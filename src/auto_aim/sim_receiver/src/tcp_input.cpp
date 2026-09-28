#include "tcp_input.hpp"

#include "common/standard_clock.hpp"

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <iostream>
#include <utility>

namespace sim_receiver {

namespace {

constexpr uint32_t kMagic = 0x44414544; // "DAED"
constexpr uint32_t kVersion = 1;
constexpr size_t kHeaderSize = 32;

uint32_t readU32(const uint8_t* p) {
    return (static_cast<uint32_t>(p[0]) << 24) |
           (static_cast<uint32_t>(p[1]) << 16) |
           (static_cast<uint32_t>(p[2]) << 8) |
           static_cast<uint32_t>(p[3]);
}

uint64_t readU64(const uint8_t* p) {
    uint64_t value = 0;
    for (int i = 0; i < 8; ++i) {
        value = (value << 8) | p[i];
    }
    return value;
}

int connectSocketAddress(const std::string& host, uint16_t port) {
    struct addrinfo hints;
    std::memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;

    struct addrinfo* result = nullptr;
    std::string service = std::to_string(port);
    if (getaddrinfo(host.c_str(), service.c_str(), &hints, &result) != 0) {
        return -1;
    }

    int fd = -1;
    for (struct addrinfo* addr = result; addr != nullptr; addr = addr->ai_next) {
        fd = ::socket(addr->ai_family, addr->ai_socktype, addr->ai_protocol);
        if (fd < 0) {
            continue;
        }
        if (::connect(fd, addr->ai_addr, addr->ai_addrlen) == 0) {
            break;
        }
        ::close(fd);
        fd = -1;
    }
    freeaddrinfo(result);

    if (fd < 0) {
        return -1;
    }

    int enabled = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &enabled, sizeof(enabled));
#ifdef SO_NOSIGPIPE
    setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &enabled, sizeof(enabled));
#endif
    int receive_buffer = 4 * 1024 * 1024;
    setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &receive_buffer, sizeof(receive_buffer));

    timeval timeout{};
    timeout.tv_usec = 500000;
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
    return fd;
}

} // namespace

TcpInput::TcpInput(std::string host, uint16_t port)
    : host_(std::move(host)), port_(port) {}

TcpInput::~TcpInput() {
    close();
}

TcpInput::TcpInput(TcpInput&& other) noexcept
    : host_(std::move(other.host_)), port_(other.port_), sock_(other.sock_),
      command_sock_(other.command_sock_), command_port_(other.command_port_),
      header_(other.header_), payload_(std::move(other.payload_)) {
    other.sock_ = -1;
    other.command_sock_ = -1;
}

TcpInput& TcpInput::operator=(TcpInput&& other) noexcept {
    if (this != &other) {
        close();
        host_ = std::move(other.host_);
        port_ = other.port_;
        sock_ = other.sock_;
        command_sock_ = other.command_sock_;
        command_port_ = other.command_port_;
        header_ = other.header_;
        payload_ = std::move(other.payload_);
        other.sock_ = -1;
        other.command_sock_ = -1;
    }
    return *this;
}

bool TcpInput::connect() {
    close();
    return connectOnce();
}

bool TcpInput::connected() const {
    return sock_ >= 0;
}

bool TcpInput::sendLine(uint16_t command_port, const std::string& line) {
    if (command_sock_ < 0 || command_port_ != command_port) {
        closeCommand();
        command_port_ = command_port;
        if (!connectCommandOnce()) {
            return false;
        }
    }

    if (writeAll(command_sock_, line)) {
        return true;
    }

    closeCommand();
    if (!connectCommandOnce()) {
        return false;
    }
    return writeAll(command_sock_, line);
}

bool TcpInput::connectOnce() {
    closeImage();
    sock_ = connectSocketAddress(host_, port_);
    return sock_ >= 0;
}

bool TcpInput::connectCommandOnce() {
    command_sock_ = connectSocketAddress(host_, command_port_);
    return command_sock_ >= 0;
}

bool TcpInput::writeAll(int fd, const std::string& data) {
    size_t sent = 0;
    while (sent < data.size()) {
#ifdef MSG_NOSIGNAL
        constexpr int kSendFlags = MSG_NOSIGNAL;
#else
        constexpr int kSendFlags = 0;
#endif
        ssize_t n = ::send(fd, data.data() + sent, data.size() - sent, kSendFlags);
        if (n < 0 && errno == EINTR) {
            continue;
        }
        if (n <= 0) {
            return false;
        }
        sent += static_cast<size_t>(n);
    }
    return true;
}

void TcpInput::closeCommand() {
    if (command_sock_ >= 0) {
        ::close(command_sock_);
        command_sock_ = -1;
    }
}

void TcpInput::closeImage() {
    if (sock_ >= 0) {
        ::close(sock_);
        sock_ = -1;
    }
}

bool TcpInput::readExact(void* buffer, size_t size) {
    if (sock_ < 0 && !connectOnce()) {
        return false;
    }

    uint8_t* out = static_cast<uint8_t*>(buffer);
    size_t done = 0;
    while (done < size) {
        ssize_t n = ::recv(sock_, out + done, size - done, 0);
        if (n == 0) {
            return false;
        }
        if (n < 0) {
            if (errno == EINTR) {
                continue;
            }
            return false;
        }
        done += static_cast<size_t>(n);
    }
    return true;
}

bool TcpInput::readFrame(SimFrame& frame) {
    if (sock_ < 0 && !connectOnce()) {
        return false;
    }

    if (!readExact(header_.data(), header_.size())) {
        closeImage();
        return false;
    }

    uint32_t magic = readU32(header_.data());
    uint32_t version = readU32(header_.data() + 4);
    uint32_t width = readU32(header_.data() + 8);
    uint32_t height = readU32(header_.data() + 12);
    uint32_t payload_len = readU32(header_.data() + 16);
    uint64_t timestamp_us = readU64(header_.data() + 20);
    uint32_t seq = readU32(header_.data() + 28);

    if (magic != kMagic || version != kVersion || payload_len == 0 ||
        payload_len > 20 * 1024 * 1024 || width == 0 || height == 0) {
        closeImage();
        return false;
    }

    payload_.resize(payload_len);
    if (!readExact(payload_.data(), payload_.size())) {
        closeImage();
        return false;
    }

    frame.local_timestamp_us = auto_aim::StandardClock::nowUs();
    cv::Mat decoded = cv::imdecode(payload_, cv::IMREAD_COLOR);
    if (decoded.empty()) {
        return false;
    }

    frame.bgr = decoded;
    frame.seq = seq;
    frame.source_timestamp_us = timestamp_us;
    return true;
}

void TcpInput::close() {
    closeImage();
    closeCommand();
}

} // namespace sim_receiver
