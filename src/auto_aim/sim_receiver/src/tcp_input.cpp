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
#include <opencv2/imgcodecs.hpp>
#include <utility>

namespace sim_receiver {

namespace {

constexpr uint32_t kMagic = 0x44414544; // "DAED"
constexpr uint32_t kVersion = 2;   // v2 = v1 + 真实云台姿态 (gimbal_yaw/pitch)

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

// 包头里的云台姿态是 f32 big-endian（Rust 侧 to_be_bytes）。
float readF32(const uint8_t* p) {
    const uint32_t bits = readU32(p);
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(value));
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

BufferPool::Buffer BufferPool::acquire(std::size_t size) {
    std::shared_ptr<std::vector<uint8_t>> storage;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!free_.empty()) {
            storage = std::move(free_.back());
            free_.pop_back();
        }
    }
    if (!storage) {
        storage = std::make_shared<std::vector<uint8_t>>();
    }
    // Exact size: the decoder consumes the whole buffer, so it must not carry
    // stale bytes from a larger previous frame.
    if (storage->size() != size) {
        storage->resize(size);
    }

    // The deleter owns one reference and returns the storage to the pool when the
    // last borrower drops it.  weak_ptr keeps this correct even if the pool is
    // destroyed while a buffer is still in flight (endpoint reconfiguration).
    std::weak_ptr<BufferPool> weak = shared_from_this();
    return Buffer(storage.get(), [weak, storage](std::vector<uint8_t>*) mutable {
        if (auto pool = weak.lock()) {
            pool->release(std::move(storage));
        }
    });
}

std::size_t BufferPool::freeCount() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return free_.size();
}

void BufferPool::release(std::shared_ptr<std::vector<uint8_t>> buffer) {
    std::lock_guard<std::mutex> lock(mutex_);
    free_.push_back(std::move(buffer));
}

TcpInput::TcpInput(std::string host, uint16_t port)
    : host_(std::move(host)), port_(port), pool_(std::make_shared<BufferPool>()) {}

TcpInput::~TcpInput() {
    close();
}

TcpInput::TcpInput(TcpInput&& other) noexcept
    : host_(std::move(other.host_)), port_(other.port_), sock_(other.sock_.load()),
      command_sock_(other.command_sock_), command_port_(other.command_port_),
      header_(other.header_), pool_(std::move(other.pool_)) {
    other.sock_.store(-1);
    other.command_sock_ = -1;
}

TcpInput& TcpInput::operator=(TcpInput&& other) noexcept {
    if (this != &other) {
        close();
        host_ = std::move(other.host_);
        port_ = other.port_;
        sock_.store(other.sock_.load());
        command_sock_ = other.command_sock_;
        command_port_ = other.command_port_;
        header_ = other.header_;
        pool_ = std::move(other.pool_);
        other.sock_.store(-1);
        other.command_sock_ = -1;
        if (!pool_) pool_ = std::make_shared<BufferPool>();
    }
    return *this;
}

bool TcpInput::connect() {
    close();
    if (!pool_) pool_ = std::make_shared<BufferPool>();
    return connectOnce();
}

bool TcpInput::connected() const {
    return sock_.load() >= 0;
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
    const int fd = connectSocketAddress(host_, port_);
    sock_.store(fd);
    return fd >= 0;
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
    const int fd = sock_.exchange(-1);
    if (fd >= 0) {
        ::close(fd);
    }
}

void TcpInput::shutdownImage() {
    const int fd = sock_.load();
    if (fd >= 0) {
        ::shutdown(fd, SHUT_RDWR);
    }
}

bool TcpInput::readExact(void* buffer, size_t size) {
    if (sock_.load() < 0 && !connectOnce()) {
        return false;
    }
    const int active = sock_.load();
    if (active < 0) {
        return false;
    }

    uint8_t* out = static_cast<uint8_t*>(buffer);
    size_t done = 0;
    while (done < size) {
        ssize_t n = ::recv(active, out + done, size - done, 0);
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

bool TcpInput::readPacket(SimPacket& packet) {
    packet = SimPacket{};
    if (sock_.load() < 0 && !connectOnce()) {
        return false;
    }

    if (!readExact(header_.data(), header_.size())) {
        closeImage();
        return false;
    }

    const uint32_t magic = readU32(header_.data());
    const uint32_t version = readU32(header_.data() + 4);
    const uint32_t width = readU32(header_.data() + 8);
    const uint32_t height = readU32(header_.data() + 12);
    const uint32_t payload_len = readU32(header_.data() + 16);
    const uint64_t timestamp_us = readU64(header_.data() + 20);
    const uint32_t seq = readU32(header_.data() + 28);
    const float gimbal_yaw = readF32(header_.data() + 32);
    const float gimbal_pitch = readF32(header_.data() + 36);

    if (magic != kMagic || version != kVersion || payload_len == 0 ||
        payload_len > 20 * 1024 * 1024 || width == 0 || height == 0) {
        closeImage();
        return false;
    }

    // No copy: the JPEG bytes land straight in a pooled buffer that the decoder
    // will release when it is done with them.
    BufferPool::Buffer payload = pool_->acquire(payload_len);
    if (!readExact(payload->data(), payload->size())) {
        closeImage();
        return false;   // payload returns itself to the pool
    }

    packet.payload = std::move(payload);
    packet.seq = seq;
    packet.source_timestamp_us = timestamp_us;
    packet.local_timestamp_us = auto_aim::StandardClock::nowUs();
    packet.width = width;
    packet.height = height;
    packet.gimbal_yaw = gimbal_yaw;
    packet.gimbal_pitch = gimbal_pitch;
    return true;
}

cv::Mat TcpInput::decode(const SimPacket& packet) {
    if (!packet.payload || packet.payload->empty()) {
        return cv::Mat();
    }
    return cv::imdecode(*packet.payload, cv::IMREAD_COLOR);
}

bool TcpInput::readFrame(SimFrame& frame) {
    SimPacket packet;
    if (!readPacket(packet)) {
        return false;
    }
    cv::Mat decoded = decode(packet);
    if (decoded.empty()) {
        return false;
    }
    frame.bgr = std::move(decoded);
    frame.seq = packet.seq;
    frame.source_timestamp_us = packet.source_timestamp_us;
    frame.local_timestamp_us = packet.local_timestamp_us;
    frame.gimbal_yaw = packet.gimbal_yaw;
    frame.gimbal_pitch = packet.gimbal_pitch;
    return true;
}

void TcpInput::close() {
    closeImage();
    closeCommand();
}

} // namespace sim_receiver
