#include "sim_link.hpp"

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <mutex>
#include <thread>
#include <utility>

#include <opencv2/imgcodecs.hpp>

namespace rp26_sim
{
namespace
{
constexpr uint32_t kMagic = 0x44414544; // "DAED"
constexpr uint32_t kVersion = 1;
constexpr size_t kHeaderSize = 32;

uint32_t readU32(const uint8_t *p)
{
    return (static_cast<uint32_t>(p[0]) << 24) | (static_cast<uint32_t>(p[1]) << 16) |
           (static_cast<uint32_t>(p[2]) << 8) | static_cast<uint32_t>(p[3]);
}

uint64_t readU64(const uint8_t *p)
{
    uint64_t value = 0;
    for (int i = 0; i < 8; ++i)
        value = (value << 8) | p[i];
    return value;
}

int connectOnce(const std::string &host, uint16_t port, bool keep_open)
{
    struct addrinfo hints;
    std::memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;

    struct addrinfo *result = nullptr;
    const std::string service = std::to_string(port);
    if (getaddrinfo(host.c_str(), service.c_str(), &hints, &result) != 0)
        return -1;

    int fd = -1;
    for (struct addrinfo *addr = result; addr != nullptr; addr = addr->ai_next)
    {
        fd = ::socket(addr->ai_family, addr->ai_socktype, addr->ai_protocol);
        if (fd < 0)
            continue;
        if (::connect(fd, addr->ai_addr, addr->ai_addrlen) == 0)
            break;
        ::close(fd);
        fd = -1;
    }
    freeaddrinfo(result);

    if (fd < 0)
        return -1;

    if (keep_open)
    {
        int nodelay = 1;
        setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &nodelay, sizeof(nodelay));
        int receive_buffer = 8 * 1024 * 1024;
        setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &receive_buffer, sizeof(receive_buffer));
        timeval timeout{};
        timeout.tv_usec = 500000;
        setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
    }
    return fd;
}
} // namespace

class SimLink::Impl
{
public:
    Impl(std::string host, uint16_t frame_port, uint16_t command_port, uint16_t telemetry_port)
        : host_(std::move(host)), frame_port_(frame_port), command_port_(command_port),
          telemetry_port_(telemetry_port)
    {
        header_.resize(kHeaderSize);
    }

    ~Impl()
    {
        stop_telemetry_.store(true);
        if (frame_socket_ >= 0)
            ::close(frame_socket_);
        if (command_socket_ >= 0)
            ::close(command_socket_);
        if (telemetry_thread_.joinable())
            telemetry_thread_.join();
    }

    void startTelemetry()
    {
        if (telemetry_thread_.joinable())
            return;
        telemetry_thread_ = std::thread([this] { telemetryLoop(); });
    }

    bool readFrame(SimFrame &frame)
    {
        if (frame_socket_ < 0 && !connectFrame())
            return false;

        if (!readExact(header_.data(), header_.size()))
        {
            ::close(frame_socket_);
            frame_socket_ = -1;
            return false;
        }

        const uint32_t magic = readU32(header_.data());
        const uint32_t version = readU32(header_.data() + 4);
        const uint32_t width = readU32(header_.data() + 8);
        const uint32_t height = readU32(header_.data() + 12);
        const uint32_t payload_len = readU32(header_.data() + 16);
        const uint64_t timestamp_us = readU64(header_.data() + 20);
        const uint32_t seq = readU32(header_.data() + 28);

        if (magic != kMagic || version != kVersion || payload_len == 0 ||
            payload_len > 20 * 1024 * 1024 || width == 0 || height == 0)
        {
            ::close(frame_socket_);
            frame_socket_ = -1;
            return false;
        }

        payload_.resize(payload_len);
        if (!readExact(payload_.data(), payload_.size()))
        {
            ::close(frame_socket_);
            frame_socket_ = -1;
            return false;
        }

        cv::Mat decoded = cv::imdecode(payload_, cv::IMREAD_COLOR);
        if (decoded.empty())
            return false;

        frame.bgr = decoded;
        frame.seq = seq;
        frame.source_timestamp_us = timestamp_us;
        return true;
    }

    bool sendLine(const std::string &line)
    {
        std::lock_guard<std::mutex> lock(command_mutex_);
        // 复用长连接：仿真的指令服务是"每连接开一个线程"，高频下发时不要每帧新建连接。
        if (command_socket_ < 0)
            command_socket_ = connectOnce(host_, command_port_, true);
        if (command_socket_ < 0)
            return false;

        size_t sent = 0;
        while (sent < line.size())
        {
            const ssize_t n = ::send(command_socket_, line.data() + sent, line.size() - sent, 0);
            if (n <= 0)
            {
                ::close(command_socket_);
                command_socket_ = -1;
                command_socket_ = connectOnce(host_, command_port_, true);
                if (command_socket_ < 0)
                    return false;
                continue;
            }
            sent += static_cast<size_t>(n);
        }
        return true;
    }

    std::vector<std::string> pollTelemetry()
    {
        std::lock_guard<std::mutex> lock(telemetry_mutex_);
        std::vector<std::string> lines;
        lines.swap(telemetry_lines_);
        return lines;
    }

private:
    bool connectFrame()
    {
        frame_socket_ = connectOnce(host_, frame_port_, true);
        return frame_socket_ >= 0;
    }

    bool readExact(void *buffer, size_t size)
    {
        uint8_t *out = static_cast<uint8_t *>(buffer);
        size_t done = 0;
        while (done < size)
        {
            const ssize_t n = ::recv(frame_socket_, out + done, size - done, 0);
            if (n == 0)
                return false;
            if (n < 0)
            {
                if (errno == EINTR)
                    continue;
                return false;
            }
            done += static_cast<size_t>(n);
        }
        return true;
    }

    void telemetryLoop()
    {
        while (!stop_telemetry_.load())
        {
            const int fd = connectOnce(host_, telemetry_port_, false);
            if (fd < 0)
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(200));
                continue;
            }

            std::string buffer;
            char chunk[1024];
            while (!stop_telemetry_.load())
            {
                const ssize_t received = ::recv(fd, chunk, sizeof(chunk), 0);
                if (received <= 0)
                    break;
                buffer.append(chunk, static_cast<size_t>(received));
                std::size_t newline = buffer.find('\n');
                while (newline != std::string::npos)
                {
                    std::string line = buffer.substr(0, newline);
                    buffer.erase(0, newline + 1);
                    if (!line.empty())
                    {
                        std::lock_guard<std::mutex> lock(telemetry_mutex_);
                        telemetry_lines_.push_back(std::move(line));
                    }
                    newline = buffer.find('\n');
                }
            }
            ::close(fd);
            if (!stop_telemetry_.load())
                std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }
    }

    std::string host_;
    uint16_t frame_port_ = 7666;
    uint16_t command_port_ = 7667;
    uint16_t telemetry_port_ = 7668;

    int frame_socket_ = -1;
    int command_socket_ = -1;
    std::mutex command_mutex_;
    std::vector<uint8_t> header_;
    std::vector<uint8_t> payload_;

    std::atomic<bool> stop_telemetry_{false};
    std::thread telemetry_thread_;
    std::mutex telemetry_mutex_;
    std::vector<std::string> telemetry_lines_;
};

SimLink::SimLink(std::string host, uint16_t frame_port, uint16_t command_port, uint16_t telemetry_port)
    : m_impl(std::make_unique<Impl>(std::move(host), frame_port, command_port, telemetry_port))
{
    m_impl->startTelemetry();
}

SimLink::~SimLink() = default;

bool SimLink::readFrame(SimFrame &frame)
{
    return m_impl->readFrame(frame);
}

bool SimLink::sendGimbal(double yaw, double pitch)
{
    char command[64];
    std::snprintf(command, sizeof(command), "GIMBAL %.17g %.17g\n", yaw, pitch);
    return m_impl->sendLine(command);
}

bool SimLink::sendFire()
{
    return m_impl->sendLine("FIRE\n");
}

bool SimLink::sendReset()
{
    return m_impl->sendLine("RESET\n");
}

std::vector<std::string> SimLink::pollTelemetry()
{
    return m_impl->pollTelemetry();
}

} // namespace rp26_sim
