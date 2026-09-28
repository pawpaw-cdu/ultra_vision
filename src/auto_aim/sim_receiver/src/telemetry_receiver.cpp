#include "sim_receiver/telemetry_receiver.hpp"

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstring>
#include <iostream>
#include <mutex>
#include <thread>
#include <utility>

namespace sim_receiver
{
    namespace
    {
        int connectOnce(const std::string& host, uint16_t port)
        {
            struct addrinfo hints;
            std::memset(&hints, 0, sizeof(hints));
            hints.ai_family = AF_UNSPEC;
            hints.ai_socktype = SOCK_STREAM;

            struct addrinfo* result = nullptr;
            const std::string service = std::to_string(port);
            if (getaddrinfo(host.c_str(), service.c_str(), &hints, &result) != 0) {
                return -1;
            }

            int fd = -1;
            for (struct addrinfo* addr = result; addr != nullptr; addr = addr->ai_next) {
                fd = ::socket(addr->ai_family, addr->ai_socktype, addr->ai_protocol);
                if (fd < 0) continue;
                if (::connect(fd, addr->ai_addr, addr->ai_addrlen) == 0) break;
                ::close(fd);
                fd = -1;
            }
            freeaddrinfo(result);
            return fd;
        }
    } // namespace

    class TelemetryReceiver::Impl
    {
    public:
        Impl(std::string host, uint16_t port) : host_(std::move(host)), port_(port) {}

        ~Impl() { stop(); }

        bool connect()
        {
            if (worker_.joinable()) return connected_.load();
            stop_requested_.store(false);
            worker_ = std::thread([this] { receiveLoop(); });
            return true;
        }

        bool connected() const { return connected_.load(); }

        std::vector<std::string> poll()
        {
            std::lock_guard<std::mutex> lock(mutex_);
            std::vector<std::string> lines;
            lines.swap(pending_);
            return lines;
        }

    private:
        void receiveLoop()
        {
            while (!stop_requested_.load()) {
                const int fd = connectOnce(host_, port_);
                if (fd < 0) {
                    connected_.store(false);
                    std::this_thread::sleep_for(std::chrono::milliseconds(200));
                    continue;
                }
                connected_.store(true);
                std::cout << "Rune telemetry connected on " << host_ << ":" << port_
                          << std::endl;

                std::string buffer;
                char chunk[1024];
                while (!stop_requested_.load()) {
                    const ssize_t received = ::recv(fd, chunk, sizeof(chunk), 0);
                    if (received <= 0) break;
                    buffer.append(chunk, static_cast<std::size_t>(received));
                    std::size_t newline = buffer.find('\n');
                    while (newline != std::string::npos) {
                        std::string line = buffer.substr(0, newline);
                        buffer.erase(0, newline + 1);
                        if (!line.empty()) {
                            std::lock_guard<std::mutex> lock(mutex_);
                            pending_.push_back(std::move(line));
                        }
                        newline = buffer.find('\n');
                    }
                }
                ::close(fd);
                connected_.store(false);
                if (!stop_requested_.load()) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(200));
                }
            }
        }

        void stop()
        {
            stop_requested_.store(true);
            if (worker_.joinable()) worker_.join();
        }

        std::string host_;
        uint16_t port_;
        std::thread worker_;
        std::mutex mutex_;
        std::vector<std::string> pending_;
        std::atomic<bool> stop_requested_{false};
        std::atomic<bool> connected_{false};
    };

    TelemetryReceiver::TelemetryReceiver(std::string host, uint16_t port)
        : impl_(std::make_unique<Impl>(std::move(host), port))
    {
    }

    TelemetryReceiver::~TelemetryReceiver() = default;

    bool TelemetryReceiver::connect() { return impl_->connect(); }
    bool TelemetryReceiver::connected() const { return impl_->connected(); }
    std::vector<std::string> TelemetryReceiver::poll() { return impl_->poll(); }
} // namespace sim_receiver
