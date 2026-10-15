#include "udp_input.hpp"

#include "common/standard_clock.hpp"

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <utility>

namespace sim_receiver {

namespace {

constexpr size_t kHeaderLen = 40;
constexpr size_t kChunkSize = 1400;          // 与仿真器一致：报文 ≤ 1460B，不触发 IP 分片
constexpr uint32_t kMagic = 0x44414544;      // "DAED"
constexpr uint16_t kSupportedVersion = 1;
constexpr uint64_t kSubscribeIntervalMs = 1000;
// 最多同时攒这么多帧的分片。正常只会有一两帧在途；这个上限是防止
// 乱序/伪造报文把 pending_ 撑爆。
constexpr size_t kMaxPendingFrames = 64;

uint16_t readU16(const uint8_t* p) {
    return static_cast<uint16_t>((static_cast<uint16_t>(p[0]) << 8) | p[1]);
}

uint32_t readU32(const uint8_t* p) {
    return (static_cast<uint32_t>(p[0]) << 24) | (static_cast<uint32_t>(p[1]) << 16) |
           (static_cast<uint32_t>(p[2]) << 8) | static_cast<uint32_t>(p[3]);
}

uint64_t readU64(const uint8_t* p) {
    uint64_t value = 0;
    for (int i = 0; i < 8; ++i) {
        value = (value << 8) | p[i];
    }
    return value;
}

float readF32(const uint8_t* p) {
    const uint32_t bits = readU32(p);
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(value));
    return std::isfinite(value) ? value : 0.0f;
}

uint64_t nowMs() {
    const auto now = auto_aim::StandardClock::now();
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count());
}

} // namespace

UdpInput::UdpInput(std::string host, uint16_t port)
    : host_(std::move(host)), port_(port), pool_(std::make_shared<BufferPool>()) {}

UdpInput::~UdpInput() {
    close();
}

void UdpInput::setTimeoutMs(int timeout_ms) {
    timeout_ms_ = timeout_ms > 0 ? timeout_ms : 100;
}

bool UdpInput::connect() {
    close();
    if (!pool_) pool_ = std::make_shared<BufferPool>();

    // 只解析目标地址用来 sendto；不 connect——多网卡主机可能从别的地址回包，
    // connect 会把那部分包直接过滤掉。
    struct addrinfo hints;
    std::memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_DGRAM;
    struct addrinfo* result = nullptr;
    const std::string service = std::to_string(port_);
    if (getaddrinfo(host_.c_str(), service.c_str(), &hints, &result) != 0) {
        return false;
    }

    const int fd = ::socket(result->ai_family, result->ai_socktype, result->ai_protocol);
    if (fd < 0) {
        freeaddrinfo(result);
        return false;
    }

    // 本地只要求"能收"：绑任意端口，仿真器按来源地址回推。
    struct sockaddr_in local;
    std::memset(&local, 0, sizeof(local));
    local.sin_family = AF_INET;
    local.sin_addr.s_addr = htonl(INADDR_ANY);
    local.sin_port = 0;
    if (::bind(fd, reinterpret_cast<struct sockaddr*>(&local), sizeof(local)) != 0) {
        std::cerr << "UDP 绑定本地端口失败: " << std::strerror(errno) << std::endl;
        freeaddrinfo(result);
        ::close(fd);
        return false;
    }

    // 一帧上百个报文是突发，内核缓冲不够就会丢；尽量放大。
    int buffer = 8 * 1024 * 1024;
    setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &buffer, sizeof(buffer));

    target_ = *reinterpret_cast<struct sockaddr_in*>(result->ai_addr);
    freeaddrinfo(result);

    sock_.store(fd);
    scratch_.resize(kHeaderLen + kChunkSize + 64);
    pending_.clear();
    delivered_any_ = false;
    last_timestamp_us_ = 0;
    last_frame_id_ = 0;

    // 立刻订阅一次，别干等第一个 1s 周期。
    subscribe();
    return true;
}

bool UdpInput::connected() const {
    return sock_.load() >= 0;
}

void UdpInput::close() {
    const int fd = sock_.exchange(-1);
    if (fd >= 0) {
        ::close(fd);
    }
    pending_.clear();   // 半帧的缓冲在这里归还池子
}

void UdpInput::subscribe() {
    const int fd = sock_.load();
    if (fd < 0) {
        return;
    }
    // 内容任意，仿真器只认来源地址。
    const char hello[] = "SUB";
    ::sendto(fd, hello, sizeof(hello) - 1, 0,
             reinterpret_cast<struct sockaddr*>(&target_), sizeof(target_));
    last_subscribe_ms_ = nowMs();
}

void UdpInput::subscribeIfDue() {
    if (nowMs() - last_subscribe_ms_ < kSubscribeIntervalMs) {
        return;
    }
    subscribe();
}

void UdpInput::prunePending(uint32_t newest_complete) {
    // 比最新拼齐帧旧 2 帧以上的，一定拼不齐了（对应"按时间戳丢旧帧"）。
    // 用有符号差值判，frame_id 回绕时也不会误删。
    for (auto it = pending_.begin(); it != pending_.end();) {
        if (static_cast<int32_t>(newest_complete - it->first) > 2) {
            it = pending_.erase(it);
        } else {
            ++it;
        }
    }
}

ReadStatus UdpInput::read(SimPacket& packet) {
    if (sock_.load() < 0 && !connect()) {
        return ReadStatus::Error;
    }

    for (;;) {
        subscribeIfDue();

        struct pollfd pfd;
        pfd.fd = sock_.load();
        pfd.events = POLLIN;
        pfd.revents = 0;
        const int ready = ::poll(&pfd, 1, timeout_ms_);
        if (ready == 0) {
            return ReadStatus::Idle;   // 这段时间没有报文：流正常，继续等
        }
        if (ready < 0) {
            if (errno == EINTR) {
                continue;
            }
            return ReadStatus::Error;
        }

        const ssize_t n = ::recv(sock_.load(), scratch_.data(), scratch_.size(), 0);
        if (n < 0) {
            if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK) {
                continue;
            }
            return ReadStatus::Error;
        }
        if (static_cast<size_t>(n) < kHeaderLen) {
            continue;   // 订阅报文之类的短包
        }

        const uint8_t* header = scratch_.data();
        if (readU32(header) != kMagic || readU16(header + 4) != kSupportedVersion ||
            readU16(header + 6) != kHeaderLen) {
            continue;
        }

        const uint32_t frame_id = readU32(header + 8);
        const uint64_t timestamp_us = readU64(header + 12);
        const uint32_t total_len = readU32(header + 20);
        const uint32_t offset = readU32(header + 24);
        const uint32_t payload_len = readU32(header + 28);
        if (total_len == 0 || total_len > 20u * 1024 * 1024 || payload_len == 0 ||
            static_cast<size_t>(n) < kHeaderLen + payload_len ||
            static_cast<uint64_t>(offset) + payload_len > total_len) {
            continue;
        }

        // 已知更旧的帧不必再攒（它不可能被交付）。
        if (delivered_any_ && timestamp_us <= last_timestamp_us_) {
            pending_.erase(frame_id);
            continue;
        }

        auto it = pending_.find(frame_id);
        if (it == pending_.end()) {
            // 攒太多半帧说明对面在灌我们拼不完的东西：丢最旧的一个。
            if (pending_.size() >= kMaxPendingFrames) {
                pending_.erase(pending_.begin());
            }
            PendingFrame fresh;
            fresh.data = pool_->acquire(total_len);
            fresh.total_len = total_len;
            fresh.timestamp_us = timestamp_us;
            fresh.gimbal_yaw = readF32(header + 32);
            fresh.gimbal_pitch = readF32(header + 36);
            it = pending_.emplace(frame_id, std::move(fresh)).first;
        }
        PendingFrame& frame = it->second;
        // 分片直接写进整帧缓冲的对应偏移：没有额外的整帧拷贝。
        std::memcpy(frame.data->data() + offset, header + kHeaderLen, payload_len);
        frame.received += payload_len;
        if (frame.received < frame.total_len) {
            continue;
        }

        // 拼齐了，但可能已经被更新的帧超过：过期就丢。
        if (delivered_any_ && timestamp_us <= last_timestamp_us_) {
            pending_.erase(frame_id);
            continue;
        }

        packet.payload = std::move(frame.data);
        packet.seq = frame_id;
        packet.source_timestamp_us = timestamp_us;
        packet.local_timestamp_us = auto_aim::StandardClock::nowUs();
        packet.gimbal_yaw = frame.gimbal_yaw;
        packet.gimbal_pitch = frame.gimbal_pitch;
        pending_.erase(frame_id);

        last_timestamp_us_ = timestamp_us;
        last_frame_id_ = frame_id;
        delivered_any_ = true;
        prunePending(frame_id);
        return ReadStatus::Frame;
    }
}

} // namespace sim_receiver
