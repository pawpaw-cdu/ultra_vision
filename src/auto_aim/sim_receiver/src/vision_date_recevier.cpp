#include "sim_receiver/vision_date_recevier.hpp"

#include "tcp_input.hpp"
#include "udp_input.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <condition_variable>
#include <deque>
#include <iostream>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#include <yaml-cpp/yaml.h>

namespace sim_receiver {

namespace {

// How many JPEGs the receiver may decode in parallel.  One socket reader drains
// the wire; N decoders turn those packets into frames.  With a single decoder
// the decode cost is back on the critical path, which is exactly what used to
// let a slow client throttle the simulator's send rate (TCP back-pressure).
int configuredDecodeThreads(int fallback) {
    if (const char* value = std::getenv("ULTRA_VISION_SIM_DECODE_THREADS")) {
        const int parsed = std::atoi(value);
        if (parsed > 0) {
            return std::min(parsed, 8);
        }
    }
    return fallback;
}

} // namespace

// Three-stage pipeline, one thread role per stage:
//
//   reader thread   socket -> [header | JPEG]      (no decode, no copy)
//   decoder threads JPEG   -> cv::Mat              (parallel, bounded queue)
//   consumer        cv::Mat (newest wins, stale frames dropped)
//
// The previous implementation did recv + imdecode back-to-back on one thread.
// While it decoded it was not reading, so the simulator's writer blocked and the
// observable frame rate of *both* ends collapsed to the client's decode rate.
// Keeping the socket drained and decoding off the wire removes that coupling.
//
// The reader sits behind PacketSource, so the transport is a detail: TCP
// (default) or the simulator's UDP fragment stream (`protocol: udp`).  Commands
// always stay on TCP 7667.
class VisionDateRecevier::Impl {
public:
    Impl(std::string host, uint16_t port, uint16_t command_port, int decode_threads)
        : input(std::move(host), port), command_port_(command_port),
          decode_threads_(decode_threads > 0 ? decode_threads : 1) {}

    ~Impl() { stop(); }

    // 图像改走 UDP 分片流（仿真器 DAEDALUS_UDP=1 的那一路）。指令仍然走 TCP。
    void useUdp(uint16_t udp_port) {
        udp_port_ = udp_port;
        udp = std::make_unique<UdpInput>(input.host(), udp_port);
    }

    void useTcp() {
        udp.reset();
    }

    void overrideEndpoint(std::string host, uint16_t port) {
        stop();
        input = TcpInput(std::move(host), port);
        if (udp) {
            // 协议不变，只把主机（必要时还有端口）搬到新地址上。
            udp = std::make_unique<UdpInput>(input.host(), udp_port_);
        }
    }

    void overrideCommandEndpoint(uint16_t port) {
        command_port_ = port;
    }

    PacketSource& image() {
        return udp ? static_cast<PacketSource&>(*udp) : static_cast<PacketSource&>(input);
    }

    bool connect() {
        if (reader.joinable()) {
            return connected_.load();
        }

        stop_requested_.store(false);
        {
            std::lock_guard<std::mutex> lock(raw_mutex);
            raw_queue.clear();
            next_packet_id = 1;
        }
        {
            std::lock_guard<std::mutex> lock(mutex);
            frame_ready = false;
            latest_id_ = 0;
            latest_frame.release();
        }

        const bool ok = image().connect();
        if (udp) {
            std::cout << "Simulator image stream: UDP " << input.host() << ':' << udp_port_
                      << " (subscribed; commands TCP " << command_port_ << ')' << std::endl;
        } else {
            std::cout << "Simulator image stream: TCP " << input.host() << ':' << input.port()
                      << std::endl;
        }
        connected_.store(ok);
        reader = std::thread([this] { readLoop(); });
        decoders.reserve(static_cast<std::size_t>(decode_threads_));
        for (int i = 0; i < decode_threads_; ++i) {
            decoders.emplace_back([this] { decodeLoop(); });
        }
        if (!command_worker.joinable()) {
            command_worker = std::thread([this] { commandLoop(); });
        }
        return true;
    }

    bool connected() const {
        return connected_.load();
    }

    uint64_t droppedPackets() const {
        return dropped_packets_.load();
    }

    bool sendFireCommand() {
        if (stop_requested_.load()) return false;
        {
            std::lock_guard<std::mutex> lock(command_mutex);
            fire_pending = true;
        }
        command_cv.notify_one();
        return true;
    }

    bool sendResetCommand() {
        if (stop_requested_.load()) return false;
        {
            std::lock_guard<std::mutex> lock(command_mutex);
            reset_pending = true;
        }
        command_cv.notify_one();
        return true;
    }

    bool sendGimbalCommand(double yaw, double pitch) {
        if (stop_requested_.load() || !std::isfinite(yaw) || !std::isfinite(pitch)) {
            return false;
        }
        {
            std::lock_guard<std::mutex> lock(command_mutex);
            gimbal_pending = GimbalCommand{yaw, pitch};
        }
        command_cv.notify_one();
        return true;
    }

    cv::Mat getFrame(uint64_t* sequence, uint64_t* source_timestamp_us,
                     uint64_t* local_timestamp_us, float* gimbal_yaw,
                     float* gimbal_pitch) {
        std::unique_lock<std::mutex> lock(mutex);
        frame_cv.wait_for(lock, std::chrono::milliseconds(100), [this] {
            return stop_requested_.load() || frame_ready;
        });
        if (stop_requested_.load() || !frame_ready) {
            return cv::Mat();
        }
        if (sequence) *sequence = latest_sequence_;
        if (source_timestamp_us) *source_timestamp_us = latest_source_timestamp_us_;
        if (local_timestamp_us) *local_timestamp_us = latest_local_timestamp_us_;
        if (gimbal_yaw) *gimbal_yaw = latest_gimbal_yaw_;
        if (gimbal_pitch) *gimbal_pitch = latest_gimbal_pitch_;
        frame_ready = false;
        return std::move(latest_frame);
    }

private:
    struct RawItem {
        SimPacket packet;
        uint64_t id = 0;   // arrival order; the consumer only ever moves forward
    };

    void readLoop() {
        PacketSource& source = image();
        while (!stop_requested_.load()) {
            SimPacket packet;
            const ReadStatus status = source.read(packet);
            if (status == ReadStatus::Idle) {
                continue;   // 流正常，只是这段时间没有帧（UDP poll 超时）
            }
            if (status == ReadStatus::Error) {
                connected_.store(false);
                for (int i = 0; i < 5 && !stop_requested_.load(); ++i) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(20));
                }
                continue;
            }

            connected_.store(true);
            {
                std::lock_guard<std::mutex> lock(raw_mutex);
                // Bounded queue: if every decoder is busy, drop the oldest frame
                // instead of growing without bound or stalling the socket.  A
                // late frame is worthless to a control loop.
                if (raw_queue.size() >= static_cast<std::size_t>(decode_threads_)) {
                    raw_queue.pop_front();
                    dropped_packets_.fetch_add(1, std::memory_order_relaxed);
                }
                raw_queue.push_back(RawItem{std::move(packet), next_packet_id++});
            }
            raw_cv.notify_one();
        }
    }

    void decodeLoop() {
        while (true) {
            RawItem item;
            {
                std::unique_lock<std::mutex> lock(raw_mutex);
                raw_cv.wait(lock, [this] {
                    return stop_requested_.load() || !raw_queue.empty();
                });
                if (raw_queue.empty()) {
                    return;   // stopping and everything already handed out
                }
                item = std::move(raw_queue.front());
                raw_queue.pop_front();
            }

            cv::Mat decoded = decodePacket(item.packet);
            if (decoded.empty()) {
                continue;
            }

            bool published = false;
            {
                std::lock_guard<std::mutex> lock(mutex);
                // Decoders run in parallel, so results can land out of order.
                // Keep only the newest: never hand the consumer a frame older
                // than one it has already seen.
                if (item.id > latest_id_) {
                    latest_id_ = item.id;
                    latest_frame = std::move(decoded);
                    latest_sequence_ = item.packet.seq;
                    latest_source_timestamp_us_ = item.packet.source_timestamp_us;
                    latest_local_timestamp_us_ = item.packet.local_timestamp_us;
                    latest_gimbal_yaw_ = item.packet.gimbal_yaw;
                    latest_gimbal_pitch_ = item.packet.gimbal_pitch;
                    frame_ready = true;
                    published = true;
                }
            }
            if (published) {
                frame_cv.notify_one();
            }
        }
    }

    void commandLoop() {
        while (!stop_requested_.load()) {
            bool fire_pending_now = false;
            bool reset_pending_now = false;
            std::optional<GimbalCommand> gimbal_pending_now;
            {
                std::unique_lock<std::mutex> lock(command_mutex);
                command_cv.wait(lock, [this] {
                    return stop_requested_.load() || fire_pending || reset_pending ||
                           gimbal_pending.has_value();
                });
                if (stop_requested_.load()) break;
                fire_pending_now = fire_pending;
                fire_pending = false;
                reset_pending_now = reset_pending;
                reset_pending = false;
                gimbal_pending_now = std::move(gimbal_pending);
                gimbal_pending.reset();
            }

            // Keep socket setup and connect delays off the image-processing loop.
            if (gimbal_pending_now.has_value()) {
                char command[96];
                std::snprintf(command, sizeof(command), "GIMBAL %.17g %.17g\n",
                              gimbal_pending_now->yaw, gimbal_pending_now->pitch);
                if (!input.sendLine(command_port_, command)) {
                    std::cerr << "Failed to send simulator gimbal command" << std::endl;
                }
            }
            if (fire_pending_now && !input.sendLine(command_port_, "FIRE\n")) {
                std::cerr << "Failed to send simulator fire command" << std::endl;
            }
            if (reset_pending_now && !input.sendLine(command_port_, "RESET\n")) {
                std::cerr << "Failed to send simulator reset command" << std::endl;
            }
        }
    }

    void stop() {
        stop_requested_.store(true);
        // Unblock the reader parked in recv() so shutdown does not wait for the
        // socket timeout; the reader still owns closing the descriptor.  (UDP
        // polls with a short timeout, so it needs no wake-up.)
        image().shutdown();
        raw_cv.notify_all();
        frame_cv.notify_all();
        command_cv.notify_all();
        if (reader.joinable()) {
            reader.join();
        }
        for (auto& decoder : decoders) {
            if (decoder.joinable()) {
                decoder.join();
            }
        }
        decoders.clear();
        if (command_worker.joinable()) {
            command_worker.join();
        }
        image().close();
        input.close();
    }

    TcpInput input;
    std::unique_ptr<UdpInput> udp;
    uint16_t udp_port_ = 7669;
    uint16_t command_port_ = 7667;
    const int decode_threads_;
    std::thread reader;
    std::vector<std::thread> decoders;
    std::thread command_worker;

    std::mutex mutex;                 // decoded-frame slot (decoders -> consumer)
    std::condition_variable frame_cv;
    cv::Mat latest_frame;
    uint64_t latest_sequence_ = 0;
    uint64_t latest_source_timestamp_us_ = 0;
    uint64_t latest_local_timestamp_us_ = 0;
    float latest_gimbal_yaw_ = 0.0f;
    float latest_gimbal_pitch_ = 0.0f;
    uint64_t latest_id_ = 0;
    bool frame_ready = false;

    std::mutex raw_mutex;             // received-but-undecoded queue (reader -> decoders)
    std::condition_variable raw_cv;
    std::deque<RawItem> raw_queue;
    uint64_t next_packet_id = 1;
    std::atomic<uint64_t> dropped_packets_{0};

    std::mutex command_mutex;
    std::condition_variable command_cv;
    struct GimbalCommand {
        double yaw = 0.0;
        double pitch = 0.0;
    };
    bool fire_pending = false;
    bool reset_pending = false;
    std::optional<GimbalCommand> gimbal_pending;
    std::atomic<bool> stop_requested_{false};
    std::atomic<bool> connected_{false};
};

VisionDateRecevier::VisionDateRecevier(const std::string& config_path)
    : impl_(std::make_unique<Impl>("127.0.0.1", 7666, 7667,
                                   configuredDecodeThreads(2))) {
    try {
        YAML::Node node = YAML::LoadFile(config_path);
        if (node["host"]) {
            overrideEndpoint(node["host"].as<std::string>(), 7666);
        }
        if (node["port"]) {
            overrideEndpoint(node["host"] ? node["host"].as<std::string>() : "127.0.0.1",
                             static_cast<uint16_t>(node["port"].as<int>()));
        }
        if (node["command_port"]) {
            overrideCommandEndpoint(static_cast<uint16_t>(node["command_port"].as<int>()));
        }
        // 图像流协议：tcp（默认）/ udp。udp 时图像走仿真器 DAEDALUS_UDP=1 那一路
        // 分片流（默认端口 7669），指令仍然走 TCP 7667。见 simulator_system/API.md §1.1。
        if (node["protocol"] && node["protocol"].as<std::string>() == "udp") {
            const uint16_t udp_port = node["udp_port"]
                ? static_cast<uint16_t>(node["udp_port"].as<int>()) : 7669;
            impl_->useUdp(udp_port);
        }
    } catch (const std::exception& err) {
        std::cerr << "Failed to load simulator config: " << err.what() << std::endl;
    }

    // 环境变量覆盖（脚本 A/B 用，不必改配置文件）：
    //   ULTRA_VISION_SIM_PROTOCOL=udp|tcp
    //   ULTRA_VISION_SIM_UDP_PORT=7669
    if (const char* protocol = std::getenv("ULTRA_VISION_SIM_PROTOCOL")) {
        const std::string value(protocol);
        if (value == "udp") {
            const char* port_env = std::getenv("ULTRA_VISION_SIM_UDP_PORT");
            const int port = port_env != nullptr ? std::atoi(port_env) : 7669;
            impl_->useUdp(static_cast<uint16_t>(std::max(1, std::min(65535, port))));
        } else if (value == "tcp") {
            impl_->useTcp();
        }
    }
}

VisionDateRecevier::~VisionDateRecevier() = default;

void VisionDateRecevier::overrideEndpoint(const std::string& host, uint16_t port) {
    impl_->overrideEndpoint(host, port);
}

void VisionDateRecevier::overrideCommandEndpoint(uint16_t port) {
    impl_->overrideCommandEndpoint(port);
}

bool VisionDateRecevier::connect() {
    return impl_->connect();
}

bool VisionDateRecevier::connected() const {
    return impl_->connected();
}

std::uint64_t VisionDateRecevier::droppedPackets() const {
    return impl_->droppedPackets();
}

bool VisionDateRecevier::sendFireCommand() {
    return impl_->sendFireCommand();
}

bool VisionDateRecevier::sendResetCommand() {
    return impl_->sendResetCommand();
}

bool VisionDateRecevier::sendGimbalCommand(double yaw, double pitch) {
    return impl_->sendGimbalCommand(yaw, pitch);
}

cv::Mat VisionDateRecevier::getFrame(uint64_t* sequence,
                                     uint64_t* source_timestamp_us,
                                     uint64_t* local_timestamp_us,
                                     float* gimbal_yaw, float* gimbal_pitch) {
    return impl_->getFrame(sequence, source_timestamp_us, local_timestamp_us,
                           gimbal_yaw, gimbal_pitch);
}

} // namespace sim_receiver
