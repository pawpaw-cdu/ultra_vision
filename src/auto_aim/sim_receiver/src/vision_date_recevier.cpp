#include "sim_receiver/vision_date_recevier.hpp"

#include "tcp_input.hpp"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <thread>
#include <utility>

#include <yaml-cpp/yaml.h>

namespace sim_receiver {

class VisionDateRecevier::Impl {
public:
    Impl(std::string host, uint16_t port, uint16_t command_port)
        : input(std::move(host), port), command_port_(command_port) {}

    ~Impl() { stop(); }

    void overrideEndpoint(std::string host, uint16_t port) {
        stop();
        input = TcpInput(std::move(host), port);
    }

    void overrideCommandEndpoint(uint16_t port) {
        command_port_ = port;
    }

    bool connect() {
        if (worker.joinable()) {
            return connected_.load();
        }

        stop_requested_.store(false);
        {
            std::lock_guard<std::mutex> lock(mutex);
            frame_ready = false;
            latest_frame.release();
        }

        connected_.store(input.connect());
        worker = std::thread([this] { receiveLoop(); });
        if (!command_worker.joinable()) {
            command_worker = std::thread([this] { commandLoop(); });
        }
        return true;
    }

    bool connected() const {
        return connected_.load();
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
                     uint64_t* local_timestamp_us) {
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
        frame_ready = false;
        return std::move(latest_frame);
    }

private:
    void receiveLoop() {
        while (!stop_requested_.load()) {
            SimFrame frame;
            if (!input.readFrame(frame)) {
                connected_.store(false);
                input.close();
                for (int i = 0; i < 5 && !stop_requested_.load(); ++i) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(20));
                }
                continue;
            }

            connected_.store(true);
            {
                std::lock_guard<std::mutex> lock(mutex);
                latest_frame = std::move(frame.bgr);
                latest_sequence_ = frame.seq;
                latest_source_timestamp_us_ = frame.source_timestamp_us;
                latest_local_timestamp_us_ = frame.local_timestamp_us;
                frame_ready = true;
            }
            frame_cv.notify_one();
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
        frame_cv.notify_all();
        command_cv.notify_all();
        if (worker.joinable()) {
            worker.join();
        }
        if (command_worker.joinable()) {
            command_worker.join();
        }
        input.close();
    }

    TcpInput input;
    uint16_t command_port_ = 7667;
    std::thread worker;
    std::thread command_worker;
    std::mutex mutex;
    std::condition_variable frame_cv;
    std::mutex command_mutex;
    std::condition_variable command_cv;
    cv::Mat latest_frame;
    uint64_t latest_sequence_ = 0;
    uint64_t latest_source_timestamp_us_ = 0;
    uint64_t latest_local_timestamp_us_ = 0;
    bool frame_ready = false;
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
    : impl_(std::make_unique<Impl>("127.0.0.1", 7666, 7667)) {
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
    } catch (const std::exception& err) {
        std::cerr << "Failed to load simulator config: " << err.what() << std::endl;
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
                                     uint64_t* local_timestamp_us) {
    return impl_->getFrame(sequence, source_timestamp_us, local_timestamp_us);
}

} // namespace sim_receiver
