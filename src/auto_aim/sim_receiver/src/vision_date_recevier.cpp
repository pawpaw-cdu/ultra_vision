#include "sim_receiver/vision_date_recevier.hpp"

#include "tcp_input.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>

#include <yaml-cpp/yaml.h>

namespace sim_receiver {

class VisionDateRecevier::Impl {
public:
    Impl(std::string host, uint16_t port)
        : input(std::move(host), port) {}

    ~Impl() { stop(); }

    void overrideEndpoint(std::string host, uint16_t port) {
        stop();
        input = TcpInput(std::move(host), port);
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
        if (!fire_worker.joinable()) {
            fire_worker = std::thread([this] { fireLoop(); });
        }
        return true;
    }

    bool connected() const {
        return connected_.load();
    }

    bool sendFireCommand() {
        if (stop_requested_.load()) return false;
        {
            std::lock_guard<std::mutex> lock(fire_mutex);
            fire_pending = true;
        }
        fire_cv.notify_one();
        return true;
    }

    cv::Mat getFrame() {
        std::unique_lock<std::mutex> lock(mutex);
        frame_cv.wait_for(lock, std::chrono::milliseconds(100), [this] {
            return stop_requested_.load() || frame_ready;
        });
        if (stop_requested_.load() || !frame_ready) {
            return cv::Mat();
        }
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
                frame_ready = true;
            }
            frame_cv.notify_one();
        }
    }

    void fireLoop() {
        while (!stop_requested_.load()) {
            bool pending = false;
            {
                std::unique_lock<std::mutex> lock(fire_mutex);
                fire_cv.wait(lock, [this] {
                    return stop_requested_.load() || fire_pending;
                });
                if (stop_requested_.load()) break;
                pending = fire_pending;
                fire_pending = false;
            }

            // Keep socket setup and connect delays off the image-processing loop.
            if (pending && !input.sendLine(7667, "FIRE\n")) {
                std::cerr << "Failed to send simulator fire command" << std::endl;
            }
        }
    }

    void stop() {
        stop_requested_.store(true);
        frame_cv.notify_all();
        fire_cv.notify_all();
        if (worker.joinable()) {
            worker.join();
        }
        if (fire_worker.joinable()) {
            fire_worker.join();
        }
        input.close();
    }

    TcpInput input;
    std::thread worker;
    std::thread fire_worker;
    std::mutex mutex;
    std::condition_variable frame_cv;
    std::mutex fire_mutex;
    std::condition_variable fire_cv;
    cv::Mat latest_frame;
    bool frame_ready = false;
    bool fire_pending = false;
    std::atomic<bool> stop_requested_{false};
    std::atomic<bool> connected_{false};
};

VisionDateRecevier::VisionDateRecevier(const std::string& config_path)
    : impl_(std::make_unique<Impl>("127.0.0.1", 7666)) {
    try {
        YAML::Node node = YAML::LoadFile(config_path);
        if (node["host"]) {
            overrideEndpoint(node["host"].as<std::string>(), 7666);
        }
        if (node["port"]) {
            overrideEndpoint(node["host"] ? node["host"].as<std::string>() : "127.0.0.1",
                             static_cast<uint16_t>(node["port"].as<int>()));
        }
    } catch (const std::exception& err) {
        std::cerr << "Failed to load simulator config: " << err.what() << std::endl;
    }
}

VisionDateRecevier::~VisionDateRecevier() = default;

void VisionDateRecevier::overrideEndpoint(const std::string& host, uint16_t port) {
    impl_->overrideEndpoint(host, port);
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

cv::Mat VisionDateRecevier::getFrame() {
    return impl_->getFrame();
}

} // namespace sim_receiver
