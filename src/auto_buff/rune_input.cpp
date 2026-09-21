#include "auto_buff/rune_input.hpp"

#include <cstdlib>
#include <filesystem>
#include <iostream>

#include <sim_receiver/telemetry_receiver.hpp>
#include <sim_receiver/vision_date_receiver.hpp>

#include "common/standard_clock.hpp"

namespace auto_aim::energy
{
    RuneInputSource::RuneInputSource() = default;
    RuneInputSource::~RuneInputSource() = default;

    bool RuneInputSource::open(const RuneConfig& config, const std::string& config_dir,
                               const std::string& video_path)
    {
        video_mode_ = !video_path.empty();
        if (video_mode_) {
            capture_.open(video_path);
            if (!capture_.isOpened()) {
                std::cerr << "Cannot open video: " << video_path << std::endl;
                return false;
            }
            std::cout << "Energy rune: video input " << video_path << std::endl;
            return true;
        }

        receiver_ = std::make_unique<sim_receiver::VisionDateRecevier>(
            config_dir + "/" + config.simulator_config);
        if (const char* host = std::getenv("ULTRA_VISION_SIM_HOST")) {
            receiver_->overrideEndpoint(host, 7666);
        }
        if (const char* port = std::getenv("ULTRA_VISION_SIM_PORT")) {
            const char* host = std::getenv("ULTRA_VISION_SIM_HOST");
            receiver_->overrideEndpoint(host ? host : "127.0.0.1",
                                        static_cast<uint16_t>(std::atoi(port)));
        }
        if (const char* command_port = std::getenv("ULTRA_VISION_SIM_COMMAND_PORT")) {
            receiver_->overrideCommandEndpoint(static_cast<uint16_t>(std::atoi(command_port)));
        }
        receiver_->connect();

        // 真值通道：仿真器把命中/状态行发到遥测端口，用于自动打分。
        bool telemetry_disabled = false;
        if (const char* disabled = std::getenv("ULTRA_VISION_NO_TELEMETRY")) {
            telemetry_disabled = std::string(disabled) != "0";
        }
        if (!telemetry_disabled) {
            std::string host = "127.0.0.1";
            uint16_t port = 7668;
            if (const char* value = std::getenv("ULTRA_VISION_SIM_HOST")) host = value;
            if (const char* value = std::getenv("ULTRA_VISION_TELEMETRY_PORT")) {
                port = static_cast<uint16_t>(std::atoi(value));
            }
            telemetry_ = std::make_unique<sim_receiver::TelemetryReceiver>(host, port);
            telemetry_->connect();
        }

        if (const char* dir = std::getenv("ULTRA_VISION_RUNE_FRAME_DIR")) {
            frame_dir_ = dir;
            std::filesystem::create_directories(frame_dir_);
            label_dump_.open(frame_dir_ + "/labels.txt", std::ios::out | std::ios::trunc);
            std::cout << "recording frames to " << frame_dir_ << std::endl;
        }
        if (const char* path = std::getenv("ULTRA_VISION_RUNE_GT_CSV")) {
            ground_truth_.open(path, std::ios::out | std::ios::trunc);
            if (ground_truth_) ground_truth_ << "local_time_us,line\n";
        }
        if (const char* reset = std::getenv("ULTRA_VISION_SIM_RESET")) {
            if (std::string(reset) != "0") receiver_->sendResetCommand();
        }
        receiver_->sendGimbalCommand(0.0, 0.0);
        std::cout << "Energy rune: simulator input " << config_dir << "/"
                  << config.simulator_config << std::endl;
        return true;
    }

    bool RuneInputSource::readFrame(Frame& frame)
    {
        frame = Frame{};
        if (video_mode_) {
            capture_ >> frame.bgr;
            if (frame.bgr.empty()) return false;
            frame.local_timestamp_us = StandardClock::nowUs();
            return true;
        }
        if (!receiver_) return false;
        frame.bgr = receiver_->getFrame(&frame.sequence, &frame.source_timestamp_us,
                                        &frame.local_timestamp_us);
        return !frame.bgr.empty();
    }

    std::vector<std::string> RuneInputSource::pollTelemetry()
    {
        if (!telemetry_) return {};
        return telemetry_->poll();
    }

    bool RuneInputSource::sendGimbal(double yaw, double pitch)
    {
        return receiver_ ? receiver_->sendGimbalCommand(yaw, pitch) : false;
    }

    bool RuneInputSource::sendFire()
    {
        return receiver_ ? receiver_->sendFireCommand() : false;
    }

    bool RuneInputSource::sendReset()
    {
        return receiver_ ? receiver_->sendResetCommand() : false;
    }
} // namespace auto_aim::energy
