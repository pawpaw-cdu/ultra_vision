#ifndef AUTO_AIM_GIMBAL_CONTROLLER_HPP
#define AUTO_AIM_GIMBAL_CONTROLLER_HPP

#include "ultra_vision/control/gimbal_aimer.hpp"

#include <atomic>
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>

namespace ultra_vision
{
    struct GimbalControllerSnapshot {
        bool valid = false;
        bool settled = false;
        double yaw = 0.0;
        double pitch = 0.0;
        double yaw_velocity = 0.0;
        double pitch_velocity = 0.0;
        double yaw_error = 0.0;
        double pitch_error = 0.0;
        double desired_yaw = 0.0;
        double desired_pitch = 0.0;
        uint64_t target_generation = 0;
        uint64_t processed_generation = 0;
        uint64_t timestamp_us = 0;
    };

    // Runs the trajectory generator at a fixed rate and sends absolute angle
    // commands through a caller-provided transport callback. The vision loop
    // only publishes target angles, so a slow camera frame cannot make the
    // gimbal move in large discrete jumps.
    class GimbalController
    {
    public:
        using SendFunction = std::function<bool(double, double)>;

        GimbalController(const GimbalAimConfig& config,
                         SendFunction send,
                         double command_rate_hz = 100.0);
        ~GimbalController();

        GimbalController(const GimbalController&) = delete;
        GimbalController& operator=(const GimbalController&) = delete;

        void setTargetAngles(const GimbalTargetAngles& target);
        GimbalControllerSnapshot snapshot() const;
        GimbalControllerSnapshot snapshotAt(uint64_t timestamp_us) const;
        void stop();

    private:
        void commandLoop();

        GimbalAimConfig config_;
        GimbalAimer aimer_;
        SendFunction send_;
        double command_period_ = 0.01;

        mutable std::mutex target_mutex_;
        GimbalTargetAngles target_;
        uint64_t target_generation_ = 0;

        mutable std::mutex snapshot_mutex_;
        GimbalControllerSnapshot snapshot_;
        mutable std::deque<GimbalControllerSnapshot> history_;
        uint64_t processed_generation_ = 0;

        std::atomic<bool> stop_requested_{false};
        std::thread command_thread_;
    };
}

#endif
