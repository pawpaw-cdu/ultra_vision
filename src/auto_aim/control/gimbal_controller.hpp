#ifndef AUTO_AIM_GIMBAL_CONTROLLER_HPP
#define AUTO_AIM_GIMBAL_CONTROLLER_HPP

#include "gimbal_aimer.hpp"

#include <atomic>
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>

namespace auto_aim
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
        // 目标角提供者（借鉴 sp_vision 的 commandgener：瞄准/弹道解算放在**控制线程**
        // 里按控制频率重算，并用"当前时刻"预测提前量，而不是每帧算一次）。
        // 返回 false 表示本周期没有可用目标 → 本周期不下发（云台保持）。
        using TargetProvider = std::function<bool(GimbalTargetAngles&, double now_seconds)>;

        GimbalController(const GimbalAimConfig& config,
                         SendFunction send,
                         double command_rate_hz = 100.0);
        ~GimbalController();

        GimbalController(const GimbalController&) = delete;
        GimbalController& operator=(const GimbalController&) = delete;

        void setTargetAngles(const GimbalTargetAngles& target);
        /// @brief 设置目标角提供者；设置后每个控制周期都会调用它重新取目标角
        ///        （与 setTargetAngles 二选一：provider 优先）。
        void setTargetProvider(TargetProvider provider);
        GimbalControllerSnapshot snapshot() const;
        GimbalControllerSnapshot snapshotAt(uint64_t timestamp_us) const;
        void stop();

    private:
        void commandLoop();

        GimbalAimConfig config_;
        GimbalAimer aimer_;
        SendFunction send_;
        TargetProvider provider_;
        mutable std::mutex provider_mutex_;
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
