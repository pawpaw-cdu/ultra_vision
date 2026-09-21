#include "auto_buff/rune_aim_bridge.hpp"

#include <algorithm>
#include <iostream>

#include "common/standard_clock.hpp"
#include "auto_buff/support/math.hpp"

namespace auto_aim::energy
{
    RuneAimBridge::RuneAimBridge(const RuneConfig& config, RuneAimer& aimer)
        : config_(config), aimer_(aimer),
          aim_buffer_(config.mode, config.max_coast_frames, config.max_distance_jump_ratio,
                      config.max_center_jump_deg)
    {
    }

    void RuneAimBridge::install(GimbalController& controller, AimSignalFilter& filter,
                                bool fire_enabled, std::function<void()> fire)
    {
        controller_ = &controller;
        filter_ = &filter;
        fire_enabled_ = fire_enabled;
        fire_ = std::move(fire);
        controller.setTargetProvider(
            [this](GimbalTargetAngles& out, double now_seconds) {
                return provide(out, now_seconds);
            });
    }

    void RuneAimBridge::publish(const Input& input)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        shared_.input = input;
    }

    RuneCommand RuneAimBridge::lastCommand() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (shared_.command_valid) return shared_.command;
        RuneCommand empty;   // control=false：帧循环据此跳过开火/记录 0
        return empty;
    }

    bool RuneAimBridge::consumeFireRequest()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const bool request = shared_.fire_request;
        shared_.fire_request = false;
        return request;
    }

    void RuneAimBridge::clear()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        shared_ = Shared{};
    }

    bool RuneAimBridge::provide(GimbalTargetAngles& out, double now_seconds)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!shared_.input.has_value()) return false;
        const Input& input = *shared_.input;

        if (!input.valid) {
            if (!input.park) return false;
            out.valid = true;
            out.yaw = input.park_yaw;
            out.pitch = input.park_pitch;
            out.yaw_velocity = 0.0;
            out.pitch_velocity = 0.0;
            return true;
        }

        // 必须用**副本**调用：aim()/aimCenter() 会原地 predict() 推进状态，
        // 用同一个 buffer 连续调用会把相位反复外推（实测到位误差 p90 从 0.8° 涨到 14.9°）。
        auto_aim::energy::RuneTarget working = input.target;
        RuneCommand cmd = input.center_only
                              ? aimer_.aimCenter(working, input.frame_time, now_seconds,
                                                 input.latency)
                              : aimer_.aim(working, input.frame_time, now_seconds, input.latency,
                                           input.slot_offset);
        shared_.command = cmd;
        shared_.command_valid = cmd.control;
        if (cmd.shoot) shared_.shoot_latched = true;
        if (cmd.shoot && input.engageable) shared_.fire_request = true;
        if (!cmd.control) return false;

        const double yaw_limit = config_.aim_yaw_limit_deg * kPi / 180.0;
        const double pitch_limit = config_.aim_pitch_limit_deg * kPi / 180.0;
        if (std::abs(cmd.yaw) > yaw_limit || std::abs(cmd.pitch) > pitch_limit) return false;

        out.valid = true;
        out.yaw = cmd.yaw;
        out.pitch = cmd.pitch;
        const double velocity_limit = config_.max_target_velocity_deg_s * kPi / 180.0;
        out.yaw_velocity = std::max(-velocity_limit, std::min(velocity_limit, cmd.yaw_velocity));
        out.pitch_velocity = std::max(-velocity_limit, std::min(velocity_limit, cmd.pitch_velocity));

        if (filter_ != nullptr && config_.aim_filter.enabled) {
            const auto filtered = filter_->update(cmd.yaw, cmd.pitch, 0, now_seconds);
            if (filtered.valid) {
                out.yaw = filtered.yaw;
                out.pitch = filtered.pitch;
                out.yaw_velocity = filtered.yaw_velocity;
                out.pitch_velocity = filtered.pitch_velocity;
            }
        }

        // 开火判定（在控制线程里做，sp_vision commandgener + shooter 同构）：
        // cmd.shoot（火控节流）、帧级可打性、云台到位误差、命中后保持、换叶，
        // 必须是**同一时刻**的值；放到 30 Hz 帧循环里采样，这个"与"几乎凑不齐。
        if (controller_ != nullptr && fire_ != nullptr && cmd.shoot && input.engageable &&
            !input.hold && !cmd.blade_switched && fire_enabled_) {
            const auto snapshot = controller_->snapshot();
            const double fire_thresh = config_.fire_thresh_deg * kPi / 180.0;
            const double yaw_error = std::abs(limitRad(out.yaw - snapshot.yaw));
            const double pitch_error = std::abs(out.pitch - snapshot.pitch);
            if (snapshot.valid && yaw_error <= fire_thresh && pitch_error <= fire_thresh) {
                shared_.fire_request = true;
            }
        }
        return true;
    }
} // namespace auto_aim::energy
