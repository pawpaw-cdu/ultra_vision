#include "ultra_vision/control/gimbal_controller.hpp"

#include "ultra_vision/core/standard_clock.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iterator>
#include <utility>

namespace ultra_vision
{
    GimbalController::GimbalController(const GimbalAimConfig& config,
                                       SendFunction send,
                                       double command_rate_hz)
        : config_(config), aimer_(config), send_(std::move(send))
    {
        if (!std::isfinite(command_rate_hz) || command_rate_hz <= 0.0) {
            command_rate_hz = 100.0;
        }
        command_period_ = 1.0 / command_rate_hz;
        command_thread_ = std::thread(&GimbalController::commandLoop, this);
    }

    GimbalController::~GimbalController()
    {
        stop();
    }

    void GimbalController::setTargetAngles(const GimbalTargetAngles& target)
    {
        if (!target.valid ||
            !std::isfinite(target.yaw) ||
            !std::isfinite(target.pitch)) {
            return;
        }

        GimbalTargetAngles bounded_target = target;
        bounded_target.yaw = GimbalAimer::normalizeAngle(target.yaw);
        bounded_target.pitch = std::max(
            config_.pitch_min, std::min(target.pitch, config_.pitch_max));

        std::lock_guard<std::mutex> lock(target_mutex_);
        if (target_.valid) {
            const double yaw_delta = GimbalAimer::normalizeAngle(
                bounded_target.yaw - target_.yaw);
            const double pitch_delta = bounded_target.pitch - target_.pitch;
            if (std::abs(yaw_delta) < 1e-4 && std::abs(pitch_delta) < 1e-4) {
                return;
            }
        }
        target_ = bounded_target;
        ++target_generation_;
    }

    GimbalControllerSnapshot GimbalController::snapshot() const
    {
        GimbalControllerSnapshot result;
        {
            std::lock_guard<std::mutex> lock(snapshot_mutex_);
            result = snapshot_;
        }
        {
            std::lock_guard<std::mutex> lock(target_mutex_);
            result.target_generation = target_generation_;
            if (result.processed_generation != target_generation_) {
                result.settled = false;
            }
        }
        return result;
    }

    GimbalControllerSnapshot GimbalController::snapshotAt(uint64_t timestamp_us) const
    {
        std::lock_guard<std::mutex> lock(snapshot_mutex_);
        if (history_.empty()) return snapshot_;

        auto upper = std::upper_bound(
            history_.begin(), history_.end(), timestamp_us,
            [](uint64_t timestamp, const GimbalControllerSnapshot& entry) {
                return timestamp < entry.timestamp_us;
            });
        if (upper == history_.begin()) return history_.front();
        if (upper == history_.end()) return history_.back();

        const auto next = upper;
        const auto previous = std::prev(upper);
        const uint64_t interval = next->timestamp_us - previous->timestamp_us;
        if (interval == 0 || interval > 100000) return *previous;

        const double ratio = std::max(
            0.0, std::min(1.0,
                static_cast<double>(timestamp_us - previous->timestamp_us) /
                    static_cast<double>(interval)));
        GimbalControllerSnapshot result = *previous;
        result.yaw = GimbalAimer::normalizeAngle(
            previous->yaw + ratio * GimbalAimer::normalizeAngle(
                next->yaw - previous->yaw));
        result.pitch = previous->pitch +
            ratio * (next->pitch - previous->pitch);
        result.desired_yaw = GimbalAimer::normalizeAngle(
            previous->desired_yaw + ratio * GimbalAimer::normalizeAngle(
                next->desired_yaw - previous->desired_yaw));
        result.desired_pitch = previous->desired_pitch +
            ratio * (next->desired_pitch - previous->desired_pitch);
        return result;
    }

    void GimbalController::stop()
    {
        stop_requested_.store(true);
        if (command_thread_.joinable()) {
            command_thread_.join();
        }
    }

    void GimbalController::commandLoop()
    {
        using Clock = std::chrono::steady_clock;
        auto next_tick = Clock::now();
        auto last_tick = next_tick;

        while (!stop_requested_.load()) {
            next_tick += std::chrono::duration_cast<Clock::duration>(
                std::chrono::duration<double>(command_period_));
            std::this_thread::sleep_until(next_tick);
            const auto now = Clock::now();
            const double dt = std::chrono::duration<double>(now - last_tick).count();
            last_tick = now;

            GimbalTargetAngles target;
            uint64_t generation = 0;
            {
                std::lock_guard<std::mutex> lock(target_mutex_);
                target = target_;
                generation = target_generation_;
            }
            if (!target.valid) continue;

            GimbalControllerSnapshot previous;
            {
                std::lock_guard<std::mutex> lock(snapshot_mutex_);
                previous = snapshot_;
            }
            const GimbalCommand command = aimer_.trackTargetAngles(
                target, previous.yaw, previous.pitch, dt);
            if (!command.valid || !send_ ||
                !send_(command.yaw, command.pitch)) {
                continue;
            }

            uint64_t latest_generation = 0;
            {
                std::lock_guard<std::mutex> lock(target_mutex_);
                latest_generation = target_generation_;
            }

            {
                std::lock_guard<std::mutex> lock(snapshot_mutex_);
                processed_generation_ = generation;
                snapshot_.valid = true;
                snapshot_.settled = command.settled && generation == latest_generation;
                snapshot_.yaw = command.yaw;
                snapshot_.pitch = command.pitch;
                snapshot_.yaw_velocity = command.yaw_velocity;
                snapshot_.pitch_velocity = command.pitch_velocity;
                snapshot_.yaw_error = command.yaw_error;
                snapshot_.pitch_error = command.pitch_error;
                snapshot_.desired_yaw = command.desired_yaw;
                snapshot_.desired_pitch = command.desired_pitch;
                snapshot_.target_generation = latest_generation;
                snapshot_.processed_generation = processed_generation_;
                snapshot_.timestamp_us = StandardClock::nowUs();
                history_.push_back(snapshot_);
                while (history_.size() > 2000) history_.pop_front();
            }
        }
    }
}
