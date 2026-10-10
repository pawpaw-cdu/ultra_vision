#include "tracker.hpp"
#include "Kalman/tracker_utils.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <optional>

using namespace auto_aim::tracker_detail;

namespace auto_aim
{
    void Tracker::noteMeasurement(bool accepted)
    {
        if (!cfg_.watchdog_enabled) return;
        if (accepted) ++accepted_since_init_;
        recent_measurements_.push_back(accepted);
        while (static_cast<int>(recent_measurements_.size()) > std::max(4, cfg_.watchdog_window)) {
            recent_measurements_.pop_front();
        }
        const int samples = static_cast<int>(recent_measurements_.size());
        int rejected = 0;
        for (bool ok : recent_measurements_) {
            if (!ok) ++rejected;
        }
        last_reject_ratio_ = samples > 0 ? static_cast<double>(rejected) / samples : 0.0;
    }

    /// @brief 收敛看门狗：量测长期被拒 或 状态物理上发散 → 直接 LOST + 下一帧强制重初始化。
    bool Tracker::watchdogTrip(const char* reason)
    {
        if (!cfg_.watchdog_enabled) return false;
        ++watchdog_resets_;
        force_reinit_ = true;
        state_ = TrackerState::LOST;
        lost_count_ = 0;
        detect_count_ = 0;
        confidence_ = 0.0;
        has_observation_ = false;
        accepted_since_init_ = 0;
        recent_measurements_.clear();
        last_reject_ratio_ = 0.0;
        std::cerr << "[watchdog] " << reason << " → 重新初始化（第 " << watchdog_resets_ << " 次）"
                  << std::endl;
        return true;
    }

    Tracker::WatchdogStats Tracker::watchdogStats() const
    {
        WatchdogStats stats;
        stats.resets = watchdog_resets_;
        stats.reject_ratio = last_reject_ratio_;
        stats.samples = static_cast<int>(recent_measurements_.size());
        // "已收敛" = 自上次（重新）初始化以来接受了足够多帧量测 + 状态没越界。
        // 对齐 sp_vision 的 Target::convergened()（`update_count_ > 3 && !diverged()`）：
        // 刚入靶/刚重捕的头几帧 ω 不可信，不允许拿它做提前量（见 TargetEstimate::converged）。
        // 注意**不要**把 `state == TRACKING` 放进判据：TEMPD_LOST 每十几帧出现一次，
        // 会让这个开关跟着抖，反而把提前量开关抖成 5°/s 级的震荡。
        if (!cfg_.watchdog_enabled) {
            stats.converged = true;
            return stats;
        }
        stats.converged = accepted_since_init_ >= cfg_.watchdog_min_converge_updates &&
            !stateDiverged();
        return stats;
    }

    /// @brief 状态发散检查（速度/角速度/位置/位置协方差越界）。
    bool Tracker::stateDiverged() const
    {
        if (!ekf_.isInit()) return true;
        const auto state = ekf_.getState();
        const double speed = std::sqrt(state(1, 0) * state(1, 0) + state(3, 0) * state(3, 0) +
                                       state(8, 0) * state(8, 0));
        const double yaw_rate = std::abs(state(6, 0));
        const double range = std::sqrt(state(0, 0) * state(0, 0) + state(2, 0) * state(2, 0) +
                                       state(7, 0) * state(7, 0));
        const double position_cov = std::max({ekf_.positionVariance(0), ekf_.positionVariance(1),
                                              ekf_.positionVariance(2)});
        return speed > cfg_.watchdog_max_speed || yaw_rate > cfg_.watchdog_max_yaw_rate ||
               range > cfg_.watchdog_max_range || position_cov > cfg_.watchdog_max_position_cov;
    }

}
