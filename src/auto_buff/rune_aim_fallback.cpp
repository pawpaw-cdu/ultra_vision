#include "auto_buff/rune_aim_fallback.hpp"

#include <cmath>

namespace auto_aim::energy
{
    void RuneAimFallback::reset()
    {
        frames_without_control_ = 0;
        parked_frames_ = 0;
        recenter_count_ = 0;
        have_last_center_ = false;
        last_center_time_ = 0.0;
        last_center_yaw_ = 0.0;
        last_center_pitch_ = 0.0;
    }

    void RuneAimFallback::noteCenter(double yaw, double pitch, double now)
    {
        last_center_yaw_ = yaw;
        last_center_pitch_ = pitch;
        last_center_time_ = now;
        have_last_center_ = true;
    }

    RuneAimFallback::Decision RuneAimFallback::decide(double now, double commanded_yaw,
                                                      double commanded_pitch)
    {
        Decision decision;
        ++frames_without_control_;

        const bool have_recent_center =
            have_last_center_ && (now - last_center_time_) <= config_.park_on_center_max_age_s;
        if (have_recent_center) {
            decision.kind = Decision::Kind::AimLastCenter;
            decision.yaw = last_center_yaw_;
            decision.pitch = last_center_pitch_;
            ++parked_frames_;
            return decision;
        }

        if (frames_without_control_ > config_.recenter_after_frames) {
            // 连"最后已知符心"都过期了（例如从没看到过符，或超过 park 时限）：
            // 这才回标定位姿，作为最后的兜底。
            frames_without_control_ = 0;
            decision.reset_estimator = true;
            if (std::abs(commanded_yaw) > config_.home_epsilon_rad ||
                std::abs(commanded_pitch) > config_.home_epsilon_rad) {
                decision.kind = Decision::Kind::Recenter;
                decision.yaw = 0.0;
                decision.pitch = 0.0;
                ++recenter_count_;
            }
        }
        return decision;
    }
} // namespace auto_aim::energy
