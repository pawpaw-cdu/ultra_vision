#include "auto_buff/control/rune_round_guard.hpp"

#include <algorithm>

namespace auto_aim::energy
{
    void RuneRoundGuard::reset()
    {
        snapshot_ = Snapshot{};
        missing_blade_frames_ = 0;
        round_start_ = 0.0;
        have_blade_position_ = false;
        hold_ = false;
        hold_phase_ = std::numeric_limits<double>::quiet_NaN();
        hold_start_time_ = 0.0;
        class_zero_frames_ = 0;
        class_phase_ = std::numeric_limits<double>::quiet_NaN();
    }

    void RuneRoundGuard::noteScoredHit(double now)
    {
        // 命中即开新一轮：仿真的 build_new_round 立刻换靶，不等"点亮消失"。
        round_start_ = now;
        snapshot_.blade_lit = true;
        missing_blade_frames_ = 0;
    }

    void RuneRoundGuard::update(const Frame& frame, double now)
    {
        snapshot_.class_hit = false;

        if (frame.solved) {
            // ---- 本轮存活与换片 ------------------------------------------
            const bool blade_jumped =
                have_blade_position_ &&
                cv::norm(frame.blade_center - last_blade_position_) > config_.blade_slot_jump_px;
            if (!snapshot_.blade_lit || blade_jumped) round_start_ = now;
            last_blade_position_ = frame.blade_center;
            have_blade_position_ = true;
            snapshot_.blade_lit = true;
            missing_blade_frames_ = 0;

            // ---- 类别命中：同一片靶 先 class0(连续) 再 class1/2 ------------
            if (frame.blade_class >= 0 && std::isfinite(frame.aim_phase)) {
                bool same_blade = std::isfinite(class_phase_);
                if (same_blade) {
                    double delta = std::fmod(std::abs(frame.aim_phase - class_phase_),
                                             config_.phase_slot_rad);
                    if (delta > 0.5 * config_.phase_slot_rad) {
                        delta = config_.phase_slot_rad - delta;
                    }
                    same_blade = delta < 0.25 * config_.phase_slot_rad;
                }
                if (!same_blade) {
                    class_phase_ = frame.aim_phase;
                    class_zero_frames_ = 0;
                }
                if (frame.blade_class == 0) {
                    ++class_zero_frames_;
                } else if (class_zero_frames_ >= 2) {
                    snapshot_.class_hit = true;
                    class_zero_frames_ = 0;
                } else {
                    class_zero_frames_ = 0;
                }
            } else {
                class_zero_frames_ = 0;
            }

            // ---- 命中后保持：直到相位跳变确认换到了另一片 ------------------
            if (frame.scored_hit) {
                hold_ = true;
                hold_phase_ = frame.aim_phase;
                hold_start_time_ = now;
            } else if (hold_) {
                // 解禁条件：**确认换片**（估计器观测接管 / 相位整跳），而不是
                const bool switched = frame.blade_switched;
                // 稳定 class-0：同一片连续 2 帧读成未激活；且（默认）要求同时
                const bool witness_ok =
                    !config_.require_active_witness || frame.active_witness > 0;
                const bool stable_inactive = class_zero_frames_ >= 2 && witness_ok;
                const bool timed_out = config_.post_hit_hold_max_s > 0.0 &&
                                       (now - hold_start_time_) > config_.post_hit_hold_max_s;
                if (switched || stable_inactive || timed_out) {
                    hold_ = false;
                }
            }
        } else {
            if (snapshot_.blade_lit &&
                ++missing_blade_frames_ > config_.round_end_frames) {
                snapshot_.blade_lit = false;
            }
            class_zero_frames_ = 0;
        }

        snapshot_.round_age_s = snapshot_.blade_lit ? now - round_start_ : 1e9;
        snapshot_.hold = hold_;
    }
} // namespace auto_aim::energy
