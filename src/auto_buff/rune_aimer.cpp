#include "auto_buff/rune_aimer.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>

#include "auto_buff/rune_target.hpp"
#include "auto_buff/support/math.hpp"
#include "auto_buff/support/trajectory.hpp"

namespace auto_aim::energy
{
    RuneAimer::RuneAimer(const RuneAimerConfig& config) : config_(config) {}

    bool RuneAimer::sendAngles(RuneTarget& target, double predict_time, double bullet_speed,
                               const Eigen::Vector3d& plate_in_buff, double& yaw_world,
                               double& pitch_up, double& fly_time, double phase_offset) const
    {
        const double speed = bullet_speed < 10.0 ? 24.0 : bullet_speed;

        // First pass: aim where the plate is now, take its flight time.
        target.predict(predict_time);
        Eigen::Vector3d aim_in_world = target.pointBuffToWorld(plate_in_buff, phase_offset);
        double distance = std::hypot(aim_in_world[0], aim_in_world[1]);
        double height = aim_in_world[2];
        const Trajectory first(speed, distance, height, config_.gravity);
        if (first.unsolvable) return false;

        // Second pass: re-predict for that flight time and re-solve.
        target.predict(first.fly_time);
        aim_in_world = target.pointBuffToWorld(plate_in_buff, phase_offset);
        distance = std::hypot(aim_in_world[0], aim_in_world[1]);
        height = aim_in_world[2];
        const Trajectory second(speed, distance, height, config_.gravity);
        if (second.unsolvable) return false;
        if (std::abs(second.fly_time - first.fly_time) > config_.max_flight_time_error) {
            return false;
        }

        yaw_world = std::atan2(aim_in_world[1], aim_in_world[0]);
        pitch_up = second.pitch;
        fly_time = second.fly_time;
        return true;
    }

    RuneCommand RuneAimer::aim(RuneTarget& target, double timestamp, double now,
                               double detection_age_override, int slot_offset)
    {
        // `slot_offset`：瞄"跟踪片往前 k 个槽位"的那片（参考 sp_vision 的换片逻辑：
        // 命中后直接转到新亮起的那片，而不是继续盯着旧片或退回符心）。
        RuneCommand command = aimImpl(target, timestamp, now, false, detection_age_override,
                                      0.0);
        if (slot_offset == 0 || !command.control) return command;
        const double phase = slot_offset * 2.0 * kPi / 5.0;
        RuneCommand shifted = aimImpl(target, timestamp, now, false, detection_age_override,
                                      phase);
        return shifted.control ? shifted : command;
    }

    RuneCommand RuneAimer::aimCenter(RuneTarget& target, double timestamp, double now,
                                     double detection_age_override)
    {
        return aimImpl(target, timestamp, now, true, detection_age_override, 0.0);
    }

    RuneCommand RuneAimer::aimImpl(RuneTarget& target, double timestamp, double now,
                                   bool center_only, double detection_age_override,
                                   double phase_offset)
    {
        RuneCommand command;
        if (target.isUnsolvable()) return command;

        // The estimator holds the state as of the capture time; add the age of
        // that frame plus the configured extra lead.
        // 提前量：优先用调用方给的"平滑延迟"；否则用此刻的即时值。
        const double detection_age = detection_age_override >= 0.0
                                         ? detection_age_override
                                         : std::max(0.0, now - timestamp);
        // 转速快时用更大的延迟补偿（sp_vision 的 decision_speed +
        // high/low_speed_delay_time 分档），慢速时用小值避免过冲。
        const double omega = target.ekfX().size() > 6 ? std::abs(target.ekfX()[6]) : 0.0;
        const double latency = omega > config_.decision_speed_rad_s
                                   ? config_.high_speed_delay
                                   : config_.low_speed_delay;
        const double future = detection_age + latency;

        double yaw_world = 0.0;
        double pitch_up = 0.0;
        double fly_time = 0.0;
        const Eigen::Vector3d aim_point_in_buff =
            center_only ? Eigen::Vector3d::Zero()
                        : Eigen::Vector3d(0.0, 0.0, config_.target_radius_m);
        if (!sendAngles(target, future, config_.bullet_speed, aim_point_in_buff, yaw_world,
                        pitch_up, fly_time, phase_offset)) {
            return command;
        }

        // z-up world yaw grows to the left; the gimbal command turns right.
        const double yaw = -yaw_world + config_.yaw_offset_deg * kPi / 180.0;
        const double pitch = pitch_up + config_.pitch_offset_deg * kPi / 180.0;
        command.yaw = yaw;
        command.pitch = pitch;
        command.fly_time = fly_time;
        command.aim_roll = target.ekfX().size() > 5 ? target.ekfX()[5] : 0.0;

        // Feedforward: probe the same aim solution a little further ahead and
        // difference it. The plate orbits at up to 120 deg/s on the large rune,
        // so the trajectory generator needs the target's angular rate to track
        // it without lagging (this is the `yaw_vel`/`pitch_vel` that sp_vision's
        // MPC planner feeds its controller).
        {
            constexpr double kProbeDt = 0.05;
            RuneTarget probe = target;
            double probe_yaw = 0.0;
            double probe_pitch = 0.0;
            double probe_fly = 0.0;
            if (sendAngles(probe, future + kProbeDt, config_.bullet_speed,
                           aim_point_in_buff, probe_yaw, probe_pitch, probe_fly,
                           phase_offset)) {
                command.yaw_velocity = -limitRad(probe_yaw - yaw_world) / kProbeDt;
                command.pitch_velocity = (probe_pitch - pitch_up) / kProbeDt;
            }
        }

        // 输出角斜率限制：把"重捕获/换片"造成的大回弹摊成平滑过渡，避免云台猛甩。
        // 注意：只在**幅值变化大**时生效，正常跟踪不受影响；换靶时也照常允许
        // （换靶本来就该快速对准，只是不再一步跳到位）。
        if (config_.max_aim_step_deg > 0.0 && last_angle_valid_) {
            const double step = config_.max_aim_step_deg * kPi / 180.0;
            const double dy = limitRad(command.yaw - last_yaw_);
            const double dp = command.pitch - last_pitch_;
            if (std::abs(dy) > step) command.yaw = last_yaw_ + std::copysign(step, dy);
            if (std::abs(dp) > step) command.pitch = last_pitch_ + std::copysign(step, dp);
        }

        const double switch_limit = config_.switch_angle_deg * kPi / 180.0;
        const bool angle_changed =
            last_angle_valid_ && (std::abs(limitRad(last_yaw_ - yaw)) > switch_limit ||
                                  std::abs(limitRad(last_pitch_ - pitch)) > switch_limit);

        // 换叶判定（顺序即优先级）：
        //   1) 状态估计器报告的"观测接管"= 真的换叶（相位/圆心跳变被二次确认）。
        //      这才是 kSwitch 语义上的换叶：中断一帧控制并抑制开火。
        //   2) 连续 switch_confirm_frames 帧角度跳变超阈值 = 兜底（估计器没报但
        //      确实在跳），同样按换叶处理。
        //   3) 其余：角度单帧跳变交给轨迹生成器和瞄准滤波消化，**不关控制**
        //      —— 关控制会让云台"冻结"，下一帧再追，看起来就是顿挫/抖动。
        if (observed_switch_) {
            switch_fanblade_ = true;
            mistake_count_ = 0;
            command.control = false;
        } else if (angle_changed) {
            ++mistake_count_;
            if (mistake_count_ >= std::max(1, config_.switch_confirm_frames)) {
                switch_fanblade_ = true;
                command.control = false;
                if (mistake_count_ > config_.max_mistakes) {
                    // 估计器持续跳变：强制重新捕获，并恢复控制，避免卡死。
                    mistake_count_ = 0;
                    command.control = true;
                }
            } else {
                switch_fanblade_ = false;
                command.control = true;
            }
        } else {
            switch_fanblade_ = false;
            mistake_count_ = 0;
            command.control = true;
        }
        command.blade_switched = switch_fanblade_;
        last_yaw_ = yaw;
        last_pitch_ = pitch;
        last_angle_valid_ = true;

        if (!last_fire_initialized_) {
            last_fire_time_ = now - config_.fire_gap_time;
            last_fire_initialized_ = true;
        }
        if (center_only) {
            // 观测过期：只把相机稳稳停在符上，等重新捕获后再打（打过期目标只会浪费）。
            command.shoot = false;
            last_fire_time_ = now;
        } else if (switch_fanblade_) {
            // Never shoot at a plate the gimbal has not settled on yet.
            command.shoot = false;
            // 但**换靶就是新一轮**：参考实现（RuneDecisionModule）在换靶时
            // reset_cooldown()，允许立刻打新靶。原来把 last_fire_time_ 设成 now
            // 等于换靶后再等一个完整 fire_gap_time 才允许开火，实测导致近一半的
            // 弹是在本轮 2.5 s 窗口失败之后才落地（我们 48% vs 基线 18%）。
            // 这里把节流"提前到上一发之前"，下一帧只要云台到位就能打。
            last_fire_time_ = now - config_.fire_gap_time;
        } else if (now - last_fire_time_ > config_.fire_gap_time) {
            command.shoot = true;
            last_fire_time_ = now;
        }
        return command;
    }
} // namespace auto_aim::energy
