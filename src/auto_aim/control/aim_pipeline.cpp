#include "aim_pipeline.hpp"

#include <algorithm>
#include <cmath>

namespace auto_aim
{
    AimPipeline::AimPipeline(const AimPipelineConfig& config, SendFn send)
        : config_(config), send_(std::move(send)), tracker_(config.tracker),
          selector_(config.selector),
          bootstrap_aimer_(config.gimbal), aim_filter_(config.aim_filter),
          // 控制器只给 (yaw, pitch)，这里把"要不要控制/要不要开火/前馈角速度"
          // 用原子量补上 —— 控制线程 100 Hz 独立跑，主循环不做任何阻塞操作。
          controller_(
              config.gimbal,
              [this](double yaw, double pitch) {
                  const bool control = control_enabled_.load(std::memory_order_relaxed);
                  const bool fire = control && fire_request_.load(std::memory_order_relaxed);
                  return send_(control, fire, yaw,
                              desired_yaw_vel_.load(std::memory_order_relaxed), 0.0, pitch,
                              desired_pitch_vel_.load(std::memory_order_relaxed), 0.0);
              },
              config.command_rate_hz),
          shooter_(config.shooter)
    {
        aim_jump_limit_ = config_.aim_jump_limit_deg > 0.0
            ? config_.aim_jump_limit_deg * M_PI / 180.0 : 1e9;
        last_heartbeat_ = std::chrono::steady_clock::now();
    }

    AimPipeline::Outcome AimPipeline::update(const FrameInput& input)
    {
        Outcome outcome;
        control_enabled_.store(input.allow_control, std::memory_order_relaxed);
        // ---- 1. 估计：只有拿到**新**观测才更新；否则保持状态并让选择器按年龄外推 ----
        // （异步神经网络跟不上相机帧率时这是常态，不是异常。）
        if (input.fresh) {
            tracker_.setCameraPose(input.pose);
            tracker_.update(input.armors, input.observation_time);
            last_detection_time_ = input.observation_time;
        }
        const double now = input.now > 0.0 ? input.now : input.observation_time;
        if (input.bullet_speed > 0.0) selector_.setProjectileSpeed(input.bullet_speed);

        const double current_yaw = input.pose.yaw;
        const double current_pitch = input.pose.pitch;
        outcome.track_usable =
            tracker_.getState() == TrackerState::TRACKING ||
            (tracker_.getState() == TrackerState::TEMP_LOST &&
             tracker_.getLostCount() <= config_.max_control_lost_frames);

        // ---- 2. 选板 + 提前量 ----
        TargetDecision decision;
        if (outcome.track_usable && config_.selector.enabled) {
            TargetEstimate estimate;
            estimate.center = tracker_.getTargetCenterWorldArray();
            estimate.velocity = tracker_.getTargetVelocityWorld();
            estimate.yaw = tracker_.getYawWorld();
            estimate.omega = tracker_.getOmega();
            estimate.armor_radius = tracker_.getArmorRadius();
            // 观测年龄 = 现在 − 最后一次**新观测**的时刻。异步检测、慢推理、
            // 仿真里的"曝光→到手"延迟都体现在这一项里（时间基由调用方给）。
            estimate.age = last_detection_time_ > 0.0
                ? std::max(0.0, now - last_detection_time_) : 0.0;
            decision = selector_.select(estimate, current_yaw, current_pitch);
        }
        outcome.decision = decision;

        // ---- 3. 瞄点：稳定跟踪用回归滤波，刚入靶用观测方向直接解 ----
        GimbalTargetAngles desired;
        if (decision.valid) {
            const auto filtered = aim_filter_.update(
                decision.target_yaw, decision.target_pitch, decision.armor_id, now);
            desired.valid = filtered.valid;
            desired.yaw = filtered.yaw;
            desired.pitch = filtered.pitch;
            desired.velocity_valid = true;
            desired.yaw_velocity = decision.target_yaw_velocity;
            desired.pitch_velocity = decision.target_pitch_velocity;
        } else if (tracker_.hasObservation()) {
            const auto& observed = tracker_.getLastObservedArmor();
            if (observed.tvec.rows == 3 && observed.solve_result) {
                const std::array<double, 3> target_camera{{
                    observed.tvec.at<double>(0),
                    observed.tvec.at<double>(1),
                    observed.tvec.at<double>(2),
                }};
                desired = bootstrap_aimer_.solveTargetAngles(target_camera, current_yaw,
                                                            current_pitch);
                desired.velocity_valid = false;
                aim_filter_.reset();
            }
        }

        // ---- 4. 反跑飞守卫 + 下发 ----
        const double tolerance = config_.fire_angle_tolerance_deg * M_PI / 180.0;
        ShootInput shoot_input;
        if (desired.valid) {
            const double jump_yaw = std::abs(
                GimbalAimer::normalizeAngle(desired.yaw - current_yaw));
            const double jump_pitch = std::abs(desired.pitch - current_pitch);
            // 只在"估计不可信"时才拦（未稳定跟踪）。稳定跟踪时瞄点偏轴 30°+ 是合法的
            // （近距靶在画面边缘就有这么大），无条件按住反而让画面更偏。
            const bool unverified = tracker_.getState() != TrackerState::TRACKING;
            if (unverified && (jump_yaw > aim_jump_limit_ || jump_pitch > aim_jump_limit_)) {
                outcome.held_for_jump = true;
            } else {
                const auto aim_snapshot = controller_.snapshot();
                const bool continuous = aim_snapshot.valid &&
                    std::abs(GimbalAimer::normalizeAngle(
                        desired.yaw - aim_snapshot.desired_yaw)) <= tolerance &&
                    std::abs(desired.pitch - aim_snapshot.desired_pitch) <= tolerance;
                outcome.aim_ready = aim_snapshot.valid && continuous &&
                    std::abs(aim_snapshot.yaw_error) <= tolerance &&
                    std::abs(aim_snapshot.pitch_error) <= tolerance;
                outcome.aim_yaw_error = aim_snapshot.yaw_error;
                outcome.aim_pitch_error = aim_snapshot.pitch_error;
                shoot_input.solved = true;
                shoot_input.ballistics_valid = aim_snapshot.valid;
                shoot_input.yaw_error = aim_snapshot.yaw_error;
                shoot_input.pitch_error = aim_snapshot.pitch_error;
                shoot_input.target_distance = cv::norm(tracker_.getTargetCenter());
                const auto velocity = tracker_.getTargetVelocity();
                shoot_input.target_angular_rate =
                    std::hypot(velocity[0], velocity[2]) /
                    std::max(0.5, shoot_input.target_distance);
                desired_yaw_vel_.store(decision.target_yaw_velocity,
                                       std::memory_order_relaxed);
                desired_pitch_vel_.store(decision.target_pitch_velocity,
                                         std::memory_order_relaxed);
                controller_.setTargetAngles(desired);
            }
        }
        if (!desired.valid) {
            desired_yaw_vel_.store(0.0, std::memory_order_relaxed);
            desired_pitch_vel_.store(0.0, std::memory_order_relaxed);
            // 无目标心跳：让下位机知道视觉还活着（mode=0 = 不控制）
            if (config_.heartbeat_hz > 0.0) {
                const auto now = std::chrono::steady_clock::now();
                const double period = 1.0 / config_.heartbeat_hz;
                if (std::chrono::duration<double>(now - last_heartbeat_).count() >= period) {
                    last_heartbeat_ = now;
                    fire_request_.store(false, std::memory_order_relaxed);
                    send_(false, false, input.pose.yaw, 0.0, 0.0, input.pose.pitch, 0.0, 0.0);
                }
            }
        }

        // ---- 5. 开火决策（判据在 ShootEvaluator，一处裁决） ----
        const auto shooter_output = shooter_.update(decision, outcome.aim_ready, now,
                                                    shoot_input);
        outcome.fire = input.fire_enabled && shooter_output.fire;
        outcome.fired_now = outcome.fire && !last_fire_;
        last_fire_ = outcome.fire;
        fire_request_.store(outcome.fire, std::memory_order_relaxed);
        last_outcome_ = outcome;
        return outcome;
    }
} // namespace auto_aim
