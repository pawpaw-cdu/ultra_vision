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
        // ---- 接管闸门：**必须"现在确实有可用目标"才允许下发控制帧** ----
        // 之前这里只判 input.allow_control（=下位机在自瞄模式），于是没有目标时
        // 控制器线程仍以 100 Hz 发 mode=1，把**上一次的目标角**当命令发下去：
        // 云台就朝一个过期目标一路走（实测：全程 armors=0 时 sent 仍到 118 Hz，
        // 云台被带到远处角度/顶到 pitch 限位）。没有可用目标时只发 mode=0 心跳。
        control_enabled_.store(input.allow_control && outcome.track_usable,
                               std::memory_order_relaxed);

        // ---- 2. 选板 + 提前量 ----
        TargetDecision decision;
        if (outcome.track_usable && config_.selector.enabled) {
            TargetEstimate estimate;
            estimate.center = tracker_.getTargetCenterWorldArray();
            estimate.velocity = tracker_.getTargetVelocityWorld();
            estimate.yaw = tracker_.getYawWorld();
            estimate.omega = tracker_.getOmega();
            estimate.armor_radius = tracker_.getArmorRadius();
            // 未收敛（刚入靶/刚重捕）时不允许用 ω 做提前量，见 TargetEstimate::converged。
            estimate.converged = tracker_.watchdogStats().converged;
            // 观测年龄 = 现在 − 最后一次**新观测**的时刻。异步检测、慢推理、
            // 仿真里的"曝光→到手"延迟都体现在这一项里（时间基由调用方给）。
            estimate.age = last_detection_time_ > 0.0
                ? std::max(0.0, now - last_detection_time_) : 0.0;
            decision = selector_.select(estimate, current_yaw, current_pitch);
        }
        outcome.decision = decision;

        // ---- 3. 瞄点：稳定跟踪用回归滤波，刚入靶用观测方向直接解 ----
        // 静止目标 + 有新观测：**直接瞄"刚观测到的那块板"**，不走选板器。
        // 为什么（2026-09-29 实测，`~/t1` 录制 35 帧）：选板器要靠模型 yaw 状态推算四块板
        // 各自的角度，而 yaw 状态在刚入靶/刚重捕时还没收敛 → 它挑到相邻 90° 的那块板
        // （obs_id=2 却 aim_id=1，35 帧只有 4 帧瞄对），云台就去追不存在的板 → 板出画（转飞）。
        // 观测到的那块板本身就是最准的瞄点（硬约束也把模型该板钉在它上面），静止时不需要
        // "挑下一块"这种逻辑，也不需要 yaw 状态。旋转靶仍走下面的选板器。
        GimbalTargetAngles desired;
        bool aimed_at_observed = false;
        // 没有**新观测**的这一帧（检测断续很常见）：不要用滑行的模型状态重新"选板" ——
        // 那正是"识别一断就跳到相邻板、云台瞎转"的来源。保持上一拍算好的瞄点即可。
        if (outcome.track_usable && !input.fresh && have_last_desired_ && last_desired_.valid) {
            desired = last_desired_;
            aimed_at_observed = true;
        }
        // ---- 静止档：按板号锁定，锁不到就保持上一拍，避免两块静止板之间来回抖 ----
        // 判据来源（2026-10-20 现场反馈）：画面里有两块**静止**板时，关联层给出的
        // "最后观测板"会在两块之间交替；原来静止档直接瞄 lastObserved ⇒ 瞄点每帧在
        // 两块板之间跳。静止靶本来不需要换板：锁住一块，看到另一块也不动。
        //   · 首次进入静止档（或离开静止档）→ 锁当前观测板；
        //   · 观测到的是锁定板 → 用它的 tvec 过一遍 aim_filter（同一 id 平滑，不重置）；
        //   · 观测到别的板 → **保持上一拍瞄点**；锁定板连续丢失超过阈值才换锁。
        const bool still_now = outcome.track_usable && tracker_.hasObservation() &&
            selector_.targetStill();
        if (!still_now) {
            still_was_active_ = false;
            still_lock_id_ = -1;
            still_lock_lost_ = 0;
        }
        if (!aimed_at_observed && still_now) {
            const auto& observed = tracker_.getLastObservedArmor();
            if (observed.solve_result && observed.tvec.rows == 3) {
                constexpr int kStillLockLostFrames = 20;   // ≈0.2~0.3 s（60~100 fps）
                const int observed_id = tracker_.getLastObservedPlateId();
                if (!still_was_active_ || still_lock_id_ < 0) {
                    still_lock_id_ = observed_id;
                    still_lock_lost_ = 0;
                }
                const bool locked_observed = observed_id == still_lock_id_;
                if (locked_observed) {
                    still_lock_lost_ = 0;
                } else if (++still_lock_lost_ > kStillLockLostFrames) {
                    // 锁定的板太久没看到（真的换板/出了视野）→ 重新锁定
                    still_lock_id_ = observed_id;
                    still_lock_lost_ = 0;
                }
                std::array<double, 3> aim_camera{{
                    observed.tvec.at<double>(0),
                    observed.tvec.at<double>(1),
                    observed.tvec.at<double>(2),
                }};
                const double distance = std::sqrt(
                    aim_camera[0] * aim_camera[0] + aim_camera[1] * aim_camera[1] +
                    aim_camera[2] * aim_camera[2]);
                const double flight_time =
                    distance / std::max(1.0, selector_.projectileSpeed());
                // 弹道下坠补偿：相机系 y 向下，下坠要让瞄点**抬高** → y 减小。
                aim_camera[1] -= 0.5 * 9.81 * flight_time * flight_time;
                const auto solved =
                    bootstrap_aimer_.solveTargetAngles(aim_camera, current_yaw, current_pitch);
                if (solved.valid && observed_id == still_lock_id_) {
                    // 用锁定板的观测更新瞄点；同一板号走 aim_filter 平滑（不再每帧 reset）
                    const auto filtered = aim_filter_.update(
                        solved.yaw, solved.pitch, observed_id, now);
                    desired.valid = filtered.valid;
                    desired.yaw = filtered.yaw;
                    desired.pitch = filtered.pitch;
                    desired.yaw_velocity = 0.0;
                    desired.pitch_velocity = 0.0;
                    aimed_at_observed = true;
                } else if (have_last_desired_ && last_desired_.valid) {
                    // 看到的是"另一块静止板"→ 保持上一拍瞄点，绝不跟着跳
                    desired = last_desired_;
                    aimed_at_observed = true;
                }
                still_was_active_ = true;
            }
        }
        // "没合格板"的触发条件（与策略无关，统计/遥测用）
        outcome.no_valid_plate = outcome.track_usable && config_.selector.enabled &&
            !decision.valid;
        outcome.gave_up = outcome.no_valid_plate && config_.no_valid_armor_policy != 0;
        if (!aimed_at_observed && decision.valid) {
            const auto filtered = aim_filter_.update(
                decision.target_yaw, decision.target_pitch, decision.armor_id, now);
            desired.valid = filtered.valid;
            desired.yaw = filtered.yaw;
            desired.pitch = filtered.pitch;
            desired.velocity_valid = true;
            desired.yaw_velocity = decision.target_yaw_velocity;
            desired.pitch_velocity = decision.target_pitch_velocity;
        } else if (!aimed_at_observed && !outcome.gave_up && tracker_.hasObservation()) {
            // 兜底分支：选板器没给出合格板时**要不要**退回去瞄"最后观测到的那块板"。
            // no_valid_armor_policy=1（放弃）时整段跳过 —— desired 保持 invalid，云台就
            // 停在上一拍瞄点上，与参考实现"没合格板就 return false"一致。
            const auto& observed = tracker_.getLastObservedArmor();
            if (observed.tvec.rows == 3 && observed.solve_result) {
                std::array<double, 3> target_camera{{
                    observed.tvec.at<double>(0),
                    observed.tvec.at<double>(1),
                    observed.tvec.at<double>(2),
                }};
                // 弹道下坠：和上面"静止档"、选板器的追板/守角档用同一套补偿。少了这一项，
                // 兜底分支的 pitch 会比正常档位低一个 drop（3 m ≈ 1.3°），一进一出就是
                // "pitch 抽一下"。
                const double distance = std::sqrt(
                    target_camera[0] * target_camera[0] + target_camera[1] * target_camera[1] +
                    target_camera[2] * target_camera[2]);
                const double flight_time = distance / std::max(1.0, selector_.projectileSpeed());
                target_camera[1] -= 0.5 * config_.selector.gravity * flight_time * flight_time;
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
        // 记录本拍最终瞄点：没有新观测的帧直接复用（见上面"保持上一拍瞄点"）。
        if (desired.valid) {
            last_desired_ = desired;
            have_last_desired_ = true;
        }
        outcome.fire = input.fire_enabled && shooter_output.fire;
        outcome.fired_now = outcome.fire && !last_fire_;
        last_fire_ = outcome.fire;
        fire_request_.store(outcome.fire, std::memory_order_relaxed);
        last_outcome_ = outcome;
        return outcome;
    }
} // namespace auto_aim
