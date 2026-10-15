#include "target_selector.hpp"
#include "control/selector_utils.hpp"

using namespace auto_aim::selector_detail;

#include <algorithm>
#include <cmath>
#include <limits>

namespace auto_aim
{

    TargetSelector::TargetSelector(const TargetSelectorConfig& config)
        : config_(config)
    {
        if (!isFinite(config_.projectile_speed) || config_.projectile_speed <= 1e-6) {
            config_.projectile_speed = 25.0;
        }
        if (!isFinite(config_.gravity) || config_.gravity < 0.0) {
            config_.gravity = 9.81;
        }
        if (!isFinite(config_.command_latency) || config_.command_latency < 0.0) {
            config_.command_latency = 0.08;
        }
        if (!isFinite(config_.decide_speed) || config_.decide_speed < 0.0) {
            config_.decide_speed = 7.0;
        }
        if (!isFinite(config_.low_speed_latency) ||
            config_.low_speed_latency < 0.0) {
            config_.low_speed_latency = 0.06;
        }
        if (!isFinite(config_.high_speed_latency) ||
            config_.high_speed_latency < config_.low_speed_latency) {
            config_.high_speed_latency = 0.13;
        }
        if (!isFinite(config_.lock_switch_margin) ||
            config_.lock_switch_margin <= 0.0) {
            config_.lock_switch_margin = 0.65;
        }
        if (!isFinite(config_.max_lead_time) || config_.max_lead_time <= 0.0) {
            config_.max_lead_time = 0.50;
        }
        if (!isFinite(config_.spin_omega_threshold) ||
            config_.spin_omega_threshold < 0.0) {
            config_.spin_omega_threshold = 2.0;
        }
        if (!isFinite(config_.coming_angle) || config_.coming_angle <= 0.0) {
            config_.coming_angle = 60.0 * kPi / 180.0;
        }
        if (!isFinite(config_.leaving_angle) || config_.leaving_angle < 0.0) {
            config_.leaving_angle = 20.0 * kPi / 180.0;
        }
        config_.coming_angle = std::min(config_.coming_angle, kPi);
        config_.leaving_angle = std::min(config_.leaving_angle, config_.coming_angle);
        if (!isFinite(config_.yaw_velocity) || config_.yaw_velocity <= 0.0) {
            config_.yaw_velocity = 6.283185307179586;
        }
        if (!isFinite(config_.pitch_velocity) || config_.pitch_velocity <= 0.0) {
            config_.pitch_velocity = 3.141592653589793;
        }
        if (!isFinite(config_.yaw_acceleration) || config_.yaw_acceleration <= 0.0) {
            config_.yaw_acceleration = 31.41592653589793;
        }
        if (!isFinite(config_.pitch_acceleration) || config_.pitch_acceleration <= 0.0) {
            config_.pitch_acceleration = 20.94395102393195;
        }
        if (!isFinite(config_.handoff_start_angle) || config_.handoff_start_angle < 0.0) {
            config_.handoff_start_angle = 10.0 * kPi / 180.0;
        }
        config_.handoff_start_angle = std::min(
            config_.handoff_start_angle, config_.coming_angle);
        if (!isFinite(config_.handoff_duration) || config_.handoff_duration <= 0.0) {
            config_.handoff_duration = 0.20;
        }
    }

    double TargetSelector::armorDelta(
        const TargetEstimate& estimate,
        int armor_id,
        double lead_time) const
    {
        const auto center = predictCenterPosition(estimate, lead_time);
        const double center_angle = std::atan2(center[2], center[0]);
        // ArmorEKF stores armor positions as center + R * radial. The visible
        // firing-side armor is therefore opposite that radial direction. Shift
        // the plate angle by pi before comparing it with the center bearing.
        const double plate_angle = estimate.yaw + estimate.omega * lead_time +
            angleForPlate(wrapPlateId(armor_id)) + kPi;
        return normalizeAngle(plate_angle - center_angle);
    }

    TargetDecision TargetSelector::select(
        const TargetEstimate& estimate_in,
        double current_yaw,
        double current_pitch)
    {
        TargetDecision result;
        if (!config_.enabled) return result;
        if (!isFinite(estimate_in.center[0]) || !isFinite(estimate_in.center[1]) ||
            !isFinite(estimate_in.center[2]) || !isFinite(estimate_in.velocity[0]) ||
            !isFinite(estimate_in.velocity[1]) || !isFinite(estimate_in.velocity[2]) ||
            !isFinite(estimate_in.yaw) || !isFinite(estimate_in.omega) ||
            !isFinite(estimate_in.armor_radius) || estimate_in.armor_radius <= 1e-6 ||
            !isFinite(current_yaw) || !isFinite(current_pitch)) {
            return result;
        }

        const double center_distance = std::sqrt(
            estimate_in.center[0] * estimate_in.center[0] +
            estimate_in.center[1] * estimate_in.center[1] +
            estimate_in.center[2] * estimate_in.center[2]);
        if (!isFinite(center_distance) || center_distance <= 1e-6) {
            return result;
        }

        // ---- 静止靶闸门：静止时把速度/角速度按 0 处理 ----
        // 见 TargetSelectorConfig::motion_gate_* 的实测数据：阈值按"噪声够不到、
        // 真实运动够得着"取，进/出各一个阈值（滞回）避免在边界抖动。
        TargetEstimate estimate_used = estimate_in;
        if (config_.motion_gate_enabled) {
            const double speed_h =
                std::hypot(estimate_in.velocity[0], estimate_in.velocity[2]);
            const double omega_abs = std::abs(estimate_in.omega);
            if (!estimate_in.converged) {
                // 还没收敛（刚入靶/刚重捕）：一律按静止处理，不给 ω 提前量。
                still_ = true;
            } else if (speed_h > config_.motion_gate_enter_speed ||
                       omega_abs > config_.motion_gate_enter_omega) {
                still_ = false;
            } else if (speed_h < config_.motion_gate_exit_speed &&
                       omega_abs < config_.motion_gate_exit_omega) {
                still_ = true;
            }   // 其余情况保持上一次判定（滞回带）
            if (still_) {
                estimate_used.velocity = {0.0, 0.0, 0.0};
                estimate_used.omega = 0.0;
            }
        } else {
            still_ = false;
        }
        const TargetEstimate& estimate = estimate_used;

        // The estimate is expressed at the time of the last detection, not now.
        // Carry its age into the lead so a slow asynchronous detector is
        // compensated instead of biasing the aim point behind the target.
        const double estimate_age = std::max(
            0.0, std::min(estimate.age, config_.max_lead_time));
        // A fast target needs a larger allowance than a slow one; see the config
        // comment. The split is on the chassis yaw rate, like the reference.
        const double latency = std::abs(estimate.omega) > config_.decide_speed
            ? config_.high_speed_latency
            : config_.low_speed_latency;
        const double seed_time = latency + estimate_age +
            center_distance / config_.projectile_speed;

        // ---- 守角档：|ω| 超过云台可达上限，追板必然失败 -----------------------
        // 瞄"预测的底盘中心方位"：正对相机的近侧板与中心方位只差 asin(R/d)≈2~3°，
        // 云台停住不动就能让每块板周期性扫过视轴，不再被板换甩飞。开火窗 = 预测
        // 弹丸到达时刻，最近的板与守角的夹角在 spin_fire_tolerance 内。
        if (config_.spin_hold_enabled &&
            std::abs(estimate.omega) > config_.chase_omega_threshold) {
            const std::array<double, 3> center_pred{{
                estimate.center[0] + estimate.velocity[0] * seed_time,
                estimate.center[1] + estimate.velocity[1] * seed_time,
                estimate.center[2] + estimate.velocity[2] * seed_time}};
            const double horizontal = std::hypot(center_pred[0], center_pred[2]);
            if (horizontal > 1e-6 && center_pred[2] > 0.0) {
                // 弹道下坠：这里必须和追板档用**同一套**补偿，否则两档在 |ω| 门限
                // 附近来回切时，命令 pitch 会一次跳一个 drop（3 m、25 m/s ≈ 1.3°，
                // 5 m ≈ 1.9°）—— 实测就是"pitch 偶尔抽一下"（2026-10-15：1 rad/s 跑
                // 里 ω 在 2.5 门限上下跳，pitch 在 0.53° 与 1.86° 之间跳 1.33°，
                // 差的正好是 drop 1.29°）。
                const double flight_time = center_distance / config_.projectile_speed;
                const double drop = 0.5 * config_.gravity * flight_time * flight_time;
                // 相机系 y 向下：下坠要让瞄点抬高 → y 减小。
                const std::array<double, 3> aim_point{{
                    center_pred[0], center_pred[1] - drop, center_pred[2]}};
                result.valid = true;
                result.armor_id = -1;
                result.aim_point = aim_point;
                result.target_yaw = std::atan2(center_pred[0], center_pred[2]);
                result.target_pitch = std::atan2(-aim_point[1], horizontal);
                result.predicted_yaw = result.target_yaw;
                result.distance = center_distance;
                result.flight_time = flight_time;
                result.lead_time = seed_time;
                result.target_yaw_velocity = 0.0;
                result.target_pitch_velocity = 0.0;

                // 预测弹丸到达时刻，哪块板离守角最近（也就正对相机的那块）。
                int nearest_plate = -1;
                double nearest_delta = std::numeric_limits<double>::max();
                for (int armor_id = 0; armor_id < 4; ++armor_id) {
                    const std::array<double, 3> plate =
                        predictPlatePosition(estimate, armor_id, seed_time);
                    if (!isFinite(plate[0]) || !isFinite(plate[2]) || plate[2] <= 0.0) {
                        continue;
                    }
                    const double bearing = std::atan2(plate[0], plate[2]);
                    const double delta = std::abs(
                        normalizeAngle(bearing - result.target_yaw));
                    if (delta < nearest_delta) {
                        nearest_delta = delta;
                        nearest_plate = armor_id;
                        result.armor_position = plate;
                    }
                }
                result.armor_id = nearest_plate;
                result.in_fire_window = nearest_plate >= 0 &&
                    nearest_delta <= config_.spin_fire_tolerance;
                lock_id_ = nearest_plate;
                return result;
            }
        }

        struct Evaluated {
            TargetDecision decision;
            double delta = 0.0;
            double cost = std::numeric_limits<double>::max();
            bool eligible = false;
        };
        std::array<Evaluated, 4> evaluated;

        int best_armor = -1;
        double best_cost = std::numeric_limits<double>::max();

        for (int armor_id = 0; armor_id < 4; ++armor_id) {
            double aim_time = seed_time;
            std::array<double, 3> armor_position{{0.0, 0.0, 0.0}};
            std::array<double, 3> aim_point{{0.0, 0.0, 0.0}};
            double distance = 0.0;
            double flight_time = 0.0;
            double raw_yaw = 0.0;
            double raw_pitch = 0.0;
            double gimbal_time = 0.0;
            bool converged = false;

            for (int iteration = 0; iteration < 10; ++iteration) {
                armor_position = predictPlatePosition(estimate, armor_id, aim_time);
                distance = std::sqrt(
                    armor_position[0] * armor_position[0] +
                    armor_position[1] * armor_position[1] +
                    armor_position[2] * armor_position[2]);
                if (!isFinite(distance) || distance <= 1e-6) break;

                flight_time = distance / config_.projectile_speed;
                aim_point = armor_position;
                aim_point[1] -= 0.5 * config_.gravity * flight_time * flight_time;
                const double horizontal = std::hypot(aim_point[0], aim_point[2]);
                raw_yaw = std::atan2(aim_point[0], aim_point[2]);
                raw_pitch = std::atan2(-aim_point[1], horizontal);
                gimbal_time = gimbalTravelTime(
                    config_, current_yaw, current_pitch, raw_yaw, raw_pitch);

                const double next_time =
                    latency + estimate_age + gimbal_time + flight_time;
                if (std::abs(next_time - aim_time) < 1e-4) {
                    aim_time = next_time;
                    converged = true;
                    break;
                }
                aim_time = next_time;
            }

            if (!converged || !isFinite(aim_time) ||
                aim_time > config_.max_lead_time) {
                continue;
            }

            armor_position = predictPlatePosition(estimate, armor_id, aim_time);
            distance = std::sqrt(
                armor_position[0] * armor_position[0] +
                armor_position[1] * armor_position[1] +
                armor_position[2] * armor_position[2]);
            if (!isFinite(distance) || distance <= 1e-6 || armor_position[2] <= 0.0) {
                continue;
            }

            flight_time = distance / config_.projectile_speed;
            aim_point = armor_position;
            aim_point[1] -= 0.5 * config_.gravity * flight_time * flight_time;
            const double horizontal = std::hypot(aim_point[0], aim_point[2]);
            raw_yaw = std::atan2(aim_point[0], aim_point[2]);
            raw_pitch = std::atan2(-aim_point[1], horizontal);
            gimbal_time = gimbalTravelTime(
                config_, current_yaw, current_pitch, raw_yaw, raw_pitch);

            const double phase = estimate.yaw + estimate.omega * aim_time +
                angleForPlate(armor_id);
            const double sin_phase = std::sin(phase);
            const double cos_phase = std::cos(phase);
            const double px = armor_position[0];
            const double pz = armor_position[2];
            const double dpx = estimate.velocity[0] -
                estimate.armor_radius * sin_phase * estimate.omega;
            const double dpz = estimate.velocity[2] +
                estimate.armor_radius * cos_phase * estimate.omega;
            const double horizontal_sq = std::max(px * px + pz * pz, 1e-9);
            const double yaw_velocity =
                (pz * dpx - px * dpz) / horizontal_sq;

            const double horizontal_distance = std::sqrt(horizontal_sq);
            const double dy = estimate.velocity[1];
            const double dh = (px * dpx + pz * dpz) / horizontal_distance;
            const double pitch_velocity =
                (-horizontal_distance * dy - aim_point[1] * dh) /
                std::max(horizontal_sq + aim_point[1] * aim_point[1], 1e-9);

            const double delta = armorDelta(estimate, armor_id, aim_time);
            const double abs_delta = std::abs(delta);
            const bool incoming = estimate.omega > config_.spin_omega_threshold
                ? delta < config_.leaving_angle
                : (estimate.omega < -config_.spin_omega_threshold
                    ? delta > -config_.leaving_angle
                    : true);

            constexpr double probe_dt = 0.02;
            const double probe_delta = armorDelta(
                estimate, armor_id, aim_time + probe_dt);
            const double delta_rate = normalizeAngle(probe_delta - delta) / probe_dt;
            double time_to_handoff = std::numeric_limits<double>::infinity();
            if (std::abs(delta_rate) > 1e-4) {
                const double handoff_delta = delta_rate > 0.0
                    ? config_.handoff_start_angle
                    : -config_.handoff_start_angle;
                time_to_handoff = std::max(0.0, (handoff_delta - delta) / delta_rate);
            }

            TargetDecision candidate;
            candidate.valid = true;
            candidate.armor_id = armor_id;
            candidate.armor_position = armor_position;
            candidate.aim_point = aim_point;
            candidate.predicted_yaw = normalizeAngle(
                estimate.yaw + estimate.omega * aim_time);
            candidate.target_yaw = raw_yaw;
            candidate.target_pitch = raw_pitch;
            candidate.target_yaw_velocity = yaw_velocity;
            candidate.target_pitch_velocity = pitch_velocity;
            candidate.distance = distance;
            candidate.flight_time = flight_time;
            candidate.aim_time = aim_time;
            candidate.gimbal_time = gimbal_time;
            candidate.lead_time = aim_time;
            candidate.time_to_handoff = time_to_handoff;
            // Spinning chassis: a plate crosses the window in a fraction of the
            // bullet's flight time, so requiring it to stay put for the whole
            // lead time (`time_to_handoff > aim_time`) is unsatisfiable and
            // silently disables firing. The reference implementation only asks
            // for "inside the window and on the side where plates appear".
            const bool spinning = std::abs(estimate.omega) >
                config_.spin_omega_threshold;
            candidate.in_fire_window = abs_delta <= config_.coming_angle &&
                incoming && (spinning || time_to_handoff > aim_time);
            evaluated[armor_id].decision = candidate;
            evaluated[armor_id].delta = delta;
            evaluated[armor_id].eligible = abs_delta <= config_.coming_angle &&
                incoming;

            if (evaluated[armor_id].eligible) {
                const double cost = abs_delta + 0.25 * gimbal_time +
                    0.05 * aim_time;
                evaluated[armor_id].cost = cost;
                if (cost < best_cost) {
                    best_cost = cost;
                    best_armor = armor_id;
                }
            }
        }

        // Plate locking: when the previously chosen plate is still eligible,
        // keep it unless another candidate is clearly cheaper. Without this the
        // selector can alternate between two plates that sit at similar angles,
        // which costs a gimbal round trip every time.
        if (lock_id_ >= 0 && lock_id_ < 4 &&
            evaluated[static_cast<std::size_t>(lock_id_)].eligible) {
            const double locked_cost =
                evaluated[static_cast<std::size_t>(lock_id_)].cost;
            if (best_armor < 0 ||
                best_cost >= locked_cost * config_.lock_switch_margin) {
                best_armor = lock_id_;
                best_cost = locked_cost;
            }
        }
        lock_id_ = best_armor;

        if (best_armor < 0) return result;

        result = evaluated[best_armor].decision;

        // No handoff blending while the chassis is spinning. The blend puts the
        // aim point *between* two plates, so neither is ever aligned with the
        // gimbal and the firing gate stays closed. During spin the plates sweep
        // continuously anyway, so aiming straight at the chosen one is both
        // simpler and correct.
        const bool spin_mode = std::abs(estimate.omega) >
            config_.spin_omega_threshold;
        if (spin_mode) return result;

        int next_armor = -1;
        double next_cost = std::numeric_limits<double>::max();
        const double handoff_lookup_time = result.aim_time +
            config_.handoff_duration;
        for (int armor_id = 0; armor_id < 4; ++armor_id) {
            if (armor_id == best_armor || !evaluated[armor_id].decision.valid) {
                continue;
            }
            const double future_delta = armorDelta(
                estimate, armor_id, handoff_lookup_time);
            const double cost = std::abs(future_delta);
            if (cost < next_cost) {
                next_cost = cost;
                next_armor = armor_id;
            }
        }

        if (next_armor >= 0 && result.time_to_handoff < config_.handoff_duration) {
            const double progress = std::max(
                0.0, std::min(1.0,
                    1.0 - result.time_to_handoff / config_.handoff_duration));
            const double alpha = progress * progress * (3.0 - 2.0 * progress);
            const TargetDecision& next = evaluated[next_armor].decision;
            result.next_armor_id = next_armor;
            result.handoff_alpha = alpha;
            result.handoff_gain = 1.0 - alpha;
            result.target_yaw = normalizeAngle(result.target_yaw + alpha *
                normalizeAngle(next.target_yaw - result.target_yaw));
            result.target_pitch += alpha * (next.target_pitch - result.target_pitch);
            result.target_yaw_velocity += alpha *
                (next.target_yaw_velocity - result.target_yaw_velocity);
            result.target_pitch_velocity += alpha *
                (next.target_pitch_velocity - result.target_pitch_velocity);
        }
        return result;
    }

}
