#include "ultra_vision/selection/target_selector.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace ultra_vision
{
    namespace
    {
        constexpr double kPi = 3.14159265358979323846;

        bool isFinite(double value)
        {
            return std::isfinite(value);
        }

        int wrapPlateId(int plate_id)
        {
            return (plate_id % 4 + 4) % 4;
        }

        double angleForPlate(int plate_id)
        {
            return (1 - plate_id) * kPi / 2.0;
        }

        std::array<double, 3> predictPlatePosition(
            const TargetEstimate& estimate,
            int plate_id,
            double time)
        {
            const double phase = estimate.yaw + estimate.omega * time + angleForPlate(plate_id);
            return {
                estimate.center[0] + estimate.velocity[0] * time +
                    estimate.armor_radius * std::cos(phase),
                estimate.center[1] + estimate.velocity[1] * time,
                estimate.center[2] + estimate.velocity[2] * time +
                    estimate.armor_radius * std::sin(phase)
            };
        }

        std::array<double, 3> predictCenterPosition(
            const TargetEstimate& estimate,
            double time)
        {
            return {
                estimate.center[0] + estimate.velocity[0] * time,
                estimate.center[1] + estimate.velocity[1] * time,
                estimate.center[2] + estimate.velocity[2] * time
            };
        }

        double normalizeAngle(double angle)
        {
            return std::atan2(std::sin(angle), std::cos(angle));
        }

        double axisTravelTime(double error, double max_velocity, double max_acceleration)
        {
            error = std::abs(error);
            if (error <= 1e-9 || max_velocity <= 1e-9 || max_acceleration <= 1e-9) {
                return 0.0;
            }

            const double acceleration_time = max_velocity / max_acceleration;
            const double acceleration_distance =
                0.5 * max_acceleration * acceleration_time * acceleration_time;
            if (error <= 2.0 * acceleration_distance) {
                return 2.0 * std::sqrt(error / max_acceleration);
            }
            return error / max_velocity + acceleration_time;
        }

        double gimbalTravelTime(const TargetSelectorConfig& config,
                                double current_yaw, double current_pitch,
                                double target_yaw, double target_pitch)
        {
            const double yaw_time = axisTravelTime(
                normalizeAngle(target_yaw - current_yaw),
                config.yaw_velocity, config.yaw_acceleration);
            const double pitch_time = axisTravelTime(
                target_pitch - current_pitch,
                config.pitch_velocity, config.pitch_acceleration);
            return std::max(yaw_time, pitch_time);
        }
    }

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
        const TargetEstimate& estimate,
        double current_yaw,
        double current_pitch)
    {
        TargetDecision result;
        if (!config_.enabled) return result;
        if (!isFinite(estimate.center[0]) || !isFinite(estimate.center[1]) ||
            !isFinite(estimate.center[2]) || !isFinite(estimate.velocity[0]) ||
            !isFinite(estimate.velocity[1]) || !isFinite(estimate.velocity[2]) ||
            !isFinite(estimate.yaw) || !isFinite(estimate.omega) ||
            !isFinite(estimate.armor_radius) || estimate.armor_radius <= 1e-6 ||
            !isFinite(current_yaw) || !isFinite(current_pitch)) {
            return result;
        }

        const double center_distance = std::sqrt(
            estimate.center[0] * estimate.center[0] +
            estimate.center[1] * estimate.center[1] +
            estimate.center[2] * estimate.center[2]);
        if (!isFinite(center_distance) || center_distance <= 1e-6) {
            return result;
        }

        const double seed_time = config_.command_latency +
            center_distance / config_.projectile_speed;

        struct Evaluated {
            TargetDecision decision;
            double delta = 0.0;
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

                const double next_time = config_.command_latency +
                    gimbal_time + flight_time;
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
            candidate.in_fire_window = abs_delta <= config_.coming_angle &&
                incoming && time_to_handoff > aim_time;
            evaluated[armor_id].decision = candidate;
            evaluated[armor_id].delta = delta;
            evaluated[armor_id].eligible = abs_delta <= config_.coming_angle &&
                incoming;

            if (evaluated[armor_id].eligible) {
                const double cost = abs_delta + 0.25 * gimbal_time +
                    0.05 * aim_time;
                if (cost < best_cost) {
                    best_cost = cost;
                    best_armor = armor_id;
                }
            }
        }

        if (best_armor < 0) return result;

        result = evaluated[best_armor].decision;

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
