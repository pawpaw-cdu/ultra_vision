#ifndef AUTO_AIM_SELECTOR_UTILS_HPP
#define AUTO_AIM_SELECTOR_UTILS_HPP

// 从 control/target_selector.cpp 抽出的无状态工具（角度/弹道/可达性/期望角）。

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

#include "control/target_selector.hpp"

namespace auto_aim
{
namespace selector_detail
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
}
}

#endif // AUTO_AIM_SELECTOR_UTILS_HPP
