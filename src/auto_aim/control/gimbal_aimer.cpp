#include "gimbal_aimer.hpp"

#include <algorithm>
#include <cmath>

namespace auto_aim
{
    namespace
    {
        constexpr double kPi = 3.14159265358979323846;

        double clamp(double value, double low, double high)
        {
            return std::max(low, std::min(value, high));
        }

        double moveToward(double value, double target, double max_delta)
        {
            return value + clamp(target - value, -max_delta, max_delta);
        }

        double stoppingVelocity(double error, double max_acceleration)
        {
            return std::sqrt(std::max(0.0, 2.0 * max_acceleration * std::abs(error)));
        }
    }

    GimbalAimer::GimbalAimer(const GimbalAimConfig& config)
        : config_(config)
    {
        if (!std::isfinite(config_.pitch_min) || !std::isfinite(config_.pitch_max) ||
            config_.pitch_min > config_.pitch_max) {
            config_.pitch_min = -70.0 * kPi / 180.0;
            config_.pitch_max = 70.0 * kPi / 180.0;
        }
        if (!std::isfinite(config_.max_yaw_velocity) || config_.max_yaw_velocity <= 0.0) {
            config_.max_yaw_velocity = 2.2;
        }
        if (!std::isfinite(config_.max_pitch_velocity) || config_.max_pitch_velocity <= 0.0) {
            config_.max_pitch_velocity = 1.5;
        }
        if (!std::isfinite(config_.max_yaw_acceleration) ||
            config_.max_yaw_acceleration <= 0.0) {
            config_.max_yaw_acceleration = 8.0;
        }
        if (!std::isfinite(config_.max_pitch_acceleration) ||
            config_.max_pitch_acceleration <= 0.0) {
            config_.max_pitch_acceleration = 6.0;
        }
        if (!std::isfinite(config_.max_yaw_jerk) || config_.max_yaw_jerk <= 0.0) {
            config_.max_yaw_jerk = 80.0;
        }
        if (!std::isfinite(config_.max_pitch_jerk) || config_.max_pitch_jerk <= 0.0) {
            config_.max_pitch_jerk = 60.0;
        }
        if (!std::isfinite(config_.yaw_response_gain) || config_.yaw_response_gain <= 0.0) {
            config_.yaw_response_gain = 10.0;
        }
        if (!std::isfinite(config_.pitch_response_gain) || config_.pitch_response_gain <= 0.0) {
            config_.pitch_response_gain = 10.0;
        }
        config_.feedforward_gain = clamp(config_.feedforward_gain, 0.0, 1.0);
        if (!std::isfinite(config_.feedforward_time_constant) ||
            config_.feedforward_time_constant <= 0.0) {
            config_.feedforward_time_constant = 0.06;
        }
        if (!std::isfinite(config_.yaw_deadband) || config_.yaw_deadband < 0.0) {
            config_.yaw_deadband = 0.0005;
        }
        if (!std::isfinite(config_.pitch_deadband) || config_.pitch_deadband < 0.0) {
            config_.pitch_deadband = 0.0005;
        }
        if (!std::isfinite(config_.settle_angle) || config_.settle_angle < 0.0) {
            config_.settle_angle = 0.6 * kPi / 180.0;
        }
        if (!std::isfinite(config_.settle_velocity) || config_.settle_velocity < 0.0) {
            config_.settle_velocity = 5.0 * kPi / 180.0;
        }
        if (!std::isfinite(config_.default_dt) || config_.default_dt <= 0.0) {
            config_.default_dt = 1.0 / 30.0;
        }
        if (!std::isfinite(config_.max_dt) || config_.max_dt <= 0.0) {
            config_.max_dt = 0.10;
        }
        reset();
    }

    GimbalTargetAngles GimbalAimer::solveTargetAngles(
        const std::array<double, 3>& target_camera,
        double current_yaw,
        double current_pitch) const
    {
        GimbalTargetAngles target;
        if (!std::isfinite(target_camera[0]) ||
            !std::isfinite(target_camera[1]) ||
            !std::isfinite(target_camera[2]) ||
            !std::isfinite(current_yaw) ||
            !std::isfinite(current_pitch)) {
            return target;
        }

        const double x = target_camera[0];
        const double y = target_camera[1];
        const double z = target_camera[2];
        const double distance = std::sqrt(x * x + y * y + z * z);

        // A target behind the image plane has an ambiguous feedback direction
        // and must not cause a half-turn command jump.
        if (z <= 0.0 || distance <= 1e-9) return target;

        // OpenCV optical coordinates differ from the gimbal's local axes:
        // optical +x = right, +y = down, +z = forward. The simulator applies
        // positive yaw to the right and positive pitch upward.
        const double local_x = x / distance;
        const double local_y = -y / distance;
        const double local_z = -z / distance;
        const double cos_yaw = std::cos(current_yaw);
        const double sin_yaw = std::sin(current_yaw);
        const double cos_pitch = std::cos(current_pitch);
        const double sin_pitch = std::sin(current_pitch);

        // Map the target from the current camera frame into the gimbal command
        // frame with the current Y-X rotation, then recover the absolute
        // command. This avoids coupling errors when both axes are non-zero.
        const double command_frame_x = cos_yaw * local_x -
            sin_yaw * (sin_pitch * local_y + cos_pitch * local_z);
        const double command_frame_y = cos_pitch * local_y - sin_pitch * local_z;
        const double command_frame_z = sin_yaw * local_x +
            cos_yaw * (sin_pitch * local_y + cos_pitch * local_z);

        target.yaw = normalizeAngle(std::atan2(command_frame_x, -command_frame_z));
        target.pitch = clamp(
            std::asin(clamp(command_frame_y, -1.0, 1.0)),
            config_.pitch_min, config_.pitch_max);
        target.valid = true;
        return target;
    }

    GimbalCommand GimbalAimer::solve(const std::array<double, 3>& target_camera,
                                     double current_yaw,
                                     double current_pitch,
                                     double dt)
    {
        return trackTargetAngles(
            solveTargetAngles(target_camera, current_yaw, current_pitch),
            current_yaw, current_pitch, dt);
    }

    GimbalCommand GimbalAimer::trackTargetAngles(const GimbalTargetAngles& desired,
                                                 double current_yaw,
                                                 double current_pitch,
                                                 double dt)
    {
        GimbalCommand command;
        if (!desired.valid || !std::isfinite(current_yaw) || !std::isfinite(current_pitch)) {
            return command;
        }

        if (!std::isfinite(dt) || dt <= 0.0) dt = config_.default_dt;
        dt = std::min(dt, config_.max_dt);

        if (!initialized_) {
            reset(current_yaw, current_pitch);
            previous_desired_yaw_ = desired.yaw;
            previous_desired_pitch_ = desired.pitch;
        } else {
            // The caller, not the controller state, is authoritative for the
            // current physical pose. This prevents a stale acknowledgement
            // from accumulating into a large command jump.
            yaw_ = normalizeAngle(current_yaw);
            pitch_ = clamp(current_pitch, config_.pitch_min, config_.pitch_max);
        }

        const double desired_yaw_rate = desired.velocity_valid
            ? desired.yaw_velocity
            : normalizeAngle(desired.yaw - previous_desired_yaw_) / dt;
        const double desired_pitch_rate = desired.velocity_valid
            ? desired.pitch_velocity
            : (desired.pitch - previous_desired_pitch_) / dt;
        const double feedforward_alpha = clamp(
            dt / config_.feedforward_time_constant, 0.0, 1.0);
        const double plausible_yaw_rate = config_.max_yaw_velocity * 2.5;
        const double plausible_pitch_rate = config_.max_pitch_velocity * 2.5;

        // A plate switch can move the desired angle by almost a quarter turn.
        // Do not interpret that discontinuity as target velocity.
        if (std::abs(desired_yaw_rate) <= plausible_yaw_rate) {
            yaw_feedforward_ += feedforward_alpha * (desired_yaw_rate - yaw_feedforward_);
        } else {
            yaw_feedforward_ = 0.0;
        }
        if (std::abs(desired_pitch_rate) <= plausible_pitch_rate) {
            pitch_feedforward_ +=
                feedforward_alpha * (desired_pitch_rate - pitch_feedforward_);
        } else {
            pitch_feedforward_ = 0.0;
        }

        const double yaw_feedforward_output = desired.velocity_valid
            ? yaw_feedforward_
            : config_.feedforward_gain * yaw_feedforward_;
        const double pitch_feedforward_output = desired.velocity_valid
            ? pitch_feedforward_
            : config_.feedforward_gain * pitch_feedforward_;
        const AxisStep yaw_step = stepAxis(
            yaw_, yaw_velocity_, yaw_acceleration_, desired.yaw,
            yaw_feedforward_output, config_.yaw_deadband,
            config_.max_yaw_velocity, config_.max_yaw_acceleration,
            config_.max_yaw_jerk, config_.yaw_response_gain, true, dt);
        const AxisStep pitch_step = stepAxis(
            pitch_, pitch_velocity_, pitch_acceleration_, desired.pitch,
            pitch_feedforward_output, config_.pitch_deadband,
            config_.max_pitch_velocity, config_.max_pitch_acceleration,
            config_.max_pitch_jerk, config_.pitch_response_gain, false, dt);

        yaw_ = yaw_step.position;
        pitch_ = clamp(pitch_step.position, config_.pitch_min, config_.pitch_max);
        yaw_velocity_ = yaw_step.velocity;
        pitch_velocity_ = pitch_step.velocity;
        yaw_acceleration_ = yaw_step.acceleration;
        pitch_acceleration_ = pitch_step.acceleration;
        previous_desired_yaw_ = desired.yaw;
        previous_desired_pitch_ = desired.pitch;

        command.yaw = yaw_;
        command.pitch = pitch_;
        command.yaw_velocity = yaw_velocity_;
        command.pitch_velocity = pitch_velocity_;
        command.yaw_acceleration = yaw_acceleration_;
        command.pitch_acceleration = pitch_acceleration_;
        command.desired_yaw = desired.yaw;
        command.desired_pitch = desired.pitch;
        command.yaw_error = normalizeAngle(desired.yaw - yaw_);
        command.pitch_error = desired.pitch - pitch_;
        command.settled =
            std::abs(command.yaw_error) <= config_.settle_angle &&
            std::abs(command.pitch_error) <= config_.settle_angle &&
            std::abs(yaw_velocity_) <= config_.settle_velocity &&
            std::abs(pitch_velocity_) <= config_.settle_velocity;
        command.valid = true;
        return command;
    }

    GimbalAimer::AxisStep GimbalAimer::stepAxis(double position,
                                                double velocity,
                                                double acceleration,
                                                double desired,
                                                double feedforward_velocity,
                                                double deadband,
                                                double max_velocity,
                                                double max_acceleration,
                                                double max_jerk,
                                                double response_gain,
                                                bool wrap_angle,
                                                double dt) const
    {
        AxisStep result{position, velocity, acceleration};
        const double error = wrap_angle
            ? normalizeAngle(desired - position)
            : desired - position;

        if (std::abs(error) <= deadband &&
            std::abs(velocity - feedforward_velocity) <= config_.settle_velocity) {
            result.position = desired;
            result.velocity = feedforward_velocity;
            result.acceleration = 0.0;
            return result;
        }

        feedforward_velocity = clamp(
            feedforward_velocity, -max_velocity, max_velocity);
        const double stopping = stoppingVelocity(error, max_acceleration);
        const double correction = clamp(response_gain * error, -stopping, stopping);
        const double target_velocity = clamp(
            feedforward_velocity + correction, -max_velocity, max_velocity);

        // Limit the change in acceleration rather than only the acceleration
        // itself. This removes the step-like onset that made plate switches
        // look like sudden gimbal hits in the simulator.
        const double target_acceleration = clamp(
            (target_velocity - velocity) / dt,
            -max_acceleration, max_acceleration);
        result.acceleration = moveToward(
            acceleration, target_acceleration, max_jerk * dt);
        result.acceleration = clamp(
            result.acceleration, -max_acceleration, max_acceleration);

        result.velocity = velocity + result.acceleration * dt;
        result.velocity = clamp(result.velocity, -max_velocity, max_velocity);
        const double relative_velocity = result.velocity - feedforward_velocity;
        result.velocity = feedforward_velocity + clamp(
            relative_velocity, -stopping, stopping);
        result.position = position + result.velocity * dt;

        const double remaining = wrap_angle
            ? normalizeAngle(desired - result.position)
            : desired - result.position;
        if (error * remaining < 0.0 && std::abs(remaining) < std::abs(error)) {
            result.position = desired;
            result.velocity = feedforward_velocity;
            result.acceleration = 0.0;
        } else if (std::abs(remaining) <= deadband &&
                   std::abs(result.velocity - feedforward_velocity) <=
                       config_.settle_velocity) {
            result.position = desired;
            result.velocity = feedforward_velocity;
            result.acceleration = 0.0;
        }
        return result;
    }

    void GimbalAimer::reset(double yaw, double pitch)
    {
        initialized_ = true;
        yaw_ = normalizeAngle(yaw);
        pitch_ = clamp(pitch, config_.pitch_min, config_.pitch_max);
        yaw_velocity_ = 0.0;
        pitch_velocity_ = 0.0;
        yaw_acceleration_ = 0.0;
        pitch_acceleration_ = 0.0;
        previous_desired_yaw_ = yaw_;
        previous_desired_pitch_ = pitch_;
        yaw_feedforward_ = 0.0;
        pitch_feedforward_ = 0.0;
    }

    double GimbalAimer::normalizeAngle(double angle)
    {
        return std::atan2(std::sin(angle), std::cos(angle));
    }
}
