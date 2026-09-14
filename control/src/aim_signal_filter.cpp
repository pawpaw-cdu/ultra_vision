#include "ultra_vision/control/aim_signal_filter.hpp"

#include <algorithm>
#include <cmath>

namespace ultra_vision
{
    namespace
    {
        double normalizeAngle(double angle)
        {
            return std::atan2(std::sin(angle), std::cos(angle));
        }

        bool finite(double value)
        {
            return std::isfinite(value);
        }
    }

    AimSignalFilter::AimSignalFilter(const AimSignalFilterConfig& config)
        : config_(config)
    {
        if (!finite(config_.process_noise_acceleration) ||
            config_.process_noise_acceleration <= 0.0) {
            config_.process_noise_acceleration = 80.0;
        }
        if (!finite(config_.measurement_noise) || config_.measurement_noise <= 0.0) {
            config_.measurement_noise = 0.0025;
        }
        if (!finite(config_.initial_angle_covariance) ||
            config_.initial_angle_covariance <= 0.0) {
            config_.initial_angle_covariance = 0.05;
        }
        if (!finite(config_.initial_velocity_covariance) ||
            config_.initial_velocity_covariance <= 0.0) {
            config_.initial_velocity_covariance = 4.0;
        }
        if (!finite(config_.reset_innovation) || config_.reset_innovation <= 0.0) {
            config_.reset_innovation = 0.35;
        }
        if (!finite(config_.reset_timeout) || config_.reset_timeout <= 0.0) {
            config_.reset_timeout = 0.20;
        }
        if (!finite(config_.max_dt) || config_.max_dt <= 0.0) {
            config_.max_dt = 0.10;
        }
    }

    void AimSignalFilter::reset()
    {
        yaw_ = AxisState{};
        pitch_ = AxisState{};
        time_initialized_ = false;
        last_timestamp_ = 0.0;
        last_target_id_ = -1;
    }

    double AimSignalFilter::updateAxis(
        AxisState& axis, double measurement, double dt, bool wrap_angle)
    {
        if (!axis.initialized) {
            axis.initialized = true;
            axis.angle = wrap_angle ? normalizeAngle(measurement) : measurement;
            axis.velocity = 0.0;
            axis.p00 = config_.initial_angle_covariance;
            axis.p01 = 0.0;
            axis.p11 = config_.initial_velocity_covariance;
            return axis.angle;
        }

        const double predicted_angle = axis.angle + axis.velocity * dt;
        const double innovation = wrap_angle
            ? normalizeAngle(measurement - predicted_angle)
            : measurement - predicted_angle;

        if (std::abs(innovation) > config_.reset_innovation) {
            axis.angle = wrap_angle ? normalizeAngle(measurement) : measurement;
            axis.velocity = 0.0;
            axis.p00 = config_.initial_angle_covariance;
            axis.p01 = 0.0;
            axis.p11 = config_.initial_velocity_covariance;
            return axis.angle;
        }

        const double q = config_.process_noise_acceleration;
        const double dt2 = dt * dt;
        const double dt3 = dt2 * dt;
        const double dt4 = dt2 * dt2;
        const double p00 = axis.p00 + 2.0 * dt * axis.p01 + dt2 * axis.p11 +
            0.25 * dt4 * q;
        const double p01 = axis.p01 + dt * axis.p11 + 0.5 * dt3 * q;
        const double p11 = axis.p11 + dt2 * q;
        const double innovation_variance = p00 + config_.measurement_noise;
        if (!finite(innovation_variance) || innovation_variance <= 1e-9) {
            return axis.angle;
        }

        const double gain_angle = p00 / innovation_variance;
        const double gain_velocity = p01 / innovation_variance;
        axis.angle = predicted_angle + gain_angle * innovation;
        axis.velocity += gain_velocity * innovation;
        axis.p00 = (1.0 - gain_angle) * p00;
        axis.p01 = (1.0 - gain_angle) * p01;
        axis.p11 = p11 - gain_velocity * p01;
        return axis.angle;
    }

    AimSignalEstimate AimSignalFilter::update(
        double raw_yaw, double raw_pitch, int target_id, double timestamp)
    {
        AimSignalEstimate estimate;
        if (!config_.enabled) {
            estimate.valid = finite(raw_yaw) && finite(raw_pitch);
            estimate.yaw = normalizeAngle(raw_yaw);
            estimate.pitch = raw_pitch;
            return estimate;
        }
        if (!finite(raw_yaw) || !finite(raw_pitch) || !finite(timestamp)) {
            return estimate;
        }

        const bool target_changed = target_id != last_target_id_;
        const bool timed_out = time_initialized_ &&
            (timestamp - last_timestamp_ > config_.reset_timeout ||
             timestamp < last_timestamp_);
        if (target_changed || timed_out || !time_initialized_) {
            yaw_ = AxisState{};
            pitch_ = AxisState{};
            estimate.reset = true;
        }

        double dt = time_initialized_ ? timestamp - last_timestamp_ : 0.0;
        dt = std::max(0.0, std::min(dt, config_.max_dt));
        estimate.yaw = updateAxis(yaw_, raw_yaw, dt, true);
        estimate.pitch = updateAxis(pitch_, raw_pitch, dt, false);
        estimate.yaw_velocity = yaw_.velocity;
        estimate.pitch_velocity = pitch_.velocity;
        estimate.valid = true;

        time_initialized_ = true;
        last_timestamp_ = timestamp;
        last_target_id_ = target_id;
        return estimate;
    }
}
