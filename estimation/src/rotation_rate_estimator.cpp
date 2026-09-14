#include "ultra_vision/estimation/rotation_rate_estimator.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace ultra_vision
{
    namespace
    {
        constexpr double kPi = 3.14159265358979323846;

        double normalizeAngle(double angle)
        {
            return std::atan2(std::sin(angle), std::cos(angle));
        }

        double nearestPlateBranch(double angle, double reference)
        {
            double best = angle;
            double best_error = std::abs(normalizeAngle(angle - reference));
            for (int branch = -2; branch <= 2; ++branch) {
                if (branch == 0) continue;
                const double candidate = angle + branch * kPi;
                const double error = std::abs(
                    normalizeAngle(candidate - reference));
                if (error < best_error) {
                    best = candidate;
                    best_error = error;
                }
            }
            return best;
        }
    }

    RotationRateEstimator::RotationRateEstimator(
        const RotationRateEstimatorConfig& config)
        : config_(config)
    {
        if (!std::isfinite(config_.window) || config_.window <= 0.0) {
            config_.window = 0.35;
        }
        if (!std::isfinite(config_.min_span) || config_.min_span <= 0.0) {
            config_.min_span = 0.12;
        }
        if (!std::isfinite(config_.max_sample_gap) ||
            config_.max_sample_gap <= 0.0) {
            config_.max_sample_gap = 0.25;
        }
        if (!std::isfinite(config_.max_rate) || config_.max_rate <= 0.0) {
            config_.max_rate = 10.0;
        }
        if (!std::isfinite(config_.min_rate) || config_.min_rate < 0.0) {
            config_.min_rate = 0.6;
        }
        if (!std::isfinite(config_.max_residual) ||
            config_.max_residual <= 0.0) {
            config_.max_residual = 0.16;
        }
        if (config_.min_slopes < 1) {
            config_.min_slopes = 4;
        }
    }

    void RotationRateEstimator::reset()
    {
        samples_.clear();
        omega_ = 0.0;
        residual_ = 0.0;
        valid_ = false;
    }

    bool RotationRateEstimator::update(
        double radial_angle, double timestamp)
    {
        if (!std::isfinite(radial_angle) || !std::isfinite(timestamp)) {
            return valid_;
        }

        if (!samples_.empty()) {
            const double gap = timestamp - samples_.back().timestamp;
            if (gap <= 0.0 || gap > config_.max_sample_gap) {
                reset();
            }
        }

        while (!samples_.empty() &&
               timestamp - samples_.front().timestamp > config_.window) {
            samples_.pop_front();
        }

        double branch = radial_angle;
        double unwrapped = branch;
        if (!samples_.empty()) {
            const double reference = samples_.back().angle;
            branch = nearestPlateBranch(radial_angle, reference);
            unwrapped = reference + normalizeAngle(branch - reference);

            const double dt = timestamp - samples_.back().timestamp;
            const double step = std::abs(unwrapped - reference);
            const double max_step = std::min(
                config_.max_rate * dt + 0.35, 2.35);
            if (dt > 1e-6 && step > max_step) {
                return valid_;
            }
        }

        samples_.push_back({timestamp, unwrapped});
        if (samples_.back().timestamp - samples_.front().timestamp <
            config_.min_span) {
            return valid_;
        }

        std::vector<double> slopes;
        slopes.reserve(samples_.size() * (samples_.size() - 1) / 2);
        for (std::size_t i = 0; i < samples_.size(); ++i) {
            for (std::size_t j = i + 1; j < samples_.size(); ++j) {
                const double span =
                    samples_[j].timestamp - samples_[i].timestamp;
                if (span < config_.min_span) continue;
                const double slope =
                    (samples_[j].angle - samples_[i].angle) / span;
                if (std::abs(slope) <= config_.max_rate) {
                    slopes.push_back(slope);
                }
            }
        }

        if (static_cast<int>(slopes.size()) < config_.min_slopes) {
            return valid_;
        }

        const std::size_t middle = slopes.size() / 2;
        std::nth_element(
            slopes.begin(), slopes.begin() + middle, slopes.end());
        double candidate = slopes[middle];
        if (slopes.size() % 2 == 0) {
            const double upper = slopes[middle];
            std::nth_element(
                slopes.begin(), slopes.begin() + middle - 1,
                slopes.begin() + middle);
            candidate = 0.5 * (slopes[middle - 1] + upper);
        }

        std::vector<double> intercepts;
        intercepts.reserve(samples_.size());
        for (const auto& sample : samples_) {
            intercepts.push_back(sample.angle - candidate * sample.timestamp);
        }
        std::nth_element(
            intercepts.begin(),
            intercepts.begin() + intercepts.size() / 2,
            intercepts.end());
        const double intercept = intercepts[intercepts.size() / 2];

        double squared_error = 0.0;
        for (const auto& sample : samples_) {
            const double prediction =
                intercept + candidate * sample.timestamp;
            const double error = normalizeAngle(sample.angle - prediction);
            squared_error += error * error;
        }
        residual_ = std::sqrt(squared_error / samples_.size());
        if (!std::isfinite(residual_) ||
            residual_ > config_.max_residual) {
            valid_ = false;
            omega_ = 0.0;
            return false;
        }

        omega_ = std::abs(candidate) < config_.min_rate ? 0.0 : candidate;
        valid_ = true;
        return true;
    }
}
