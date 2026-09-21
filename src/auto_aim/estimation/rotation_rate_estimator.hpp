#ifndef AUTO_AIM_ROTATION_RATE_ESTIMATOR_HPP
#define AUTO_AIM_ROTATION_RATE_ESTIMATOR_HPP

#include <deque>

namespace auto_aim
{
    struct RotationRateEstimatorConfig {
        double window = 0.35;
        double min_span = 0.12;
        double max_sample_gap = 0.25;
        double max_rate = 10.0;
        double min_rate = 0.6;
        double max_residual = 0.16;
        int min_slopes = 4;
    };

    class RotationRateEstimator
    {
    public:
        explicit RotationRateEstimator(
            const RotationRateEstimatorConfig& config = {});

        void reset();
        bool update(double radial_angle, double timestamp);

        double omega() const { return valid_ ? omega_ : 0.0; }
        bool valid() const { return valid_; }
        double residual() const { return residual_; }

    private:
        struct Sample {
            double timestamp = 0.0;
            double angle = 0.0;
        };

        RotationRateEstimatorConfig config_;
        std::deque<Sample> samples_;
        // Smoothed sample interval, used to size the fitting window. A fixed
        // window cannot hold enough samples once the detector runs at a few Hz.
        double sample_interval_ = 0.0;
        double omega_ = 0.0;
        double residual_ = 0.0;
        bool valid_ = false;
    };
}

#endif
