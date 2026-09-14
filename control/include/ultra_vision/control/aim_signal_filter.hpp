#ifndef AUTO_AIM_AIM_SIGNAL_FILTER_HPP
#define AUTO_AIM_AIM_SIGNAL_FILTER_HPP

namespace ultra_vision
{
    struct AimSignalFilterConfig {
        bool enabled = true;
        double process_noise_acceleration = 80.0; // rad^2/s^4
        double measurement_noise = 0.0025;         // rad^2
        double initial_angle_covariance = 0.05;    // rad^2
        double initial_velocity_covariance = 4.0;  // rad^2/s^2
        double reset_innovation = 0.35;            // rad
        double reset_timeout = 0.20;               // s
        double max_dt = 0.10;                      // s
    };

    struct AimSignalEstimate {
        bool valid = false;
        bool reset = false;
        double yaw = 0.0;
        double pitch = 0.0;
        double yaw_velocity = 0.0;
        double pitch_velocity = 0.0;
    };

    // Recursive constant-velocity regression for raw absolute aim angles.
    // It gates target switches and angular jumps instead of trusting a single
    // selector output unconditionally.
    class AimSignalFilter
    {
    public:
        explicit AimSignalFilter(const AimSignalFilterConfig& config = {});

        AimSignalEstimate update(double raw_yaw, double raw_pitch,
                                 int target_id, double timestamp);
        void reset();

    private:
        struct AxisState {
            bool initialized = false;
            double angle = 0.0;
            double velocity = 0.0;
            double p00 = 0.0;
            double p01 = 0.0;
            double p11 = 0.0;
        };

        double updateAxis(AxisState& axis, double measurement, double dt,
                          bool wrap_angle);

        AimSignalFilterConfig config_;
        AxisState yaw_;
        AxisState pitch_;
        bool time_initialized_ = false;
        double last_timestamp_ = 0.0;
        int last_target_id_ = -1;
    };
}

#endif
