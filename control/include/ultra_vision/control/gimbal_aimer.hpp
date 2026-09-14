#ifndef AUTO_AIM_GIMBAL_AIMER_HPP
#define AUTO_AIM_GIMBAL_AIMER_HPP

#include <array>

namespace ultra_vision
{
    struct GimbalAimConfig {
        // Physical limits are expressed in SI units. They are intentionally
        // frame-rate independent, unlike the previous "degrees per frame"
        // limits which changed behavior when the processing rate changed.
        double pitch_min = -70.0 * 3.14159265358979323846 / 180.0;
        double pitch_max = 70.0 * 3.14159265358979323846 / 180.0;
        double max_yaw_velocity = 2.2;       // rad/s
        double max_pitch_velocity = 1.5;     // rad/s
        double max_yaw_acceleration = 8.0;   // rad/s^2
        double max_pitch_acceleration = 6.0; // rad/s^2
        double max_yaw_jerk = 80.0;          // rad/s^3
        double max_pitch_jerk = 60.0;        // rad/s^3
        double yaw_response_gain = 10.0;     // 1/s
        double pitch_response_gain = 10.0;   // 1/s
        double feedforward_gain = 0.85;
        double feedforward_time_constant = 0.06; // s
        double yaw_deadband = 0.0005;        // rad
        double pitch_deadband = 0.0005;      // rad
        double settle_angle = 0.6 * 3.14159265358979323846 / 180.0;
        double settle_velocity = 5.0 * 3.14159265358979323846 / 180.0;
        double default_dt = 1.0 / 30.0;      // s
        double max_dt = 0.10;                // s
    };

    struct GimbalCommand {
        bool valid = false;
        double yaw = 0.0;
        double pitch = 0.0;
        double yaw_velocity = 0.0;
        double pitch_velocity = 0.0;
        double yaw_acceleration = 0.0;
        double pitch_acceleration = 0.0;
        double desired_yaw = 0.0;
        double desired_pitch = 0.0;
        double yaw_error = 0.0;
        double pitch_error = 0.0;
        bool settled = false;
    };

    struct GimbalTargetAngles {
        bool valid = false;
        bool velocity_valid = false;
        double yaw = 0.0;
        double pitch = 0.0;
        double yaw_velocity = 0.0;
        double pitch_velocity = 0.0;
    };

    // Converts a target position expressed in the current camera frame into
    // absolute gimbal angles, then advances a jerk-limited trajectory toward
    // that target. The class has no transport or simulator dependency.
    class GimbalAimer
    {
    public:
        explicit GimbalAimer(const GimbalAimConfig& config = {});

        // Stateless angle solver. This is the only part that depends on the
        // target geometry and current camera pose.
        GimbalTargetAngles solveTargetAngles(
            const std::array<double, 3>& target_camera,
            double current_yaw,
            double current_pitch) const;

        // Stateful trajectory generator. dt is the elapsed time since the
        // previous command and must be supplied by the caller so the same
        // controller behaves identically at different frame rates.
        GimbalCommand solve(const std::array<double, 3>& target_camera,
                            double current_yaw,
                            double current_pitch,
                            double dt);

        // Advances the trajectory toward angles already expressed in the
        // target/command frame. This is used by high-rate command schedulers
        // that are independent from the vision frame rate.
        GimbalCommand trackTargetAngles(const GimbalTargetAngles& desired,
                                        double current_yaw,
                                        double current_pitch,
                                        double dt);

        void reset(double yaw = 0.0, double pitch = 0.0);

        static double normalizeAngle(double angle);

    private:
        struct AxisStep {
            double position = 0.0;
            double velocity = 0.0;
            double acceleration = 0.0;
        };

        AxisStep stepAxis(double position,
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
                          double dt) const;

        GimbalAimConfig config_;
        bool initialized_ = false;
        double yaw_ = 0.0;
        double pitch_ = 0.0;
        double yaw_velocity_ = 0.0;
        double pitch_velocity_ = 0.0;
        double yaw_acceleration_ = 0.0;
        double pitch_acceleration_ = 0.0;
        double previous_desired_yaw_ = 0.0;
        double previous_desired_pitch_ = 0.0;
        double yaw_feedforward_ = 0.0;
        double pitch_feedforward_ = 0.0;
    };
}

#endif
