#ifndef AUTO_AIM_TARGET_SELECTOR_HPP
#define AUTO_AIM_TARGET_SELECTOR_HPP

#include <array>

namespace ultra_vision
{
    struct TargetSelectorConfig {
        bool enabled = true;
        double projectile_speed = 25.0;
        double gravity = 9.81;
        double command_latency = 0.08;
        double max_lead_time = 0.50;
        double spin_omega_threshold = 2.0;
        double coming_angle = 1.0471975511965976;  // 60 degrees
        double leaving_angle = 0.3490658503988659; // 20 degrees
        double yaw_velocity = 6.283185307179586;   // rad/s
        double pitch_velocity = 3.141592653589793; // rad/s
        double yaw_acceleration = 31.41592653589793;   // rad/s^2
        double pitch_acceleration = 20.94395102393195; // rad/s^2
        double handoff_start_angle = 0.17453292519943295; // 10 degrees
        double handoff_duration = 0.20; // s
    };

    // Pure estimator output consumed by the selector. All values are in the
    // world frame so the selector is independent of camera rotation.
    struct TargetEstimate {
        std::array<double, 3> center{{0.0, 0.0, 0.0}};
        std::array<double, 3> velocity{{0.0, 0.0, 0.0}};
        double yaw = 0.0;
        double omega = 0.0;
        double armor_radius = 0.21;
    };

    // The selector's only output. The gimbal consumes `aim_point`; the
    // visualizer consumes `armor_position` and `predicted_yaw`; the firing
    // decision consumes `in_fire_window` plus gimbal readiness.
    struct TargetDecision {
        bool valid = false;
        int armor_id = -1;
        std::array<double, 3> armor_position{{0.0, 0.0, 0.0}};
        std::array<double, 3> aim_point{{0.0, 0.0, 0.0}};
        double predicted_yaw = 0.0;
        double target_yaw = 0.0;
        double target_pitch = 0.0;
        double target_yaw_velocity = 0.0;
        double target_pitch_velocity = 0.0;
        double distance = 0.0;
        double flight_time = 0.0;
        double aim_time = 0.0;
        double gimbal_time = 0.0;
        double lead_time = 0.0;
        double handoff_gain = 1.0;
        double handoff_alpha = 0.0;
        double time_to_handoff = 0.0;
        int next_armor_id = -1;
        bool in_fire_window = false;
    };

    class TargetSelector
    {
    public:
        explicit TargetSelector(const TargetSelectorConfig& config = {});

        TargetDecision select(const TargetEstimate& estimate,
                              double current_yaw,
                              double current_pitch);

    private:
        double armorDelta(const TargetEstimate& estimate,
                          int armor_id,
                          double lead_time) const;

        TargetSelectorConfig config_;
    };
}

#endif
