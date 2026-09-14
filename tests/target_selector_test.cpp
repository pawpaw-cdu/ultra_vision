#include "ultra_vision/selection/target_selector.hpp"

#include <array>
#include <cmath>
#include <iostream>

namespace
{
    bool near(double lhs, double rhs, double tolerance = 1e-6)
    {
        return std::abs(lhs - rhs) <= tolerance;
    }

    bool expect(bool condition, const char* message)
    {
        if (!condition) std::cerr << "FAILED: " << message << std::endl;
        return condition;
    }

    ultra_vision::TargetEstimate makeEstimate()
    {
        ultra_vision::TargetEstimate estimate;
        estimate.center = {0.0, 0.0, 5.0};
        estimate.velocity = {0.0, 0.0, 0.0};
        estimate.yaw = 0.0;
        estimate.omega = 0.0;
        estimate.armor_radius = 0.21;
        return estimate;
    }
}

int main()
{
    bool passed = true;
    ultra_vision::TargetSelectorConfig config;
    config.command_latency = 0.08;
    config.max_lead_time = 1.0;
    ultra_vision::TargetSelector selector(config);

    const auto stationary = selector.select(makeEstimate(), 0.0, 0.0);
    const double stationary_distance = 4.79;
    const double stationary_flight = stationary_distance / config.projectile_speed;
    const double expected_drop = 0.5 * config.gravity *
        stationary_flight * stationary_flight;
    passed &= expect(stationary.valid, "stationary prediction must be valid");
    passed &= expect(stationary.armor_id == 2, "stationary target must select the near armor");
    passed &= expect(near(stationary.armor_position[0], 0.0, 1e-6),
                     "front plate must stay centered in x");
    passed &= expect(near(stationary.aim_point[1], -expected_drop, 1e-6),
                     "ballistic compensation must move the aim point upward");
    passed &= expect(near(stationary.target_yaw, 0.0, 1e-6) &&
                         stationary.target_pitch > 0.0,
                     "selector must output an absolute home-frame gimbal angle");
    passed &= expect(near(stationary.flight_time, stationary_flight, 1e-6),
                     "flight time must use the predicted distance");

    auto positive_omega = makeEstimate();
    positive_omega.omega = 3.0;
    const auto positive = selector.select(positive_omega, 0.0, 0.0);
    passed &= expect(positive.valid, "positive omega prediction must be valid");
    passed &= expect(positive.armor_id == 3,
                     "positive omega must select the right incoming armor");

    auto negative_omega = makeEstimate();
    negative_omega.omega = -3.0;
    const auto negative = selector.select(negative_omega, 0.0, 0.0);
    passed &= expect(negative.valid, "negative omega prediction must be valid");
    passed &= expect(negative.armor_id == 1,
                     "negative omega must select the left incoming armor");

    auto low_omega = makeEstimate();
    low_omega.omega = 0.05;
    ultra_vision::TargetSelector low_speed_selector(config);
    const auto low = low_speed_selector.select(low_omega, 0.0, 0.0);
    passed &= expect(low.valid && low.armor_id == 2,
                     "small angular travel must not switch armor");

    auto moving = makeEstimate();
    moving.velocity = {1.0, 0.0, -0.5};
    const auto moving_decision = selector.select(moving, 0.0, 0.0);
    passed &= expect(moving_decision.valid, "moving target decision must be valid");
    passed &= expect(moving_decision.aim_point[0] > 0.15,
                     "linear x velocity must be included in lead prediction");
    passed &= expect(moving_decision.aim_point[2] < 5.21,
                     "linear z velocity must be included in lead prediction");
    passed &= expect(moving_decision.target_yaw > 0.0,
                     "absolute yaw must follow the predicted home-frame position");

    auto far_estimate = makeEstimate();
    far_estimate.center = {0.0, 0.0, 30.0};
    const auto far = selector.select(far_estimate, 0.0, 0.0);
    passed &= expect(!far.valid, "prediction beyond max lead time must be rejected");

    if (!passed) return 1;
    std::cout << "target_selector_test passed" << std::endl;
    return 0;
}
