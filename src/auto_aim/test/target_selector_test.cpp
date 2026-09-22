#include "control/target_selector.hpp"

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

    auto_aim::TargetEstimate makeEstimate()
    {
        auto_aim::TargetEstimate estimate;
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
    auto_aim::TargetSelectorConfig config;
    config.command_latency = 0.08;
    config.max_lead_time = 1.0;
    auto_aim::TargetSelector selector(config);

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
    auto_aim::TargetSelector low_speed_selector(config);
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

    // ---- 静止靶闸门（2026-09-29）------------------------------------------
    // 实车静止靶上 EKF 的 |v|/|ω| 噪声（|v| p90 200 mm/s、|ω| p90 3.7°/s）会通过
    // 提前量 + 云台角速度前馈变成可见抖动。闸门必须在"噪声量级"上判静止、
    // 在"真实运动量级"上判在动。
    {
        auto noise = makeEstimate();
        noise.velocity = {0.20, 0.0, -0.15};   // |v| = 250 mm/s：噪声量级
        noise.omega = 0.06;                    // 3.4°/s：噪声量级
        auto_aim::TargetSelector gate_selector(config);
        const auto still_decision = gate_selector.select(noise, 0.0, 0.0);
        passed &= expect(gate_selector.targetStill(),
                         "measured EKF noise must be treated as a still target");
        passed &= expect(near(still_decision.target_yaw_velocity, 0.0, 1e-9) &&
                             near(still_decision.target_pitch_velocity, 0.0, 1e-9),
                         "still target must not send a velocity feedforward to the gimbal");
        const auto still_baseline = gate_selector.select(makeEstimate(), 0.0, 0.0);
        passed &= expect(near(still_decision.aim_point[0], still_baseline.aim_point[0], 1e-9) &&
                             near(still_decision.aim_point[2], still_baseline.aim_point[2], 1e-9),
                         "still target must not apply the velocity lead");

        auto spin = makeEstimate();
        spin.omega = 2.0;   // 115°/s：真实小陀螺
        auto_aim::TargetSelector spin_selector(config);
        const auto spin_decision = spin_selector.select(spin, 0.0, 0.0);
        passed &= expect(!spin_selector.targetStill() && spin_decision.valid,
                         "real small-gyro rate must pass the motion gate immediately");

        auto translate = makeEstimate();
        translate.velocity = {0.8, 0.0, -0.8};   // 1.13 m/s：真实平移
        auto_aim::TargetSelector pan_selector(config);
        const auto pan_decision = pan_selector.select(translate, 0.0, 0.0);
        passed &= expect(!pan_selector.targetStill() && pan_decision.valid &&
                             pan_decision.aim_point[0] > 0.1,
                         "real translation must pass the motion gate immediately");
    }

    if (!passed) return 1;
    std::cout << "target_selector_test passed" << std::endl;
    return 0;
}
