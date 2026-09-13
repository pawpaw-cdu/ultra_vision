#include "control/gimbal_aimer.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>

namespace
{
    constexpr double kPi = 3.14159265358979323846;

    bool near(double lhs, double rhs, double tolerance = 1e-9)
    {
        return std::abs(lhs - rhs) <= tolerance;
    }

    bool expect(bool condition, const char* message)
    {
        if (!condition) std::cerr << "FAILED: " << message << std::endl;
        return condition;
    }

    std::array<double, 3> targetFromDesiredCommand(
        double current_yaw,
        double current_pitch,
        double desired_yaw,
        double desired_pitch)
    {
        const double desired_cos_pitch = std::cos(desired_pitch);
        const double desired_x = std::sin(desired_yaw) * desired_cos_pitch;
        const double desired_y = std::sin(desired_pitch);
        const double desired_z = -std::cos(desired_yaw) * desired_cos_pitch;

        const double current_cos_yaw = std::cos(current_yaw);
        const double current_sin_yaw = std::sin(current_yaw);
        const double current_cos_pitch = std::cos(current_pitch);
        const double current_sin_pitch = std::sin(current_pitch);

        const double yaw_frame_x = current_cos_yaw * desired_x +
            current_sin_yaw * desired_z;
        const double yaw_frame_y = desired_y;
        const double yaw_frame_z = -current_sin_yaw * desired_x +
            current_cos_yaw * desired_z;

        const double local_x = yaw_frame_x;
        const double local_y = current_cos_pitch * yaw_frame_y +
            current_sin_pitch * yaw_frame_z;
        const double local_z = -current_sin_pitch * yaw_frame_y +
            current_cos_pitch * yaw_frame_z;

        return {local_x, -local_y, -local_z};
    }

    std::array<double, 3> targetFromAbsoluteYaw(double current_yaw, double target_yaw)
    {
        return targetFromDesiredCommand(current_yaw, 0.0, target_yaw, 0.0);
    }

    double simulateConvergence(auto_aim::GimbalAimer& aimer,
                               double target_yaw,
                               double dt,
                               double duration)
    {
        double current_yaw = 0.0;
        for (double elapsed = 0.0; elapsed < duration; elapsed += dt) {
            const auto command = aimer.solve(
                targetFromAbsoluteYaw(current_yaw, target_yaw),
                current_yaw, 0.0, dt);
            if (!command.valid) break;
            current_yaw = command.yaw;
        }
        return current_yaw;
    }
}

int main()
{
    bool passed = true;
    auto_aim::GimbalAimConfig config;
    config.max_yaw_velocity = 2.2;
    config.max_pitch_velocity = 1.5;
    config.max_yaw_acceleration = 8.0;
    config.max_pitch_acceleration = 6.0;
    config.max_yaw_jerk = 80.0;
    config.max_pitch_jerk = 60.0;
    auto_aim::GimbalAimer aimer(config);

    const auto center = aimer.solveTargetAngles({0.0, 0.0, 5.0}, 0.0, 0.0);
    passed &= expect(center.valid, "center target must be valid");
    passed &= expect(near(center.yaw, 0.0), "center target yaw must be zero");
    passed &= expect(near(center.pitch, 0.0), "center target pitch must be zero");

    const auto right = aimer.solveTargetAngles({1.0, 0.0, 5.0}, 0.0, 0.0);
    passed &= expect(right.yaw > 0.0, "right target must produce positive yaw");
    passed &= expect(near(right.yaw, std::atan2(1.0, 5.0)),
                     "right target must produce the expected absolute yaw");

    const auto up = aimer.solveTargetAngles({0.0, -1.0, 5.0}, 0.0, 0.0);
    passed &= expect(up.pitch > 0.0, "up target must produce positive pitch");
    passed &= expect(near(up.pitch, std::atan2(1.0, 5.0)),
                     "up target must produce the expected absolute pitch");

    constexpr double desired_yaw = 0.65;
    constexpr double desired_pitch = 0.35;
    const auto combined = aimer.solveTargetAngles(
        targetFromDesiredCommand(-0.7, 0.3, desired_yaw, desired_pitch), -0.7, 0.3);
    passed &= expect(near(combined.yaw, desired_yaw, 1e-8) &&
                     near(combined.pitch, desired_pitch, 1e-8),
                     "combined yaw and pitch must round-trip through the solver");

    const auto clamped = aimer.solveTargetAngles({0.0, -0.2, 1.0}, 0.0, 1.3);
    passed &= expect(near(clamped.pitch, config.pitch_max),
                     "pitch must be clamped to the configured software limit");

    const auto behind = aimer.solveTargetAngles({0.0, 0.0, -1.0}, 0.0, 0.0);
    passed &= expect(!behind.valid, "target behind camera must be rejected");

    const auto invalid = aimer.solveTargetAngles({0.0, 0.0, std::nan("")}, 0.0, 0.0);
    passed &= expect(!invalid.valid, "non-finite target must be rejected");

    auto_aim::GimbalAimer trajectory_aimer(config);
    const auto initial = trajectory_aimer.solve({1.0, 0.0, 5.0}, 0.0, 0.0, 1.0 / 30.0);
    passed &= expect(initial.valid && std::abs(initial.yaw) <=
                         config.max_yaw_velocity / 30.0 + 1e-12,
                     "the first trajectory sample must not jump to the target");
    const auto first_step = trajectory_aimer.solve({1.0, 0.0, 5.0}, 0.0, 0.0, 1.0 / 30.0);
    passed &= expect(first_step.yaw > 0.0 && first_step.yaw < std::atan2(1.0, 5.0),
                     "the trajectory must move toward the target gradually");
    passed &= expect(std::abs(first_step.yaw_velocity) <= config.max_yaw_velocity + 1e-9,
                     "commanded yaw velocity must stay inside its physical limit");
    passed &= expect(std::abs(first_step.yaw - initial.yaw) <=
                         config.max_yaw_velocity / 30.0 + 1e-12,
                     "one yaw step must not exceed the velocity limit times dt");

    auto_aim::GimbalAimer aimer_30hz(config);
    auto_aim::GimbalAimer aimer_60hz(config);
    const double yaw_30hz = simulateConvergence(aimer_30hz, 0.70, 1.0 / 30.0, 1.0);
    const double yaw_60hz = simulateConvergence(aimer_60hz, 0.70, 1.0 / 60.0, 1.0);
    passed &= expect(std::abs(yaw_30hz - 0.70) < 0.01,
                     "30 Hz trajectory must converge to the requested angle");
    passed &= expect(std::abs(yaw_60hz - 0.70) < 0.01,
                     "60 Hz trajectory must converge to the requested angle");
    passed &= expect(std::abs(yaw_30hz - yaw_60hz) < 0.01,
                     "trajectory must be substantially frame-rate independent");

    auto_aim::GimbalAimer switch_aimer(config);
    double current_yaw = 0.0;
    double current_pitch = 0.0;
    auto_aim::GimbalCommand switch_command;
    const double dt = 1.0 / 30.0;
    for (int frame = 0; frame < 90; ++frame) {
        switch_command = switch_aimer.solve(
            targetFromAbsoluteYaw(current_yaw, 0.70), current_yaw, current_pitch, dt);
        current_yaw = switch_command.yaw;
    }
    const double before_switch_yaw = current_yaw;
    const auto switched = switch_aimer.solve(
        targetFromAbsoluteYaw(current_yaw, -0.70), current_yaw, current_pitch, dt);
    passed &= expect(std::abs(switched.yaw - before_switch_yaw) <=
                         config.max_yaw_velocity * dt + 1e-12,
                     "an armor-plate switch must not create a position jump");
    passed &= expect(std::abs(switched.yaw_acceleration -
                              switch_command.yaw_acceleration) <=
                         config.max_yaw_jerk * dt + 1e-9,
                     "an armor-plate switch must not create an acceleration step");

    auto_aim::GimbalAimConfig model_feedforward_config = config;
    model_feedforward_config.feedforward_gain = 0.2;
    model_feedforward_config.max_yaw_acceleration = 100.0;
    auto_aim::GimbalAimer model_feedforward_aimer(model_feedforward_config);
    double modeled_yaw = 0.0;
    auto_aim::GimbalCommand modeled_command;
    for (int frame = 0; frame < 60; ++frame) {
        auto_aim::GimbalTargetAngles modeled_target;
        modeled_target.valid = true;
        modeled_target.velocity_valid = true;
        modeled_target.yaw = modeled_yaw + 1.0 * dt;
        modeled_target.yaw_velocity = 1.0;
        modeled_command = model_feedforward_aimer.trackTargetAngles(
            modeled_target, modeled_yaw, 0.0, dt);
        modeled_yaw = modeled_command.yaw;
    }
    passed &= expect(std::abs(modeled_command.yaw_velocity - 1.0) < 0.05,
                     "model-derived feedforward must not be scaled by the legacy gain");

    if (!passed) return 1;
    std::cout << "gimbal_aimer_test passed" << std::endl;
    return 0;
}
