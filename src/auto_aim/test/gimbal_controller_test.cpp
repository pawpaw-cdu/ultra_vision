#include "control/gimbal_controller.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

namespace
{
    struct SentCommand {
        double yaw = 0.0;
        double pitch = 0.0;
        std::chrono::steady_clock::time_point sent_at;
    };

    bool expect(bool condition, const char* message)
    {
        if (!condition) std::cerr << "FAILED: " << message << std::endl;
        return condition;
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

    constexpr double command_rate = 100.0;
    std::mutex commands_mutex;
    std::vector<SentCommand> commands;
    auto_aim::GimbalController controller(
        config,
        [&commands, &commands_mutex](double yaw, double pitch) {
            std::lock_guard<std::mutex> lock(commands_mutex);
            commands.push_back({yaw, pitch, std::chrono::steady_clock::now()});
            return true;
        },
        command_rate);

    auto_aim::GimbalTargetAngles target;
    target.valid = true;
    target.yaw = 0.70;
    target.pitch = 0.20;
    controller.setTargetAngles(target);

    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    auto_aim::GimbalControllerSnapshot snapshot;
    while (std::chrono::steady_clock::now() < deadline) {
        snapshot = controller.snapshot();
        if (snapshot.valid && snapshot.settled) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }

    passed &= expect(snapshot.valid && snapshot.settled,
                     "the command scheduler must settle on a stationary target");
    passed &= expect(std::abs(snapshot.yaw - target.yaw) < 0.01 &&
                         std::abs(snapshot.pitch - target.pitch) < 0.01,
                     "the command scheduler must converge to both target angles");

    std::vector<SentCommand> copied_commands;
    {
        std::lock_guard<std::mutex> lock(commands_mutex);
        copied_commands = commands;
    }
    passed &= expect(copied_commands.size() >= 50,
                     "the command scheduler must send at the configured high rate");

    bool command_steps_limited = true;
    for (size_t i = 1; i < copied_commands.size(); ++i) {
        const double yaw_step = std::abs(auto_aim::GimbalAimer::normalizeAngle(
            copied_commands[i].yaw - copied_commands[i - 1].yaw));
        const double pitch_step = std::abs(
            copied_commands[i].pitch - copied_commands[i - 1].pitch);
        const double elapsed = std::chrono::duration<double>(
            copied_commands[i].sent_at - copied_commands[i - 1].sent_at).count();
        // Callback timestamps are sampled just after the controller computes
        const double bounded_elapsed = std::max(0.0, elapsed) + 1.0 / command_rate;
        command_steps_limited &=
            yaw_step <= config.max_yaw_velocity * bounded_elapsed + 1e-9 &&
            pitch_step <= config.max_pitch_velocity * bounded_elapsed + 1e-9;
    }

    passed &= expect(command_steps_limited,
                     "streamed yaw commands must remain velocity limited");

    target.yaw = -0.70;
    target.pitch = -0.20;
    controller.setTargetAngles(target);
    const auto before_switch = snapshot;
    const double yaw_before_switch = snapshot.yaw;
    std::this_thread::sleep_for(std::chrono::milliseconds(15));
    const auto switched = controller.snapshot();
    const double switch_elapsed = std::max(
        0.0, (switched.timestamp_us - before_switch.timestamp_us) / 1e6) +
        1.0 / command_rate;
    passed &= expect(std::abs(switched.yaw - yaw_before_switch) <=
                         config.max_yaw_velocity * switch_elapsed + 1e-9,
                     "a target switch must not create a streamed position jump");

    target.pitch = 2.0;
    controller.setTargetAngles(target);
    std::this_thread::sleep_for(std::chrono::milliseconds(1200));
    {
        std::lock_guard<std::mutex> lock(commands_mutex);
        copied_commands = commands;
    }
    bool pitch_within_limit = true;
    for (const auto& command : copied_commands) {
        pitch_within_limit &= command.pitch <= config.pitch_max + 1e-9 &&
            command.pitch >= config.pitch_min - 1e-9;
    }
    passed &= expect(pitch_within_limit,
                     "the controller must clamp every transmitted pitch command");

    controller.stop();
    if (!passed) return 1;
    std::cout << "gimbal_controller_test passed" << std::endl;
    return 0;
}
