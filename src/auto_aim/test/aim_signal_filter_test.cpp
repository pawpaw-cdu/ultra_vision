#include "control/aim_signal_filter.hpp"

#include <cmath>
#include <iostream>

namespace
{
    bool expect(bool condition, const char* message)
    {
        if (!condition) std::cerr << "FAILED: " << message << std::endl;
        return condition;
    }
}

int main()
{
    bool passed = true;
    auto_aim::AimSignalFilterConfig config;
    config.process_noise_acceleration = 40.0;
    config.measurement_noise = 0.0025;
    config.reset_innovation = 0.50;
    auto_aim::AimSignalFilter filter(config);

    double raw_squared_error = 0.0;
    double filtered_squared_error = 0.0;
    int samples = 0;
    const double dt = 1.0 / 30.0;
    for (int frame = 0; frame < 120; ++frame) {
        const double time = frame * dt;
        const double truth = 0.4 * std::sin(0.9 * time);
        const double noise = (frame % 2 == 0) ? 0.025 : -0.025;
        const auto estimate = filter.update(truth + noise, 0.1, 2, time);
        if (frame >= 30) {
            raw_squared_error += (truth + noise - truth) * (truth + noise - truth);
            filtered_squared_error += (estimate.yaw - truth) * (estimate.yaw - truth);
            ++samples;
        }
    }
    passed &= expect(filtered_squared_error < raw_squared_error * 0.35,
                     "filtered angle regression must reduce measurement noise");

    auto_aim::AimSignalFilter switch_filter(config);
    for (int frame = 0; frame < 10; ++frame) {
        switch_filter.update(0.2, 0.1, 2, frame * dt);
    }
    const auto switched = switch_filter.update(1.2, 0.1, 3, 10.0 * dt);
    passed &= expect(switched.reset && std::abs(switched.yaw - 1.2) < 1e-9,
                     "changing armor target must reset the angle regression");

    const auto timed_out = switch_filter.update(1.25, 0.1, 3, 1.0);
    passed &= expect(timed_out.reset && std::abs(timed_out.yaw - 1.25) < 1e-9,
                     "a long time gap must reset the angle regression");

    if (!passed) return 1;
    std::cout << "aim_signal_filter_test passed" << std::endl;
    return 0;
}
