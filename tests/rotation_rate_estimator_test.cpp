#include "ultra_vision/estimation/rotation_rate_estimator.hpp"

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
    constexpr double dt = 1.0 / 30.0;
    ultra_vision::RotationRateEstimatorConfig config;

    ultra_vision::RotationRateEstimator stationary(config);
    for (int frame = 0; frame < 120; ++frame) {
        const double time = frame * dt;
        const double noise = (frame % 2 == 0) ? 0.06 : -0.06;
        stationary.update(noise, time);
    }
    passed &= expect(std::abs(stationary.omega()) < 0.2,
                     "stationary radial angle must not create spin");

    ultra_vision::RotationRateEstimator spinning(config);
    for (int frame = 0; frame < 120; ++frame) {
        const double time = frame * dt;
        const double noise = (frame % 3 == 0) ? 0.05 : -0.025;
        spinning.update(3.0 * time + noise, time);
    }
    passed &= expect(std::abs(spinning.omega() - 3.0) < 0.25,
                     "constant small-gyro phase must recover stable angular rate");

    const double before_outlier = spinning.omega();
    spinning.update(3.0 * (120 * dt) + 0.8, 120 * dt);
    passed &= expect(std::abs(spinning.omega() - before_outlier) < 0.35,
                     "a single phase outlier must not create an angular-rate jump");

    if (!passed) return 1;
    std::cout << "rotation_rate_estimator_test passed, stationary="
              << stationary.omega() << ", spin=" << spinning.omega()
              << std::endl;
    return 0;
}
