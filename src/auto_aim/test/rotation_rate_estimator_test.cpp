#include "estimation/rotation_rate_estimator.hpp"

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
    auto_aim::RotationRateEstimatorConfig config;

    auto_aim::RotationRateEstimator stationary(config);
    for (int frame = 0; frame < 120; ++frame) {
        const double time = frame * dt;
        const double noise = (frame % 2 == 0) ? 0.06 : -0.06;
        stationary.update(noise, time);
    }
    passed &= expect(std::abs(stationary.omega()) < 0.2,
                     "stationary radial angle must not create spin");

    auto_aim::RotationRateEstimator spinning(config);
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

    // ---- 阶跃（装甲板翻面）不能被当成转速 ---------------------------------
    // 实车实测：位置被硬约束钉住时，对称装甲板的 PnP 姿态会在 ±90° 上翻面
    // （1.57 rad 的单步跳变），窗口内的中位斜率会变成 1.57/窗口 ≈ 0.8 rad/s
    // （44°/s），乘 386 ms 提前量就是瞄点甩 5°。见 rotation_rate_estimator.hpp。
    {
        auto_aim::RotationRateEstimator stepped(config);
        for (int frame = 0; frame < 60; ++frame) {
            stepped.update(0.0, frame * dt);
        }
        const double flip_time = 60 * dt + 0.1;   // 跨一次丢失后再观测
        stepped.update(1.57, flip_time);
        passed &= expect(std::abs(stepped.omega()) < 0.6,
                         "a 90-degree plate flip must not be reported as a rotation rate");
    }

    if (!passed) return 1;
    std::cout << "rotation_rate_estimator_test passed, stationary="
              << stationary.omega() << ", spin=" << spinning.omega()
              << std::endl;
    return 0;
}
