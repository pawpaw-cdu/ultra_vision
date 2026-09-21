// Self-checking test for the energy-rune estimator and aimer.
//
// The pipeline is fed analytic observations of a rotating rune (no image and
// no simulator needed), so the numerical behaviour can be checked anywhere:
//   * the small rune turns at a fixed pi/3 rad/s and the filter must follow it;
//   * the large rune turns at a*sin(w*t + phi) + (2.09 - a) and the filter must
//     identify a and w well enough to predict better than a constant-rate model;
//   * the aimer must point at the predicted plate with the simulator's angle
//     convention (positive yaw right, positive pitch up).

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "auto_buff/rune_aimer.hpp"
#include "auto_buff/rune_target.hpp"
#include "auto_buff/rune_types.hpp"
#include "auto_buff/support/math.hpp"

namespace
{
    using namespace auto_aim::energy;

    int failures = 0;

    void check(bool condition, const std::string& what)
    {
        if (!condition) {
            std::printf("FAIL: %s\n", what.c_str());
            ++failures;
        }
    }

    double wrap(double angle) { return limitRad(angle); }

    // Ground-truth rune: the mechanism sits d metres in front of the camera and
    // the plate orbits in the y-z plane of the z-up world frame.
    struct RuneTruth
    {
        Eigen::Vector3d center{3.0, 0.0, 0.0};
        double radius = 0.7;

        Eigen::Vector3d platePosition(double roll) const
        {
            const Eigen::Matrix3d R = rotationMatrix(Eigen::Vector3d(0.0, 0.0, roll));
            return R * Eigen::Vector3d(0.0, 0.0, radius) + center;
        }

        PowerRune observation(double roll) const
        {
            PowerRune rune;
            rune.r_center = cv::Point2f(320.0f, 240.0f);
            rune.light_num = 1;
            rune.fanblades.emplace_back(FanBlade());
            rune.ypd_in_world = xyz2ypd(center);
            rune.ypr_in_world = Eigen::Vector3d(0.0, 0.0, roll);
            rune.blade_xyz_in_world = platePosition(roll);
            rune.blade_ypd_in_world = xyz2ypd(rune.blade_xyz_in_world);
            // RuneTarget::getTarget 只接受"solved"的观测（未解算的观测绝不允许
            // 进入估计器，见 PowerRune::solved 的注释）。合成观测必须显式置位，
            // 否则整个自检会静默失效（滤波器一次都不初始化）。
            rune.solved = true;
            return rune;
        }
    };

    // Fixed-rate small rune: the filter has to lock onto pi/3 rad/s.
    void testSmallRune()
    {
        const RuneTruth truth;
        RuneTarget target(RuneMode::Small);
        RuneAimerConfig aimer_config;
        RuneAimer aimer(aimer_config);

        const double rate = kPi / 3.0;
        const double dt = 1.0 / 30.0;
        const double start = 0.4;
        double last_roll = 0.0;
        double last_time = 0.0;
        for (int frame = 0; frame < 90; ++frame) {
            const double t = frame * dt;
            const double roll = start + rate * t;
            const auto rune = truth.observation(roll);
            target.getTarget(rune, t);
            if (frame == 89) {
                last_roll = roll;
                last_time = t;
            }
        }
        check(!target.isUnsolvable(), "small rune: filter must be solvable");

        const double roll_error = wrap(target.ekfX()[5] - last_roll);
        check(std::abs(roll_error) < 7.0 * kPi / 180.0,
              "small rune: roll tracking error < 7 deg (got " +
                  std::to_string(roll_error * 180.0 / kPi) + " deg)");

        // One-step prediction must advance at the fixed rate.
        RuneTarget predicted = target;
        const double horizon = 0.15;
        predicted.predict(horizon);
        const double predicted_roll = predicted.ekfX()[5];
        const double truth_roll = start + rate * (last_time + horizon);
        const double predicted_error = wrap(predicted_roll - truth_roll);
        check(std::abs(predicted_error) < 4.0 * kPi / 180.0,
              "small rune: 0.15 s prediction error < 4 deg (got " +
                  std::to_string(predicted_error * 180.0 / kPi) + " deg)");

        // The command must point at that predicted plate.
        RuneTarget aim_target = target;
        const RuneCommand command = aimer.aim(aim_target, last_time, last_time);
        check(command.control, "small rune: aimer must produce a control command");
        const Eigen::Vector3d plate = truth.platePosition(truth_roll);
        const double expected_yaw = -std::atan2(plate[1], plate[0]);
        const double expected_pitch =
            std::atan2(plate[2], std::hypot(plate[0], plate[1]));
        check(std::abs(wrap(command.yaw - expected_yaw)) < 1.5 * kPi / 180.0,
              "small rune: aim yaw matches the predicted plate (got " +
                  std::to_string((command.yaw - expected_yaw) * 180.0 / kPi) + " deg)");
        // The aimer fires on a ballistic solution, so its pitch sits a degree
        // or two above the straight line to the plate (gravity compensation).
        check(std::abs(command.pitch - expected_pitch) < 3.5 * kPi / 180.0,
              "small rune: aim pitch matches the predicted plate (got " +
                  std::to_string((command.pitch - expected_pitch) * 180.0 / kPi) + " deg)");
    }

    // 观测到"已激活的那片"（class 1/2）时，估计器的相位必须留在**跟踪槽位**上，
    // 而不是跟着观测片跑：否则瞄点整跳一片，云台就在两片之间来回甩。
    // 这就是 PowerRune::slot_offset 的作用（现场实测：92 帧 >2° 瞄准跳变里
    // 有 10 帧落在"观测到已激活片"的帧上）。
    void testForeignSlotObservation()
    {
        const RuneTruth truth;
        RuneTarget target(RuneMode::Small);
        const double rate = kPi / 3.0;
        const double dt = 0.05;
        const double slot = 2.0 * kPi / 5.0;

        double roll = 0.3;
        double now = 0.0;
        for (int frame = 0; frame < 20; ++frame) {   // 先让滤波器收敛在跟踪槽位
            now += dt;
            roll += rate * dt;
            target.getTarget(truth.observation(roll), now);
        }
        const double tracked_before = target.ekfX()[5];

        const int offset = 2;   // 观测片比跟踪槽位超前 2 个槽位
        for (int frame = 0; frame < 10; ++frame) {
            now += dt;
            roll += rate * dt;
            PowerRune observed = truth.observation(roll + offset * slot);
            observed.slot_offset = offset;
            observed.slot_offset_valid = true;
            target.getTarget(observed, now);
        }

        const double tracked_after = target.ekfX()[5];
        const double drift = std::abs(wrap(tracked_after - roll));
        check(drift < 10.0 * kPi / 180.0,
              "foreign-slot observation: the phase stays on the tracked slot (got " +
                  std::to_string(drift * 180.0 / kPi) + " deg off)");
        check(std::abs(wrap(tracked_after - tracked_before - rate * 10.0 * dt)) <
                  10.0 * kPi / 180.0,
              "foreign-slot observation: the phase still advances at the rune rate");
    }

    // Variable-rate large rune: omega(t) = a sin(w t + phi) + 2.09 - a.
    void testLargeRune()
    {
        const RuneTruth truth;
        RuneTarget target(RuneMode::Large);
        RuneAimer aimer(RuneAimerConfig{});

        const double a = 1.0;
        const double w = 1.95;
        const double phi = 0.3;
        const double bias = 2.09 - a;
        const double start = -0.2;
        const auto rollAt = [&](double t) {
            return start - a / w * std::cos(w * t + phi) + a / w * std::cos(phi) + bias * t;
        };
        const auto rateAt = [&](double t) { return a * std::sin(w * t + phi) + bias; };

        const double dt = 1.0 / 30.0;
        const int frames = 120; // 4 s: long enough to identify the sinusoid
        for (int frame = 0; frame < frames; ++frame) {
            const double t = frame * dt;
            target.getTarget(truth.observation(rollAt(t)), t);
        }
        check(!target.isUnsolvable(), "large rune: filter must be solvable");
        check(std::abs(target.ekfX()[5] - wrap(rollAt((frames - 1) * dt))) <
                  12.0 * kPi / 180.0,
              "large rune: roll tracking error < 12 deg");
        check(std::abs(target.ekfX()[8] - w) < 0.30,
              "large rune: identified omega within 0.30 rad/s (got " +
                  std::to_string(target.ekfX()[8]) + ")");
        check(std::abs(target.ekfX()[7] - a) < 0.35,
              "large rune: identified amplitude within 0.35 (got " +
                  std::to_string(target.ekfX()[7]) + ")");

        // Prediction quality versus a constant pi/3 rad/s extrapolation.
        const double last_time = (frames - 1) * dt;
        const double horizon = 0.25;
        RuneTarget predicted = target;
        predicted.predict(horizon);
        const double truth_roll = rollAt(last_time + horizon);
        const double model_error = std::abs(wrap(predicted.ekfX()[5] - truth_roll));
        const double constant_error =
            std::abs(wrap(rollAt(last_time) + (kPi / 3.0) * horizon - truth_roll));
        check(model_error < constant_error,
              "large rune: variable-rate prediction beats the constant-rate model (got " +
                  std::to_string(model_error * 180.0 / kPi) + " deg vs " +
                  std::to_string(constant_error * 180.0 / kPi) + " deg)");
        check(model_error < 10.0 * kPi / 180.0,
              "large rune: 0.25 s prediction error < 10 deg (got " +
                  std::to_string(model_error * 180.0 / kPi) + " deg)");

        // Sanity: the identified rate must track the truth, not a constant.
        const double rate = rateAt(last_time);
        check(std::abs(target.ekfX()[6] - rate) < 0.45,
              "large rune: instantaneous rate within 0.45 rad/s (got " +
                  std::to_string(target.ekfX()[6]) + " vs " + std::to_string(rate) + ")");
    }
} // namespace

int main()
{
    testSmallRune();
    testLargeRune();
    testForeignSlotObservation();
    if (failures == 0) {
        std::printf("rune_target_test: all checks passed\n");
        return 0;
    }
    std::printf("rune_target_test: %d check(s) failed\n", failures);
    return 1;
}
