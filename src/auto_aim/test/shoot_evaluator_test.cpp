#include "control/shoot_evaluator.hpp"
#include <cstdio>
namespace {
int failures = 0;
void check(bool ok, const char* what) { if (!ok) { std::printf("FAIL: %s\n", what); ++failures; } }
}
int main()
{
    auto_aim::ShootEvaluator evaluator;
    auto_aim::ShootInput in;
    in.solved = true; in.ballistics_valid = true; in.target_distance = 6.0;
    in.yaw_error = 0.01; in.pitch_error = 0.01; in.target_angular_rate = 1.0;
    check(evaluator.evaluate(in) == auto_aim::ShootVerdict::Fire, "settled target fires");
    auto not_settled = in; not_settled.yaw_error = 0.30;
    check(evaluator.evaluate(not_settled) == auto_aim::ShootVerdict::NotSettled, "large yaw error blocks");
    auto fast = in; fast.target_angular_rate = 5.0;
    check(evaluator.evaluate(fast) == auto_aim::ShootVerdict::RateTooHigh, "spinning target blocks");
    auto far = in; far.target_distance = 20.0;
    check(evaluator.evaluate(far) == auto_aim::ShootVerdict::OutOfRange, "far target blocks");
    auto no_solve = in; no_solve.ballistics_valid = false;
    check(evaluator.evaluate(no_solve) == auto_aim::ShootVerdict::BallisticsInvalid, "no ballistics blocks");
    auto hit = in; hit.already_hit_this_target = true;
    check(evaluator.evaluate(hit) == auto_aim::ShootVerdict::AlreadyHit, "one shot per target");
    auto cool = in; cool.cooldown_ready = false;
    check(evaluator.evaluate(cool) == auto_aim::ShootVerdict::Cooldown, "cooldown blocks");
    if (failures == 0) { std::printf("shoot_evaluator_test: all checks passed\n"); return 0; }
    std::printf("shoot_evaluator_test: %d check(s) failed\n", failures);
    return 1;
}
