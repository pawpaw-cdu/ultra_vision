#include "control/shooter.hpp"

#include <iostream>

namespace
{
    bool expect(bool condition, const char* message)
    {
        if (!condition) std::cerr << "FAILED: " << message << std::endl;
        return condition;
    }

    auto_aim::TargetDecision makeDecision(int armor_id, bool in_window,
                                          double handoff_gain)
    {
        auto_aim::TargetDecision decision;
        decision.valid = true;
        decision.armor_id = armor_id;
        decision.in_fire_window = in_window;
        decision.handoff_gain = handoff_gain;
        return decision;
    }
}

int main()
{
    bool passed = true;
    auto_aim::ShooterConfig config;
    config.min_fire_interval = 0.10;
    config.ready_handoff_gain = 0.95;
    config.end_handoff_gain = 0.65;
    auto_aim::Shooter shooter(config);

    auto output = shooter.update({}, false, 0.0);
    passed &= expect(output.state == auto_aim::ShooterState::END &&
                         output.end_branch,
                     "invalid target must select the END branch");

    output = shooter.update(makeDecision(2, false, 1.0), false, 1.0);
    passed &= expect(output.state == auto_aim::ShooterState::ERROR &&
                         output.ready_branch && output.error_branch &&
                         output.error_reason ==
                             auto_aim::ShooterErrorReason::OUT_OF_FIRE_WINDOW,
                     "READY branch must report an out-of-window error");

    output = shooter.update(makeDecision(2, true, 1.0), false, 1.1);
    passed &= expect(output.state == auto_aim::ShooterState::ERROR &&
                         output.error_reason ==
                             auto_aim::ShooterErrorReason::GIMBAL_ERROR,
                     "READY branch must report a gimbal-error leaf");

    output = shooter.update(makeDecision(2, true, 1.0), true, 1.2);
    passed &= expect(output.state == auto_aim::ShooterState::FIRING && output.fire,
                     "shooter must fire after entering the window and becoming ready");

    output = shooter.update(makeDecision(2, true, 1.0), true, 1.25);
    passed &= expect(output.state == auto_aim::ShooterState::FIRING && !output.fire,
                     "shooter must respect the firing cooldown");

    output = shooter.update(makeDecision(2, true, 1.0), true, 1.31);
    passed &= expect(output.state == auto_aim::ShooterState::FIRING && output.fire,
                     "shooter must resume fire after the cooldown");

    output = shooter.update(makeDecision(2, true, 0.50), true, 1.40);
    passed &= expect(output.state == auto_aim::ShooterState::END && !output.fire,
                     "shooter must leave firing when the window is ending");

    output = shooter.update(makeDecision(3, false, 0.50), false, 1.50);
    passed &= expect(output.state == auto_aim::ShooterState::ERROR &&
                         output.ready_branch && output.error_branch &&
                         output.error_reason ==
                             auto_aim::ShooterErrorReason::OUT_OF_FIRE_WINDOW,
                     "a changed target must re-enter the ready tree before its window");

    output = shooter.update(makeDecision(3, true, 1.0), true, 1.60);
    passed &= expect(output.state == auto_aim::ShooterState::FIRING && output.fire,
                     "the next target must enter firing normally");

    auto_aim::Shooter recovering_shooter(config);
    recovering_shooter.update(makeDecision(2, true, 1.0), true, 2.0);
    output = recovering_shooter.update(makeDecision(2, true, 0.5), true, 2.1);
    passed &= expect(output.state == auto_aim::ShooterState::END,
                     "an ending window must stop continuous fire");
    output = recovering_shooter.update(makeDecision(2, true, 1.0), false, 2.2);
    passed &= expect(output.state == auto_aim::ShooterState::ERROR &&
                         output.error_reason ==
                             auto_aim::ShooterErrorReason::GIMBAL_ERROR,
                     "the same armor must recover into the ready tree after a tracking gap");
    output = recovering_shooter.update(makeDecision(2, true, 1.0), true, 2.3);
    passed &= expect(output.state == auto_aim::ShooterState::FIRING && output.fire,
                     "a recovered target must fire again after readiness");

    if (!passed) return 1;
    std::cout << "shooter_test passed" << std::endl;
    return 0;
}
