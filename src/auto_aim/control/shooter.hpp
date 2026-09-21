#ifndef AUTO_AIM_SHOOTER_HPP
#define AUTO_AIM_SHOOTER_HPP

#include "target_selector.hpp"
#include "shoot_evaluator.hpp"

namespace auto_aim
{
    enum class ShooterState { IDLE, READY, FIRING, ERROR, END };

    enum class ShooterErrorReason {
        NONE,
        OUT_OF_FIRE_WINDOW,
        GIMBAL_ERROR,
        // 以下来自 ShootEvaluator 的结论（决策②集中在那一处，见 shoot_evaluator.hpp）
        NO_TARGET,
        BALLISTICS_INVALID,
        ALREADY_HIT,
        COOLDOWN,
        RATE_TOO_HIGH,
        OUT_OF_RANGE,
        NOT_SETTLED,
    };

    struct ShooterConfig {
        double min_fire_interval = 0.10;
        double ready_handoff_gain = 0.95;
        double end_handoff_gain = 0.65;
    };

    struct ShooterOutput {
        ShooterState state = ShooterState::IDLE;
        bool fire = false;
        bool ready_branch = false;
        bool firing_branch = false;
        bool error_branch = false;
        bool end_branch = false;
        ShooterErrorReason error_reason = ShooterErrorReason::NONE;
    };

    class Shooter
    {
    public:
        explicit Shooter(const ShooterConfig& config = {});

        ShooterOutput update(const TargetDecision& decision,
                             bool aim_ready,
                             double timestamp,
                             const ShootInput& shoot_input = {});

        ShooterState state() const { return state_; }

    private:
        struct TickContext {
            const TargetDecision* decision = nullptr;
            bool aim_ready = false;
            double timestamp = 0.0;
            // 判据输入（由入口填）。填了就以 ShootEvaluator 的结论为准。
            ShootInput shoot_input;
            bool has_verdict = false;
            ShootVerdict verdict = ShootVerdict::NoTarget;
        };

        ShooterOutput fireDecisionRoot(const TickContext& context);
        ShooterOutput readyBranch(const TickContext& context);
        ShooterOutput firingLeaf(const TickContext& context);
        ShooterOutput errorBranch(const TickContext& context);
        ShooterOutput endBranch();

        ShooterConfig config_;
        ShootEvaluator evaluator_;
        ShooterState state_ = ShooterState::IDLE;
        int last_armor_id_ = -1;
        bool last_fire_time_initialized_ = false;
        double last_fire_time_ = 0.0;
    };
}

#endif
