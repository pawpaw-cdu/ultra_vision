#include "shooter.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>

namespace auto_aim
{
    Shooter::Shooter(const ShooterConfig& config)
        : config_(config)
    {
        if (!std::isfinite(config_.min_fire_interval) ||
            config_.min_fire_interval < 0.0) {
            config_.min_fire_interval = 0.10;
        }
        config_.ready_handoff_gain = std::max(
            0.0, std::min(config_.ready_handoff_gain, 1.0));
        config_.end_handoff_gain = std::max(
            0.0, std::min(config_.end_handoff_gain, config_.ready_handoff_gain));

        // 判据参数：默认值在 ShootEvaluatorConfig 里；这里允许用环境变量覆盖，
        // 方便 A/B（例如"角速度上限 3.2 vs 4.5"这种），不用改 yaml、不用重编译。
        ShootEvaluatorConfig eval;
        if (const char* value = std::getenv("ULTRA_VISION_SHOOT_MAX_RATE")) {
            eval.max_angular_rate = std::atof(value);
        }
        if (const char* value = std::getenv("ULTRA_VISION_SHOOT_TOL_GROWTH")) {
            eval.tolerance_growth_per_m = std::atof(value);
        }
        if (const char* value = std::getenv("ULTRA_VISION_SHOOT_YAW_TOL")) {
            eval.yaw_tolerance = std::atof(value);
        }
        if (const char* value = std::getenv("ULTRA_VISION_SHOOT_PITCH_TOL")) {
            eval.pitch_tolerance = std::atof(value);
        }
        evaluator_ = ShootEvaluator(eval);
    }

    ShooterOutput Shooter::update(
        const TargetDecision& decision, bool aim_ready, double timestamp,
        const ShootInput& shoot_input)
    {
        TickContext context{&decision, aim_ready, timestamp};
        // 决策②：只要有完整的判据输入，就由 ShootEvaluator 一处裁决；
        // 否则退回旧的 aim_ready 布尔（旧调用/单测仍然可用）。
        context.shoot_input = shoot_input;
        context.has_verdict = shoot_input.solved || shoot_input.already_hit_this_target;
        if (context.has_verdict) {
            context.verdict = evaluator_.evaluate(shoot_input);
            context.aim_ready = context.verdict == ShootVerdict::Fire;
        }
        ShooterOutput output = fireDecisionRoot(context);
        last_armor_id_ = decision.armor_id;
        output.state = state_;
        return output;
    }

    ShooterOutput Shooter::fireDecisionRoot(const TickContext& context)
    {
        if (!context.decision->valid) return endBranch();

        const bool target_changed = context.decision->armor_id != last_armor_id_;
        const bool gain_recovered =
            context.decision->handoff_gain >= config_.ready_handoff_gain;
        const bool ready_eligible = target_changed ||
            (context.decision->handoff_gain > config_.end_handoff_gain &&
             (gain_recovered || !context.decision->in_fire_window));
        if (!ready_eligible) return endBranch();

        return readyBranch(context);
    }

    ShooterOutput Shooter::readyBranch(const TickContext& context)
    {
        if (context.decision->in_fire_window && context.aim_ready &&
            context.decision->handoff_gain > config_.end_handoff_gain) {
            return firingLeaf(context);
        }
        return errorBranch(context);
    }

    ShooterOutput Shooter::firingLeaf(const TickContext& context)
    {
        ShooterOutput output;
        output.ready_branch = true;
        output.firing_branch = true;
        if (!context.decision->in_fire_window ||
            context.decision->handoff_gain <= config_.end_handoff_gain) {
            return endBranch();
        }
        if (!context.aim_ready) {
            state_ = ShooterState::ERROR;
            output.state = state_;
            output.error_branch = true;
            output.error_reason = ShooterErrorReason::GIMBAL_ERROR;
            return output;
        }

        state_ = ShooterState::FIRING;
        output.state = state_;
        const bool cooldown_elapsed = !last_fire_time_initialized_ ||
            !std::isfinite(context.timestamp) ||
            context.timestamp - last_fire_time_ >= config_.min_fire_interval;
        if (cooldown_elapsed) {
            output.fire = true;
            last_fire_time_initialized_ = true;
            last_fire_time_ = context.timestamp;
        }
        return output;
    }

    ShooterOutput Shooter::errorBranch(const TickContext& context)
    {
        ShooterOutput output;
        output.ready_branch = true;
        output.error_branch = true;
        state_ = ShooterState::ERROR;
        output.state = state_;
        if (!context.decision->in_fire_window ||
            context.decision->handoff_gain <= config_.end_handoff_gain) {
            output.error_reason = ShooterErrorReason::OUT_OF_FIRE_WINDOW;
        } else if (context.has_verdict) {
            // 把判据结论映射成原因码：日志里能直接看出"被哪一条挡住"。
            switch (context.verdict) {
            case ShootVerdict::NoTarget: output.error_reason = ShooterErrorReason::NO_TARGET; break;
            case ShootVerdict::BallisticsInvalid:
                output.error_reason = ShooterErrorReason::BALLISTICS_INVALID; break;
            case ShootVerdict::AlreadyHit: output.error_reason = ShooterErrorReason::ALREADY_HIT; break;
            case ShootVerdict::Cooldown: output.error_reason = ShooterErrorReason::COOLDOWN; break;
            case ShootVerdict::RateTooHigh:
                output.error_reason = ShooterErrorReason::RATE_TOO_HIGH; break;
            case ShootVerdict::OutOfRange: output.error_reason = ShooterErrorReason::OUT_OF_RANGE; break;
            case ShootVerdict::NotSettled: output.error_reason = ShooterErrorReason::NOT_SETTLED; break;
            case ShootVerdict::Fire: output.error_reason = ShooterErrorReason::GIMBAL_ERROR; break;
            }
        } else {
            output.error_reason = ShooterErrorReason::GIMBAL_ERROR;
        }
        return output;
    }

    ShooterOutput Shooter::endBranch()
    {
        ShooterOutput output;
        output.end_branch = true;
        state_ = ShooterState::END;
        output.state = state_;
        return output;
    }
}
