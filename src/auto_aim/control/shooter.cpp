#include "shooter.hpp"

#include <algorithm>
#include <cmath>

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
    }

    ShooterOutput Shooter::update(
        const TargetDecision& decision, bool aim_ready, double timestamp)
    {
        TickContext context{&decision, aim_ready, timestamp};
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
        output.error_reason = (!context.decision->in_fire_window ||
            context.decision->handoff_gain <= config_.end_handoff_gain)
            ? ShooterErrorReason::OUT_OF_FIRE_WINDOW
            : ShooterErrorReason::GIMBAL_ERROR;
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
