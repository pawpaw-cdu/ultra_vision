#include "auto_buff/rune_fire_gate.hpp"

#include <algorithm>

namespace auto_aim::energy
{
    void RuneFireGate::reset()
    {
        engageable_until_.fill(0.0);
        snapshot_ = Snapshot{};
    }

    void RuneFireGate::update(const Frame& frame, double now)
    {
        snapshot_ = Snapshot{};
        snapshot_.planned_slot = frame.planned_slot;
        snapshot_.planned_slot_class = frame.planned_slot_class;
        if (!frame.solved || frame.planned_slot < 0 || frame.planned_slot >= 5) {
            return;
        }

        const std::size_t slot = static_cast<std::size_t>(frame.planned_slot);
        snapshot_.slot_activated = frame.activated[slot];
        snapshot_.class_known = frame.planned_slot_class >= 0;
        if (snapshot_.slot_activated) {
            engageable_until_[slot] = 0.0;
            return;
        }

        if (frame.planned_slot_class == 0) {
            engageable_until_[slot] =
                std::max(engageable_until_[slot], now + config_.engageable_latch_s);
        } else if (frame.planned_slot_class > 0) {
            // Activated-looking evidence is a veto, never a model-building gate.
            engageable_until_[slot] = 0.0;
            return;
        }

        snapshot_.latched = engageable_until_[slot] > now;
        snapshot_.engageable =
            frame.observation_fresh && snapshot_.latched;
    }
} // namespace auto_aim::energy
