#ifndef AUTO_AIM_ENERGY_RUNE_FIRE_GATE_HPP
#define AUTO_AIM_ENERGY_RUNE_FIRE_GATE_HPP

// Fire eligibility is deliberately separate from observation/model building.
//
// RuneDetector may construct a geometry/phase observation from an activated
// blade when that is the best visible measurement. That observation is useful
// for the five-slot model, but it must never authorise a shot. This class owns
// only the firing decision for one planned physical slot.

#include <array>

namespace auto_aim::energy
{
    class RuneFireGate
    {
    public:
        struct Config
        {
            /// How long class-0 evidence for a physical slot remains usable
            /// while the detector misses that blade.
            double engageable_latch_s = 0.8;
        };

        struct Frame
        {
            bool solved = false;
            bool observation_fresh = false;
            int planned_slot = -1;
            /// Class observed on this physical slot this frame:
            /// -1 = unknown, 0 = inactive, >0 = activated.
            int planned_slot_class = -1;
            std::array<bool, 5> activated{};
        };

        struct Snapshot
        {
            bool engageable = false;
            bool slot_activated = false;
            bool class_known = false;
            bool latched = false;
            int planned_slot = -1;
            int planned_slot_class = -1;
        };

        RuneFireGate() = default;
        explicit RuneFireGate(const Config& config) : config_(config) {}

        void update(const Frame& frame, double now);
        void reset();
        Snapshot snapshot() const { return snapshot_; }
        bool engageable() const { return snapshot_.engageable; }

    private:
        Config config_;
        std::array<double, 5> engageable_until_{};
        Snapshot snapshot_;
    };
} // namespace auto_aim::energy

#endif // AUTO_AIM_ENERGY_RUNE_FIRE_GATE_HPP
