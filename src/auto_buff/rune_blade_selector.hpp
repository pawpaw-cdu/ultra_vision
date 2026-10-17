#ifndef AUTO_AIM_ENERGY_RUNE_BLADE_SELECTOR_HPP
#define AUTO_AIM_ENERGY_RUNE_BLADE_SELECTOR_HPP

// Post-hit blade selection.
//
// Observation/model building may legitimately keep tracking the blade that was
// just activated. The fire target must be selected separately: after a scored
// hit, search among visible class-0/bright candidates, exclude the hit blade,
// and require temporal confirmation before handing the new physical slot to the
// aimer and fire gate.

#include <array>
#include <vector>

#include <opencv2/core.hpp>

namespace auto_aim::energy
{
    class RuneBladeSelector
    {
    public:
        struct Config
        {
            int confirm_frames = 2;
            /// Reject candidates this close to the blade that was just hit.
            double hit_clearance_orbit_ratio = 0.45;
            double class0_bonus = 1.0;
            double lit_bonus = 0.35;
        };

        struct Candidate
        {
            int class_id = 0;
            cv::Point2f center{0.0f, 0.0f};
            int slot = -1;
            bool lit = false;
        };

        struct Frame
        {
            bool solved = false;
            bool observation_fresh = false;
            bool hold = false;
            bool scored_hit = false;
            int observed_slot = -1;
            int hit_slot = -1;
            cv::Point2f hit_center{0.0f, 0.0f};
            float orbit_px = 0.0f;
            std::array<bool, 5> activated{};
            std::vector<Candidate> candidates;
        };

        struct Snapshot
        {
            bool valid = false;
            bool searching = false;
            bool retargeting = false;
            bool confirmed = false;
            int planned_slot = -1;
            int retarget_offset = 0;
            int target_class = -1;
            cv::Point2f target_center{0.0f, 0.0f};
        };

        RuneBladeSelector() = default;
        explicit RuneBladeSelector(const Config& config) : config_(config) {}

        void update(const Frame& frame, double now);
        void reset();
        Snapshot snapshot() const { return snapshot_; }

    private:
        struct Selection
        {
            bool valid = false;
            int slot = -1;
            int class_id = -1;
            cv::Point2f center{0.0f, 0.0f};
            double score = -1e9;
        };

        Selection chooseCandidate(const Frame& frame) const;
        int uniqueUnactivatedSlot(const Frame& frame) const;

        Config config_;
        bool searching_ = false;
        int hit_slot_ = -1;
        cv::Point2f hit_center_{0.0f, 0.0f};
        int pending_slot_ = -1;
        int pending_frames_ = 0;
        Selection selected_;
        Snapshot snapshot_;
    };
} // namespace auto_aim::energy

#endif // AUTO_AIM_ENERGY_RUNE_BLADE_SELECTOR_HPP
