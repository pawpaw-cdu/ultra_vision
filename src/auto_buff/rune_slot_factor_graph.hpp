#ifndef AUTO_AIM_ENERGY_RUNE_SLOT_FACTOR_GRAPH_HPP
#define AUTO_AIM_ENERGY_RUNE_SLOT_FACTOR_GRAPH_HPP

// Discrete factor-graph / Viterbi layer for energy-rune slot identity.
//
// The continuous estimator already tracks the rune pose and phase. This layer
// solves the remaining discrete ambiguity in one step:
//   target slot in {0..4} x activation mask in {0..31}.
// At most 160 states are evaluated per frame. Candidate-to-slot assignment is
// one-to-one, and target identity, activation monotonicity and candidate
// semantics are combined as soft factors instead of independent hard gates.

#include <array>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>

namespace auto_aim::energy
{
    class RuneSlotFactorGraph
    {
    public:
        struct Config
        {
            /// Maximum candidate distance in units of mean adjacent slot spacing.
            double assignment_gate_spacing = 0.45;
            /// Matching reward makes "used slots" preferable to dropping a close
            /// candidate, while the gate still rejects unrelated detections.
            double assignment_reward = 0.25;
            double inactive_assignment_reward = 0.10;
            int max_assignment_candidates = 10;

            double target_miss_cost = 0.35;
            double inactive_class_reward = 0.70;
            double active_class_cost = 0.50;
            double activated_target_cost = 2.20;
            double active_mask_reward = 0.35;
            double inactive_mask_cost = 0.25;
            double hint_match_reward = 1.00;
            double hint_mismatch_cost = 1.00;

            double target_switch_cost = 0.55;
            double activation_set_cost = 0.35;
            /// Activation is monotonic in the physical reward model, but network
            /// classes are noisy. Keep removal as a finite soft penalty so a
            /// false one-frame hint can decay instead of latching for the round.
            double activation_remove_cost = 0.40;
        };

        struct Frame
        {
            bool solved = false;
            int anchor_slot = -1;  // physical slot represented by slot_centers[0]
            std::array<cv::Point2f, 5> slot_centers{};
            std::vector<std::pair<int, cv::Point2f>> candidates;
            std::array<bool, 5> activated_hint{};
            bool round_active = true;
        };

        struct Snapshot
        {
            bool valid = false;
            bool updated = false;
            int target_slot = -1;
            std::array<bool, 5> activated{};
            std::array<int, 5> assigned_class{{-1, -1, -1, -1, -1}};
            std::array<double, 5> slot_residual{};
            int assigned_count = 0;
            bool target_observed = false;
            double best_score = 0.0;
            double margin = 0.0;
        };

        RuneSlotFactorGraph() = default;
        explicit RuneSlotFactorGraph(const Config& config) : config_(config) {}

        void update(const Frame& frame, double now);
        void reset();
        const Snapshot& snapshot() const { return snapshot_; }

        static int encodeState(int target, int mask)
        {
            return (mask & 31) * 5 + target;
        }
        static int targetOf(int state) { return state % 5; }
        static int maskOf(int state) { return state / 5; }

    private:
        double emissionCost(int target, int mask) const;
        void assignCandidates(const Frame& frame);
        void initialize(const Frame& frame);
        void advance(const Frame& frame);

        Config config_;
        Snapshot snapshot_;
        std::array<double, 160> previous_{};
        std::array<double, 160> current_{};
        bool initialized_ = false;
        bool round_active_ = true;
        double now_ = 0.0;
    };
} // namespace auto_aim::energy

#endif // AUTO_AIM_ENERGY_RUNE_SLOT_FACTOR_GRAPH_HPP
