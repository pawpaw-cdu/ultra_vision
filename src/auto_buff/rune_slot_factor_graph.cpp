#include "auto_buff/rune_slot_factor_graph.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace auto_aim::energy
{
    namespace
    {
        constexpr double kInf = 1e9;
    }

    void RuneSlotFactorGraph::reset()
    {
        snapshot_ = Snapshot{};
        previous_.fill(-kInf);
        current_.fill(-kInf);
        initialized_ = false;
        round_active_ = true;
        now_ = 0.0;
    }

    void RuneSlotFactorGraph::assignCandidates(const Frame& frame)
    {
        snapshot_.assigned_class.fill(-1);
        snapshot_.slot_residual.fill(-1.0);
        snapshot_.assigned_count = 0;

        double spacing = 0.0;
        for (int slot = 0; slot < 5; ++slot) {
            const int next = (slot + 1) % 5;
            spacing += cv::norm(frame.slot_centers[static_cast<std::size_t>(next)] -
                                frame.slot_centers[static_cast<std::size_t>(slot)]);
        }
        spacing /= 5.0;
        if (!(spacing > 1.0)) return;

        struct CandidateRef
        {
            int source_index = -1;
            int class_id = 0;
            cv::Point2f center{};
            double min_distance = 0.0;
        };
        std::vector<CandidateRef> candidates;
        candidates.reserve(frame.candidates.size());
        const double gate = config_.assignment_gate_spacing * spacing;
        for (std::size_t i = 0; i < frame.candidates.size(); ++i) {
            CandidateRef candidate;
            candidate.source_index = static_cast<int>(i);
            candidate.class_id = frame.candidates[i].first;
            candidate.center = frame.candidates[i].second;
            candidate.min_distance = std::numeric_limits<double>::max();
            for (const cv::Point2f& slot_center : frame.slot_centers) {
                candidate.min_distance = std::min(
                    candidate.min_distance,
                    static_cast<double>(cv::norm(candidate.center - slot_center)));
            }
            if (candidate.min_distance <= gate) candidates.push_back(candidate);
        }
        std::sort(candidates.begin(), candidates.end(),
                  [](const CandidateRef& lhs, const CandidateRef& rhs) {
                      return lhs.min_distance < rhs.min_distance;
                  });
        const int count = std::min(
            static_cast<int>(candidates.size()),
            std::max(0, config_.max_assignment_candidates));
        if (count <= 0) return;
        candidates.resize(static_cast<std::size_t>(count));

        const int masks = 1 << count;
        const std::vector<double> no_assignment(masks, 0.0);
        std::vector<std::vector<double>> dp(6, std::vector<double>(masks, kInf));
        std::vector<std::vector<int>> choice(6, std::vector<int>(masks, -1));
        dp[0][0] = 0.0;
        for (int slot = 0; slot < 5; ++slot) {
            for (int mask = 0; mask < masks; ++mask) {
                if (dp[static_cast<std::size_t>(slot)][static_cast<std::size_t>(mask)] >= kInf) {
                    continue;
                }
                const double previous =
                    dp[static_cast<std::size_t>(slot)][static_cast<std::size_t>(mask)];
                if (previous < dp[static_cast<std::size_t>(slot + 1)]
                                  [static_cast<std::size_t>(mask)]) {
                    dp[static_cast<std::size_t>(slot + 1)][static_cast<std::size_t>(mask)] =
                        previous;
                    choice[static_cast<std::size_t>(slot + 1)]
                          [static_cast<std::size_t>(mask)] = -1;
                }
                for (int candidate = 0; candidate < count; ++candidate) {
                    const int bit = 1 << candidate;
                    if (mask & bit) continue;
                    const double distance = cv::norm(
                        candidates[static_cast<std::size_t>(candidate)].center -
                        frame.slot_centers[static_cast<std::size_t>(slot)]);
                    if (distance > gate) continue;
                    const double reward =
                        config_.assignment_reward +
                        (candidates[static_cast<std::size_t>(candidate)].class_id == 0
                             ? config_.inactive_assignment_reward
                             : 0.0);
                    const double cost = distance / spacing - reward;
                    const int next_mask = mask | bit;
                    const double total = previous + cost;
                    if (total < dp[static_cast<std::size_t>(slot + 1)]
                                     [static_cast<std::size_t>(next_mask)]) {
                        dp[static_cast<std::size_t>(slot + 1)]
                          [static_cast<std::size_t>(next_mask)] = total;
                        choice[static_cast<std::size_t>(slot + 1)]
                              [static_cast<std::size_t>(next_mask)] = candidate;
                    }
                }
            }
        }

        int mask = 0;
        double best = kInf;
        for (int candidate_mask = 0; candidate_mask < masks; ++candidate_mask) {
            if (dp.back()[static_cast<std::size_t>(candidate_mask)] < best) {
                best = dp.back()[static_cast<std::size_t>(candidate_mask)];
                mask = candidate_mask;
            }
        }

        std::array<int, 5> assigned{{-1, -1, -1, -1, -1}};
        for (int slot = 5; slot > 0; --slot) {
            const int candidate =
                choice[static_cast<std::size_t>(slot)][static_cast<std::size_t>(mask)];
            if (candidate < 0) continue;
            assigned[static_cast<std::size_t>(slot - 1)] = candidate;
            mask &= ~(1 << candidate);
        }
        for (int slot = 0; slot < 5; ++slot) {
            const int candidate = assigned[static_cast<std::size_t>(slot)];
            if (candidate < 0) continue;
            const int physical_slot = (frame.anchor_slot + slot) % 5;
            const CandidateRef& matched = candidates[static_cast<std::size_t>(candidate)];
            snapshot_.assigned_class[static_cast<std::size_t>(physical_slot)] =
                matched.class_id;
            snapshot_.slot_residual[static_cast<std::size_t>(physical_slot)] =
                cv::norm(matched.center - frame.slot_centers[static_cast<std::size_t>(slot)]) /
                spacing;
            ++snapshot_.assigned_count;
        }
    }

    double RuneSlotFactorGraph::emissionCost(int target, int mask) const
    {
        double cost = 0.0;
        const int target_class = snapshot_.assigned_class[static_cast<std::size_t>(target)];
        if (target_class < 0) {
            cost -= config_.target_miss_cost;
        } else if (target_class == 0) {
            cost += config_.inactive_class_reward;
        } else {
            cost -= config_.active_class_cost;
        }
        if (mask & (1 << target)) cost -= config_.activated_target_cost;

        for (int slot = 0; slot < 5; ++slot) {
            const bool activated = (mask & (1 << slot)) != 0;
            const bool hint = snapshot_.activated[static_cast<std::size_t>(slot)];
            cost += activated == hint ? config_.hint_match_reward
                                      : -config_.hint_mismatch_cost;
            const int class_id = snapshot_.assigned_class[static_cast<std::size_t>(slot)];
            if (class_id > 0) {
                cost += activated ? config_.active_mask_reward
                                  : -config_.inactive_mask_cost;
            } else if (class_id == 0 && activated) {
                cost -= config_.inactive_mask_cost;
            }
        }
        return cost;
    }

    void RuneSlotFactorGraph::initialize(const Frame& frame)
    {
        (void)frame;
        for (int mask = 0; mask < 32; ++mask) {
            for (int target = 0; target < 5; ++target) {
                const int state = encodeState(target, mask);
                previous_[static_cast<std::size_t>(state)] =
                    emissionCost(target, mask);
            }
        }
        initialized_ = true;
    }

    void RuneSlotFactorGraph::advance(const Frame& frame)
    {
        (void)frame;
        current_.fill(-kInf);
        for (int next_mask = 0; next_mask < 32; ++next_mask) {
            for (int next_target = 0; next_target < 5; ++next_target) {
                const int next_state = encodeState(next_target, next_mask);
                const double emission = emissionCost(next_target, next_mask);
                double best = -kInf;
                for (int prev_mask = 0; prev_mask < 32; ++prev_mask) {
                    const int added = next_mask & ~prev_mask;
                    const int removed = prev_mask & ~next_mask;
                    const int added_count =
                        __builtin_popcount(static_cast<unsigned int>(added));
                    const int removed_count =
                        __builtin_popcount(static_cast<unsigned int>(removed));
                    for (int prev_target = 0; prev_target < 5; ++prev_target) {
                        const int prev_state = encodeState(prev_target, prev_mask);
                        const double prior = previous_[static_cast<std::size_t>(prev_state)];
                        if (prior <= -kInf * 0.5) continue;
                        const double transition =
                            -(prev_target == next_target ? 0.0 : config_.target_switch_cost) -
                            config_.activation_set_cost * added_count -
                            config_.activation_remove_cost * removed_count;
                        best = std::max(best, prior + transition);
                    }
                }
                current_[static_cast<std::size_t>(next_state)] = best + emission;
            }
        }
        previous_ = current_;
    }

    void RuneSlotFactorGraph::update(const Frame& frame, double now)
    {
        now_ = now;
        snapshot_.updated = false;
        for (int slot = 0; slot < 5; ++slot) {
            snapshot_.activated[static_cast<std::size_t>(slot)] =
                frame.activated_hint[static_cast<std::size_t>(slot)];
        }
        if (!round_active_ || !frame.round_active) {
            if (round_active_ && !frame.round_active) reset();
            round_active_ = frame.round_active;
        }
        if (!frame.round_active) return;
        if (!frame.solved || frame.anchor_slot < 0 || frame.anchor_slot >= 5) {
            return;
        }

        assignCandidates(frame);
        if (!initialized_) {
            initialize(frame);
        } else {
            advance(frame);
        }

        int best_state = 0;
        int second_state = -1;
        for (int state = 1; state < 160; ++state) {
            if (previous_[static_cast<std::size_t>(state)] >
                previous_[static_cast<std::size_t>(best_state)]) {
                second_state = best_state;
                best_state = state;
            } else if (second_state < 0 ||
                       previous_[static_cast<std::size_t>(state)] >
                           previous_[static_cast<std::size_t>(second_state)]) {
                second_state = state;
            }
        }

        snapshot_.valid = true;
        snapshot_.updated = true;
        snapshot_.target_slot = targetOf(best_state);
        const int best_mask = maskOf(best_state);
        for (int slot = 0; slot < 5; ++slot) {
            snapshot_.activated[static_cast<std::size_t>(slot)] =
                (best_mask & (1 << slot)) != 0;
        }
        snapshot_.target_observed =
            snapshot_.assigned_class[static_cast<std::size_t>(snapshot_.target_slot)] >= 0;
        snapshot_.best_score = previous_[static_cast<std::size_t>(best_state)];
        snapshot_.margin =
            second_state >= 0
                ? snapshot_.best_score - previous_[static_cast<std::size_t>(second_state)]
                : 0.0;
    }
} // namespace auto_aim::energy
