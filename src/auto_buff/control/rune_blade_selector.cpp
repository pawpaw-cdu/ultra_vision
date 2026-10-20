#include "auto_buff/control/rune_blade_selector.hpp"

#include <algorithm>
#include <cmath>

namespace auto_aim::energy
{
    void RuneBladeSelector::reset()
    {
        searching_ = false;
        hit_slot_ = -1;
        hit_center_ = {0.0f, 0.0f};
        pending_slot_ = -1;
        pending_frames_ = 0;
        selected_ = Selection{};
        snapshot_ = Snapshot{};
    }

    int RuneBladeSelector::uniqueUnactivatedSlot(const Frame& frame) const
    {
        int slot = -1;
        int count = 0;
        for (int i = 0; i < 5; ++i) {
            if (frame.activated[static_cast<std::size_t>(i)]) continue;
            if (i == hit_slot_) continue;
            slot = i;
            ++count;
        }
        return count == 1 ? slot : -1;
    }

    RuneBladeSelector::Selection RuneBladeSelector::chooseCandidate(
        const Frame& frame) const
    {
        Selection best;
        const double clearance =
            std::max(1.0f, frame.orbit_px) * config_.hit_clearance_orbit_ratio;
        for (const Candidate& candidate : frame.candidates) {
            if (candidate.slot < 0 || candidate.slot >= 5) continue;
            if (frame.activated[static_cast<std::size_t>(candidate.slot)]) continue;
            if (candidate.slot == hit_slot_) continue;
            if (frame.orbit_px > 1.0f &&
                cv::norm(candidate.center - frame.hit_center) < clearance) {
                continue;
            }

            double score = candidate.class_id == 0 ? config_.class0_bonus : 0.0;
            if (candidate.lit) score += config_.lit_bonus;
            // Activated-class evidence is a weaker candidate, not an outright
            if (candidate.class_id > 0) score -= 0.15;
            if (score > best.score) {
                best.valid = true;
                best.slot = candidate.slot;
                best.class_id = candidate.class_id;
                best.center = candidate.center;
                best.score = score;
            }
        }
        return best;
    }

    void RuneBladeSelector::update(const Frame& frame, double now)
    {
        (void)now;
        if (frame.scored_hit && frame.hit_slot >= 0) {
            searching_ = true;
            hit_slot_ = frame.hit_slot;
            hit_center_ = frame.hit_center;
            pending_slot_ = -1;
            pending_frames_ = 0;
            selected_ = Selection{};
        }
        if (frame.hold) searching_ = true;
        if (selected_.valid && frame.activated[static_cast<std::size_t>(selected_.slot)]) {
            selected_ = Selection{};
            searching_ = true;
        }

        if (searching_ && frame.solved && frame.observation_fresh) {
            const Selection candidate = chooseCandidate(frame);
            if (candidate.valid) {
                if (candidate.slot == pending_slot_) {
                    ++pending_frames_;
                } else {
                    pending_slot_ = candidate.slot;
                    pending_frames_ = 1;
                }
                if (pending_frames_ >= std::max(1, config_.confirm_frames)) {
                    selected_ = candidate;
                    searching_ = false;
                    pending_slot_ = -1;
                    pending_frames_ = 0;
                }
            } else {
                pending_slot_ = -1;
                pending_frames_ = 0;
                const int unique = uniqueUnactivatedSlot(frame);
                if (unique >= 0) {
                    selected_.valid = true;
                    selected_.slot = unique;
                    selected_.class_id = -1;
                    selected_.center = {0.0f, 0.0f};
                    searching_ = false;
                }
            }
        }

        snapshot_ = Snapshot{};
        snapshot_.searching = searching_;
        snapshot_.retargeting = searching_ || selected_.valid;
        if (selected_.valid &&
            !frame.activated[static_cast<std::size_t>(selected_.slot)]) {
            snapshot_.valid = true;
            snapshot_.confirmed = true;
            snapshot_.planned_slot = selected_.slot;
            snapshot_.target_class = selected_.class_id;
            snapshot_.target_center = selected_.center;
        } else if (!searching_ && frame.observed_slot >= 0 &&
                   !frame.activated[static_cast<std::size_t>(frame.observed_slot)]) {
            snapshot_.valid = true;
            snapshot_.confirmed = true;
            snapshot_.planned_slot = frame.observed_slot;
        }
        if (snapshot_.valid && frame.observed_slot >= 0) {
            snapshot_.retarget_offset =
                ((snapshot_.planned_slot - frame.observed_slot) % 5 + 5) % 5;
        }
    }
} // namespace auto_aim::energy
