#include "auto_buff/rune_slot_lattice.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>

#include <opencv2/core.hpp>

namespace auto_aim::energy
{
    double RuneSlotLattice::slotStep() const
    {
        return 2.0 * CV_PI / 5.0;
    }

    void RuneSlotLattice::reset()
    {
        snapshot_ = Snapshot{};
        anchored_ = false;
        last_evidence_time_.fill(-1e9);
        strong_.fill(false);
        now_ = 0.0;
        reference_phase_ = 0.0;
        have_seen_ = false;
        decay_tick_ = 0;
        round_active_ = true;
        votes_.fill(0);
    }

    void RuneSlotLattice::rotateBooking(int shift)
    {
        const int step = ((shift % 5) + 5) % 5;
        if (step == 0) return;
        std::array<int, 5> votes{};
        std::array<double, 5> evidence{};
        std::array<bool, 5> strong{};
        for (int i = 0; i < 5; ++i) {
            const int target = (i + step) % 5;
            votes[static_cast<std::size_t>(target)] = votes_[static_cast<std::size_t>(i)];
            evidence[static_cast<std::size_t>(target)] = last_evidence_time_[static_cast<std::size_t>(i)];
            strong[static_cast<std::size_t>(target)] = strong_[static_cast<std::size_t>(i)];
        }
        votes_ = votes;
        last_evidence_time_ = evidence;
        strong_ = strong;
    }

    void RuneSlotLattice::clearActivation()
    {
        votes_.fill(0);
        decay_tick_ = 0;
        last_evidence_time_.fill(-1e9);
        strong_.fill(false);
        snapshot_.activated.fill(false);
        snapshot_.votes = votes_;
    }

    void RuneSlotLattice::vote(std::array<int, 5>& votes, int slot, int amount) const
    {
        if (slot < 0 || slot >= 5) return;
        const std::size_t index = static_cast<std::size_t>(slot);
        const int capped = std::min(votes[index] + amount, 3 * config_.activate_votes);
        votes[index] = std::max(capped, 0);
    }

    void RuneSlotLattice::refreshActivated() const
    {
        for (std::size_t slot = 0; slot < votes_.size(); ++slot) {
            const bool fresh = (now_ - last_evidence_time_[slot]) <= config_.evidence_max_age_s;
            // 票数够 + 证据新鲜；或者"自己打中的那片"（整轮保留，真值语义如此）。
            snapshot_.activated[slot] =
                (votes_[slot] >= config_.activate_votes && fresh) || strong_[slot];
        }
        snapshot_.votes = votes_;
    }

    bool RuneSlotLattice::candidateSlotOffset(const Frame& frame, const cv::Point2f& candidate,
                                              double target_phase, double slot_step,
                                              int& offset) const
    {
        if (frame.slot_centers_valid && !config_.use_measured_angles) {
            // 最近邻：5 个槽位在图像里的预测位置由解算层投影给出，天然含透视压缩。
            int nearest = -1;
            double nearest_distance = 0.0;
            double spacing = 0.0;
            int counted = 0;
            for (std::size_t i = 0; i < frame.slot_centers.size(); ++i) {
                const double distance = cv::norm(frame.slot_centers[i] - candidate);
                if (nearest < 0 || distance < nearest_distance) {
                    nearest = static_cast<int>(i);
                    nearest_distance = distance;
                }
                const std::size_t next = (i + 1) % frame.slot_centers.size();
                spacing += cv::norm(frame.slot_centers[next] - frame.slot_centers[i]);
                ++counted;
            }
            spacing = counted > 0 ? spacing / counted : 0.0;
            const double gate = config_.assign_max_spacing_ratio * spacing;
            // 5 个投影中心重合（退化，比如缺少相机标定）时不能用作依据，退回角度法。
            if (spacing > 1.0) {
                if (nearest < 0 || nearest_distance > gate) return false;
                offset = nearest;
                return true;
            }
        }

        // 退化路径（没有投影槽位可用时）：按候选与靶心的图像夹角折算。
        const cv::Point2f hub = frame.hub;
        const double phase =
            std::atan2(candidate.y - hub.y, candidate.x - hub.x);
        double delta = phase - target_phase;
        while (delta > CV_PI) delta -= 2.0 * CV_PI;
        while (delta < -CV_PI) delta += 2.0 * CV_PI;
        offset = static_cast<int>(std::lround(delta / slot_step));
        return true;
    }

    bool RuneSlotLattice::slotOffsetForCandidate(const cv::Point2f& candidate_px,
                                                 int& offset) const
    {
        Frame probe;
        probe.hub = snapshot_.hub;
        probe.target_center = snapshot_.target_center;
        probe.solved = true;
        probe.slot_centers = snapshot_.slot_centers;
        probe.slot_centers_valid = snapshot_.slot_centers_valid;
        const double target_phase = std::atan2(probe.target_center.y - probe.hub.y,
                                               probe.target_center.x - probe.hub.x);
        int raw = 0;
        if (!candidateSlotOffset(probe, candidate_px, target_phase, slotStep(), raw)) {
            return false;
        }
        offset = ((raw % 5) + 5) % 5;
        return true;
    }

    void RuneSlotLattice::update(const Frame& frame, double now)
    {
        const double slot_step = slotStep();
        now_ = now;
        // 诊断：把本帧投影出的槽位中心带进快照（离线核对归属用）。
        snapshot_.slot_centers = frame.slot_centers;
        snapshot_.slot_centers_valid = frame.slot_centers_valid;
        snapshot_.hub = frame.hub;
        snapshot_.target_center = frame.target_center;

        // 回合结束（没有点亮扇叶了）→ 机关复位。真值里每一轮开始时掩码都是 00000，
        // 这里跟着清空，否则上一轮的记账会一直污染下一轮（实测过记的主因之一）。
        if (round_active_ && !frame.round_active) {
            clearActivation();
            if (debug_) {
                std::cerr << "[rune lattice] round ended: cleared activation bookkeeping"
                          << std::endl;
            }
        }
        round_active_ = frame.round_active;

        // 证据衰减：每 decay_every_frames 帧没有新的证据就各槽位减一票。
        // 单帧的 ±1 槽折算噪声会在衰减里自己掉下去，只有持续一致的证据才能置位。
        if (++decay_tick_ >= std::max(1, config_.decay_every_frames)) {
            decay_tick_ = 0;
            for (int& value : votes_) value = std::max(0, value - 1);
        }

        snapshot_.switch_confirmed = false;
        retired_centers_.clear();
        std::array<bool, 5> observed_active{};   // 本帧观测到的"已激活片"集合
        if (frame.solved) {
            // 换片 = "刚才还在瞄的那片不再是被点亮的那片" ⇒ 它已经激活了。
            // 被换下的片在**本帧**图像里的位置：把本帧靶心绕圆心反向转 step×72°。
            // 只认 1~2 片的跳变（更大的跳变按误判处理）。
            const auto note_retired = [&](int step) {
                if (!config_.book_retired_on_switch || step == 0) return;
                const int count = std::min(std::abs(step), 2);
                const double sign = step > 0 ? 1.0 : -1.0;
                const cv::Point2f vector = frame.target_center - frame.hub;
                for (int k = 1; k <= count; ++k) {
                    const double angle = -sign * static_cast<double>(k) * slot_step;
                    const double c = std::cos(angle);
                    const double s = std::sin(angle);
                    retired_centers_.emplace_back(
                        frame.hub.x + static_cast<float>(vector.x * c - vector.y * s),
                        frame.hub.y + static_cast<float>(vector.x * s + vector.y * c));
                }
            };
            // ---- 换片判据（深大）：观测绝对相位 vs 运动模型预测 -----------------
            // 预测：把上一帧的状态相位按运动模型的角速度外推到本帧；
            // 观测：本帧解算出的绝对相位（平面内角）。
            // 偏差 > 阈值且连续 N 帧一致（同一个 72° 整数倍）⇒ 确认换片。
            if (config_.switch_by_phase && anchored_ && have_last_phase_time_ &&
                frame.observed_phase_valid) {
                const double dt = std::max(0.0, now - last_phase_time_);
                const double predicted = frame.roll + frame.phase_rate * dt;
                double delta = frame.observed_phase - predicted;
                while (delta > CV_PI) delta -= 2.0 * CV_PI;
                while (delta < -CV_PI) delta += 2.0 * CV_PI;
                const int step = static_cast<int>(std::lround(delta / slot_step));
                if (std::abs(delta) > config_.phase_switch_threshold_rad && step != 0) {
                    if (step == pending_step_) {
                        ++pending_offset_frames_;
                    } else {
                        pending_step_ = step;
                        pending_offset_frames_ = 1;
                        offset_applied_ = false;
                    }
                    if (!offset_applied_ &&
                        pending_offset_frames_ >= config_.switch_confirm_frames) {
                        note_retired(step);
                        blade_index_ = ((blade_index_ + step) % 5 + 5) % 5;
                        if (config_.rotate_booking_on_switch) rotateBooking(step);
                        snapshot_.switch_confirmed = true;
                        offset_applied_ = true;
                    }
                } else {
                    pending_step_ = 0;
                    pending_offset_frames_ = 0;
                    offset_applied_ = false;
                }
            }
            last_phase_time_ = now;
            have_last_phase_time_ = true;
            // 锚点只用 class 0 的片：class0 才代表"未激活、该打的那块装甲板"。
            if (!anchored_ && frame.target_class == 0) {
                reference_phase_ = frame.roll;
                blade_index_ = 0;   // 锚点：第一次锁定的 class-0 片记为 0 号
                anchored_ = true;
                clearActivation();
                if (debug_) std::cerr << "[rune lattice] anchored on a class-0 blade" << std::endl;
            }
            if (anchored_) {
                // 锁定片相对跟踪片的几何偏移（带符号 -2~2）；估计器折算/记账都要用。
                int offset_signed = 0;
                const double target_phase =
                    std::atan2(frame.target_center.y - frame.hub.y,
                               frame.target_center.x - frame.hub.x);
                const bool offset_ok =
                    candidateSlotOffset(frame, frame.target_center, target_phase, slot_step,
                                        offset_signed);
                snapshot_.observed_offset_signed = offset_signed;
                snapshot_.observed_offset = ((offset_signed % 5) + 5) % 5;
                snapshot_.observed_offset_valid = offset_ok;

                // ---- 换片判据：二选一 ------------------------------------------
                // (a) 深大 RP-26Rune 式（switch_by_geometry）：用运动模型外推的相位
                //     比观测 —— 我们的等价量是"几何偏移应为 0"，连续 N 帧非零即确认；
                // (b) 状态相位跳变（默认）：换片时状态自己会跳 ±72° 的整数倍。
                if (config_.switch_by_geometry) {
                    if (offset_ok && offset_signed != 0) {
                        if (offset_signed != pending_offset_) {
                            pending_offset_ = offset_signed;
                            pending_offset_frames_ = 1;
                            offset_applied_ = false;
                        } else {
                            ++pending_offset_frames_;
                        }
                        if (!offset_applied_ &&
                            pending_offset_frames_ >= config_.switch_confirm_frames) {
                            note_retired(offset_signed);
                            blade_index_ = ((blade_index_ + offset_signed) % 5 + 5) % 5;
                            if (config_.rotate_booking_on_switch) rotateBooking(offset_signed);
                            snapshot_.switch_confirmed = true;
                            offset_applied_ = true;
                        }
                    } else {
                        pending_offset_ = 0;
                        pending_offset_frames_ = 0;
                        offset_applied_ = false;
                    }
                } else if (!config_.switch_by_phase && have_last_roll_) {
                    const double delta = frame.roll - last_roll_;
                    if (std::abs(delta) > 30.0 * CV_PI / 180.0) {
                        const int step = static_cast<int>(std::lround(delta / slot_step));
                        if (step != 0) {
                            note_retired(step);
                            blade_index_ = ((blade_index_ + step) % 5 + 5) % 5;
                            if (config_.rotate_booking_on_switch) rotateBooking(step);
                            snapshot_.switch_confirmed = true;
                        }
                    }
                }
                last_roll_ = frame.roll;
                have_last_roll_ = true;
                snapshot_.slot_id = blade_index_;
                last_seen_time_ = now;
                have_seen_ = true;
            }

            // 已激活记账：按"与当前靶心的相位差"把过滤前的候选折算到槽位，投票。
            // 同时（用同一套几何）算出"本帧锁定的那片"相对跟踪槽位偏了几个槽位：
            // 观测到已激活片时，估计器要靠它把观测折算回跟踪槽位。
            if (frame.round_active && anchored_ && snapshot_.slot_id >= 0) {
                const cv::Point2f hub = frame.hub;
                const double target_phase = std::atan2(frame.target_center.y - hub.y,
                                                       frame.target_center.x - hub.x);
                for (const auto& candidate : frame.candidates) {
                    if (candidate.first == 0) continue;   // 未激活片不投票
                    int offset = 0;
                    if (!candidateSlotOffset(frame, candidate.second, target_phase, slot_step,
                                             offset)) {
                        continue;   // 离所有槽位都太远：不算证据
                    }
                    const int slot = static_cast<int>(
                        (((snapshot_.slot_id + offset) % 5) + 5) % 5);   // 物理片编号
                    vote(votes_, slot, config_.vote_gain);
                    last_evidence_time_[static_cast<std::size_t>(slot)] = now;
                    observed_active[static_cast<std::size_t>(slot)] = true;
                }
                // 换片推出来的"被换下的那片"：已经确认过的物理事件，一次给满票
                // （不依赖网络类别，也不需要命中反馈）。
                for (const cv::Point2f& retired : retired_centers_) {
                    int offset = 0;
                    if (!candidateSlotOffset(frame, retired, target_phase, slot_step, offset)) {
                        continue;
                    }
                    const int slot =
                        static_cast<int>((((snapshot_.slot_id + offset) % 5) + 5) % 5);
                    vote(votes_, slot, config_.activate_votes);
                    last_evidence_time_[static_cast<std::size_t>(slot)] = now;
                    observed_active[static_cast<std::size_t>(slot)] = true;
                    if (debug_) {
                        std::cerr << "[rune lattice] retired blade booked as activated: slot "
                                  << slot << " (offset " << offset << ")" << std::endl;
                    }
                }
            }
        } else if (have_seen_ && now - last_seen_time_ > config_.reset_silence_s) {
            // 长时间完全没有亮片 → 机关复位：清空"已激活"记账，编号保留。
            clearActivation();
            have_seen_ = false;
            if (debug_) {
                std::cerr << "[rune lattice] long silence: cleared activation bookkeeping"
                          << std::endl;
            }
        }

        // 命中反馈是强证据（我们确知弹丸打上去了）：给当时开火的槽位加一大票。
        if (frame.scored_hit && frame.fired_slot >= 0) {
            const std::size_t slot = static_cast<std::size_t>(frame.fired_slot);
            vote(votes_, slot, 3 * config_.activate_votes / 2);
            last_evidence_time_[slot] = now;
            strong_[slot] = true;   // 我们确知打中了这片：整轮保留
        }

        // ---- 命中锚重锚 ----------------------------------------------------
        // 我们打中的片（strong_）是真值锚；若"本帧观测到的已激活片"在某个非零
        // 循环移位下更吻合 strong_，说明参考系漂了，连续确认后把它转回来。
        if (anchored_ && snapshot_.slot_id >= 0) {
            int observed_count = 0;
            int strong_count = 0;
            for (int i = 0; i < 5; ++i) {
                if (observed_active[static_cast<std::size_t>(i)]) ++observed_count;
                if (strong_[static_cast<std::size_t>(i)]) ++strong_count;
            }
            if (observed_count > 0 && strong_count > 0) {
                int best_shift = 0;
                int best_score = -1;
                int second_score = -1;
                for (int shift = 0; shift < 5; ++shift) {
                    int score = 0;
                    for (int i = 0; i < 5; ++i) {
                        if (observed_active[static_cast<std::size_t>(i)] &&
                            strong_[static_cast<std::size_t>((i + shift) % 5)]) {
                            ++score;
                        }
                    }
                    if (score > best_score) {
                        second_score = best_score;
                        best_score = score;
                        best_shift = shift;
                    } else if (score > second_score) {
                        second_score = score;
                    }
                }
                const bool unambiguous = best_score > second_score;
                if (unambiguous && best_shift != 0) {
                    if (best_shift == reanchor_shift_) {
                        ++reanchor_frames_;
                    } else {
                        reanchor_shift_ = best_shift;
                        reanchor_frames_ = 1;
                    }
                    if (reanchor_frames_ >= config_.reanchor_confirm_frames) {
                        blade_index_ = ((blade_index_ + best_shift) % 5 + 5) % 5;
                        snapshot_.slot_id = blade_index_;
                        rotateBooking(best_shift);
                        reanchor_frames_ = 0;
                    }
                } else {
                    reanchor_shift_ = 0;
                    reanchor_frames_ = 0;
                }
            }
        }

        refreshActivated();
        if (std::all_of(snapshot_.activated.begin(), snapshot_.activated.end(),
                        [](bool value) { return value; })) {
            clearActivation();   // 5 片打满 → 机关复位，清空记账
        }
    }
} // namespace auto_aim::energy
