// Self-checking tests for the two pure state classes extracted out of the frame
// loop: RuneRoundGuard (round / hold / fire window) and RuneSlotLattice
// (1..5 slot booking). No image, no simulator, no OpenVINO — they are fed
// synthetic per-frame inputs, which is exactly why they were split out.

#include <array>
#include <cmath>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

#include "auto_buff/rune_round_guard.hpp"
#include "auto_buff/rune_slot_lattice.hpp"
#include "auto_buff/rune_aim_fallback.hpp"
#include "auto_buff/rune_orbit_fit.hpp"

namespace
{
    using namespace auto_aim::energy;

    int failures = 0;

    void check(bool condition, const std::string& what)
    {
        if (!condition) {
            std::printf("FAIL: %s\n", what.c_str());
            ++failures;
        }
    }

    constexpr double kPi = 3.14159265358979323846;
    constexpr double kSlot = 2.0 * kPi / 5.0;

    RuneRoundGuard::Frame solvedFrame(cv::Point2f blade, double phase, int cls)
    {
        RuneRoundGuard::Frame frame;
        frame.solved = true;
        frame.blade_center = blade;
        frame.aim_phase = phase;
        frame.blade_class = cls;
        return frame;
    }

    // ---- 回合记账 / 开火窗口 -------------------------------------------------
    void testRoundWindow()
    {
        RuneRoundGuard::Config config;
        config.fire_start_delay_s = 0.15;
        config.round_window_s = 2.5;
        config.round_end_frames = 5;
        RuneRoundGuard guard(config);

        const cv::Point2f blade(300.0f, 240.0f);
        guard.update(solvedFrame(blade, 0.1, 0), 10.0);
        auto snapshot = guard.snapshot();
        check(snapshot.blade_lit, "round: starts when a lit blade appears");
        check(std::abs(snapshot.round_age_s) < 1e-9, "round: age starts at 0");
        check(!guard.windowOpen(0.3), "round: closed during the start delay");

        guard.update(solvedFrame(blade, 0.2, 0), 10.2);
        check(guard.windowOpen(0.3), "round: open after the start delay");
        // 年龄 0.2 s：飞行 2.3 s 时 2.5 s 刚好落在窗口末端内；再远就超窗口。
        check(guard.windowOpen(2.2), "round: open while the round ends inside the window");
        check(!guard.windowOpen(2.4), "round: closed when the flight time overruns the window");

        // 连续 round_end_frames 帧没有解算结果 → 本轮结束
        RuneRoundGuard::Frame nothing;
        for (int i = 0; i < config.round_end_frames; ++i) {
            guard.update(nothing, 10.4 + 0.05 * i);
        }
        check(guard.snapshot().blade_lit, "round: survives round_end_frames-1 blank frames");
        guard.update(nothing, 11.0);
        snapshot = guard.snapshot();
        check(!snapshot.blade_lit, "round: ends after round_end_frames blank frames");
        check(snapshot.round_age_s > 1e8, "round: age is 'never' once the round ended");
        check(!guard.windowOpen(0.0), "round: no window after the round ended");
    }

    void testBladeJumpRestartsRound()
    {
        RuneRoundGuard::Config config;
        config.blade_slot_jump_px = 25.0;
        RuneRoundGuard guard(config);

        guard.update(solvedFrame(cv::Point2f(300.0f, 240.0f), 0.0, 0), 5.0);
        guard.update(solvedFrame(cv::Point2f(305.0f, 241.0f), 0.1, 0), 5.5);
        check(std::abs(guard.snapshot().round_age_s - 0.5) < 1e-9,
              "round: small blade motion does not restart the round");
        // 72° 换叶：图像里整跳一大截 → 重开一轮
        guard.update(solvedFrame(cv::Point2f(380.0f, 260.0f), 0.2, 0), 6.0);
        check(std::abs(guard.snapshot().round_age_s) < 1e-9,
              "round: a blade jump larger than blade_slot_jump_px restarts the round");
    }

    void testClassHitNeedsInactiveFirst()
    {
        RuneRoundGuard guard;
        const cv::Point2f blade(300.0f, 240.0f);

        // 同一片靶：连续两帧 class 0，再看到 class 1 → 判命中
        guard.update(solvedFrame(blade, 0.30, 0), 1.0);
        guard.update(solvedFrame(blade, 0.31, 0), 1.1);
        check(!guard.snapshot().class_hit, "class hit: needs an activated reading to fire");
        guard.update(solvedFrame(blade, 0.32, 1), 1.2);
        check(guard.snapshot().class_hit, "class hit: class0 x2 then class1 counts as a hit");

        // 新的一片靶（相位差大）直接报 class 1：不能算命中（否则每片新靶都误判）
        RuneRoundGuard fresh;
        fresh.update(solvedFrame(blade, 0.30, 0), 1.0);
        fresh.update(solvedFrame(blade, 0.30 + 0.6 * kSlot, 1), 1.1);
        check(!fresh.snapshot().class_hit,
              "class hit: an activated reading on a different blade is not a hit");
    }

    void testPostHitHold()
    {
        RuneRoundGuard guard;
        const cv::Point2f blade(300.0f, 240.0f);
        RuneRoundGuard::Frame hit = solvedFrame(blade, 0.5, 1);
        hit.scored_hit = true;
        guard.update(hit, 2.0);
        check(guard.snapshot().hold, "hold: engages on a scored hit");

        // 靶一直在转，0.10 rad 的相位漂移不该解禁
        guard.update(solvedFrame(blade, 0.60, 1), 2.1);
        check(guard.snapshot().hold, "hold: small phase drift keeps the gate closed");
        // 相位漂移超过 0.30 rad 也**不**解禁：符在转，漂移自己就会走到阈值，
        // 那样会在 0.3 s 后无条件放行、把刚打过的片再打一遍（实测 66% 的命中如此）。
        guard.update(solvedFrame(blade, 0.95, 1), 2.2);
        check(guard.snapshot().hold, "hold: a phase drift alone no longer releases");

        // 确认换片（估计器观测接管）→ 解禁
        RuneRoundGuard::Frame switched = solvedFrame(blade, 1.05, 0);
        switched.blade_switched = true;
        guard.update(switched, 2.3);
        check(!guard.snapshot().hold, "hold: releases on a confirmed blade switch");

        // 兜底：没有换片信号时，超过 post_hit_hold_max_s 也要放行（防闸门卡死）
        RuneRoundGuard::Config config;
        config.post_hit_hold_max_s = 1.2;
        RuneRoundGuard timeout_guard(config);
        RuneRoundGuard::Frame held = solvedFrame(blade, 0.5, 1);
        held.scored_hit = true;
        timeout_guard.update(held, 10.0);
        timeout_guard.update(solvedFrame(blade, 0.6, 1), 10.5);
        check(timeout_guard.snapshot().hold, "hold: still held before the timeout");
        timeout_guard.update(solvedFrame(blade, 0.7, 1), 11.3);
        check(!timeout_guard.snapshot().hold, "hold: forced release after the max hold time");

        // 新靶被稳定读成 class 0（同一片连续 2 帧）→ 立刻放行，不必等估计器确认接管
        RuneRoundGuard::Config quick;
        quick.post_hit_hold_max_s = 5.0;
        quick.require_active_witness = true;   // 这一支专门测"必须有已激活片作证"
        RuneRoundGuard fast(quick);
        RuneRoundGuard::Frame hit2 = solvedFrame(blade, 0.5, 1);
        hit2.scored_hit = true;
        fast.update(hit2, 20.0);
        fast.update(solvedFrame(blade, 0.6, 0), 20.1);
        check(fast.snapshot().hold,
              "hold: one class-0 frame after the hit is not enough (avoid re-shooting "
              "the blade we just activated)");
        fast.update(solvedFrame(blade, 0.7, 0), 20.2);   // 没有"已激活片作证"
        check(fast.snapshot().hold,
              "hold: two class-0 frames without an activated-blade witness stay closed "
              "(that reading is probably the blade we just hit)");
        RuneRoundGuard::Frame witnessed = solvedFrame(blade, 0.8, 0);
        witnessed.active_witness = 1;   // 同帧里还看到一片 class 1/2
        fast.update(witnessed, 20.3);
        check(!fast.snapshot().hold,
              "hold: releases once the same blade reads class 0 twice with a witness");

        // 关掉"作证"要求（A/B 用）→ 回到原来的宽松规则
        RuneRoundGuard::Config loose;
        loose.post_hit_hold_max_s = 5.0;
        loose.require_active_witness = false;
        RuneRoundGuard loose_guard(loose);
        RuneRoundGuard::Frame hit3 = solvedFrame(blade, 0.5, 1);
        hit3.scored_hit = true;
        loose_guard.update(hit3, 30.0);
        loose_guard.update(solvedFrame(blade, 0.6, 0), 30.1);
        loose_guard.update(solvedFrame(blade, 0.7, 0), 30.2);
        check(!loose_guard.snapshot().hold,
              "hold: without the witness requirement two class-0 frames are enough");
    }

    void testScoredHitRestartsRound()
    {
        RuneRoundGuard guard;
        guard.update(solvedFrame(cv::Point2f(300.0f, 240.0f), 0.0, 0), 3.0);
        guard.noteScoredHit(7.0);      // 仿真 build_new_round 立刻换靶
        guard.update(solvedFrame(cv::Point2f(340.0f, 240.0f), 0.1, 0), 7.0);
        check(std::abs(guard.snapshot().round_age_s) < 1e-9,
              "round: a scored hit starts a new round immediately");
    }

    // ---- 槽位格点 -----------------------------------------------------------
    RuneSlotLattice::Frame latticeFrame(double roll, int cls, cv::Point2f target_center,
                                        std::vector<std::pair<int, cv::Point2f>> candidates)
    {
        RuneSlotLattice::Frame frame;
        frame.solved = true;
        frame.roll = roll;
        frame.target_class = cls;
        frame.hub = cv::Point2f(0.0f, 0.0f);
        frame.target_center = target_center;
        frame.candidates = std::move(candidates);
        return frame;
    }

    /// 带"投影槽位中心"的帧：换片里程计要靠几何偏移来推动，这正是流水线里的形态。
    RuneSlotLattice::Frame slotFrame(double roll, int cls, int observed_slot,
                                     bool blade_switched, double radius_px = 60.0,
                                     double observed_phase_offset = 0.0)
    {
        (void)blade_switched;
        RuneSlotLattice::Frame frame;
        frame.solved = true;
        frame.roll = roll;
        frame.target_class = cls;
        frame.hub = cv::Point2f(0.0f, 0.0f);
        for (int k = 0; k < 5; ++k) {
            const double angle = -kPi / 2.0 + k * kSlot;
            frame.slot_centers[static_cast<std::size_t>(k)] =
                cv::Point2f(static_cast<float>(radius_px * std::cos(angle)),
                            static_cast<float>(radius_px * std::sin(angle)));
        }
        frame.slot_centers_valid = true;
        frame.target_center = frame.slot_centers[static_cast<std::size_t>((observed_slot % 5 + 5) % 5)];
        frame.blade_switched = blade_switched;
        // 绝对相位（深大式换片判据的观测）：默认与状态一致（无换片）。
        frame.observed_phase_valid = true;
        frame.observed_phase = roll + observed_phase_offset;
        frame.phase_rate = 0.0;
        return frame;
    }

    void testLatticeAnchoring()
    {
        // 换片里程计（绝对 ID）现在是非默认的可选项，测它要显式打开。
        RuneSlotLattice::Config id_config;
        id_config.switch_by_phase = true;
        RuneSlotLattice lattice(id_config);
        // 没有 class 0 的观测不锚定（class 1/2 是"已经打过的片"）
        lattice.update(slotFrame(1.0, 1, 0, false), 1.0);
        check(lattice.slotId() == -1, "lattice: does not anchor on an activated blade");

        lattice.update(slotFrame(1.0, 0, 0, false), 1.1);
        check(lattice.slotId() == 0, "lattice: anchors on a class-0 blade (blade 0)");

        // 符在转：状态相位每帧只走几度 ⇒ 片号不动（旋转不是换片）
        double roll = 1.0;
        for (int i = 0; i < 10; ++i) {
            roll += 3.0 * kPi / 180.0;
            lattice.update(slotFrame(roll, 0, 0, false), 1.2 + 0.033 * i);
        }
        check(lattice.slotId() == 0, "lattice: continuous rotation does not advance the blade id");

        // 换片：**绝对相位**比运动模型预测偏 72°（深大判据），连续 2 帧确认
        roll += 0.05;
        lattice.update(slotFrame(roll, 0, 0, false, 60.0, 0.0), 2.00);
        lattice.update(slotFrame(roll + 0.05, 0, 0, false, 60.0, kSlot), 2.033);
        check(lattice.slotId() == 0, "lattice: one deviating frame is not enough");
        lattice.update(slotFrame(roll + 0.10, 0, 0, false, 60.0, kSlot), 2.066);
        check(lattice.slotId() == 1, "lattice: a confirmed +72 deg phase jump advances the id");

        // 观测回到预测（无偏差）后，下一次换片才能再次推进
        lattice.update(slotFrame(roll + 0.15, 0, 0, false, 60.0, 0.0), 2.1);
        check(lattice.slotId() == 1, "lattice: returning to the prediction keeps the id");
        lattice.update(slotFrame(roll + 0.20, 0, 0, false, 60.0, -2.0 * kSlot), 2.13);
        lattice.update(slotFrame(roll + 0.25, 0, 0, false, 60.0, -2.0 * kSlot), 2.16);
        check(lattice.slotId() == 4, "lattice: a -144 deg phase jump moves the id back two");
    }

    void testLatticeActivationBookkeeping()
    {
        // 投票阈值 4、每帧 +2 ⇒ 需要连续两帧指向同一槽位才置位（单帧噪声不置位）。
        RuneSlotLattice::Config config;
        config.activate_votes = 4;
        config.vote_gain = 2;
        RuneSlotLattice lattice(config);
        // 靶心在 12 点方向（图像 y 向下 ⇒ 圆心上方），已激活片在它 +72° 处
        const cv::Point2f target(0.0f, -100.0f);
        const cv::Point2f activated(100.0f * std::sin(kSlot), -100.0f * std::cos(kSlot));
        const std::vector<std::pair<int, cv::Point2f>> candidates{{0, target}, {1, activated}};
        lattice.update(latticeFrame(0.0, 0, target, candidates), 1.0);

        auto snapshot = lattice.snapshot();
        check(snapshot.slot_id == 0, "lattice: aiming at slot 0");
        check(!snapshot.activated[1],
              "lattice: a single frame of class-1 evidence is not enough (no instant flip)");
        check(snapshot.votes[1] == config.vote_gain,
              "lattice: the first frame casts a vote for slot 1");

        lattice.update(latticeFrame(0.0, 0, target, candidates), 1.05);
        snapshot = lattice.snapshot();
        check(snapshot.activated[1] && !snapshot.activated[0],
              "lattice: consistent evidence over two frames marks slot 1 activated");

        // 命中反馈是强证据：一票就够（没检到的那一槽也能补上）
        RuneSlotLattice::Frame hit = latticeFrame(0.0, 0, target, {});
        hit.scored_hit = true;
        hit.fired_slot = 3;
        lattice.update(hit, 1.1);
        check(lattice.activated()[3], "lattice: the fired slot is booked on a scored hit");
    }

    /// 换片 = "刚才还在瞄的那片不再被点亮" ⇒ 它已经激活。这是新模型（只给未激活
    /// 目标片）下**唯一**不依赖网络类别、也不依赖命中反馈的激活证据，所以要有单测
    /// 钉住两件事：(1) 被换下的片被记账；(2) 槽位号是**物理片号**，换片不改变它。
    void testLatticeRetiredSlotBooksActivation()
    {
        RuneSlotLattice::Config config;
        config.switch_by_phase = false;          // 用最简单的"roll 跳变"判换片
        config.rotate_booking_on_switch = false; // 槽位号是物理片号 ⇒ 不该再旋转
        RuneSlotLattice lattice(config);

        // 某一槽位在图像里的绝对位置（12 点方向为 0 号，顺时针 +72°/槽）。
        const auto slot_point = [](int slot, double radius) {
            const double angle = -kPi / 2.0 + slot * kSlot;
            return cv::Point2f(static_cast<float>(radius * std::cos(angle)),
                               static_cast<float>(radius * std::sin(angle)));
        };

        // 第 1 帧：锚定在点亮的 0 号片；另有一片已激活（class 1）在 2 号槽位。
        RuneSlotLattice::Frame frame = slotFrame(0.0, 0, 0, false, 60.0);
        frame.candidates = {{1, slot_point(2, 60.0)}};
        lattice.update(frame, 1.0);
        lattice.update(frame, 1.033);   // 投票阈值 4 / 每帧 +2 → 连续两帧才置位
        check(lattice.activated()[2], "retired-slot: class-1 evidence books slot 2");

        // 第 2 帧：打中了 —— 点亮的那片换到 1 号槽位（状态相位跳 +72°）。
        RuneSlotLattice::Frame switched = slotFrame(kSlot, 0, 1, false, 60.0);
        switched.blade_switched = true;
        switched.candidates = {{1, slot_point(2, 60.0)}};   // 已激活的那片还在
        lattice.update(switched, 1.066);

        // 被换下的 0 号片（物理片号不变）应当立刻记成"已激活"……
        check(lattice.activated()[0],
              "retired-slot: the blade we were aiming at is booked as activated on a "
              "confirmed switch (geometry only, no network class)");
        // ……而新点亮的那片（1 号）不能被记成已激活，否则开火闸门会把它一起挡掉。
        check(!lattice.activated()[1],
              "retired-slot: the newly lit blade is NOT booked as activated");
        // 槽位号是物理片号：换片前记在 2 号的那片，换片后仍在 2 号。
        check(lattice.activated()[2],
              "retired-slot: slot numbering is the physical blade index (invariant under "
              "a switch), so the earlier booking stays at slot 2");
    }

    void testLatticeRoundResetAndDecay()
    {
        RuneSlotLattice::Config config;
        config.activate_votes = 4;
        config.vote_gain = 2;
        config.decay_every_frames = 4;
        RuneSlotLattice lattice(config);

        const cv::Point2f target(0.0f, -100.0f);
        const cv::Point2f activated(100.0f * std::sin(kSlot), -100.0f * std::cos(kSlot));
        const std::vector<std::pair<int, cv::Point2f>> candidates{{1, activated}};
        lattice.update(latticeFrame(0.0, 0, target, candidates), 1.0);
        lattice.update(latticeFrame(0.0, 0, target, candidates), 1.05);
        check(lattice.activated()[1], "lattice: two consistent frames book slot 1");

        // 回合结束（没有点亮扇叶）→ 掩码归零，编号保留
        RuneSlotLattice::Frame ended = latticeFrame(0.0, 0, target, {});
        ended.round_active = false;
        lattice.update(ended, 1.2);
        const auto after_round = lattice.snapshot();
        check(!after_round.activated[1], "lattice: a finished round clears the mask");
        check(after_round.slot_id == 0, "lattice: the round reset keeps the slot numbering");

        // 单帧噪声会被衰减掉，不会置位
        RuneSlotLattice noise(config);
        noise.update(latticeFrame(0.0, 0, target, {{1, activated}}), 2.0);
        for (int i = 0; i < 8; ++i) {
            noise.update(latticeFrame(0.0, 0, target, {}), 2.05 + 0.05 * i);
        }
        check(!noise.activated()[1], "lattice: one noisy frame decays away without booking");
    }

    void testLatticeSilenceAndFullReset()
    {
        RuneSlotLattice::Config config;
        config.reset_silence_s = 5.0;
        config.activate_votes = 4;
        RuneSlotLattice lattice(config);
        const cv::Point2f target(0.0f, -100.0f);
        // 靶心在 12 点方向，已激活片在它 +144°（两个槽位）处 → 槽位 2 记账
        const cv::Point2f two_slots_ahead(100.0f * std::sin(2.0 * kSlot),
                                          -100.0f * std::cos(2.0 * kSlot));
        for (int i = 0; i < 2; ++i) {
            lattice.update(latticeFrame(0.0, 0, target, {{1, two_slots_ahead}}), 1.0 + 0.05 * i);
        }
        check(lattice.activated()[2],
              "lattice: an activated blade two slots ahead books slot 2");

        // 长时间静默 → 清空"已激活"记账，但编号保留
        RuneSlotLattice::Frame blank;
        lattice.update(blank, 10.0);
        const auto after_silence = lattice.snapshot();
        check(!after_silence.activated[0] && !after_silence.activated[1] &&
                  !after_silence.activated[2] && !after_silence.activated[3] &&
                  !after_silence.activated[4],
              "lattice: long silence clears the activation bookkeeping");

        // 5 片打满 → 机关复位，清空记账（每个槽位放一个已激活片）
        RuneSlotLattice full(config);
        std::vector<std::pair<int, cv::Point2f>> every_slot;
        for (int k = 0; k < 5; ++k) {
            const double angle = k * kSlot;
            every_slot.emplace_back(
                1, cv::Point2f(100.0f * std::sin(angle), -100.0f * std::cos(angle)));
        }
        for (int i = 0; i < 2; ++i) {
            full.update(latticeFrame(0.0, 0, target, every_slot), 1.0 + 0.05 * i);
        }
        // 打满之前先确认这 5 个点确实落在 5 个不同的槽位上
        RuneSlotLattice probe(config);
        auto one_by_one = every_slot;
        std::array<bool, 5> seen{};
        for (std::size_t i = 0; i + 1 < one_by_one.size(); ++i) {
            for (int repeat = 0; repeat < 2; ++repeat) {
                probe.update(latticeFrame(0.0, 0, target, {one_by_one[i]}), 1.0 + 0.05 * repeat);
            }
            const auto booked = probe.snapshot().votes;
            for (std::size_t slot = 0; slot < booked.size(); ++slot) {
                if (booked[slot] > 0) seen[slot] = true;
            }
        }
        std::size_t distinct = 0;
        for (bool value : seen) distinct += value ? 1u : 0u;
        check(distinct == 4, "lattice: the four candidates book four distinct slots");
        const auto reset = full.snapshot();
        std::string mask_debug;
        for (bool value : reset.activated) mask_debug += value ? '1' : '0';
        std::string votes_debug;
        for (int value : reset.votes) votes_debug += std::to_string(value);
        check(!reset.activated[0] && !reset.activated[1] && !reset.activated[2] &&
                  !reset.activated[3] && !reset.activated[4],
              "lattice: five booked slots reset the mechanism (mask=" + mask_debug +
                  " votes=" + votes_debug + ")");
    }

    // ---- 丢目标时的云台策略 ---------------------------------------------------
    void testFallbackParkAndRecenter()
    {
        using Fallback = RuneAimFallback;
        Fallback::Config config;
        config.park_on_center_max_age_s = 1.5;
        config.recenter_after_frames = 3;

        // 从没见过符心：计数到阈值就回中
        Fallback blind(config);
        for (int i = 0; i < 3; ++i) {
            const auto decision = blind.decide(1.0 + 0.1 * i, 0.5, 0.2);
            check(decision.kind == Fallback::Decision::Kind::None,
                  "fallback: no recenter before the frame threshold");
        }
        const auto home = blind.decide(1.4, 0.5, 0.2);
        check(home.kind == Fallback::Decision::Kind::Recenter,
              "fallback: recenters after recenter_after_frames with no centre at all");
        check(home.reset_estimator, "fallback: recentering resets the estimator");
        check(blind.recenterCount() == 1, "fallback: recenter is counted");

        // 已经接近标定位姿：仍然复位估计器，但不必再发一条回中指令
        Fallback at_home(config);
        for (int i = 0; i < 3; ++i) at_home.decide(1.0 + 0.1 * i, 0.0, 0.0);
        const auto already_home = at_home.decide(1.4, 0.0, 0.0);
        check(already_home.kind == Fallback::Decision::Kind::None,
              "fallback: no home command when the gimbal already sits at home");
        check(already_home.reset_estimator,
              "fallback: the estimator is reset even if no home command is sent");
        check(at_home.recenterCount() == 0, "fallback: nothing counted when already at home");

        // 有新鲜符心：优先停靠，且停靠角就是最后已知符心角
        Fallback parked(config);
        parked.noteCenter(0.42, -0.11, 10.0);
        const auto keep = parked.decide(10.5, 0.3, 0.1);
        check(keep.kind == Fallback::Decision::Kind::AimLastCenter,
              "fallback: parks on the last known centre while it is fresh");
        check(std::abs(keep.yaw - 0.42) < 1e-12 && std::abs(keep.pitch + 0.11) < 1e-12,
              "fallback: park angles are the last known centre angles");
        check(parked.parkedFrames() == 1, "fallback: parked frames are counted");

        // 符心过期 → 不再停靠，回中
        Fallback stale(config);
        stale.noteCenter(0.42, -0.11, 10.0);
        stale.decide(12.0, 0.3, 0.1);   // 过期（2.0 s > 1.5 s），计数 1
        stale.decide(12.1, 0.3, 0.1);
        stale.decide(12.2, 0.3, 0.1);
        const auto stale_home = stale.decide(12.3, 0.3, 0.1);
        check(stale_home.kind == Fallback::Decision::Kind::Recenter,
              "fallback: a stale centre falls through to recentering");
        check(stale.parkedFrames() == 0, "fallback: a stale centre is never used to park");

        // 中间恢复过控制 → 计数清零，重新开始数
        Fallback recovers(config);
        recovers.decide(20.0, 0.5, 0.2);
        recovers.decide(20.1, 0.5, 0.2);
        recovers.noteControlAvailable();
        recovers.decide(20.2, 0.5, 0.2);
        recovers.decide(20.3, 0.5, 0.2);
        check(recovers.decide(20.4, 0.5, 0.2).kind == Fallback::Decision::Kind::None,
              "fallback: a frame with a usable target restarts the lost-target count");
    }
    // ---- 回转椭圆拟合 ---------------------------------------------------------
    void testOrbitFit()
    {
        // 造一条"椭圆轨道"的**相对向量**（靶心−符心）：半长轴 70、半短轴 55、倾斜 20°。
        // 注意拟合的是相对量，所以椭圆中心应当在原点附近。
        RuneOrbitFit fit;
        const cv::Point2f center(0.0f, 0.0f);
        const double a = 70.0, b = 55.0, tilt = 20.0 * kPi / 180.0;
        double now = 0.0;
        for (int i = 0; i < 40; ++i) {
            const double t = 2.0 * kPi * i / 40.0;
            const cv::Point2f local(static_cast<float>(a * std::cos(t)),
                                    static_cast<float>(b * std::sin(t)));
            const cv::Point2f rotated(
                static_cast<float>(local.x * std::cos(tilt) - local.y * std::sin(tilt)),
                static_cast<float>(local.x * std::sin(tilt) + local.y * std::cos(tilt)));
            now += 0.05;
            fit.add(center + rotated, now);
        }
        const auto result = fit.fit(now);
        check(result.valid, "orbit fit: a full-arc synthetic ellipse is fitted");
        check(cv::norm(result.center) < 1.5,
              "orbit fit: the relative-vector ellipse is centred at the origin (" +
                  std::to_string(cv::norm(result.center)) + " px off)");
        check(std::abs(result.semi_major_px - a) < 2.0,
              "orbit fit: recovers the semi-major axis (got " +
                  std::to_string(result.semi_major_px) + " vs 70)");

        // 被云台平移污染的样本（椭圆中心偏得远）必须被拒 → 退回瞬时偏移
        RuneOrbitFit shifted;
        double ts = 0.0;
        for (int i = 0; i < 30; ++i) {
            const double ang = 2.0 * kPi * i / 30.0;
            ts += 0.05;
            shifted.add(cv::Point2f(static_cast<float>(60.0 + a * std::cos(ang)),
                                    static_cast<float>(-40.0 + b * std::sin(ang))), ts);
        }
        check(!shifted.fit(ts).valid,
              "orbit fit: a translation-contaminated cloud is rejected");

        // 样本几乎共线（短弧/直线）不许当作有效椭圆 → 调用方会退回瞬时偏移
        RuneOrbitFit narrow;
        double t2 = 0.0;
        for (int i = 0; i < 12; ++i) {
            const float x = 380.0f + 4.0f * i;    // 一条直线上的点
            t2 += 0.05;
            narrow.add(cv::Point2f(x, 260.0f + 0.25f * (x - 400.0f)), t2);
        }
        check(!narrow.fit(t2).valid, "orbit fit: collinear samples are rejected");

        // 少量野值不影响结果
        RuneOrbitFit noisy;
        double t3 = 0.0;
        for (int i = 0; i < 30; ++i) {
            const double ang = 2.0 * kPi * i / 30.0;
            const cv::Point2f local(static_cast<float>(a * std::cos(ang)),
                                    static_cast<float>(b * std::sin(ang)));
            cv::Point2f point = center + local;
            if (i % 10 == 3) point += cv::Point2f(25.0f, -18.0f);   // 野值
            t3 += 0.05;
            noisy.add(point, t3);
        }
        const auto noisy_result = noisy.fit(t3);
        check(noisy_result.valid && std::abs(noisy_result.semi_major_px - a) < 4.0,
              "orbit fit: outliers are rejected by the refit (got " +
                  std::to_string(noisy_result.semi_major_px) + " vs 70)");
    }
} // namespace

int main()
{
    testRoundWindow();
    testBladeJumpRestartsRound();
    testClassHitNeedsInactiveFirst();
    testPostHitHold();
    testScoredHitRestartsRound();
    testLatticeAnchoring();
    testLatticeActivationBookkeeping();
    testLatticeRetiredSlotBooksActivation();
    testLatticeSilenceAndFullReset();
    testLatticeRoundResetAndDecay();
    testFallbackParkAndRecenter();
    testOrbitFit();
    if (failures == 0) {
        std::printf("rune_pipeline_state_test: all checks passed\n");
        return 0;
    }
    std::printf("rune_pipeline_state_test: %d check(s) failed\n", failures);
    return 1;
}
