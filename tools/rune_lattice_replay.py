#!/usr/bin/env python3
"""离线回放槽位格点（RuneSlotLattice）：扫参数不用每次都跑仿真器。

用法：
    python3 tools/rune_lattice_replay.py <run_dir> [--sweep]

输入用一次真实跑出来的 vision.csv（含 `cands` 候选列、`ekf_roll_deg`、`cx/cy`、
`r_center_*`、`blade_class`）与 ground_truth.csv（真值 activated 掩码）。
这里把 rune_slot_lattice.cpp 的逻辑照着实现一遍，然后：
  * 默认：用当前配置（activate_votes=4 / gain=2 / decay=4）核对该次跑的数据；
  * --sweep：把 (阈值, 每帧增益, 衰减周期) 组合扫一遍，输出"掩码一致率 / 片数
    一致率 / 闸门误挡帧数"，用来选参数（避免"跑一轮 40 s 只能试一个组合"）。

注意：这是**回放**，不是替代真跑。回放里 `round_active` 用"最近 5 帧内有没有
解算结果"近似（真跑里由 RuneRoundGuard 给），命中反馈用遥测 scored=1 的帧近似。
"""

import argparse
import bisect
import csv
import math
from pathlib import Path

SLOT = 2.0 * math.pi / 5.0


def load(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def load_truth(run_dir):
    out = []
    for row in load(f"{run_dir}/ground_truth.csv"):
        tokens = dict(tok.split("=", 1) for tok in row["line"].split() if "=" in tok)
        if "activated" not in tokens or tokens.get("highlighted") in (None, "-"):
            continue
        out.append((int(row["local_time_us"]), tokens["activated"],
                    int(tokens.get("scored", "0"))))
    return out


def rotate(bits, shift):
    return bits[shift:] + bits[:shift]


def angle_offset(cx, cy, hub, target_phase):
    """图像夹角法：候选与靶心的夹角差 ÷72° 取整。"""
    phase = math.atan2(cy - hub[1], cx - hub[0])
    delta = phase - target_phase
    while delta > math.pi:
        delta -= 2 * math.pi
    while delta < -math.pi:
        delta += 2 * math.pi
    return round(delta / SLOT)


def assign_offset(cx, cy, hub, target_phase, slot_centers, mode, spacing_ratio=0.45):
    """返回相对当前瞄准槽位的偏移；None = 无法归属（离所有槽位都太远）。

    注意 -1 是合法偏移（后一片 = 槽位 4），所以不能用 -1 当哨兵。
    """
    if mode == "nearest" and slot_centers:
        nearest, nearest_distance = None, 0.0
        spacing, counted = 0.0, 0
        for index, (sx, sy) in enumerate(slot_centers):
            distance = math.hypot(sx - cx, sy - cy)
            if nearest is None or distance < nearest_distance:
                nearest, nearest_distance = index, distance
            nx, ny = slot_centers[(index + 1) % len(slot_centers)]
            spacing += math.hypot(nx - sx, ny - sy)
            counted += 1
        spacing = spacing / counted if counted else 0.0
        if spacing > 1.0:
            return nearest if nearest_distance <= spacing_ratio * spacing else None
    return angle_offset(cx, cy, hub, target_phase)


def replay(frames, truth, votes_to_activate, gain, decay_every, assign_mode="nearest"):
    """frames: [(t_us, solved, roll, hub, target_center, target_class, cands, fired)]"""
    activated = [0] * 5
    votes = [0] * 5
    anchored = False
    reference = 0.0
    decay_tick = 0
    round_active = True
    slot_id = -1
    last_solved_index = -10**9
    records = []

    for index, (t_us, solved, roll, hub, target_center, target_class, cands, slot_centers, fired) in \
            enumerate(frames):
        if solved:
            last_solved_index = index
        active = (index - last_solved_index) <= 5     # 近似 RuneRoundGuard 的 blade_lit
        if round_active and not active:
            activated = [0] * 5
            votes = [0] * 5
            decay_tick = 0
        round_active = active

        decay_tick += 1
        if decay_tick >= max(1, decay_every):
            decay_tick = 0
            votes = [max(0, value - 1) for value in votes]

        if solved:
            if not anchored and target_class == 0:
                reference = roll
                anchored = True
                activated = [0] * 5
                votes = [0] * 5
            if anchored:
                slot_index = round((roll - reference) / SLOT)
                slot_id = (slot_index % 5 + 5) % 5
            if active and anchored and slot_id >= 0:
                target_phase = math.atan2(target_center[1] - hub[1], target_center[0] - hub[0])
                for cls, cx, cy in cands:
                    if cls == 0:
                        continue
                    offset = assign_offset(cx, cy, hub, target_phase, slot_centers, assign_mode)
                    if offset is None:
                        continue
                    slot = ((slot_id + offset) % 5 + 5) % 5
                    votes[slot] = min(votes[slot] + gain, 3 * votes_to_activate)

        scored = truth.get(index, 0)
        if scored and slot_id >= 0:
            votes[slot_id] = min(votes[slot_id] + 3 * votes_to_activate // 2,
                                 3 * votes_to_activate)

        activated = [1 if value >= votes_to_activate else 0 for value in votes]
        if all(activated):
            activated = [0] * 5
            votes = [0] * 5
        records.append((activated, slot_id))
    return records


def evaluate(records, truth_masks):
    pairs = [(ours, truth_masks[index])
             for index, (ours, slot) in enumerate(records)
             if truth_masks.get(index) and slot >= 0]
    if not pairs:
        return None
    identical = sum(1 for ours, theirs in pairs if ours == theirs)
    popcount = sum(1 for ours, theirs in pairs if sum(ours) == sum(theirs))
    best_shift = max(range(5),
                     key=lambda shift: sum(1 for ours, theirs in pairs
                                           if rotate(ours, shift) == theirs))
    shifted = sum(1 for ours, theirs in pairs if rotate(ours, best_shift) == theirs)
    over = sum(1 for ours, theirs in pairs if sum(ours) > sum(theirs))
    under = sum(1 for ours, theirs in pairs if sum(ours) < sum(theirs))
    return {
        "n": len(pairs),
        "identical": identical / len(pairs),
        "popcount": popcount / len(pairs),
        "shift": best_shift,
        "shifted": shifted / len(pairs),
        "over": over,
        "under": under,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--assign", choices=("nearest", "angle"), default="nearest",
                        help="候选→槽位的归属方式")
    args = parser.parse_args()

    truth_rows = load_truth(args.run_dir)
    truth_times = [row[0] for row in truth_rows]
    rows = load(f"{args.run_dir}/vision.csv")

    frames = []
    truth_masks = {}
    for row in rows:
        now_us = int(row["time_us"])
        cands = []
        for item in filter(None, row.get("cands", "").split("|")):
            cls, x, y = item.split(":")
            cands.append((int(cls), float(x), float(y)))
        solved = abs(float(row["cx"])) > 1e-6 or abs(float(row["cy"])) > 1e-6
        slot_centers = []
        for item in filter(None, row.get("slot_centers", "").split("|")):
            sx, sy = item.split(":")
            slot_centers.append((float(sx), float(sy)))
        frames.append((now_us, solved, math.radians(float(row["ekf_roll_deg"])),
                       (float(row["r_center_x"]), float(row["r_center_y"])),
                       (float(row["cx"]), float(row["cy"])),
                       int(row["blade_class"]), cands, slot_centers, int(row["fired"])))
        index = bisect.bisect_left(truth_times, now_us)
        best = None
        for candidate in (index - 1, index, index + 1):
            if 0 <= candidate < len(truth_rows):
                delta = abs(truth_times[candidate] - now_us) / 1e6
                if best is None or delta < best[0]:
                    best = (delta, candidate)
        if best and best[0] <= 0.25:
            truth_masks[len(frames) - 1] = [int(ch) for ch in truth_rows[best[1]][1]]
            if truth_rows[best[1]][2]:
                pass
    scored_frames = set()
    for index in truth_masks:
        now_us = frames[index][0]
        jump = bisect.bisect_left(truth_times, now_us)
        for candidate in (jump - 1, jump, jump + 1):
            if 0 <= candidate < len(truth_rows) and truth_rows[candidate][2] == 1:
                scored_frames.add(index)
    truth_scored = {index: 1 if index in scored_frames else 0 for index in truth_masks}

    print(f"帧数 {len(frames)}，对齐上真值的 {len(truth_masks)}")
    if not args.sweep:
        records = replay(frames, truth_scored, 4, 2, 4, args.assign)
        result = evaluate(records, truth_masks)
        print(f"当前配置 (votes=4, gain=2, decay=4, assign={args.assign}):")
        for key, value in result.items():
            print(f"   {key:10s} {value}")
        return 0

    print(f"{'votes':>6} {'gain':>5} {'decay':>6} {'mask':>8} {'popcnt':>8} "
          f"{'shift':>6} {'shifted':>8} {'over':>5} {'under':>6}")
    best = None
    for votes in (1, 2, 3, 4, 5, 6):
        for gain in (1, 2, 3):
            for decay in (2, 4, 8, 10**6):
                records = replay(frames, truth_scored, votes, gain, decay, args.assign)
                result = evaluate(records, truth_masks)
                if result is None:
                    continue
                score = result["popcount"] + result["identical"]
                if best is None or score > best[0]:
                    best = (score, votes, gain, decay, result)
                print(f"{votes:6d} {gain:5d} {decay if decay < 10**6 else -1:6d} "
                      f"{result['identical']:8.1%} {result['popcount']:8.1%} "
                      f"{result['shift']:6d} {result['shifted']:8.1%} "
                      f"{result['over']:5d} {result['under']:6d}")
    _, votes, gain, decay, result = best
    print(f"\n最优（按 掩码一致率+片数一致率）: votes={votes} gain={gain} "
          f"decay={decay if decay < 10**6 else 'never'} → mask {result['identical']:.1%} "
          f"popcount {result['popcount']:.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
