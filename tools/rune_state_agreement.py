#!/usr/bin/env python3
"""槽位记账 vs 仿真真值：我们的 `slot_activated_mask` 到底对不对？

用法：
    python3 tools/rune_state_agreement.py <run_dir>

数据来源（都由 tools/rune_sim_test.sh 产出）：
  vision.csv       我们逐帧的 slot_id / slot_activated_mask（自己数出来的已激活槽位）
  ground_truth.csv 仿真逐条 RUNE 遥测：activated=NNNNN（真值掩码）、highlighted（亮片数）

为什么要专门量这个：开火闸门"别打已经打过的槽位"（fire_block_on_hit_slot）
完全建立在这份记账上；记账错了，闸门只会让我们少打有效片。
这里的对齐方法：两边的槽位编号起点未知（我们锚定在"第一次看到 class 0"的相位上），
所以对 5 个循环移位都算一遍，取最好的那个 —— 报告里明确写出用了哪个移位。
"""

import argparse
import bisect
import csv
import statistics as st


def load(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def parse_truth(run_dir):
    """[(local_time_us, mask_int, highlighted), ...]"""
    out = []
    for row in load(f"{run_dir}/ground_truth.csv"):
        line = row["line"]
        mask = None
        highlighted = None
        for token in line.split():
            if token.startswith("activated="):
                mask = token.split("=", 1)[1]
            elif token.startswith("highlighted="):
                highlighted = token.split("=", 1)[1]
        if mask is None:
            continue
        if highlighted in (None, "-"):
            continue
        try:
            out.append((int(row["local_time_us"]), mask, int(highlighted)))
        except ValueError:
            continue
    return out


def mask_bits(mask):
    return [int(character) for character in mask]


def rotate(bits, shift):
    return bits[shift:] + bits[:shift]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--tol", type=float, default=0.25,
                        help="与最近遥测样本允许的最大时间差（秒）")
    args = parser.parse_args()

    truth = parse_truth(args.run_dir)
    if not truth:
        print("没有可用的遥测行")
        return 1
    truth_times = [t for t, _, _ in truth]

    rows = load(f"{args.run_dir}/vision.csv")
    pairs = []          # (our bits, truth bits, our slot_id, truth highlighted)
    for row in rows:
        if int(row.get("slot_id", "-1")) < 0:
            continue
        mask = row.get("slot_activated_mask", "").strip()
        if len(mask) != 5:
            continue
        now_us = int(row["time_us"])
        index = bisect.bisect_left(truth_times, now_us)
        best = None
        for candidate in (index - 1, index, index + 1):
            if 0 <= candidate < len(truth):
                delta = abs(truth_times[candidate] - now_us) / 1e6
                if best is None or delta < best[0]:
                    best = (delta, candidate)
        if best is None or best[0] > args.tol:
            continue
        pairs.append((mask_bits(mask), mask_bits(truth[best[1]][1]),
                      int(row["slot_id"]), truth[best[1]][2]))

    if not pairs:
        print("没有对齐上的帧（检查 slot_id 是否已锚定 / 时间轴是否对齐）")
        return 1

    print(f"对齐帧数            : {len(pairs)}")
    identical = sum(1 for ours, theirs, _, _ in pairs if ours == theirs)
    popcount_ok = sum(1 for ours, theirs, _, _ in pairs if sum(ours) == sum(theirs))
    print(f"掩码完全一致(未移位) : {identical / len(pairs):6.1%}")
    print(f"已激活个数一致       : {popcount_ok / len(pairs):6.1%}")

    best_shift, best_identical = 0, -1
    for shift in range(5):
        hits = sum(1 for ours, theirs, _, _ in pairs
                   if rotate(ours, shift) == theirs)
        if hits > best_identical:
            best_shift, best_identical = shift, hits
    print(f"最优循环移位         : {best_shift}（一致率 {best_identical / len(pairs):.1%}）")

    ours_count = [sum(ours) for ours, _, _, _ in pairs]
    theirs_count = [sum(theirs) for _, theirs, _, _ in pairs]
    print(f"已激活片数 我方 p50={st.median(ours_count):.0f} 真值 p50={st.median(theirs_count):.0f}"
          f"  最大 我方 {max(ours_count)} / 真值 {max(theirs_count)}")

    # 闸门安全性：目标槽位是否落在"真值也认为已激活"的位子上 —— 这是
    # fire_block_on_hit_slot 会挡掉的帧占比（挡错=白白少打一发）。
    blocked = wrong = 0
    for ours, theirs, slot, _ in pairs:
        index = (slot + best_shift) % 5
        aimed_activated = theirs[index] == 1
        we_block = ours[slot] == 1
        if we_block:
            blocked += 1
            if not aimed_activated:
                wrong += 1
    wrong_share = wrong / blocked if blocked else 0.0
    print(f"闸门会挡掉的帧       : {blocked} / {len(pairs)} ({blocked / len(pairs):.1%})，"
          f"其中真值认为'这片还没打过'的 {wrong} 帧 ({wrong_share:.1%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
