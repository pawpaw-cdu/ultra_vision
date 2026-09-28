#!/usr/bin/env python3
"""能量机关检测/估计的量化评分。

用法：
    python3 tools/rune_model_score.py <run_dir> [--truth-distance 6.342] [--pad 1.36]

<run_dir> 由 tools/rune_sim_test.sh 产出（vision.csv / ground_truth.csv / frames/）。
输出四类指标，用于"换模型 / 换距离 / 换 pad"前后的横向对比：

  1. 检出率     非 LOST 帧占全部处理帧的比例
  2. 延迟       推理延迟 p50 / p90（毫秒）
  3. 距离精度   PnP 圆心距离相对仿真真值的中位误差（真值 = 相机预置里的距离）
  4. 状态一致性（有类别输出后启用）class 0 框数 与 遥测 highlighted 片数 的逐帧一致率

后两项是这套量化里最有价值的：距离误差直接决定弹丸落点，
状态一致性直接决定"该打哪一片"。
"""

import argparse
import bisect
import csv
import math
import re
import statistics as st
from pathlib import Path


def load(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def percentile(values, fraction):
    if not values:
        return float("nan")
    values = sorted(values)
    return values[min(len(values) - 1, int(fraction * (len(values) - 1)))]


def telemetry_states(run_dir):
    """逐帧遥测：{(mode, face): [(t, phase, highlighted_count), ...]}"""
    out = []
    gt = Path(run_dir) / "ground_truth.csv"
    if not gt.exists():
        return out
    for row in load(gt):
        line = row["line"]
        if "event=state" not in line:
            continue
        phase = re.search(r"phase=(\w+)", line)
        highlighted = re.search(r"highlighted=([^ ]+)", line)
        count = 0 if not highlighted or highlighted.group(1) == "-" else len(
            highlighted.group(1).split(","))
        activated = re.search(r"activated=([01]+)", line)
        out.append({
            "time_us": int(row["local_time_us"]),
            "phase": phase.group(1) if phase else "?",
            "lit": count,
            "activated": activated.group(1).count("1") if activated else -1,
        })
    return out


def class_availability_by_activated(run_dir, rows):
    """按"已经打了几片"分组统计：有候选 / 有 class0 候选 的帧占比。

    这是当前最要紧的一张表：模型对**未激活**（class 0）的判别会随着机关被激活
    而变差，而检测器的类别闸门只放行 class 0 —— 于是越接近激活，越拿不到观测。
    """
    states = telemetry_states(run_dir)
    if not states:
        return None
    times = [s["time_us"] for s in states]
    groups = {}
    for row in rows:
        now = int(row["time_us"])
        index = bisect.bisect_left(times, now)
        best = None
        for candidate in (index - 1, index, index + 1):
            if 0 <= candidate < len(states):
                delta = abs(times[candidate] - now) / 1e6
                if best is None or delta < best[0]:
                    best = (delta, candidate)
        if best is None or best[0] > 0.25:
            continue
        activated = states[best[1]]["activated"]
        if activated < 0:
            continue
        bucket = groups.setdefault(activated, [0, 0, 0])
        bucket[0] += 1
        if float(row["cand_n"]) > 0:
            bucket[1] += 1
        if float(row.get("class0", 0)) > 0:
            bucket[2] += 1
    return groups


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    # 真值 = 相机到 R 标的**斜距**：预置里相机 (-4.512,0.55,-4.424)、R 标 (-0.16,2.45,-0.22)
    # ⇒ 水平 6.05 m、抬高 1.90 m ⇒ 斜距 6.342 m。以前误用水平的 6.05，导致所有
    # 距离偏差被系统性高估约 4.6%（见 docs/energy_rune_issue_audit.md §17）。
    parser.add_argument("--truth-distance", type=float, default=6.342,
                        help="仿真相机到圆盘中心的真实距离（米），来自相机预置")
    parser.add_argument("--pad", type=float, default=None)
    args = parser.parse_args()

    rows = load(Path(args.run_dir) / "vision.csv")
    if not rows:
        print("no vision.csv rows")
        return

    tracked = [r for r in rows if r["status"] != "LOST"]
    latency = [float(r["latency_ms"]) for r in rows]
    distances = [float(r["meas_dis"]) for r in tracked if float(r["meas_dis"]) > 1.0]

    # "有观测" 与 "非 LOST" 是两件事：状态是 TRACK/TEMP_LOST 里大量帧只是**在滑行**
    # （估计器外推），真正拿到本帧解算结果的帧少得多。这一列才是检测/解算的真实
    # 占空比 —— 它决定槽位记账有多少证据、瞄准有多少在靠预测。
    fresh = [r for r in rows if abs(float(r["cx"])) > 1e-6]
    with_candidates = [r for r in rows if float(r["cand_n"]) > 0]
    with_class0 = [r for r in rows if float(r.get("class0", 0)) > 0]

    print(f"run          : {args.run_dir}")
    print(f"pad          : {args.pad if args.pad is not None else 'config default'}")
    print(f"frames       : {len(rows)}")
    print(f"detect rate  : {100.0 * len(tracked) / len(rows):.1f}%")
    print(f"fresh obs    : {100.0 * len(fresh) / len(rows):.1f}%  "
          f"(本帧真的解算出观测；其余帧靠滑行)")
    print(f"candidates   : {100.0 * len(with_candidates) / len(rows):.1f}%  "
          f"class0 可用 {100.0 * len(with_class0) / len(rows):.1f}%  "
          f"→ 有候选但无 class0（类别闸门会挡住该帧）"
          f" {100.0 * (len(with_candidates) - len(with_class0)) / len(rows):.1f}%")
    print(f"latency ms   : p50 {percentile(latency, 0.5):.1f}  p90 {percentile(latency, 0.9):.1f}")
    if distances:
        median = st.median(distances)
        error = 100.0 * (median - args.truth_distance) / args.truth_distance
        print(f"PnP distance : median {median:.2f} m (truth {args.truth_distance:.2f} m, "
              f"{error:+.1f}%), p10 {percentile(distances, 0.1):.2f}, "
              f"p90 {percentile(distances, 0.9):.2f}")
    else:
        print("PnP distance : no valid samples")

    # 状态一致性：需要模型输出类别时才严格成立；这里先用"遥测点亮片数"分布
    # 把 each 相位的占比列出来，方便与后面模型 class 0 的框数对照。
    states = telemetry_states(args.run_dir)
    if states:
        by_phase = {}
        for s in states:
            by_phase.setdefault(s["phase"], []).append(s["lit"])
        summary = ", ".join(
            f"{phase}: {len(v)}帧/lit中位{int(st.median(v))}" for phase, v in sorted(by_phase.items()))
        print(f"telemetry    : {summary}")

    groups = class_availability_by_activated(args.run_dir, rows)
    if groups:
        print("class0 可用性 : 按真值[已激活片数]分组")
        for activated in sorted(groups):
            frames, candidates, class0 = groups[activated]
            print(f"  activated={activated}: {frames:4d} 帧  "
                  f"有候选 {100.0 * candidates / frames:5.1f}%  "
                  f"class0 可用 {100.0 * class0 / frames:5.1f}%")
    print("note         : detect rate 是[非 LOST 帧]，含大量滑行帧；"
          "判断检测能力要看 fresh obs / class0 可用性这两行。")


if __name__ == "__main__":
    main()
