#!/usr/bin/env python3
"""能量机关运行曲线：一次跑完直接看四条曲线，而不是只看激活次数。

用法：
    python3 tools/rune_curve.py <run_dir> [--truth-distance 6.342] [--out curve.png]

<run_dir> 由 tools/rune_sim_test.sh 产出（vision.csv / sim.log）。
输出（同时打印到终端 + 画成图片）：

  1. 可用性曲线   每帧 status 与是否有目标 → 目标占空比
  2. 相位曲线     aim_phase_deg（图像相位，靶心相对 R 标） vs ekf_roll_deg
  3. 距离曲线     meas_dis / R_dis，和真值线对比
  4. 每轮命中     sim.log 里每一轮是否被击中（连续命中数 = 激活的关键）

设计原则（见 docs/energy_rune_vs_rp26.md §5）：先看曲线再改代码，一次只改一个因素。
"""

import argparse
import csv
import math
import re
import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt  # noqa: E402


def load_rows(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def as_float(value, default=float("nan")):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def rounds_from_sim(sim_log, hits):
    """sim.log -> 轮次起点列表 + 每轮是否命中（小符 face=1）。"""
    starts = []
    current = None
    if not Path(sim_log).exists():
        return starts, []
    for line in Path(sim_log).read_text(errors="ignore").splitlines():
        match = re.search(
            r"event=state t=([\d.]+) face=(\d) mode=(\w+) phase=(\w+).*highlighted=([^ ]+)", line
        )
        if match and match.group(2) == "1" and match.group(3) == "small":
            stamp = float(match.group(1))
            key = (match.group(4), match.group(5))
            if key != current:
                current = key
                if match.group(4) == "activating":
                    starts.append(stamp)
    unique = []
    for stamp in starts:
        if not unique or stamp - unique[-1] > 0.2:
            unique.append(stamp)
    marks = []
    for index, stamp in enumerate(unique):
        end = unique[index + 1] if index + 1 < len(unique) else float("inf")
        # 命中按 [本轮起点, 下一轮起点) 归属，不加容差：加容差会把边界上的同一次
        # 命中同时算进相邻两轮（曾因此虚报"连续 5 轮命中"）。
        marks.append(any(stamp <= hit < end for hit in hits))
    return unique, marks


def longest_run(marks):
    best = run = 0
    for mark in marks:
        run = run + 1 if mark else 0
        best = max(best, run)
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--truth-distance", type=float, default=6.342)
    parser.add_argument("--out", default=None, help="图片输出路径（默认 <run_dir>/curve.png）")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    rows = load_rows(run_dir / "vision.csv")
    if not rows:
        print("no vision.csv rows")
        return

    raw_times = [as_float(row["time_us"]) / 1e6 for row in rows]
    sim_times = [as_float(row.get("sim_t")) for row in rows]
    use_sim_time = any(value > 1.0 for value in sim_times)
    if use_sim_time:
        times = sim_times
        print("  时间轴：仿真时间（sim_t 列）——可直接和轮次/命中对齐")
    else:
        times = [value - raw_times[0] for value in raw_times]
    latency = [as_float(row["latency_ms"]) for row in rows]
    status_ok = [0.0 if row["status"] == "LOST" else 1.0 for row in rows]
    has_target = [1.0 if as_float(row["cx"]) > 0 or as_float(row["r_center_x"]) > 0 else 0.0
                  for row in rows]
    phase_aim = [as_float(row.get("aim_phase_deg")) for row in rows]
    phase_ekf = [as_float(row.get("ekf_roll_deg")) for row in rows]
    distance_obs = [as_float(row["meas_dis"]) for row in rows]
    distance_ekf = [as_float(row["R_dis"]) for row in rows]
    plate_r = [as_float(row.get("plate_r_px")) for row in rows]

    # ---- 每轮命中（sim.log） ----
    hits = []
    sim_log = run_dir / "sim.log"
    if sim_log.exists():
        for line in sim_log.read_text(errors="ignore").splitlines():
            match = re.search(
                r"event=hit t=([\d.]+) face=(\d) mode=(\w+) phase=(\w+) hit=(-?\d+)"
                r" accurate=(\d+) scored=(\d+)",
                line,
            )
            if match and match.group(2) == "1" and match.group(3) == "small" and match.group(7) == "1":
                hits.append(float(match.group(1)))
    round_starts, round_marks = rounds_from_sim(sim_log, sorted(set(round(h, 1) for h in hits)))

    # ---- 终端摘要 ----
    track_rate = 100.0 * sum(status_ok) / len(rows)
    target_rate = 100.0 * sum(has_target) / len(rows)
    valid_obs = [value for value in distance_obs if value > 1.0]
    print(f"run            : {run_dir}")
    print(f"frames         : {len(rows)}  时间跨度 {max(times) - min(times):.1f} s")
    print(f"TRACK 占空比   : {track_rate:.1f}%   有目标占空比: {target_rate:.1f}%")
    print(f"latency ms     : p50 {st.median(latency):.1f}")
    if valid_obs:
        median = st.median(valid_obs)
        print(f"distance       : obs 中位 {median:.2f} m (真值 {args.truth_distance:.2f}, "
              f"{100.0 * (median - args.truth_distance) / args.truth_distance:+.1f}%)")
    print(f"plate_r px     : 中位 {st.median([v for v in plate_r if v > 0] or [float('nan')]):.1f}")
    if round_marks:
        print(f"轮次           : {len(round_marks)} 轮，命中 {sum(round_marks)} 轮 "
              f"({100.0 * sum(round_marks) / len(round_marks):.0f}%)，"
              f"最长连续命中 {longest_run(round_marks)}（需要 5 才算激活）")
        print("     " + "".join("Y" if mark else "." for mark in round_marks))

    # ---- 云台抖动/换靶指标（用户反馈的"两片已激活扇叶之间来回动"） ----
    aim = [as_float(row["aim_yaw"]) for row in rows]
    phase = [as_float(row.get("aim_phase_deg")) for row in rows]
    steps = [abs(aim[i] - aim[i - 1]) for i in range(1, len(aim))
             if aim[i] == aim[i] and aim[i - 1] == aim[i - 1] and aim[i] != 0.0 and aim[i - 1] != 0.0]
    jumps = 0
    for i in range(1, len(phase)):
        if phase[i] != phase[i] and phase[i - 1] != phase[i - 1]:
            continue
        if phase[i] == 0.0 or phase[i - 1] == 0.0:
            continue
        if abs((phase[i] - phase[i - 1] + 180.0) % 360.0 - 180.0) > 20.0:
            jumps += 1
    # 云台大角度摆动：用户现场反馈的"莫名朝一侧大角度转再归位"。
    sent = [as_float(row["sent_yaw"]) for row in rows]
    big_swings = 0
    max_swing = 0.0
    for i in range(1, len(sent)):
        if sent[i] != sent[i] or sent[i - 1] != sent[i - 1] or sent[i] == 0.0 or sent[i - 1] == 0.0:
            continue
        delta = abs(sent[i] - sent[i - 1])
        max_swing = max(max_swing, delta)
        if delta > math.radians(15.0):
            big_swings += 1
    print("云台下发角     : |Δsent_yaw|>15° 的帧 %d，最大 %.1f°  <- 大角度甩飞指标（应接近 0）"
          % (big_swings, math.degrees(max_swing)))
    if steps:
        ordered = sorted(math.degrees(value) for value in steps)
        print(f"云台目标角     : 逐帧|Δaim_yaw| 中位 {ordered[len(ordered) // 2]:.2f}° "
              f"p90 {ordered[int(0.9 * (len(ordered) - 1))]:.2f}° | "
              f"目标相位跳变(换靶) {jumps} 次  ← 左右拉扯的直接指标")

    # ---- 画图 ----
    figure, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=False)

    axes[0].fill_between(times, status_ok, step="post", alpha=0.35, label="TRACK (not LOST)")
    axes[0].plot(times, has_target, linewidth=1.0, label="has target")
    axes[0].set_ylabel("availability")
    axes[0].set_ylim(-0.1, 1.2)
    axes[0].legend(loc="upper right", fontsize=8)

    axes[1].plot(times, phase_aim, ".", markersize=2, label="image phase (plate vs R-mark)")
    axes[1].plot(times, phase_ekf, ".", markersize=2, label="EKF phase")
    axes[1].set_ylabel("deg")
    axes[1].legend(loc="upper right", fontsize=8)

    axes[2].plot(times, distance_obs, ".", markersize=2, label="observed distance")
    axes[2].plot(times, distance_ekf, "-", linewidth=1.0, label="EKF distance")
    axes[2].axhline(args.truth_distance, color="k", linestyle="--", linewidth=1,
                    label=f"truth {args.truth_distance:.2f} m")
    axes[2].set_ylabel("m")
    axes[2].legend(loc="upper right", fontsize=8)

    if round_starts:
        axes[3].bar(range(len(round_marks)), [1 if m else 0.15 for m in round_marks],
                    color=["tab:green" if m else "tab:red" for m in round_marks])
        axes[3].set_xticks(range(len(round_marks)))
        axes[3].set_xticklabels([f"{value:.1f}" for value in round_starts], fontsize=7)
        axes[3].set_ylabel("rounds")
        axes[3].set_xlabel("round start (sim time, s)")

    # ---- 扇叶 ID 记账一致性（与仿真遥测的 activated 掩码逐帧比对）----
    # ---- 云台控制链在瞄准上的效果（结合 aim_px / aim_world / sent 数据）----
    if any(row.get("aim_px") for row in rows):
        aim_px = [as_float(row.get("aim_px")) for row in rows]
        aim_py = [as_float(row.get("aim_py")) for row in rows]
        dx = [value - 320.0 for value in aim_px if value > 0.0]
        dy = [value - 240.0 for value in aim_py if value > 0.0]
        if dx and dy:
            def p90(values):
                ordered = sorted(abs(value) for value in values)
                return ordered[int(0.9 * (len(ordered) - 1))]

            print("瞄准残差(相对画面中心): 水平 中位 %+.1f px p90 %.1f px (%.2f° / %.2f°) | 竖直 中位 %+.1f px "
                  "p90 %.1f px (弹道抬升+跟踪，%.2f° / %.2f°)"
                  % (st.median(dx), p90(dx), math.degrees(st.median(dx) / 579.4),
                     math.degrees(p90(dx) / 579.4), st.median(dy), p90(dy),
                     math.degrees(st.median(dy) / 579.4), math.degrees(p90(dy) / 579.4)))
        track = [abs(as_float(row.get("aim_yaw")) - as_float(row.get("sent_yaw")))
                 for row in rows
                 if as_float(row.get("aim_yaw")) == as_float(row.get("aim_yaw"))
                 and as_float(row.get("sent_yaw")) == as_float(row.get("sent_yaw"))
                 and as_float(row.get("aim_yaw")) != 0.0]
        if track:
            ordered = sorted(math.degrees(value) for value in track)
            print("云台跟踪      : |aim_yaw − sent_yaw| 中位 %.2f°  p90 %.2f°"
                  "  <- 轨迹生成器的跟踪误差（steady 很小、换靶瞬态较大）"
                  % (ordered[len(ordered) // 2], ordered[int(0.9 * (len(ordered) - 1))]))

    if sim_log.exists():
        events = []
        for line in sim_log.read_text(errors="ignore").splitlines():
            match = re.search(r"event=state t=([\d.]+) face=1 mode=small .*activated=(\d+)", line)
            if match:
                events.append((float(match.group(1)), match.group(2)))
        events.sort()

        def mask_at(stamp):
            best = None
            for value_time, value in events:
                if value_time <= stamp:
                    best = value
                else:
                    break
            return best

        compared = 0
        mismatched = 0
        active = 0
        for row in rows:
            stamp = as_float(row.get("sim_t"))
            ours = row.get("slot_activated_mask", "")
            if stamp != stamp or len(ours) != 5:
                continue
            truth = mask_at(stamp)
            if not truth:
                continue
            if ours.count("1") > 0 or truth.count("1") > 0:
                active += 1
            compared += 1
            if truth.count("1") != ours.count("1"):
                mismatched += 1
        if compared:
            print("slot 记账一致 : 有激活片时 %d 帧，与遥测不一致 %d 帧 (%.0f%%)"
                  "  <- 这个数降下来之前，扇叶 ID 闸门不要开"
                  % (active, mismatched, 100.0 * mismatched / compared))

    figure.tight_layout()
    out = Path(args.out) if args.out else run_dir / "curve.png"
    figure.savefig(out, dpi=110)
    print(f"curve saved    : {out}")


if __name__ == "__main__":
    main()
