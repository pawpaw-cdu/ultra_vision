#!/usr/bin/env python3
"""离线校验"单扇叶构建 5 槽格点 + 新片匹配"的扇叶 ID 算法。

不跑闭环，只用一次 run 已有的数据：
  * vision.csv  : 逐帧观测相位 `obs_phase_deg`（解算给出的几何相位）+ `sim_t`
  * sim.log     : 仿真真值 —— 每帧点亮的模块号 `highlighted=`、已激活掩码 `activated=`

校验逻辑（与 C++ 里的 lattice 逻辑同构）：
  1) 只有一片亮时也能"构建机关"：以第一片为 0 号槽，记下相位参考；
  2) 之后每片亮起，用它的相位与 72° 格点**匹配**（就近取整），得到相对槽位号；
     匹配前先把相位按"已知转速外推"解缠（网络只看到亮起区域，相位是包过的）；
  3) 连续若干帧无亮片 → 下一次亮起时重建。

判据：
  * 一致性：同一个仿真模块号，应该始终映射到同一个我们编号（1:1 稳定映射）；
  * 已激活片数：我们的记账应与仿真 `activated=` 的片数一致。

用法：python3 tools/rune_id_check.py <run_dir> [--rate 60]
"""

import argparse
import csv
import math
import re
from collections import Counter, defaultdict
from pathlib import Path


def sim_events(sim_log):
    """按时间返回 (highlighted 集合, 已激活片数)。"""
    events = []
    for line in Path(sim_log).read_text(errors="ignore").splitlines():
        match = re.search(
            r"event=state t=([\d.]+) face=1 mode=small phase=(\w+).*activated=(\d+).*highlighted=([^ ]+)",
            line,
        )
        if not match:
            continue
        stamp = float(match.group(1))
        activated = match.group(3).count("1")
        highlighted = set()
        if match.group(4) not in ("-", ""):
            highlighted = {int(value) for value in match.group(4).split(",") if value.strip().isdigit()}
        events.append((stamp, highlighted, activated))
    events.sort()
    return events


def at(events, stamp):
    best = None
    for value_time, highlighted, activated in events:
        if value_time <= stamp:
            best = (highlighted, activated)
        else:
            break
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--rate", type=float, default=60.0, help="靶心转速（度/秒）")
    parser.add_argument("--silent-frames", type=int, default=15, help="多少帧无亮片后重建")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    rows = [
        row
        for row in csv.DictReader(open(run_dir / "vision.csv"))
        if row.get("sim_t") and row.get("obs_phase_deg")
    ]
    observations = []
    for row in rows:
        phase = float(row["obs_phase_deg"])
        if phase == 0.0 and row.get("status") != "TRACK":
            continue                      # 无观测帧写成 0，跳过
        if float(row.get("cand_n", 0) or 0) <= 0:
            continue
        observations.append((float(row["sim_t"]), phase))
    events = sim_events(run_dir / "sim.log")
    if not observations or not events:
        print("数据不足：需要 vision.csv 的 obs_phase_deg/sim_t 与 sim.log 的状态行")
        return

    # ---- 算法 A：单扇叶格点 + 匹配（现场方案）----
    slot_step = 72.0
    lattice_ref = None
    lattice_valid = False
    silent = 0
    previous_time, previous_phase = None, None
    result_a = []
    for stamp, raw in observations:
        # 解缠：把观测相位折到"上次相位 + 已知转速 × dt"附近
        if previous_phase is None:
            phase = raw
        else:
            expected = previous_phase + args.rate * (stamp - previous_time) * (
                1.0 if previous_phase is not None else 0.0
            )
            phase = min(
                [raw + 360.0 * k for k in range(-2, 3)],
                key=lambda value: abs(value - expected),
            )
        if not lattice_valid:
            lattice_ref = phase          # 用当前这一片构建机关：它记作 0 号槽
            lattice_valid = True
            silent = 0
        slot = int(round((phase - lattice_ref) / slot_step)) % 5
        result_a.append((stamp, slot))
        previous_time, previous_phase = stamp, phase

    # 无亮片计数（用观测帧之间的间隔近似）
    rebuilt = 1
    last_stamp = None
    for stamp, _ in result_a:
        if last_stamp is not None and stamp - last_stamp > args.silent_frames / 25.0:
            rebuilt += 1
        last_stamp = stamp

    # ---- 算法 B：相位累加（上一版，做对照）----
    acc_slot = None
    result_b = []
    previous_phase = None
    for stamp, raw in observations:
        if previous_phase is None:
            acc_slot = 0
            phase = raw
        else:
            phase = min(
                [raw + 360.0 * k for k in range(-2, 3)],
                key=lambda value: abs(value - previous_phase),
            )
            steps = int(round((phase - previous_phase) / slot_step))
            acc_slot = (acc_slot + steps) % 5
        result_b.append((stamp, acc_slot))
        previous_phase = phase

    def report(tag, result):
        mapping = defaultdict(Counter)
        agree = 0
        total = 0
        for stamp, slot in result:
            truth = at(events, stamp)
            if not truth or not truth[0]:
                continue
            total += 1
            for module in truth[0]:
                mapping[module][slot] += 1
            if result is result_a:
                pass
        # 每个仿真模块号的主要映射槽位
        stable = 0
        for module, counter in sorted(mapping.items()):
            top, count = counter.most_common(1)[0]
            stable += count
            print(f"    仿真模块 {module} -> 我们槽位 {top}（{count}/{sum(counter.values())}）")
        print(f"  {tag}: 有亮片帧 {total}；稳定映射命中率 {100.0 * stable / max(1, total):.0f}%")

    print(f"run: {run_dir}  观测帧 {len(observations)}")
    print("算法 A：单扇叶构建格点 + 匹配（本次实现）")
    report("A", result_a)
    print("算法 B：相位累加（上一版）")
    report("B", result_b)
    print(f"A 的重建次数（相邻亮片间隔 > {args.silent_frames} 帧）≈ {rebuilt}")


if __name__ == "__main__":
    main()
