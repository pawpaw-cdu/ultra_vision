#!/usr/bin/env python3
"""闭环方向判定：从实车录制（node.cpp 的 ULTRA_VISION_RECORD）看"瞄准误差在收敛还是发散"。

为什么需要它：云台"一进控制就转一个角度"可能是
  ① 下发改的符号反 → 误差随云台运动越来越大（发散，越追越远）；
  ② 仿射常数（坐标系/零点）没对上 → 误差收敛到一个非零常数（云台稳定停在错的角度）；
  ③ 就是这么设计的（靶板本来就不在视轴上，云台该转过去）。
三者用同一份录制就能分开：算 `err = 目标角 − yaw_sign·实际回传角`，看 |err| 的趋势。

用法：
    tools/loop_error_check.py <数据集目录> [--yaw-sign -1]
输出：两轴各自的 |err| 前/后半段均值、斜率、<2° 占比，以及"误差 ↔ 编码器运动"的相关性
（正相关 = 编码器在**远离**目标 = 发散）。
"""
import csv
import math
import os
import re
import sys


def readRows(folder):
    path = os.path.join(folder, "meta.csv")
    rows = []
    with open(path) as handle:
        for row in csv.DictReader(handle):
            try:
                rows.append({
                    "t": float(row["t"]),
                    "raw_yaw": float(row["yaw"]),
                    "raw_pitch": float(row["pitch"]),
                    "aim_yaw": float(row["aim_yaw"]),
                    "aim_pitch": float(row["aim_pitch"]),
                    "armors": int(float(row["armors"])),
                    "state": int(float(row["tracker_state"])),
                    "mode": int(float(row["mode"])),
                })
            except (KeyError, TypeError, ValueError):
                continue
    return rows


def configYawSign():
    try:
        text = open("configs/serial.yaml").read()
        match = re.search(r"(?m)^\s*yaw_sign:\s*(-?[0-9.]+)", text)
        if match and float(match.group(1)) < 0.0:
            return -1.0
    except OSError:
        pass
    return 1.0


def wrap(angle):
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def trend(xs, ys):
    n = len(xs)
    if n < 3:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    denom = sum((x - mx) ** 2 for x in xs)
    if denom < 1e-12:
        return 0.0
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom


def corr(xs, ys):
    n = len(xs)
    if n < 3:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx < 1e-12 or syy < 1e-12:
        return 0.0
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(sxx * syy)


def report(name, rows, yaw_sign):
    # 只统计"有目标 + 在下发控制"的帧
    active = [r for r in rows if r["armors"] > 0 and r["mode"] > 0 and r["state"] in (1, 2)]
    if len(active) < 20:
        print(f"{name}: 有效帧太少（{len(active)}）——要录制时靶板可见、并且控制是开的")
        return

    times = [r["t"] for r in active]
    signed = []      # 目标 − 实际（带符号）
    errors = []      # |目标 − 实际|
    rates = []       # 编码器角速度（带符号）
    for index, row in enumerate(active):
        if name == "yaw":
            actual = yaw_sign * row["raw_yaw"]
            desired = row["aim_yaw"]
        else:
            actual = row["raw_pitch"]
            desired = row["aim_pitch"]
        signed.append(math.degrees(wrap(desired - actual)))
        errors.append(abs(signed[-1]))
        if index + 1 < len(active):
            dt = active[index + 1]["t"] - row["t"]
            rates.append((active[index + 1]["raw_pitch" if name == "pitch" else "raw_yaw"]
                          - row["raw_pitch" if name == "pitch" else "raw_yaw"]) / dt
                         if dt > 1e-6 else 0.0)
        else:
            rates.append(0.0)

    half = len(errors) // 2
    early = sum(errors[:half]) / max(1, half)
    late = sum(errors[half:]) / max(1, len(errors) - half)
    settled = sum(1 for e in errors if e < 2.0) / len(errors)
    slope = trend(times, errors)
    # 带符号误差 vs 编码器运动方向：同号 = 编码器朝目标走（收敛），异号 = 在远离（发散）
    align = corr(signed[:-1], rates[:-1])
    print(f"{name}: 帧 {len(errors)}；|err| 前半均值 {early:.2f}° 后半均值 {late:.2f}°；"
          f"斜率 {slope:+.3f} °/s；|err|<2° 占比 {100.0 * settled:.0f}%")
    # 主判据：|err| 是在涨、在收、还是停在一个常数上
    if late > max(2.0, 1.3 * max(early, 0.5)) and slope > 0.05:
        verdict = "**发散**：误差越追越大 → 这一轴的下发符号反了"
    elif late < 0.7 * max(early, 0.5) and slope < -0.05:
        verdict = "收敛：编码器在朝目标走"
    else:
        verdict = (f"停在常数误差上（{late:.1f}°）→ 不是符号问题，是"
                   "坐标系/零点差（或靶板本来就没在视轴上）")
    print(f"      主判据 → {verdict}")
    print(f"      参考 corr(带符号误差, 编码器角速度) = {align:+.3f}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    folder = sys.argv[1]
    yaw_sign = configYawSign()
    if "--yaw-sign" in sys.argv:
        yaw_sign = float(sys.argv[sys.argv.index("--yaw-sign") + 1])
    rows = readRows(folder)
    if not rows:
        print(f"读不到 {folder}/meta.csv")
        return 1
    print(f"帧 {len(rows)}，yaw_sign = {yaw_sign:+.0f}")
    report("yaw", rows, yaw_sign)
    report("pitch", rows, yaw_sign)
    return 0


if __name__ == "__main__":
    sys.exit(main())
