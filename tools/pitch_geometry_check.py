#!/usr/bin/env python3
"""从 `replay_dataset --dump aim.csv` 的逐帧表里判断**垂直几何**方向对不对。

背景（2026-09-29）：yaw 的符号当初是用"画面方位角 vs 回传 yaw 的相关系数"量出来的
（结论 yaw_sign=-1）；pitch 从来没这么量过，所以"板往下移、枪口却往上抬"到底是
①下发映射反 还是 ②流水线垂直几何（手眼外参 / pitch 符号）反 —— 需要分开判。

判据（全部来自同一段录制，云台不控制、人手动抬落 pitch）：
    θ = 靶板在**相机系里的竖直角度**（向上为正）= -atan2(obs_y, hypot(obs_x, obs_z))
    p = 当帧姿态 pitch（raw 回传）
    物理关系 k = dθ/dp     ← 由数据拟合（k<0 表示"p 增大 = 相机抬头"）
    流水线要求 Δ = aim_pitch - p 应该等于 **-θ/k**（把偏差消掉所需的动作）
    若 corr(Δ, -θ/k) < 0 → 流水线的垂直几何是**反的** ⇒ 该查外参/pitch 符号；
    若 > 0 → 流水线自己的方向自洽 ⇒ 反的是"下发映射/下位机闭环"那一侧。

用法：
    tools/pitch_geometry_check.py <aim.csv>
"""
import csv
import math
import sys


def load(path):
    rows = []
    with open(path) as handle:
        for row in csv.DictReader(handle):
            try:
                if float(row.get("obs_z_mm") or 0) <= 1.0:   # 该帧没有观测
                    continue
                rows.append({k: float(v) for k, v in row.items()})
            except (TypeError, ValueError):
                continue
    return rows


def slope(xs, ys):
    n = len(xs)
    if n < 3:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    denom = sum((x - mx) ** 2 for x in xs)
    if denom < 1e-9:
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
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return sxy / math.sqrt(sxx * syy)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    rows = load(sys.argv[1])
    if len(rows) < 20:
        print(f"有效观测帧太少（{len(rows)}）：录制里要有靶板、并且要手动抬落 pitch")
        return 1

    # 靶板的竖直角（向上为正）与姿态 pitch
    theta = []
    pitch = []
    aim_delta = []
    for row in rows:
        horizontal = math.hypot(row["obs_x_mm"], row["obs_z_mm"])
        theta.append(-math.degrees(math.atan2(row["obs_y_mm"], max(horizontal, 1.0))))
        p = row["pose_pitch_deg"] if abs(row["pose_pitch_deg"]) > 1e-9 else row["raw_pitch_deg"]
        pitch.append(p)
        aim_delta.append(row["aim_pitch_deg"] - p)

    k = slope(pitch, theta)                     # dθ/dp
    spread = max(pitch) - min(pitch)
    print(f"有效帧 {len(rows)}，pitch 变化范围 {spread:.1f}°")
    print(f"物理关系 dθ/dp = {k:+.3f} °/°  "
          f"（{'pitch 增大=抬头' if k < 0 else 'pitch 增大=低头' if k > 0 else '没变化'}）")
    print(f"靶板竖直角 θ：{min(theta):+.2f}° ~ {max(theta):+.2f}°")
    if abs(spread) < 3.0:
        print("⚠️ pitch 基本没动：这段录制量不出方向，请手动抬落 pitch 再录")
        return 1
    if abs(k) < 0.2:
        print("⚠️ dθ/dp 太小：观测噪声占主导，方向判不出来")
        return 1

    need = [-t / k for t in theta]              # 把偏差消掉所需的 Δ
    r = corr(aim_delta, need)
    print(f"corr(流水线要求的 Δ, 消除偏差所需的方向) = {r:+.3f}（样本 {len(aim_delta)}）")
    if r > 0.5:
        print("结论：**流水线垂直几何自洽**（方向对）⇒ 反的是下发映射/下位机闭环那一侧，"
              "改 configs/serial.yaml 的 command_*_pitch_sign。")
    elif r < -0.5:
        print("结论：**流水线垂直几何是反的**（外参/pitch 符号问题）⇒ 下发映射不用动；"
              "应给 pitch 加符号并重新解手眼外参（见 docs/auto_aim_aim_loop.md）。")
    else:
        print("结论：相关太弱，判不出来：确认这段里靶板一直在画面内、且 pitch 有来回动。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
