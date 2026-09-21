#!/usr/bin/env python3
"""角度域核账：把"提前量/瞄准"拆成可以在真值上对账的数字。

为什么必须用角度域
------------------
平移工况下"锁定的那块板"会进出视野、还会换板。位置域比较（estimate 的 xyz
直接比 truth 的 xyz）会因为换板而整块跳变 —— 之前用质心差分量出过
91/154/221 m/s 的假速度就是这么来的。角度域不会：四块板相隔 90°，
1~2° 的瞄准误差不可能认错板；就算换了板，也只是换到相邻的那块，
角速度方向本来就一致。

真值口径（关键，别搞错参考系）
------------------------------
truth.csv 是"板中心在**当前相机系**里的位置"，而仿真里的相机是被我们下发的
云台指令带着转的（simulator/src/main.rs 的 `remote_camera_gimbal`）。
selector.csv 的 target_yaw/target_pitch 是**home 系**角度（和云台指令同一个
参考系，因为估计器用 cmd_yaw/cmd_pitch 把观测反旋回 home 系）。
所以真值要先转回 home 系：

    yaw_home   = atan2(x, z) + cmd_yaw
    pitch_home = atan2(-y, hypot(x, z)) + cmd_pitch

这条关系是实测出来的，不是猜的：给仿真发 `GIMBAL +5deg 0`，靶面 apparent
yaw 立刻变成 -5deg；发 `GIMBAL 0 +5deg`，apparent pitch 变成 0.28-5=-4.7deg。

指标
----
* `pred`：selector 在 t 时刻预测的 (t+lead) 时刻瞄准角 vs 真值在 t+lead 时刻的
  最近板角 → Δyaw/Δpitch（deg）、横向误差（cm = Δyaw×距离）、
  "提前量欠量"（ms = -Δyaw/真值该板角速度）。
* `now`：同一时刻的指向残差（云台闭环把靶面压到光学轴的能力）。
* 占空：有 selector 行 / 有 observation 行的比例（丢靶会直接掉这个数）。
* 分档：sim.log 里的 `[move]` 行给出整车速度，把帧分成静止/平移两档。

注意 pitch 里的**弹道补偿**：selector 的 target_pitch 瞄的是
"抬 0.5·g·t_fly² 之后的瞄准点"，不是板中心。7.7 m / 25 m/s 时这一项
= 3.1~3.5deg（0.47 m），是**故意的**，不是误差。工具默认把这一项减掉，
`pitch(去弹道)` 才是真正要看的数；`--no-ballistic` 可以看原始值核对。

用法：
    tools/auto_aim_aim_check.py /tmp/trans1/uv0 [--label 平移1.5m/s] [--table]
"""

import argparse
import bisect
import csv
import math
import re
import statistics as st
import sys


def num(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_rows(path, columns):
    """读 csv，返回 [(time_us, (col...))] 按时间排序；缺列/坏行直接跳过。"""
    rows = []
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            t = num(row.get("time_us"))
            if t is None:
                continue
            values = tuple(num(row.get(c)) for c in columns)
            rows.append((int(t), values))
    rows.sort(key=lambda item: item[0])
    return rows


class Truth:
    """truth.csv：每帧四块板在相机系里的位置（plate_id 是仿真给的稳定编号）。"""

    def __init__(self, path):
        self.frames = {}          # time_us -> {plate_id: (x, y, z)}
        self.times = []
        with open(path, newline="") as handle:
            for row in csv.DictReader(handle):
                t = num(row.get("time_us"))
                pid = num(row.get("plate_id"))
                x, y, z = num(row.get("x")), num(row.get("y")), num(row.get("z"))
                if None in (t, pid, x, y, z):
                    continue
                self.frames.setdefault(int(t), {})[int(pid)] = (x, y, z)
        self.times = sorted(self.frames)

    def frame_at(self, t_us, tolerance_us=25_000):
        if not self.times:
            return None, None
        index = bisect.bisect_left(self.times, t_us)
        best = None
        for candidate in (index - 1, index, index + 1):
            if 0 <= candidate < len(self.times):
                delta = abs(self.times[candidate] - t_us)
                if best is None or delta < best[0]:
                    best = (delta, self.times[candidate])
        if best is None or best[0] > tolerance_us:
            return None, None
        return best[1], self.frames[best[1]]

    def rate(self, plate_id, t_us, span_us=40_000, commanded=None):
        """该板的角速度 (yaw_dot, pitch_dot) rad/s，±span 有限差分。

        传了 `commanded` 就返回 **home 系**角速度：apparent 角速度里混着
        "相机自己跟着云台指令转"的成分（实测能到 20~50 deg/s，比靶自己的运动
        大一个量级），不扣掉就会把云台的伺服动作算成"靶在动"。
        home 系角速度 = d(apparent + cmd)/dt，正好是预测器该复现的量。
        """
        before = self.frame_at(t_us - span_us)
        after = self.frame_at(t_us + span_us)
        if before[1] is None or after[1] is None:
            return None
        if plate_id not in before[1] or plate_id not in after[1]:
            return None
        dt = (after[0] - before[0]) / 1e6
        if not (0.005 < dt < 0.2):
            return None
        if commanded is not None:
            command_before = commanded(before[0])
            command_after = commanded(after[0])
            if command_before is None or command_after is None:
                return None
        else:
            command_before = command_after = (0.0, 0.0)

        def angles(point, command):
            x, y, z = point
            horizontal = math.hypot(x, z)
            return (math.atan2(x, z) + command[0],
                    math.atan2(-y, max(horizontal, 1e-9)) + command[1])

        y0, p0 = angles(before[1][plate_id], command_before)
        y1, p1 = angles(after[1][plate_id], command_after)
        return ((math.atan2(math.sin(y1 - y0), math.cos(y1 - y0))) / dt,
                (p1 - p0) / dt)

    def velocity(self, plate_id, t_us, span_us=40_000):
        """该板在相机系里的线速度 m/s（有限差分）。"""
        before = self.frame_at(t_us - span_us)
        after = self.frame_at(t_us + span_us)
        if before[1] is None or after[1] is None:
            return None
        if plate_id not in before[1] or plate_id not in after[1]:
            return None
        dt = (after[0] - before[0]) / 1e6
        if not (0.005 < dt < 0.2):
            return None
        return [(after[1][plate_id][k] - before[1][plate_id][k]) / dt for k in range(3)]


def parse_move_log(path):
    """sim.log 的 `[move] t=.. v=(x,y)` 行 → 绝对时间 → 速度大小（m/s）。"""
    entries = []
    try:
        with open(path, errors="replace") as handle:
            for line in handle:
                match = re.search(r"\[move\] t=([0-9.]+) v=\(([-0-9.]+),([-0-9.]+)\)", line)
                if match:
                    entries.append((float(match.group(1)), math.hypot(
                        float(match.group(2)), float(match.group(3)))))
    except FileNotFoundError:
        return []
    entries.sort()
    return entries


def speed_at(entries, t_us, epoch_us):
    """按仿真内部相对时间查速度（[move] 的 t 是仿真启动后的秒数）。"""
    if not entries:
        return None
    elapsed = (t_us - epoch_us) / 1e6
    index = bisect.bisect_right([e[0] for e in entries], elapsed) - 1
    if index < 0:
        return entries[0][1]
    return entries[index][1]


def fmt(values, unit="", scale=1.0, digits=2):
    if not values:
        return "n=0"
    ordered = sorted(v * scale for v in values)
    return (f"n={len(ordered):<4} 中位 {st.median(ordered):+.{digits}f} "
            f"p10 {ordered[int(0.1 * (len(ordered) - 1))]:+.{digits}f} "
            f"p90 {ordered[int(0.9 * (len(ordered) - 1))]:+.{digits}f}{unit}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", help="<out>/uv0 这种目录")
    parser.add_argument("--label", default="")
    parser.add_argument("--table", action="store_true", help="打印每秒明细")
    parser.add_argument("--projectile-speed", type=float, default=25.0)
    parser.add_argument("--no-ballistic", action="store_true",
                        help="不做弹道补偿扣除（看原始 pitch 残差）")
    args = parser.parse_args()
    GRAVITY = 9.81

    def ballistic_pitch(distance):
        """aim_point 比板中心高 0.5·g·t_fly²，换算成俯仰角（度）。"""
        if args.no_ballistic or args.projectile_speed <= 1e-6:
            return 0.0
        flight = distance / args.projectile_speed
        drop = 0.5 * GRAVITY * flight * flight
        return math.degrees(math.atan2(drop, distance))

    base = args.run_dir.rstrip("/")
    truth = Truth(f"{base}/truth.csv")
    if not truth.times:
        print(f"{base}: truth.csv 是空的", file=sys.stderr)
        return 2
    selectors = load_rows(f"{base}/selector.csv", (
        "target_yaw", "target_pitch", "lead", "armor_id", "in_fire"))
    observations = load_rows(f"{base}/observation.csv", (
        "cmd_yaw", "cmd_pitch", "state"))
    cmd_times = [row[0] for row in observations]
    epoch_us = truth.times[0]
    moves = parse_move_log(f"{base}/sim.log")

    def commanded(t_us):
        """该时刻的云台指令角（home 系）；用最近的前后两帧里更近的那个。"""
        if not observations:
            return None
        index = bisect.bisect_left(cmd_times, t_us)
        best = None
        for candidate in (index - 1, index):
            if 0 <= candidate < len(observations):
                delta = abs(cmd_times[candidate] - t_us)
                if best is None or delta < best[0]:
                    best = (delta, observations[candidate][1])
        if best is None or best[0] > 100_000:
            return None
        return best[1]

    def home_angles(plate_point, command):
        """相机系真值 → home 系绝对角（与 selector 的 target_yaw 同参考系）。"""
        x, y, z = plate_point
        cmd_yaw, cmd_pitch = command[0], command[1]
        yaw = math.atan2(x, z) + cmd_yaw
        pitch = math.atan2(-y, max(math.hypot(x, z), 1e-9)) + cmd_pitch
        return yaw, pitch

    def angle_gap(a, b):
        return math.atan2(math.sin(a - b), math.cos(a - b))

    pred_yaw, pred_pitch, point_yaw, point_pitch = [], [], [], []
    pred_lateral_cm, deficits_ms, distances = [], [], []
    rates_deg, residual_perp_deg = [], []
    rows_out = []
    for t_us, (target_yaw, target_pitch, lead, armor_id, in_fire) in selectors:
        if None in (target_yaw, target_pitch, lead):
            continue
        # now：同一时刻的指向残差（闭环把靶面压到光学轴的能力）
        command_now = commanded(t_us)
        frame_now = truth.frame_at(t_us)
        if command_now and frame_now[1]:
            best = None
            for pid, point in frame_now[1].items():
                yaw, pitch = home_angles(point, command_now)
                gap = abs(angle_gap(yaw, target_yaw)) + abs(pitch - target_pitch)
                if best is None or gap < best[0]:
                    best = (gap, yaw, pitch, pid, math.sqrt(sum(v * v for v in point)))
            if best:
                point_yaw.append(math.degrees(angle_gap(target_yaw, best[1])))
                point_pitch.append(math.degrees(target_pitch - best[2])
                                   - ballistic_pitch(best[4]))

        if not (0.005 < lead < 0.6):
            continue
        t_future = t_us + int(lead * 1e6)
        command_future = commanded(t_future)
        frame_future = truth.frame_at(t_future)
        if not command_future or not frame_future[1]:
            continue
        best = None
        for pid, point in frame_future[1].items():
            yaw, pitch = home_angles(point, command_future)
            gap = abs(angle_gap(yaw, target_yaw)) + abs(pitch - target_pitch)
            if best is None or gap < best[0]:
                best = (gap, yaw, pitch, pid)
        if best is None:
            continue
        _, yaw_true, pitch_true, pid = best
        delta_yaw = angle_gap(target_yaw, yaw_true)
        distance = math.sqrt(sum(v * v for v in frame_future[1][pid]))
        delta_pitch = (target_pitch - pitch_true
                       - math.radians(ballistic_pitch(distance)))
        delta_yaw_deg = math.degrees(delta_yaw)
        delta_pitch_deg = math.degrees(delta_pitch)
        pred_yaw.append(delta_yaw_deg)
        pred_pitch.append(delta_pitch_deg)
        pred_lateral_cm.append(delta_yaw * distance * 100.0)
        distances.append(distance)

        # 把误差投影到"真值该板此刻的角运动方向"上：
        #   分量① 平行于运动 → 提前量欠量（ms）
        #   分量② 垂直于运动 → 真正的指向偏置（不随时间演化，滤波改不了）
        rate = truth.rate(pid, t_future, commanded=commanded)
        deficit = None
        residual_perp = None
        rate_deg = None
        if rate:
            omega_yaw, omega_pitch = math.degrees(rate[0]), math.degrees(rate[1])
            rate_deg = math.hypot(omega_yaw, omega_pitch)
            norm_sq = omega_yaw * omega_yaw + omega_pitch * omega_pitch
            if norm_sq > 1e-9:
                projected = (delta_yaw_deg * omega_yaw + delta_pitch_deg * omega_pitch) / norm_sq
                deficit = -projected * 1000.0
                residual_perp = math.hypot(delta_yaw_deg - projected * omega_yaw,
                                           delta_pitch_deg - projected * omega_pitch)
        rows_out.append({
            "t_us": t_us, "speed": speed_at(moves, t_us, epoch_us),
            "dyaw": delta_yaw_deg, "dpitch": delta_pitch_deg,
            "lateral_cm": delta_yaw * distance * 100.0, "distance": distance,
            "deficit_ms": deficit, "perp_deg": residual_perp,
            "rate_deg": rate_deg, "armor_id": armor_id, "in_fire": in_fire,
        })
        if rate_deg is not None:
            rates_deg.append(rate_deg)
        if deficit is not None and abs(deficit) < 600 and rate_deg and rate_deg > 1.0:
            deficits_ms.append(deficit)
        if residual_perp is not None and rate_deg and rate_deg > 1.0:
            residual_perp_deg.append(residual_perp)

    label = args.label or base
    print(f"=== {label} ===")
    span = (truth.times[-1] - truth.times[0]) / 1e6
    print(f"真值帧 {len(truth.times)} 跨度 {span:.1f}s | selector 行 {len(selectors)} "
          f"| observation 行 {len(observations)}")
    if selectors:
        covered = (selectors[-1][0] - selectors[0][0]) / 1e6
        print(f"有决策的时长 {covered:.1f}s / 真值 {span:.1f}s "
              f"= {100.0 * covered / max(span, 1e-6):.0f}%")
    print(f"指向残差 now : yaw {fmt(point_yaw, 'deg', digits=2)} | "
          f"pitch {fmt(point_pitch, 'deg', digits=2)}")
    print(f"预测残差 pred: yaw {fmt(pred_yaw, 'deg', digits=2)} | "
          f"pitch {fmt(pred_pitch, 'deg', digits=2)}")
    if distances:
        print(f"横向误差     : {fmt(pred_lateral_cm, 'cm', digits=1)} | "
              f"距离中位 {st.median(distances):.2f}m")
    else:
        print("横向误差     : n=0")
    print(f"提前量欠量   : {fmt(deficits_ms, 'ms', digits=0)} "
          f"(仅 |ω|>1deg/s 的行)")
    print(f"指向偏置(⊥)  : {fmt(residual_perp_deg, 'deg', digits=2)} "
          f"(扣掉提前量后剩下的部分)")
    if rates_deg:
        print(f"真值板角速度 : 中位 {st.median(rates_deg):.2f} p90 "
              f"{sorted(rates_deg)[int(0.9 * (len(rates_deg) - 1))]:.2f} deg/s")

    if moves and rows_out:
        still = [r for r in rows_out if r["speed"] is not None and r["speed"] < 0.2]
        moving = [r for r in rows_out if r["speed"] is not None and r["speed"] >= 0.2]
        print(f"分档(仿真整车速度): 静止 {len(still)} 帧 | 平移 {len(moving)} 帧")
        for name, group in (("静止", still), ("平移", moving)):
            if not group:
                continue
            lag = [r["deficit_ms"] for r in group
                   if r["deficit_ms"] is not None and r["rate_deg"] and r["rate_deg"] > 1.0]
            lat = [r["lateral_cm"] for r in group]
            print(f"  {name}: 横向中位 {st.median(lat):+.1f}cm"
                  + (f" | 欠量中位 {st.median(lag):+.0f}ms (n={len(lag)})" if lag else ""))

    if args.table and rows_out:
        print(" t(s)     v(m/s)  Δyaw(deg)  横向(cm)  欠量(ms)  ⊥(deg) |ω|(deg/s) fire")
        for row in rows_out[::max(1, len(rows_out) // 40)]:
            speed = row["speed"]
            deficit = row["deficit_ms"]
            perp = "--" if row["perp_deg"] is None else f"{row['perp_deg']:.2f}"
            rate = "--" if row["rate_deg"] is None else f"{row['rate_deg']:.2f}"
            lag = "--" if deficit is None else f"{deficit:+.0f}"
            print(f"{(row['t_us'] - epoch_us) / 1e6:7.2f}"
                  f" {speed if speed is not None else float('nan'):7.2f}"
                  f" {row['dyaw']:+10.2f} {row['lateral_cm']:+9.1f}"
                  f" {lag:>8} {perp:>8} {rate:>9} {row['in_fire']:4.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
