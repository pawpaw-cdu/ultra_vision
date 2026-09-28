#!/usr/bin/env python3
"""差分法探针：在静态机位录制帧上，量"帧间差分找点亮扇叶"的可用性。

思路（很多队伍在实车上的做法）：
    相机静止/近似静止时，**静止的亮区（地面反光等）在帧间差里会相互抵消**，
    只有"又亮又在动"的那片扇叶会留下明显的差分量 ⇒ 用帧间差 + 阈值 + 连通域
    就能把点亮扇叶拎出来，作为网络检测的补充（网络漏检的帧它可能还在）。

本脚本做两件事：
1. 与网络（geom_compare 输出的 NN 检测）在**同一批帧**上对比：谁在哪一帧找到了点亮片；
2. 对"网络没检出、差分检出了"的帧做**运动学校验**：相机静止 + 圆心/半径已知 +
   小符角速度固定 60°/s ⇒ 由最近一帧 NN 的位置外推，检查差分给出的方位是否吻合
   （吻合才说明差分找到的是真的那片，而不是噪声/反光）。

用法：
    python3 tools/rune_diff_probe.py --frames <dir> --nn <nn.csv> [--mode small]
"""

import argparse
import csv
import math
import statistics
from pathlib import Path

import cv2
import numpy as np

K_SMALL = math.pi / 3.0      # 小符固定角速度 rad/s
ORBIT_PX = 68.0              # 半径 0.7 m @ ~6 m，f=579.4


def load_frames(directory, pattern):
    frames = []
    for path in sorted(Path(directory).glob(pattern)):
        name = path.stem
        try:
            ts = int(name.rsplit("_", 1)[1]) * 1e-6
        except (ValueError, IndexError):
            continue
        frames.append((ts, path))
    frames.sort()
    return frames


def load_nn(path):
    """geom_compare 的 CSV：一行一条候选；我们只取 best（is_best=1）的靶心位置。"""
    detections = {}
    for row in csv.DictReader(open(path)):
        if row.get("is_best") != "1":
            continue
        ts = float(row["t_sim_us"]) * 1e-6
        detections[round(ts, 4)] = (
            float(row["ours_plate_x"]),
            float(row["ours_plate_y"]),
            float(row["ours_k2_x"]),
            float(row["ours_k2_y"]),
        )
    return detections


def diff_candidates(prev, cur, hub, threshold, orbit=ORBIT_PX, compensate=False):
    """返回 (候选中心列表)。候选 = 差分连通域里"落在轨道环带内"的。"""
    g0 = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    g1 = cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY)
    if compensate:
        # 云台在动时，先用相位相关估计全局平移并补偿，再差分；否则差分会把整幅
        # 背景都算成变化（实拍实测：blob 面积 15k~110k px²，与网络差 ~190 px）。
        (dx, dy), response = cv2.phaseCorrelate(np.float32(g0), np.float32(g1))
        if abs(dx) < 60 and abs(dy) < 60:
            matrix = np.float32([[1, 0, -dx], [0, 1, -dy]])
            g0 = cv2.warpAffine(g0, matrix, (g1.shape[1], g1.shape[0]),
                                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    diff = cv2.absdiff(g1, g0)
    diff = cv2.GaussianBlur(diff, (5, 5), 0)
    _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    count, _, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    candidates = []
    for index in range(1, count):  # 0 是背景
        area = float(stats[index, cv2.CC_STAT_AREA])
        if area < 6.0:
            continue
        cx, cy = float(centroids[index][0]), float(centroids[index][1])
        radius = math.dist((cx, cy), hub)
        if not (0.4 * orbit <= radius <= 2.5 * orbit):
            continue  # 不在轨道环带上（贴地面反光/机构中心等）
        candidates.append((cx, cy, area, radius))
    candidates.sort(key=lambda c: -c[2])
    return candidates


def angle_of(point, hub):
    return math.atan2(point[1] - hub[1], point[0] - hub[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", required=True)
    parser.add_argument("--nn", required=True)
    parser.add_argument("--pattern", default="raw_*.png")
    parser.add_argument("--mode", default="small", choices=["small", "large"])
    parser.add_argument("--threshold", type=float, default=20.0)
    parser.add_argument("--validate-gap", type=float, default=0.2,
                        help="运动学校验允许的最大外推间隔（秒）；超过就不算数")
    parser.add_argument("--hub-gap", type=float, default=0.15,
                        help="用最近一帧网络的 k2 当圆心时，允许的最大间隔（秒）。"
                             "实拍里云台在动，圆心必须逐帧给；超过该间隔的帧跳过")
    parser.add_argument("--motion-compensate", action="store_true",
                        help="差分前先做全局平移补偿（云台在动的实拍素材需要）")
    args = parser.parse_args()

    frames = load_frames(args.frames, args.pattern)
    nn = load_nn(args.nn)
    print(f"帧数 {len(frames)}，网络检出帧 {len(nn)}（{100*len(nn)/max(1,len(frames)):.0f}%）")
    if len(frames) < 3:
        return 1

    # 圆心：优先"逐帧用最近一帧网络的 k2"（实拍云台在动），否则退回全帧中位数（静态机位）。
    nn_times_all = sorted(nn)
    static_hub = None
    if nn:
        static_hub = (statistics.median(v[2] for v in nn.values()),
                      statistics.median(v[3] for v in nn.values()))
    orbit = ORBIT_PX
    if nn:
        radii = [math.dist((v[0], v[1]), (v[2], v[3])) for v in nn.values()]
        if radii:
            orbit = statistics.median(radii)
    print(f"静态圆心参考 = ({static_hub[0]:.1f},{static_hub[1]:.1f})  轨道半径中位 {orbit:.1f}px"
          if static_hub else "无网络参考")

    def hub_for(t):
        """返回 (hub, 是否逐帧参考)；超过 hub_gap 就返回 None。"""
        if not nn_times_all:
            return None, False
        ref = min(nn_times_all, key=lambda x: abs(x - t))
        if abs(ref - t) <= args.hub_gap:
            return (nn[ref][2], nn[ref][3]), True
        return None, False

    prev = None
    rows = []
    for ts, path in frames:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        if prev is not None and prev.shape == image.shape:
            hub, fresh = hub_for(ts)
            if hub is None:
                prev = image
                continue
            cands = diff_candidates(prev, image, hub, args.threshold, orbit,
                                    args.motion_compensate)
            best = cands[0] if cands else None
            rows.append({"t": ts, "cand": best, "n": len(cands)})
        prev = image
    n_ok = sum(1 for r in rows if r["cand"] is not None)
    print(f"差分检出帧 {n_ok}/{len(rows)}（{100*n_ok/max(1,len(rows)):.0f}%）")

    # 与网络对比（按时间最近匹配，±20 ms）
    nn_times = sorted(nn)
    agree, both, diff_only, nn_only = [], 0, 0, 0
    validated = 0
    checked = 0
    for row in rows:
        t = row["t"]
        nearest = min(nn_times, key=lambda x: abs(x - t)) if nn_times else None
        has_nn = nearest is not None and abs(nearest - t) <= 0.02
        if has_nn and row["cand"] is not None:
            both += 1
            d = math.dist((row["cand"][0], row["cand"][1]), nn[nearest][:2])
            agree.append(d)
        elif has_nn:
            nn_only += 1
        elif row["cand"] is not None:
            diff_only += 1
            # 运动学校验：用最近一帧 NN 的位置 + 固定角速度外推到本帧
            if nn_times:
                ref = min(nn_times, key=lambda x: abs(x - t))
                dt = t - ref
                hub_now, _ = hub_for(t)
                if abs(dt) <= args.validate_gap and hub_now is not None:
                    hub = hub_now
                    checked += 1
                    base = nn[ref]
                    omega = (1.0 if args.mode == "small" else 1.0) * K_SMALL
                    # 旋向：用参考帧附近 NN 的相位变化判断
                    later = [x for x in nn_times if x > ref]
                    sign = 1.0
                    if later:
                        nxt = later[0]
                        if nxt - ref < 0.6:
                            a0 = angle_of((base[0], base[1]), hub)
                            a1 = angle_of((nn[nxt][0], nn[nxt][1]), hub)
                            dd = (a1 - a0 + math.pi) % (2 * math.pi) - math.pi
                            sign = 1.0 if dd > 0 else -1.0
                    predicted = (hub[0] + orbit * math.cos(angle_of((base[0], base[1]), hub)
                                                              + sign * omega * dt),
                                 hub[1] + orbit * math.sin(angle_of((base[0], base[1]), hub)
                                                              + sign * omega * dt))
                    err = math.dist((row["cand"][0], row["cand"][1]), predicted)
                    if err < 20.0:
                        validated += 1

    print(f"\n对比（阈值 {args.threshold}）：")
    print(f"  两者都有 {both}   仅网络有 {nn_only}   仅差分有 {diff_only}")
    if agree:
        print(f"  两者位置差：中位 {statistics.median(agree):.1f} px "
              f"p90 {sorted(agree)[int(len(agree)*0.9)-1]:.1f} px")
    if checked:
        print(f"  仅差分帧的运动学校验：{validated}/{checked} 落在预测位置 20 px 内"
              f"（{100*validated/checked:.0f}%）")
    both_areas = [r["cand"][2] for r in rows
                  if r["cand"] is not None and nn_times and
                  min(abs(x - r["t"]) for x in nn_times) <= 0.02]
    only_areas = [r["cand"][2] for r in rows
                  if r["cand"] is not None and nn_times and
                  min(abs(x - r["t"]) for x in nn_times) > 0.02]
    if both_areas and only_areas:
        print(f"  差分连通域面积：网络也检出的帧 中位 {statistics.median(both_areas):.0f} px²，"
              f"仅差分的帧 中位 {statistics.median(only_areas):.0f} px²")
    print(f"  合并后可用帧（两者取并集）≈ {both + nn_only + diff_only} / {len(rows)}"
          f"（{100*(both+nn_only+diff_only)/max(1,len(rows)):.0f}%）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
