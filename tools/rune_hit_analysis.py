#!/usr/bin/env python3
"""命中归因：把"打中了但没计分"拆成可执行的原因。

用法：
    python3 tools/rune_hit_analysis.py <run_dir>

输入：
    <run_dir>/vision.csv         视觉端逐帧 CSV（含 aim/lit/picked/亮度/slot 等诊断列）
    <run_dir>/ground_truth.csv   仿真遥测（RUNE event=hit/scored/highlighted）

输出：每次命中一行的归因 + 汇总。分类规则（按优先级）：

    A 无有效目标就开火      : 击发时刻 target_valid=0（估计器无解，瞄的是滑行/符心）
    B 锁定暗片（选片错）    : picked_bright 明显低于同帧最亮候选 cand_bright
    C 瞄点偏离锁定片        : |aim_px - picked_px| 超过 LATERAL_PX
    D 瞄点正确但仍未计分    : 以上都不是 → 剩下的只可能是弹道/命中时刻的时间错

其中 B 会同时给出"最亮片（=点亮片）离我们锁定片的像素距离"，用于判断下一次
该改选片还是改提前量。
"""

import csv
import math
import statistics
import sys
from pathlib import Path

# 判定阈值（都用像素/亮度原始量，避免引入角度换算的额外假设）
BRIGHT_GAP_FOR_DARK_PICK = 30.0   # picked 比最亮暗这么多 → 认为锁到了暗片
RADIAL_TOL_PX = 25.0              # 瞄点相对锁定片的"径向"偏差上限（换片会体现为径向/大角度偏差）
TANGENTIAL_TOL_PX = 40.0          # 切向偏差（扣掉提前量后）上限


def load_vision(path):
    rows = []
    for row in csv.DictReader(open(path)):
        try:
            row["_t"] = float(row["sim_t"])
        except (KeyError, ValueError):
            continue
        rows.append(row)
    rows.sort(key=lambda r: r["_t"])
    return rows


def load_hits(path):
    hits = []
    for line in open(path):
        if "event=hit" not in line:
            continue
        fields = dict(part.split("=", 1) for part in line.split() if "=" in part)
        hits.append(
            {
                "t": float(fields["t"]),
                "index": int(fields["hit"]),
                "accurate": fields.get("accurate", "0") == "1",
                "scored": fields.get("scored", "0") == "1",
                "phase": fields.get("phase", ""),
                "highlighted": fields.get("highlighted", "-"),
                "mode": fields.get("mode", ""),
            }
        )
    return hits


def nearest(rows, t):
    best = None
    for row in rows:
        if best is None or abs(row["_t"] - t) < abs(best["_t"] - t):
            best = row
    return best


def fire_frame(rows, hit_time):
    """命中时刻往前推一个飞行时间，得到"击发那一帧"（用上一帧记的 fly_time）。"""
    before = [r for r in rows if r["_t"] <= hit_time]
    if not before:
        return None
    fly = 0.25
    try:
        fly = float(before[-1]["fly_time"]) or 0.25
    except (KeyError, ValueError):
        pass
    return nearest(rows, hit_time - fly)


def motion_hat(rows, row):
    """用图像里相邻有效帧的靶心位移方向估计"运动方向"（避免依赖 roll/spd 的符号约定）。"""
    try:
        index = rows.index(row)
    except ValueError:
        return None
    for offset in range(1, 8):
        for neighbor in (rows[index - offset] if index - offset >= 0 else None,
                         rows[index + offset] if index + offset < len(rows) else None):
            if neighbor is None or neighbor.get("target_valid") != "1":
                continue
            lx = float(row.get("picked_px") or 0.0)
            ly = float(row.get("picked_py") or 0.0)
            nx = float(neighbor.get("picked_px") or 0.0)
            ny = float(neighbor.get("picked_py") or 0.0)
            if lx == 0.0 or nx == 0.0:
                continue
            dx, dy = nx - lx, ny - ly
            norm = math.hypot(dx, dy)
            if norm > 2.0:
                return (dx / norm, dy / norm)
    return None


def classify(row, rows):
    if row is None:
        return "?", "no csv row"
    if row.get("target_valid") != "1":
        return "A 无有效目标就开火", "target_valid=0"
    picked = float(row.get("picked_bright") or 0.0)
    cand = float(row.get("cand_bright") or 0.0)
    if cand > 0 and cand - picked > BRIGHT_GAP_FOR_DARK_PICK:
        lit = (float(row.get("lit_px") or 0.0), float(row.get("lit_py") or 0.0))
        lock = (float(row.get("picked_px") or 0.0), float(row.get("picked_py") or 0.0))
        gap = math.dist(lit, lock) if lit != (0.0, 0.0) and lock != (0.0, 0.0) else -1.0
        return "B 锁定暗片(选片错)", f"picked={picked:.0f} 最亮={cand:.0f} 距离={gap:.0f}px"

    aim = (float(row.get("aim_px") or 0.0), float(row.get("aim_py") or 0.0))
    lock = (float(row.get("picked_px") or 0.0), float(row.get("picked_py") or 0.0))
    hub = (float(row.get("r_center_x") or 0.0), float(row.get("r_center_y") or 0.0))
    if aim == (0.0, 0.0) or lock == (0.0, 0.0):
        return "D 瞄点正确仍未计分", "aim/lock 缺失"

    # 把"瞄点 - 锁定片"分解成径向（是否同一片/同一圈）与切向（提前量是否合理）。
    radial = (lock[0] - hub[0], lock[1] - hub[1])
    r_norm = math.hypot(*radial)
    if r_norm < 1.0:
        return "D 瞄点正确仍未计分", "hub 无效"
    r_hat = (radial[0] / r_norm, radial[1] / r_norm)
    t_hat = (-r_hat[1], r_hat[0])
    delta = (aim[0] - lock[0], aim[1] - lock[1])
    radial_err = delta[0] * r_hat[0] + delta[1] * r_hat[1]
    tangential = delta[0] * t_hat[0] + delta[1] * t_hat[1]

    # 理论提前量幅值：|spd| × 飞行时间 × 像素半径；方向取图像里的真实运动方向。
    try:
        spd = float(row.get("spd") or 0.0)
        fly = float(row.get("fly_time") or 0.0)
    except ValueError:
        spd, fly = 0.0, 0.0
    lead_expected = abs(spd) * fly * r_norm
    m_hat = motion_hat(rows, row)
    if m_hat is None:
        return "D 瞄点正确仍未计分", "无相邻帧可判运动方向"
    along_motion = delta[0] * m_hat[0] + delta[1] * m_hat[1]
    lateral = delta[0] * (-m_hat[1]) + delta[1] * m_hat[0]
    lead_err = along_motion - lead_expected

    if abs(radial_err) > RADIAL_TOL_PX:
        return "C 瞄到别的片/别的圈", (
            f"径向偏差={radial_err:.0f}px 沿运动={along_motion:.0f}px 半径={r_norm:.0f}px")
    if abs(lateral) > TANGENTIAL_TOL_PX:
        return "C 瞄到别的片/别的圈", (
            f"侧向={lateral:.0f}px 沿运动={along_motion:.0f}px 理论提前量={lead_expected:.0f}px")
    if abs(lead_err) > TANGENTIAL_TOL_PX:
        return "E 提前量/时间错", (
            f"沿运动={along_motion:.0f}px 理论提前量={lead_expected:.0f}px 残差={lead_err:.0f}px"
            f"(≈{lead_err / max(1e-3, abs(spd) * r_norm):.2f}s)")
    return "D 瞄点正确仍未计分", (
        f"沿运动={along_motion:.0f}px 理论={lead_expected:.0f}px 侧向={lateral:.0f}px")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    run_dir = Path(sys.argv[1])
    rows = load_vision(run_dir / "vision.csv")
    hits = load_hits(run_dir / "ground_truth.csv")
    if not rows or not hits:
        print(f"没有数据：rows={len(rows)} hits={len(hits)}")
        return 1

    # 同一次开火可能在一个极短时间窗里产生多条 hit 事件（弹丸继续撞到别的模块），
    # 先把它们按 0.05 s 归并成"一次开火"，否则会把"一发弹打中两片"算成两发。
    shots = []
    for hit in sorted(hits, key=lambda h: h["t"]):
        if shots and hit["t"] - shots[-1]["t"] <= 0.05:
            shots[-1]["scored"] = shots[-1]["scored"] or hit["scored"]
            shots[-1]["indexes"].append(hit["index"])
        else:
            shots.append(
                {
                    "t": hit["t"],
                    "scored": hit["scored"],
                    "indexes": [hit["index"]],
                    "phase": hit["phase"],
                    "highlighted": hit["highlighted"],
                }
            )

    print(f"命中事件 {len(hits)} 条 → 归并为 {len(shots)} 次开火")
    counts = {}
    print(f"{'命中时刻':>8} {'hit':>4} {'计分':>4} {'phase':>10} {'点亮片':>7} "
          f"{'slot':>4} {'target_valid':>12} {'分类':>22} 说明")
    for shot in shots:
        hit = shot
        row = fire_frame(rows, hit["t"])
        label, note = classify(row, rows)
        counts[label] = counts.get(label, 0) + 1
        print(f"{hit['t']:8.2f} {str(shot['indexes']):>4} {int(hit['scored']):>4} {hit['phase']:>10} "
              f"{hit['highlighted']:>7} {row.get('slot_id','?') if row else '?':>4} "
              f"{row.get('target_valid','?') if row else '?':>12} {label:>22} {note}")

    print("\n汇总：")
    total = sum(counts.values())
    for label in sorted(counts):
        print(f"  {label:24s} {counts[label]:3d} / {total}")
    scored = sum(1 for h in shots if h["scored"])
    print(f"  计分命中 {scored}/{total} ({100*scored/max(1,total):.0f}%)")

    # 逐发开火归因：这是"每发弹为什么没进"的主口径（命中事件只覆盖打中的弹）。
    fired_rows = [r for r in rows if r.get("fired") == "1"]
    if fired_rows:
        print(f"\n逐发归因（fired=1 共 {len(fired_rows)} 发）：")
        hit_times = [h["t"] for h in hits]
        for r in fired_rows:
            t = r["_t"]
            follow = [h for h in hits if 0.05 < h["t"] - t < 0.6]
            outcome = "miss(完全没碰到机关)"
            if follow:
                outcome = "scored" if any(h["scored"] for h in follow) else "hit-未计分"
            note = ""
            if r.get("target_valid") != "1":
                note = "发火时无有效目标"
            else:
                cand = float(r.get("cand_bright") or 0.0)
                picked = float(r.get("picked_bright") or 0.0)
                if cand > 0 and cand - picked > BRIGHT_GAP_FOR_DARK_PICK:
                    note = "锁的是暗片"
            print(f"  t={t:6.2f} slot={r.get('slot_id','?'):>2} target_valid={r.get('target_valid','?')} "
                  f"|aim-lit|={math.dist((float(r.get('aim_px') or 0), float(r.get('aim_py') or 0)), (float(r.get('lit_px') or 0), float(r.get('lit_py') or 0))):6.1f}px "
                  f"-> {outcome} {note}")

    # 瞄点 vs 锁定片 / 锁定片 vs 最亮片 的分布（只统计有效目标帧）
    aim_lock, lock_lit = [], []
    for row in rows:
        if row.get("target_valid") != "1":
            continue
        aim = (float(row.get("aim_px") or 0.0), float(row.get("aim_py") or 0.0))
        lock = (float(row.get("picked_px") or 0.0), float(row.get("picked_py") or 0.0))
        lit = (float(row.get("lit_px") or 0.0), float(row.get("lit_py") or 0.0))
        if aim != (0.0, 0.0) and lock != (0.0, 0.0):
            aim_lock.append(math.dist(aim, lock))
        if lock != (0.0, 0.0) and lit != (0.0, 0.0):
            lock_lit.append(math.dist(lock, lit))
    if aim_lock:
        print(f"\n有效目标帧：|瞄点-锁定片| 中位 {statistics.median(aim_lock):.1f} px "
              f"p90 {sorted(aim_lock)[int(len(aim_lock)*0.9)-1]:.1f} px")
    if lock_lit:
        print(f"有效目标帧：|锁定片-最亮片| 中位 {statistics.median(lock_lit):.1f} px "
              f"p90 {sorted(lock_lit)[int(len(lock_lit)*0.9)-1]:.1f} px "
              f"（>30px 占 {100*sum(1 for x in lock_lit if x>30)/len(lock_lit):.0f}%）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
