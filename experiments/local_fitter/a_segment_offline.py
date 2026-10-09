#!/usr/bin/env python3
"""A 段离线验证：检测流 -> 单物理板「正面弧段」-> 模式A 拟合 ω。

输入：仿真器导出的 ULTRA_VISION_DETECTIONS_CSV（每帧**全部**装甲板检测：
      time_us,source_time_us,idx,u_lt,v_lt,u_lb,v_lb,u_rt,v_rt,u_rb,v_rb,x,y,z）
      + 可选 truth.csv（每帧 4 块板真值，用于核对身份）。

A1 身份：把检测按"最近邻+常速预测"关联成航迹（track=一块物理板）。单帧 PnP 的
        yaw 无法唯一确定板号（4 块板相差 90°，四重简并），所以这里用**时序一致
        的航迹**当物理身份；绝对板号由 EKF 的预测 yaw 才能定。
A2 分割：同一航迹内，帧间隔 > gap 即断段。
A3 门控：只留"正对相机"的帧：图像两灯条中心距 sep > gate·W·fx/depth
        （sep/sep_max = cos(板相对相机偏角)）。
拟合：每段 >= min_frames 帧，用与 LocalTrajectoryFitter 完全相同的刚体模型
      （θ=θ0+ω·dt，C,h,θ0,ω,漂移）做 LM，输出 rmse 与 ω。

用法：
  a_segment_offline.py --det det.csv [--truth truth.csv] [--png out.png]
                       [--R 0.21] [--W 0.1272] [--L 0.0598] [--gate 0.5] [--off -3.22]
"""
import argparse, collections, csv, math
import numpy as np


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", required=True)
    ap.add_argument("--truth", default="")
    ap.add_argument("--png", default="")
    ap.add_argument("--fx", type=float, default=1447.0)
    ap.add_argument("--fy", type=float, default=1447.0)
    ap.add_argument("--cx", type=float, default=720.0)
    ap.add_argument("--cy", type=float, default=540.0)
    ap.add_argument("--R", type=float, default=0.21)
    ap.add_argument("--W", type=float, default=0.1272)
    ap.add_argument("--L", type=float, default=0.0598)
    ap.add_argument("--gate", type=float, default=0.5)
    ap.add_argument("--off", type=float, default=-3.22, help="det.source_time 相对 truth 时间的偏移(s)")
    ap.add_argument("--track_gate", type=float, default=90.0, help="A1 关联门限(px)")
    ap.add_argument("--max_gap", type=float, default=0.15, help="断段间隔(s)")
    ap.add_argument("--min_frames", type=int, default=6)
    ap.add_argument("--zmin", type=float, default=2.0)
    ap.add_argument("--zmax", type=float, default=4.0)
    return ap.parse_args()


def load_dets(a):
    D = []
    for r in csv.DictReader(open(a.det)):
        z = float(r["z"])
        if not (a.zmin < z < a.zmax):
            continue
        lt = (float(r["u_lt"]), float(r["v_lt"]))
        lb = (float(r["u_lb"]), float(r["v_lb"]))
        rt = (float(r["u_rt"]), float(r["v_rt"]))
        rb = (float(r["u_rb"]), float(r["v_rb"]))
        D.append(dict(t=int(r["source_time_us"]) * 1e-6, lt=lt, lb=lb, rt=rt, rb=rb,
                      u=(lt[0] + lb[0] + rt[0] + rb[0]) / 4,
                      v=(lt[1] + lb[1] + rt[1] + rb[1]) / 4,
                      sep=(rt[0] + rb[0]) / 2 - (lt[0] + lb[0]) / 2,
                      z=z, x=float(r["x"]), y=float(r["y"])))
    D.sort(key=lambda d: d["t"])
    return D


def track(D, gate_px, max_gap):
    """A1: 最近邻 + 常速预测关联成航迹。"""
    tracks = []
    for d in D:
        best = None
        for tr in tracks:
            dt = d["t"] - tr["t"]
            if dt <= 0 or dt > 0.30:
                continue
            px, py = tr["u"] + tr["vu"] * dt, tr["v"] + tr["vv"] * dt
            dist = math.hypot(d["u"] - px, d["v"] - py)
            if dist < gate_px and (best is None or dist < best[0]):
                best = (dist, tr)
        if best is None:
            tracks.append(dict(id=len(tracks), pts=[d], t=d["t"], u=d["u"], v=d["v"], vu=0.0, vv=0.0))
        else:
            tr = best[1]
            dt = d["t"] - tr["t"]
            if dt > 0:
                a = 0.5
                tr["vu"] = a * (d["u"] - tr["u"]) / dt + (1 - a) * tr["vu"]
                tr["vv"] = a * (d["v"] - tr["v"]) / dt + (1 - a) * tr["vv"]
            tr["pts"].append(d)
            tr["t"], tr["u"], tr["v"] = d["t"], d["u"], d["v"]
    return tracks


def segment_and_gate(tracks, a, fx):
    """A2 断段 + A3 正对门控；返回 [(track_id, [dets], truth_pid_optional)]"""
    segs = []
    for tr in tracks:
        pts = tr["pts"]
        runs = [[pts[0]]]
        for x, y in zip(pts, pts[1:]):
            (runs.append([y]) if y["t"] - x["t"] > a.max_gap else runs[-1].append(y))
        for run in runs:
            fg = [d for d in run if d["sep"] > a.gate * a.W * fx / d["z"]]
            sub = [[fg[0]]] if fg else []
            for x, y in zip(fg, fg[1:]):
                (sub.append([y]) if y["t"] - x["t"] > a.max_gap else sub[-1].append(y))
            for s in sub:
                if len(s) >= a.min_frames:
                    segs.append((tr["id"], s))
    return segs


def fit_segment(seg, a):
    """与 LocalTrajectoryFitter 相同的刚体模型做 LM。"""
    rows = []
    for d in seg:
        Lt, Lb = (d["lt"], d["lb"]) if d["lt"][1] < d["lb"][1] else (d["lb"], d["lt"])
        Rt, Rb = (d["rt"], d["rb"]) if d["rt"][1] < d["rb"][1] else (d["rb"], d["rt"])
        rows.append((d["t"], +a.W / 2, Lt, Lb))
        rows.append((d["t"], -a.W / 2, Rt, Rb))
    t0 = rows[0][0]

    def resid(P):
        Cx, Cy, Cz, h, th0, om, vx, vy, vz = P
        r = []
        for (t, off, top, bot) in rows:
            dt = t - t0
            th = th0 + om * dt
            for b, (uu, vv) in ((a.L / 2, top), (-a.L / 2, bot)):
                pvx = a.R * math.cos(th) - off * math.sin(th)
                pvy = a.R * math.sin(th) + off * math.cos(th)
                X = pvy + Cx + vx * dt
                Y = -(h + b) + Cy + vy * dt
                Z = pvx + Cz + vz * dt
                r += [a.fx * X / Z + a.cx - uu, a.fy * Y / Z + a.cy - vv]
        return np.array(r)

    def lm(P, iters=200):
        P = np.array(P, float)
        lam = 1e-3
        r = resid(P)
        cost = r @ r
        for _ in range(iters):
            J = np.zeros((len(r), len(P)))
            for j in range(len(P)):
                dP = np.zeros(len(P))
                dP[j] = 1e-6 * max(1, abs(P[j]))
                J[:, j] = (resid(P + dP) - r) / dP[j]
            A = J.T @ J + lam * np.eye(len(P))
            try:
                step = np.linalg.solve(A, -(J.T @ r))
            except np.linalg.LinAlgError:
                break
            Pn = P + step
            rn = resid(Pn)
            cn = rn @ rn
            if cn < cost:
                P, r, cost = Pn, rn, cn
                lam = max(lam * 0.5, 1e-9)
            else:
                lam *= 3
        return P, math.sqrt(cost / len(r))

    best = None
    for om0 in (3, -3, 6, -6):
        P, rm = lm([0, 0, 3, 0.1, 0, om0, 0, 0, 0])
        if best is None or rm < best[1]:
            best = (P, rm)
    return best


def truth_lookup(path, off):
    T = collections.defaultdict(list)
    for r in csv.DictReader(open(path)):
        T[int(r["time_us"])].append((float(r["x"]), float(r["y"]), float(r["z"]), int(r["plate_id"])))
    tf = np.array(sorted(T))

    def pid(t, x, y, z):
        i = int(np.argmin(np.abs(tf - (t * 1e6 - off * 1e6))))
        return min((math.dist((x, y, z), (px, py, pz)), p) for (px, py, pz, p) in T[tf[i]])[1]

    return pid


def main():
    a = parse()
    D = load_dets(a)
    tracks = track(D, a.track_gate, a.max_gap)
    segs = segment_and_gate(tracks, a, a.fx)
    pid = truth_lookup(a.truth, a.off) if a.truth else None

    print("dets=%d  tracks=%d  segments(>=%d)=%d" % (len(D), len(tracks), a.min_frames, len(segs)))
    rows = []
    for tid, s in segs:
        P, rm = fit_segment(s, a)
        arc = abs(s[-1]["t"] - s[0]["t"]) * abs(P[5]) * 180 / math.pi
        pur = ""
        if pid is not None:
            c = collections.Counter(pid(d["t"], d["x"], d["y"], d["z"]) for d in s)
            mp, mc = c.most_common(1)[0]
            pur = "truth=%d(%d/%d)" % (mp, mc, len(s))
            if mc / len(s) >= 0.6:
                pur = "OK " + pur
        rows.append((tid, len(s), arc, P[5], rm, s, pur))

    om = np.array([r[3] for r in rows])
    rm = np.array([r[4] for r in rows])
    print("%-5s %4s %6s %9s %7s  %s" % ("track", "n", "arc°", "omega", "rmse", "identity"))
    for tid, n, arc, o, r, s, pur in rows:
        print("%-5d %4d %6.0f %+9.3f %6.2fpx  %s" % (tid, n, arc, o, r, pur))
    print("SUMMARY: omega mean=%.3f std=%.3f  rmse mean=%.2f px  n_seg=%d"
          % (om.mean(), om.std(), rm.mean(), len(rows)))

    if a.png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(13, 5))
        cmap = plt.get_cmap("tab20")
        for i, (tid, n, arc, o, r, s, pur) in enumerate(rows):
            ts = [(d["t"] - D[0]["t"]) for d in s]
            ax[0].plot(ts, [d["u"] for d in s], "-o", ms=3, lw=1, color=cmap(i % 20))
        if pid is not None:
            for id_, tr in enumerate(tracks):
                pass
        ax[0].set_xlabel("t (s)"); ax[0].set_ylabel("plate-center u (px)")
        ax[0].set_title("A1/A2/A3 single-plate front arcs (one color per segment)"); ax[0].grid(alpha=.3)
        ax[1].axhline(-3.0, color="k", ls="--", lw=1, label="truth omega=-3.0")
        ax[1].bar(range(len(rows)), om, color="tab:blue", alpha=.8)
        ax[1].set_xlabel("segment #"); ax[1].set_ylabel("fitted omega (rad/s)")
        ax[1].set_title("per-segment fitted omega"); ax[1].legend(); ax[1].grid(alpha=.3)
        fig.tight_layout(); fig.savefig(a.png, dpi=110)
        print("wrote", a.png)


if __name__ == "__main__":
    main()
