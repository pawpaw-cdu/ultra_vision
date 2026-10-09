#!/usr/bin/env bash
# 局部拟合器效果实验：扫描 (距离 × 转速)，对比"局部拟合 ON/OFF"下 EKF 的 ω
# 稳定性，以及拟合器自己输出的 ω / 重投影 rmse / 有效率。
#
# 产出：data/spin_sweep.png（ω 稳定性 / ω 偏差 / 有效率）、data/spin_sweep_ekf.png（时域）。
#
# 用法：experiments/local_fitter/exp_spin_sweep.sh [out_dir=/tmp/exp]
# 前提：Ultra_Vision 已构建（build_sim/auto_aim_sim），仿真器已构建。
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="${1:-/tmp/exp}"
DATA="$REPO/experiments/local_fitter/data"
mkdir -p "$OUT"
cd "$REPO"

for dist in 3.0 6.5; do
  for spd in 3 5 8; do
    for mode in off on; do
      tag="d${dist}_w${spd}_${mode}"
      if [ "$mode" = on ]; then
        E="ULTRA_VISION_LOCAL_FIT=1 ULTRA_VISION_LOCAL_FIT_ALPHA_DEG=0 ULTRA_VISION_LOCAL_FIT_LOG=$OUT/${tag}_fit.csv"
      else
        E=""
      fi
      echo "[run] $tag"
      env $E ULTRA_VISION_DISABLE_CONTROL=1 DAEDALUS_LIGHT_CSV_PATH="$OUT/lb_${tag}.csv" \
        tools/auto_aim_sim_test.sh --no-build --seconds 15 --camera armor_camera_far \
        --distance "$dist" --gyro 1 --gyro-speed "$spd" --out "$OUT/$tag" >/dev/null 2>&1
    done
  done
done

python3 - "$OUT" "$DATA" <<'PY'
import os, sys, statistics
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
OUT, DATA = sys.argv[1], sys.argv[2]

def sel(p):
    t = open(p).read().splitlines(); h = t[0].split(','); i = h.index('omega')
    return np.array([float(r.split(',')[i]) for r in t[1:] if r.split(',')[i] not in ('', 'nan')])

conds = [(d, s) for d in (3.0, 6.5) for s in (3, 5, 8)]
rows = []
for d, s in conds:
    o = sel(f"{OUT}/d{d}_w{s}_off/uv0/selector.csv"); n = sel(f"{OUT}/d{d}_w{s}_on/uv0/selector.csv")
    ot, nt = o[len(o)//3:], n[len(n)//3:]
    fw, fr, ok, tot = [], [], 0, 0
    for ln in open(f"{OUT}/d{d}_w{s}_on_fit.csv"):
        p = ln.strip().split(',')
        if len(p) != 5: continue
        tot += 1
        if p[4] == '1': ok += 1; fw.append(float(p[2])); fr.append(float(p[3]))
    rows.append(dict(d=d, s=s, off_m=np.mean(ot), off_s=np.std(ot), on_m=np.mean(nt), on_s=np.std(nt),
                     fitw=statistics.median(fw) if fw else float('nan'),
                     fitrm=statistics.median(fr) if fr else float('nan'), valid=ok/max(1,tot)))

print("%6s %5s | %8s %6s | %8s %6s | %8s %7s | %6s" % ("dist","w","EKFoff","std","EKFon","std","fit w","rmse","valid"))
for r in rows:
    print("%6.1f %5d | %+8.2f %6.2f | %+8.2f %6.2f | %+8.3f %7.2f | %5.0f%%" % (
        r['d'],r['s'],r['off_m'],r['off_s'],r['on_m'],r['on_s'],r['fitw'],r['fitrm'],r['valid']*100))

fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
for d, c in ((3.0, 'tab:blue'), (6.5, 'tab:red')):
    rs = [r for r in rows if r['d'] == d]
    ax[0].plot([r['s'] for r in rs], [r['off_s'] for r in rs], 'o--', color=c, label=f"dist {d}m OFF")
    ax[0].plot([r['s'] for r in rs], [r['on_s'] for r in rs], 'o-', color=c, label=f"dist {d}m ON")
    ax[1].plot([r['s'] for r in rs], [abs(r['fitw'])-r['s'] for r in rs], 'o-', color=c, label=f"dist {d}m")
    ax[2].plot([r['s'] for r in rs], [r['valid']*100 for r in rs], 'o-', color=c, label=f"dist {d}m")
ax[0].set_xlabel("truth ω (rad/s)"); ax[0].set_ylabel("EKF ω std (rad/s)"); ax[0].set_title("EKF ω stability (lower=better)"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
ax[1].axhline(0, color='k', lw=.8); ax[1].set_xlabel("truth ω (rad/s)"); ax[1].set_ylabel("fit ω bias (rad/s)"); ax[1].set_title("local-fit ω bias"); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
ax[2].set_xlabel("truth ω (rad/s)"); ax[2].set_ylabel("valid fit fraction (%)"); ax[2].set_title("local-fit validity"); ax[2].legend(fontsize=8); ax[2].grid(alpha=.3)
fig.tight_layout(); fig.savefig(f"{DATA}/spin_sweep.png", dpi=110)

fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
for ax2, (dd, ww) in zip(axes, [(6.5, 3), (6.5, 8)]):
    for mode, c, lab in (('off', 'tab:gray', 'EKF ω (fit OFF)'), ('on', 'tab:blue', 'EKF ω (fit ON)')):
        t = open(f"{OUT}/d{dd}_w{ww}_{mode}/uv0/selector.csv").read().splitlines(); h = t[0].split(',')
        i, j = h.index('omega'), h.index('time_us'); R = [r.split(',') for r in t[1:]]
        om = np.array([float(r[i]) for r in R]); tt = (np.array([float(r[j]) for r in R]) - float(R[0][j])) * 1e-6
        ax2.plot(tt, om, lw=1, color=c, label=lab)
    ax2.axhline(ww, color='k', ls='--', lw=1, label=f"truth ω={ww}")
    ax2.set_title(f"dist {dd} m, spin {ww} rad/s"); ax2.set_xlabel("t (s)"); ax2.set_ylabel("EKF ω (rad/s)")
    ax2.set_ylim(0, ww*1.9); ax2.legend(fontsize=8); ax2.grid(alpha=.3)
fig.tight_layout(); fig.savefig(f"{DATA}/spin_sweep_ekf.png", dpi=110)
print("wrote", f"{DATA}/spin_sweep.png", "and spin_sweep_ekf.png")
PY
