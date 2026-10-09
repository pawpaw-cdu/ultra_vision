#!/usr/bin/env python3
"""从 data/*.csv 生成局部拟合器的结果曲线（data/*.png）。
用法： python3 plots.py   （在 experiments/local_fitter 下运行）"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")


def read(name):
    with open(os.path.join(DATA, name), newline="") as f:
        return list(csv.DictReader(f))


def fig_multiplate():
    rows = read("identifiability.csv")
    modes = ["bar1", "plate", "multi"]
    Ns = sorted({int(r["N"]) for r in rows})
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for m in modes:
        a = [next(float(r["alpha_end_deg"]) for r in rows if r["mode"] == m and int(r["N"]) == n) for n in Ns]
        r_ = [next(float(r["R_end"]) for r in rows if r["mode"] == m and int(r["N"]) == n) for n in Ns]
        ax[0].plot(Ns, a, "o-", label=m)
        ax[1].plot(Ns, r_, "o-", label=m)
    ax[0].axhline(15, ls="--", c="k", lw=1); ax[0].axhspan(14, 16, color="k", alpha=0.06)
    ax[0].set_xlabel("window N (frames)"); ax[0].set_ylabel("converged alpha (deg)"); ax[0].set_title("alpha convergence (true 15)")
    ax[1].axhline(0.20, ls="--", c="k", lw=1); ax[1].axhspan(0.18, 0.22, color="k", alpha=0.06)
    ax[1].set_xlabel("window N (frames)"); ax[1].set_ylabel("converged R (m)"); ax[1].set_title("R convergence (true 0.20)")
    ax[0].legend(); ax[1].legend()
    fig.tight_layout(); fig.savefig(os.path.join(DATA, "multiplate_identifiability.png"), dpi=120); plt.close(fig)


def fig_slow_trace():
    rows = [r for r in read("multi_slow.csv") if r["N"] == "40" and r["mode"] == "multi"]
    keys = sorted({(r["init_alpha_deg"], r["init_R"]) for r in rows})
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for a0, R0 in keys:
        sub = [r for r in rows if r["init_alpha_deg"] == a0 and r["init_R"] == R0]
        t = [float(r["t"]) for r in sub]
        ax[0].plot(t, [float(r["alpha_deg"]) for r in sub], label=f"a0={a0},R0={R0}")
        ax[1].plot(t, [float(r["R"]) for r in sub], label=f"a0={a0},R0={R0}")
    ax[0].axhline(15, ls="--", c="k"); ax[1].axhline(0.20, ls="--", c="k")
    ax[0].set_xlabel("t (s)"); ax[0].set_ylabel("alpha (deg)"); ax[0].set_title("slow-recursive alpha")
    ax[1].set_xlabel("t (s)"); ax[1].set_ylabel("R (m)"); ax[1].set_title("slow-recursive R")
    ax[0].legend(fontsize=8); ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(DATA, "slow_recursive.png"), dpi=120); plt.close(fig)


def fig_model_switch():
    rows = read("model_switch.csv")
    fig, ax = plt.subplots(figsize=(7, 4))
    for N in sorted({r["N"] for r in rows}):
        sub = [r for r in rows if r["N"] == N]
        ax.plot([float(r["t"]) for r in sub], [float(r["R"]) for r in sub], "-", label=f"N={N}")
    ax.axvline(5.0, ls=":", c="r"); ax.text(5.05, 0.205, "R 0.20->0.28", color="r", fontsize=8)
    ax.axhline(0.28, ls="--", c="k", lw=1)
    ax.set_xlabel("t (s)"); ax.set_ylabel("R (m)"); ax.set_title("model switch: R estimate (true step to 0.28)")
    ax.legend(); fig.tight_layout(); fig.savefig(os.path.join(DATA, "model_switch.png"), dpi=120); plt.close(fig)


def fig_observability():
    rows = read("observability.csv")
    Cz15 = [float(r["Cz"]) for r in rows if r["alpha_deg"] == "15"]
    d15 = [float(r["d_deltaU_dtheta_px_per_rad"]) for r in rows if r["alpha_deg"] == "15"]
    Cz0 = [float(r["Cz"]) for r in rows if r["alpha_deg"] == "0"]
    d0 = [float(r["d_deltaU_dtheta_px_per_rad"]) for r in rows if r["alpha_deg"] == "0"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(Cz15, d15, "o-", label="alpha=15deg")
    ax.plot(Cz0, d0, "s-", label="alpha=0")
    ax.axvline(0.40, ls=":", c="r"); ax.text(0.41, min(d15) * 0.4, "Cz=2R", color="r", fontsize=8)
    ax.set_xlabel("Cz (m)"); ax.set_ylabel("d(u_top-u_bot)/dtheta (px/rad)")
    ax.set_title("frontal diff observability: alpha=0 is always 0"); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(DATA, "observability.png"), dpi=120); plt.close(fig)


def fig_pnp():
    rows = read("pnp_compare.csv")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    vals = {
        "PnP raw": [float(r["yaw_pnp_err_deg"]) for r in rows],
        "PnP 解分支": [float(r["yaw_pnp_branch_deg"]) for r in rows],
        "Fit": [float(r["yaw_fit_err_deg"]) for r in rows],
    }
    ax[0].hist(vals["PnP raw"], bins=40, alpha=0.6, label="PnP raw")
    ax[0].hist(vals["PnP 解分支"], bins=40, alpha=0.6, label="PnP branch-resolved")
    ax[0].hist(vals["Fit"], bins=40, alpha=0.9, label="multi-frame fit")
    ax[0].set_xlabel("yaw error (deg)"); ax[0].set_ylabel("count"); ax[0].legend(); ax[0].set_title("yaw error distribution")
    cs = [abs(float(r["cosang"])) for r in rows]
    ax[1].scatter(cs, vals["PnP 解分支"], s=4, alpha=0.4, label="PnP branch-resolved")
    ax[1].scatter(cs, vals["Fit"], s=4, alpha=0.6, label="fit")
    ax[1].set_xlabel("|cos(viewing angle)| (1=frontal)"); ax[1].set_ylabel("yaw error (deg)")
    ax[1].set_title("error vs viewing angle"); ax[1].legend()
    fig.tight_layout(); fig.savefig(os.path.join(DATA, "pnp_vs_fit.png"), dpi=120); plt.close(fig)


def fig_value():
    rows = read("value_analysis.csv")
    d = [float(r["d_m"]) for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(d, [float(r["ekf_yaw_pnp_deg"]) for r in rows], "o-", label="PnP-only EKF")
    ax[0].plot(d, [float(r["ekf_yaw_fit_deg"]) for r in rows], "s-", label="local-fit EKF")
    ax[0].set_xlabel("distance (m)"); ax[0].set_ylabel("steady-state yaw sigma (deg)")
    ax[0].set_yscale("log"); ax[0].set_title("EKF yaw sigma vs distance"); ax[0].legend()
    ax[1].plot(d, [float(r["miss_pnp_cm"]) for r in rows], "o-", label="PnP-only")
    ax[1].plot(d, [float(r["miss_fit_cm"]) for r in rows], "s-", label="local-fit")
    ax[1].axhline(6.75, ls="--", c="k", lw=1, label="armor half-width")
    ax[1].set_xlabel("distance (m)"); ax[1].set_ylabel("miss at target (cm)")
    ax[1].set_title("aim miss distance"); ax[1].legend()
    fig.tight_layout(); fig.savefig(os.path.join(DATA, "value_vs_distance.png"), dpi=120); plt.close(fig)

    rows2 = read("value_vs_noise.csv")
    sp = [float(r["sigma_px"]) for r in rows2]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(sp, [float(r["yaw_pnp_deg"]) for r in rows2], "o-", label="PnP-only")
    ax[0].plot(sp, [float(r["yaw_fit_deg"]) for r in rows2], "s-", label="local-fit")
    ax[0].set_xlabel("corner noise sigma (px)"); ax[0].set_ylabel("yaw error (deg)")
    ax[0].set_yscale("log"); ax[0].set_title("yaw error vs corner noise @3m"); ax[0].legend()
    ax[1].plot(sp, [float(r["hit_pnp"]) for r in rows2], "o-", label="PnP-only")
    ax[1].plot(sp, [float(r["hit_fit"]) for r in rows2], "s-", label="local-fit")
    ax[1].set_xlabel("corner noise sigma (px)"); ax[1].set_ylabel("hit probability")
    ax[1].set_title("hit probability vs corner noise @3m"); ax[1].legend()
    fig.tight_layout(); fig.savefig(os.path.join(DATA, "value_vs_noise.png"), dpi=120); plt.close(fig)


if __name__ == "__main__":
    have = set(os.listdir(DATA))
    if "identifiability.csv" in have:
        fig_multiplate()
    if "multi_slow.csv" in have:
        fig_slow_trace()
    if "model_switch.csv" in have:
        fig_model_switch()
    if "observability.csv" in have:
        fig_observability()
    if "pnp_compare.csv" in have:
        fig_pnp()
    if "value_analysis.csv" in have:
        fig_value()
    print("wrote data/*.png")
