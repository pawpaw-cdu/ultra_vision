#!/usr/bin/env python3
"""深大 RP-26Rune 阈值扫描：量出"他们的传统链路在仿真器画面上卡在哪"。

用法：
    python3 tools/rp26_sim/sweep.py --seconds 8 [--open-loop] [--preset low] [--mode small]

对每个配置：把 power_rune.json 拷到临时目录、按 dot 路径改几个键，然后跑 run.sh，
再统计：
    probe_total      RuneObservationRefiner 每个候选打出的日志条数
    projectable      其中"装甲板+灯臂+R标都可投影"的比例（决定能否进入位姿优化）
    pose_ok          位姿优化成功次数（= 候选点进入相位滤波的次数，来自他们的调试图像计数）
    dir_ok           相位滤波确认旋转方向的次数（确认后才能真正输出目标）
    find_frames      CSV 里 is_find=1 的帧占比（云台是否被他们接管）
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HARNESS_ROOT = Path(__file__).resolve().parent

# 扫描一律从"他们的原始配置"出发（config/power_rune.rp26_origin.json），
# 这样 baseline 场景才是真正的深大原参数；harness 默认跑的是平台适配版
# （config/power_rune.json）。
DEFAULT_CONFIG = HARNESS_ROOT / "config" / "power_rune.rp26_origin.json"
DETECT_CONFIG = HARNESS_ROOT / "config" / "rp26_sim.json"


def set_dotted(config, dotted_key, value):
    node = config
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


SCENARIOS = {
    "baseline": {},
    "mask30": {"feature_extract.red_minus_blue2bin_threshold": 30.0},
    "solidity045": {"contour_descriptor.solidity_threshold_rectangular": 0.45},
    "aspect090": {"contour_descriptor.aspect_ratio_relative_error_threshold": 0.9},
    "relaxed": {
        "contour_descriptor.solidity_threshold_rectangular": 0.45,
        "contour_descriptor.aspect_ratio_relative_error_threshold": 0.9,
    },
    "relaxed_mask30": {
        "feature_extract.red_minus_blue2bin_threshold": 30.0,
        "contour_descriptor.solidity_threshold_rectangular": 0.45,
        "contour_descriptor.aspect_ratio_relative_error_threshold": 0.9,
    },
    # 只对齐弹道：仿真器弹丸 25 m/s、LinearDamping(0.05) ≈ 阻力系数 0.043、无马格努斯。
    "ballistic_sim": {
        "rune_ballistic_model.bullet_flying_speed": 25.0,
        "rune_ballistic_model.drag_coefficient": 0.043,
        "rune_ballistic_model.magnus_acceleration": 0.0,
    },
    # 掩膜 + 弹道一起对齐（"仿真器适配"组合）
    "sim_fit": {
        "feature_extract.red_minus_blue2bin_threshold": 30.0,
        "rune_ballistic_model.bullet_flying_speed": 25.0,
        "rune_ballistic_model.drag_coefficient": 0.043,
        "rune_ballistic_model.magnus_acceleration": 0.0,
    },
    "sim_fit_relaxed": {
        "feature_extract.red_minus_blue2bin_threshold": 30.0,
        "contour_descriptor.solidity_threshold_rectangular": 0.45,
        "contour_descriptor.aspect_ratio_relative_error_threshold": 0.9,
        "rune_ballistic_model.bullet_flying_speed": 25.0,
        "rune_ballistic_model.drag_coefficient": 0.043,
        "rune_ballistic_model.magnus_acceleration": 0.0,
    },
    # 观测率适配：他们的相位滤波器要求"≤0.1 s 间隔的连续 20 帧观测"，
    # 而本平台（仿真器 + 五点网络）在一轮 2.5 s 点灯窗口里给不出这么密的观测，
    # 所以把这两个键按本平台的观测率放宽（仍是他们自己的配置接口）。
    "obs_adapt": {
        "feature_extract.red_minus_blue2bin_threshold": 30.0,
        "rune_ballistic_model.bullet_flying_speed": 25.0,
        "rune_ballistic_model.drag_coefficient": 0.043,
        "rune_ballistic_model.magnus_acceleration": 0.0,
        "small_phase_estimate.max_data_interval": 0.25,
        "small_phase_estimate.min_size_to_confirm_rotation": 10,
    },
    "obs_adapt_relaxed": {
        "feature_extract.red_minus_blue2bin_threshold": 30.0,
        "contour_descriptor.solidity_threshold_rectangular": 0.45,
        "contour_descriptor.aspect_ratio_relative_error_threshold": 0.9,
        "rune_ballistic_model.bullet_flying_speed": 25.0,
        "rune_ballistic_model.drag_coefficient": 0.043,
        "rune_ballistic_model.magnus_acceleration": 0.0,
        "small_phase_estimate.max_data_interval": 0.25,
        "small_phase_estimate.min_size_to_confirm_rotation": 10,
    },
    # 再加数据寿命：他们的 data_life=0.2 s 意味着"观测断超 0.2 s 就退化为瞄符心、
    # 且禁止开火"，而本平台的观测间隙常常 0.5 s 以上，于是云台一直在"回符心"、
    # 从不真正开火。这里把 data_life/recover_time 按本平台观测间隙放宽。
    "platform_fit": {
        "feature_extract.red_minus_blue2bin_threshold": 30.0,
        "rune_ballistic_model.bullet_flying_speed": 25.0,
        "rune_ballistic_model.drag_coefficient": 0.043,
        "rune_ballistic_model.magnus_acceleration": 0.0,
        "small_phase_estimate.max_data_interval": 0.25,
        "small_phase_estimate.min_size_to_confirm_rotation": 10,
        "temp.data_life": 0.6,
        "temp.recover_time": 0.4,
    },
    "platform_fit_relaxed": {
        "feature_extract.red_minus_blue2bin_threshold": 30.0,
        "contour_descriptor.solidity_threshold_rectangular": 0.45,
        "contour_descriptor.aspect_ratio_relative_error_threshold": 0.9,
        "rune_ballistic_model.bullet_flying_speed": 25.0,
        "rune_ballistic_model.drag_coefficient": 0.043,
        "rune_ballistic_model.magnus_acceleration": 0.0,
        "small_phase_estimate.max_data_interval": 0.25,
        "small_phase_estimate.min_size_to_confirm_rotation": 10,
        "temp.data_life": 0.6,
        "temp.recover_time": 0.4,
    },
    # 再把开火状态机的三个键按"稀疏观测"改：他们的初冷却 0.3 s + 连发阈值 0.04 s
    # 是给"连续观测、100~200 Hz 火控"设计的；观测断流时每次重新建目标都会重置冷却，
    # 导致一轮里最多只能打 1 发。
    "platform_fit_fire": {
        "feature_extract.red_minus_blue2bin_threshold": 30.0,
        "rune_ballistic_model.bullet_flying_speed": 25.0,
        "rune_ballistic_model.drag_coefficient": 0.043,
        "rune_ballistic_model.magnus_acceleration": 0.0,
        "small_phase_estimate.max_data_interval": 0.25,
        "small_phase_estimate.min_size_to_confirm_rotation": 10,
        "temp.data_life": 0.6,
        "temp.recover_time": 0.4,
        "small_rune_decision_module.fire_cooldown_time_init": 0.05,
        "small_rune_decision_module.fire_remaining_time_threshold": 0.2,
        "small_rune_decision_module.fire_cooldown_time": 0.3,
    },
}


def write_config(name, overrides, out_root):
    config_dir = out_root / f"cfg_{name}"
    config_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads(DEFAULT_CONFIG.read_text())
    for key, value in overrides.items():
        set_dotted(config, key, value)
    # ensure_ascii=False：OpenCV 的 JSON 解析器不支持 \uXXXX 转义，中文注释必须原样写。
    (config_dir / "power_rune.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False)
    )
    shutil.copy(DETECT_CONFIG, config_dir / "rp26_sim.json")
    return config_dir


def summarize(name, run_dir, log_path, csv_path):
    log = log_path.read_text(errors="ignore") if log_path.exists() else ""
    probe = re.findall(r"\[REFINE_PROBE\].*projectable=(\d)", log)
    projectable = sum(1 for value in probe if value == "1")
    dir_ok = len(re.findall(r"\[update_candidate_targets\](顺时针|逆时针)", log))

    find_frames = 0
    frames = 0
    if csv_path.exists():
        import csv as csv_module

        for row in csv_module.DictReader(csv_path.open()):
            frames += 1
            if row.get("is_find") == "1":
                find_frames += 1

    ground_truth = Path(str(csv_path) + ".ground_truth")
    hits = scored = activated = 0
    if ground_truth.exists():
        text = ground_truth.read_text(errors="ignore")
        hits = text.count("event=hit")
        scored = text.count("scored=1")
        activated = text.count("event=activated")

    print(
        f"{name:16s} probe={len(probe):4d} projectable={projectable:4d}"
        f" ({projectable / max(1, len(probe)) * 100:5.1f}%)"
        f" dir_ok={dir_ok:3d} find={find_frames}/{frames}"
        f" hits={hits} scored={scored} activated={activated}  [{run_dir}]"
    )
    return {
        "name": name,
        "probe": len(probe),
        "projectable": projectable,
        "dir_ok": dir_ok,
        "find_frames": find_frames,
        "frames": frames,
        "hits": hits,
        "scored": scored,
        "activated": activated,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--preset", default="low")
    parser.add_argument("--mode", default="small")
    parser.add_argument("--open-loop", action="store_true")
    parser.add_argument("--out", default="/tmp/rp26_sweep")
    parser.add_argument("--only", default="")
    args = parser.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    names = [
        name for name in SCENARIOS if not args.only or name in args.only.split(",")
    ]
    results = []
    for name in names:
        config_dir = write_config(name, SCENARIOS[name], out_root)
        run_dir = out_root / f"run_{name}"
        run_dir.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env["RP26_CONFIG_DIR"] = str(config_dir)
        command = [
            str(HARNESS_ROOT / "run.sh"),
            "--preset",
            args.preset,
            "--mode",
            args.mode,
            "--seconds",
            str(args.seconds),
            "--out",
            str(run_dir),
        ]
        if args.open_loop:
            command.append("--open-loop")
        subprocess.run(command, env=env, check=False, stdout=subprocess.DEVNULL)
        results.append(summarize(name, run_dir, run_dir / "rp26.log", run_dir / "rp26.csv"))

    print("\n按 projectable 比例排序：")
    for item in sorted(results, key=lambda r: -r["projectable"] / max(1, r["probe"])):
        print(
            f"  {item['name']:16s} {item['projectable']}/{item['probe']}"
            f"  dir_ok={item['dir_ok']} find={item['find_frames']}"
        )


if __name__ == "__main__":
    main()
