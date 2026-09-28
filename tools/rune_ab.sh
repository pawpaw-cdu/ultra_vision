#!/usr/bin/env bash
# 批量 A/B：同一预设下跑 N 轮，输出每轮指标与分布（中位数/四分位），
# 用来判断"某配置是否真的更好"——单轮 30 s 的命中率方差约 ±15%，
# N<5 时结论不可信（实测三组配置各 3 轮，中位数 0%/33%/36% 完全分不开）。
#
# 用法：
#   tools/rune_ab.sh --runs 10 --seconds 30 --preset low --mode small \
#                    --a configs [--b /tmp/cfg_other] [--truth 6.05]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
runs=10
seconds=30
preset=low
mode=small
truth=6.05
config_a="$REPO_ROOT/configs"
config_b=""
pad=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --runs) runs="$2"; shift 2 ;;
    --seconds) seconds="$2"; shift 2 ;;
    --preset) preset="$2"; shift 2 ;;
    --mode) mode="$2"; shift 2 ;;
    --truth) truth="$2"; shift 2 ;;
    --a) config_a="$2"; shift 2 ;;
    --b) config_b="$2"; shift 2 ;;
    --pad) pad="$2"; shift 2 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

run_group() {
  local tag="$1" cfg_dir="$2"
  echo "=== group ${tag}: ${cfg_dir}"
  for i in $(seq 1 "$runs"); do
    local out="/tmp/rune_ab_${tag}_${i}"
    local args=(--preset "$preset" --mode "$mode" --seconds "$seconds" --repeat 1 --out "$out")
    [[ -n "$pad" ]] && args+=(--pad "$pad")
    ULTRA_VISION_CONFIG_DIR="$cfg_dir" "$REPO_ROOT/tools/rune_sim_test.sh" "${args[@]}" >/dev/null 2>&1 || true
  done
  python3 - "$tag" "$runs" <<'PY'
import glob, re, statistics as st, subprocess, sys
tag, runs = sys.argv[1], int(sys.argv[2])
rates, swings, aims, seqs = [], [], [], []
for index in range(1, runs + 1):
    run_dir = f"/tmp/rune_ab_{tag}_{index}"
    out = subprocess.run(["python3", "tools/rune_curve.py", run_dir, "--truth-distance", "6.05"],
                         capture_output=True, text=True).stdout
    rate = re.search(r"命中 (\d+) 轮 \((\d+)%\)", out)
    seq = re.search(r"^\s+([.Y]+)$", out, re.M)
    aim = re.search(r"p90 ([\d.]+)°", out)
    swing = re.search(r">15° 的帧 (\d+)", out)
    if not rate:
        continue
    rates.append(int(rate.group(2)))
    seqs.append(seq.group(1) if seq else "?")
    aims.append(float(aim.group(1)) if aim else 0.0)
    swings.append(int(swing.group(1)) if swing else 0)
if not rates:
    print("  没有可用数据")
    raise SystemExit
rates_sorted = sorted(rates)
print(f"  n={len(rates)}  每轮命中 %s" % rates)
print(f"  中位 {st.median(rates):.0f}%%  四分位 {rates_sorted[len(rates)//4]}~{rates_sorted[3*len(rates)//4]}%%"
      f"  最大连续命中 {max((s.count('Y') for s in seqs), default=0)}")
print(f"  aim p90 中位 {st.median(aims):.2f}deg   甩飞帧 %s (中位 {st.median(swings):.0f})" % swings)
for seq in seqs:
    print(f"    {seq}")
PY
}

run_group a "$config_a"
if [[ -n "$config_b" ]]; then
  run_group b "$config_b"
fi
