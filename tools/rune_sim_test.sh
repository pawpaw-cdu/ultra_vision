#!/usr/bin/env bash
# 能量机关仿真回归测试：启动仿真器 -> 跑视觉端 -> 汇总击打/激活结果。
#
# 用法：
#   tools/rune_sim_test.sh [--preset near|far] [--seconds 90] [--mode small|large]
#                          [--score 0.7] [--pad 1.6] [--out /tmp/rune_test]
#
# 产出：
#   <out>/vision.csv    视觉端逐帧遥测（含瞄准角、估计相位、下发角）
#   <out>/ground_truth.csv  仿真器击打/状态真值
#   <out>/sim.log       仿真器标准输出（含每一行 RUNE 事件）
#   <out>/summary.txt   本次结果摘要（开火/命中/得分/激活）
#
# 约定：脚本只负责编排，仿真器与视觉端都用仓库里已有的可执行文件；
# 测试完成后会关闭自己启动的仿真器进程。
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SIM_DIR="${SIM_DIR:-$REPO_ROOT/../simulator_system/simulator}"
SIM_BIN="$SIM_DIR/target/release/daedalus"
RUNE_BIN="$REPO_ROOT/build_rune/src/auto_buff/auto_buff_demo"

preset="near"
seconds=90
mode="small"
score=""
pad=""
repeat=1
record=0
out="/tmp/rune_test"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --preset) preset="$2"; shift 2 ;;
    --seconds) seconds="$2"; shift 2 ;;
    --mode) mode="$2"; shift 2 ;;
    --score) score="$2"; shift 2 ;;
    --pad) pad="$2"; shift 2 ;;
    --repeat) repeat="$2"; shift 2 ;;
    --record-frames) record=1; shift ;;
    --record-stride) record_stride="$2"; shift 2 ;;
    --out) out="$2"; shift 2 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

camera_config="$REPO_ROOT/tools/sim/rune_camera_${preset}.yaml"
[[ -f "$camera_config" ]] || { echo "no such preset: $preset" >&2; exit 2; }
[[ -x "$SIM_BIN" ]] || { echo "simulator not built: $SIM_BIN" >&2; exit 2; }
[[ -x "$RUNE_BIN" ]] || { echo "vision not built: $RUNE_BIN (cmake --target rune_demo)" >&2; exit 2; }

mkdir -p "$out"
pkill -f 'target/release/daedalus' 2>/dev/null || true
sleep 1

run_once() {
  local iter="$1"
  local run_dir="$out"
  [[ "$repeat" -gt 1 ]] && run_dir="$out/run$iter"
  mkdir -p "$run_dir"

(
  cd "$SIM_DIR"
  DAEDALUS_SCENARIO=energy_rune \
  DAEDALUS_CAMERA_CONFIG="$camera_config" \
  DAEDALUS_RUNE_ONLY=1 \
  DAEDALUS_TELEMETRY_ADDR=127.0.0.1:7668 \
  DAEDALUS_TCP_ADDR=127.0.0.1:7666 \
  DAEDALUS_COMMAND_ADDR=127.0.0.1:7667 \
  DAEDALUS_CAPTURE_WIDTH=640 \
  DAEDALUS_CAPTURE_HEIGHT=480 \
  DAEDALUS_CAPTURE_FPS=30 \
  "$SIM_BIN" > "$run_dir/sim.log" 2>&1 &
  echo $! > "$run_dir/sim.pid"
)
sleep 6

vision_env=(
  ULTRA_VISION_SIM_HOST=127.0.0.1
  ULTRA_VISION_NO_DISPLAY=1
  ULTRA_VISION_RUNE_MODE="$mode"
  ULTRA_VISION_RUNE_TEST_SECONDS="$seconds"
  ULTRA_VISION_RUNE_CSV="$run_dir/vision.csv"
  ULTRA_VISION_RUNE_GT_CSV="$run_dir/ground_truth.csv"
)
if [[ "$record" == "1" ]]; then
  vision_env+=(ULTRA_VISION_RUNE_FRAME_DIR="$run_dir/frames")
  vision_env+=(ULTRA_VISION_RUNE_FRAME_STRIDE="${record_stride:-5}")
fi
if [[ -n "$score" || -n "$pad" ]]; then
  # 覆盖检出阈值 / 预填充比例：复制一份配置到输出目录再改
  cp "$REPO_ROOT/configs/buff.yaml" "$run_dir/buff.yaml"
  [[ -n "$score" ]] && sed -i '' "s/^    score_threshold: .*/    score_threshold: $score/" "$run_dir/buff.yaml"
  [[ -n "$pad" ]] && sed -i '' "s/^    input_pad_scale: .*/    input_pad_scale: $pad/" "$run_dir/buff.yaml"
  vision_env+=(ULTRA_VISION_CONFIG_DIR="$run_dir")
fi

(
  cd "$REPO_ROOT"
  env "${vision_env[@]}" "$RUNE_BIN" "${ULTRA_VISION_CONFIG_DIR:-$REPO_ROOT/configs}" \
      > "$run_dir/vision.log" 2>&1 || true
)

  kill "$(cat "$run_dir/sim.pid")" 2>/dev/null || true
  sleep 1
}

for i in $(seq 1 "$repeat"); do
  run_once "$i"
done

{
  echo "preset      : $preset ($camera_config)"
  echo "mode        : $mode"
  echo "seconds     : $seconds"
  [[ -n "$score" ]] && echo "score_thresh: $score"
  [[ -n "$pad" ]] && echo "input_pad   : $pad"
  echo
  for i in $(seq 1 "$repeat"); do
    local_log="$out/vision.log"; [[ "$repeat" -gt 1 ]] && local_log="$out/run$i/vision.log"
    echo "--- run $i"
    grep -E "shots fired|hits on target|scored hits|activations" "$local_log" || true
  done
  echo
  hits_total=0; act_total=0
  for i in $(seq 1 "$repeat"); do
    log="$out/sim.log"; [[ "$repeat" -gt 1 ]] && log="$out/run$i/sim.log"
    # 注意：仿真器发的是 `phase=activated`（不是 event=activated），历史上这里一直
    # 统计不到 → summary 永远显示 0。这里按"掩码打满 11111"数，才是真正的激活次数。
    h=$(grep -c 'event=hit' "$log" 2>/dev/null || true)
    a=$(grep -c 'activated=11111' "$log" 2>/dev/null || true)
    hits_total=$((hits_total + ${h:-0})); act_total=$((act_total + ${a:-0}))
  done
  echo "sim hit events    : $hits_total"
  echo "sim activations   : $act_total (掩码打满 11111 的行数; >0 即打满过)"
} | tee "$out/summary.txt"
