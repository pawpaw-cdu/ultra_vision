#!/usr/bin/env bash
# 深大 RP-26Rune 能量机关算法 + simulator_system 仿真器 的一键回归。
#
#   tools/rp26_sim/run.sh [--preset low|low_large|mid|far] [--mode small|large]
#                         [--seconds 30] [--out /tmp/rp26_test]
#                         [--width 640] [--height 480] [--fps 30] [--open-loop]
#                         [-- <额外的 rp26_sim 参数>]
#
# 数据流与 Ultra_Vision 的 tools/rune_sim_test.sh 完全一致（同一个仿真器、同样的端口）：
#   daedalus 7666 帧 -> rp26_sim（深大检测+算法）-> 7667 云台/开火指令；7668 遥测打分。
#
# 产出：
#   <out>/rp26.csv           逐帧：云台姿态、他们的下发角、像素残差、自洽性检查
#   <out>/rp26.csv.ground_truth  仿真器 RUNE 遥测行原文
#   <out>/rp26.log           视觉端标准输出（含 glog）
#   <out>/sim.log            仿真器标准输出
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SIM_DIR="${SIM_DIR:-$REPO_ROOT/../simulator_system/simulator}"
SIM_BIN="$SIM_DIR/target/release/daedalus"
HARNESS_BIN="${RP26_SIM_BIN:-$REPO_ROOT/build_rp26_sim/bin/rp26_sim}"

preset="low"
mode="small"
seconds=30
out="/tmp/rp26_test"
width="${RP26_CAPTURE_WIDTH:-640}"
height="${RP26_CAPTURE_HEIGHT:-480}"
fps="${RP26_CAPTURE_FPS:-}"
extra=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --preset) preset="$2"; shift 2 ;;
    --mode) mode="$2"; shift 2 ;;
    --seconds) seconds="$2"; shift 2 ;;
    --out) out="$2"; shift 2 ;;
    --width) width="$2"; shift 2 ;;
    --height) height="$2"; shift 2 ;;
    --fps) fps="$2"; shift 2 ;;
    --) shift; extra=("$@"); break ;;
    *) extra+=("$1"); shift ;;
  esac
done

camera_config="$REPO_ROOT/tools/sim/rune_camera_${preset}.yaml"
[[ -f "$camera_config" ]] || { echo "no such preset: $preset" >&2; exit 2; }
[[ -x "$SIM_BIN" ]] || { echo "simulator not built: $SIM_BIN" >&2; exit 2; }
[[ -x "$HARNESS_BIN" ]] || { echo "harness not built: $HARNESS_BIN (cmake --build build_rp26_sim)" >&2; exit 2; }

mkdir -p "$out"
pkill -f 'target/release/daedalus' 2>/dev/null || true
sleep 1

(
  cd "$SIM_DIR"
  DAEDALUS_SCENARIO=energy_rune \
  DAEDALUS_CAMERA_CONFIG="$camera_config" \
  DAEDALUS_RUNE_ONLY=1 \
  DAEDALUS_TELEMETRY_ADDR=127.0.0.1:7668 \
  DAEDALUS_TCP_ADDR=127.0.0.1:7666 \
  DAEDALUS_COMMAND_ADDR=127.0.0.1:7667 \
  DAEDALUS_CAPTURE_WIDTH="$width" \
  DAEDALUS_CAPTURE_HEIGHT="$height" \
  ${fps:+DAEDALUS_CAPTURE_FPS="$fps"} \
  "$SIM_BIN" > "$out/sim.log" 2>&1 &
  echo $! > "$out/sim.pid"
)
sleep 6

set +e
RP26_CONFIG_DIR="${RP26_CONFIG_DIR:-$REPO_ROOT/tools/rp26_sim/config}" \
"$HARNESS_BIN" --mode "$mode" --seconds "$seconds" --csv "$out/rp26.csv" \
  ${extra[@]+"${extra[@]}"} > "$out/rp26.log" 2>&1
harness_status=$?
set -e

kill "$(cat "$out/sim.pid")" 2>/dev/null || true
sleep 1

echo "--- rp26_sim (exit=$harness_status)"
tail -25 "$out/rp26.log"
echo "--- ground truth（命中/激活）"
grep -c "event=hit" "$out/rp26.csv.ground_truth" 2>/dev/null | sed 's/^/hits: /' || true
grep -c "event=activated" "$out/rp26.csv.ground_truth" 2>/dev/null | sed 's/^/activated: /' || true
echo "--- 输出文件"
ls -l "$out/rp26.csv" "$out/rp26.log" "$out/sim.log" 2>/dev/null || true
