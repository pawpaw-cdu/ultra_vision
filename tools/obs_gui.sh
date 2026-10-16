#!/usr/bin/env bash
# 一键起"仿真器 + 算法 GUI"两个窗口，用于观察不同小陀螺转速下的跟踪/重投影效果。
#
# 用法:
#   tools/obs_gui.sh [gyro_speed_rad_s=3.0] [distance_m=3.0] [camera_preset=armor_camera_far]
# 例:
#   tools/obs_gui.sh            # 3 rad/s，3 m
#   tools/obs_gui.sh 5.0 6.5    # 5 rad/s，6.5 m
#
# 本脚本**不注入任何仿真/算法行为宏**：转速/距离/相机是场景参数（用命令行给），
# 其余一切（检测、PnP、Tracker、EKF、云台控制、开火……）全部来自 `configs/`。
# 唯一显式传的两个环境变量是"连哪台机器、用哪种协议"，它们必须和本脚本起仿真器
# 时用的传输方式一致（历史坑：configs/simulator.yaml 里写 udp 而脚本起的是 tcp，
# 视觉端会一直 "Waiting for simulator frames"）。
#
# 转速在仿真器**启动时**固定（DAEDALUS_SMALL_GYRO_SPEED），换转速要重跑本脚本。
# Ctrl-C 退出并关掉仿真器。
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SIM_DIR="$REPO/../simulator_system/simulator"
SPEED="${1:-3.0}"
DIST="${2:-3.0}"
CAM="${3:-armor_camera_far}"
RUN="/tmp/obs_gui"

[[ -x "$REPO/build_sim/auto_aim_sim" ]] || { echo "先构建 build_sim/auto_aim_sim" >&2; exit 2; }
mkdir -p "$RUN"

# 相机预设按"距靶距离"重新摆位（场景参数；不改其它任何量）。
camera_config="$REPO/tools/sim/${CAM}.yaml"
if [[ -n "$DIST" ]]; then
  camera_config="$RUN/camera.yaml"
  python3 - "$REPO/tools/sim/${CAM}.yaml" "$camera_config" "$DIST" <<'PY'
import math, re, sys
src, dst, want = sys.argv[1], sys.argv[2], float(sys.argv[3])
text = open(src).read()
block = re.search(r"(?ms)^armor_chassis:\s*\n((?:\s+.*\n)+)", text)
section = block.group(0)
pos = re.search(r"position:\s*\[([^\]]*)\]", section)
tgt = re.search(r"target:\s*\[([^\]]*)\]", section)
p = [float(v) for v in pos.group(1).split(",")]
q = [float(v) for v in tgt.group(1).split(",")]
norm = math.dist(p, q)
direction = [(p[i] - q[i]) / norm for i in range(3)]
new = [q[i] + direction[i] * want for i in range(3)]
new_text = section[:pos.start(1)] + ", ".join(f"{v:.4f}" for v in new) + section[pos.end(1):]
open(dst, "w").write(text.replace(section, new_text))
PY
fi

# 退出时连仿真器的子进程一起收干净（run_host.sh 会 `exec target/release/daedalus`，
# 但 cargo 构建阶段可能有子进程残留 —— 先收子进程再收父进程，最后补一记硬杀）。
cleanup() {
  local pid=""
  [[ -f "$RUN/sim.pid" ]] && pid="$(cat "$RUN/sim.pid")"
  if [[ -n "$pid" ]]; then
    pkill -TERM -P "$pid" 2>/dev/null || true
    kill -TERM "$pid" 2>/dev/null || true
    sleep 1
    pkill -KILL -P "$pid" 2>/dev/null || true
    kill -KILL "$pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT

( cd "$SIM_DIR" && \
  DAEDALUS_RUNE_ONLY=0 DAEDALUS_SMALL_GYRO=1 DAEDALUS_SMALL_GYRO_SPEED="$SPEED" \
  DAEDALUS_GIMBAL_MODE=vision \
  DAEDALUS_CAMERA_CONFIG="$camera_config" \
  DAEDALUS_CSV_PATH="$RUN/truth.csv" DAEDALUS_LIGHT_CSV_PATH="$RUN/light.csv" \
  DAEDALUS_CAPTURE_WIDTH=1440 DAEDALUS_CAPTURE_HEIGHT=1080 DAEDALUS_CAPTURE_FPS=60 \
  DAEDALUS_TCP_ADDR=127.0.0.1:7666 DAEDALUS_COMMAND_ADDR=127.0.0.1:7667 \
  DAEDALUS_TELEMETRY_ADDR=127.0.0.1:7668 \
  DAEDALUS_WINDOW_WIDTH=960 DAEDALUS_WINDOW_HEIGHT=720 \
  nohup ./run_host.sh > "$RUN/sim.log" 2>&1 & echo $! > "$RUN/sim.pid" )

for _ in $(seq 1 120); do nc -z 127.0.0.1 7666 2>/dev/null && break; sleep 1; done
if ! nc -z 127.0.0.1 7666 2>/dev/null; then echo "仿真器没起来:" >&2; tail -5 "$RUN/sim.log" >&2; exit 3; fi
echo "[obs] sim up: gyro=${SPEED} rad/s dist=${DIST} m；算法 GUI 启动中（Ctrl-C 退出）..."

# 传输方式：和上面起仿真器用的 tcp:7666 对齐；其余参数一律走 configs/。
ULTRA_VISION_SIM_HOST=127.0.0.1 \
ULTRA_VISION_SIM_PROTOCOL=tcp \
  "$REPO/build_sim/auto_aim_sim"
