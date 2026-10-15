#!/usr/bin/env bash
# 观测模式：弹两个窗口（仿真器预览 + 算法端 "Ultra Vision"），相机静止（视觉端不接管
# 云台，ULTRA_VISION_DISABLE_CONTROL=1），用于看不同小陀螺转速下的观测/跟踪效果。
# OBS_GUI_CONTROL=1 时**不冻结云台**：视觉端按正常闭环接管云台，用于看"追板/守角"
# 实际跟得上跟不上（云台动了、目标可能出画，这是正常现象）。
#
# 用法:
#   tools/obs_gui.sh [gyro_speed_rad_s=3.0] [distance_m=3.0] [camera_preset=armor_camera_far]
# 例:
#   tools/obs_gui.sh 2.0          # 2 rad/s
#   tools/obs_gui.sh 5.0 6.5      # 5 rad/s，6.5 m
#   ULTRA_VISION_LOCAL_FIT=1 ULTRA_VISION_LOCAL_FIT_ALPHA_DEG=0 tools/obs_gui.sh 3.0   # 同时开局部拟合器
#   OBS_GUI_CONTROL=1 tools/obs_gui.sh 3.0                       # 云台不冻结，闭环跟随
#
# 转速在仿真器**启动时**确定（DAEDALUS_SMALL_GYRO_SPEED，进程内固定），换转速要重跑本脚本。
# Ctrl-C 退出并关掉仿真器。
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SIM_DIR="${SIM_DIR:-$REPO/../simulator_system/simulator}"
SPEED="${1:-3.0}"
DIST="${2:-3.0}"
CAM="${3:-armor_camera_far}"
RUN="${RUN:-/tmp/obs_gui}"
# 云台开关：默认冻结（观测模式，相机不动）。OBS_GUI_CONTROL=1 → 不设
# ULTRA_VISION_DISABLE_CONTROL，视觉端按正常闭环下发云台指令（跟得上/跟飞都看得见）。
OBS_GUI_CONTROL="${OBS_GUI_CONTROL:-0}"

[[ -x "$REPO/build_sim/auto_aim_sim" ]] || { echo "先构建 build_sim/auto_aim_sim" >&2; exit 2; }
mkdir -p "$RUN"
# ⚠ 这里必须用 `configs/.`（把**内容**拷进已存在的目录）。写成 `cp -R configs "$RUN/configs"`
# 时：RUN 目录第一次是新建的没问题，但**第二次运行（目录已存在）会拷成
# $RUN/configs/configs/**，而视觉端读的 $RUN/configs/tracker.yaml 还是**上一次的旧配置**。
# 踩过的坑：2026-10-10 之后 hard_plate_constraint 改成了 false，但 /tmp/obs_gui 里那份
# 陈旧配置一直是 true —— 实测同一命令、同转速（5rad/s）：
#     陈旧(true)  模型板误差 41.2/102.6px(p50/p90)，车心偏差中位 69.8px，>60px 占 55.6%
#     新鲜(false) 模型板误差 19.2/ 38.0px(p50/p90)，车心偏差中位  7.1px，>60px 占  5.1%
# 也就是"重投影的模型整块跑到机器人旁边"那种现象。
rm -rf "$RUN/configs/configs" 2>/dev/null || true
cp -R "$REPO/configs/." "$RUN/configs/"

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

cleanup() { [[ -f "$RUN/sim.pid" ]] && kill "$(cat "$RUN/sim.pid")" 2>/dev/null || true; }
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
if [[ "$OBS_GUI_CONTROL" == "0" ]]; then
  echo "[obs] sim up: gyro=${SPEED} rad/s dist=${DIST} m; 视觉端 GUI（云台冻结，只观测）..."
else
  echo "[obs] sim up: gyro=${SPEED} rad/s dist=${DIST} m; 视觉端 GUI（**云台不冻结**，闭环接管，会跟着转）..."
fi

# 协议要**两侧一致**：本脚本起仿真器时没有给 DAEDALUS_UDP=1（= TCP），但
# configs/simulator.yaml 里写的是 udp ⇒ 视觉端会去订阅 UDP 7669 而永远等不到帧
# （现象就是窗口里一直 "Waiting for simulator frames..."）。所以这里默认钉 TCP，
# 想用 UDP 就： DAEDALUS_UDP=1 ULTRA_VISION_SIM_PROTOCOL=udp tools/obs_gui.sh …
# ⚠ 注释必须写在整条命令**之前**：把 # 塞进反斜杠续行的中间，会把后面（包括
#   启动命令本身）整段注释掉，而日志文件还是上一轮的残留 —— 看起来"跑了其实没跑"。
# 云台开关只影响**视觉端**：DISABLE_CONTROL 存在时视觉端不发云台命令（相机静止）。
export ULTRA_VISION_CONFIG_DIR="$RUN/configs"
export ULTRA_VISION_SIM_HOST=127.0.0.1
export ULTRA_VISION_SIM_PROTOCOL="${ULTRA_VISION_SIM_PROTOCOL:-tcp}"
export ULTRA_VISION_ESTIMATE_CSV="$RUN/estimate.csv"
export ULTRA_VISION_OBSERVATION_CSV="$RUN/observation.csv"
export ULTRA_VISION_SELECTOR_CSV="$RUN/selector.csv"
if [[ "$OBS_GUI_CONTROL" == "0" ]]; then
  export ULTRA_VISION_DISABLE_CONTROL=1
else
  # 调用方 shell 里可能残留着这个变量，显式清掉，否则"不冻结"会静默失效。
  unset ULTRA_VISION_DISABLE_CONTROL
fi
"$REPO/build_sim/auto_aim_sim" > "$RUN/vision.log" 2>&1
