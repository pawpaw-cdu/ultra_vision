#!/usr/bin/env bash
# auto_aim 仿真回归：起仿真器（armor_chassis 场景）→ 跑视觉端 → 出真值对比。
#
# 与 tools/rune_sim_test.sh 同构，区别是场景换成正对装甲板的固定相机
# （simulator/config/camera.yaml 里 scenario: armor_chassis），仿真器会把**真值**
# （四块板在相机系里的位置）写成 dataset.csv：
#   <out>/truth.csv      仿真真值（DAEDALUS_CSV_PATH）
#   <out>/estimate.csv   我方估计（ULTRA_VISION_ESTIMATE_CSV）
#   <out>/observation.csv 我方观测（含 PnP 的输出）
#   <out>/sim.log        仿真器日志
# 之后用 tools/truth_regression.py 做逐级误差对比。
#
# 用法：
#   tools/auto_aim_sim_test.sh [--seconds 30] [--uv 0|1] [--sigma 1.5] [--out /tmp/auto_aim]
#                               [--no-build]
#                               [--camera armor_camera|armor_camera_mid|armor_camera_far]
#                               [--distance 5.2]        # 沿视线挪相机，覆盖预置距离
#                               [--pan 0.8]             # 横向平移工况（±2 m 来回）
#                               [--pan-excursion 2.0] [--pan-lead 6.0]
#                               [--gyro 1]              # 靶车开小陀螺（旋转靶回归）
#                               [--qpos 10]             # 扫 process_noise_pos
#
# 三个工况对应的回归姿势（改滤波/瞄准任何一处，这三条都要跑）：
#   静止： --camera armor_camera_far
#   平移： --camera armor_camera_far --pan 0.8
#   旋转： --camera armor_camera_far --gyro 1
# 结论口径见 docs/auto_aim_aim_loop.md，量化工具 tools/auto_aim_aim_check.py。
#
# --uv 1 打开 UV（像素重投影）观测（configs/tracker.yaml 的 uv_observation），
# 这是本轮加的 A/B 开关；其余参数透传到视觉端的环境变量。
#
# 顺序（按使用习惯）：**先编译 Ultra_Vision**，再启动仿真器
# （simulator_system/simulator/run_host.sh，它自己会 cargo build 再运行）。
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SIM_DIR="${SIM_DIR:-$REPO_ROOT/../simulator_system/simulator}"
SIM_BIN="$SIM_DIR/target/release/daedalus"
AIM_BIN="${AIM_BIN:-$REPO_ROOT/build_sim/auto_aim}"

seconds=30
uv=0
compact=0
uv_height=""
camera_preset="armor_camera"
capture="1440x1080"
distance=""
pan=""
pan_excursion="2.0"
pan_lead="6.0"
gyro=0
qpos=""
sigma=""
out="/tmp/auto_aim_sim"
build=1
truth_override=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --seconds) seconds="$2"; shift 2 ;;
    --uv) uv="$2"; shift 2 ;;
    --camera) camera_preset="$2"; shift 2 ;;
    --capture) capture="$2"; shift 2 ;;
    --distance) distance="$2"; shift 2 ;;
    --pan) pan="$2"; shift 2 ;;
    --pan-excursion) pan_excursion="$2"; shift 2 ;;
    --pan-lead) pan_lead="$2"; shift 2 ;;
    # `--gyro 1`：让靶车开小陀螺（仿真里 3 rad/s ≈ 172 deg/s，和实车小陀螺同量级）。
    # 这是"旋转靶"回归工况；改滤波器 Q 的时候必须跑它，别把旋转档调坏。
    --gyro) gyro="$2"; shift 2 ;;
    --qpos) qpos="$2"; shift 2 ;;
    --compact) compact="$2"; shift 2 ;;
    --uv-height) uv_height="$2"; shift 2 ;;
    --sigma) sigma="$2"; shift 2 ;;
    --out) out="$2"; shift 2 ;;
    --truth) truth_override="$2"; shift 2 ;;
    --no-build) build=0; shift ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

[[ -x "$SIM_BIN" ]] || { echo "simulator not built: $SIM_BIN" >&2; exit 2; }
[[ -x "$AIM_BIN" ]] || { echo "auto_aim not built: $AIM_BIN (cmake --target auto_aim)" >&2; exit 2; }

# 1) 先把 Ultra_Vision 编译好，再起仿真器（避免"编译把正在跑的进程换掉"）。
if [[ "$build" == "1" ]]; then
  echo "building auto_aim ..."
  cmake --build "$REPO_ROOT/build_sim" -j8 --target auto_aim >/dev/null
fi

mkdir -p "$out"
run_dir="$out/uv$uv"
mkdir -p "$run_dir"

# 相机-靶车距离：默认用预置文件里的值；`--distance 5.2` 会沿着同一条视线把相机
# 挪到指定距离（写成 <out>/camera.yaml，不动仓库里的预置）。距离一变，
# 靶面像素尺寸(= f·h/d) 和 FOV 能覆盖的横向范围(≈ 2·d·tan(FOV/2)) 一起变，
# 所以"识别不到"和"平移出画"这两件事都要盯这个数。
camera_config="$REPO_ROOT/tools/sim/${camera_preset}.yaml"
if [[ -n "$distance" ]]; then
  camera_config="$run_dir/camera.yaml"
  python3 - "$REPO_ROOT/tools/sim/${camera_preset}.yaml" "$camera_config" "$distance" <<'PY'
import math, re, sys
src, dst, want = sys.argv[1], sys.argv[2], float(sys.argv[3])
text = open(src).read()
block = re.search(r"(?ms)^armor_chassis:\s*\n((?:\s+.*\n)+)", text)
if not block:
    sys.exit(f"{src}: 找不到 armor_chassis: 段")
section = block.group(0)
pos = re.search(r"position:\s*\[([^\]]*)\]", section)
tgt = re.search(r"target:\s*\[([^\]]*)\]", section)
if not (pos and tgt):
    sys.exit(f"{src}: armor_chassis 段缺 position/target")
p = [float(v) for v in pos.group(1).split(",")]
q = [float(v) for v in tgt.group(1).split(",")]
norm = math.dist(p, q)
direction = [(p[i] - q[i]) / norm for i in range(3)]
new = [q[i] + direction[i] * want for i in range(3)]
new_text = section[:pos.start(1)] + ", ".join(f"{v:.4f}" for v in new) + section[pos.end(1):]
open(dst, "w").write(text.replace(section, new_text))
print(f"camera distance {norm:.2f} -> {want:.2f} m, position {new}")
PY
fi

# `--pan 0.8`：横向平移回归工况（速度 m/s，默认 ±2 m 来回，先静止 6 s）。
# 为什么要"先静止 6 s"：视觉端比仿真器晚 3~4.5 s 起来（脚本 sleep 2 s + 建连 +
# 首帧），靶车要是已经在动，第一帧就在画面边缘、ROI 又是按那一帧初始化的，
# 很容易锁到地上的假板（实测：真值 6.5 m，锁出来的"板"在 1~2 m），
# 然后云台被假目标带着转 80~90deg 出画，整轮再也回不来 —— 就是"识别不到"的现场。
# 6 s 静止先让它在正中锁定，再开始扫。
# 为什么限幅 ±2 m：6.5 m / 41deg 的横向半宽只有 2.4 m，超了就出画。
if [[ -n "$pan" ]]; then
  if [[ -n "${DAEDALUS_MOVE_LOOP:-}" || -n "${DAEDALUS_MOVE_SCHEDULE:-}" ]]; then
    echo "警告：--pan 与 DAEDALUS_MOVE_* 同时给了，用 --pan。" >&2
  fi
  export DAEDALUS_MOVE_LOOP="$(python3 - "$pan" "$pan_excursion" "$pan_lead" <<'PY'
import sys
speed, excursion, lead = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
leg = excursion / speed
print(f"{lead:g},0,0;{leg:.3f},{speed},0;1,0,0;{leg:.3f},{ -speed},0;1,0,0")
PY
)"
  echo "平移工况: DAEDALUS_MOVE_LOOP=$DAEDALUS_MOVE_LOOP"
fi

# 仿真器：**已经在跑就用它**（绝不重复启动、也绝不去 pkill 别人的进程）；
# 没在跑才自己起一个，而且结束时只关自己起的这个。
started_sim=0
if nc -z 127.0.0.1 7666 2>/dev/null; then
  # 需要"脚本化运动"时**绝不能附着手动开的仿真器**：它不会带 DAEDALUS_MOVE_* 环境
  # 变量，于是运动根本不生效，测出来的其实是键盘操作（这是真踩过的坑）。
  if [[ -n "${DAEDALUS_MOVE_LOOP:-}" || -n "${DAEDALUS_MOVE_SCHEDULE:-}" ]]; then
    echo "错误：127.0.0.1:7666 上已有仿真器在跑，但它不会带 DAEDALUS_MOVE_*。" >&2
    echo "      请先关掉手动开的仿真器（或设 SIM_FORCE_LAUNCH=1 由脚本自己起）。" >&2
    if [[ "${SIM_FORCE_LAUNCH:-0}" != "1" ]]; then exit 4; fi
  fi
  echo "detected a running simulator on 127.0.0.1:7666 — attaching to it"
else
  (
    cd "$SIM_DIR"
    # 用仓库自带的 run_host.sh 启动（它先 cargo build 再 exec daedalus）；
    # 场景/相机位姿由 DAEDALUS_CAMERA_CONFIG 指向**本仓的相机预置**决定。
    DAEDALUS_RUNE_ONLY=0 \
    DAEDALUS_SMALL_GYRO="$gyro" \
    DAEDALUS_CAMERA_CONFIG="$camera_config" \
    DAEDALUS_CSV_PATH="$run_dir/truth.csv" \
    DAEDALUS_CAPTURE_WIDTH=${capture%x*} \
    DAEDALUS_CAPTURE_HEIGHT=${capture#*x} \
    DAEDALUS_WINDOW_WIDTH=960 \
    DAEDALUS_WINDOW_HEIGHT=720 \
    DAEDALUS_CAPTURE_FPS=30 \
    DAEDALUS_TELEMETRY_ADDR=127.0.0.1:7668 \
    DAEDALUS_TCP_ADDR=127.0.0.1:7666 \
    DAEDALUS_COMMAND_ADDR=127.0.0.1:7667 \
    ./run_host.sh > "$run_dir/sim.log" 2>&1 &
    echo $! > "$run_dir/sim.pid"
  )
  started_sim=1
fi

# 等仿真器把 TCP 口打开（run_host.sh 里可能还有一次 cargo build）。
for _ in $(seq 1 120); do
  if nc -z 127.0.0.1 7666 2>/dev/null; then break; fi
  sleep 1
done
if ! nc -z 127.0.0.1 7666 2>/dev/null; then
  echo "仿真器没起来（7666 未监听）" >&2
  [[ -f "$run_dir/sim.log" ]] && tail -n 20 "$run_dir/sim.log" >&2
  exit 3
fi
sleep 2

# UV 开关通过一份临时配置目录注入（不改动仓库里的 configs/）。
cp -R "$REPO_ROOT/configs" "$run_dir/configs"
python3 - "$run_dir/configs/tracker.yaml" "$uv" "$sigma" "$compact" "$uv_height" "$qpos" <<'PY'
import re, sys
path, uv, sigma, compact, uv_height, qpos = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6]
text = open(path).read()
text = re.sub(r"(?m)^(\s*uv_observation:\s*).*$", lambda m: m.group(1) + ("true" if uv == "1" else "false"), text)
text = re.sub(r"(?m)^(\s*uv_compact:\s*).*$", lambda m: m.group(1) + ("true" if compact == "1" else "false"), text)
if qpos:
    # 平移过程噪声（Q 的 q_a）A/B：平移靶跟不上时先扫这一个。
    text = re.sub(r"(?m)^(\s*process_noise_pos:\s*).*$", lambda m: m.group(1) + qpos, text)
if uv_height:
    text = re.sub(r"(?m)^(\s*uv_armor_height:\s*).*$", lambda m: m.group(1) + uv_height, text)
if sigma:
    text = re.sub(r"(?m)^(\s*uv_sigma_px:\s*).*$", lambda m: m.group(1) + sigma, text)
open(path, "w").write(text)
PY

(
  cd "$REPO_ROOT"
  env \
    ULTRA_VISION_CONFIG_DIR="$run_dir/configs" \
    ULTRA_VISION_NO_DISPLAY=1 \
    ULTRA_VISION_SIM_HOST=127.0.0.1 \
    ULTRA_VISION_TEST_SECONDS="$seconds" \
    ULTRA_VISION_ESTIMATE_CSV="$run_dir/estimate.csv" \
    ULTRA_VISION_OBSERVATION_CSV="$run_dir/observation.csv" \
    ULTRA_VISION_SELECTOR_CSV="$run_dir/selector.csv" \
    "$AIM_BIN" > "$run_dir/vision.log" 2>&1 || true
)

# 真值来源：自己起的仿真器写到 $run_dir/truth.csv；附着模式（仿真器是别人起的）
# 就用它自己写的那份（默认在 simulator/ 目录下的 dataset.csv，可用 --truth 指定）。
if [[ -n "$truth_override" ]]; then
  cp "$truth_override" "$run_dir/truth.csv"
elif [[ "$started_sim" == "0" && -f "$SIM_DIR/dataset.csv" ]]; then
  cp "$SIM_DIR/dataset.csv" "$run_dir/truth.csv"
fi

# 只关自己起的仿真器；附着模式下不动别人的进程。
if [[ "$started_sim" == "1" ]]; then
  kill "$(cat "$run_dir/sim.pid")" 2>/dev/null || true
fi
sleep 1

{
  echo "uv_observation : $uv"
  echo "seconds        : $seconds"
  echo "camera         : $camera_config"
  echo "输出目录       : $run_dir"
  echo
  echo "--- 真值对比（truth_regression.py）"
  python3 "$REPO_ROOT/tools/truth_regression.py" \
      --truth "$run_dir/truth.csv" \
      --observation "$run_dir/observation.csv" \
      --estimate "$run_dir/estimate.csv" \
      --selector "$run_dir/selector.csv" 2>&1 || true
  echo
  echo "--- 发火/跟踪摘要（vision.log 尾部）"
  tail -n 12 "$run_dir/vision.log" || true
}
