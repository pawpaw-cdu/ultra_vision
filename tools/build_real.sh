#!/usr/bin/env bash
# 真机模式一键构建：海康 MVS 相机 + C 板串口协议。
#
#   tools/build_real.sh [构建目录]        # 默认 build_real
#   tools/build_real.sh build             # NUC 上把默认目录 build/ 配成真机模式
#
# 和仿真模式的区别只有两点：
#   · -DULTRA_VISION_AUTO_AIM_SIMULATOR=OFF → 编 node.cpp（真机入口）而不是 node_sim.cpp；
#   · -DULTRA_VISION_USE_HIK_CAMERA=ON     → 编 io/camera/HikCamera.cpp（MVS SDK）。
# 两者共用同一套 configs/ 与同一套估计/控制/开火代码，切换只影响 IO 层。
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${1:-$REPO_ROOT/build_real}"
MVS_ROOT="${ULTRA_VISION_HIK_SDK_ROOT:-/opt/MVS}"

# OpenVINO（若用神经网络检测器）需要先 source setupvars.sh，否则会被自动跳过。
if [[ -z "${OpenVINO_DIR:-}" ]] && [[ -f /opt/intel/openvino_2024.6.0/setupvars.sh ]]; then
  # setupvars.sh 里有未定义变量，碰到 set -u 会直接退出，这里临时放开
  set +u
  # shellcheck disable=SC1091
  source /opt/intel/openvino_2024.6.0/setupvars.sh >/dev/null
  set -u
fi

echo "[build_real] 构建目录 $BUILD_DIR，MVS SDK $MVS_ROOT"
cmake -S "$REPO_ROOT" -B "$BUILD_DIR" \
  -DULTRA_VISION_AUTO_AIM_SIMULATOR=OFF \
  -DULTRA_VISION_USE_HIK_CAMERA=ON \
  -DULTRA_VISION_HIK_SDK_ROOT="$MVS_ROOT" \
  -DULTRA_VISION_BUILD_AUTO_BUFF=OFF \
  -DULTRA_VISION_BUILD_RUNE=OFF
cmake --build "$BUILD_DIR" -j"$(nproc 2>/dev/null || sysctl -n hw.ncpu)" \
  --target auto_aim hik_probe gimbal_probe
echo "[build_real] 完成：$BUILD_DIR/auto_aim"
