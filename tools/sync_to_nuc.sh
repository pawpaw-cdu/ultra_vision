#!/usr/bin/env bash
# 把本仓库同步到远端（NUC / 工控机 / VM），**逐目录映射、保持结构**。
#
# 为什么单独写一个脚本：`rsync -ac src/ io/ configs/ tools/ <远端>/` 这种"多个源 +
# 目录目标"的写法，会把 configs/、tools/ 的**内容**平铺到远端根目录，把源码和配置
# 撒一地（2026-09-27 真踩过，NUC 根目录出现了一堆 hsv_tuner.cpp / camera.yaml ...）。
# 正确写法是每个目录各自 rsync 到同名目录（本脚本做的事）。
#
#   tools/sync_to_nuc.sh [目标] [--dry-run] [--delete]
#     目标默认 nuc:~/Ultra_Vision
#     --delete 会删掉远端多余文件（默认不删，避免误伤现场产物）
#
# 不同步的东西：构建目录、git、运行产物（dataset.csv / *.jpg / *_probe.png 等）。
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="nuc:~/Ultra_Vision"
EXTRA_ARGS=()
for arg in "$@"; do
  case "$arg" in
    --dry-run) EXTRA_ARGS+=(--dry-run --itemize-changes) ;;
    --delete) EXTRA_ARGS+=(--delete) ;;
    -*) EXTRA_ARGS+=("$arg") ;;
    *) DEST="$arg" ;;
  esac
done

EXCLUDES=(
  --exclude 'build*/'
  --exclude '.git/'
  --exclude '.DS_Store'
  --exclude '__pycache__/'
  --exclude 'dataset.csv'
  --exclude 'CMakeFiles/'
  --exclude '*.jpg'
  --exclude '*.png'
  --exclude 'hand_eye.yaml'
  --exclude 'camera_intrinsics.yaml'
)

# 逐目录/逐文件，目标路径与源一一对应
ITEMS=(src io tools configs common docs tests models CMakeLists.txt README.md .gitignore)
for item in "${ITEMS[@]}"; do
  [[ -e "$REPO_ROOT/$item" ]] || continue
  echo "[sync] $item → $DEST/$item"
  if [[ -d "$REPO_ROOT/$item" ]]; then
    # 目录**必须两端都带尾斜杠**：否则 rsync 会把目录本身塞进目标，变成 src/src/
    rsync -ac "${EXCLUDES[@]}" ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} \
      "$REPO_ROOT/$item/" "$DEST/$item/"
  else
    rsync -ac "${EXCLUDES[@]}" ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} \
      "$REPO_ROOT/$item" "$DEST/$item"
  fi
done
echo "[sync] 完成。远端构建：ssh ${DEST%%:*} 'cd ~/Ultra_Vision && cmake --build build -j8'"
