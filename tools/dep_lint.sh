#!/usr/bin/env bash
# 分层依赖 lint（B4）：用 rg 把 B3 定死的依赖方向钉住。
#
# 依赖方向（docs/layer_dependencies.md）：
#     io/            最低层：只搬字节/帧，不 include 任何上层
#     perception/    检测 + PnP（可依赖 Kalman 的类型 + visualization 的投影）
#     Kalman/        观测/估计（只依赖 perception 的 PnP 原语 + common）
#     control/       选择/控制（依赖 Kalman；下发经 io/gimbal 的协议帧）
#     visualization/ 纯投影/绘图（只依赖 Kalman 的数据类型）
#     app/           组合层：装配以上各层（唯一的"知道所有层"的地方）
#     node_*.cpp     入口：只做 IO + 调 app/
#
# 硬规则（违反 → 退出码 1）：perception→control、perception→io、Kalman→control、
#   Kalman→io、io→上层、node_*.cpp 直接碰滤波器核心（armor_ekf / setNoise 等）。
# 软规则（只提示，不失败）：node_*.cpp 过大、node_*.cpp 直接 include 算法层头文件。
#
# 用法：tools/dep_lint.sh        # CI / 本地都能跑，无副作用
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$REPO_ROOT/src/auto_aim"
fail=0

hard() { # layer_dir  regex  message
    local hits
    hits="$(rg -n --no-heading "$2" "$SRC/$1" 2>/dev/null || true)"
    if [[ -n "$hits" ]]; then
        echo "✗ [硬规则] $3"
        echo "$hits" | sed 's/^/    /'
        fail=1
    else
        echo "✓ $3"
    fi
}

warn() { # layer_dir  regex  message
    local hits
    hits="$(rg -n --no-heading "$2" "$SRC/$1" 2>/dev/null || true)"
    if [[ -n "$hits" ]]; then
        echo "⚠ [软规则] $3"
        echo "$hits" | sed 's/^/    /'
    else
        echo "✓ $3"
    fi
}

echo "== 分层依赖 lint（${SRC}） =="

# ---- 硬规则：跨层反向依赖 -------------------------------------------------
hard perception '#include\s+"control/'  "perception/ 不得 include control/（检测不得依赖控制）"
hard perception '#include\s+"io/'       "perception/ 不得 include io/（检测不得依赖硬件）"
hard Kalman     '#include\s+"control/'  "Kalman/ 不得 include control/（估计不得依赖控制）"
hard Kalman     '#include\s+"io/'       "Kalman/ 不得 include io/（估计不得依赖硬件）"
hard io         '#include\s+"(perception|Kalman|control|app|visualization)/' \
                "io/ 是最低层，不得 include 任何上层"

# ---- 硬规则 + 软规则：入口（node_*.cpp）-----------------------------------
entry_hard() { # regex  message
    local hits
    hits="$(rg -n --no-heading "$1" "$SRC"/node_*.cpp 2>/dev/null || true)"
    if [[ -n "$hits" ]]; then
        echo "✗ [硬规则] $2"
        echo "$hits" | sed 's/^/    /'
        fail=1
    else
        echo "✓ $2"
    fi
}
entry_hard '#include\s+"Kalman/armor_ekf\.hpp"' \
           "node_*.cpp 不得直接 include 滤波器核心（走 AimPipeline/Tracker）"
entry_hard '(setNoise|setProcessNoise|setMeasurementNoise|setQ\(|setR\()' \
           "node_*.cpp 不得直接设滤波器噪声参数（属于算法层）"

# ---- 软规则：入口厚薄（thin composition root）------------------------------
for f in "$SRC"/node_sim.cpp "$SRC"/node.cpp; do
    [[ -f "$f" ]] || continue
    n="$(wc -l < "$f" | tr -d ' ')"
    tag="✓"
    [[ "$n" -gt 300 ]] && tag="⚠"
    printf '    %s %-28s %s 行（薄入口目标 ≤300）\n' "$tag" "${f#$SRC/}" "$n"
done

echo
if [[ "$fail" -ne 0 ]]; then
    echo "dep_lint: 有硬规则违反，见上。"
    exit 1
fi
echo "dep_lint: 硬规则全过。"
