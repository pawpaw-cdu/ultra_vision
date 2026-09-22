# 分层依赖方向（B3）

> 目标：把 `src/auto_aim/` 的依赖方向**定死并可检查**，让"能力模块 + 薄组合根"和
> sp_vision 的 `tasks/ + src/*.cpp` 一一对应。检查脚本：`tools/dep_lint.sh`。

## 1. 层与职责

| 层 | 目录 | 职责 | 可以依赖 |
|---|---|---|---|
| 硬件驱动 | `io/` | 搬字节/帧：相机（Hik/Galaxy）取图、串口收发、云台协议成帧 | 只依赖 `common/` 与 3rd-party SDK |
| 检测 | `perception/` | 灯条/神经网络检测、PnP、动态 ROI（`armor_source`） | `common/`、`Kalman/`（类型）、`visualization/`（投影） |
| 估计 | `Kalman/` | 观测关联、EKF、状态推进（`tracker`/`armor_ekf`） | `common/`、`perception/`（PnP 原语） |
| 控制 | `control/` | 选板、提前量、云台轨迹、开火判据（`aim_pipeline`） | `Kalman/`、`control/`、`io/gimbal`（协议帧） |
| 可视化 | `visualization/` | 投影与曲线（纯函数/绘图） | `Kalman/`（数据类型） |
| 配置 | `config/` | YAML → 各层配置结构体 | `Kalman/`、`control/`（只读它们的 Config 类型） |
| 组合 | `app/` | **唯一**知道所有层的地方：装配、CSV/曲线/叠加显示、仿真入口阶段 | 以上全部 |
| 入口 | `node_*.cpp` | 只做 IO：解析配置目录、建 `app/` 对象、跑主循环 | `app/`、`io/`、`common/` |

## 2. 依赖方向（单向）

```
io/  →  perception/  →  Kalman/  →  control/  →  io/(下发帧)
             │              │           │
             └───── visualization/ ◄────┘
                        config/  （只读配置类型）
                          app/   （装配以上全部）
                     node_*.cpp  （只调 app/ + io/）
```

一句话：**数据往上（io→perception→Kalman→control），命令往下（control→io 下发帧），
`app/` 是唯一的横切装配点，入口只做 IO。**

## 3. 硬规则（`tools/dep_lint.sh` 会失败的情况）

1. `perception/` 不得 `#include "control/..."` —— 检测不得依赖控制。
2. `perception/` 不得 `#include "io/..."` —— 检测不得依赖硬件。
3. `Kalman/` 不得 `#include "control/..."` —— 估计不得依赖控制。
4. `Kalman/` 不得 `#include "io/..."` —— 估计不得依赖硬件。
5. `io/` 不得 `#include` 任何上层（`perception/ Kalman/ control/ app/ visualization/`）。
6. `node_*.cpp` 不得直接 `#include "Kalman/armor_ekf.hpp"`，也不得调用
   `setNoise/setProcessNoise/setMeasurementNoise/...` —— 滤波器参数属于算法层。

## 4. 软规则（只提示）

- `node_*.cpp` 行数（thin composition root 目标 ≤300 行）。
- `node_*.cpp` 直接 include 算法层头文件（应通过 `app/` 暴露的接口）。

## 5. 当前的跨层边（lint 扫描结果，2026-10）

| 来源层 | 去向 | 是否允许 | 说明 |
|---|---|---|---|
| `perception/` | `Kalman/tracker.hpp` | ✓ | 检测需要 `TrackerState`/`Armor` 类型做动态 ROI |
| `perception/` | `visualization/projection.hpp` | ✓ | PnP 用投影原语 |
| `Kalman/` | `perception/pnp_solver.hpp` | ✓ | `tracker` 用 `getArmor3DPoints` 精修位姿 |
| `control/` | `Kalman/tracker.hpp` | ✓ | 控制读估计状态 |
| `control/` | `io/gimbal/gimbal.hpp` | ✓ | 下发帧格式（`gimbal_command_frame.hpp`） |
| `config/` | `Kalman/`、`control/` | ✓ | 只读 Config 类型 |
| `app/` | 全部 | ✓ | 组合层 |

没有发现反向依赖；`io/` 对上层零依赖。

## 6. 入口收尾（A1 / A2）

- `node_sim.cpp`（150 行）：只 include `app/`，主循环在入口、阶段在 `app/sim_runner`。
- `node.cpp`（17 行）：只 include `app/`，真机逻辑在 `app/real_runner`。
- 两个入口现在都只依赖 `app/`（+ 标准库/OpenCV）；`Kalman/`/`perception/`/`control/`
  的 include 都在能力层与 `app/` 内部。
- 下一步（NUC 验证）：把 `app/real_runner` 的 `run()` 拆成
  `initialize()` + 阶段方法，见 `docs/real_entry_refactor_plan.md`。
