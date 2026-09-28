# 能量机关（auto_buff）— 当前架构

> 2026-09-24 重构：**与自瞄拆成两个平行任务**，结构对齐 sp_vision：
>
> ```
> src/auto_aim/      自瞄（原样保留）
> src/auto_buff/     能量机关：buff_demo（薄入口）+ rune_*（算法）+ support/
> tests/             单元测试（buff_target_test / buff_state_test），不进算法目录
> tools/             脚本 + buff_model_bench（模型选型探针）
> ```
>
> 构建目标（对齐 sp_vision 的 `tasks/auto_buff`：算法是 **OBJECT 库**、入口是薄 executable）：
>
> | 目标 | 类型 | 内容 |
> |---|---|---|
> | `auto_buff` | OBJECT 库 | 算法本体 + 诊断层（可被任意入口/测试复用） |
> | `auto_buff_demo` | executable | 薄入口：参数/环境 → 配置 → `runRune()` |
> | `buff_target_test` / `buff_state_test` | 测试 | 顶层 `tests/`，`ctest -R buff` 运行 |
> | `buff_model_bench` | 工具 | 模型延迟/输入尺寸选型 |
>
> 旧的 `src/auto_buff/energy_node.cpp` + 那一代 `energy_demo`（华南虎式 demo、硬编码
> 路径、只吃本地视频）**已删除**（git 历史里可查），它与现在的流水线是同一件事的两代。

## 1. 一句话

**五点关键点网络 → 选靶（亮点判据 + 锁定滞回）→ PnP 位姿 → EKF（圆心/相位）→
弹道解算 → 100 Hz 云台轨迹 + 开火许可**；激活状态机用"图像相位 + 命中反馈"记账。
模型不重训：直接用 `models/openvino/RuneDetectionModel-master/model/model-0624.onnx`
（3 类 + 5 点，输入 640×480、归一化 0~1）。

## 2. 运行时拓扑

```
┌──────────────────────┐                ┌────────────────────────────────────────────┐
│ daedalus（Rust 仿真）│  7666 TCP 帧   │ rune_demo（C++ 视觉 + 火控）               │
│ 640×480 @30fps q95   │ ─────────────► │  ①帧接收线程：解码 JPEG → 最新帧           │
│                      │                │  ②遥测线程：RUNE 事件行 → 队列             │
│ 指令 7667 TCP        │ ◄───────────── │  ③云台线程 100 Hz：轨迹生成 + 下发         │
│ 遥测 7668 TCP        │ ─────────────► │  ④主线程 ~20-24 fps：视觉/火控主循环       │
└──────────────────────┘                └────────────────────────────────────────────┘
```

| 线程 | 频率 | 职责 | 与其他线程的耦合 |
|---|---|---|---|
| ① 帧接收 | 跟仿真 30 fps | TCP 收流、JPEG 解码、发布"最新帧" | 主线程取走一帧 |
| ② 遥测 | 事件驱动 | 解析命中/状态行，算出 `本地时钟−仿真时钟` 偏移 | 主循环每帧 `poll()` |
| ③ 云台 | 100 Hz | 加加速度受限轨迹 + 速度前馈，发 `GIMBAL` 行 | 主循环只更新目标角 |
| ④ 主循环 | 20~24 fps（受推理限制） | 检测 → 选靶 → PnP → EKF → 瞄准 → 激活 → 开火 → CSV/显示 | 取最新帧，不阻塞 ②③ |

实测：单帧 p50 **41 ms**（OpenVINO CPU 推理占绝大部分），即仿真给 30 fps 我们只跑 20~24 fps。

## 3. 模块与职责

| 文件 | 职责 | 关键点 |
|---|---|---|
| `rune_model.*` | OpenVINO 推理 + 解码（`[1,18,6300]`：3 类分 + 5×(x,y,conf)） | 归一化 0~1；坐标映射回原图；中心距 NMS 30 px；`lastBestScore()` 供诊断 |
| `rune_types.*` | `FanBlade`（四点+靶心+R 标）/ `PowerRune`（5 个 72° 格点、相位观测） | `phase_rad` = 平面内"圆心→靶心"向量角 |
| `rune_detector.*` | 候选 → 一个扇叶：亮点判据（小窗峰值−大窗均值）+ 锁定滞回 + 类别过滤 | 输出 `litBladeCenters()`、`classCounts()`、`targetClass()` |
| `rune_solver.*` | 靶面四点 PnP（IPPE）→ 世界系圆心/姿态/球坐标 | 双解按"圆心反投影落在 R 标上"选；深度用 `focal·0.7/像素偏移` 校正 |
| `rune_target.*` | EKF：小符 7 维（yaw/pitch/dis/roll/ω…）、大符 10 维（正弦 ω 模型） | 两条测量：①圆心+相位 ②靶心位置；含野值门（距离/圆心跳变） |
| `rune_aimer.*` | 两遍弹道解 + 瞄准角滤波 → `yaw/pitch/fly_time/shoot` | `predict_time` + `fire_gap_time` 节流；换靶时重置节流 |
| `rune_round_guard.*` | 一轮记账：窗口（`round_window_s`）、类别命中（class0→class1/2）、命中后保持 | 输入图像相位 + 命中反馈；输出 `blade_lit/round_age/hold/class_hit` |
| `rune_slot_lattice.*` | 槽位格点：单扇叶锚定 + 每帧重新对齐 + 已激活记账（丢帧不重锚） | 输出 `slot_id` 与 5 位 `activated` 掩码（进 CSV） |
| `rune_aim_fallback.*` | 丢目标策略：停最后已知符心 → 兜底回标定位姿 + 复位估计器 | `park_on_center_max_age_s`、`recenter_after_frames` |
| `rune_aim_bridge.*` | 帧循环 ↔ 100 Hz 控制线程的瞄准桥 | 帧循环只发布状态；开火请求由控制线程判定并回取 |
| `rune_config.hpp` | `configs/buff.yaml` → 各结构体 | 见 §6 |
| `rune_node.cpp` | 入口：仿真 TCP 或离线视频；CSV/调试/可视化 | 见 §7 环境变量 |
| `support/` | EKF、正弦拟合、球坐标、弹道等纯数学 | 无 IO，可单测 |

## 4. 数据流（逐级）

```
帧(640×480) ─► RuneModel.detect ─► 候选[cls, conf, 5×kpt, center]
                                          │
                    RuneDetector.buildRune（类别过滤 → 亮点排序 → 锁定滞回）
                                          ▼
                                   FanBlade + r_center(R 标)
                                          │
                    RuneSolver.solve（PnP + 双解选择 + 深度校正 + 相位观测）
                                          ▼
                    PowerRune{xyz/ypd/ypr, blade_xyz/blade_ypd, phase_rad}
                                          │
                    RuneTarget.getTarget（野值门 → EKF 两条测量 → 状态）
                                          ▼
        RuneAimer.aim（预测 → 弹道 → yaw/pitch/fly_time/shoot）
                                          │
        RuneRoundGuard.update（一轮窗口/类别命中/命中保持）
        RuneSlotLattice.update（槽位记账 → slot_id + activated 掩码）
                                          ▼
        GimbalController（100 Hz 轨迹）─► GIMBAL/FIRE ─► 仿真器；CSV/曲线/画面
```

每级的失败行为：候选为空 → `TEMP_LOST`（≤20 帧）→ `LOST`；解算失败/野值 → 滑行（coast）；
没有可用目标 → 先停在最后已知符心（`park_on_center_max_age_s=1.5 s` 内），
连续 15 帧仍无目标才回标定位姿并复位估计器（`rune_aim_fallback`）。

## 5. "谁决定什么"（决策点分布）

| 决策 | 在哪 | 依据 |
|---|---|---|
| 用哪些候选 | `rune_detector.cpp` | `score_threshold`、`min_plate_radius_px`、`require_inactive_class` |
| 锁哪一片 | 同上 | 亮点判据 + `lock_break_px` / `lock_max_miss` 滞回 |
| 圆心/相位 | `rune_solver.cpp` | PnP 双解选择 + `phase_rad` 几何相位 |
| 状态是否可用 | `rune_target.cpp` | EKF 野值门 + 发散保护（小符 |ω|≈π/3） |
| 何时换叶/开火 | `rune_aimer.cpp` | `switch_angle_deg`、`fire_gap_time`、云台到位判据 |
| 本轮是否有效 | `rune_round_guard.cpp` | 点亮窗口 + 类别命中（class0→class1/2）+ 命中保持 |
| 瞄的是第几片 / 哪几片已打 | `rune_slot_lattice.cpp` | 单扇叶锚定 + 每帧重新对齐（丢帧不重锚）+ 投影槽位最近邻归属 + 投票阈值 + 回合复位 |
| 丢目标时云台怎么办 | `rune_aim_fallback.cpp` | 最后已知符心（1.5 s 内）优先，兜底回标定位姿 |
| 是否卡开火 | `rune_round_guard.cpp` + `rune_aim_bridge.cpp` | 类别闸门、`hold`、云台到位误差（控制线程判定） |

> 注意：这条链上"打哪一片/能不能打"的判断分散在上述多处，是当前最主要的可维护性
> 与可解释性风险（见 `docs/energy_rune_architecture.md` §5 A3）。

## 6. 配置（`configs/buff.yaml`）

| 段 | 内容 | 备注 |
|---|---|---|
| `camera` | 内参、畸变、标定分辨率、自动缩放 | 与仿真 640×480 / FOV45° 对齐 |
| `model` | 路径、输入尺寸、布局、类别数、关键点数、阈值、`input_scale`、`input_pad_scale` | `input_scale=0.00392`（0~1）；`plate_point_order` 标定物点顺序 |
| `detector` | 选靶与锁定参数、`multi_candidate`、`require_inactive_class`、圆心搜索兜底 | 亮点判据窗口就在这一段的代码里 |
| `geometry` | 回转半径、靶面半宽、深度校正、距离闸门、相机→云台外参 | 物理量，实车需重测 |
| `aimer` | 提前时间、弹速、云台角包线、换叶阈值、开火节流 | `fire_gap_time` 当前 0.7 s |
| `activation` | 窗口、总目标数、`gate_fire` | 规则层 |
| `gimbal` | 速度/加速度/加加速度限幅、位置环/前馈增益、下发频率 | 与 `configs/tracker.yaml` 同义 |
| `aim_filter` | 瞄准角递归滤波（过程噪声/观测噪声/重置阈值） | 抹平换叶尖点 |

## 7. 运行、调试与评估

```bash
# 端到端仿真回归（自动起停仿真器 + 统计命中/激活）
tools/rune_sim_test.sh --preset low --mode small --seconds 30 --repeat 2
tools/rune_sim_test.sh --preset low_large --mode large --seconds 30 --pad 1.36 --record-frames

# 曲线（可用性 / 相位 / 距离 / 每轮命中），是"误差来源"的第一手数据
python3 tools/rune_curve.py <run_dir> --truth-distance 6.05

# 单次跑分（检出率 / 延迟 / 距离）
python3 tools/rune_model_score.py <run_dir> --truth-distance 6.05

# 槽位记账 vs 仿真真值（掩码一致率 / 闸门误挡）
python3 tools/rune_state_agreement.py <run_dir>

# 用录下来的 CSV 离线扫描格点参数（不必每次跑仿真）
python3 tools/rune_lattice_replay.py <run_dir> --sweep --assign nearest

# 纯算法自检（无仿真、无模型）
# 注意：需要带 -DULTRA_VISION_BUILD_TESTS=ON 配置构建目录，否则没有该目标：
#   cmake -S . -B build_rune -DULTRA_VISION_BUILD_RUNE=ON -DULTRA_VISION_BUILD_TESTS=ON
./build_rune/src/auto_buff/rune_target_test
./build_rune/src/auto_buff/rune_pipeline_state_test

# 模型延迟/输入尺寸选型
./build_rune/src/auto_buff/rune_model_bench models/openvino/....xml --image x.png

# 扇叶状态分类（经典面积特征）的**离线标定探针**：编译同一份
# rune_blade_state.cpp，只把 hub/轨道/靶心换成人工标注（网络检不到的"已激活"
# 外观只能人标），于是两张标注帧的量在同一条代码路径上、可以直接比
./build_rune/buff_state_probe /tmp/frame.png --hub 947,440 --orbit 125 \
    --blade 970,320 --blade 843,377 --dump /tmp/probe.png
# 也可以只给相位，自动摆 5 片：--n5 11   （相位 0 = 正上方，顺时针）

# 注意：状态分类的阈值**尚未可信**（两类实测重叠，见
# docs/energy_rune_issue_audit.md §46），现在只写进 CSV 观察，不接开火闸门。
```

常用环境变量：`ULTRA_VISION_RUNE_CSV`（逐帧 CSV）、`ULTRA_VISION_RUNE_DEBUG=1`
（打印 `[frame]/[rune detect]/[rune pick]/[fire]` 轨迹）、`ULTRA_VISION_NO_DISPLAY=1`、
`ULTRA_VISION_RUNE_DEBUG_ALL=1`（让 `[rune decode] best class score` **每帧**都打，
排查"网络到底看没看见"时必须开，否则 30 帧取模会把结论带偏）、
`ULTRA_VISION_DISABLE_FIRE=1`、`ULTRA_VISION_RUNE_MODE=small|large`、
`ULTRA_VISION_RUNE_TEST_SECONDS`、`ULTRA_VISION_RUNE_VIDEO=<mp4>`（离线回放）、
`ULTRA_VISION_SIM_HOST/PORT/COMMAND_PORT/TELEMETRY_PORT`、`ULTRA_VISION_SIM_RESET=1`。

## 8. 坐标与符号约定

- 估计在世界系完成：z-up，yaw 绕 z，pitch 向上为正；原点/朝向取标定时相机位姿
  （仿真里 `GIMBAL 0 0`，正对圆心）。
- 下发角度 `yaw_cmd = -yaw_world`、`pitch_cmd = +pitch_up`（仿真正 yaw 向右、正 pitch 向上），
  换算集中在 `RuneAimer::aim()`。
- 相位：`roll=0` 表示靶心在画面 12 点方向，增大为顺时针；几何相位与 PnP roll
  实测只差 0.4°（std），EKF 跟踪误差 2.6°（std）。

## 9. 已知限制

- 几何是**单帧**的（无平面/圆心时间窗），观测距离逐帧 5.8~7.7 m（真值 6.05）；
- 感知全串行，帧率被推理锁定（20~24 fps，仿真给 30 fps）；
- 时间基准不统一：位姿快照用"到达时刻"而非曝光时刻，提前量不含推理时间（~40 ms 欠补偿）；
- 失目标时没有"瞄符心"保底，盲目期 1~2 s；
- 开火是固定节流 + 布尔闸门，没有"命中时刻"概念（实测曾 48% 的弹在本轮失败后落地）；
- 选靶/类别/亮点判据含有针对仿真的域假设，迁移到实车需重新标定。
- **召回的主控量是"机关在网络输入里的大小"**：同一个视频换画布缩放系数，召回能
  差 4~9 倍（buff_two 2%↔18%、buff_three 61%↔81%、仿真 4%↔90%）。三条最优曲线
  都落在"输入里轨道半径 ≈ 50~60 px"（实拍 ~60、仿真 ~50）。
  `input_pad_adaptive: true` 可以自动把它钉住（实测与手调的各自最优等价），
  但**默认关**：默认 `input_pad_scale: 1.6` 按实拍定，跑仿真时用
  `--pad 1.3` 或把 target 设 52。详见 `docs/energy_rune_issue_audit.md` §47。
- **新模型对"已激活"外观是盲的**：实拍里从"打完一片"那一刻起，网络对已激活片的
  类别分长期 ≤0.01（不是被阈值筛掉），只对未激活目标片出框。所以已激活片的几何
  只能靠槽位格点/相位外推补，不能指望"网络给 ROI + 经典判状态"。见 §48。
- **没有 class0 观测时不再"救回"已激活片**（`rescue_missing_inactive: false`）：
  三家参考实现（sp_vision 按亮片数+位置连续性选靶、rm_vision_core 在 inactive 为空
  时本帧不给目标、RP-26Rune 要求至少一片未激活才做整符配准）都不允许已激活片成为
  瞄准目标。改前多片已激活时观测跳变 0.9/0.3 次每秒、改后 0.3/0.1；仿真小符有效
  命中 17 → 21、大符 9 → 12，实拍 buff_three 跟踪率 81% 不变。见 §49/§50。
- **只剩一片未激活时打不出去**（现场"差临门一脚"）已修（§51）：①几何连续优先于
  类别过滤（参考实现都不让类别决定目标）；②开火可打性锁存 `engageable_latch_s: 0.8`
  （4 片已激活时网络对最后一片的检出率只有 ~19%，而开火请求是 0.7 s 节流的脉冲）；
  ③`fire_block_on_hit_slot: true`（槽位记账有了几何来源之后才敢开）。实测小符完整
  激活轮数 1/8 → 3/12、命中有效率 28~34% → 42~58%、**大符完整激活 0 → 2**，
  实拍 buff_three 81% 跟踪不变。
- **槽位绝对编号仍会漂**（与仿真真值的"最优循环移位一致率" ~40%），是换片判据 ±1 槽
  的误触发造成的；它现在的用途只剩"别打刚打过的那片"（够用），要继续做 ID 匹配得先修它。

以上每一条的证据、影响与建议都在 `docs/energy_rune_architecture.md` §5。
