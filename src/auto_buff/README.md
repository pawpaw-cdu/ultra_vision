# 能量机关（auto_buff）— 当前架构

> 2026-09-24 重构：**与自瞄拆成两个平行任务**，结构对齐 sp_vision：
>
> ```
> src/auto_aim/      自瞄
> src/auto_buff/
>   app/             入口、输入源、Runner
>   config/          buff.yaml -> RuneConfig
>   common/          公共数据类型、跟踪生命周期
>   detection/       模型、检测器、观测/靶面精修
>   estimation/      PnP 解算、目标 EKF、槽位格点
>   control/         瞄准、桥接、扇叶选择、开火门、回合/兜底状态
>   support/         无 IO 的数学与滤波工具
> tests/             单元测试，不进算法目录
> tools/sim/         仿真相机预设
> ```
>
> 构建目标（对齐 sp_vision 的 `tasks/auto_buff`：算法是 **OBJECT 库**、入口是薄 executable）：
>
> | 目标 | 类型 | 内容 |
> |---|---|---|
> | `auto_buff` | OBJECT 库 | 算法本体 + 诊断层（可被任意入口/测试复用） |
> | `auto_buff_demo` | executable | 薄入口：参数/环境 → 配置 → `runRune()` |
> | `buff_target_test` / `buff_state_test` | 测试 | 顶层 `tests/`，`ctest -R buff` 运行 |
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
| `detection/rune_model.*` | OpenVINO 推理 + 五点解码 | 坐标映射；中心距 NMS |
| `detection/rune_detector.*` | 候选选择、亮度判据、锁定滞回、观测精修 | 输出类别、亮点和候选几何 |
| `detection/rune_refiner.*` | 颜色轮廓描述子与 R 标融合 | 观测级几何精修 |
| `detection/rune_plate_refiner.*` | 网络 ROI → 经典靶面中心精修 | 只修中心，保持网络尺度 |
| `estimation/rune_solver.*` | 四点 PnP、世界系几何、相位观测 | IPPE 双解选择 |
| `estimation/rune_target.*` | 小/大符 EKF 与预测 | 圆心、相位和角速度状态 |
| `estimation/rune_slot_lattice.*` | 五槽位锚定和激活记账 | 输出槽号与激活掩码 |
| `control/rune_aimer.*` | 弹道解算和瞄准角生成 | 预测飞行时间与节流 |
| `control/rune_aim_bridge.*` | 帧循环 ↔ 100 Hz 控制线程 | 发布最新目标状态 |
| `control/rune_blade_selector.*` | 命中后选择下一片 | 排除命中片和已激活片 |
| `control/rune_fire_gate.*` | 计划槽位开火准入 | 与观测/建模解耦 |
| `control/rune_round_guard.*` | 回合窗口、命中保持 | 输入相位与命中反馈 |
| `control/rune_aim_fallback.*` | 丢目标停靠与回中策略 | 最后符心优先 |
| `config/rune_config.*` | `configs/buff.yaml` → 各结构体 | 集中读取配置 |
| `common/rune_types.*` | `FanBlade` / `PowerRune` 数据契约 | 世界系几何与相位 |
| `common/rune_track_lifecycle.hpp` | 检测状态时间滞回 | TRACK/LOST |
| `app/rune_runner.*` | 观测→建模→选片→控制→火控主循环 | 薄编排 |
| `app/rune_input.*` | 仿真/视频输入、遥测、指令输出 | IO 与算法隔离 |
| `app/buff_demo.cpp` | 参数和环境 → Runner | 薄入口 |
| `support/` | EKF、正弦拟合、球坐标、弹道数学 | 无 IO |

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
| 用哪些候选构造模型 | `rune_detector.cpp` | 置信度、靶面几何、相位连续、亮度和锁定滞回；**类别不决定是否建模** |
| 锁哪一片 | 同上 | 亮点判据 + `lock_break_px` / `lock_max_miss` 滞回 |
| 圆心/相位 | `rune_solver.cpp` | PnP 双解选择 + `phase_rad` 几何相位 |
| 状态是否可用 | `rune_target.cpp` | EKF 野值门 + 发散保护（小符 |ω|≈π/3） |
| 何时换叶/开火 | `rune_aimer.cpp` | `switch_angle_deg`、`fire_gap_time`、云台到位判据 |
| 本轮是否有效 | `rune_round_guard.cpp` | 点亮窗口 + 类别命中（class0→class1/2）+ 命中保持 |
| 瞄的是第几片 / 哪几片已打 | `rune_slot_lattice.cpp` | 单扇叶锚定 + 每帧重新对齐（丢帧不重锚）+ 投影槽位最近邻归属 + 投票阈值 + 回合复位 |
| 命中后选择下一片 | `rune_blade_selector.cpp` | 排除命中片 + 远离命中位置 + class0/亮片软打分 + 连续帧确认 |
| 计划槽位能否开火 | `rune_fire_gate.cpp` | 计划物理槽 + 激活掩码 + 该槽 class0/class1 证据 + 短暂锁存 |
| 丢目标时云台怎么办 | `rune_aim_fallback.cpp` | 最后已知符心（1.5 s 内）优先，兜底回标定位姿 |
| 是否卡开火 | `RuneFireGate` + `rune_round_guard.cpp` + `rune_aim_bridge.cpp` | 独立火控准入、`hold`、云台到位误差 |

> 观测/建模与开火准入已经分开：class1/2 的已激活片可以参与五槽模型和估计器更新，
> 但只有 `RuneFireGate` 判定计划槽位可打时才会发出 FIRE。

## 6. 配置（`configs/buff.yaml`）

| 段 | 内容 | 备注 |
|---|---|---|
| `camera` | 内参、畸变、标定分辨率、自动缩放 | 与仿真 640×480 / FOV45° 对齐 |
| `model` | 路径、输入尺寸、布局、类别数、关键点数、阈值、`input_scale`、`input_pad_scale` | `input_scale=0.00392`（0~1）；`plate_point_order` 标定物点顺序 |
| `detector` | 类无关的观测选片与锁定参数、`multi_candidate`、圆心搜索兜底 | 火控准入不在这里 |
| `geometry` | 回转半径、靶面半宽、深度校正、距离闸门、相机→云台外参 | 物理量，实车需重测 |
| `aimer` | 提前时间、弹速、云台角包线、换叶阈值、开火节流 | `fire_gap_time` 当前 0.7 s |
| `activation` | 窗口、总目标数、`gate_fire` | 规则层 |
| `gimbal` | 速度/加速度/加加速度限幅、位置环/前馈增益、下发频率 | 与 `configs/tracker.yaml` 同义 |
| `aim_filter` | 瞄准角递归滤波（过程噪声/观测噪声/重置阈值） | 抹平换叶尖点 |

## 7. 运行、调试与评估

```bash
# 构建（算法是 OBJECT 库，入口是薄 executable）
cmake -S . -B build_rune -DULTRA_VISION_BUILD_RUNE=ON -DULTRA_VISION_BUILD_TESTS=ON
cmake --build build_rune -j

# 端到端：仿真 / 视频 / 相机（环境变量见 buff_demo.cpp）
./build_rune/auto_buff_demo

# 纯算法自检（无仿真、无模型）
./build_rune/tests/buff_target_test
./build_rune/tests/buff_state_test
```

> 本节原先还有一整套仿真回归（`rune_sim_test.sh`）、曲线/跑分/槽位一致/格点离线回放
> 脚本、模型选型探针与扇叶状态分类标定探针，它们已随 uv 收敛移出主线（内容在 git 历史里）。
> 扇叶状态分类的阈值仍然**不可信**（两类实测重叠），不接开火闸门。

常用环境变量：`ULTRA_VISION_RUNE_DEBUG=1`
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
  `--pad 1.3` 或把 target 设 52。详见 git 历史里的 `docs/energy_rune_issue_audit.md` §47。
- **已激活外观的类别召回不稳定**：部分实拍段里网络对已激活片几乎不给框，仿真固定
  相机下则能稳定给出 class1。因此激活片可以作为几何观测，但类别不能单独承担
  建模门或火控门。
- **已激活片可以参与观测和建模，但不能获得开火许可**。观测层不再使用 class0
  硬过滤；`RuneFireGate` 只看计划槽位的激活掩码和类别证据。固定相机实验里，
  软观测把激活窗口内的新模型构造率从 25~31% 提升到 90.4%。
- **只剩一片未激活时打不出去**（现场"差临门一脚"）已修（§51）：①几何连续优先于
  类别过滤（参考实现都不让类别决定目标）；②开火可打性锁存 `engageable_latch_s: 0.8`
  （4 片已激活时网络对最后一片的检出率只有 ~19%，而开火请求是 0.7 s 节流的脉冲）；
  ③计划槽位激活时禁止重复开火（槽位记账有了几何来源之后才敢开）。实测小符完整
  激活轮数 1/8 → 3/12、命中有效率 28~34% → 42~58%、**大符完整激活 0 → 2**，
  实拍 buff_three 81% 跟踪不变。
- **槽位绝对编号仍会漂**（与仿真真值的"最优循环移位一致率" ~40%），是换片判据 ±1 槽
  的误触发造成的；它现在的用途只剩"别打刚打过的那片"（够用），要继续做 ID 匹配得先修它。

以上每一条的证据、影响与建议都在 `docs/energy_rune_architecture.md` §5。
