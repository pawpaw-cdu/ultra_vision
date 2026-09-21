# 能量机关（大小符）跟踪与激活：现状分析与并入 Ultra_Vision 的设计方案

> 范围：把 `buff_detector` / `Ultra_Vision/src/auto_buff` 的能力升级为
> “大/小能量机关的跟踪 + 激活”，并把它并入 `Ultra_Vision` 的 `auto_aim` 主线架构。
>
> 本文的目标行为规格取自仿真系统 `../simulator_system/simulator/src/robomaster/power_rune.rs`，
> 它是当前唯一可执行的、把 RM 规则量化过的实现。

## 0. 结论摘要

1. `buff_detector` 与 `Ultra_Vision/src/auto_buff` 是同一份“扇叶检测 + 圆盘建模”代码的两个世代，
   `Ultra_Vision` 里那份更新（检测配置重构成 `HsvRange`），但丢掉了 `buff_detector` 工作区里的
   TCP 模拟器输入和“单扇叶补全模型”两个改动。
2. 现有能力只覆盖“小符观测”：算出圆心、半径、5 个扇叶角度，画出可击打四边形。**没有**大符变速模型、
   **没有**亮/灭（激活态）识别、**没有**激活流程状态机、**没有**云台闭环、**没有**并入配置/遥测/测试体系。
3. `Ultra_Vision` 已经有成熟的自瞄五层链路（检测 → PnP → 整车 EKF → 选板 → 云台/火控）和
   仿真器复用件（`sim_receiver`、`config_loader`、`GimbalAimer`、`GimbalController`、`Shooter`、CSV 遥测）。
   能量机关应该做成**同一架构内的第二条 pipeline**，而不是继续长成独立 demo。
4. `src/auto_aim/perception/energy_detector.{hpp,cpp}`（两帧法定圆 + 等角预测）是一次没有完成的并入尝试，
   **没有调用方、没有进 CMake**；`src/auto_buff/energy_node.cpp`（硬编码 `/home/pawpaw-ubuntu/...` 的单文件原型）
   同样是死代码。两者都不应该在新的设计里被继承。
5. 关键设计判断：能量机关的“跟踪”本质是**圆盘上 5 个固定角间隔目标的角度域估计**（相位 + 角速度），
   而不是像素域的匈牙利匹配；大符的角速度是**正弦变化**的（`ω(t)=a·sin(ω₀t)+b`，峰值恒为 2.09 rad/s），
   必须用带参数的模型或自适应滤波器，恒定角速度外推在 20 秒激活窗口内会累积到几十度的相位误差。

## 1. 两个代码库的关系与现状

| 仓库 / 路径 | 角色 | 入口 | 状态 |
|---|---|---|---|
| `buff_detector/`（独立 git 仓库，HEAD `3f96611`） | 能量机关算法的上游原型 | `energy_demo`：本地视频或 TCP 模拟器 | 工作区有未提交改动：TCP 接收、单扇叶补全、`debug=1` |
| `Ultra_Vision/src/auto_buff/` | 同一份代码搬进主工程（`15c909b` 起，`47cf1d6`/`85158a3` 继续改） | `energy_demo`：仅本地视频 | 检测配置重构为 `HsvRange`；缺失上面的 TCP/单扇叶改动 |
| `Ultra_Vision/src/auto_buff/energy_node.cpp` | 更早的单文件原型 | 无 | 死代码，未进任何 CMake target |
| `Ultra_Vision/src/auto_aim/perception/energy_detector.{hpp,cpp}` | 一次未完成的并入尝试 | 无 | 死代码，未进任何 CMake target |
| `Ultra_Vision/src/auto_aim/` | 自瞄主线：检测 → PnP → 整车 EKF → 选板 → 云台/火控 | `auto_aim`（相机版 `node.cpp` / 仿真版 `node_sim.cpp`） | 活跃，有单测、有遥测、有调参 |

证据：

```bash
# 上游工作区独有的两个改动
diff -u Ultra_Vision/src/auto_buff/src/tracker.cpp buff_detector/src/tracker.cpp
#   -> 单扇叶场景补全 5 扇叶模型、ref_index 兜底（只在上游工作区）
#   -> buff_detector 还新增了 sim_receiver/ 与 main.cpp 的 --tcp 模式
# 死代码确认
rg -n "energy_detector|EnergyDetector" Ultra_Vision --glob '!build*'   # 只有自身文件命中
rg -n "energy_node" Ultra_Vision/**/CMakeLists.txt                    # 无命中
```

### 1.1 Ultra_Vision 自瞄主线（并入的目标架构）

```text
相机帧（Galaxy 相机 / TCP 模拟器）
  -> Detector            灯条 + 装甲板配对（传统）或 YOLO11 四点（OpenVINO，可异步线程）
  -> PnP                 solveArmorPnP -> rvec/tvec（相机系）
  -> Tracker             整车状态估计：ArmorEKF(9 维) + 板号关联 + 状态机
  -> TargetSelector      选板、提前量、可射击窗口 -> TargetDecision
  -> AimSignalFilter     瞄准角递归滤波
  -> GimbalController    100 Hz 加加速度受限轨迹 -> GIMBAL yaw/pitch
  -> Shooter             火控状态机 -> FIRE
```

各层与共用件的现成位置：

| 能力 | 位置 | 能量机关能否直接复用 |
|---|---|---|
| YAML 解析（两入口共享） | `src/auto_aim/config_loader.hpp` | 复用，新增能量机关小节 |
| 相机/仿真输入 | `io/camera/GalaxyCamera.*`、`src/auto_aim/sim_receiver/` | 直接复用 |
| PnP 求解 | `src/auto_aim/perception/pnp_solver.*` | 复用接口，换几何模型 |
| 状态估计 | `src/auto_aim/Kalman/armor_ekf.hpp`、`Kalman/tracker.*` | 只复用“时间戳/相机姿态”约定，模型要新写 |
| 角度解算 + 轨迹 | `src/auto_aim/control/gimbal_aimer.*` | **直接复用**（与仿真器相机约定一致） |
| 高频指令调度 | `src/auto_aim/control/gimbal_controller.*` | 直接复用 |
| 火控 | `src/auto_aim/control/shooter.*` | 复用状态机，输入换成能量机关决策 |
| 遥测 | `node_sim.cpp` 的 `ULTRA_VISION_*_CSV` | 复用写法，新增 rune CSV |
| 回归工具 | `tools/truth_regression.py`、`src/auto_aim/test/` | 复用，扩展 rune 阶段 |

### 1.2 现有能量机关链路（buff/auto_buff）

```text
BGR -> HSV/BGR 掩膜 -> 形态学开闭
     -> 轮廓几何筛选（面积/矩形度/细长度/圆度/凸度）-> 矩心 = 扇叶中心
     -> R 标：面积 200~400、圆度>=0.3、半径 10~35 -> 圆心 + 半径×1.2
     -> Tracker：首帧按角度排序初始化 5 个扇叶 ID
                贪心匹配（丢失扇叶加惩罚）-> 角度 EMA -> 丢失扇叶按恒定角速度外推
                -> TrackerResult{圆心, 半径×1.2, 5 个扇叶中心/角度/击打四边形}
     -> PnP：4 点 -> SOLVEPNP_IPPE，物点 320mm × 382mm
     -> 可视化：圆盘 + 5 个击打四边形 + ROI
```

已知问题（都已实测踩到）：

- 常量角速度（配置 `angular_velocity_deg: 50.5`）只对**小符**成立，且与仿真器真值 60°/s（π/3）不符；
- 扇叶 ID 靠像素距离贪心匹配，扇叶彼此间距只有 `2·R·sin36°`，遮挡或丢帧后容易串号；
- 半径用“圆心到扇叶中心平均距离”，圆心来自一个面积 200–400 px 的小圆标记，抖动直接进模型；
- 是否点亮（`BladeStatus::ACTIVE/NOACTIVE/NOTHING`）在 `energy_node.cpp` 里用**轮廓面积**区分
  （7000–11000 视为未点亮、13000–15000 视为点亮），这是拟合出来的经验值，换场地/换机关就失效，
  而且 `types.hpp` 里定义了这个枚举后从未使用；
- PnP 物点尺寸 320mm × 382mm 来自实车标定，与仿真模型（扇叶靶面约 300mm × 210mm）不一致，
  在仿真里距离会系统性偏大约 25%；
- 没有时间基准：`dt` 由 `steady_clock` 算出，与图像时间戳无关。

## 2. 仿真器里的权威目标行为（可直接当规格）

来源：`../simulator_system/simulator/src/robomaster/power_rune.rs`、`simulator/src/main.rs`、
`simulator/src/assets/POWER.glb`、`simulator/config/camera.yaml`、`API.md`。

### 2.1 场景

- `scenario: energy_rune`（`rune-only` 特性下启动）加载 `POWER.glb`，包含**两个互相独立的机关面**：
  `FACE_1` = 小符（红方，顺时针）、`FACE_2` = 大符（蓝方，逆时针）。
  两者中心相距约 0.62 m，同屏可见。
- 相机固定在 `position [1.738, 2.45, 1.731]`，`looking_at [-0.16, 2.45, -0.22]`
  —— 恰好是 `FACE_1`（小符）的圆心，视线方向 ≈ 圆盘转轴 `(1,0,1)/√2`。
- 鼠标/F3 在 rune-only 下被禁用，但 **`GIMBAL <yaw> <pitch>` 仍然驱动相机**：
  `camera.rotation = home.rotation · R_y(-yaw) · R_x(pitch)`，pitch 限幅 ±80°。
  这与 `Ultra_Vision` 的 `GimbalAimer::solveTargetAngles()` 约定完全一致，可直接复用。
- 按 `FIRE` 从相机位置沿光轴以 25 m/s 发射弹丸（冷却 0.05 s，有 44.5 g 质量、重力 9.81、阻尼 0.05）。
- rune 场景**不导出 `dataset.csv`**（只有 `armor_chassis` 场景导出），所以目前没有能量机关真值回放基线。

### 2.2 几何

| 量 | 值 | 依据 |
|---|---|---|
| 扇叶中心回转半径 | 0.699 m | `FACE_1_TARGET_i_*` 节点局部平移在圆盘平面内的半径 |
| 扇叶数 / 角间隔 | 5 / 72° | `build_targets()` 的 `1..=5` |
| 扇叶靶面尺寸 | ≈ 0.21 m（盘内）× 0.30 m（竖直）× 0.21 m（沿轴） | 靶面 mesh 的 POSITION bbox |
| 圆心高度 | 2.447 m | `FACE_1.translation` |
| 每个扇叶 4 个视觉变体 | `DISABLED` / `ACTIVE`（点亮）/ `ACTIVATED`（已命中）/ `COMPLETED` | `RuneVisual::apply()` |
| 额外视觉线索 | `LEGGING_PROGRESSING` 只在 `Activating` 态可见 | `build_targets()` 的 `only_activating` |

### 2.3 运动学

- **小符**：恒定角速度 `ROTATION_BASELINE_SMALL = π/3 rad/s`（60 °/s），红方顺时针、蓝方逆时针（相机视角）。
- **大符**：进入激活态时随机生成

  ```text
  a     ∈ [0.780, 1.045]   rad/s
  ω₀    ∈ [1.884, 2.0]     rad/s     （正弦周期 3.14 ~ 3.33 s）
  b     = 2.090 − a
  ω(t)  = a·sin(ω₀·t) + b
  ```

  由此：**峰值转速恒为 2.090 rad/s ≈ 119.7 °/s**（与 `a` 无关），谷值 `2.090 − 2a ∈ [0, 0.53] rad/s`，
  `ω` 的均值 `b ∈ [1.045, 1.310] rad/s ≈ 60~75 °/s`（与小符 60 °/s 同量级，这就是“小符是常数先验”的来源）。
  `(a, ω₀)` 每次进入激活时重随机，但在同一次激活会话的多轮之间保持不变。
- 大符**只在激活态**使用变转速；其余状态退回小符的恒定转速。

### 2.4 激活状态机（把“激活”需求讲清楚的核心）

```text
Inactive(1.0s) ──► Activating ──命中全部目标──► Activated(6.0s) ──► Inactive
                       │
                       ├─ 主窗口 2.5s 超时 ──► Failed(1.5s) ──► Inactive
                       └─ 全局 20s 超时    ──► Failed(1.5s) ──► Inactive
```

| 模式 | 每轮点亮 | 命中规则 | 完成条件 |
|---|---|---|---|
| 小符 | 随机 1 个未激活扇叶 | 命中该扇叶即记 1 次，随即开启新一轮（重新随机 1 个） | 5 个扇叶全部激活（共 5 轮） |
| 大符 | 随机 2 个未激活扇叶 | 命中第 1 个后开启 **1.0 s 二次窗口**，窗口内命中第 2 个才算本轮成功；二次窗口超时则**重新开一轮**（不判失败） | 5 个扇叶全部激活（2+2+1，共 3 轮） |

补充规则：命中**非点亮**扇叶在当前仿真里被忽略（`FUNNY = true`，不判失败、不推进进度）；
每轮主窗口 2.5 s，全局 20 s，超时/判失败后回到 `Inactive` 并 1 s 后重新点亮。

对视觉的含义：**激活 = 在正确的时间把弹丸打到当前点亮的扇叶上**。
因此视觉必须输出三件事：(a) 哪几个扇叶当前点亮；(b) 它们在未来 `t_flight + t_gimbal` 时刻的角度；
(c) 已经点亮的扇叶不能再打（否则浪费 2.5 s 窗口）。

### 2.5 命中与火控约束

- 弹丸初速 25 m/s、直线飞行、受重力：2.7 m 距离飞行 0.108 s，重力下坠约 5.7 cm（≈1.2° 俯仰补偿）。
- 同一时间扇叶最多转 `ω·t_flight`：大符峰值 ≈ 0.226 rad = 13°，必须做提前角。
- 命中判定用 `ACTIVE / ACTIVATED / COMPLETED / DISABLED` 四种靶面网格的碰撞体，四者重叠，
  所以“打到靶面”即触发 `on_target_hit(逻辑索引)`，坏命中在规则层被忽略。

## 3. 差距分析

| 能力 | buff_detector / auto_buff | auto_aim 主线 | 目标 |
|---|---|---|---|
| 扇叶中心 + 圆心检测 | 有（BGR/HSV 几何筛选） | 无（死代码不算） | 沿用并参数化 |
| 5 扇叶模型 | 有（贪心匹配 + 常量 ω 外推） | 无 | 角度域估计 |
| 小符运动模型 | 有（常量 ω，值不匹配） | 无 | 60 °/s 先验 + 观测融合 |
| **大符变速模型** | 无 | 无 | `a·sin(ω₀t)+b` 或自适应 EKF |
| **亮/灭（激活态）识别** | 名义上有枚举，实际用面积阈值且未使用 | 无 | 发光强度 + 多帧投票的三态判定 |
| **激活流程状态机** | 无 | 无 | 小符/大符两套，含窗口与超时 |
| 云台闭环瞄准 | 无 | 有（GimbalAimer/Controller） | 直接复用 |
| 火控/开火许可 | 无 | 有（Shooter） | 复用，输入换成能量机关决策 |
| 时间基准 | `steady_clock` | 图像时间戳（`StandardClock`） | 统一到图像时间戳 |
| PnP 几何 | 有（320×382mm，与仿真不符） | 有（装甲板） | 重新标定 + 配置化 |
| 配置驱动 | 部分（detector/tracker/solve 三个 YAML） | `config_loader.hpp` 统一 | 并入 `configs/` |
| 遥测/回归 | 无 | CSV + `truth_regression.py` | 复用并扩展 |
| 单元测试 | 无 | 有 | 新增能量机关测试 |

顺带发现的三处“抽屉里的资产”：

1. `configs/buff.yaml`：原先写好的 `buff_type/small_buff/large_buff/kalman/roi/hsv` 草案
   **没有任何代码加载它**（只有 `tools/hsv_tuner.cpp` 把它当字符串模板打印）。它已被改写为
   新链路的正式配置（见第 6 节）。
2. **`sp_vision_25` 里已有成套的能量机关实现和已训练模型**，这是本次并入的真正起点：
   `tasks/auto_buff/`（`buff_detector/solver/target/aimer/type` + `yolo11_buff`）、
   `assets/yolo11_buff_int8.{xml,bin}`（单类 `fanblade`、每扇叶 6 关键点、letterbox pad 114）、
   `tools/{extended_kalman_filter,ransac_sine_fitter,math_tools,trajectory}.{hpp,cpp}`。
   **不需要重新训练**；`models/train.py` 只是 3 行历史遗留。
3. `sp_vision_25/tests/auto_buff_test.cpp` 与 `src/auto_buff_debug.cpp`：参考实现的完整调用顺序
   （detector → solver → target → aimer → 指令），移植时按它对齐。

## 4. 目标架构：作为 auto_aim 的第二条 pipeline

### 4.1 目录与文件

```text
src/auto_buff/
  rune_types.{hpp,cpp}     # FanBlade / PowerRune：把点亮扇叶放回 5 个 72° 格点
  rune_model.{hpp,cpp}     # OpenVINO 推理 + YOLO11 关键点解码（[1,17,8400]）
  rune_detector.{hpp,cpp}  # 关键点 -> 扇叶 -> 圆心（阈值+膨胀+最近轮廓精定位）
  rune_solver.{hpp,cpp}    # 靶面 4 角点 PnP -> z-up 世界系（几何配置化）
  rune_target.{hpp,cpp}    # EKF：小符 7 维匀速、大符 10 维 a·sin(ωt+φ)+2.09−a
  rune_aimer.{hpp,cpp}     # 检测延迟 + 飞行时间迭代预测 -> 云台角 + 开火
  rune_config.hpp          # buff.yaml -> 各层配置
  rune_node.cpp            # 入口：TCP 模拟器 / 离线视频
  support/                 # 移植自 sp_vision 的 EKF、RANSAC 正弦拟合、球坐标、弹道
  test/rune_target_test.cpp# 解析观测下的估计/预测/瞄准自检
configs/buff.yaml          # 由“草案”升级为正式配置（rune 段）
```

### 4.2 数据流

```text
图像帧(时间戳)
  -> RuneModel/Detector  关键点 -> 点亮扇叶 -> 圆心（5 格点定向）
  -> RuneSolver          靶面 PnP -> z-up 世界系的圆心(ypd)/姿态(ypr)/靶心(ypd)
  -> RuneTarget          EKF：圆心 + yaw + roll + ω（大符再辨识 a、ω₀、φ）
  -> RuneAimer           按飞行时间迭代预测靶心 -> yaw/pitch + 开火脉冲
  -> sim_receiver        GIMBAL / FIRE（与 node_sim.cpp 同一通道）
```

### 4.3 复用与新增

- **直接复用**：`sim_receiver`（帧+时间戳+命令回传）、CSV 遥测写法、
  `sp_vision_25` 的整套能量机关算法与已训练模型（移植进本模块，不依赖 sp_vision 构建）。
- **新增**：本模块自己的配置加载（`rune_config.hpp`）、坐标/符号适配层
  （把仿真器的云台角与 sp_vision 的 z-up 世界系对齐）、内参随帧尺寸缩放、自检测试。
- **后续接入**：激活状态机（点亮集合 + 命中窗口）、`GimbalController`/`Shooter` 复用、
  `truth_regression` 报告。
- **不继承**：`energy_node.cpp`、`auto_aim/perception/energy_detector.*`、`auto_buff` 的贪心匹配、
  常量 `dt=0.033` 与 `steady_clock` 时间基准。

## 5. 算法设计要点

### 5.1 检测层

1. **用 `sp_vision_25` 已训练的 `yolo11_buff_int8` 关键点网络**（单类 `fanblade`），
   **不重新训练**。每个扇叶 6 个关键点 = 靶面 4 角 + 靶心 + 臂上点，输出 `[1,17,8400]`；
   letterbox 到 640×640、pad 114、RGB、/255。
   **补充（选型实测）**：能量机关没有 yolov5 版本——本工程的 `yolov5.xml` 是装甲板 rp24
   轻量模型；能量机关可选的只有 YOLOv8n/YOLO11n-pose 系列。本机实测 fp16 比 int8 更快
   （p50 19.5~21.6 ms vs 25.8~29.7 ms），因此默认换成 fp16（同一批权重，检出能力不变）；
   预处理改用 `cv::dnn::blobFromImage` 后 6.2 ms → 1.2 ms，端到端 23 FPS → 42 FPS。
   静态导出无法运行时 reshape（anchor 是常量），要更轻必须按 `imgsz=416` 重新导出，
   代码已支持（动态输入自动 reshape、静态输入按模型自身尺寸、letterbox 跟随实际尺寸）。
   详见 `src/auto_buff/README.md` 的“模型与速度”。
2. 只有**点亮的扇叶**才会被网络检出，所以“哪些扇叶亮着”本身就由检测结果给出；
   扇叶的三态（禁用/点亮/已命中）不需要额外分类器，但需要**跨帧的格点一致性**
   （见 `PowerRune`：新亮起的扇叶=距离已点亮集合最远的那个）。
3. 圆心由关键点 5、6 外推（`(臂点 − 靶心)×1.4 + 靶心`），再用灰度阈值 + 膨胀 +
   最近近圆轮廓精定位；实拍里这一步比直接取 R 标面积阈值稳。
4. 结构上给出与 `Detector` 相同的“输入图像 → 输出带不确定性的观测”边界；不在这里做状态估计。

已知边界：模型是仿真风格训练的，实拍里“扇叶熄灭、只剩靶心环亮着”的形态会漏检
（见第 8 节风险 2）。

### 5.2 圆盘状态估计

最终采用的是 sp_vision 的 z-up 世界系模型（已实现），因为它同时吃下“圆心在动”和
“扇叶在转”两件事，且与云台角直接同构：

```text
小符 x(7)  = [R_yaw, v_R_yaw, R_pitch, R_dis, yaw, roll, ω]
大符 x(10) = 小符 + [a, ω₀, φ]，其中 ω(t) = a·sin(ω₀ t + φ) + (2.09 − a)
观测 z1(4) = [R_yaw, R_pitch, R_dis, roll]        （圆心 + 姿态）
观测 z2(3) = 靶心球坐标 ypd                          （点亮扇叶）
```

- 小符：`ω` 初值取 ±π/3，方向由 `Voter` 投票（连续帧 roll 变化方向）锁定；`ω` 只通过
  状态转移与 roll 的协方差间接被观测，等价于“常数角速度先验 + 观测修正”。
- 大符：`(a, ω₀, φ)` 是 EKF 状态的一部分，用 `a·sin(ω₀t+φ)+2.09−a` 做非线性传播；
  `b = 2.09 − a` 这条仿真器给出的强约束直接写进模型，少估一个参数。
- 角度通道按 **72° 格点折叠**（`|Δroll| > π/12` 时找最近格点），所以换扇叶不会打断跟踪。
- 发散守卫：小符 `|ω|` 必须落在 π/3 ± 10°；大符 `a ∈ [0.52, 1.57]`、`ω₀ ∈ [1.26, 3.0]`，
  越界即重新初始化。
- 丢失时按模型外推，丢帧超过阈值后重置；`TEMP_LOST / LOST` 状态机已实现。

### 5.3 提前角与可射击窗口

```text
future   = (now − 图像时间戳) + predict_time      （检测延迟 + 额外提前量）
θ        = θ_state(t + future)                    （EKF 传播，大符按正弦模型积分）
p        = R_buff2world(θ) · (0,0,0.7) + 圆心      （靶心世界坐标）
解一次弹道 -> t_flight；再用 t_flight 重新预测 θ 并重解弹道
|Δt_flight| < 0.01 s 才接受，否则放弃本轮（避免发散时硬发指令）
指令 yaw = −atan2(p.y, p.x)，pitch = 弹道上仰角（正=上）
```

点火约束（已实现）：瞄准角相对上一帧跳变超过 5° 判为换叶，**换叶期间与之后一段时间不开火**；
连续换叶超过 3 次则强制重新捕获；稳定后按 `fire_gap_time`（默认 0.7 s）节流开火。
仿真里 `FIRE` 冷却 0.05 s、弹丸 25 m/s，0.7 s 的间隔对 2.5 s 窗口有足够裕量。

待补：把“点亮集合 + 已命中集合 + 窗口计时”接进来（5.4），目前只按“最优点亮扇叶”打。

### 5.4 激活状态机

```text
输入：本帧点亮扇叶集合、扇叶三态、命中反馈（或自身开火时刻 + 超时）
状态：Waiting / Engaging(窗口计时) / AwaitingSecondHit(仅大符, 1.0s) / Done / Failed
输出：目标扇叶索引、是否允许开火、已命中集合、剩余窗口时间
```

- 小符：单目标 → 命中后立刻 `已激活 ∪= {i}` → 选择下一个点亮扇叶（检测给出，不要自己预测随机数）。
- 大符：两目标 → 命中第 1 个后进入 1.0 s 二次窗口，优先选择第二个点亮扇叶；
  超时则按“新一轮 2 个点亮”重新初始化。
- 5 个全部命中即 `Done`；主窗口 2.5 s、全局 20 s 是硬约束，状态机必须自己计时，
  不能依赖“命中事件”一定可见（帧率/遮挡会丢反馈）。

## 6. 配置设计

`configs/buff.yaml` 已升级为正式配置，由 `energy/rune_config.hpp` 加载（不再走
`config_loader.hpp`，避免和装甲板主线耦合）。实际字段：

```yaml
rune:
  enabled: true
  mode: small               # small | large（可用 ULTRA_VISION_RUNE_MODE 覆盖）
  visualize: true
  simulator_config: simulator.yaml
  camera:                   # 标定分辨率 + 内参，帧尺寸不同时按比例缩放
    camera_matrix: [...]    # 640x480 / FOV 45° -> fx=fy=579.411, cx=320, cy=240
    distort_coeffs: [...]
    calibration_width: 640
    calibration_height: 480
    auto_scale_intrinsics: true
  model:
    path: models/openvino/yolo11_buff_int8.xml   # 复用 sp_vision 的模型
    device: CPU
    input_size: 640
    score_threshold: 0.7
    nms_threshold: 0.4
    pad_value: 114
    num_threads: 4
  detector:
    max_lost_frames: 20
    gray_threshold: 100
    dilate_size: 5
    center_mask_ratio: 0.8
    center_extrapolation: 1.4
    multi_candidate: false       # true 时同时输出所有点亮扇叶（激活状态机需要）
    fallback_single_blade: true
  geometry:
    target_radius_m: 0.700       # 圆心 -> 靶心（仿真真值；实车需标定）
    target_half_width_m: 0.127   # 靶面 254 mm
    arm_point_radius_m: 0.220
    R_camera2gimbal: [1,0,0, 0,1,0, 0,0,1]
    t_camera2gimbal: [0,0,0]
  aimer:
    yaw_offset_deg: 0.0
    pitch_offset_deg: 0.0
    fire_gap_time: 0.7
    predict_time: 0.12
    bullet_speed: 25.0
    gravity: 9.81
    switch_angle_deg: 5.0
    max_mistakes: 3
```

## 7. 分阶段实施计划

| 阶段 | 内容 | 验收标准 |
|---|---|---|
| **M1 骨架 + 检测 + PnP** ✅ | `src/auto_buff/` 全模块（模型/检测/解算/估计/瞄准/入口/自检）、`configs/buff.yaml` 正式化、`rune_demo` 与 `rune_target_test`、模型入库 `models/openvino/` | 已达成：可编译、可跑视频与仿真输入、实拍跟踪段距离/角速度数值自洽、自检全过 |
| **M2 跟踪** ✅（数值）/ ⏳（仿真） | EKF 跟踪 + 72° 格点折叠 + 丢失外推 + 时间戳统一 | 自检：小符跟踪 < 7°、0.15 s 预测 < 4°；大符 4 s 内辨识 ω 误差 < 0.30 rad/s，0.25 s 预测优于匀速外推。**待接仿真 rune-only 场景做端到端复核** |
| **M3 点亮集合** ⏳ | `multi_candidate=true` 下的多扇叶格点分配 + 点亮集合输出 + 跨帧三态一致性 | 点亮扇叶识别准确率 > 99%；小符/大符“本轮该打哪个”不串号 |
| **M4 激活状态机 + 火控** ⏳ | 窗口/超时/已命中集合状态机、`Shooter` 与 `GimbalController` 接入、`FIRE` 节流 | rune-only 仿真里自动完成小符 5 次命中、大符 2+2+1；记录激活成功率、超时率、平均用时 |
| **M5 并入与回归** ⏳ | 并入 `node_sim.cpp`（`ULTRA_VISION_RUNE_MODE` 切换）、`truth_regression` 扩展、实车标定 | `ctest` 全绿；两条 pipeline 互不影响；有能量机关阶段误差报告 |

配套建议：给仿真器补一份能量机关真值导出（当前只有 `armor_chassis` 会写 `dataset.csv`），
字段建议 `time_us,face,mode,target_index,state,cx,cy,R,phase,omega`，
否则 M2/M3/M4 的量化验收只能靠人工看视频。

## 8. 风险与待确认

1. **实车几何未知**：老 `auto_buff` 里的 `ENERGY_LENGTH/WIDTH = 382/320 mm` 与仿真靶面不符；
   现在采用 sp_vision 标定的几何（半径 0.700 m、靶面 254 mm、臂点 0.220 m），
   实车仍需现场复测（几何全部在 `geometry.*`，改配置即可）。
2. **实拍与仿真外观不一致**（已实测）：模型是仿真风格训练的，`buff_red.mp4` 前 148 帧
   五个扇叶全亮时能稳定跟踪（ω 中位数 52°/s、距离 3.4 m）；第 148 帧后机关只留一个
   靶心环发光，网络直接漏检。→ 上实车前需要用现场素材微调检测（补数据或退回到
   传统视觉 + 发光强度分类），或者只把本模块用在仿真/训练场。
3. **坏命中是否判失败**：仿真 `FUNNY=true` 忽略误击，真实规则（以及后续仿真版本）可能改为判失败，
   这会影响火控的保守程度（是否必须在窗口内才开火）。
4. **规则版本**：小符 60 °/s、大符 `a·sin(ω₀t)+b`、2.5 s / 1 s / 20 s 窗口都取自当前仿真器实现，
   需要与参赛赛季规则手册再核对一次。
5. **算力**：能量机关若与装甲板自瞄同时运行（相机会同时看到机关与车），需要决定是共用一帧、
   分线程，还是按 `buff_type` 互斥；仿真器 rune-only 模式下二者不同屏，实车需要统一规划。
6. **模型来源**：能量机关的模型已存在于 `sp_vision_25`（`assets/yolo11_buff_int8.{xml,bin}`，
   单类 `fanblade` + 6 关键点），**不打算自训练**，`models/train.py` 是历史遗留。
   本模块已把模型复制到 `models/openvino/` 并直接从配置引用；如果 `sp_vision` 侧后续更新
   模型，需要同步复制一次（或把路径指回 sp_vision 的 assets 目录）。
7. **仿真输入尺寸**：仿真器 `DAEDALUS_CAPTURE_WIDTH/HEIGHT` 默认 640×480，与配置内参一致；
   若改成其它分辨率，`auto_scale_intrinsics` 会按比例缩放内参，但更稳妥的是重新标定。
