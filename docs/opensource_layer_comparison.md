# 与开源方案的分层对照（E10）

> 只对比**结构 / 执行 / 驱动**，不含算法调参。对照对象：
> `sp_vision_25`、`jlu_vision_26`、`rmcs_auto_aim_v2`、`awakening`、本项目 `Ultra_Vision`。
> 依据：各仓库的目录分层与入口写法（本地源码，2026-10）。

## 0. 各家的分层骨架

| 项目 | 能力层目录 | 入口目录 | 入口形态 |
|---|---|---|---|
| sp_vision_25 | `tasks/{auto_aim,auto_buff,omniperception}`（检测/跟踪/解算/瞄准/开火各自成文件） | `src/*.cpp`（14 个，每个 82~195 行） | **薄组合根**：构造模块 + 短循环 |
| jlu_vision_26 | `src/{auto_aim,auto_buff,odom_coord,...}` + `src/tools`、`src/hardware` | `startup/` 里的启动器 | 模块化 + xmake，`hardware/tools` 单独成层 |
| rmcs_auto_aim_v2 | `src/module/*`（plugin）+ `src/kernel/*`（核心实现） | ROS2 组件（`package.xml`/`plugins.xml`） | **插件化**：kernel 实现 + module 装配 |
| awakening | `src/tasks/*`（含 `base/` 公共）+ `src/runtime`、`src/_rcl` | `src/runtime` 驱动 | 任务化 + 运行时；3rdparty 全内嵌 |
| **Ultra_Vision** | `perception/` `Kalman/` `control/` `config/` `visualization/` + `app/` | `node_sim.cpp`(150)、`node.cpp`(1640→重构中) | 能力库 `auto_aim_core` + 两个入口 |

## 1. 硬件层（相机 / 串口 / 云台）

| 项目 | 相机 | 串口/总线 | 云台下发 | 时间戳 |
|---|---|---|---|---|
| sp_vision | `io/{hikrobot,mindvision,usbcamera}` 多相机并列 | `io/serial` | `io/gimbal` 直接绝对角 | 相机 SDK 时间戳 |
| jlu_vision | `src/hardware/camera`、`src/hardware/imu` 分离 | `src/hardware/serial` | 由 `hardware` 层统一 | 有 |
| rmcs | `src/io/capturer` + `recorder` | 走 ROS2 话题 | kernel 内 `transform_tree` | 走 ROS 时间 |
| awakening | 3rdparty `daedalus_interface` | `_rcl` 运行时 | `tasks` 内下发 | 有 |
| **Ultra_Vision** | `io/camera/{Hik,Galaxy}` **同名接口**二选一 | `io/serial`（termios+超时读） | `io/gimbal`（43B 收/29B 发，读线程） | Hik 有 `FrameTime`；**Galaxy 缺**（C5/C7） |

**可借鉴**：sp_vision/jlu 把相机驱动做成"多实现 + 统一接口"，Ultra_Vision 已对齐；
差在**时间戳契约要两条相机都满足**（本项目 Hik 有、Galaxy 缺）。

## 2. 识别层（检测 / PnP）

| 项目 | 结构 | 备注 |
|---|---|---|
| sp_vision | `tasks/auto_aim/{yolo,detector,classifier}` + `solver`（PnP）分离 | 传统/神经网络并列成文件 |
| jlu_vision | `src/auto_aim/armor_detector` 独立包 | xmake 包边界 |
| rmcs | `kernel/detector` + `module/detector`（实现/装配分离） | |
| awakening | `tasks/auto_aim/armor_omni` | |
| **Ultra_Vision** | `perception/{detector,nn_detector,pnp_solver,armor_source}` | `armor_source` 把"检测+PnP+动态 ROI"合成一个前端 |

**可借鉴**：rmcs 的"kernel 实现 / module 装配"边界值得学；Ultra_Vision 用
`armor_source` 起到了类似作用（入口不再各自拼检测）。

## 3. 观测层（关联 / 估计）

| 项目 | 结构 | 备注 |
|---|---|---|
| sp_vision | `tasks/auto_aim/{tracker,target,voter}`（跟踪/目标/投票分离） | |
| jlu_vision | `src/auto_aim/armor_tracker` + `fake_armors_publisher`（便于离线/回放） | 有假数据注入 |
| rmcs | `kernel/{pose_estimator,tracker}` | |
| awakening | `tasks/auto_aim/armor_omni` + FSM | |
| **Ultra_Vision** | `Kalman/{tracker,armor_ekf,plate_ekf,tracker_*}` + `estimation/rotation_rate_estimator` | 关联/生命周期/尺度/watchdog 拆成多个 TU |

**可借鉴**：jlu 的 `fake_armors_publisher` —— 让观测层可**脱离相机**回归；我们已有
`tools/replay_dataset`，方向一致。

## 4. 决策层（选板 / 提前量 / 开火）

| 项目 | 结构 | 备注 |
|---|---|---|
| sp_vision | `tasks/auto_aim/{aimer,planner,shooter}` | planner 出轨迹，shooter 单独判据 |
| jlu_vision | `armor_tracker` 内决策 | |
| rmcs | `kernel/fire_control` | |
| awakening | `tasks/auto_aim/auto_aim_fsm`（显式状态机） | |
| **Ultra_Vision** | `control/{target_selector,gimbal_aimer,shooter,aim_pipeline}` | `aim_pipeline` 把选板→提前量→云台→开火串成一条链，两入口共用 |

**可借鉴**：sp_vision 把 `aimer`(期望姿态) 与 `planner`(轨迹/速度前馈) 分开 —— 这正是
我们 `gimbal_command_frame` + 速度前馈在做的事；awakening 的显式 FSM 适合把
"捕获/跟踪/开火/失锁"讲清楚。

## 5. 控制/执行层（下发 / 频率 / 时间基）

| 项目 | 下发 | 控制环频率 | 时间基 |
|---|---|---|---|
| sp_vision | `io/gimbal.send(control,fire,yaw,yaw_vel,yaw_acc,pitch,...)` | 独立规划线程（MPC） | 帧时间戳 → 姿态插值 |
| jlu_vision | 由 hardware 层下发 | | |
| rmcs | ROS2 话题 | ROS 定时 | |
| awakening | `_rcl` 运行时 | | |
| **Ultra_Vision** | `AimPipeline` 控制线程按 `command_rate_hz` 下发 | 100Hz（可评估 200/500，C6） | 仿真已统一；真机待补曝光时刻（C7） |

**可借鉴**：sp_vision 的"发送结构体里直接带 vel/acc 前馈"我们已对齐；差在
**下发频率与固件内插行为的确认**（C6，需电控）。

## 6. 我们相对开源的结构优势 / 待补

**已对齐或更好**

- 能力库 `auto_aim_core` + 两个薄入口，仿真/真机**同一份估计/控制/开火**（开源里少有两入口共享到这一层）。
- `app/` 组合层把配置/CSV/曲线/叠加显示集中，避免入口各自抄一套。
- 仿真保真度：云台限速跟随 + 真实姿态回传，闭环与实车同构。

**待补（都在执行/驱动层）**

- `node.cpp` 仍是 1640 行的厚入口（A2）；软规则已把它标红。
- 相机时间戳契约两条驱动不一致（C5/C7）。
- 下发频率/协议字段需与电控确认（C6）。
- 决策/控制的可观测性（如 in_fire_window 占空、plate_switches）我们已有 CSV，开源里少见——可继续用 `docs/layer_error_budget.md` 的表做回归。
