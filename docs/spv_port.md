# 用 sp_vision_25 的 tracker/solver 当底座（移植说明）

## 为什么换底座

真机实测（同一份检测流、同一个"模型 vs 本帧所有检测"指标，最近板心距离）：

| 谁 | 帧数 | 中位 | p90 |
|---|---|---|---|
| Ultra_Vision 原来的 tracker | 1784 | **46.6 px** | 58.8 |
| sp_vision_25（本目录移植版） | 1171 | **3.0 px** | 27.7 |

我们那套的问题不在滤波器数学（状态/观测方程本来就是照抄 sp_vision 的），而在事后加的补丁：
`geometry_learn` 在实车上会漂（实测 18 秒里 l 从 +3.5cm 走到 −16cm、撞到两端的钳位），
而 `hard_plate_constraint` 一关，模型又没有任何东西钉在观测上。
既然 sp_vision 的底盘构造+重投影在真机上已被验证"非常优秀"，就直接用它当底座，
**局部拟合器只在这个底座上做 A/B**。

## 搬了什么（`src/spv/`）

| 文件 | 来源 | 改动 |
|---|---|---|
| `armor.{hpp,cpp}` | `tasks/auto_aim/armor.*` | namespace → `spv` |
| `target.{hpp,cpp}` | `tasks/auto_aim/target.*` | 同上（**滤波器本体，逻辑未改**） |
| `tracker.{hpp,cpp}` | `tasks/auto_aim/tracker.*` | 同上 + 删掉"全向感知"重载 |
| `solver.{hpp,cpp}` | `tasks/auto_aim/solver.*` | 同上 + 物点尺寸改成可配（见下） |
| `tools/extended_kalman_filter.{hpp,cpp}` | `tools/*` | namespace → `spv` |
| `tools/math_tools.{hpp,cpp}` | `tools/*` | 同上 |
| `tools/logger.hpp` | `tools/logger.hpp` | **换成零依赖桩**（原版用 spdlog，开发机没有） |

**没有搬**：aimer / shooter / planner / voter / detector / omniperception / 轨迹解算。
本次只验证"底盘构造 + 重投影 + 滤波器 + 局部拟合器"，瞄准与开火仍用我们自己的。

## 坐标系怎么对齐

sp_vision 的链路：
```
p_world = R_gimbal2world · (R_camera2gimbal · p_cam + t_camera2gimbal)
```
我们的链路：
```
p_world = (base_to_world · cameraToBaseRotation(yaw, pitch) · cameraToGimbal) · p_cam
```
所以适配层（`tools/spv_replay.cpp`）令

- `R_camera2gimbal` = `CameraPose::camera_to_gimbal`（手眼标定），`t_camera2gimbal` = 手眼平移
- `R_gimbal2imubody` = 单位阵，每帧 `set_R_gimbal2world(Quaterniond(base_to_world · cameraToBaseRotation(yaw, pitch)))`

两条链路就逐位等价，sp_vision 的 Solver/Tracker 拿到的是真机同一个观测。

## 物点尺寸（**必须与 PnP 同源**）

sp_vision 原版写死 `灯条 56mm / 小板宽 135mm`；我们标出来的是 `60mm / 131.5mm`。
同一组关键点用两套物点会带来 4~7% 的距离系统偏差，A/B 就不公平了，所以改成环境变量：

```
SPV_LIGHTBAR_LEN=0.060 SPV_SMALL_ARMOR_WIDTH=0.1315 SPV_BIG_ARMOR_WIDTH=0.221
```

不设就是 sp_vision 原值。

## 两个坑（都已处理）

1. **角点顺序**：我们 `Points_2D` = 左下/左上/右上/右下，sp_vision 要 TL/TR/BR/BL
   → 适配层循环移位。
2. **颜色过滤**：sp_vision 的 `Tracker::track()` **忽略 `use_enemy_color` 参数、无条件**
   按 `enemy_color` 过滤。所以生成的配置里 `enemy_color` 必须和建 `Armor` 时给的
   `color_id` 一致，否则所有检测会被 `remove_if` 掉（第一次跑就是这样，tracking=0）。

## 第三个坑：外参的**坐标系约定**（2026-10-11）

NUC 上 sp_vision 跑的是 `~/sp_vision_25/configs/standard3.yaml`（现场版，已抄成
`configs/spv_extrinsics.yaml`）。但**不能把它的 `R_camera2gimbal` 直接拼进我们的链路**：

- 它那份是给 sp_vision 自己的云台系标的（x 前 / y 左 / z 上）；
- 我们的 `cameraToBaseRotation` 输出的是我们的体坐标系（x 右 / y 下 / z 前）；
- 两者相差一个固定的 90° 置换。直接拼会把世界系的"竖直"拧歪，而 sp_vision 的模型
  假设四块板在水平面内绕中心转 —— 竖直一歪模型就散。

实测（同一份实车检测流，指标 = 检测中心 ↔ 最近模型板心）：

| 配置 | 中位 |
|---|---|
| 自洽（单位外参，spv 的总变换 = 我们的 world←camera） | **3.9 px** |
| 硬拼 spv 的 R_camera2gimbal（带补偿 R_g2w=W·R^T） | 27.1 px |
| 硬拼 configs/camera.yaml 的 R_camera2gimbal | 149 px |

**结论**：重投影对"外参选哪一组"本来就是不变的（前后用同一套就抵消），真正敏感的是
**坐标系约定是否自洽**。要 100% 复刻 sp_vision 的链路，正确做法不是搬外参数字，而是
连它的"云台角→世界系"那一步一起搬 —— 它用的是 **C 板 IMU 四元数**（`set_R_gimbal2world(q)`），
不是 yaw/pitch。所以下一步应该：把 IMU 四元数也录进检测流（real_runner 里已经有
`io::ImuSample`），再喂给它。现在默认用自己的自洽口径（`SPV_USE_SPV_EXTRINSICS=1` 可切换）。

## 局部拟合器的接入状态（`--local-fit`）

已经接好了：spv 的状态就是 `[x,vx,y,vy,z,vz,a,w,r,l,h]`，`a/w` 直接是底盘 yaw 与角速度，
所以注入只要一路 `1×11` 的 H（不需要像我们原来那套处理 sin/cos）。工具里也做了：
单板弧关联（同帧多板混进一条弧会让 rmse 直接爆）、**去畸变**、**用 EKF 中心热启动**。

但**这一份录制上拟合器还没收敛**：`fitWithSlow` 的 rmse 在 37~97px（门限 ~17px），
166 次尝试 0 次有效。注意同一份数据用 `replay_obs_csv --local-fit` 却出过 7.95px ——
因为它沿用了 `local_fit.min_depth=1.5` 的过滤，**把 1.2~1.5m 的近距帧全丢了**。
也就是说：**拟合器在近距（1.2~1.5m）这一段是失效的**，这是接下来要查的第一件事。

## 怎么用

```bash
# 先录一份（真机）：
ULTRA_VISION_DISABLE_CONTROL=1 ULTRA_VISION_DETECTIONS_CSV=/tmp/det.csv ./build/auto_aim_real

# 再用 sp_vision 的底座离线回放同一份检测流：
SPV_LIGHTBAR_LEN=0.060 SPV_SMALL_ARMOR_WIDTH=0.1315 \
  ./build_sim/spv_replay --detections /tmp/det.csv --out /tmp/spv.csv
```

`--out` 逐帧落 sp_vision 的 11 维状态（`x vx y vy z vz a w r l h`）+ 重投影指标。
