# 局部拟合器（小陀螺 θ/ω + 慢层 α/R）接入说明

> 算法验证与实测结论见 `experiments/local_fitter/README.md`（四板联合后 α 收敛 15±1、
> R 进入 ±0.02；多帧拟合 yaw std ≈ 0.01° ≪ 单帧 PnP）。本文只讲**在 Ultra_Vision 里怎么接**。

## 1. 定位

正面观测时单帧 PnP 的 yaw 病态（左右边像素差趋零）。局部拟合器用短窗灯条轨迹拟合出
**绝对 yaw + 角速度**，作为 EKF 的强时序观测，并保留 α、R 作慢变量在线收敛。

**默认关闭**（`configs/tracker.yaml` 的 `tracker.local_fit.enabled: false`）。与
局部拟合器是**可选组件**：单独成库，不改变 `auto_aim_sim/real` 的
默认依赖与行为。

## 2. 分层与依赖

```
perception/ ──▶ Kalman/（tracker/armor_ekf）──▶ control/
                     ▲
estimation/local_trajectory_fitter （可选库 auto_aim_local_fit，Ceres）
                     ▲ 由 app 注入回调
common/types.hpp（LightBarObservation / LocalFitResult）
```

- **`estimation/local_trajectory_fitter.{hpp,cpp}`**：拟合本体（快层 Ceres 多板联合
  θ/ω，慢层确定性 Gauss-Newton 出 α/R 与 JᵀJ）。编译为 `auto_aim_local_fit`
  （仅当找到 Ceres；不进 `auto_aim_core`）。
- **`Kalman/armor_ekf.hpp`**：新增 `updateWithLocalFit(LocalFitResult)`——按质量缩放
  噪声、走 NIS 门控，用 `updateAngle`+`updateOmega` 注入（保留 ω↔sin/cos 交叉协方差，
  不覆盖状态）。
- **`Kalman/tracker_localfit.cpp`**：线程安全灯条缓冲 + `pushArmorLightBars` +
  `runLocalFitNow`（同步）/`startLocalFitWorker`（异步线程）+ `applyLatestLocalFit`。
  Tracker 通过 `setLocalFitter(callback)` 注入具体拟合算法 → **core 不引入 Ceres**。
- **`common/types.hpp`**：数据契约 `LightBarObservation`、`LocalFitResult`。
- **`configs/tracker.yaml`**：`tracker.local_fit` 段。

## 3. 数据流

```
每帧检测到 Armor
  └─ tracker.pushArmorLightBars(armor)      // 两根灯条端点入缓冲（滑窗）
       └─ [异步线程] runLocalFitNow()        // 低频：喂给注入的拟合器
            └─ latestLocalFit()  →  applyLatestLocalFit()
                 └─ 相机系→EKF 系换算后 ekf_.updateWithLocalFit(conv, …)
```

时间对齐：拟合结果带时间戳，`applyLatestLocalFit` 丢弃比当前帧老过
`max_result_age` 的结果；EKF 侧由调用时机决定外推。

## 4. 开关与配置（`tracker.local_fit`）

| 字段 | 说明 |
|---|---|
| `enabled` | 总开关（默认 false） |
| `window_frames` / `min_bars` | 滑窗帧数 / 最少灯条数（覆盖相位 ≥ π/3 才有 ω 可辨识性） |
| `period_frames` / `max_result_age` | 触发周期 / 结果最大年龄 |
| `yaw_noise` / `omega_noise` | 拟合 yaw、ω 的观测噪声（质量再缩放） |
| `yaw_rate_sign` | 相机系→整车系 yaw 符号（**标定项**，默认 +1） |
| `also_update_center` | 是否把拟合中心也当观测（默认否，先只给 yaw/ω） |

## 5. 怎么用（app 层接线）

```cpp
auto_aim::LocalTrajectoryFitter fitter(/*LocalFitConfig*/);
tracker.setLocalFitter([&](const std::vector<auto_aim::LightBarObservation>& obs,
                           auto_aim::LocalFitResult& out) {
    return fitter.fit(obs, alpha_cur, R_cur, out);   // alpha/R 来自慢层 SlowCalib
});
tracker.startLocalFitWorker();
// 每帧检测后： tracker.pushArmorLightBars(armor); … tracker.applyLatestLocalFit();
```

## 6. 单测 / 回归

```bash
cmake -S . -B build_sim -DULTRA_VISION_BUILD_TESTS=ON
cmake --build build_sim -j8 --target local_trajectory_fitter_test
./build_sim/local_trajectory_fitter_test   # ctest 里名 local_trajectory_fitter_test
```

覆盖：快层四板联合（正面，`sigma=0.5` → yaw<3°、ω<0.2、rmse~0.5px）、慢层一步 refine
（α 从 13°、R 从 0.18 → 进入 15±1 / 0.20±0.02）。

## 7. 现状与后续

- 本分支（`codex/test`）已接入组件 + EKF 钩子 + 缓冲/线程 + 配置 + 单测，**默认关**。
- 待办：① 在 `app/sim_runner` 里真正接线并做 A/B（`experiments/local_fitter` 已有离线
  脚本）；② 标定 `yaw_rate_sign` 与相机→整车 yaw 换算；③ 真机前端仍需先修曝光时间戳
  与角点精度（见 `experiments/local_fitter/CORNER_ACCURACY.md`）。
