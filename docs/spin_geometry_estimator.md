# 小陀螺灯条轨迹几何估计器（spin_geometry_estimator）

> 定位：异步架构里的**低频全局几何先验 / 初始化器**。默认**不接管主链路**，
> 只作为可选组件（`auto_aim_spin_geometry`）提供 + 单测，不进 `auto_aim_core`。

## 思路

跟踪一根灯条**上/下端点**在像素系的轨迹：pitch=0 时两端点半径相同 → 轨迹近"矩形"；
pitch≠0 时两端点半径差 ≈ L·sinα → 轨迹呈"扇带形"。用这条轨迹反解旋转中心 C、
回转半径 R、灯条 pitch、旋转轴方向、yaw(t)，再据刚体几何外推同一辆车的其余三块板。

## 数学模型

相机针孔 `u=fx·X/Z+cx, v=fy·Y/Z+cy`。旋转轴 `â=normalize((tx,1,tz))`，中心 C。
取 `e1=normalize(x̂-(x̂·â)â)、e2=e1×â`，则相位 θ 处

```
r̂=cosθ·e1+sinθ·e2,  t̂=-sinθ·e1+cosθ·e2
灯条中心 B=C+R·r̂+c·t̂
长轴   d=cosα·â+sinα·r̂
端点   E±=B±(L/2)·d
```

未知量：`C(3), R, α, tx, tz, yaw(t)`；`c`（灯条切向偏移）与 `L`（灯条长度）已知——
**L 提供尺度**（单条空间圆的投影有深度歧义，用 L 的投影长度定死）。

## 实测结论（见 `/tmp/gyro_traj_demo`，可复现）

1. **yaw(t) 必须用三次 B 样条**：真实变速 `θ=3t+0.4·sin(2π·1.0·t)` 下，二次多项式
   彻底失效（yaw 106°）；样条能精确表达，但**必须配多起点初值**（θ0 × 转向），否则会落进
   "近/远"错误分支。加多起点后 60 帧档 → **yaw 0.62° / 中心 10 mm / 外推 0.24 px**。
2. **旋转轴不竖直是最大风险**：只差 ~6° 时，固定竖直轴模型让中心偏 34 mm、外推最大偏
   10 px；把轴当 2 DOF 一起估 → 轴残差 0.14°、误差回到常规水平。**接入前必须开轴估计。**
3. **单灯条最短窗口 ≈ 200° 转角**（约 1.1 s @3 rad/s）。低于 ~160° 出现"近/远"双解，
   且**重投影误差不能当有效性判据**（对/错分支都 ~0.5 px）。本模块用**弧覆盖**
   （`arc_coverage_deg`）门控，实测门限取 190°。
4. 噪声 / 5% 外点(Huber) / 近距强透视 都稳；只看**半圈遮挡**也能收敛（已知 L 提供尺度）。

## API

```cpp
#include "estimation/spin_geometry_estimator.hpp"

auto_aim::SpinGeometryConfig cfg;      // K / bar_offset_c / bar_length_L / distance_hint / estimate_axis
auto_aim::SpinGeometryEstimator est(cfg);
std::vector<auto_aim::SpinBarObservation> obs;  // 每帧：t + plate/side + 上/下端点像素
if (est.fit(obs)) {
    const auto& r = est.result();      // center / radius / pitch / axis / arc_coverage_deg
    double yaw_now = r.yawAt(t);
    auto bars = est.allBars(yaw_now);  // 外推 4 板 × 2 灯条
}
```

> **务必给多根灯条**：同一帧里把能看到的灯条都塞进来（同一 `t`、不同 `plate`/`side`）。
> 只给一根会频繁解错（见下）。

## 真实噪声下的表现（关键）

把噪声换成贴近实检的模式——**端点高斯噪声 + 整块板共享的"框抖动"（相关误差）+ 丢帧 +
外点**——再扫窗口，对比"单灯条 vs 多灯条联合(4 板 × 2)"：

| 噪声档(端点σ/框抖动/丢帧/外点) | 窗口 | 单灯条 yaw | 多灯条 yaw | 多灯条中心 |
|---|---|---|---|---|
| mild (1.0px / 2px / 15% / 3%@8px) | 2.0 s | 1.5° | **0.55°** | 16 mm |
| mild | 1.2 s | 11.0° | **0.50°** | 50 mm |
| medium (2.0 / 5 / 30% / 5%@12px) | 2.0 s | 50.6° | **1.03°** | 78 mm |
| medium | 1.2 s | 13.9° | **1.06°** | 83 mm |
| harsh (3.0 / 8 / 40% / 8%@20px) | 2.0 s | 51.3° | **1.66°** | 35 mm |
| harsh | 1.2 s | 96.7° | **2.31°** | 60 mm |
| medium + 只看近侧半圈 | 2.0 s | 59.4° | **1.89°** | 4 mm |
| medium + 只看近侧半圈 | 1.2 s | 72.1° | **1.82°** | 48 mm |

**结论**：
- 真实噪声下**单灯条基本不可用**（频繁落进"近/远"错误分支，yaw 50~97°）；
- **多灯条联合是必要条件**：yaw 回到 0.5~2.3°，连 harsh + 半圈遮挡也稳；
- 中心误差在 medium 档仍有 ~80 mm（>50 mm 目标）→ 想再准，要么端点更准（角点精修）、
  要么给更多帧/更多板，要么把"框抖动"这类相关噪声显式建模进 R。
- ⚠️ 重投影误差不能当有效性判据；用**净转角覆盖**门控（≥190°）。

## 构建 / 测试

可选依赖 Ceres（找到才编译；`auto_aim_sim`/`auto_aim_real` 不受影响）：

```bash
cmake -S . -B build_sim -DULTRA_VISION_BUILD_TESTS=ON
cmake --build build_sim -j8 --target spin_geometry_estimator_test
./build_sim/spin_geometry_estimator_test      # ctest 里是第 19 个用例
```

未装 Ceres 时会打印 "auto_aim_spin_geometry: 跳过"，其余构建照常。

## 后续接入（尚未做，按此顺序）

1. **离线验证**：用 `tools/replay_dataset` 录一段小陀螺，离线跑本估计器，与现有 tracker 的
   yaw/中心逐帧对比，量化收益。
2. **低频只读**：`app/` 起一个 5~10 Hz worker，滑窗跑估计器，只打日志/曲线，不改控制。
3. **加约束以缩短窗口**：现在只跟一根灯条；改为联合"同一块板两根灯条 / 多块板"的残差，
   打破近/远歧义（这是缩短到 ~200° 以下、走向在线可用的关键）。
4. **前置条件**：真机需先修 C7 的曝光时间戳；并做稳"同一根灯条"的跨帧关联。
5. 最后才考虑让它在 `Tracker` LOST/DETECTING 时做全局初始化、给 `selector` 预测四板。

## 真实检测端点实测（sim，3 m，静止相机）—— 当前**不可用**

用 `ULTRA_VISION_DETECTIONS_CSV`（新增的可选记录器，默认关）+ `ULTRA_VISION_DISABLE_CONTROL=1`
（冻结云台，相机静止）录一段真实小陀螺（3 m、3 rad/s、12 s），把每帧**全部检测**的
左右两根灯条端点导出，离线喂给本估计器：

- 相机确为静止：真值旋转中心 C 的时间标准差 ≈ 1.5 mm；R≈0.21 m；距离 3.0 m。
- **观测精度是瓶颈**：检测角点相对真值板的残差中位 **22.9 px**（min 9.8 / p90 36.7）；
  PnP tvec 相对真值板中位 **9.8 cm**（rp24），换 yolo11 头 **11.7 cm**（没有更好）；
  其中**深度方向最差：dz 标准差 32.5 cm**（PnP 深度病态），还混有 z≈0.8 m 的假框。
- 而 3 m 处小灯条在画面里只有 **~30 px 长**——**观测噪声 ≈ 信号**，SNR≈1。

→ 结论：**用当前检测器（NN 关键点）的原始端点，轨迹几何估计不可用**（拟合发散到
R 上界、右 pitch 限幅）。这不是几何模型的问题，而是**观测精度不够**。

**要让它成立，必须先提高端点精度**（任一）：
1. 角点精修到 ~1 px（现在的 `keypoint_refiner` 在本场景下没达到）；
2. 缩短交战距离 / 换大板（1~2 m 时灯条 60~120 px）；
3. 不喂原始检测，改喂 **tracker 融合后的平滑观测**；
4. 或者只用传统灯条检测器（端点更稳）在近距离上跑。

复现：`/tmp/gyro_real`(rp24) / `/tmp/gyro_real_y11`(yolo11) 有 `detections.csv` + `truth.csv`；
`/tmp/label_bars.py` 做真值标注、`/tmp/gyro_traj_demo/build/replay_bars` 做离线拟合。
