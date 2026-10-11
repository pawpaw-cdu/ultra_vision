# 局部拟合器与 Tracker/TargetSelector 职责交集审计报告

## 审计时间
2026-10-11

## 审计范围

必审（存在性已核对，提示词里的三个文件名与实际不符，已按实际文件审计）：

| 提示词里的路径 | 实际文件 | 说明 |
|---|---|---|
| `Kalman/tracker.cpp` / `tracker.hpp` | ✅ 存在 | |
| `Kalman/tracker_localfit.cpp` / `tracker_localfit.hpp` | ⚠️ 只有 `.cpp` | 声明在 `tracker.hpp`，没有独立头 |
| `Kalman/association.cpp` | ⚠️ 实为 `tracker_association.cpp` | |
| `Kalman/lifecycle.cpp` | ⚠️ 实为 `tracker_lifecycle.cpp` | |
| `Kalman/armor_ekf.cpp` | ⚠️ 头文件实现 `armor_ekf.hpp`（无 .cpp） | |
| `common/types.hpp` | ⚠️ 在仓库根 `common/types.hpp` | 不是 `src/auto_aim/common/` |
| 其余 | ✅ 存在 | |

另外审计了 `Kalman/tracker_accessors.cpp`（模型构造/几何同步）、
`config/tracker_config_loader.hpp`（开关是否真的接上）、
`app/real_runner.cpp` / `app/sim_runner.cpp`（调用链）、
`perception/armor_source.cpp`（ROI 与 Tracker 状态的关系）。

## 审计结果汇总

| 审计项 | 状态 | 发现的问题数 |
|---|---|---|
| A: LocalFit 前端依赖选板 | ⚠️ 有问题（软耦合 + 相位整体偏移） | 2 |
| B: TargetSelector 依赖 LocalFit | ✅ 通过 | 0 |
| C: lifecycle 影响弧段 | ⚠️ 间接影响（经 ROI 的输入侧耦合） | 1 |
| D: EKF 相位转换 | ❌ 严重 | 3 |
| E: LocalFitResult 字段 | ⚠️ 有问题（缺 `ref_armor_id`） | 1 |
| F: Push/Apply 调用链 | ⚠️ 有问题（**慢层输出没接上**，与选板无关） | 2 |
| G: 共享状态 | ⚠️ 有三套 id + 两套相位表不一致 | 2 |

**一句话结论**：职责边界在**代码结构上**是干净的（LocalFit 不看 Tracker 状态、TargetSelector 不看 LocalFit），
但**数据语义上**有两处硬伤：① 拟合器的 yaw 观测因为 EKF 雅可比写错而**完全没生效**；
② 拟合器的慢层几何（R/α/h）**没有任何调用点写回模型**。这两条比"职责重叠"严重得多。

---

## 详细发现

### 审计项 A —— LocalFit 前端是否依赖 Tracker 的选板状态

**匹配代码**

```cpp
// app/real_runner.cpp:1005   （sim_runner.cpp:623 同）
for (const auto& a : frame_input.armors) {
    tracker.pushArmorLightBars(a, observation_time, tracker.plateIdFor(a));
}

// Kalman/tracker.hpp:601
int plateIdFor(const Armor& armor) const { return selectPlateId(armor); }   // ← 读 EKF 状态

// Kalman/tracker.hpp:598-599（接口注释）
/// armor_id 必须是**板号 0..3**（不是网络类别 id）——用 plateIdFor() 取。
void pushArmorLightBars(const Armor& armor, double timestamp, int armor_id);

// Kalman/tracker_localfit.cpp:131,153
const int pid = (arc->plate_hint >= 0 && arc->plate_hint <= 3) ? arc->plate_hint : 0;
...
o.armor_id = pid;          // ← 直接进拟合器的 plate_yaw_offsets[armor_id]

// estimation/local_trajectory_fitter.cpp:159-165
const double yaw_off = cfg.plate_yaw_offsets[o.armor_id];
const double theta = f.yawAt(o.timestamp) + yaw_off;
```

**问题分析**

1. **不是"只喂选中板"**：调用点传的是**本帧全部检测**（`frame_input.armors` / `source_result_.armors`），
   而且 push 发生在 `pipeline.update()` / `pipeline_->update()` **之前**（real_runner.cpp:1005 vs 1010；
   sim_runner.cpp:623 vs 629），所以同帧的关联/选板结果不会泄进缓冲。这一层是干净的。
2. **但 `armor_id` 本身来自 Tracker 的选板**：`plateIdFor()` → `selectPlateId()`（tracker_association.cpp）→
   `matchCost()` → `ekf_.predictYpd()` / `yawStd()`。也就是说拟合器拿到的"板号"是
   **EKF 状态 + 上一帧选板（`plate_id_`，带 0.65 倍滞回）**的产物，不是观测自身的属性。
3. 该 `armor_id` 在拟合器里**不是提示而是硬偏移**：`plate_yaw_offsets[armor_id]` 直接决定
   拟合出的 `yaw` 代表哪块板的相位。选板错 k 号 → 拟合 yaw 整体差 k·90°。
   目前靠 `applyLatestLocalFit()` 的"模 90° 吸附"掩盖（tracker_localfit.cpp:351-355），
   属于**治标**。

**状态**：⚠️ 有重合（软耦合）。**设计上的错**是"用 EKF 状态去给观测贴板号"，
**实现上的错**是把这个不可靠的板号当成了拟合模型的硬参数。

**修复建议**：`pushArmorLightBars` 只吃原始观测（`armor_id = -1`，表示未知），
板号由拟合器自己在弧段内**相对**确定（它已经有 `seed_cx/seed_cz` 的两段式相对身份方案，
tracker_localfit.cpp:105-107），或由调用方传**网络分类器的板面数字**（`Armor::number`，
sp_vision_25 的 `armor.name` 路线）而不是几何选板结果。

### 审计项 B —— TargetSelector 是否依赖 LocalFit

**检索结果**：`grep -n "local_fit\|LocalFit\|segment\|arc" control/target_selector.{cpp,hpp} control/aim_pipeline.cpp`
→ **零匹配**。`target_selector.hpp` 只 `#include <array>`。

TargetSelector 的输入是 `aim_pipeline.cpp:73-86` 组装出来的 `TargetEstimate`
（center/velocity/yaw/omega/armor_radius/converged/age），全部来自 Tracker 的 getter；
它自己用 `predictPlatePosition()` + `angleForPlate()` 推四块板（target_selector.cpp:181-201, 217-346）。
LocalFit 只通过 EKF 状态**间接**影响它 —— 这正是期望的"只在 EKF 融合处交汇"。

**状态**：✅ 通过（0 个问题）。

### 审计项 C —— Tracker 的 lifecycle 是否影响 LocalFit 的弧段

**匹配代码（直接）**：`grep -n "state_\|IDLE\|TRACKING\|TEMP_LOST" tracker_localfit.cpp` → **零匹配**。
弧段的创建/暂停/清理只由观测流决定：

- 正对门控暂停而非清空：tracker_localfit.cpp:88-95
- 间距超时清空重来：:97-101
- 长时间不活跃删除：:179-184
- worker 只挑"还活着的弧"：:199-210

**匹配代码（间接，实为问题）**

```cpp
// app/real_runner.cpp:866   （sim_runner.cpp:450 同）
source_result = armor_source.update(image, frame_sequence, ts,
    tracker.getEstimatedArmorPositions(), tracker.getState(), /*image_will_be_modified=*/true);

// perception/armor_source.cpp:76
if (projected.size() >= 2 && tracker_state != TrackerState::LOST) { ... ROI 收缩到预测的四块板附近 ... }
```

**问题分析**：拟合器的输入是**检测结果**，而检测的范围（ROI）由 Tracker 的
**预测模型位置 + 状态**决定。模型被拖偏时 ROI 跟着偏，新转进画面的板可能落在 ROI 外 →
该板根本不出现在 `armors` 里 → 拟合器的弧断掉/饿死。这条耦合在模式上就是
"模型错 → 看不到修正模型所需的数据"的正反馈，与审计项 A 的相位问题叠加时尤其危险。

**状态**：⚠️ 1 个问题（实现上的耦合）。
**修复建议**：LocalFit 的输入不要复用"给 tracker 用的收缩 ROI 检测结果"；
让 worker 以一个**独立的、覆盖可视角的 ROI**（或全帧/分块）跑一路低帧率检测；
或至少在 `|ω|` 大 / 弧段饥饿时把 ROI 放大回全帧。

### 审计项 D —— EKF 融合是否做了相位转换 ❌

**`updateWithLocalFit` 完整实现**（`Kalman/armor_ekf.hpp:769-778`）

```cpp
double updateWithLocalFit(
    const LocalFitResult& fit, double yaw_noise = 0.02, double omega_noise = 0.05,
    double nis_threshold = std::numeric_limits<double>::infinity()) {
    if (!initialized_ || !fit.valid) return 0.0;
    const double q = std::clamp(fit.quality, 0.05, 1.0);
    const double yaw_w = updateAngle(fit.yaw, /*plate_id=*/0, yaw_noise / q, nis_threshold);
    const double omega_w = updateOmega(fit.omega, omega_noise / q, nis_threshold);
    return std::max(yaw_w, omega_w);
}
```

**结论 1：没有任何相位转换**。`plate_id` 被**写死为 0**，`LocalFitResult` 里也没有
`ref_armor_id` 可用来换算。调用方只能靠"模 90° 吸附"（tracker_localfit.cpp:351-355）
把未知的 k·90° 消掉。

**结论 2（更严重）：`updateAngle` 的雅可比本身写错了。**

```cpp
// Kalman/armor_ekf.hpp:635-643
const double sin_total = x_(4,0) * cos_plate + x_(5,0) * sin_plate;
const double cos_total = x_(5,0) * cos_plate - x_(4,0) * sin_plate;
const double predicted_angle = std::atan2(sin_total, cos_total);
Matrix<1, kStateDim> H;
H(0, 4) = cos_total;
H(0, 5) = -sin_total;
```

状态是 `(x4,x5) = (sinθ, cosθ)`，`predicted = θ + plate_angle`，正确雅可比是
`H = (cosθ, -sinθ)`（与同文件 `ypdJacobian` 第 3 行 `jacobian(3,4)=cos_theta;
jacobian(3,5)=-sin_theta;`（:976-977）一致）。代码用的是**旋转之后**的
`(cos_total, -sin_total) = (cos(θ+p), -sin(θ+p))` —— 只有当 `p = angleForPlate(id) = 0`，
也就是 `id = 1` 时才等于正确值。

**有限差分核对**（`/tmp/h_probe`，只读探针，不改仓库）：

```
plate=0 theta= 37.0deg | H_code=(-0.602,-0.799) H_fd=(+0.799,-0.602) | dtheta_gain code -0.000 fd +1.000 (应=1)
plate=1 theta= 37.0deg | H_code=(+0.799,-0.602) H_fd=(+0.799,-0.602) | dtheta_gain code +1.000 fd +1.000 (应=1)
plate=2 theta= 37.0deg | H_code=(+0.602,+0.799) H_fd=(+0.799,-0.602) | dtheta_gain code +0.000 fd +1.000 (应=1)
plate=3 theta= 37.0deg | H_code=(-0.799,+0.602) H_fd=(+0.799,-0.602) | dtheta_gain code -1.000 fd +1.000 (应=1)
```

**直接注入核对**（`/tmp/angle_probe`：告诉滤波器"某块板的相位比预测大 5°"，看底盘 yaw 实际怎么动）：

```
plate_id=0  观测偏 +5.0deg -> 底盘 yaw 变化  +0.000 deg  (期望 +5.000)
plate_id=1  观测偏 +5.0deg -> 底盘 yaw 变化  +4.751 deg  (期望 +5.000)
plate_id=2  观测偏 +5.0deg -> 底盘 yaw 变化  +0.000 deg  (期望 +5.000)
plate_id=3  观测偏 +5.0deg -> 底盘 yaw 变化  -4.751 deg  (期望 +5.000)   ← 符号相反
```

**后果**：

1. `updateWithLocalFit` 固定用 `plate_id = 0` → **拟合器的 yaw 观测是完全的空操作**。
   这就是历史上"开了 `use_yaw` 却毫无变化"的根因（commit 8e0c663 把它归因于
   "结果太老被老化门限丢掉"，并加了 ω 外推；外推修的是"新鲜度"，修不了"雅可比为 0"）。
2. 依赖它的三套机制**全部空转**：模 90° 相位吸附（tracker_localfit.cpp:349-355）、
   拟合 yaw 常值偏置在线学习（:357-372）、按 `|ω|` 混合拟合/EKF yaw（:379-397）。
3. `joint_solve` 的 yaw 通道同样走 `updateAngle(..., 0, ...)`（tracker.cpp:534-537）→ 也是空的。
4. 旧路径 `tracker.cpp:95` `ekf_.updateAngle(*radial_angle, plate_id_, ...)`：只有
   `plate_id_ = 1` 正确，`0/2` 无效，`3` **反向**（正反馈）。该路径只在
   `armorYpdObservation()` 返回 `nullopt` 时触发（PnP 的 rvec 缺失/退化），
   属于低频但危险的兜底分支。
5. 主路径 `updateYpd` 用的是**正确**的 `ypdJacobian`（:976-977），所以日常表现正常
   —— 这也解释了为什么这个 bug 一直没暴露：只有 LocalFit / joint_solve / 兜底分支会踩到。

**状态**：❌ 严重（3 个问题：无相位转换、雅可比错、受影响机制静默空转）。

### 审计项 E —— LocalFitResult 字段完整性

```cpp
// common/types.hpp:120-134
struct LocalFitResult {
    bool   valid = false;
    double timestamp = 0.0;            // 窗口末端时间
    double center[3] = {0,0,0};        // 旋转中心（相机系）
    double yaw = 0.0;                  // chassis yaw
    double omega = 0.0;
    double alpha = 0.0;                // 慢层
    double R = 0.20;                   // 慢层
    double h = 0.0;                    // 慢层
    double drift[3] = {0,0,0};
    double reprojection_rmse_px = 0.0;
    double quality = 0.0;
    int    num_bars = 0;
};
```

对照要求的 8 个字段：`timestamp` ✅、`center_cam`（命名 `center`）✅、`radius`（命名 `R`）✅、
`theta`（命名 `yaw`）✅、`omega` ✅、`reprojection_rmse`（命名 `reprojection_rmse_px`）✅、
`valid` ✅；**`ref_armor_id` ❌ 缺失**。

**后果**：调用方无法知道拟合输出代表哪块板，只能用"模 90° 吸附"猜；
而且 `R` 的语义是"绕旋转中心的回转半径"，在矩形底盘 + `l/h` 分组几何下
与"车心到板心的距离"不再等价，落到 `plate_distance[4]` 时会带系统性偏差。

**状态**：⚠️ 1 个问题（缺 `ref_armor_id`；命名与语义需要补文档）。

### 审计项 F —— Push / Apply 调用链是否干净

**全部调用点**

| 调用 | 位置 | 上下文 |
|---|---|---|
| `setLocalFitter` | real_runner.cpp:622 / sim_runner.cpp:246 | 注入 Ceres 版拟合器（编译期 `ULTRA_VISION_HAVE_LOCAL_FIT`） |
| `startLocalFitWorker` | real_runner.cpp:628 / sim_runner.cpp:261 | 起异步 worker（30 ms 周期） |
| `pushArmorLightBars` | real_runner.cpp:1005 / sim_runner.cpp:623 | **本帧全部检测**，在 `pipeline.update()` **之前** |
| `applyLatestLocalFit` | real_runner.cpp:1010 / sim_runner.cpp:640 | 在 `pipeline.update()` **之后**，**无条件**调用（不按 Tracker 状态分支） |

→ 输入侧**没有**经过选板过滤 ✅；apply 也**不是**只在 `TRACKING` 调用 ✅
（内部只要求 `cfg_.local_fit.enabled && ekf_.isInit()`，tracker_localfit.cpp:314）。

**问题**：慢层输出的写回**断了**。

```cpp
// Kalman/tracker_accessors.cpp:130   ← 定义存在
void Tracker::updateChassisFromFit(double radius, double armor_pitch_rad, double height_up)

// 全仓检索：updateChassisFromFit 只有"声明(tracker.hpp:430) + 定义"两处，零调用点。
// Tracker::setArmorRadius（tracker.hpp:512）只被单测调用：
//   test/armor_ekf_test.cpp:622  online_tracker.setArmorRadius(0.185);
// local_fit_apply_geometry_（tracker.hpp:650）全仓没有任何自增 →
//   状态行的 `geometry=` 恒为 0（real_runner.cpp:2034）。
```

git 证据：commit `8e0c663`（"延迟量测…"）删除了

```cpp
-        if (cfg_.local_fit.use_fit_radius && slow_ok) {
-            updateChassisFromFit(fit->R, fit->alpha, fit->h);
-            ++local_fit_apply_geometry_;
-        }
```

**后果**：拟合器算出来的 `R / α / h` **完全没有写回模型几何**；
`local_fit.use_fit_radius` / `radius_alpha` / `radius_max_step` 是**死配置**；
`geometry=` 这个诊断量永远为 0（"拟合器没生效"的假象）。
同时 real_runner.cpp:986-992 反过来把 tracker 的几何高度喂给拟合器当 `h_prior` ——
即"tracker → 拟合器"这条是通的，"拟合器 → tracker"这条是断的，闭环开口。

**状态**：⚠️ 2 个问题（慢层未写回、死配置 + 误导性遥测）。

### 审计项 G —— 是否存在共享的 `current_armor_id`

`grep -rn "current_armor_id\|current_target\|selected_id\|active_armor\|currentTargetId"`
→ **零匹配**：不存在三方共用的单一 id 状态 ✅。

但存在**三套语义重叠的 id**：

| 模块 | id 名字 | 位置 |
|---|---|---|
| Tracker | `plate_id_`（当前关联板）/ `previous_plate_id_` / `last_observed_plate_id_` | tracker.hpp 私有成员；`selectPlateId` 在 tracker_association.cpp |
| TargetSelector | `TargetEstimate::armor_id` / `TargetDecision::armor_id` / `lock_id_` | target_selector.hpp:113,129,154,159；target_selector.cpp:181-201,321 |
| LocalFit | `LightBarObservation::armor_id` / `LocalFitArc::plate_hint` | tracker_localfit.cpp:108,118-119,131,153 |

**关键问题：两套 id→相位表不一致。**

```
ArmorEKF::angleForPlate        = [ +90,  0, -90, -180 ]  (deg)  // common/chassis_model.hpp armorPhaseOffset=(1-k)·90°
拟合器 plate_yaw_offsets 默认   = [   0, 90, 180, 270 ]  (deg)   // local_trajectory_fitter.hpp:88-89
```

差值是 `+90/−90/+90/−90`，即 id 被做了一次反射（0↔1、2↔3）。
因为差值都是 90° 的整数倍，在**四块板完全对称**时二者等价，被"模 90° 吸附"掩盖；
但当前仓库已经启用逐板几何（`plate_distance[4]`、`l/h` 分组 {0,2}/{1,3}），
此时"yaw" 与 "yaw+90°" **不再等价**（组会互换），相位表不一致会直接变成模型分叉。

**状态**：⚠️ 2 个问题（相位表两套、id 语义重叠但无命名规范）。

---

## 修复优先级

| 优先级 | 问题 | 代码位置 | 修复成本 | 影响 |
|---|---|---|---|---|
| **P0-1** | `updateAngle` 雅可比用了旋转后的 cos/sin，导致 plate 0/2 完全无效、plate 3 反号；`updateWithLocalFit`/`joint_solve` 固定用 plate 0 → **拟合器 yaw 通道全程空操作** | armor_ekf.hpp:641-643 | 2 行 | 极高：LocalFit 存在的唯一目的就是"正面 yaw 病态时提供强 Yaw 观测"，现在是空转；还让三套 yaw 机制（吸附/偏置学习/|ω|混合）全部无效 |
| **P0-2** | 慢层 `R/α/h` 无调用点写回模型（`updateChassisFromFit` 是死代码） | tracker_accessors.cpp:130；tracker_localfit.cpp applyLatestLocalFit | 5~10 行 | 高：几何永远停在 yaml 写死值；`use_fit_radius` 等是死配置；`geometry=` 遥测误导 |
| **P1-1** | `LocalFitResult` 缺 `ref_armor_id`，`updateWithLocalFit` 无相位换算 | common/types.hpp:120；armor_ekf.hpp:774 | 小 | 高：相位靠猜；未来换板/多板联合必然出错 |
| **P1-2** | EKF 与拟合器两套 id→相位表不一致 | chassis_model.hpp:44 vs local_trajectory_fitter.hpp:88 | 小 | 中→高：四板对称时被掩盖，启用逐板几何后爆发 |
| **P2-1** | 拟合器输入复用"给 tracker 的收缩 ROI"，与 Tracker 状态耦合（模型偏→看不到修正数据） | armor_source.cpp:76；real_runner.cpp:866 | 中 | 中：弧段饥饿/正反馈 |
| **P2-2** | `pushArmorLightBars` 的 `armor_id` 来自几何选板（EKF 状态），拟合 yaw 整体相位受其支配 | tracker.hpp:601；tracker_localfit.cpp:131,153 | 中 | 中：已在用"模 90° 吸附"兜住 |
| **P3** | 三套 id 缺少命名与语义规范；`Armor::number`（分类器数字）未用于身份 | 全局 | 中 | 低→中：可参考 sp_vision_25 用数字做身份 |

---

## 验证方案（每个问题修完后如何确认消失）

**P0-1**

1. 单测：对 `plate_id = 0..3` 各跑一次"观测相位 +5°"的更新，断言底盘 yaw 变化
   ∈ `(3°, 5°]` 且符号为正（当前实测：0/2 为 0、3 为 −4.75°）。
2. 单测：`updateWithLocalFit(fit)`，其中 `fit.yaw = getYaw() + 0.05`（其余量不动），
   断言 `getYaw()` 明显朝 `fit.yaw` 移动（当前实测完全不动）。
3. 离线 A/B：`tools/replay_obs_csv` 打开 `local_fit.enabled` + `use_yaw`，
   比较 `trusted_fit`（模型-观测残差）与车心稳定性；修复后"开 use_yaw"必须**有变化**。
4. 仿真：`--gyro 1 --gyro-speed 3.0`，看 `yaw` 列（MODEL_CSV）是否与真值同相。

**P0-2**

1. `geometry=` 计数在拟合有效时应开始增长（不再恒为 0）。
2. 状态行 `Rgeo=` 应从 `0.250/0.250/0.250/0.250(cfg)` 缓慢收敛到实测半径
   （实车 ≈0.20~0.24），且**限幅**内（单次 ≤ `radius_max_step`）。
3. 关/开 `use_fit_radius` 的 A/B：四块板重投影像素残差应下降且不出现发散。

**P1-1 / P1-2**

1. 单测：构造一个"参考板 = 3 号"的拟合结果，断言注入后 EKF 的
   `getPlateAngle(3)` 与 fit.yaw 一致，且**不依赖**模 90° 吸附。
2. 一致性单测：`angleForPlate(k)` 与 `plate_yaw_offsets[k]` 在 k=0..3 上逐一相等
   （或显式记录两者的固定换算关系）。

**P2-1**

1. 统计：拟合器弧段数 `arcs`、`cand0`（有灯条但没够 min_bars 的弧）在
   ROI 收缩与全帧两种模式下对比；全帧模式下 `cand0` 应显著下降。

---

## 回归测试建议

1. **新增 `armor_ekf_angle_test`**：`updateAngle` 对 4 个板位的增益/符号 + 有限差分雅可比
   一致性（把 `/tmp/h_probe`、`/tmp/angle_probe` 固化成 gtest）。
2. **新增 `local_fit_injection_test`**：`updateWithLocalFit` 的 yaw / ω 两个通道
   各自单独打开时状态都确实变化。
3. **接口固定测试**：`LocalFitResult` 若加字段，必须有构造/序列化测试，
   避免加载器与结构体默认值漂移（仓库已有 `config_dump` 这个思路）。
4. **相位约定测试**：EKF `angleForPlate` ↔ `chassis_model::armorPhaseOffset` ↔
   拟合器 `plate_yaw_offsets` ↔ `target_selector::angleForPlate` 四处**同一张表**，
   用一条断言把四处绑起来。
5. **既有 21 项 ctest 全绿**作为底线（含 `armor_geometry_test`、`armor_ekf_test`）。

---

## 附：审计过程中发现的、与"职责交集"无关但需要先处理的事实

**实车小电脑（`ssh nuc`）的工作树与本地不一致**：

```
HEAD: 9358ba4 local-fit: 真机入口接入局部拟合器 + 把运行组件从 BUILD_TESTS 里拿出来
git status: 22 files changed, 2634 insertions(+), 311 deletions(-)   （脏工作树）
```

本地 `codex/test` 的 HEAD 是 `113810c`。也就是说：**之前贴出的实车日志来自
`9358ba4 + 22 个未提交改动`这棵混合树**，本报告的结论只对本地代码成立。
特别地，那些日志里出现过 `geometry=1624`，而**当前源代码里
`local_fit_apply_geometry_` 没有任何自增点**（本地与 NUC 都一样，已核对），
所以那条日志必然来自更早的构建。建议：先让两边代码对齐，再解释任何实车数字。
