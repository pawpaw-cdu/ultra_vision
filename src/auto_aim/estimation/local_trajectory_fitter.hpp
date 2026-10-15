#ifndef AUTO_AIM_ESTIMATION_LOCAL_TRAJECTORY_FITTER_HPP
#define AUTO_AIM_ESTIMATION_LOCAL_TRAJECTORY_FITTER_HPP

// 局部拟合器（快层 θ/ω + 慢层 α/R）——短窗小陀螺轨迹拟合。
//
// 提示词要点：
//   · 快层：固定 α、R，用 Ceres 拟合 C,h,θ0,ω,β（+微弱漂移速度先验），
//     多板联合（同帧多块装甲板共享 C/h/θ0/ω，各板 yaw 相差固定角）。
//   · 慢层：固定快层最优参数，只放开 α、R 做一小步（确定性 Gauss-Newton，
//     输出 JᵀJ 对角信息量），带遗忘/门控交给调用方（SlowCalib）融合。
//
// 依赖：Eigen3 + Ceres（可选组件，与 spin_geometry_estimator 同款：
//       只有找到 Ceres 时才编译，不链接进 auto_aim_core）。
// 实测结论见 experiments/local_fitter/README.md：单板短窗 R 不可辨识，
// 四板联合后 α 收敛到 15±1、R 进入 ±0.02。

#include <array>
#include <chrono>
#include <vector>

#include "common/types.hpp"

namespace auto_aim
{
    struct LocalFitIntrinsics
    {
        double fx = 800.0, fy = 800.0, cx = 432.0, cy = 960.0;
    };

    struct LocalTrajectoryFitConfig
    {
        LocalFitIntrinsics K;
        double lightbar_len = 0.10;     // 灯条长度 L（板高，米）
        double plate_width = 0.135;     // 装甲板宽度 W（米；决定两灯条切向偏移 ±W/2）
        double gimbal_pitch = 0.0;      // 云台俯仰（rad，构造 R_cv）
        double pitch_offset = 0.0;      // 俯仰偏置（rad，加在 gimbal_pitch 上：相机安装俯仰，现场标定）
        // 直接用完整 R_cv（车辆系→相机系，行主序）覆盖上面的 Rx(gimbal_pitch)·R_base。
        // 用于把"相机→云台安装旋转"（hand-eye R_camera2gimbal）纳入模型——实车必须有。
        bool use_rcv = false;
        std::array<double, 9> rcv{{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}};
        std::array<double, 3> center_init{{0.0, 0.0, 3.0}};
        double h_init = 0.10;           // 灯条中心高度初值（车辆系）
        // ---- 高度 h（板心相对旋转中心的高度，车辆系，向上为正）的**慢层先验** ----
        // h 在快层里是自由变量，而它与 Cz 强耦合（板心的 v 坐标同时受两者影响）：
        // 拟合器会把本该属于 Cz 的误差吸进 h。给它一个慢层先验（σ 内自由），
        // 就等于把"板心高度是机械常数"这条硬约束**显式**加进残差 —— 部分打破 h-Cz 耦合。
        // 先验值由 tracker 慢层几何提供（geometry.plate_height，缓慢收敛）。
        // 量化前提（本项目实车实测）：相机与板心高度差仅 ~6cm（obs_y/cz≈0.02），
        // ∂v/∂Cz ≈ f·Δh/Cz² ≈ 16 px/m（不是 44 px/m），所以这条约束的**绝对增益有限**，
        // 主要价值是"别让 h 乱跑"。
        double h_prior = 0.0;
        double h_prior_sigma = 0.05;    // m；<=0 表示不加先验
        double omega_init = 3.0;        // 角速度初值（rad/s，多起点用）
        int omega_model_order = 1;      // 1=常 ω；2=带角加速度 β
        int max_iterations = 100;
        // 多起点的**总时间预算**（秒）：多起点是 16 个初值 × 每个最多 max_solver_time_s，
        // 在 NUC 上实测单次弧段拟合要 2.6~4.6s（fitms=），把 worker 占满、快层 ω 拿不到
        // 新鲜结果。这里给整个多起点循环一个总预算：预算用完就不再试新初值，
        // 直接用当前最好的解。慢层(R/α/h)本来就是秒级更新，够用。
        double multi_start_budget_s = 0.35;
        bool use_huber = true;
        double huber_delta = 2.0;       // px
        // 慢层（α、R）硬边界与先验
        double alpha_prior = 15.0 * M_PI / 180.0;
        double alpha_lo = 8.0 * M_PI / 180.0, alpha_hi = 25.0 * M_PI / 180.0;
        double R_prior = 0.20, R_lo = 0.10, R_hi = 0.35;
        // 慢层在线（估计 R）融合参数
        double R_learn = 0.5;           // 学习率（对 refineSlow 的 GN 步）
        double R_step_max = 0.01;       // 单次 R 变化上限（m）
        double R_freeze_band = 0.05;    // |Cz−2R| 小于它就冻结 R（不可辨识）
        double R_info_min = 1e-6;       // JᵀJ 对角信息量下限（低于则冻结）
        double R_prior_gain = 0.05;     // 弱先验回拉系数（防 R–Cz 耦合漂移）
        // 有效性
        double max_rmse_px = 5.0;
        // **角度**门限（度）：有效性的主判据。为什么必须有它：
        // 同一套几何误差，像素残差 ∝ f；本项目仿真相机 f≈1444，而实车 f≈2390（1.65×），
        // 只用 max_rmse_px 会把实车上 6~13px 的好拟合全判成无效（实车 log: fit=0/3、pub=0）。
        // 实际门限取 max(max_rmse_px, fx·tan(max_rmse_deg))，即"绝对像素"当保底，
        // 角度门限随分辨率自动放大。
        double max_rmse_deg = 0.4;
        // Ceres 单次求解的时间上限（秒）。NUC 这类低功耗 x86 上，一次弧段拟合
        // （fitWithSlow 会跑 2~3 次快层 + 慢层）可能要几百 ms；不限时会让 worker
        // 几秒才转一圈，结果算完就已经"十几秒前"了（实车 log: fit=1/2、age=11.8s）。
        // 注意：这是**安全网**，不是常规限速。仿真实测：0.05s → 有效解 17%（被掐断）；
        // 0.25s → 99%、5s → 同样 99%，所以 0.25s 足够且能防住病态初值。
        // 慢层结果的工龄由 LocalFitConfig::max_slow_result_age 兜（默认 2s）。
        double max_solver_time_s = 0.25;
        // 板 yaw 偏移（armor_id → 该板相对**底盘 yaw** 的相位偏移）。
        //
        // **必须与 ArmorEKF::angleForPlate / chassis_model::armorPhaseOffset 同一张表**
        // —— 也就是 (1−k)·90°：plate 0 → +90°（车左侧）、1 → 0°（正面）、
        // 2 → −90°、3 → −180°。理由：拟合输出的 `LocalFitResult::yaw` 是给 EKF 当
        // 底盘 yaw 用的（ArmorEKF::updateWithLocalFit 以"零偏移板"注入），
        // 两张表不一致时拟合 yaw 与 EKF yaw 会整体差 ±90°/180°：四块板对称时被
        // "模 90° 吸附"掩盖，启用逐板几何（plate_distance / l / h）后就变成模型分叉。
        // 2026-10-11 从 {0, 90, 180, 270} 改成本表（两者差一次 id 反射 0↔1、2↔3，
        // 模型的几何解不变，只是 θ 的代表元对齐）。见
        // git 历史里的 docs/local_fit_tracker_responsibility_audit.md 审计项 G。
        std::vector<double> plate_yaw_offsets{
            0.5 * M_PI, 0.0, -0.5 * M_PI, -M_PI};
    };

    class LocalTrajectoryFitter
    {
    public:
        /// @brief 快层拟合结果（内部模型参数：theta(t)=theta0+omega·t+½β·t²，
        ///        center(t)=center+drift·(t−t0)）。
        struct FastFit
        {
            bool valid = false;
            double center[3] = {0, 0, 0};
            double h = 0.0;
            double theta0 = 0.0, omega = 0.0, beta = 0.0;
            double drift[3] = {0, 0, 0};
            double t0 = 0.0;
            double rmse_px = 0.0;
            // 相位用**相对时间** (t - t0)：绝对时间戳（仿真/真机的 µs 纪元）会让 ω·t 溢出精度。
            double yawAt(double t) const { const double dt = t - t0; return theta0 + omega * dt + 0.5 * beta * dt * dt; }
        };

        explicit LocalTrajectoryFitter(LocalTrajectoryFitConfig config = {}) : config_(std::move(config)) {}

        void setConfig(const LocalTrajectoryFitConfig& config) { config_ = config; }
        const LocalTrajectoryFitConfig& config() const { return config_; }

        /// @brief 快层：固定 α、R，拟合 C,h,θ0,ω,β（多板联合）。
        bool fitFast(const std::vector<LightBarObservation>& observations, double alpha_fixed,
                     double R_fixed, FastFit& out) const;

        /// @brief 一步到位：fitFast + 用 α/R 先验算 yaw(t_end)，填 LocalFitResult。
        bool fit(const std::vector<LightBarObservation>& observations, double alpha_fixed,
                 double R_fixed, LocalFitResult& out) const;

        /// @brief 慢层一步：固定快层最优参数，只放开 α、R；输出增量与 JᵀJ 信息量。
        void refineSlow(const std::vector<LightBarObservation>& observations,
                        const FastFit& fast, double& alpha, double& R,
                        double& info_alpha, double& info_R) const;

        // ---- 慢层在线状态（α 固定、R 在线估计）--------------------------------------
        // 每步：fitFast(α,R) → refineSlow 对 R 走一小步（确定性 GN + JᵀJ 信息量）→
        // 门控融合（Cz≈2R / 信息量不足时冻结；步长上限；先验范围）。
        void seedSlow(double alpha, double R) { slow_alpha_ = alpha; slow_R_ = R; }
        double slowAlpha() const { return slow_alpha_; }
        double slowR() const { return slow_R_; }
        bool fitWithSlow(const std::vector<LightBarObservation>& observations, LocalFitResult& out);

    private:
        LocalTrajectoryFitConfig config_;
        double slow_alpha_ = 0.0;
        double slow_R_ = 0.20;
    };
}  // namespace auto_aim

#endif  // AUTO_AIM_ESTIMATION_LOCAL_TRAJECTORY_FITTER_HPP
