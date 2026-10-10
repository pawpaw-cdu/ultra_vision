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
        double omega_init = 3.0;        // 角速度初值（rad/s，多起点用）
        int omega_model_order = 1;      // 1=常 ω；2=带角加速度 β
        int max_iterations = 100;
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
        // 板 yaw 偏移（armor_id → 偏移角）。默认四板 90° 均布。
        std::vector<double> plate_yaw_offsets{0.0, M_PI / 2, M_PI, 3.0 * M_PI / 2};
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
