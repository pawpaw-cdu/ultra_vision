#ifndef ULTRA_VISION_AUTO_AIM_ESTIMATION_POSE_2D_SOLVER_HPP
#define ULTRA_VISION_AUTO_AIM_ESTIMATION_POSE_2D_SOLVER_HPP

// 6D PnP → 4D（机械约束）→ **2D**（拟合器给 ψ、t_z，只用角点解 t_x,t_y）的最后一环。
// 对应文档《从 6D PnP 到 2D 约束优化的完整数学推导》§5-§7。
//
// **一个反直觉但省事的事实**：在"ψ 与 t_z 固定"的前提下这个问题**严格线性**，
// 不需要高斯-牛顿迭代——对 t_x、t_y 各是一元二次最小二乘，可以直接闭式解出来
// （推导见 .cpp）。方案里写的"2×2 线性系统，几乎瞬间可解"是对的，而且比它更强：
// 一次都不需要迭代（只有放开 t_z 时才用几轮 GN）。
//
// 约定（与 perception/pnp_solver.cpp 的 armorKeypointBasis + getArmor3DPoints
// **逐位一致**——必须如此，否则这里的 ψ 与 refineArmorYaw() 的 ψ 不是同一个角）：
//     物点顺序 lb, lt, rt, rb：  P_i = tvec + y_i·tangent(ψ) + z_i·vertical(ψ, pitch)
//     u = fx·X/Z + cx,  v = fy·Y/Z + cy
//
// 与局部拟合器的连接（本模块存在的理由）：
//     fitterArmorPose() 把拟合器的刚体解 (C, θ, R, h, α) 经 R_cv 换成
//     ① 板心在相机系的位置 ② 该板法向的 ψ ③ 板心深度 t_z。
//     ⚠ **不要拿拟合器的 C_z 当 t_z**：C_z 是"旋转中心"深度，与板心深度相差
//     [R_cv·(R·n̂ + h·ẑ)]_z（R=0.2、θ=90° 时就是 20 cm），见
//     docs/plan_audit_6d_pose.md §2.1。

#include <array>

#include <opencv2/core.hpp>

#include "common/types.hpp"

namespace auto_aim
{
    struct Pose2DInput
    {
        /// 四角像素（**顺序 lb, lt, rt, rb**；已去畸变；uv 与 PnP 同源）
        std::array<cv::Point2f, 4> corners{};
        /// 物点（`getArmor3DPoints` 的顺序与尺寸）。只用 y/z 分量（板面内两轴）。
        std::array<cv::Point3f, 4> object{};
        double fx = 0.0, fy = 0.0, cx = 0.0, cy = 0.0;
        /// 板法向方位角（rad），与 `refineArmorYaw` 同口径。
        double psi = 0.0;
        /// 板心深度（m）。`solve_tz=false` 时按已知量固定。
        double tz = 0.0;
        /// 机械俯仰（RM 规范 ≈15°）。**它决定单板 yaw 是否一阶可观测**。
        double pitch = 15.0 * 3.14159265358979323846 / 180.0;
        /// true：把 t_z 也放开（3 自由度，用几轮 GN）。
        bool solve_tz = false;
    };

    struct Pose2DOutput
    {
        bool valid = false;
        double tx = 0.0, ty = 0.0, tz = 0.0;
        double rmse_px = 0.0;      // 8 个残差（4 角 × u,v）的 RMSE
        int iterations = 0;
        /// 最小的那个 Z_i（诊断：≤0 表示有点跑到相机背面，直接判无效）
        double min_depth = 0.0;
    };

    /// 固定 (ψ, t_z) 解 (t_x, t_y)；`solve_tz` 时再放开 t_z。
    Pose2DOutput solvePose2D(const Pose2DInput& input);

    /// 一维 ψ 扫描的产物：最优 ψ + 该处残差 + **代价曲率**（可观测性代理量）。
    /// `info` 是 profile cost 对 ψ 的二阶导（单位 px²/rad²）；正面观测时它接近 0
    /// ——这正是单帧 PnP yaw 病态的本质，也是"拟合器有没有必要"的判据。
    struct YawScanResult
    {
        double psi = 0.0;
        double rmse_px = 0.0;
        double info = 0.0;
    };

    /// 在 `psi_center ± span` 内以 `step` 扫 ψ，返回 profile cost 最小的那一点。
    YawScanResult scanYaw(const Pose2DInput& input, double psi_center, double span, double step);

    /// 拟合器 → 2D 求解器需要的量。
    struct FitterArmorPose
    {
        bool valid = false;
        cv::Vec3d center{0, 0, 0};   // 板心（相机系，m）
        double psi = 0.0;            // 该板法向的方位角（与 refineArmorYaw 同口径）
        double radius = 0.0;         // 板心到旋转中心的水平距离（诊断）
    };

    /// 把拟合器的刚体解换成"某一块板的板心位置 + ψ"。
    /// @param plate_id 板号 0..3（与 `armorPhaseOffset` 同一张表）
    /// @param R_cv     拟合器用的"车体系→相机系"3×3（**行主序**，9 个 double）
    FitterArmorPose fitterArmorPose(const LocalFitResult& fit, int plate_id, const double R_cv[9]);
}

#endif  // ULTRA_VISION_AUTO_AIM_ESTIMATION_POSE_2D_SOLVER_HPP
