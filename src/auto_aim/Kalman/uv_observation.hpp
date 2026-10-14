#ifndef ULTRA_VISION_AUTO_AIM_KALMAN_UV_OBSERVATION_HPP
#define ULTRA_VISION_AUTO_AIM_KALMAN_UV_OBSERVATION_HPP

// UV-EKF（刚体分解）的观测契约：**直接用四个灯条端点的像素**更新整车状态。
//
// 与局部拟合器的区别：拟合器要"一段弧 + 非凸优化（Ceres）+ 慢层"，
// 这里是把同一个刚体模型**放进 EKF 的观测函数**里，每帧一次、无额外优化器。
//
// 模型（与 estimation/local_trajectory_fitter 的 barEndpointV **同构**，常数都取实测值）：
//     p(a,b) = armorPosition(k) + a·t̂ + b·b̂        a = ±W/2（两根灯条），b = ±L/2（上下端点）
//     t̂ = (−sinφ, 0, cosφ)   切向（水平）
//     b̂ = −(sinα·n̂ + cosα·ŷ) 灯条方向（顶端朝**车内**；实测 α≈14.5°）
//     n̂ = (cosφ, 0, sinφ),  φ = θ + armorPhaseOffset(k)
//
// **差分信号与平均信号**：
//   Δ = p(+L/2) − p(−L/2) = L·b̂     —— 只依赖 θ（与 C 无关）→ 差分给旋转信息；
//   M = (p(+L/2) + p(−L/2))/2        —— 含 C 与 θ             → 平均给平移信息。
// 这就是方案里"刚体分解"的数学内容。
//
// ⚠ **与方案文档的一处修正**：方案建议"先做一次差分更新、再做一次平均更新"。
// 那是**近似**——差分与平均来自**同一组像素**、必然相关，分两次顺序更新等于假设两者独立。
// 这里改为**一次 8 维更新**，把"差分噪声 σ_d / 平均噪声 σ_a"写进观测协方差 R 的分块结构：
//     Var(top) = Var(bot) = σ_a² + σ_d²/4
//     Cov(top, bot)       = σ_a² − σ_d²/4
// 数学等价、且不需要"两者独立"这个不成立的假设。

#include <array>
#include <cmath>

#include "common/types.hpp"

namespace auto_aim
{
    /// 灯条几何（实测/官方标定值，见 docs/design_decision_local_fit.md）：
    ///   W = 0.135（官方小装甲长 135mm）、L = 0.0551（灯条 top-bottom 实测 55.11mm）、
    ///   α = 14.5°（从真实关键点 + 相机姿态量出来）。
    struct UvGeometry
    {
        double plate_width = 0.135;
        double lightbar_len = 0.0551;
        double alpha = 14.5 * 3.14159265358979323846 / 180.0;
        double alpha_sin() const { return std::sin(alpha); }
        double alpha_cos() const { return std::cos(alpha); }
    };

    struct UvObservation
    {
        /// 四个灯条端点像素，顺序 = [左灯条上, 左灯条下, 右灯条上, 右灯条下]
        /// （与 Armor::Points_2D 的 lb/lt/rt/rb 是同一组点，只是顺序不同）。
        std::array<cv::Point2f, 4> corners{};
        int plate_id = -1;

        // 相机
        double fx = 0.0, fy = 0.0, cx = 0.0, cy = 0.0;
        /// world→camera 的旋转（**行主序** 9 个数）与相机光心在世界系的位置：
        ///   p_cam = R_w2c · (p_world − camera_origin)
        std::array<double, 9> R_world2camera{{1, 0, 0, 0, 1, 0, 0, 0, 1}};
        std::array<double, 3> camera_origin{{0, 0, 0}};

        /// 观测噪声：差分（同一灯条上端点 − 下端点）与平均（两端点中点）的像素标准差。
        /// 实测量级：端点噪声 ≈0.67px（静止段同板逐帧位移/√2，n=1524）。
        double sigma_diff_px = 0.7;
        double sigma_avg_px = 0.7;
    };
}

#endif  // ULTRA_VISION_AUTO_AIM_KALMAN_UV_OBSERVATION_HPP
