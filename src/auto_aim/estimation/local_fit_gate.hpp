#ifndef ULTRA_VISION_AUTO_AIM_ESTIMATION_LOCAL_FIT_GATE_HPP
#define ULTRA_VISION_AUTO_AIM_ESTIMATION_LOCAL_FIT_GATE_HPP

// 局部拟合结果的**分层门控**：ω 与 yaw 的可信条件完全不同，必须分开判。
//
// 背景（2026-10-12 用实车检测流量出来的，见 git 历史里的 docs/plan_audit_6d_pose.md §0）：
// 把拟合器的 yaw 与整车 EKF 的 yaw 对比（模 90°），按拟合 rmse 分层：
//
//     fit rmse < 3 px    n=312    中位 +0.49°   std  0.57°
//     fit rmse 3~8 px    n=223    中位 +1.21°   std  9.67°
//     fit rmse 8~16 px   n=1185   中位 +5.67°   std 22.56°
//
// 也就是说：**yaw 只在"拟合很好"那一档才可信**（σ_ψ≈0.6°，与方案假设的 0.5~1° 一致），
// 而 rmse 8~16px 那一大坨（占 68%）的 yaw 是**22° 量级**的垃圾；ω 却一直稳
// （静止段 ω≈0、自旋段落在 [0, 5.8]，是"无滞后"的那一路观测）。
//
// 而修复前代码里 yaw 通道**只有** `use_yaw` + `|ω|` 混合 + 参考板门控，**没有质量门**：
// 自旋时 |ω|≥2.5 就给满权重，等于把 σ_ψ=22° 的那一坨直接注进 EKF。
// 这正是 yaml 里那句"直接注入会让慢转工况从 0.3px 劣化到 30px"的来源。
//
// 所以这里给出两把纯函数尺子（可单测、不依赖 Tracker/EKF）：
//   ① rmse 门：模型和观测贴不贴（噪声/建模误差的代理）；
//   ② 相位跨度门：这条弧有没有**看到足够的旋转**——覆盖不足时代价函数在 yaw 方向
//      是平谷，残差可以很小而 yaw 仍然不可信（方案 §3.3 的边界条件）。

#include <cmath>

#include "common/types.hpp"

namespace auto_aim
{
    struct LocalFitGateConfig
    {
        /// yaw 通道允许的最大拟合 rmse（像素）。**实测拐点**：>3px 时 σ_ψ 从 0.6° 跳到 10~23°。
        /// 注意这是**绝对像素**阈值：换分辨率/换焦距要重标（ω 通道不受它影响）。
        double yaw_max_rmse_px = 3.0;
        /// yaw 通道要求的最小相位跨度（度）。24 帧窗口下 |ω|=1 rad/s ≈ 23°，
        /// 与既有 `yaw_use_omega_min = 1.0` 同量级；显式写出来是为了让"为什么慢转不喂 yaw"可解释。
        double yaw_min_phase_span_deg = 20.0;

        /// ω 通道允许的最大拟合 rmse（像素）。
        /// 2026-10-12 实测修正了一个一直以来的假设："ω 在任何档都稳"是**错的**——
        /// 把 min_depth 放宽到 0.8 后注入次数 36→343，底盘一致性指标从 32.6px 劣化到
        /// 69.7px（此时 yaw 已被门全部挡下，害人的只能是 ω 通道）。所以 ω 也要过质量门：
        /// 拟合没贴住观测时，连角速度一起丢掉。<=0 表示不门（旧行为，A/B 用）。
        double omega_max_rmse_px = 3.0;
    };

    enum class LocalFitYawVerdict
    {
        ok = 0,
        bad_rmse,     // 模型没贴住观测 → yaw 不可信
        bad_span,     // 弧段没看到足够的旋转 → yaw 不可辨识
        invalid       // 结果本身无效
    };

    inline const char* localFitYawVerdictName(LocalFitYawVerdict v)
    {
        switch (v)
        {
            case LocalFitYawVerdict::ok: return "ok";
            case LocalFitYawVerdict::bad_rmse: return "bad_rmse";
            case LocalFitYawVerdict::bad_span: return "bad_span";
            default: return "invalid";
        }
    }

    inline LocalFitYawVerdict evaluateLocalFitYaw(
        const LocalFitResult& fit, const LocalFitGateConfig& cfg)
    {
        if (!fit.valid) return LocalFitYawVerdict::invalid;
        if (!std::isfinite(fit.reprojection_rmse_px)) return LocalFitYawVerdict::bad_rmse;
        if (cfg.yaw_max_rmse_px > 0.0 && fit.reprojection_rmse_px > cfg.yaw_max_rmse_px)
            return LocalFitYawVerdict::bad_rmse;
        const double span_deg = std::abs(fit.phase_span_rad) * 180.0 / 3.14159265358979323846;
        if (cfg.yaw_min_phase_span_deg > 0.0 && span_deg < cfg.yaw_min_phase_span_deg)
            return LocalFitYawVerdict::bad_span;
        return LocalFitYawVerdict::ok;
    }

    /// ω 通道的质量门。**与 yaw 的区别只在"不看相位跨度"**：ω 的符号/大小在静态段
    /// 也能被弧段约束住（静止时 ω≈0 是有效观测，正好压掉单帧 PnP 造出来的假转速），
    /// 所以 ω 不需要跨度门；但同样需要 rmse 门（见 omega_max_rmse_px 注释）。
    inline bool localFitOmegaUsable(const LocalFitResult& fit, const LocalFitGateConfig& cfg)
    {
        if (!fit.valid || !std::isfinite(fit.omega)) return false;
        if (cfg.omega_max_rmse_px > 0.0 && std::isfinite(fit.reprojection_rmse_px) &&
            fit.reprojection_rmse_px > cfg.omega_max_rmse_px)
            return false;
        return true;
    }
}

#endif  // ULTRA_VISION_AUTO_AIM_ESTIMATION_LOCAL_FIT_GATE_HPP
