#pragma once
#include "types.h"
#include <vector>

namespace rm
{
    struct FitConfig
    {
        Vec3 center_init{0, 0, 3};
        double h_init = 0.10;
        double theta0_init = 0.0, omega_init = 3.0;
        int omega_model_order = 1;      // 1 = 常 ω；2 = 带 β
        int max_iterations = 100;
        double function_tolerance = 1e-10, gradient_tolerance = 1e-10;
        bool use_huber = true;
        double huber_delta = 2.0;
    };

    struct FitResult
    {
        bool success = false;
        Vec3 center_cam{0, 0, 0};
        Vec3 drift_velocity{0, 0, 0};
        double h = 0.0;
        double theta0 = 0.0, omega = 0.0, beta = 0.0;
        double reprojection_rmse = 0.0;
        int num_iterations = 0;
        double alpha_used = 0.0, R_used = 0.0;
        double yawAt(double t) const { return theta0 + omega * t + 0.5 * beta * t * t; }
        Vec3 centerAt(double t, double t0) const { return center_cam + drift_velocity * (t - t0); }
    };

    // 快层：α、R 固定（来自慢层），估计 C,h,θ0,ω,β。
    FitResult fitTrajectory(const std::vector<LightBarObservation>& observations,
                            const CameraIntrinsics& K, double gimbal_pitch,
                            double alpha_fixed, double R_fixed, double L,
                            const FitConfig& cfg);

    // 慢层一步：固定快层最优参数，只放开 α、R 迭代一小步；输出增量与信息量 (JᵀJ 对角)。
    void refineSlowParams(const std::vector<LightBarObservation>& observations,
                          const CameraIntrinsics& K, double gimbal_pitch, double L,
                          const FitResult& fast, double& alpha, double& R,
                          double& info_alpha, double& info_R);
}  // namespace rm
