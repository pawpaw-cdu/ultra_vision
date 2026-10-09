#pragma once
#include "types.h"
#include <string>
#include <vector>

namespace rm
{
    struct SimulationConfig
    {
        CameraIntrinsics K;
        int image_width = 864, image_height = 1920;
        Vec3 center_cam{0, 0, 3};
        double radius = 0.20;
        double pitch_install = 15.0 * M_PI / 180.0;
        double gimbal_pitch = 0.0;
        double omega0 = 3.0, omega_accel = 0.0, theta0 = 0.0;
        double lightbar_len = 0.10, lightbar_h = 0.10;
        Vec3 drift_velocity{0, 0, 0};
        Vec3 drift_accel{0, 0, 0};
        int num_frames = 200;
        double dt = 0.01;
        double pixel_noise_std = 0.5;
        bool enable_noise = true;
        // 贴近实战的坏条件
        double outlier_frac = 0.0;      // 外点帧比例
        double outlier_px = 0.0;        // 外点幅值
        double box_jitter_px = 0.0;     // 整块板共享的框抖动（相关误差）
        double dropout_frac = 0.0;      // 随机丢帧
        double timeline_jitter_std = 0.0; // 时间戳抖动（秒）
        double gimbal_pitch_rate = 0.0; // 俯仰随时间变化率（rad/s，模拟时变）
        unsigned seed = 7;
    };

    struct SimulationFrame
    {
        double timestamp = 0.0;
        Vec2 top_px{0, 0}, bottom_px{0, 0};
        Vec3 center_cam_gt{0, 0, 0};
        double yaw_gt = 0.0, omega_gt = 0.0, gimbal_pitch_gt = 0.0;
    };

    std::vector<SimulationFrame> runSimulation(const SimulationConfig& cfg, const std::string& csv_path);
}  // namespace rm
