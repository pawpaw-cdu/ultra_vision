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
