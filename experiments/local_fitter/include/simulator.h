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
        // 多装甲板 / 车型参数（提示词 4.5 刚体分解、§八 车型切换）
        int num_plates = 1;             // 1 = 单板（旧行为）；4 = 整车四板 90° 均布
        double plate_width = 0.135;     // 装甲板宽度 W(m)（仅多板/四角时用）
        bool gate_visibility = false;   // 只保留朝向相机的板（自遮挡）
        double visibility_half_angle = 75.0 * M_PI / 180.0;
        double radius_step_time = -1.0; // >=0 时，t 超过该时刻 R 跳到 radius_after（车型切换）
        double radius_after = 0.28;
        unsigned seed = 7;
    };

    struct SimulationFrame
    {
        double timestamp = 0.0;
        Vec2 top_px{0, 0}, bottom_px{0, 0};
        Vec3 center_cam_gt{0, 0, 0};
        double yaw_gt = 0.0, omega_gt = 0.0, gimbal_pitch_gt = 0.0;
    };

    // 一块装甲板在某一帧的观测：4 个角点像素 [LT, LB, RT, RB]。
    struct PlateObservation
    {
        int armor_id = 0;
        Vec2 corners_px[4];
        double confidence = 1.0;
    };

    struct MultiFrame
    {
        double timestamp = 0.0;
        std::vector<PlateObservation> plates;
        Vec3 center_cam_gt{0, 0, 0};
        double yaw_gt = 0.0, omega_gt = 0.0, gimbal_pitch_gt = 0.0, radius_gt = 0.0;
        bool any_visible = false;
    };

    std::vector<SimulationFrame> runSimulation(const SimulationConfig& cfg, const std::string& csv_path);

    // 多装甲板仿真：四板 90° 均布，每板 4 角点；可选自遮挡门控与车型切换。
    std::vector<MultiFrame> runMultiSimulation(const SimulationConfig& cfg, const std::string& csv_path);
}  // namespace rm
