#ifndef ULTRA_VISION_AUTO_AIM_KALMAN_UV_OBSERVATION_HPP
#define ULTRA_VISION_AUTO_AIM_KALMAN_UV_OBSERVATION_HPP

// UV-EKF（刚体分解）的观测契约：**直接用四个灯条端点的像素**更新整车状态。

#include <array>
#include <cmath>

#include "common/types.hpp"

namespace auto_aim
{
    /// 灯条几何（实测/官方标定值）：
    ///   W = 0.135
    ///   α = 14.5
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
