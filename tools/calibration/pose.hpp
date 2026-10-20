#ifndef ULTRA_VISION_CALIBRATION_POSE_HPP
#define ULTRA_VISION_CALIBRATION_POSE_HPP

#include <cmath>

#include <opencv2/core.hpp>

namespace calibration
{
    /// Gimbal/base rotation with the same Y-then-X convention used by Tracker.
    inline cv::Matx33d gimbalToBase(double yaw, double pitch)
    {
        const double cy = std::cos(yaw), sy = std::sin(yaw);
        const double cp = std::cos(pitch), sp = std::sin(pitch);
        return cv::Matx33d(cy, sy * sp, sy * cp,
                           0.0, cp, -sp,
                           -sy, cy * sp, cy * cp);
    }

    inline cv::Matx33d quaternionToRotation(double w, double x, double y, double z)
    {
        const double norm = std::sqrt(w * w + x * x + y * y + z * z);
        if (norm < 1e-12) return cv::Matx33d::eye();
        w /= norm;
        x /= norm;
        y /= norm;
        z /= norm;
        return cv::Matx33d(
            1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
            2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
            2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y));
    }
} // namespace calibration

#endif // ULTRA_VISION_CALIBRATION_POSE_HPP
