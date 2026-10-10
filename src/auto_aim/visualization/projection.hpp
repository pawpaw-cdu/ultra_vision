#ifndef ULTRA_VISION_AUTO_AIM_VISUALIZATION_PROJECTION_HPP
#define ULTRA_VISION_AUTO_AIM_VISUALIZATION_PROJECTION_HPP

// Overlay helpers shared by the hardware and simulator entry points. They draw
// the estimator's chassis model back onto the detection image, so the operator
// sees all four armor plates of the estimated vehicle instead of only the
// plates that happened to be detected in the current frame.

#include <cmath>
#include <cstddef>
#include <vector>

#include <opencv2/opencv.hpp>

#include "Kalman/armor_ekf.hpp"
#include "common/armor_geometry.hpp"

namespace auto_aim
{
    inline void drawQuad(cv::Mat& image, const std::vector<cv::Point2f>& points,
                         const cv::Scalar& color, int thickness)
    {
        if (points.size() != 4) return;
        for (std::size_t i = 0; i < points.size(); ++i) {
            cv::line(image, points[i], points[(i + 1) % points.size()],
                     color, thickness, cv::LINE_AA);
        }
    }

    inline bool projectPoint(const cv::Mat& point, const cv::Mat& camera_matrix,
                             const cv::Mat& dist_coeffs,
                             cv::Point2f& image_point)
    {
        if (point.rows != 3 || point.cols != 1) return false;
        const std::vector<cv::Point3f> object_points = {
            cv::Point3f(static_cast<float>(point.at<double>(0)),
                        static_cast<float>(point.at<double>(1)),
                        static_cast<float>(point.at<double>(2)))
        };
        std::vector<cv::Point2f> image_points;
        cv::projectPoints(object_points, cv::Mat::zeros(3, 1, CV_64F),
                          cv::Mat::zeros(3, 1, CV_64F),
                          camera_matrix, dist_coeffs, image_points);
        if (image_points.empty()) return false;
        image_point = image_points.front();
        return true;
    }

    // Armor plates are flat rigid bodies tangent to the chassis circle. The plate
    // outline is built in the **plate frame** (tangent × bar axis × normal) and the
    // three axes must all live in the same frame as `center` (normally the camera
    // frame). 用 common/armor_geometry.hpp 的 ArmorAxes 造轴，别在这里另起一套。
    //
    // 历史坑（都会被这个版本消掉）：
    //   ① 用世界系 yaw 造切向 → 相机一有 pitch，画出来的板就被掰一个角度；
    //   ② 竖直轴取"世界竖直"→ 漏掉真机装甲板的 15° 安装倾角（灯条其实是斜的）；
    //   ③ 角点组装时只把竖直轴的 y 分量加进 y、把切向的 y 分量丢了 →
    //      相机有 pitch 时四角在 y 上系统性错位（≈ tangent.y·half_width）。
    //      现在三轴都是三维向量，`center + t·u + v·w` 一次算完，没有特例。
    inline std::vector<cv::Point2f> projectArmorCameraFrame(
        const cv::Mat& center, const cv::Vec3d& tangent, const cv::Vec3d& bar_axis,
        double half_width, double half_height, const cv::Mat& camera_matrix,
        const cv::Mat& dist_coeffs)
    {
        if (center.rows != 3 || center.cols != 1) return {};

        const cv::Vec3d plate_center(center.at<double>(0), center.at<double>(1),
                                     center.at<double>(2));
        ArmorAxes axes;
        axes.tangent = tangent;
        axes.bar = bar_axis;
        const std::vector<cv::Vec3d> corners =
            armorCorners(plate_center, axes, half_width, half_height);

        std::vector<cv::Point3f> object_points;
        object_points.reserve(corners.size());
        for (const auto& p : corners)
            object_points.emplace_back(static_cast<float>(p[0]), static_cast<float>(p[1]),
                                       static_cast<float>(p[2]));

        std::vector<cv::Point2f> image_points;
        cv::projectPoints(object_points, cv::Mat::zeros(3, 1, CV_64F),
                          cv::Mat::zeros(3, 1, CV_64F),
                          camera_matrix, dist_coeffs, image_points);
        return image_points;
    }

    /// @brief 旧接口（按世界系 yaw 造切向、竖直轴由调用方给）——**只在相机姿态
    ///        接近 0 时正确**，且无法表达 15° 安装倾角。保留仅为兼容老工具，
    ///        新代码请用 Tracker::armorAxesCamera() + projectArmorCameraFrame()。
    inline std::vector<cv::Point2f> projectArmor(
        const cv::Mat& center, int plate_id, double yaw, const cv::Vec3d& bar_axis,
        double half_width, double half_height, const cv::Mat& camera_matrix,
        const cv::Mat& dist_coeffs)
    {
        const double total_angle = yaw + ArmorEKF::angleForPlate(plate_id);
        const cv::Vec3d tangent(-std::sin(total_angle), 0.0, std::cos(total_angle));
        return projectArmorCameraFrame(center, tangent, bar_axis, half_width, half_height,
                                       camera_matrix, dist_coeffs);
    }
} // namespace auto_aim

#endif // ULTRA_VISION_AUTO_AIM_VISUALIZATION_PROJECTION_HPP
