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

    // Armor plates are flat and tangent to the chassis circle, so the plate
    // outline is built from the radial phase and a shared vertical axis.
    inline std::vector<cv::Point2f> projectArmor(
        const cv::Mat& center, int plate_id, double yaw,
        const cv::Vec3d& vertical_axis, double half_width, double half_height,
        const cv::Mat& camera_matrix, const cv::Mat& dist_coeffs)
    {
        if (center.rows != 3 || center.cols != 1) return {};

        const double total_angle = yaw + ArmorEKF::angleForPlate(plate_id);
        const cv::Vec3d tangent(-std::sin(total_angle), 0.0, std::cos(total_angle));
        const double cx = center.at<double>(0);
        const double cy = center.at<double>(1);
        const double cz = center.at<double>(2);

        const std::vector<cv::Point3f> object_points = {
            cv::Point3f(cx - tangent[0] * half_width - vertical_axis[0] * half_height,
                        cy - vertical_axis[1] * half_height,
                        cz - tangent[2] * half_width - vertical_axis[2] * half_height),
            cv::Point3f(cx - tangent[0] * half_width + vertical_axis[0] * half_height,
                        cy + vertical_axis[1] * half_height,
                        cz - tangent[2] * half_width + vertical_axis[2] * half_height),
            cv::Point3f(cx + tangent[0] * half_width + vertical_axis[0] * half_height,
                        cy + vertical_axis[1] * half_height,
                        cz + tangent[2] * half_width + vertical_axis[2] * half_height),
            cv::Point3f(cx + tangent[0] * half_width - vertical_axis[0] * half_height,
                        cy - vertical_axis[1] * half_height,
                        cz + tangent[2] * half_width - vertical_axis[2] * half_height)
        };

        std::vector<cv::Point2f> image_points;
        cv::projectPoints(object_points, cv::Mat::zeros(3, 1, CV_64F),
                          cv::Mat::zeros(3, 1, CV_64F),
                          camera_matrix, dist_coeffs, image_points);
        return image_points;
    }
} // namespace auto_aim

#endif // ULTRA_VISION_AUTO_AIM_VISUALIZATION_PROJECTION_HPP
