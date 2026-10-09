#ifndef AUTO_AIM_TRACKER_UTILS_HPP
#define AUTO_AIM_TRACKER_UTILS_HPP

// 从 Kalman/tracker.cpp 抽出的无状态工具（几何/变换/YPD 观测构造）。
// 放在 tracker_detail 命名空间里，tracker.cpp 用 using namespace 引入，调用点不用改。

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>

#include "Kalman/tracker.hpp"

namespace auto_aim
{
namespace tracker_detail
{
namespace
{
        Vec<3> toVec(const cv::Mat& tvec)
        {
            Vec<3> result;
            result(0, 0) = tvec.at<double>(0);
            result(1, 0) = tvec.at<double>(1);
            result(2, 0) = tvec.at<double>(2);
            return result;
        }

        cv::Mat toMat(const Vec<3>& value)
        {
            return (cv::Mat_<double>(3, 1) <<
                value(0, 0), value(1, 0), value(2, 0));
        }

        /// @brief NIS 硬门限 → 软门限换算。soft_nis_gate 打开时返回 +inf，
        ///        即"始终应用更新、只把 NIS 留给诊断"（对齐 sp_vision 的
        ///        recent_nis_failures：它绝不因为 NIS 而丢观测）。硬门限在高速
        ///        自旋下有正反馈——ω 没学到 → 残差大 → 拒更新 → ω 永远学不到。
        ///        UV 路径不经过这里，仍保留硬门限（用来挡矩形板的镜像二义）。
        double effectiveNisThreshold(bool soft_nis_gate, double hard_threshold)
        {
            return soft_nis_gate ? std::numeric_limits<double>::infinity()
                                 : hard_threshold;
        }

        Matrix<3, 3> cameraToBaseRotation(double yaw, double pitch)
        {
            const double cy = std::cos(yaw);
            const double sy = std::sin(yaw);
            const double cp = std::cos(pitch);
            const double sp = std::sin(pitch);

            // Simulator camera pose in Bevy coordinates:
            // R_home_camera = R_y(-yaw) * R_x(pitch). OpenCV optical axes are
            // converted with diag(1, -1, -1), so the camera-to-home transform
            // is S * R_home_camera * S.
            Matrix<3, 3> bevy_rotation;
            bevy_rotation(0, 0) = cy;
            bevy_rotation(0, 1) = -sy * sp;
            bevy_rotation(0, 2) = -sy * cp;
            bevy_rotation(1, 0) = 0.0;
            bevy_rotation(1, 1) = cp;
            bevy_rotation(1, 2) = -sp;
            bevy_rotation(2, 0) = sy;
            bevy_rotation(2, 1) = cy * sp;
            bevy_rotation(2, 2) = cy * cp;

            const std::array<double, 3> optical_sign{{1.0, -1.0, -1.0}};
            Matrix<3, 3> result;
            for (int row = 0; row < 3; ++row) {
                for (int column = 0; column < 3; ++column) {
                    result(row, column) = optical_sign[row] *
                        bevy_rotation(row, column) * optical_sign[column];
                }
            }
            return result;
        }

        // camera←gimbal 的旋转（手眼外参），默认单位阵
        Matrix<3, 3> cameraToGimbal(const CameraPose& pose)
        {
            Matrix<3, 3> rotation;
            for (int row = 0; row < 3; ++row) {
                for (int column = 0; column < 3; ++column) {
                    rotation(row, column) = pose.camera_to_gimbal[row][column];
                }
            }
            return rotation;
        }

        Vec<3> cameraToGimbalTranslation(const CameraPose& pose)
        {
            Vec<3> translation;
            for (int row = 0; row < 3; ++row) {
                translation(row, 0) = pose.camera_to_gimbal_translation[row];
            }
            return translation;
        }

        // 相机系点 → 世界系：先按手眼外参到云台系（含平移），再由 yaw/pitch 与
        // base_to_world 转到世界系。外参为单位阵/零时与旧行为完全一致。
        Matrix<3, 3> cameraToWorldRotation(const CameraPose& pose)
        {
            Matrix<3, 3> base_to_world;
            for (int row = 0; row < 3; ++row) {
                for (int column = 0; column < 3; ++column) {
                    base_to_world(row, column) = pose.base_to_world[row][column];
                }
            }
            return base_to_world * cameraToBaseRotation(pose.yaw, pose.pitch) *
                cameraToGimbal(pose);
        }

        Vec<3> pointToWorld(const CameraPose& pose, const Vec<3>& camera_point)
        {
            const Vec<3> gimbal_point =
                cameraToGimbal(pose) * camera_point + cameraToGimbalTranslation(pose);
            const Matrix<3, 3> base_to_world = [&pose] {
                Matrix<3, 3> matrix;
                for (int row = 0; row < 3; ++row) {
                    for (int column = 0; column < 3; ++column) {
                        matrix(row, column) = pose.base_to_world[row][column];
                    }
                }
                return matrix;
            }();
            return base_to_world * (cameraToBaseRotation(pose.yaw, pose.pitch) * gimbal_point);
        }

        Vec<3> pointToCamera(const CameraPose& pose, const Vec<3>& world_point)
        {
            const Matrix<3, 3> base_to_world = [&pose] {
                Matrix<3, 3> matrix;
                for (int row = 0; row < 3; ++row) {
                    for (int column = 0; column < 3; ++column) {
                        matrix(row, column) = pose.base_to_world[row][column];
                    }
                }
                return matrix;
            }();
            const Vec<3> gimbal_point =
                cameraToBaseRotation(pose.yaw, pose.pitch).transpose() *
                (base_to_world.transpose() * world_point);
            return cameraToGimbal(pose).transpose() *
                (gimbal_point - cameraToGimbalTranslation(pose));
        }

        Vec<3> transformVector(const Matrix<3, 3>& rotation, const Vec<3>& value)
        {
            Vec<3> result;
            for (int row = 0; row < 3; ++row) {
                for (int column = 0; column < 3; ++column) {
                    result(row, 0) += rotation(row, column) * value(column, 0);
                }
            }
            return result;
        }

        Vec<3> transformCvVector(const Matrix<3, 3>& rotation, const cv::Mat& value)
        {
            Vec<3> input;
            input(0, 0) = value.at<double>(0);
            input(1, 0) = value.at<double>(1);
            input(2, 0) = value.at<double>(2);
            return transformVector(rotation, input);
        }

        std::optional<cv::Mat> transformRotation(
            const Matrix<3, 3>& frame_rotation, const cv::Mat& rvec)
        {
            if (rvec.empty()) return std::nullopt;

            cv::Mat camera_rotation;
            cv::Rodrigues(rvec, camera_rotation);
            Matrix<3, 3> object_to_home;
            for (int row = 0; row < 3; ++row) {
                for (int column = 0; column < 3; ++column) {
                    for (int index = 0; index < 3; ++index) {
                        object_to_home(row, column) += frame_rotation(row, index) *
                            camera_rotation.at<double>(index, column);
                    }
                }
            }

            cv::Mat home_rotation(3, 3, CV_64F);
            for (int row = 0; row < 3; ++row) {
                for (int column = 0; column < 3; ++column) {
                    home_rotation.at<double>(row, column) = object_to_home(row, column);
                }
            }

            cv::Mat home_rvec;
            cv::Rodrigues(home_rotation, home_rvec);
            return home_rvec;
        }

        std::optional<double> armorRadialAngle(const Armor& armor)
        {
            if (!armor.solve_result || armor.rvec.empty() || armor.tvec.empty()) {
                return std::nullopt;
            }

            cv::Mat rotation;
            cv::Rodrigues(armor.rvec, rotation);
            cv::Vec3d normal(
                rotation.at<double>(0, 0),
                rotation.at<double>(1, 0),
                rotation.at<double>(2, 0));
            const cv::Vec3d position(
                armor.tvec.at<double>(0),
                armor.tvec.at<double>(1),
                armor.tvec.at<double>(2));

            // The visible armor face points toward the camera. Its horizontal
            // normal is also the radial direction from chassis center to plate.
            if (normal.dot(position) > 0.0) normal *= -1.0;
            const double norm = cv::norm(normal);
            if (norm < 1e-6) return std::nullopt;

            normal /= norm;
            return std::atan2(normal[2], normal[0]);
        }

        std::optional<YpdObservation> armorYpdObservation(
            const Armor& armor, int plate_id, const TrackerConfig& config)
        {
            if (!armor.solve_result || armor.tvec.empty()) return std::nullopt;
            const auto radial_angle = armorRadialAngle(armor);
            if (!radial_angle) return std::nullopt;

            const double x = armor.tvec.at<double>(0);
            const double y = armor.tvec.at<double>(1);
            const double z = armor.tvec.at<double>(2);
            const double horizontal = std::hypot(x, z);
            const double distance = std::sqrt(x * x + y * y + z * z);
            if (!std::isfinite(distance) || distance <= 1e-4) return std::nullopt;

            YpdObservation observation;
            observation.plate_id = plate_id;
            observation.yaw = std::atan2(x, z);
            observation.pitch = std::atan2(y, horizontal);
            observation.distance = distance;
            observation.armor_yaw = *radial_angle;
            observation.yaw_noise = config.bearing_measure_noise;
            observation.pitch_noise = config.pitch_measure_noise;
            if (config.ypd_distance_sigma_coeff > 0.0) {
                // 物理量纲：σ_dist = coeff · d（上科大 2026 §5B.2）。
                const double sigma = config.ypd_distance_sigma_coeff * distance;
                observation.distance_noise = sigma * sigma;
            } else {
                observation.distance_noise = config.measure_noise *
                    (1.0 + config.range_noise_distance_scale * distance * distance);
            }
            observation.armor_yaw_noise = config.angle_measure_noise;
            return observation;
        }

        Vec<3> centerFromRadialAngle(
            const Vec<3>& observation, double radial_angle, double radius,
            double armor_y_offset = 0.0)
        {
            Vec<3> center;
            center(0, 0) = observation(0, 0) - radius * std::cos(radial_angle);
            center(1, 0) = observation(1, 0) - armor_y_offset;
            center(2, 0) = observation(2, 0) - radius * std::sin(radial_angle);
            return center;
        }
}
}

}
#endif // AUTO_AIM_TRACKER_UTILS_HPP
