#ifndef AUTO_AIM_APP_OVERLAY_HPP
#define AUTO_AIM_APP_OVERLAY_HPP

// 组合层：瞄准结果叠画（整车轮廓 + 观测板高亮 + 提前点）。
// 原本内联在 node_sim.cpp 的显示分支里；抽出来便于两个入口共用。

#include <string>

#include <opencv2/opencv.hpp>

#include "Kalman/tracker.hpp"
#include "control/gimbal_controller.hpp"
#include "control/shooter.hpp"
#include "control/target_selector.hpp"
#include "visualization/projection.hpp"

namespace auto_aim
{
namespace app
{
        inline void drawAimOverlay(cv::Mat& image, const Tracker& tracker,
                           const TargetDecision& last_decision,
                           const GimbalControllerSnapshot& last_gimbal_command,
                           const std::string& shooter_status,
                           const cv::Mat& camera_matrix, const cv::Mat& dist_coeffs)
{
        if (
            tracker.getState() != auto_aim::TrackerState::LOST &&
            tracker.getTargetCenter().rows == 3) {
            const bool has_observation = tracker.hasObservation();
            const int observed_plate = has_observation
                ? tracker.getLastObservedPlateId() : -1;

            const bool large_armor = has_observation &&
                tracker.getLastObservedArmor().armor_type == auto_aim::large;
            // 模型口径与 real_runner 一致：输出状态（双模型权重生效后）而非纯整车 EKF。
            // 这里没有"当前帧时刻/当前姿态"，所以 dt=0、用 tracker 内部存的姿态；
            // 真机/仿真入口请用 modelAt(dt, 当前帧姿态)（real_runner / SimRunner）。
            const auto model = tracker.modelAt(0.0);

            for (int plate = 0; plate < 4; ++plate) {
                if (!model.valid) continue;
                const auto_aim::ArmorPose& pose = model.poses[plate];
                const cv::Mat center_mat = (cv::Mat_<double>(3, 1) << pose.center[0],
                                            pose.center[1], pose.center[2]);
                const auto corners = auto_aim::projectArmorCameraFrame(
                    center_mat, pose.axes.tangent, pose.axes.bar,
                    tracker.armorHalfWidth(plate), tracker.armorHalfHeight(),
                    camera_matrix, dist_coeffs);
                if (corners.size() != 4) continue;
                if (plate != observed_plate) {
                    auto_aim::drawQuad(image, corners, cv::Scalar(255, 220, 0), 1);
                    cv::putText(image, "E" + std::to_string(plate),
                                corners[0] + cv::Point2f(2.0f, -3.0f),
                                cv::FONT_HERSHEY_SIMPLEX, 0.4,
                                cv::Scalar(255, 220, 0), 1, cv::LINE_AA);
                }
            }

            if (has_observation) {
                const auto& observed = tracker.getLastObservedArmor();
                const std::vector<cv::Point2f> corners = {
                    observed.left.top, observed.right.top,
                    observed.right.bottom, observed.left.bottom
                };
                auto_aim::drawQuad(image, corners, cv::Scalar(0, 255, 0), 2);
                cv::putText(image, "DETECTED P" + std::to_string(observed_plate),
                            observed.left.top + cv::Point2f(-8.0f, -8.0f),
                            cv::FONT_HERSHEY_SIMPLEX, 0.45,
                            cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
            }

            cv::Mat center = tracker.getTargetCenter();
            if (model.valid) {
                center = (cv::Mat_<double>(3, 1) << model.center_camera[0],
                          model.center_camera[1], model.center_camera[2]);
            }
            cv::Point2f chassis_center;
            if (auto_aim::projectPoint(center, camera_matrix, dist_coeffs,
                                       chassis_center)) {
                cv::circle(image, chassis_center, 4, cv::Scalar(0, 0, 255), -1);
            }
            cv::putText(
                image,
                cv::format("chassis omega=%+.2f rad/s", tracker.getOmega()),
                cv::Point(8, 22), cv::FONT_HERSHEY_SIMPLEX, 0.55,
                cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
            if (last_gimbal_command.valid) {
                cv::putText(
                    image,
                    cv::format("gimbal yaw=%+.3f pitch=%+.3f rad",
                               last_gimbal_command.yaw, last_gimbal_command.pitch),
                    cv::Point(8, 46), cv::FONT_HERSHEY_SIMPLEX, 0.48,
                    cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
            }
            cv::putText(
                image,
                cv::format("shooter %s",
                           shooter_status.c_str()),
                cv::Point(8, 68), cv::FONT_HERSHEY_SIMPLEX, 0.48,
                cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
            if (last_decision.valid) {
                const auto selected_center_camera = tracker.worldToCameraPoint(
                    last_decision.armor_position);
                const cv::Mat selected_center = (cv::Mat_<double>(3, 1) <<
                    selected_center_camera[0],
                    selected_center_camera[1],
                    selected_center_camera[2]);
                const double selected_half_width =
                    tracker.armorHalfWidth(last_decision.armor_id);
                // 提前点那块板用的是**决策里的预测 yaw**（打到的那一刻的相位），
                // 所以姿态由预测相位重建，而不是读当前状态。
                const double predicted_plate_angle = last_decision.predicted_yaw +
                    ArmorEKF::angleForPlate(last_decision.armor_id);
                const auto_aim::ArmorAxes selected_axes = auto_aim::armorAxesFromRadial(
                    tracker.worldDirectionToCamera(cv::Vec3d(
                        std::cos(predicted_plate_angle), 0.0, std::sin(predicted_plate_angle))),
                    tracker.worldDirectionToCamera(cv::Vec3d(0.0, -1.0, 0.0)),
                    tracker.getArmorTiltDeg() * CV_PI / 180.0);
                const auto selected_corners = auto_aim::projectArmorCameraFrame(
                    selected_center, selected_axes.tangent, selected_axes.bar,
                    selected_half_width, tracker.armorHalfHeight(),
                    camera_matrix, dist_coeffs);
                auto_aim::drawQuad(image, selected_corners,
                                   cv::Scalar(0, 165, 255), 2);
                if (!selected_corners.empty()) {
                    cv::putText(
                        image,
                        "SELECT P" + std::to_string(last_decision.armor_id),
                        selected_corners[0] + cv::Point2f(2.0f, -5.0f),
                        cv::FONT_HERSHEY_SIMPLEX, 0.45,
                        cv::Scalar(0, 165, 255), 1, cv::LINE_AA);
                }

                const auto aim_point_camera = tracker.worldToCameraPoint(
                    last_decision.aim_point);
                const cv::Mat lead_point = (cv::Mat_<double>(3, 1) <<
                    aim_point_camera[0],
                    aim_point_camera[1],
                    aim_point_camera[2]);
                cv::Point2f lead_image;
                if (auto_aim::projectPoint(lead_point, camera_matrix, dist_coeffs,
                                           lead_image)) {
                    cv::drawMarker(image, lead_image, cv::Scalar(255, 0, 255),
                                   cv::MARKER_CROSS, 18, 2, cv::LINE_AA);
                    cv::putText(
                        image,
                        cv::format("AIM P%d t=%.3fs d=%.2fm",
                                   last_decision.armor_id,
                                   last_decision.lead_time,
                                   last_decision.distance),
                        lead_image + cv::Point2f(8.0f, -8.0f),
                        cv::FONT_HERSHEY_SIMPLEX, 0.45,
                        cv::Scalar(255, 0, 255), 1, cv::LINE_AA);
                }
            }
        } else {
            cv::putText(image, "No chassis estimate", cv::Point(8, 22),
                        cv::FONT_HERSHEY_SIMPLEX, 0.55,
                        cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
        }
}
}
}

#endif // AUTO_AIM_APP_OVERLAY_HPP
