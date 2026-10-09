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
            const auto estimated_positions = tracker.getEstimatedArmorPositions();
            const double yaw = tracker.getYaw();
            const bool has_observation = tracker.hasObservation();
            const int observed_plate = has_observation
                ? tracker.getLastObservedPlateId() : -1;

            const bool large_armor = has_observation &&
                tracker.getLastObservedArmor().armor_type == auto_aim::large;
            const double half_width = large_armor ? 0.1125 : 0.0675;
            const double half_height = 0.0275;
            cv::Vec3d vertical_axis(0.0, 1.0, 0.0);
            if (has_observation && !tracker.getLastObservedArmor().rvec.empty()) {
                cv::Mat rotation;
                cv::Rodrigues(tracker.getLastObservedArmor().rvec, rotation);
                const cv::Mat local_vertical = (cv::Mat_<double>(3, 1) << 0.0, 0.0, 1.0);
                const cv::Mat camera_vertical = rotation * local_vertical;
                cv::Vec3d candidate(
                    camera_vertical.at<double>(0),
                    camera_vertical.at<double>(1),
                    camera_vertical.at<double>(2));
                const double norm = cv::norm(candidate);
                if (norm > 0.5 && std::abs(candidate[1]) > 0.7) {
                    vertical_axis = candidate / norm;
                }
            }

            for (int plate = 0; plate < 4; ++plate) {
                const auto corners = auto_aim::projectArmor(
                    estimated_positions[plate], plate, yaw, vertical_axis,
                    half_width, half_height,
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

            const auto center = tracker.getTargetCenter();
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
                    last_decision.armor_id == observed_plate && large_armor
                    ? 0.1125 : 0.0675;
                const auto selected_corners = auto_aim::projectArmor(
                    selected_center, last_decision.armor_id,
                    tracker.worldToCameraYaw(last_decision.predicted_yaw), vertical_axis,
                    selected_half_width, half_height,
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
