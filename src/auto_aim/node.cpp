#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>

#include "common/types.hpp"
#include "common/standard_clock.hpp"
#include "config_loader.hpp"
#include "visualization/projection.hpp"
#include "perception/detector.hpp"
#include "perception/pnp_solver.hpp"
#include "Kalman/tracker.hpp"
#include "io/camera/GalaxyCamera.hpp"

#ifndef ULTRA_VISION_CONFIG_DIR
#define ULTRA_VISION_CONFIG_DIR "configs"
#endif

namespace
{
    const char* trackerStateName(auto_aim::TrackerState state)
    {
        switch (state) {
        case auto_aim::TrackerState::DETECTING: return "DETECTING";
        case auto_aim::TrackerState::TRACKING: return "TRACKING";
        case auto_aim::TrackerState::TEMP_LOST: return "TEMP_LOST";
        case auto_aim::TrackerState::LOST: return "LOST";
        }
        return "UNKNOWN";
    }

    // The estimated plates are drawn as flat quads. Their vertical axis comes
    // from the most recent PnP pose so the overlay follows the target's tilt
    // instead of assuming a perfectly level camera.
    cv::Vec3d verticalAxisFromObservation(const auto_aim::Armor& armor)
    {
        const cv::Vec3d fallback(0.0, 1.0, 0.0);
        if (armor.rvec.empty()) return fallback;

        cv::Mat rotation;
        cv::Rodrigues(armor.rvec, rotation);
        const cv::Mat local_vertical = (cv::Mat_<double>(3, 1) << 0.0, 0.0, 1.0);
        const cv::Mat camera_vertical = rotation * local_vertical;
        const cv::Vec3d candidate(
            camera_vertical.at<double>(0),
            camera_vertical.at<double>(1),
            camera_vertical.at<double>(2));
        const double norm = cv::norm(candidate);
        if (norm > 0.5 && std::abs(candidate[1]) > 0.7) {
            return candidate / norm;
        }
        return fallback;
    }
} // namespace

int main(int argc, char* argv[])
{
    std::string config_dir = ULTRA_VISION_CONFIG_DIR;
    if (argc > 1) config_dir = argv[1];
    if (const char* env = std::getenv("ULTRA_VISION_CONFIG_DIR")) config_dir = env;

    const auto config_path = [&](const std::string& name) {
        return config_dir + "/" + name;
    };

    // ------------------- 1. Load configuration -------------------
    const YAML::Node camera_file = YAML::LoadFile(config_path("camera.yaml"));
    const YAML::Node detector_file = YAML::LoadFile(config_path("detector.yaml"));
    const YAML::Node tracker_file = YAML::LoadFile(config_path("tracker.yaml"));

    const YAML::Node camera_node = camera_file["camera"];
    const std::string camera_name = camera_node["name"].as<std::string>("galaxy");
    const int device_index = camera_node["device_index"].as<int>(1);
    const std::string serial_number = camera_node["serial_number"].as<std::string>("");
    const int camera_timeout = camera_node["timeout_ms"].as<int>(1000);

    const YAML::Node intrinsic_node =
        (camera_name == "hikcamera") ? camera_file["hikcamera"] : camera_file["galaxy"];
    const cv::Mat camera_matrix =
        auto_aim::readMatFromYaml(intrinsic_node["camera_matrix"]);
    const cv::Mat dist_coeffs =
        auto_aim::readMatFromYaml(intrinsic_node["dist_coeffs"]);

    const auto_aim::DetectorConfig det_cfg =
        auto_aim::loadDetectorConfig(detector_file);
    const auto_aim::TrackerConfig track_cfg =
        auto_aim::loadTrackerConfig(tracker_file);
    const auto_aim::PnpGeometry pnp_geometry =
        auto_aim::loadPnpGeometry(tracker_file);

    // ------------------- 2. Initialize the Galaxy camera -------------------
    rm_ultra::GalaxyCamera camera;
    const bool camera_ready = serial_number.empty()
        ? camera.init("", device_index)
        : camera.init(serial_number, device_index);
    if (!camera_ready) {
        std::cerr << "Failed to init Galaxy camera!" << std::endl;
        return -1;
    }
    std::cout << "Galaxy camera initialized successfully." << std::endl;

    auto_aim::CAMERA_MATRIX = camera_matrix;
    auto_aim::DIST_COEFFS = dist_coeffs;

    // ------------------- 3. Whole-chassis estimator -------------------
    // One tracker for the whole vehicle, matching the simulator entry. Until
    // the gimbal link reports absolute angles, CameraPose keeps its identity
    // default, so the estimate is expressed in the camera frame. Calling
    // tracker.setCameraPose() before update() is the only change needed to run
    // the same estimator in the gimbal/world frame once that feedback exists.
    auto_aim::Tracker tracker(track_cfg);
    auto_aim::Detector detector(cv::Mat(), det_cfg);

    cv::Mat frame;
    cv::Mat image;
    cv::Mat gray;
    cv::Mat binary;
    cv::Mat dilated;
    std::vector<std::vector<cv::Point>> contours;
    std::vector<auto_aim::Light> lights;
    std::vector<auto_aim::Armor> armors;

    int processed_frames = 0;
    auto report_time = std::chrono::steady_clock::now();
    int key = -1;

    while (key != 27 && key != 'e' && key != 'E') {
        // ------------------- 4. Grab a frame -------------------
        if (!camera.getImage(frame, camera_timeout)) {
            std::cerr << "Failed to get image from camera" << std::endl;
            break;
        }
        if (frame.empty()) continue;

        // The frame is processed at its native resolution. Downscaling here
        // would require scaling the intrinsics by the same factor, otherwise
        // every PnP distance is wrong.
        image = frame;

        // ------------------- 5. Detection and PnP -------------------
        detector.gray_img(image, gray);
        detector.binary_img(gray, binary, det_cfg.binary_threshold);
        detector.open_close_img(binary, dilated);

        contours.clear();
        detector.find_contours(dilated, contours);
        lights.clear();
        detector.find_lights(contours, lights, image, det_cfg.enemy_color);
        armors.clear();
        if (!lights.empty()) {
            detector.find_armors(lights, armors);
        }

        int pnp_count = 0;
        for (auto& armor : armors) {
            armor.solve_result = auto_aim::solveArmorPnP(
                armor, auto_aim::CAMERA_MATRIX, auto_aim::DIST_COEFFS,
                pnp_geometry.small_width, pnp_geometry.large_width,
                pnp_geometry.height);
            if (armor.solve_result) ++pnp_count;
        }

        // ------------------- 6. Whole-chassis update -------------------
        // All detections of one frame are associated and fused in a single
        // batch update, so several plates never update the shared chassis
        // center more than once.
        const double timestamp = auto_aim::StandardClock::nowSeconds();
        tracker.update(armors, timestamp);
        ++processed_frames;

        // ------------------- 7. Visualization -------------------
        for (const auto& light : lights) {
            cv::circle(image, light.top, 2, cv::Scalar(0, 255, 255), 1);
            cv::circle(image, light.bottom, 2, cv::Scalar(0, 255, 255), 1);
            cv::line(image, light.top, light.bottom, cv::Scalar(0, 0, 255), 1);
        }
        for (const auto& armor : armors) {
            cv::line(image, armor.left.top, armor.right.bottom,
                     cv::Scalar(0, 255, 2), 1);
            cv::line(image, armor.left.bottom, armor.right.top,
                     cv::Scalar(0, 255, 2), 1);
        }

        const bool has_observation = tracker.hasObservation();
        const int observed_plate =
            has_observation ? tracker.getLastObservedPlateId() : -1;
        const auto_aim::Armor& observed_armor = tracker.getLastObservedArmor();
        const bool large_armor =
            has_observation && observed_armor.armor_type == auto_aim::large;
        const double half_width =
            (large_armor ? pnp_geometry.large_width : pnp_geometry.small_width) * 0.5;
        const double half_height = pnp_geometry.height * 0.5;
        const cv::Vec3d vertical_axis = has_observation
            ? verticalAxisFromObservation(observed_armor)
            : cv::Vec3d(0.0, 1.0, 0.0);

        if (tracker.getState() != auto_aim::TrackerState::LOST) {
            const auto estimated_positions = tracker.getEstimatedArmorPositions();
            const double yaw = tracker.getYaw();
            for (int plate = 0; plate < 4; ++plate) {
                const auto corners = auto_aim::projectArmor(
                    estimated_positions[plate], plate, yaw, vertical_axis,
                    half_width, half_height,
                    auto_aim::CAMERA_MATRIX, auto_aim::DIST_COEFFS);
                if (corners.size() != 4 || plate == observed_plate) continue;
                auto_aim::drawQuad(image, corners, cv::Scalar(255, 220, 0), 1);
                cv::putText(image, "E" + std::to_string(plate),
                            corners[0] + cv::Point2f(2.0f, -3.0f),
                            cv::FONT_HERSHEY_SIMPLEX, 0.4,
                            cv::Scalar(255, 220, 0), 1, cv::LINE_AA);
            }

            cv::Point2f chassis_center;
            if (auto_aim::projectPoint(tracker.getTargetCenter(),
                                       auto_aim::CAMERA_MATRIX,
                                       auto_aim::DIST_COEFFS, chassis_center)) {
                cv::circle(image, chassis_center, 4, cv::Scalar(0, 0, 255), -1);
            }
        }

        if (has_observation) {
            const std::vector<cv::Point2f> corners = {
                observed_armor.left.top, observed_armor.right.top,
                observed_armor.right.bottom, observed_armor.left.bottom
            };
            auto_aim::drawQuad(image, corners, cv::Scalar(0, 255, 0), 2);
            cv::putText(image, "DETECTED P" + std::to_string(observed_plate),
                        observed_armor.left.top + cv::Point2f(-8.0f, -8.0f),
                        cv::FONT_HERSHEY_SIMPLEX, 0.45,
                        cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
        }

        cv::putText(image,
                    cv::format("state=%s armors=%d pnp=%d",
                               trackerStateName(tracker.getState()),
                               static_cast<int>(armors.size()), pnp_count),
                    cv::Point(8, 22), cv::FONT_HERSHEY_SIMPLEX, 0.55,
                    cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
        if (tracker.getState() != auto_aim::TrackerState::LOST) {
            cv::putText(image,
                        cv::format("yaw=%+.3f rad omega=%+.2f rad/s lost=%d",
                                   tracker.getYaw(), tracker.getOmega(),
                                   tracker.getLostCount()),
                        cv::Point(8, 46), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                        cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
        }

        cv::imshow("Detection & Tracking", image);
        key = cv::waitKey(1);

        // ------------------- 8. Status -------------------
        const auto now = std::chrono::steady_clock::now();
        const double elapsed = std::chrono::duration<double>(now - report_time).count();
        if (elapsed >= 1.0) {
            std::cout << "Processed FPS: " << processed_frames / elapsed
                      << ", lights: " << lights.size()
                      << ", armors: " << armors.size()
                      << ", PnP: " << pnp_count
                      << ", tracker: " << trackerStateName(tracker.getState())
                      << ", fresh: " << tracker.hasObservation()
                      << ", lost: " << tracker.getLostCount()
                      << ", omega: " << tracker.getOmega() << std::endl;
            report_time = now;
            processed_frames = 0;
        }
    }

    return 0;
}
