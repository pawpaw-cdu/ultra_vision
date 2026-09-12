#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>

#include <sim_receiver/vision_date_receiver.hpp>
#include "common/types.hpp"
#include "perception/detector.hpp"
#include "perception/pnp_solver.hpp"
#include "Kalman/tracker.hpp"

#ifndef ULTRA_VISION_CONFIG_DIR
#define ULTRA_VISION_CONFIG_DIR "configs"
#endif

namespace {

cv::Mat readMatFromYaml(const YAML::Node& node)
{
    const int rows = node["rows"].as<int>();
    const int cols = node["cols"].as<int>();
    const std::vector<double> data = node["data"].as<std::vector<double>>();
    cv::Mat matrix(rows, cols, CV_64F);
    for (int row = 0; row < rows; ++row) {
        for (int column = 0; column < cols; ++column) {
            matrix.at<double>(row, column) = data[row * cols + column];
        }
    }
    return matrix;
}

cv::Mat makeCameraMatrix(int rows, int columns, double vertical_fov_degrees)
{
    const double focal_length = 0.5 * rows /
        std::tan(vertical_fov_degrees * CV_PI / 360.0);
    return (cv::Mat_<double>(3, 3) <<
        focal_length, 0.0, 0.5 * columns,
        0.0, focal_length, 0.5 * rows,
        0.0, 0.0, 1.0);
}

double steadyTimestamp()
{
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

std::vector<cv::Point2f> projectArmor(
    const cv::Mat& center,
    int plate_id,
    double yaw,
    const cv::Vec3d& vertical_axis,
    double half_width,
    double half_height,
    const cv::Mat& camera_matrix,
    const cv::Mat& dist_coeffs)
{
    const double total_angle = yaw + auto_aim::ArmorEKF::angleForPlate(plate_id);
    const double tangent_x = -std::sin(total_angle);
    const double tangent_z = std::cos(total_angle);
    const cv::Vec3d tangent(tangent_x, 0.0, tangent_z);
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
    cv::projectPoints(
        object_points,
        cv::Mat::zeros(3, 1, CV_64F),
        cv::Mat::zeros(3, 1, CV_64F),
        camera_matrix,
        dist_coeffs,
        image_points);
    return image_points;
}

bool projectPoint(
    const cv::Mat& point,
    const cv::Mat& camera_matrix,
    const cv::Mat& dist_coeffs,
    cv::Point2f& image_point)
{
    if (point.rows != 3 || point.cols != 1) return false;
    std::vector<cv::Point3f> object_points = {
        cv::Point3f(
            static_cast<float>(point.at<double>(0)),
            static_cast<float>(point.at<double>(1)),
            static_cast<float>(point.at<double>(2)))
    };
    std::vector<cv::Point2f> image_points;
    cv::projectPoints(
        object_points, cv::Mat::zeros(3, 1, CV_64F), cv::Mat::zeros(3, 1, CV_64F),
        camera_matrix, dist_coeffs, image_points);
    if (image_points.empty()) return false;
    image_point = image_points.front();
    return true;
}

void drawQuad(cv::Mat& image, const std::vector<cv::Point2f>& points,
              const cv::Scalar& color, int thickness)
{
    if (points.size() != 4) return;
    for (std::size_t i = 0; i < points.size(); ++i) {
        cv::line(image, points[i], points[(i + 1) % points.size()], color, thickness, cv::LINE_AA);
    }
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

    const YAML::Node camera_cfg = YAML::LoadFile(config_path("camera.yaml"));
    const YAML::Node detector_cfg = YAML::LoadFile(config_path("detector.yaml"));
    const YAML::Node tracker_cfg = YAML::LoadFile(config_path("tracker.yaml"));

    bool use_simulator_camera = false;
    double simulator_fov_degrees = 45.0;
    int max_processing_width = 960;
    if (camera_cfg["simulator"]) {
        use_simulator_camera = true;
        simulator_fov_degrees = camera_cfg["simulator"]["fov_degrees"].as<double>(45.0);
        max_processing_width = camera_cfg["simulator"]["max_processing_width"].as<int>(960);
    }

    auto_aim::DetectorConfig det_cfg;
    det_cfg.enemy_color = detector_cfg["detector"]["enemy_color"].as<int>();
    det_cfg.binary_threshold = detector_cfg["detector"]["binary_threshold"].as<int>();
    const auto light = detector_cfg["detector"]["light"];
    det_cfg.light_min_ratio = light["min_ratio"].as<double>();
    det_cfg.light_max_ratio = light["max_ratio"].as<double>();
    det_cfg.light_max_angle = light["max_angle"].as<double>();
    det_cfg.light_min_contour_points = light["min_contour_points"].as<int>();
    const auto armor = detector_cfg["detector"]["armor"];
    det_cfg.armor_height_ratio_min = armor["height_ratio_min"].as<double>();
    det_cfg.armor_height_ratio_max = armor["height_ratio_max"].as<double>();
    det_cfg.armor_angle_diff_max = armor["angle_diff_max"].as<double>();
    det_cfg.armor_width_to_height_min = armor["width_to_height_min"].as<double>();
    det_cfg.armor_width_to_height_max = armor["width_to_height_max"].as<double>();
    det_cfg.armor_large_ratio_thresh = armor["large_armor_ratio_thresh"].as<double>();

    const auto color = detector_cfg["detector"]["color"];
    det_cfg.color_use_detect = color["use_color_detect"].as<bool>();
    det_cfg.use_hsv = color["method"].as<std::string>() == "hsv";
    if (det_cfg.use_hsv) {
        const auto hsv = color["hsv"];
        det_cfg.hue_red_low1 = hsv["red_low1"].as<int>();
        det_cfg.hue_red_high1 = hsv["red_high1"].as<int>();
        det_cfg.hue_red_low2 = hsv["red_low2"].as<int>();
        det_cfg.hue_red_high2 = hsv["red_high2"].as<int>();
        det_cfg.hue_blue_low = hsv["blue_low"].as<int>();
        det_cfg.hue_blue_high = hsv["blue_high"].as<int>();
        det_cfg.sat_min = hsv["sat_min"].as<int>();
        det_cfg.val_min = hsv["val_min"].as<int>();
        det_cfg.color_area_ratio = hsv["area_ratio"].as<double>();
    } else {
        const auto bgr = color["bgr"];
        det_cfg.color_red_threshold = bgr["red_threshold"].as<double>();
        det_cfg.color_blue_threshold = bgr["blue_threshold"].as<double>();
    }

    auto_aim::TrackerConfig track_cfg;
    track_cfg.min_detect_frames = tracker_cfg["tracker"]["min_detect_frames"].as<int>();
    track_cfg.max_lost_frames = tracker_cfg["tracker"]["max_lost_frames"].as<int>();
    track_cfg.max_match_distance = tracker_cfg["tracker"]["max_match_distance"].as<double>();
    const auto kalman = tracker_cfg["tracker"]["kalman"];
    track_cfg.process_noise_pos = kalman["process_noise_pos"].as<double>();
    track_cfg.process_noise_vel = kalman["process_noise_vel"].as<double>();
    track_cfg.measure_noise = kalman["measure_noise"].as<double>();
    track_cfg.default_dt = kalman["default_dt"].as<double>();
    track_cfg.armor_radius = kalman["armor_radius"].as<double>(0.21);
    track_cfg.y_noise_mult = kalman["y_noise_mult"].as<double>(30.0);
    track_cfg.p_cov_min = kalman["p_cov_min"].as<double>(1e-3);
    track_cfg.angle_measure_noise = kalman["angle_measure_noise"].as<double>(0.04);
    track_cfg.max_angle_error = kalman["max_angle_error"].as<double>(0.70);
    track_cfg.angle_match_weight = kalman["angle_match_weight"].as<double>(0.08);
    track_cfg.position_nis_threshold = kalman["position_nis_threshold"].as<double>(7.815);
    track_cfg.angle_nis_threshold = kalman["angle_nis_threshold"].as<double>(3.841);

    sim_receiver::VisionDateReceiver receiver(config_path("simulator.yaml"));
    receiver.connect();

    auto_aim::Tracker tracker(track_cfg);
    auto_aim::Detector detector(cv::Mat(), det_cfg);

    int processing_height = 0;
    cv::Mat camera_matrix;
    cv::Mat dist_coeffs = cv::Mat::zeros(5, 1, CV_64F);
    cv::Mat frame;
    cv::Mat image;
    cv::Mat gray;
    cv::Mat binary;
    std::vector<std::vector<cv::Point>> contours;
    std::vector<auto_aim::Light> lights;
    std::vector<auto_aim::Armor> armors;

    double last_fire_time = 0.0;
    auto fps_start = std::chrono::steady_clock::now();
    int fps_frames = 0;
    int missing_frames = 0;
    int key = -1;

    while (key != 27 && key != 'e' && key != 'E') {
        frame = receiver.getFrame();
        if (frame.empty()) {
            ++missing_frames;
            if (missing_frames == 1 || missing_frames % 30 == 0) {
                std::cerr << "Waiting for simulator frames..." << std::endl;
            }
            key = cv::waitKey(1);
            continue;
        }
        missing_frames = 0;

        if (max_processing_width > 0 && frame.cols > max_processing_width) {
            const double scale = static_cast<double>(max_processing_width) / frame.cols;
            cv::resize(frame, image, cv::Size(), scale, scale, cv::INTER_AREA);
        } else {
            image = frame;
        }

        if (use_simulator_camera && image.rows != processing_height) {
            camera_matrix = makeCameraMatrix(image.rows, image.cols, simulator_fov_degrees);
            processing_height = image.rows;
        }

        detector.gray_img(image, gray);
        cv::GaussianBlur(gray, gray, cv::Size(3, 3), 0);
        detector.binary_img(gray, binary, det_cfg.binary_threshold);

        contours.clear();
        detector.find_contours(binary, contours);
        lights.clear();
        detector.find_lights(contours, lights, image, det_cfg.enemy_color);
        armors.clear();
        if (!lights.empty()) {
            detector.find_armors(lights, armors);
        }

        int pnp_count = 0;
        for (auto& detected_armor : armors) {
            detected_armor.solve_result = auto_aim::solveArmorPnP(
                detected_armor, camera_matrix, dist_coeffs);
            if (detected_armor.solve_result) ++pnp_count;
        }

        const double timestamp = steadyTimestamp();
        tracker.update(armors, timestamp);

        if (tracker.getState() == auto_aim::TrackerState::TRACKING &&
            timestamp - last_fire_time >= 0.1) {
            receiver.sendFireCommand();
            last_fire_time = timestamp;
        }

        if (tracker.getState() != auto_aim::TrackerState::LOST && tracker.getTargetCenter().rows == 3) {
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
                const auto corners = projectArmor(
                    estimated_positions[plate], plate, yaw, vertical_axis,
                    half_width, half_height,
                    camera_matrix, dist_coeffs);
                if (corners.size() != 4) continue;
                if (plate != observed_plate) {
                    drawQuad(image, corners, cv::Scalar(255, 220, 0), 1);
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
                drawQuad(image, corners, cv::Scalar(0, 255, 0), 2);
                cv::putText(image, "DETECTED P" + std::to_string(observed_plate),
                            observed.left.top + cv::Point2f(-8.0f, -8.0f),
                            cv::FONT_HERSHEY_SIMPLEX, 0.45,
                            cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
            }

            const auto center = tracker.getTargetCenter();
            cv::Point2f chassis_center;
            if (projectPoint(center, camera_matrix, dist_coeffs, chassis_center)) {
                cv::circle(image, chassis_center, 4, cv::Scalar(0, 0, 255), -1);
            }
            cv::putText(
                image,
                cv::format("chassis omega=%+.2f rad/s", tracker.getOmega()),
                cv::Point(8, 22), cv::FONT_HERSHEY_SIMPLEX, 0.55,
                cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
        } else {
            cv::putText(image, "No chassis estimate", cv::Point(8, 22),
                        cv::FONT_HERSHEY_SIMPLEX, 0.55,
                        cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
        }

        ++fps_frames;
        const auto now = std::chrono::steady_clock::now();
        const double elapsed = std::chrono::duration<double>(now - fps_start).count();
        if (elapsed >= 1.0) {
            std::cout << "Processed FPS: " << fps_frames / elapsed
                      << ", armors: " << armors.size()
                      << ", PnP: " << pnp_count << std::endl;
            fps_start = now;
            fps_frames = 0;
        }

        cv::resize(image, image, cv::Size(640, 480), 0.0, 0.0, cv::INTER_AREA);
        cv::imshow("Detection & Tracking", image);
        key = cv::waitKey(1);
    }

    return 0;
}
