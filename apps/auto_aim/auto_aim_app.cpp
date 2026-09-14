#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>

#include "ultra_vision/io/vision_date_recevier.hpp"
#include "ultra_vision/core/types.hpp"
#include "ultra_vision/core/standard_clock.hpp"
#include "ultra_vision/core/logger.hpp"
#include "ultra_vision/control/aim_signal_filter.hpp"
#include "ultra_vision/control/gimbal_aimer.hpp"
#include "ultra_vision/control/gimbal_controller.hpp"
#include "ultra_vision/control/shooter.hpp"
#include "ultra_vision/selection/target_selector.hpp"
#include "ultra_vision/perception/detector.hpp"
#include "ultra_vision/perception/pnp_solver.hpp"
#include "ultra_vision/estimation/tracker.hpp"

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
    const double total_angle = yaw + ultra_vision::ArmorEKF::angleForPlate(plate_id);
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

const char* shooterStateName(ultra_vision::ShooterState state)
{
    switch (state) {
    case ultra_vision::ShooterState::IDLE: return "IDLE";
    case ultra_vision::ShooterState::READY: return "READY";
    case ultra_vision::ShooterState::FIRING: return "FIRING";
    case ultra_vision::ShooterState::ERROR: return "ERROR";
    case ultra_vision::ShooterState::END: return "END";
    }
    return "UNKNOWN";
}

const char* shooterErrorName(ultra_vision::ShooterErrorReason reason)
{
    switch (reason) {
    case ultra_vision::ShooterErrorReason::NONE: return "NONE";
    case ultra_vision::ShooterErrorReason::OUT_OF_FIRE_WINDOW: return "OUT_OF_FIRE_WINDOW";
    case ultra_vision::ShooterErrorReason::GIMBAL_ERROR: return "GIMBAL_ERROR";
    }
    return "UNKNOWN";
}

std::string shooterStatusName(ultra_vision::ShooterState state,
                              ultra_vision::ShooterErrorReason reason)
{
    std::string status = shooterStateName(state);
    if (state == ultra_vision::ShooterState::ERROR) {
        status += "(" + std::string(shooterErrorName(reason)) + ")";
    }
    return status;
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

namespace ultra_vision::app
{
int runAutoAim(int argc, char* argv[], bool debug_enabled)
{
    const char* log_path = std::getenv("ULTRA_VISION_LOG");
    ultra_vision::Logger::configure(
        debug_enabled, log_path ? log_path : std::string());

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

    ultra_vision::DetectorConfig det_cfg;
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

    ultra_vision::TrackerConfig track_cfg;
    track_cfg.min_detect_frames = tracker_cfg["tracker"]["min_detect_frames"].as<int>();
    track_cfg.max_lost_frames = tracker_cfg["tracker"]["max_lost_frames"].as<int>();
    track_cfg.reacquire_frames =
        tracker_cfg["tracker"]["reacquire_frames"].as<int>(track_cfg.reacquire_frames);
    track_cfg.max_match_distance = tracker_cfg["tracker"]["max_match_distance"].as<double>();
    track_cfg.reacquire_center_error =
        tracker_cfg["tracker"]["reacquire_center_error"].as<double>(
            track_cfg.reacquire_center_error);
    track_cfg.reacquire_max_dt =
        tracker_cfg["tracker"]["reacquire_max_dt"].as<double>(track_cfg.reacquire_max_dt);
    track_cfg.max_camera_pose_dt =
        tracker_cfg["tracker"]["max_camera_pose_dt"].as<double>(
            track_cfg.max_camera_pose_dt);
    if (tracker_cfg["tracker"]["rotation_rate"]) {
        const auto rotation = tracker_cfg["tracker"]["rotation_rate"];
        track_cfg.rotation_rate.window =
            rotation["window"].as<double>(track_cfg.rotation_rate.window);
        track_cfg.rotation_rate.min_span =
            rotation["min_span"].as<double>(track_cfg.rotation_rate.min_span);
        track_cfg.rotation_rate.max_sample_gap =
            rotation["max_sample_gap"].as<double>(
                track_cfg.rotation_rate.max_sample_gap);
        track_cfg.rotation_rate.max_rate =
            rotation["max_rate"].as<double>(track_cfg.rotation_rate.max_rate);
        track_cfg.rotation_rate.min_rate =
            rotation["min_rate"].as<double>(track_cfg.rotation_rate.min_rate);
        track_cfg.rotation_rate.max_residual =
            rotation["max_residual"].as<double>(
                track_cfg.rotation_rate.max_residual);
        track_cfg.rotation_rate.min_slopes =
            rotation["min_slopes"].as<int>(track_cfg.rotation_rate.min_slopes);
    }
    const auto kalman = tracker_cfg["tracker"]["kalman"];
    track_cfg.process_noise_pos = kalman["process_noise_pos"].as<double>();
    track_cfg.process_noise_vel = kalman["process_noise_vel"].as<double>();
    track_cfg.measure_noise = kalman["measure_noise"].as<double>();
    track_cfg.default_dt = kalman["default_dt"].as<double>();
    track_cfg.armor_radius = kalman["armor_radius"].as<double>(0.21);
    track_cfg.y_noise_mult = kalman["y_noise_mult"].as<double>(30.0);
    track_cfg.p_cov_min = kalman["p_cov_min"].as<double>(1e-3);
    track_cfg.angle_measure_noise = kalman["angle_measure_noise"].as<double>(0.04);
    track_cfg.bearing_measure_noise = kalman["bearing_measure_noise"].as<double>(0.0025);
    track_cfg.pitch_measure_noise = kalman["pitch_measure_noise"].as<double>(0.005);
    track_cfg.range_noise_distance_scale =
        kalman["range_noise_distance_scale"].as<double>(0.1);
    track_cfg.ypd_nis_threshold = kalman["ypd_nis_threshold"].as<double>(9.488);
    track_cfg.max_angle_error = kalman["max_angle_error"].as<double>(0.70);
    track_cfg.angle_match_weight = kalman["angle_match_weight"].as<double>(0.08);
    track_cfg.position_nis_threshold = kalman["position_nis_threshold"].as<double>(7.815);
    track_cfg.angle_nis_threshold = kalman["angle_nis_threshold"].as<double>(3.841);
    if (kalman["armor_y_offsets"]) {
        const auto offsets = kalman["armor_y_offsets"].as<std::vector<double>>();
        if (offsets.size() == track_cfg.armor_y_offsets.size()) {
            std::copy(offsets.begin(), offsets.end(), track_cfg.armor_y_offsets.begin());
        } else {
            std::cerr << "Ignoring armor_y_offsets: expected four values" << std::endl;
        }
    }

    float pnp_small_width = 0.135f;
    float pnp_large_width = 0.225f;
    float pnp_armor_height = 0.055f;
    if (tracker_cfg["tracker"]["pnp"]) {
        const auto pnp = tracker_cfg["tracker"]["pnp"];
        pnp_small_width = pnp["small_width"].as<float>(pnp_small_width);
        pnp_large_width = pnp["large_width"].as<float>(pnp_large_width);
        pnp_armor_height = pnp["height"].as<float>(pnp_armor_height);
    }

    ultra_vision::GimbalAimConfig gimbal_cfg;
    double gimbal_command_rate_hz = 100.0;
    if (tracker_cfg["tracker"]["gimbal"]) {
        const auto gimbal = tracker_cfg["tracker"]["gimbal"];
        const double pitch_min_degrees = gimbal["pitch_min_degrees"].as<double>(-70.0);
        const double pitch_max_degrees = gimbal["pitch_max_degrees"].as<double>(70.0);
        gimbal_cfg.pitch_min = pitch_min_degrees * CV_PI / 180.0;
        gimbal_cfg.pitch_max = pitch_max_degrees * CV_PI / 180.0;
        gimbal_cfg.max_yaw_velocity =
            gimbal["max_yaw_velocity_degrees_per_sec"].as<double>(120.0) * CV_PI / 180.0;
        gimbal_cfg.max_pitch_velocity =
            gimbal["max_pitch_velocity_degrees_per_sec"].as<double>(80.0) * CV_PI / 180.0;
        gimbal_cfg.max_yaw_acceleration =
            gimbal["max_yaw_acceleration_degrees_per_sec2"].as<double>(600.0) *
            CV_PI / 180.0;
        gimbal_cfg.max_pitch_acceleration =
            gimbal["max_pitch_acceleration_degrees_per_sec2"].as<double>(450.0) *
            CV_PI / 180.0;
        gimbal_cfg.max_yaw_jerk =
            gimbal["max_yaw_jerk_degrees_per_sec3"].as<double>(5000.0) * CV_PI / 180.0;
        gimbal_cfg.max_pitch_jerk =
            gimbal["max_pitch_jerk_degrees_per_sec3"].as<double>(4000.0) * CV_PI / 180.0;
        gimbal_cfg.yaw_response_gain =
            gimbal["yaw_response_gain"].as<double>(gimbal_cfg.yaw_response_gain);
        gimbal_cfg.pitch_response_gain =
            gimbal["pitch_response_gain"].as<double>(gimbal_cfg.pitch_response_gain);
        gimbal_cfg.feedforward_gain =
            gimbal["feedforward_gain"].as<double>(gimbal_cfg.feedforward_gain);
        gimbal_cfg.feedforward_time_constant =
            gimbal["feedforward_time_constant"].as<double>(
                gimbal_cfg.feedforward_time_constant);
        gimbal_cfg.settle_angle = gimbal["settle_angle_degrees"].as<double>(0.6) *
            CV_PI / 180.0;
        gimbal_cfg.settle_velocity =
            gimbal["settle_velocity_degrees_per_sec"].as<double>(5.0) * CV_PI / 180.0;
        gimbal_cfg.max_dt = gimbal["max_control_dt"].as<double>(gimbal_cfg.max_dt);
        gimbal_command_rate_hz = gimbal["command_rate_hz"].as<double>(100.0);
    }

    ultra_vision::TargetSelectorConfig selector_cfg;
    ultra_vision::ShooterConfig shooter_cfg;
    ultra_vision::AimSignalFilterConfig aim_filter_cfg;
    int max_control_lost_frames = 5;
    double fire_angle_tolerance = 1.0 * CV_PI / 180.0;
    const bool fire_enabled = std::getenv("ULTRA_VISION_DISABLE_FIRE") == nullptr;
    const YAML::Node selector_node = tracker_cfg["tracker"]["selector"]
        ? tracker_cfg["tracker"]["selector"]
        : tracker_cfg["tracker"]["predictive_aim"];
    if (selector_node) {
        selector_cfg.enabled = selector_node["enabled"].as<bool>(true);
        selector_cfg.projectile_speed = selector_node["projectile_speed"].as<double>(25.0);
        selector_cfg.gravity = selector_node["gravity"].as<double>(9.81);
        selector_cfg.command_latency = selector_node["command_latency"].as<double>(0.08);
        selector_cfg.max_lead_time = selector_node["max_lead_time"].as<double>(0.50);
        max_control_lost_frames =
            selector_node["max_control_lost_frames"].as<int>(5);
        selector_cfg.spin_omega_threshold =
            selector_node["spin_omega_threshold"].as<double>(2.0);
        selector_cfg.coming_angle = selector_node["coming_angle_degrees"].as<double>(60.0) *
            CV_PI / 180.0;
        selector_cfg.leaving_angle = selector_node["leaving_angle_degrees"].as<double>(20.0) *
            CV_PI / 180.0;
        selector_cfg.yaw_velocity =
            selector_node["gimbal_yaw_velocity_degrees_per_sec"].as<double>(360.0) *
            CV_PI / 180.0;
        selector_cfg.pitch_velocity =
            selector_node["gimbal_pitch_velocity_degrees_per_sec"].as<double>(180.0) *
            CV_PI / 180.0;
        selector_cfg.yaw_acceleration =
            selector_node["gimbal_yaw_acceleration_degrees_per_sec2"].as<double>(1800.0) *
            CV_PI / 180.0;
        selector_cfg.pitch_acceleration =
            selector_node["gimbal_pitch_acceleration_degrees_per_sec2"].as<double>(1200.0) *
            CV_PI / 180.0;
        selector_cfg.handoff_start_angle =
            selector_node["handoff_start_angle_degrees"].as<double>(10.0) *
            CV_PI / 180.0;
        selector_cfg.handoff_duration =
            selector_node["handoff_duration"].as<double>(selector_cfg.handoff_duration);
        const YAML::Node shooter_node = tracker_cfg["tracker"]["shooter"];
        shooter_cfg.min_fire_interval = shooter_node
            ? shooter_node["min_fire_interval"].as<double>(0.10)
            : selector_node["min_fire_interval"].as<double>(0.10);
        shooter_cfg.ready_handoff_gain = shooter_node
            ? shooter_node["ready_handoff_gain"].as<double>(
                shooter_cfg.ready_handoff_gain)
            : shooter_cfg.ready_handoff_gain;
        shooter_cfg.end_handoff_gain = shooter_node
            ? shooter_node["end_handoff_gain"].as<double>(shooter_cfg.end_handoff_gain)
            : shooter_cfg.end_handoff_gain;
        fire_angle_tolerance = (shooter_node
            ? shooter_node["fire_angle_tolerance_degrees"].as<double>(3.0)
            : selector_node["fire_angle_tolerance_degrees"].as<double>(3.0)) *
            CV_PI / 180.0;
    }
    if (tracker_cfg["tracker"]["aim_filter"]) {
        const auto aim = tracker_cfg["tracker"]["aim_filter"];
        aim_filter_cfg.enabled = aim["enabled"].as<bool>(true);
        aim_filter_cfg.process_noise_acceleration =
            aim["process_noise_acceleration"].as<double>(
                aim_filter_cfg.process_noise_acceleration);
        aim_filter_cfg.measurement_noise = aim["measurement_noise"].as<double>(
            aim_filter_cfg.measurement_noise);
        aim_filter_cfg.reset_innovation =
            aim["reset_innovation_degrees"].as<double>(20.0) * CV_PI / 180.0;
        aim_filter_cfg.reset_timeout = aim["reset_timeout"].as<double>(
            aim_filter_cfg.reset_timeout);
        aim_filter_cfg.max_dt = aim["max_dt"].as<double>(aim_filter_cfg.max_dt);
    }

    ultra_vision::io::VisionDateReceiver receiver(config_path("simulator.yaml"));
    const char* simulator_host = std::getenv("ULTRA_VISION_SIM_HOST");
    const char* simulator_port = std::getenv("ULTRA_VISION_SIM_PORT");
    if (simulator_host || simulator_port) {
        const std::string host = simulator_host ? simulator_host : "127.0.0.1";
        const uint16_t port = static_cast<uint16_t>(std::max(
            1, std::min(65535, simulator_port ? std::atoi(simulator_port) : 7666)));
        receiver.overrideEndpoint(host, port);
    }
    if (const char* command_port = std::getenv("ULTRA_VISION_SIM_COMMAND_PORT")) {
        receiver.overrideCommandEndpoint(static_cast<uint16_t>(std::max(
            1, std::min(65535, std::atoi(command_port)))));
    }
    receiver.connect();
    receiver.sendGimbalCommand(0.0, 0.0);

    ultra_vision::Tracker tracker(track_cfg);
    ultra_vision::Detector detector(cv::Mat(), det_cfg);
    ultra_vision::GimbalController gimbal_controller(
        gimbal_cfg,
        [&receiver](double yaw, double pitch) {
            return receiver.sendGimbalCommand(yaw, pitch);
        },
        gimbal_command_rate_hz);
    ultra_vision::TargetSelector target_selector(selector_cfg);
    ultra_vision::Shooter shooter(shooter_cfg);
    ultra_vision::AimSignalFilter aim_signal_filter(aim_filter_cfg);

    std::ofstream estimate_recorder;
    if (const char* estimate_path = std::getenv("ULTRA_VISION_ESTIMATE_CSV")) {
        estimate_recorder.open(estimate_path, std::ios::out | std::ios::trunc);
        if (estimate_recorder) {
            estimate_recorder << "x,y,z,color,distance,time_us,plate_id\n";
        } else {
            std::cerr << "Failed to open estimate CSV: " << estimate_path << std::endl;
        }
    }

    std::ofstream observation_recorder;
    if (const char* observation_path = std::getenv("ULTRA_VISION_OBSERVATION_CSV")) {
        observation_recorder.open(observation_path, std::ios::out | std::ios::trunc);
        if (observation_recorder) {
            observation_recorder
                << "x,y,z,rx,ry,rz,tracker_x,tracker_y,tracker_z,"
                   "cmd_yaw,cmd_pitch,time_us,plate_id,state\n";
        } else {
            std::cerr << "Failed to open observation CSV: " << observation_path << std::endl;
        }
    }

    std::ofstream selector_recorder;
    if (const char* selector_path = std::getenv("ULTRA_VISION_SELECTOR_CSV")) {
        selector_recorder.open(selector_path, std::ios::out | std::ios::trunc);
        if (selector_recorder) {
            selector_recorder
                << "time_us,center_x,center_y,center_z,yaw,omega,armor_id,"
                   "armor_x,armor_y,armor_z,target_yaw,target_pitch,lead,"
                   "target_yaw_velocity,target_pitch_velocity,gimbal_time,handoff,in_fire\n";
        } else {
            std::cerr << "Failed to open selector CSV: " << selector_path << std::endl;
        }
    }

    int processing_height = 0;
    cv::Mat camera_matrix;
    cv::Mat dist_coeffs = cv::Mat::zeros(5, 1, CV_64F);
    cv::Mat frame;
    cv::Mat image;
    cv::Mat gray;
    cv::Mat binary;
    std::vector<std::vector<cv::Point>> contours;
    std::vector<ultra_vision::Light> lights;
    std::vector<ultra_vision::Armor> armors;

    double commanded_yaw = 0.0;
    double commanded_pitch = 0.0;
    ultra_vision::GimbalControllerSnapshot last_gimbal_command;
    ultra_vision::TargetDecision last_decision;
    ultra_vision::ShooterState shooter_state = ultra_vision::ShooterState::IDLE;
    ultra_vision::ShooterErrorReason shooter_error =
        ultra_vision::ShooterErrorReason::NONE;
    int aim_valid_updates = 0;
    int aim_settled_updates = 0;
    int plate_switches = 0;
    auto fps_start = std::chrono::steady_clock::now();
    int fps_frames = 0;
    int missing_frames = 0;
    int key = -1;

    while (key != 27 && key != 'e' && key != 'E') {
        uint64_t frame_sequence = 0;
        uint64_t frame_local_timestamp_us = 0;
        frame = receiver.getFrame(
            &frame_sequence, nullptr, &frame_local_timestamp_us);
        if (frame.empty()) {
            ++missing_frames;
            if (missing_frames == 1 || missing_frames % 30 == 0) {
                std::cerr << "Waiting for simulator frames..." << std::endl;
            }
            key = cv::waitKey(1);
            continue;
        }
        missing_frames = 0;

        // All estimator, control, and command-history timestamps use the same
        // local monotonic clock. The simulator source timestamp is metadata
        // only and is never compared with local wall-clock time.
        last_gimbal_command = gimbal_controller.snapshotAt(frame_local_timestamp_us);
        if (last_gimbal_command.valid) {
            commanded_yaw = last_gimbal_command.yaw;
            commanded_pitch = last_gimbal_command.pitch;
        }
        ultra_vision::CameraPose camera_pose;
        camera_pose.valid = last_gimbal_command.valid;
        camera_pose.timestamp_valid = last_gimbal_command.valid;
        camera_pose.timestamp =
            ultra_vision::StandardClock::secondsFromUs(frame_local_timestamp_us);
        camera_pose.yaw = commanded_yaw;
        camera_pose.pitch = commanded_pitch;
        tracker.setCameraPose(camera_pose);

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
            detected_armor.solve_result = ultra_vision::solveArmorPnP(
                detected_armor, camera_matrix, dist_coeffs,
                pnp_small_width, pnp_large_width, pnp_armor_height);
            if (detected_armor.solve_result) ++pnp_count;
        }

        // Use the capture timestamp carried by the simulator frame. Syncing to
        // processing completion time predicts the chassis tens of milliseconds
        // into the future, which both offsets the rendered model and overdrives
        // the gimbal during small-gyro motion.
        const double timestamp =
            ultra_vision::StandardClock::secondsFromUs(frame_local_timestamp_us);
        tracker.update(armors, timestamp);

        if (observation_recorder && tracker.hasObservation() &&
            !tracker.getLastObservedArmor().tvec.empty()) {
            const auto& observed = tracker.getLastObservedArmor();
            const cv::Mat center = tracker.getTargetCenter();
            observation_recorder << std::fixed << std::setprecision(6)
                << observed.tvec.at<double>(0) << ','
                << observed.tvec.at<double>(1) << ','
                << observed.tvec.at<double>(2) << ','
                << observed.rvec.at<double>(0) << ','
                << observed.rvec.at<double>(1) << ','
                << observed.rvec.at<double>(2) << ','
                << center.at<double>(0) << ','
                << center.at<double>(1) << ','
                << center.at<double>(2) << ','
                << commanded_yaw << ',' << commanded_pitch << ','
                << frame_local_timestamp_us << ','
                << tracker.getLastObservedPlateId() << ','
                << static_cast<int>(tracker.getState()) << '\n';
        }

        if (estimate_recorder &&
            tracker.getState() != ultra_vision::TrackerState::LOST &&
            tracker.getState() != ultra_vision::TrackerState::DETECTING) {
            const auto estimated_positions = tracker.getEstimatedArmorPositions();
            estimate_recorder << std::fixed << std::setprecision(6);
            for (int plate = 0; plate < 4; ++plate) {
                if (estimated_positions[plate].rows != 3 ||
                    estimated_positions[plate].cols != 1) {
                    continue;
                }
                const double x = estimated_positions[plate].at<double>(0);
                const double y = estimated_positions[plate].at<double>(1);
                const double z = estimated_positions[plate].at<double>(2);
                estimate_recorder << x << ',' << y << ',' << z << ",0,"
                                  << std::sqrt(x * x + y * y + z * z) << ','
                                  << frame_local_timestamp_us << ',' << plate << '\n';
            }
        }

        const bool tracker_usable =
            tracker.getState() == ultra_vision::TrackerState::TRACKING ||
            (tracker.getState() == ultra_vision::TrackerState::TEMP_LOST &&
             tracker.getLostCount() <= max_control_lost_frames);
        // The selector consumes the estimator's world-frame state and produces
        // one decision shared by the gimbal, firing logic, and visualization.
        if (tracker_usable) {
            const int previous_armor_id = last_decision.valid
                ? last_decision.armor_id : -1;
            ultra_vision::TargetDecision decision;
            if (selector_cfg.enabled) {
                ultra_vision::TargetEstimate estimate;
                estimate.center = tracker.getTargetCenterWorldArray();
                estimate.velocity = tracker.getTargetVelocityWorld();
                estimate.yaw = tracker.getYawWorld();
                estimate.omega = tracker.getOmega();
                estimate.armor_radius = tracker.getArmorRadius();
                decision = target_selector.select(
                    estimate, commanded_yaw, commanded_pitch);
            }
            last_decision = decision;

            if (selector_recorder && decision.valid) {
                selector_recorder << std::fixed << std::setprecision(6)
                    << frame_local_timestamp_us << ','
                    << tracker.getTargetCenterWorldArray()[0] << ','
                    << tracker.getTargetCenterWorldArray()[1] << ','
                    << tracker.getTargetCenterWorldArray()[2] << ','
                    << tracker.getYawWorld() << ','
                    << tracker.getOmega() << ','
                    << decision.armor_id << ','
                    << decision.armor_position[0] << ','
                    << decision.armor_position[1] << ','
                    << decision.armor_position[2] << ','
                    << decision.target_yaw << ','
                    << decision.target_pitch << ','
                    << decision.lead_time << ','
                    << decision.target_yaw_velocity << ','
                    << decision.target_pitch_velocity << ','
                    << decision.gimbal_time << ','
                    << decision.handoff_gain << ','
                    << static_cast<int>(decision.in_fire_window) << '\n';
            }

            if (decision.valid && previous_armor_id >= 0 &&
                decision.armor_id != previous_armor_id) {
                ++plate_switches;
            }

            bool aim_ready = false;
            double aim_yaw_error = 0.0;
            double aim_pitch_error = 0.0;
            if (decision.valid) {
                const auto filtered_aim = aim_signal_filter.update(
                    decision.target_yaw, decision.target_pitch,
                    decision.armor_id, timestamp);
                ultra_vision::GimbalTargetAngles desired;
                desired.valid = filtered_aim.valid;
                desired.yaw = filtered_aim.yaw;
                desired.pitch = filtered_aim.pitch;
                desired.velocity_valid = true;
                desired.yaw_velocity = decision.target_yaw_velocity;
                desired.pitch_velocity = decision.target_pitch_velocity;
                if (desired.valid) {
                    ++aim_valid_updates;

                    // The selector owns target choice and fire timing. The
                    // controller only reports whether its current absolute
                    // angle is aligned with the selector's planned intercept.
                    const auto aim_snapshot = gimbal_controller.snapshot();
                    const bool target_is_continuous = aim_snapshot.valid &&
                        std::abs(ultra_vision::GimbalAimer::normalizeAngle(
                            desired.yaw - aim_snapshot.desired_yaw)) <=
                            fire_angle_tolerance &&
                        std::abs(desired.pitch - aim_snapshot.desired_pitch) <=
                            fire_angle_tolerance;
                    aim_ready = aim_snapshot.valid &&
                        aim_snapshot.processed_generation ==
                            aim_snapshot.target_generation &&
                        target_is_continuous &&
                        std::abs(aim_snapshot.yaw_error) <= fire_angle_tolerance &&
                        std::abs(aim_snapshot.pitch_error) <= fire_angle_tolerance;
                    aim_yaw_error = aim_snapshot.yaw_error;
                    aim_pitch_error = aim_snapshot.pitch_error;
                    if (aim_ready) ++aim_settled_updates;
                    gimbal_controller.setTargetAngles(desired);
                }
            }

            const auto shooter_output = shooter.update(
                decision, aim_ready, timestamp);
            shooter_state = shooter_output.state;
            shooter_error = shooter_output.error_reason;
            if (fire_enabled && shooter_output.fire) {
                receiver.sendFireCommand();
                std::cout << cv::format(
                    "FIRE state=%s omega=%+.2f armor=%d lead=%.3fs "
                    "yaw_err=%+.1fdeg pitch_err=%+.1fdeg",
                    shooterStatusName(
                        shooter_output.state, shooter_output.error_reason).c_str(),
                    tracker.getOmega(),
                    decision.armor_id,
                    decision.lead_time,
                    aim_yaw_error * 180.0 / CV_PI,
                    aim_pitch_error * 180.0 / CV_PI)
                          << std::endl;
            }
        } else if (!tracker_usable) {
            const auto shooter_output = shooter.update({}, false, timestamp);
            shooter_state = shooter_output.state;
            shooter_error = shooter_output.error_reason;
        }

        if (tracker.getState() != ultra_vision::TrackerState::LOST && tracker.getTargetCenter().rows == 3) {
            const auto estimated_positions = tracker.getEstimatedArmorPositions();
            const double yaw = tracker.getYaw();
            const bool has_observation = tracker.hasObservation();
            const int observed_plate = has_observation
                ? tracker.getLastObservedPlateId() : -1;

            const bool large_armor = has_observation &&
                tracker.getLastObservedArmor().armor_type == ultra_vision::large;
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
                           shooterStatusName(shooter_state, shooter_error).c_str()),
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
                const auto selected_corners = projectArmor(
                    selected_center, last_decision.armor_id,
                    tracker.worldToCameraYaw(last_decision.predicted_yaw), vertical_axis,
                    selected_half_width, half_height,
                    camera_matrix, dist_coeffs);
                drawQuad(image, selected_corners, cv::Scalar(0, 165, 255), 2);
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
                if (projectPoint(lead_point, camera_matrix, dist_coeffs, lead_image)) {
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

        ++fps_frames;
        const auto now = std::chrono::steady_clock::now();
        const double elapsed = std::chrono::duration<double>(now - fps_start).count();
        if (elapsed >= 1.0) {
            const char* tracker_state = "LOST";
            if (tracker.getState() == ultra_vision::TrackerState::DETECTING) {
                tracker_state = "DETECTING";
            } else if (tracker.getState() == ultra_vision::TrackerState::TRACKING) {
                tracker_state = "TRACKING";
            } else if (tracker.getState() == ultra_vision::TrackerState::TEMP_LOST) {
                tracker_state = "TEMP_LOST";
            }
            std::cout << "Processed FPS: " << fps_frames / elapsed
                      << ", armors: " << armors.size()
                      << ", PnP: " << pnp_count
                      << ", tracker: " << tracker_state
                      << ", fresh: " << tracker.hasObservation()
                      << ", lost: " << tracker.getLostCount()
                      << ", omega: " << tracker.getOmega()
                      << ", aim_settled: " << aim_settled_updates << "/"
                      << aim_valid_updates
                      << ", shooter: " << shooterStatusName(shooter_state, shooter_error)
                      << ", plate_switches: " << plate_switches << std::endl;
            if (ultra_vision::Logger::debugEnabled()) {
                ultra_vision::Logger::log(
                    ultra_vision::LogLevel::DEBUG,
                    "pipeline",
                    cv::format("tracker=%s fresh=%d omega=%.4f",
                               tracker_state,
                               tracker.hasObservation() ? 1 : 0,
                               tracker.getOmega()));
            }
            fps_start = now;
            fps_frames = 0;
            aim_valid_updates = 0;
            aim_settled_updates = 0;
            plate_switches = 0;
        }

        cv::resize(image, image, cv::Size(640, 480), 0.0, 0.0, cv::INTER_AREA);
        cv::imshow("Detection & Tracking", image);
        key = cv::waitKey(1);
    }

    ultra_vision::Logger::shutdown();
    return 0;
}
}
