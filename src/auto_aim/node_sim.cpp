#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>

#include <sim_receiver/vision_date_receiver.hpp>
#include "common/types.hpp"
#include "common/standard_clock.hpp"
#include "config_loader.hpp"
#include "visualization/projection.hpp"
#include "control/aim_signal_filter.hpp"
#include "control/gimbal_aimer.hpp"
#include "control/gimbal_controller.hpp"
#include "control/shooter.hpp"
#include "control/target_selector.hpp"
#include "perception/detector.hpp"
#include "perception/async_detector.hpp"
#include "perception/pnp_solver.hpp"
#include "Kalman/tracker.hpp"

#ifndef ULTRA_VISION_CONFIG_DIR
#define ULTRA_VISION_CONFIG_DIR "configs"
#endif

namespace {

cv::Mat makeCameraMatrix(int rows, int columns, double vertical_fov_degrees)
{
    const double focal_length = 0.5 * rows /
        std::tan(vertical_fov_degrees * CV_PI / 360.0);
    return (cv::Mat_<double>(3, 3) <<
        focal_length, 0.0, 0.5 * columns,
        0.0, focal_length, 0.5 * rows,
        0.0, 0.0, 1.0);
}

const char* shooterStateName(auto_aim::ShooterState state)
{
    switch (state) {
    case auto_aim::ShooterState::IDLE: return "IDLE";
    case auto_aim::ShooterState::READY: return "READY";
    case auto_aim::ShooterState::FIRING: return "FIRING";
    case auto_aim::ShooterState::ERROR: return "ERROR";
    case auto_aim::ShooterState::END: return "END";
    }
    return "UNKNOWN";
}

const char* shooterErrorName(auto_aim::ShooterErrorReason reason)
{
    switch (reason) {
    case auto_aim::ShooterErrorReason::NONE: return "NONE";
    case auto_aim::ShooterErrorReason::OUT_OF_FIRE_WINDOW: return "OUT_OF_FIRE_WINDOW";
    case auto_aim::ShooterErrorReason::GIMBAL_ERROR: return "GIMBAL_ERROR";
    }
    return "UNKNOWN";
}

std::string shooterStatusName(auto_aim::ShooterState state,
                              auto_aim::ShooterErrorReason reason)
{
    std::string status = shooterStateName(state);
    if (state == auto_aim::ShooterState::ERROR) {
        status += "(" + std::string(shooterErrorName(reason)) + ")";
    }
    return status;
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

    const auto_aim::DetectorConfig det_cfg = auto_aim::loadDetectorConfig(detector_cfg);

    const auto_aim::TrackerConfig track_cfg = auto_aim::loadTrackerConfig(tracker_cfg);
    const auto_aim::PnpGeometry pnp_geometry = auto_aim::loadPnpGeometry(tracker_cfg);

    // Optional neural-network detector. It replaces the light-bar pipeline but
    // produces the same Armor type, so PnP and everything downstream is shared.
    // It runs on its own thread so that a slow inference never throttles the
    // camera and control loop.
    std::unique_ptr<auto_aim::AsyncArmorDetector> nn_detector;
#ifdef ULTRA_VISION_USE_OPENVINO
    if (auto_aim::neuralDetectorEnabled(detector_cfg)) {
        auto_aim::AsyncDetectorConfig async_cfg;
        async_cfg.detector =
            auto_aim::loadNeuralDetectorConfig(detector_cfg, config_dir);
        if (!async_cfg.detector.model_path.empty()) {
            nn_detector = std::make_unique<auto_aim::AsyncArmorDetector>(async_cfg);
            std::cout << "Neural armor detector: " << async_cfg.detector.model_path
                      << " (device " << async_cfg.detector.device << ", async)"
                      << std::endl;
        } else {
            std::cerr << "Neural detector enabled but model_path is empty; "
                         "falling back to the classical detector" << std::endl;
        }
    }
#else
    if (auto_aim::neuralDetectorEnabled(detector_cfg)) {
        std::cerr << "Neural detector requested but this build has no OpenVINO; "
                     "falling back to the classical detector" << std::endl;
    }
#endif
    const bool use_neural_detector = nn_detector != nullptr;

    auto_aim::GimbalAimConfig gimbal_cfg;
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

    auto_aim::TargetSelectorConfig selector_cfg;
    auto_aim::ShooterConfig shooter_cfg;
    auto_aim::AimSignalFilterConfig aim_filter_cfg;
    int max_control_lost_frames = 5;
    double fire_angle_tolerance = 1.0 * CV_PI / 180.0;
    const bool fire_enabled = std::getenv("ULTRA_VISION_DISABLE_FIRE") == nullptr;
    const bool show_display = std::getenv("ULTRA_VISION_NO_DISPLAY") == nullptr;
    // Restrict inference to the projected chassis region while tracking.
    // OFF by default: it was measured not to speed anything up, because the
    // detector letterboxes the crop back to the network's fixed 640x640 input.
    // It becomes useful only together with a smaller-input model export.
    const bool dynamic_roi =
        (std::getenv("ULTRA_VISION_DYNAMIC_ROI") != nullptr) &&
        std::string(std::getenv("ULTRA_VISION_DYNAMIC_ROI")) != "0";
    const YAML::Node selector_node = tracker_cfg["tracker"]["selector"]
        ? tracker_cfg["tracker"]["selector"]
        : tracker_cfg["tracker"]["predictive_aim"];
    if (selector_node) {
        selector_cfg.enabled = selector_node["enabled"].as<bool>(true);
        selector_cfg.projectile_speed = selector_node["projectile_speed"].as<double>(25.0);
        selector_cfg.gravity = selector_node["gravity"].as<double>(9.81);
        selector_cfg.command_latency = selector_node["command_latency"].as<double>(0.08);
        selector_cfg.decide_speed =
            selector_node["decide_speed"].as<double>(selector_cfg.decide_speed);
        selector_cfg.low_speed_latency =
            selector_node["low_speed_latency"].as<double>(selector_cfg.low_speed_latency);
        selector_cfg.high_speed_latency =
            selector_node["high_speed_latency"].as<double>(selector_cfg.high_speed_latency);
        selector_cfg.lock_switch_margin =
            selector_node["lock_switch_margin"].as<double>(selector_cfg.lock_switch_margin);
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

    sim_receiver::VisionDateReceiver receiver(config_path("simulator.yaml"));
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
    // A previous run can leave the camera pointing anywhere, which makes the
    // next run see nothing and makes any A/B comparison meaningless. Ask for a
    // clean scene unless the caller wants to keep the manual state.
    if (const char* reset = std::getenv("ULTRA_VISION_SIM_RESET")) {
        if (std::string(reset) != "0") {
            receiver.sendResetCommand();
            std::cout << "Requested a simulator RESET for a reproducible run"
                      << std::endl;
        }
    }
    receiver.sendGimbalCommand(0.0, 0.0);

    auto_aim::Tracker tracker(track_cfg);
    auto_aim::Detector detector(cv::Mat(), det_cfg);
    auto_aim::GimbalController gimbal_controller(
        gimbal_cfg,
        [&receiver](double yaw, double pitch) {
            return receiver.sendGimbalCommand(yaw, pitch);
        },
        gimbal_command_rate_hz);
    auto_aim::GimbalAimer bootstrap_aimer(gimbal_cfg);
    auto_aim::TargetSelector target_selector(selector_cfg);
    auto_aim::Shooter shooter(shooter_cfg);
    auto_aim::AimSignalFilter aim_signal_filter(aim_filter_cfg);

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
    std::vector<auto_aim::Light> lights;
    std::vector<auto_aim::Armor> armors;

    double commanded_yaw = 0.0;
    double commanded_pitch = 0.0;
    // Time of the detection the estimator's state was last updated from. The
    // selector uses (now - this) as its lead-time compensation.
    double last_detection_time = 0.0;
    auto_aim::GimbalControllerSnapshot last_gimbal_command;
    auto_aim::TargetDecision last_decision;
    auto_aim::ShooterState shooter_state = auto_aim::ShooterState::IDLE;
    auto_aim::ShooterErrorReason shooter_error =
        auto_aim::ShooterErrorReason::NONE;
    int aim_valid_updates = 0;
    int aim_settled_updates = 0;
    int plate_switches = 0;
    int stable_spin_updates = 0;
    auto fps_start = std::chrono::steady_clock::now();
    int fps_frames = 0;
    // Main-loop stage timing. The detector already runs on its own thread, so
    // this tells where the remaining time goes: waiting for the next frame,
    // everything the loop does with it, and the HighGUI display.
    double prof_wait_ms = 0.0;
    double prof_work_ms = 0.0;
    double prof_display_ms = 0.0;
    int missing_frames = 0;
    int key = -1;

    while (key != 27 && key != 'e' && key != 'E') {
        const auto loop_start = std::chrono::steady_clock::now();
        uint64_t frame_sequence = 0;
        uint64_t frame_local_timestamp_us = 0;
        frame = receiver.getFrame(
            &frame_sequence, nullptr, &frame_local_timestamp_us);
        const auto after_frame = std::chrono::steady_clock::now();
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
        auto_aim::CameraPose camera_pose;
        camera_pose.valid = last_gimbal_command.valid;
        camera_pose.timestamp_valid = last_gimbal_command.valid;
        camera_pose.timestamp =
            auto_aim::StandardClock::secondsFromUs(frame_local_timestamp_us);
        camera_pose.yaw = commanded_yaw;
        camera_pose.pitch = commanded_pitch;

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

        armors.clear();
        lights.clear();
        int pnp_count = 0;
        bool fresh_detection = false;
        if (use_neural_detector) {
            // Never wait for inference: hand the frame over and, if a result has
            // finished since the last iteration, consume it. A result keeps the
            // timestamp of the frame it came from, so the estimate is updated
            // on the correct time base instead of treating a stale detection as
            // if it described the current frame.
            // Dynamic ROI: once the chassis is locked, the network only has to
            // look where the four plates can be, so inference cost scales with
            // the region instead of the whole frame. Before the first lock, or
            // when disabled, the full frame is used.
            cv::Rect inference_roi;
            if (dynamic_roi) {
                const auto estimated_positions = tracker.getEstimatedArmorPositions();
                std::vector<cv::Point2f> projected;
                for (const auto& position : estimated_positions) {
                    cv::Point2f point;
                    if (auto_aim::projectPoint(position, camera_matrix, dist_coeffs,
                                               point)) {
                        projected.push_back(point);
                    }
                }
                if (projected.size() >= 2 &&
                    tracker.getState() != auto_aim::TrackerState::LOST) {
                    cv::Rect box = cv::boundingRect(projected);
                    const int margin =
                        static_cast<int>(0.35 * std::max(box.width, box.height)) + 24;
                    box.x -= margin;
                    box.y -= margin;
                    box.width += 2 * margin;
                    box.height += 2 * margin;
                    box &= cv::Rect(0, 0, image.cols, image.rows);
                    if (box.width >= 96 && box.height >= 96) {
                        inference_roi = box;
                    }
                }
            }
            nn_detector->submit(image, frame_sequence, frame_local_timestamp_us,
                                inference_roi);

            auto_aim::AsyncArmorDetector::Result nn_result;
            if (nn_detector->takeLatest(nn_result)) {
                const double detection_time =
                    auto_aim::StandardClock::secondsFromUs(nn_result.timestamp_us);
                for (auto& detected_armor : nn_result.armors) {
                    detected_armor.solve_result = auto_aim::solveArmorPnP(
                        detected_armor, camera_matrix, dist_coeffs,
                        pnp_geometry.small_width, pnp_geometry.large_width,
                        pnp_geometry.height);
                    if (detected_armor.solve_result) ++pnp_count;
                }
                armors = nn_result.armors;
                if (std::getenv("ULTRA_VISION_NN_DEBUG")) {
                    std::cerr << "[nn] seq=" << nn_result.sequence
                              << " detections=" << nn_result.armors.size()
                              << " solved=" << pnp_count
                              << " rows=" << camera_matrix.rows
                              << " ts=" << nn_result.timestamp_us << std::endl;
                }

                // Camera pose as it was when that frame was captured.
                const auto detection_snapshot =
                    gimbal_controller.snapshotAt(nn_result.timestamp_us);
                auto_aim::CameraPose detection_pose;
                detection_pose.valid = detection_snapshot.valid;
                detection_pose.timestamp_valid = detection_snapshot.valid;
                detection_pose.timestamp = detection_time;
                detection_pose.yaw = detection_snapshot.valid
                    ? detection_snapshot.yaw : commanded_yaw;
                detection_pose.pitch = detection_snapshot.valid
                    ? detection_snapshot.pitch : commanded_pitch;
                tracker.setCameraPose(detection_pose);

                const double timestamp = detection_time;
                tracker.update(armors, timestamp);
                last_detection_time = detection_time;
                fresh_detection = true;
            } else {
                // No new detection yet. Leave the estimator where it is: it
                // holds the state as of the last detection, and the selector
                // extrapolates from that with its own lead. Advancing the
                // filter to "now" here and then feeding it a detection that is
                // ~200 ms old would inject that detection at a time the state
                // has already moved past.
            }
        } else {
            tracker.setCameraPose(camera_pose);
            detector.gray_img(image, gray);
            cv::GaussianBlur(gray, gray, cv::Size(3, 3), 0);
            detector.binary_img(gray, binary, det_cfg.binary_threshold);

            contours.clear();
            detector.find_contours(binary, contours);
            detector.find_lights(contours, lights, image, det_cfg.enemy_color);
            if (!lights.empty()) {
                detector.find_armors(lights, armors);
            }

            for (auto& detected_armor : armors) {
                detected_armor.solve_result = auto_aim::solveArmorPnP(
                    detected_armor, camera_matrix, dist_coeffs,
                    pnp_geometry.small_width, pnp_geometry.large_width,
                    pnp_geometry.height);
                if (detected_armor.solve_result) ++pnp_count;
            }

            // Use the capture timestamp carried by the simulator frame. Syncing
            // to processing completion time predicts the chassis tens of
            // milliseconds into the future, which both offsets the rendered
            // model and overdrives the gimbal during small-gyro motion.
            const double timestamp =
                auto_aim::StandardClock::secondsFromUs(frame_local_timestamp_us);
            tracker.update(armors, timestamp);
            fresh_detection = true;
        }

        // Control-side time base: always the frame being processed, regardless
        // of how old the detection that just fed the estimator is.
        const double timestamp =
            auto_aim::StandardClock::secondsFromUs(frame_local_timestamp_us);
        (void)fresh_detection;

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
            tracker.getState() != auto_aim::TrackerState::LOST &&
            tracker.getState() != auto_aim::TrackerState::DETECTING) {
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
            tracker.getState() == auto_aim::TrackerState::TRACKING ||
            (tracker.getState() == auto_aim::TrackerState::TEMP_LOST &&
             tracker.getLostCount() <= max_control_lost_frames);
        // The selector consumes the estimator's world-frame state and produces
        // one decision shared by the gimbal, firing logic, and visualization.
        if (tracker_usable) {
            const int previous_armor_id = last_decision.valid
                ? last_decision.armor_id : -1;
            auto_aim::TargetDecision decision;
            if (selector_cfg.enabled) {
                auto_aim::TargetEstimate estimate;
                estimate.center = tracker.getTargetCenterWorldArray();
                estimate.velocity = tracker.getTargetVelocityWorld();
                estimate.yaw = tracker.getYawWorld();
                estimate.omega = tracker.getOmega();
                estimate.armor_radius = tracker.getArmorRadius();
                estimate.age = last_detection_time > 0.0
                    ? std::max(0.0, timestamp - last_detection_time) : 0.0;
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

            // The bootstrap warm-up must not depend on the rotation-rate
            // estimate. A stationary chassis legitimately never reports a rate,
            // and once the detector runs asynchronously its lower sample rate
            // starves the radial-angle estimator (four slopes inside a 0.35 s
            // window are no longer reachable at ~5 Hz), which used to pin the
            // loop in bootstrap forever. Gate on the selector having a usable
            // decision instead; the warm-up frames still apply.
            const bool estimate_ready = decision.valid;
            stable_spin_updates = estimate_ready
                ? std::min(stable_spin_updates + 1, 1000)
                : 0;
            const bool bootstrap_control =
                tracker.hasObservation() &&
                (!estimate_ready || stable_spin_updates < 5);

            bool aim_ready = false;
            double aim_yaw_error = 0.0;
            double aim_pitch_error = 0.0;
            auto_aim::GimbalTargetAngles desired;
            if (bootstrap_control) {
                const auto& observed = tracker.getLastObservedArmor();
                if (observed.tvec.rows == 3 && observed.tvec.cols == 1) {
                    const std::array<double, 3> target_camera{{
                        observed.tvec.at<double>(0),
                        observed.tvec.at<double>(1),
                        observed.tvec.at<double>(2)
                    }};
                    desired = bootstrap_aimer.solveTargetAngles(
                        target_camera, commanded_yaw, commanded_pitch);
                    desired.velocity_valid = false;
                    aim_signal_filter.reset();
                }
            } else if (decision.valid) {
                const auto filtered_aim = aim_signal_filter.update(
                    decision.target_yaw, decision.target_pitch,
                    decision.armor_id, timestamp);
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
                        std::abs(auto_aim::GimbalAimer::normalizeAngle(
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
                bootstrap_control ? auto_aim::TargetDecision{} : decision,
                aim_ready,
                timestamp);
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

        ++fps_frames;
        const auto now = std::chrono::steady_clock::now();
        const double elapsed = std::chrono::duration<double>(now - fps_start).count();
        if (elapsed >= 1.0) {
            const char* tracker_state = "LOST";
            if (tracker.getState() == auto_aim::TrackerState::DETECTING) {
                tracker_state = "DETECTING";
            } else if (tracker.getState() == auto_aim::TrackerState::TRACKING) {
                tracker_state = "TRACKING";
            } else if (tracker.getState() == auto_aim::TrackerState::TEMP_LOST) {
                tracker_state = "TEMP_LOST";
            }
            std::cout << "Processed FPS: " << fps_frames / elapsed
                      << ", lights: " << lights.size()
                      << ", armors: " << armors.size()
                      << ", PnP: " << pnp_count
                      << ", det_fps: " << (use_neural_detector
                             ? nn_detector->fps() : fps_frames / elapsed)
                      << ", det_ms: " << (use_neural_detector
                             ? nn_detector->latencyMs() : 0.0)
                      << ", tracker: " << tracker_state
                      << ", fresh: " << tracker.hasObservation()
                      << ", lost: " << tracker.getLostCount()
                      << ", omega: " << tracker.getOmega()
                      << ", rate: " << tracker.getRotationRateOmega()
                      << ", dec: " << last_decision.valid
                      << ", spin: " << tracker.hasRotationRate()
                      << ", aim_settled: " << aim_settled_updates << "/"
                      << aim_valid_updates
                      << ", shooter: " << shooterStatusName(shooter_state, shooter_error)
                      << ", plate_switches: " << plate_switches << std::endl;
            if (fps_frames > 0) {
                const double n = static_cast<double>(fps_frames);
                std::cout << "  loop ms: wait=" << prof_wait_ms / n
                          << " work=" << prof_work_ms / n
                          << " display=" << prof_display_ms / n << std::endl;
            }
            prof_wait_ms = prof_work_ms = prof_display_ms = 0.0;
            fps_start = now;
            fps_frames = 0;
            aim_valid_updates = 0;
            aim_settled_updates = 0;
            plate_switches = 0;
        }

        cv::resize(image, image, cv::Size(640, 480), 0.0, 0.0, cv::INTER_AREA);
        const auto before_display = std::chrono::steady_clock::now();
        // HighGUI costs more than the whole processing pipeline (measured
        // ~15 ms/frame vs ~0.9 ms of work), and a robot has no display at all.
        // Keep it for debugging, allow turning it off to see the real rate.
        if (show_display) {
            cv::imshow("Detection & Tracking", image);
            key = cv::waitKey(1);
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            key = 1;
        }
        const auto after_display = std::chrono::steady_clock::now();
        const auto ms = [](auto from, auto to) {
            return std::chrono::duration<double, std::milli>(to - from).count();
        };
        prof_wait_ms += ms(loop_start, after_frame);
        prof_work_ms += ms(after_frame, before_display);
        prof_display_ms += ms(before_display, after_display);
    }

    return 0;
}
