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
#include "visualization/curve_plotter.hpp"
#include "control/aim_pipeline.hpp"
#include "perception/armor_source.hpp"
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
    case auto_aim::ShooterErrorReason::NO_TARGET: return "NO_TARGET";
    case auto_aim::ShooterErrorReason::BALLISTICS_INVALID: return "BALLISTICS_INVALID";
    case auto_aim::ShooterErrorReason::ALREADY_HIT: return "ALREADY_HIT";
    case auto_aim::ShooterErrorReason::COOLDOWN: return "COOLDOWN";
    case auto_aim::ShooterErrorReason::RATE_TOO_HIGH: return "RATE_TOO_HIGH";
    case auto_aim::ShooterErrorReason::OUT_OF_RANGE: return "OUT_OF_RANGE";
    case auto_aim::ShooterErrorReason::NOT_SETTLED: return "NOT_SETTLED";
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

    // 配置只在 config_loader.hpp 里解析一次（默认值也只在结构体里）：
    // 硬件入口 node.cpp 以后接瞄准链时读的是同一份，不会再各写一套默认值。
    const auto_aim::GimbalAimConfig gimbal_cfg =
        auto_aim::loadGimbalConfig(tracker_cfg);
    const double gimbal_command_rate_hz =
        auto_aim::loadGimbalCommandRateHz(tracker_cfg);
    const auto_aim::TargetSelectorConfig selector_cfg =
        auto_aim::loadSelectorConfig(tracker_cfg);
    const auto_aim::ShooterConfig shooter_cfg =
        auto_aim::loadShooterConfig(tracker_cfg);
    const auto_aim::AimSignalFilterConfig aim_filter_cfg =
        auto_aim::loadAimFilterConfig(tracker_cfg);
    const bool fire_enabled = std::getenv("ULTRA_VISION_DISABLE_FIRE") == nullptr;
    const bool show_display = std::getenv("ULTRA_VISION_NO_DISPLAY") == nullptr;
    // Restrict inference to the projected chassis region while tracking.
    // OFF by default: it was measured not to speed anything up, because the
    // detector letterboxes the crop back to the network's fixed 640x640 input.
    // It becomes useful only together with a smaller-input model export.
    // 默认开：4 m 工况实测（20 s）可用观测 116 → 319 行、FIRING 段 12 → 15。
    // ULTRA_VISION_DYNAMIC_ROI=0 可关（A/B 用）。
    const bool dynamic_roi =
        std::getenv("ULTRA_VISION_DYNAMIC_ROI") == nullptr ||
        std::string(std::getenv("ULTRA_VISION_DYNAMIC_ROI")) != "0";
    // 旧配置把 min_fire_interval 放在 selector 段，这里保留兼容。
    double shooter_min_fire_interval = shooter_cfg.min_fire_interval;
    if (!tracker_cfg["tracker"]["shooter"] && tracker_cfg["tracker"]["selector"]) {
        shooter_min_fire_interval =
            tracker_cfg["tracker"]["selector"]["min_fire_interval"].as<double>(
                shooter_min_fire_interval);
    }
    const int max_control_lost_frames =
        auto_aim::loadMaxControlLostFrames(tracker_cfg);
    const double fire_angle_tolerance =
        auto_aim::loadFireAngleToleranceDegrees(tracker_cfg) * CV_PI / 180.0;

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

    // 真机与仿真共用同一套"检测前端 + 瞄准流水线"（见 armor_source / aim_pipeline）。
    // 这个入口只保留仿真专属的东西：TCP 帧源、曝光→到手的时间基、曲线/CSV/显示。
    // 仿真相机内参：分辨率固定时由 makeCameraMatrix() 按 FOV 重建，先占位声明。
    cv::Mat camera_matrix;
    cv::Mat dist_coeffs = cv::Mat::zeros(5, 1, CV_64F);

    auto_aim::ArmorSourceConfig source_cfg;
    source_cfg.classical = det_cfg;
    source_cfg.pnp = pnp_geometry;
    source_cfg.dynamic_roi = dynamic_roi;
#ifdef ULTRA_VISION_USE_OPENVINO
    source_cfg.use_neural = use_neural_detector;
    source_cfg.neural = auto_aim::loadNeuralDetectorConfig(detector_cfg, config_dir);
#endif
    auto_aim::ArmorSource armor_source(source_cfg, camera_matrix, dist_coeffs);

    auto_aim::AimPipelineConfig pipeline_cfg;
    pipeline_cfg.tracker = track_cfg;
    pipeline_cfg.selector = selector_cfg;
    pipeline_cfg.gimbal = gimbal_cfg;
    pipeline_cfg.shooter = shooter_cfg;
    pipeline_cfg.aim_filter = aim_filter_cfg;
    pipeline_cfg.command_rate_hz = gimbal_command_rate_hz;
    pipeline_cfg.max_control_lost_frames = max_control_lost_frames;
    pipeline_cfg.fire_angle_tolerance_deg = fire_angle_tolerance * 180.0 / CV_PI;
    pipeline_cfg.heartbeat_hz = 0.0;   // 仿真链路不需要心跳
    auto_aim::AimPipeline pipeline(
        pipeline_cfg, [&receiver](bool /*control*/, bool /*fire*/, double yaw, double yaw_vel,
                                  double yaw_acc, double pitch, double pitch_vel,
                                  double pitch_acc) {
            (void)yaw_vel; (void)yaw_acc; (void)pitch_vel; (void)pitch_acc;
            return receiver.sendGimbalCommand(yaw, pitch);
        });
    auto_aim::Tracker& tracker = pipeline.tracker();
    auto_aim::GimbalController& gimbal_controller = pipeline.controller();
    int previous_armor_id = -1;
    bool aim_ready = false;
    int aim_jump_rejections = 0;
    double aim_yaw_error = 0.0;
    double aim_pitch_error = 0.0;
    auto_aim::AimPipeline::Outcome aim_snapshots_for_curves;

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
                   "cmd_yaw,cmd_pitch,time_us,plate_id,state,"
                   // 检测到的四个角点像素（lb, lt, rt, rb）。为了能**离线**重解 PnP /
                   // 重投影做标定（不必每次改尺寸都重跑仿真）。
                   "u0,v0,u1,v1,u2,v2,u3,v3,source_time_us\n";
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
    // 分段耗时（ULTRA_VISION_PROF=1 时每秒打印一次）。只看 wait/work/display
    // 三个数不够：work 里混着 ROI 投影、帧拷贝、PnP、EKF、CSV 格式化、overlay
    // 绘制，不拆开就不知道该优化谁。和 sp_vision 一样先量再改。
    constexpr int kStageCount = 7;
    const char* const kStageNames[kStageCount] = {
        "roi", "submit", "pnp+ekf", "csv", "control", "overlay", "resize"};
    double prof_stage_sum[kStageCount] = {0.0};
    double prof_ekf_cpu_sum = 0.0;
    // 进帧速率（只看本机时钟，跨机器比对时间戳不可靠）：net_fps 是每秒真正
    // 拿到的帧数，empty 是 getFrame 超时返回空的次数。自瞄需要 ≥15~20 fps，
    // 掉到个位数时先看仿真器日志里的 [tcp] encoded_fps（那边才是源头）。
    int net_frames = 0;
    int net_empty = 0;
    // 关联层统计（每秒清零）：检测框数 / 被"模棱两可"丢掉的、被尺度闸门丢掉的、
    // 真正接受的观测数。`lost` 一直涨但 detections 很多时，看这三个数就知道
    // 是关联层把观测全拒了，还是检测器在给假框。
    // 曝光→收到的延迟：每秒取中位，喂给选择器当"观测年龄"的固定附加项。
    // ULTRA_VISION_SOURCE_DELAY=0 可关掉（A/B 用）。
    std::vector<double> source_delay_samples_us;
    double observation_source_delay = 0.0;
    const bool source_delay_enabled = [] {
        const char* value = std::getenv("ULTRA_VISION_SOURCE_DELAY");
        return value == nullptr || std::string(value) != "0";
    }();
    // 反"云台跑飞"守卫：瞄点相对当前下发角超过这个角度就不甩（见下发处的注释）。
    // ULTRA_VISION_AIM_JUMP_LIMIT_DEG 可调，0 = 关。
    const double aim_jump_limit = [] {
        const char* value = std::getenv("ULTRA_VISION_AIM_JUMP_LIMIT_DEG");
        const double degrees = value != nullptr ? std::atof(value) : 30.0;
        return degrees > 0.0 ? degrees * CV_PI / 180.0 : 1e9;
    }();
    int assoc_detections = 0;
    int assoc_ambiguous = 0;
    int assoc_nomatch = 0;
    int assoc_scale = 0;
    int assoc_accepted = 0;
    double net_recv_ms = 0.0;
    uint64_t net_first_local_us = 0;
    uint64_t net_prev_local_us = 0;
    const bool prof_stages = std::getenv("ULTRA_VISION_PROF") != nullptr;
    int missing_frames = 0;
    int key = -1;
    // 定时退出（ULTRA_VISION_TEST_SECONDS）：仿真回归脚本要"跑 N 秒然后干净退出"，
    // 这样 CSV 的 ofstream 正常析构、最后几行不会因为 SIGTERM 丢掉。
    // 与 auto_buff 的 ULTRA_VISION_RUNE_TEST_SECONDS 同一个套路。
    const char* test_seconds_env = std::getenv("ULTRA_VISION_TEST_SECONDS");
    const double test_seconds = test_seconds_env != nullptr ? std::atof(test_seconds_env) : 0.0;
    const auto loop_start_time = std::chrono::steady_clock::now();

    // 实时曲线（可视化调试）：
    //   不设 ULTRA_VISION_CURVES → **自动**：有窗口、或要落曲线 CSV 时才采；
    //   ULTRA_VISION_CURVES=1/0 → 强制开 / 关。
    // 采一条曲线要算观测角、观测距离和"瞄点像素误差"（一次投影），实测 ~1 ms/帧；
    // 无显示又不要 CSV 的时候（回归全走这条）这份开销纯属白给，所以默认不采。
    const char* curves_env = std::getenv("ULTRA_VISION_CURVES");
    const bool curves_enabled = curves_env != nullptr
        ? std::string(curves_env) != "0"
        : (show_display || std::getenv("ULTRA_VISION_CURVE_CSV") != nullptr);
    auto_aim::visualization::CurvePlotter curves(300);
    // panel 号决定"哪两条画在同一个坐标系里比较"：同 panel 共用纵轴。
    // yaw 命令/观测一格、pitch 命令/观测一格、像素误差一格、距离观测/估计一格。
    curves.addCurve("aim_yaw_cmd [deg]", cv::Scalar(80, 220, 255), 90.0, 0);
    curves.addCurve("armor_yaw_obs [deg]", cv::Scalar(80, 255, 80), 90.0, 0);
    curves.addCurve("aim_pitch_cmd [deg]", cv::Scalar(255, 180, 80), 60.0, 1);
    curves.addCurve("armor_pitch_obs [deg]", cv::Scalar(80, 180, 255), 60.0, 1);
    curves.addCurve("aim_px_err [px]", cv::Scalar(255, 80, 255), 120.0, 2);
    curves.addCurve("est_dist [m]", cv::Scalar(200, 200, 200), 12.0, 3);
    curves.addCurve("obs_dist [m]", cv::Scalar(120, 120, 255), 12.0, 3);
    if (const char* curve_csv = std::getenv("ULTRA_VISION_CURVE_CSV")) {
        curves.openCsv(curve_csv);
    }

    while (key != 27 && key != 'e' && key != 'E') {
        if (test_seconds > 0.0 &&
            std::chrono::duration<double>(std::chrono::steady_clock::now() - loop_start_time)
                    .count() >= test_seconds) {
            break;
        }
        const auto loop_start = std::chrono::steady_clock::now();
        uint64_t frame_sequence = 0;
        uint64_t frame_local_timestamp_us = 0;
        // 仿真器给的**曝光时刻**（sim 时钟）。与 local 时间戳相差"网络+处理+节流"
        // 的整条延迟（实测 ~57 ms），用真值做回归时必须按它对齐，否则那点延迟在
        // 云台转动时会以"相机系位移"的形式混进观测误差里。
        uint64_t frame_source_timestamp_us = 0;
        frame = receiver.getFrame(
            &frame_sequence, &frame_source_timestamp_us, &frame_local_timestamp_us);
        const auto after_frame = std::chrono::steady_clock::now();
        if (frame.empty()) {
            ++missing_frames;
            ++net_empty;
            if (missing_frames == 1 || missing_frames % 30 == 0) {
                std::cerr << "Waiting for simulator frames..." << std::endl;
            }
            key = cv::waitKey(1);
            continue;
        }
        missing_frames = 0;
        // 曝光→到手 的延迟采样（两边都是 Unix 时钟；Mac/NUC 有 NTP，
        // 常数级的钟差不影响"这段延迟有多大"的结论，只把绝对值平移）。
        // 这一项就是"观测年龄"里**原来被漏掉**的一段：检测用的是仿真端
        // 曝光时刻的像，而我们原来只用"本机收到的时刻"当观测时间。
        if (frame_source_timestamp_us > 0 && frame_local_timestamp_us > frame_source_timestamp_us) {
            source_delay_samples_us.push_back(
                static_cast<double>(frame_local_timestamp_us - frame_source_timestamp_us));
        }
        // 进帧速率统计（本机时钟，跨机器时间戳不可比）
        ++net_frames;
        net_recv_ms += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - loop_start).count();
        if (net_first_local_us == 0) {
            net_first_local_us = frame_local_timestamp_us;
        } else {
            net_prev_local_us = frame_local_timestamp_us;
        }

        auto stage_mark = after_frame;
        double stage_ms[kStageCount] = {0.0};
        double stage_ekf_cpu_ms = 0.0;
        const auto lap = [&stage_mark, &stage_ms](int index) {
            const auto now = std::chrono::steady_clock::now();
            stage_ms[index] +=
                std::chrono::duration<double, std::milli>(now - stage_mark).count();
            stage_mark = now;
        };
        // 同一段再量一次**线程 CPU 时间**：墙钟远大于 CPU 时间就说明是"被抢 CPU/
        // 换出"，不是算法真的算了那么久（机器忙的时候最容易误判成算法变慢）。
        const auto thread_cpu_ms = [] {
            timespec ts{};
            clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
            return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
        };
        double ekf_cpu_mark = thread_cpu_ms();

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
            // 进图尺寸一变就把"实际处理尺寸 + 现算的内参"打出来：仿真器给 640x480
            // 还是 1440x1080，PnP 的尺度和检出率完全是两回事，必须看一眼就知道。
            // 期望值来自 configs/camera.yaml 的 simulator.max_processing_width，
            // 它和实车海康的 1440x1080 / 41°(f≈1447) 是同一套标定。
            const bool size_mismatch =
                max_processing_width > 0 && image.cols < max_processing_width;
            std::cout << "[input] " << image.cols << "x" << image.rows
                      << " fov=" << simulator_fov_degrees
                      << " f=" << camera_matrix.at<double>(0, 0)
                      << " cx=" << camera_matrix.at<double>(0, 2)
                      << " cy=" << camera_matrix.at<double>(1, 2)
                      << " (标定尺寸 " << max_processing_width << " 宽)"
                      << (size_mismatch ? "  ← 偏小！" : "") << std::endl;
            if (size_mismatch) {
                std::cerr
                    << "[input] 警告：仿真帧只有 " << image.cols << "x" << image.rows
                    << "，比 configs/camera.yaml 标定的 "
                    << max_processing_width
                    << " 宽小 —— 靶面像素按面积缩水 "
                    << (static_cast<double>(max_processing_width) / image.cols) *
                           (static_cast<double>(max_processing_width) / image.cols)
                    << " 倍，检测率/PnP 尺度都会和标定对不上。\n"
                    << "[input]       仿真器要用匹配的采集尺寸启动：\n"
                    << "[input]         simulator_system/simulator/run_host.sh"
                       "（默认已是 1440x1080）\n"
                    << "[input]        或 DAEDALUS_CAPTURE_WIDTH=1440"
                       " DAEDALUS_CAPTURE_HEIGHT=1080，"
                       "或 tools/auto_aim_sim_test.sh --capture 1440x1080"
                    << std::endl;
            }
        }
        // UV 观测需要同一时刻的内参（尺寸变化时重建过，所以每帧同步一次）。
        if (camera_matrix.rows == 3 && camera_matrix.cols == 3) {
            tracker.setCameraIntrinsics(camera_matrix.at<double>(0, 0),
                                        camera_matrix.at<double>(1, 1),
                                        camera_matrix.at<double>(0, 2),
                                        camera_matrix.at<double>(1, 2));
        }

        // ------------------- 5. 检测（共用组件：传统/神经网络 + 动态 ROI + PnP）----
        armor_source.setIntrinsics(camera_matrix, dist_coeffs);
        const auto_aim::ArmorSourceResult source_result = armor_source.update(
            image, frame_sequence, frame_local_timestamp_us,
            tracker.getEstimatedArmorPositions(), tracker.getState(),
            /*image_will_be_modified=*/show_display);
        const std::vector<auto_aim::Light>& lights = source_result.lights;
        const std::vector<auto_aim::Armor>& armors = source_result.armors;
        const int pnp_count = source_result.pnp_count;
        bool fresh_detection = source_result.fresh;
        lap(2);


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
                << static_cast<int>(tracker.getState());
            if (observed.Points_2D.size() >= 4) {
                for (int i = 0; i < 4; ++i) {
                    observation_recorder << ',' << observed.Points_2D[static_cast<std::size_t>(i)].x
                                         << ',' << observed.Points_2D[static_cast<std::size_t>(i)].y;
                }
            } else {
                for (int i = 0; i < 8; ++i) observation_recorder << ",nan";
            }
            observation_recorder << ',' << frame_source_timestamp_us << '\n';
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
        lap(3);

        // ------------------- 6. 估计 → 选板 → 瞄准 → 开火（共用流水线）--------
        // 和真机 node.cpp 跑的是同一份实现：估计、提前量、云台轨迹、开火判据、
        // 反跑飞守卫都在 pipeline 里，两个入口只负责各自的 IO 与显示。
        {
            // 观测时刻：仿真里检测用的是**曝光那一刻**的像，所以要把
            // "曝光→到手"的延迟从收到帧的时刻里减掉（真机串口链路是同步的，直接用帧时刻）。
            const bool source_delay_usable = source_delay_enabled &&
                observation_source_delay > 0.0005 && observation_source_delay < 0.120;
            const uint64_t source_delay_us = source_delay_usable
                ? static_cast<uint64_t>(observation_source_delay * 1e6) : 0;
            const double observation_time =
                auto_aim::StandardClock::secondsFromUs(source_result.timestamp_us) -
                source_delay_us / 1e6;

            auto_aim::CameraPose detection_pose;
            const auto detection_snapshot =
                gimbal_controller.snapshotAt(source_result.timestamp_us - source_delay_us);
            detection_pose.valid = detection_snapshot.valid;
            detection_pose.timestamp_valid = detection_snapshot.valid;
            detection_pose.timestamp = observation_time;
            detection_pose.yaw = detection_snapshot.valid
                ? detection_snapshot.yaw : commanded_yaw;
            detection_pose.pitch = detection_snapshot.valid
                ? detection_snapshot.pitch : commanded_pitch;

            auto_aim::AimPipeline::FrameInput frame;
            frame.armors = source_result.armors;
            frame.fresh = source_result.fresh;
            frame.observation_time = observation_time;
            frame.now = auto_aim::StandardClock::secondsFromUs(frame_local_timestamp_us);
            frame.pose = detection_pose;
            frame.allow_control = true;         // 仿真里没有下位机，始终由我们控制
            frame.fire_enabled = fire_enabled;
            // 关联层统计：从 tracker 读（组件内部统计，见 assoc: 日志）
            if (source_result.fresh) {
                assoc_detections += static_cast<int>(source_result.armors.size());
                assoc_nomatch += tracker.getDroppedNoMatch();
                assoc_ambiguous += tracker.getDroppedAmbiguous();
                assoc_scale += tracker.getDroppedByScale();
                assoc_accepted += tracker.getAcceptedObservations();
            }
            const auto outcome = pipeline.update(frame);

            last_decision = outcome.decision;
            aim_ready = outcome.aim_ready;
            aim_yaw_error = outcome.aim_yaw_error;
            aim_pitch_error = outcome.aim_pitch_error;
            if (outcome.decision.valid) ++aim_valid_updates;
            if (outcome.aim_ready) ++aim_settled_updates;
            if (outcome.held_for_jump) ++aim_jump_rejections;
            if (outcome.fired_now) {
                receiver.sendFireCommand();
                std::cout << cv::format(
                    "FIRE state=FIRING omega=%+.2f armor=%d lead=%.3fs "
                    "yaw_err=%+.1fdeg pitch_err=%+.1fdeg",
                    tracker.getOmega(), outcome.decision.armor_id, outcome.decision.lead_time,
                    outcome.aim_yaw_error * 180.0 / CV_PI,
                    outcome.aim_pitch_error * 180.0 / CV_PI)
                          << std::endl;
            }
            if (outcome.decision.valid && previous_armor_id >= 0 &&
                outcome.decision.armor_id != previous_armor_id) {
                ++plate_switches;
            }
            previous_armor_id = outcome.decision.valid ? outcome.decision.armor_id : -1;
            shooter_state = outcome.fire ? auto_aim::ShooterState::FIRING
                                         : auto_aim::ShooterState::IDLE;
            shooter_error = auto_aim::ShooterErrorReason::NONE;
            aim_snapshots_for_curves = outcome;
            last_detection_time = observation_time;

            // selector.csv：保留原口径（时间戳同样是**观测时刻**）
            if (selector_recorder && outcome.decision.valid) {
                selector_recorder << std::fixed << std::setprecision(6)
                    << frame_local_timestamp_us << ','
                    << tracker.getTargetCenterWorldArray()[0] << ','
                    << tracker.getTargetCenterWorldArray()[1] << ','
                    << tracker.getTargetCenterWorldArray()[2] << ','
                    << tracker.getYawWorld() << ','
                    << tracker.getOmega() << ','
                    << outcome.decision.armor_id << ','
                    << outcome.decision.armor_position[0] << ','
                    << outcome.decision.armor_position[1] << ','
                    << outcome.decision.armor_position[2] << ','
                    << outcome.decision.target_yaw << ','
                    << outcome.decision.target_pitch << ','
                    << outcome.decision.lead_time << ','
                    << outcome.decision.target_yaw_velocity << ','
                    << outcome.decision.target_pitch_velocity << ','
                    << outcome.decision.gimbal_time << ','
                    << outcome.decision.handoff_gain << ','
                    << static_cast<int>(outcome.decision.in_fire_window) << '\n';
            }
        }

        lap(4);

        // 这一整段是**纯绘制**（只读 tracker 状态 + 投影 + 画框画字）。没有窗口时
        // 原来照画不误，实测 0.4~0.8 ms/帧；省掉它之后 work 只剩帧拷贝那一项。
        if (show_display &&
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
        } else if (show_display) {
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
                      << ", det_fps: " << (armor_source.neuralEnabled()
                             ? armor_source.neuralFps() : fps_frames / elapsed)
                      << ", det_ms: " << armor_source.neuralLatencyMs()
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
                if (prof_stages) {
                    std::cout << "  stage ms:";
                    double accounted = 0.0;
                    for (int stage = 0; stage < kStageCount; ++stage) {
                        const double value = prof_stage_sum[stage] / n;
                        accounted += value;
                        std::cout << ' ' << kStageNames[stage] << '=' << value;
                    }
                    std::cout << " | 合计 " << accounted << " / work "
                              << prof_work_ms / n
                              << " | pnp+ekf CPU " << prof_ekf_cpu_sum / n << std::endl;
                }
            }
            // 进帧速率：自瞄需要 ≥15~20 fps；掉到个位数先看仿真器日志的
            // [tcp] encoded_fps（源头在那边），再决定是换机器跑仿真还是查网络。
            if (net_frames > 0) {
                const double span = net_prev_local_us > net_first_local_us
                    ? (net_prev_local_us - net_first_local_us) / 1e6 : 0.0;
                const double net_fps = span > 1e-3
                    ? (net_frames - 1) / span : static_cast<double>(net_frames);
                if (!source_delay_samples_us.empty()) {
                    std::vector<double> sorted = source_delay_samples_us;
                    std::sort(sorted.begin(), sorted.end());
                    observation_source_delay =
                        sorted[sorted.size() / 2] / 1e6;   // 秒，供下一秒的观测时间用
                    double worst = sorted.back() / 1000.0;
                    std::cout << "  source_delay: median=" << observation_source_delay * 1000.0
                              << "ms max=" << worst << "ms (曝光→到手，已计入观测年龄"
                              << (source_delay_enabled ? "" : "，当前被 ULTRA_VISION_SOURCE_DELAY=0 关掉")
                              << ")" << std::endl;
                    if (source_delay_enabled &&
                        (observation_source_delay <= 0.0005 || observation_source_delay >= 0.120)) {
                        std::cerr << "[net] source_delay=" << observation_source_delay * 1000.0
                                  << " ms 超出合理区间(0~120)，**不用于补偿** —— "
                                     "两台机器的时钟大概率没对齐（NTP 没同步/差时区）；"
                                     "先对齐时钟，或 ULTRA_VISION_SOURCE_DELAY=0 关掉这一项。"
                                  << std::endl;
                    }
                }
                source_delay_samples_us.clear();
                if (aim_jump_rejections > 0) {
                    std::cout << "  aim_guard: 按住 " << aim_jump_rejections
                              << " 次（瞄点跳变超过 " << aim_jump_limit * 180.0 / CV_PI
                              << "°）" << std::endl;
                }
                aim_jump_rejections = 0;
                std::cout << "  assoc: det=" << assoc_detections
                          << " nomatch=" << assoc_nomatch
                          << " ambiguous=" << assoc_ambiguous
                          << " scale=" << assoc_scale
                          << " accepted=" << assoc_accepted << "\n";
                std::cout << "  net: fps=" << net_fps
                          << " wait+recv+decode=" << net_recv_ms / net_frames << "ms"
                          << " empty=" << net_empty << std::endl;
                if (net_fps < 12.0) {
                    std::cerr << "[net] 进帧只有 " << net_fps
                              << " fps（自瞄要 ≥15~20）。先看**仿真器**那边的日志："
                              << "[tcp] encoded_fps= 若是低值，瓶颈在渲染/编码（软件渲染最常见，"
                              << "把仿真器放到有 GPU 的机器上用 ULTRA_VISION_SIM_HOST 连过去）；"
                              << "若 encoded_fps 正常而这里低，就是网络/本机解码。"
                              << std::endl;
                }
            }
            assoc_detections = assoc_nomatch = assoc_ambiguous = assoc_scale = assoc_accepted = 0;
            net_frames = 0;
            net_empty = 0;
            net_recv_ms = 0.0;
            net_first_local_us = 0;
            net_prev_local_us = 0;
            prof_wait_ms = prof_work_ms = prof_display_ms = 0.0;
            for (int stage = 0; stage < kStageCount; ++stage) prof_stage_sum[stage] = 0.0;
            prof_ekf_cpu_sum = 0.0;
            fps_start = now;
            fps_frames = 0;
            aim_valid_updates = 0;
            aim_settled_updates = 0;
            plate_switches = 0;
        }

        lap(5);
        // 这一下采样**只为调试窗口**：ULTRA_VISION_NO_DISPLAY=1（回归全走这条路）时
        // 原来照样每帧做一次 1440x1080 → 640x480 的 INTER_AREA，实测 15~18 ms/帧，
        // 比整套 PnP+EKF+控制（~0.5 ms）贵 30 倍。没有窗口就整段跳过。
        if (show_display) {
            cv::resize(image, image, cv::Size(640, 480), 0.0, 0.0, cv::INTER_AREA);
        }
        lap(6);
        const auto before_display = std::chrono::steady_clock::now();

        // ---- 实时数学曲线：装甲板位姿 vs 瞄准方向（sp_vision 式的调试习惯）--------
        // 图像只能说明"这一帧对上了"；滞后/超前/跳变/偏置只在时间序列上看得见。
        // 四条量：命令角 vs 观测角（yaw、pitch）、瞄点像素误差、距离（观测 vs 估计）。
        if (curves_enabled) {
            const auto& observed_armor = tracker.getLastObservedArmor();
            double observed_yaw = std::nan("");
            double observed_pitch = std::nan("");
            double observed_distance = std::nan("");
            double aim_pixel_error = std::nan("");
            if (observed_armor.solve_result && observed_armor.tvec.rows == 3) {
                const double x = observed_armor.tvec.at<double>(0);
                const double y = observed_armor.tvec.at<double>(1);
                const double z = observed_armor.tvec.at<double>(2);
                observed_yaw = std::atan2(x, z) * 180.0 / CV_PI;
                observed_pitch = std::atan2(-y, std::hypot(x, z)) * 180.0 / CV_PI;
                observed_distance = std::sqrt(x * x + y * y + z * z);
                if (observed_armor.Points_2D.size() >= 4 && camera_matrix.rows == 3) {
                    cv::Point2f center(0.0f, 0.0f);
                    for (const cv::Point2f& point : observed_armor.Points_2D) center += point;
                    center *= 0.25f;
                    const cv::Mat estimated_center = tracker.getTargetCenter();
                    cv::Point2f aim_pixel;
                    if (auto_aim::projectPoint(estimated_center, camera_matrix, dist_coeffs,
                                               aim_pixel)) {
                        aim_pixel_error = cv::norm(aim_pixel - center);
                    }
                }
            }
            const cv::Mat estimated_center = tracker.getTargetCenter();
            const double estimated_distance = cv::norm(estimated_center);
            curves.pushAll({commanded_yaw * 180.0 / CV_PI, observed_yaw,
                            commanded_pitch * 180.0 / CV_PI, observed_pitch, aim_pixel_error,
                            estimated_distance, observed_distance});
        }
        // HighGUI costs more than the whole processing pipeline (measured
        // ~15 ms/frame vs ~0.9 ms of work), and a robot has no display at all.
        // Keep it for debugging, allow turning it off to see the real rate.
        // 检测图与曲线合成**一个**窗口（竖直：图在上、曲线在下）：原来两个窗口
        // 要来回看，而且曲线窗里同一 panel 的两条曲线图例画在同一个坐标上（叠字）。
        if (show_display) {
            const cv::Mat display = curves_enabled ? curves.renderStacked(image) : image;
            cv::imshow("Ultra Vision", display);
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
        for (int stage = 0; stage < kStageCount; ++stage) {
            prof_stage_sum[stage] += stage_ms[stage];
        }
        prof_ekf_cpu_sum += stage_ekf_cpu_ms;
    }

    return 0;
}
