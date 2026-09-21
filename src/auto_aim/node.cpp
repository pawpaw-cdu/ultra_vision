#include <atomic>
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
#include "control/aim_pipeline.hpp"
#include "perception/armor_source.hpp"
#include "io/gimbal/gimbal.hpp"
#if defined(ULTRA_VISION_USE_HIK_CAMERA)
#include "io/camera/HikCamera.hpp"
using CameraType = rm_ultra::HikCamera;   // 海康（MVS SDK）
#else
#include "io/camera/GalaxyCamera.hpp"
using CameraType = rm_ultra::GalaxyCamera; // 大恒（Galaxy SDK）
#endif

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

    // ------------------- 1.5 下位机链路 + 控制链配置 -------------------
    // 实车这一侧和仿真入口用**同一套** config_loader 与控制类（上一轮统一过），
    // 区别只有两条输入来源：相机来自 SDK，姿态/弹速来自下位机串口帧。
    const YAML::Node serial_file = YAML::LoadFile(config_path("serial.yaml"));
    const io::SerialConfig serial_cfg = auto_aim::loadSerialConfig(serial_file);
    const bool use_imu_world_frame =
        (serial_file["serial"] ? serial_file["serial"]["use_imu_world_frame"].as<bool>(false)
                               : false);
    // ---- 共用的检测前端 + 瞄准流水线（和仿真入口同一份实现）----
    auto_aim::ArmorSourceConfig source_cfg;
    source_cfg.classical = det_cfg;
    source_cfg.pnp = pnp_geometry;
    source_cfg.dynamic_roi = std::getenv("ULTRA_VISION_DYNAMIC_ROI") == nullptr ||
        std::string(std::getenv("ULTRA_VISION_DYNAMIC_ROI")) != "0";
#ifdef ULTRA_VISION_USE_OPENVINO
    source_cfg.use_neural = auto_aim::neuralDetectorEnabled(detector_file);
    source_cfg.neural = auto_aim::loadNeuralDetectorConfig(detector_file, config_dir);
#endif

    auto_aim::AimPipelineConfig pipeline_cfg;
    pipeline_cfg.tracker = track_cfg;
    pipeline_cfg.selector = auto_aim::loadSelectorConfig(tracker_file);
    pipeline_cfg.gimbal = auto_aim::loadGimbalConfig(tracker_file);
    pipeline_cfg.shooter = auto_aim::loadShooterConfig(tracker_file);
    pipeline_cfg.aim_filter = auto_aim::loadAimFilterConfig(tracker_file);
    pipeline_cfg.command_rate_hz = auto_aim::loadGimbalCommandRateHz(tracker_file);
    pipeline_cfg.max_control_lost_frames = auto_aim::loadMaxControlLostFrames(tracker_file);
    pipeline_cfg.fire_angle_tolerance_deg =
        auto_aim::loadFireAngleToleranceDegrees(tracker_file);
    pipeline_cfg.aim_jump_limit_deg = [] {
        const char* value = std::getenv("ULTRA_VISION_AIM_JUMP_LIMIT_DEG");
        return value != nullptr ? std::atof(value) : 30.0;
    }();
    pipeline_cfg.heartbeat_hz = 20.0;   // 真机链路必须有心跳（mode=0）

    // 下位机链路 + 控制权：只有 C 板把 mode 置成"自瞄"且链路在线，才允许我们接管。
    const bool fire_enabled = std::getenv("ULTRA_VISION_DISABLE_FIRE") == nullptr;
    io::Gimbal gimbal(serial_cfg);
    std::cout << "[io] 下位机链路: device=" << serial_cfg.device
              << " connected=" << (gimbal.connected() ? "yes" : "no")
              << " imu_world_frame=" << (use_imu_world_frame ? "on" : "off") << std::endl;

    auto pipe_send = [&gimbal](bool control, bool fire, double yaw, double yaw_vel,
                               double yaw_acc, double pitch, double pitch_vel, double pitch_acc) {
        return gimbal.send(control, fire, yaw, yaw_vel, yaw_acc, pitch, pitch_vel, pitch_acc);
    };
    auto_aim::AimPipeline pipeline(pipeline_cfg, pipe_send);
    auto_aim::ArmorSource armor_source(source_cfg, camera_matrix, dist_coeffs);
    // 相机→云台外参（手眼标定结果）。没标定过就是单位阵/零平移，与仿真口径一致。
    const auto_aim::CameraExtrinsics extrinsics =
        auto_aim::loadCameraExtrinsics(camera_file, camera_name);

    // ------------------- 2. Initialize the Galaxy camera -------------------
    CameraType camera;   // 由 ULTRA_VISION_USE_HIK_CAMERA 决定（见 include 处）
    const bool camera_ready = serial_number.empty()
        ? camera.init("", device_index)
        : camera.init(serial_number, device_index);
    if (!camera_ready) {
        std::cerr << "Failed to init camera (" << camera_name << ")!" << std::endl;
        return -1;
    }
    std::cout << "Camera initialized successfully." << std::endl;
#if defined(ULTRA_VISION_USE_HIK_CAMERA)
    // 曝光/增益按 configs/camera.yaml 的 hikcamera 段设成固定值。自动曝光在比赛里
    // 会被靶面灯光带跑、暗光下收敛要好几秒（实测只跑 0.3 s 时画面全黑），所以默认关。
    if (const YAML::Node hik = camera_file["hikcamera"]) {
        const bool auto_exposure = hik["auto_exposure"].as<bool>(false);
        const bool auto_gain = hik["auto_gain"].as<bool>(false);
        const double exposure_ms = hik["exposure_ms"].as<double>(6.0);
        const double gain = hik["gain"].as<double>(12.0);
        if (auto_exposure) {
            camera.setAutoExposure(true);
        } else {
            camera.setExposureTime(exposure_ms * 1000.0);   // 驱动按微秒
        }
        if (auto_gain) {
            camera.setAutoGain(true);
        } else {
            camera.setGain(gain);
        }
        std::cout << "[io] 曝光 " << (auto_exposure ? std::string("auto") : std::to_string(exposure_ms) + "ms")
                  << " 增益 " << (auto_gain ? std::string("auto") : std::to_string(gain) + "dB")
                  << std::endl;
    }
#endif

    auto_aim::CAMERA_MATRIX = camera_matrix;
    auto_aim::DIST_COEFFS = dist_coeffs;

    // ------------------- 3. Whole-chassis estimator -------------------
    // One tracker for the whole vehicle, matching the simulator entry. Until
    // the gimbal link reports absolute angles, CameraPose keeps its identity
    // default, so the estimate is expressed in the camera frame. Calling
    // tracker.setCameraPose() before update() is the only change needed to run
    // the same estimator in the gimbal/world frame once that feedback exists.
    auto_aim::Tracker& tracker = pipeline.tracker();   // 估计器归流水线所有
    // UV 观测（cfg.uv_observation）用：真实相机的内参在一次运行里固定。
    if (camera_matrix.rows == 3 && camera_matrix.cols == 3) {
        tracker.setCameraIntrinsics(camera_matrix.at<double>(0, 0),
                                    camera_matrix.at<double>(1, 1),
                                    camera_matrix.at<double>(0, 2),
                                    camera_matrix.at<double>(1, 2));
    }

    cv::Mat frame;
    cv::Mat image;

    uint64_t sent_frames_previous_ = 0;
    uint64_t frame_sequence = 0;
    auto_aim::TargetDecision last_decision;
    bool aim_ready = false;
    double aim_yaw_error = 0.0;
    double aim_pitch_error = 0.0;
    bool allow_control_now = false;
    bool last_source_fresh_ = false;
    // 无目标时的心跳计时（见下发处注释）
    auto last_heartbeat = std::chrono::steady_clock::now();
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

        // ---- IO 层自检：确认海康给的是"正常图像"而不是黑帧/通道错 ----
        // ULTRA_VISION_DUMP_FRAME=/tmp/frame.png 落一帧；每 30 帧打印通道均值。
        // 正常室内画面 B/G/R 同量级；若 G 远高于 R/B（发绿）或 R/B 互换，
        // 说明 Bayer 解码/像素格式不对（相机报 BayerRG8，我们请求转 BGR8_Packed）。
        if (const char* dump_path = std::getenv("ULTRA_VISION_DUMP_FRAME")) {
            static bool dumped = false;
            if (!dumped) {
                dumped = true;
                cv::imwrite(dump_path, frame);
                std::cout << "[io] dumped " << dump_path << " size=" << frame.cols << "x"
                          << frame.rows << std::endl;
            }
        }
        {
            static int probe_count = 0;
            if (++probe_count % 30 == 1) {
                const cv::Scalar mean = cv::mean(frame);
                std::cout << "[io] frame " << frame.cols << "x" << frame.rows << " mean B/G/R = "
                          << mean[0] << "/" << mean[1] << "/" << mean[2] << std::endl;
            }
        }

        // The frame is processed at its native resolution. Downscaling here
        // would require scaling the intrinsics by the same factor, otherwise
        // every PnP distance is wrong.
        image = frame;

        // ------------------- 5. 检测（共用组件：传统/神经网络 + 动态 ROI + PnP）----
        armor_source.setIntrinsics(auto_aim::CAMERA_MATRIX, auto_aim::DIST_COEFFS);
        const auto_aim::ArmorSourceResult source_result = armor_source.update(
            image, ++frame_sequence, auto_aim::StandardClock::nowUs(),
            tracker.getEstimatedArmorPositions(), tracker.getState(),
            /*image_will_be_modified=*/true);   // 本入口会在这张图上画框
        const std::vector<auto_aim::Light>& lights = source_result.lights;
        const std::vector<auto_aim::Armor>& armors = source_result.armors;
        const int pnp_count = source_result.pnp_count;
        last_source_fresh_ = source_result.fresh;
        // 注意：**不要**在没有新推理结果的帧上提前 continue —— 那会让画面隔帧丢框
        // （相机 80~100 fps、NN 25~40 fps，正好差拍频闪）。异步只影响"要不要更新
        // 估计器"，绘制与决策每帧都跑，用估计器里的状态画。
        const double observation_time =
            auto_aim::StandardClock::secondsFromUs(source_result.timestamp_us);

        // ------------------- 6. 下位机姿态 + 整车估计 -------------------
        // 观测是在**相机系**里解出来的，要放进底盘/世界系，必须知道拍照那一刻的
        // 相机姿态：yaw/pitch 用下位机回传的云台绝对角（它的 gimbal_yaw_motor.
        // absolute_angle），base_to_world 可选地用 IMU 四元数（底盘小陀螺时估计
        // 不会跟着底盘转）。这一步不做，估计就永远活在相机系里 —— 底盘一转就漂。
        const double timestamp = auto_aim::StandardClock::nowSeconds();
        const io::GimbalState gimbal_state = gimbal.state();
        auto_aim::CameraPose camera_pose;
        camera_pose.valid = gimbal_state.valid;
        camera_pose.timestamp_valid = gimbal_state.valid;
        camera_pose.timestamp = timestamp;
        camera_pose.yaw = gimbal_state.yaw;
        camera_pose.pitch = gimbal_state.pitch;
        camera_pose.camera_to_gimbal = extrinsics.rotation;
        camera_pose.camera_to_gimbal_translation = extrinsics.translation;
        if (use_imu_world_frame) {
            io::ImuSample imu;
            if (gimbal.imuAt(std::chrono::steady_clock::now(), imu)) {
                // world<-base：四元数 (w,x,y,z) → 旋转矩阵（列主序存进 base_to_world）。
                const double w = imu.w, x = imu.x, y = imu.y, z = imu.z;
                camera_pose.base_to_world = {{
                    {{1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)}},
                    {{2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)}},
                    {{2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)}},
                }};
            }
        }
        ++processed_frames;

        // ------------------- 6.5 估计 → 选板 → 瞄准 → 开火 -------------------
        // 全部交给共用流水线：真机和仿真跑的是同一份决策代码。
        camera_pose.timestamp = observation_time;
        auto_aim::AimPipeline::FrameInput frame_input;
        frame_input.armors = source_result.armors;
        frame_input.fresh = source_result.fresh;
        frame_input.observation_time = observation_time;
        frame_input.now = auto_aim::StandardClock::nowSeconds();
        frame_input.pose = camera_pose;
        frame_input.bullet_speed = gimbal_state.bullet_speed;
        allow_control_now = gimbal.mode() == io::GimbalMode::AUTO_AIM && gimbal.connected();
        frame_input.allow_control = allow_control_now;
        frame_input.fire_enabled = fire_enabled;
        const auto outcome = pipeline.update(frame_input);
        last_decision = outcome.decision;
        aim_ready = outcome.aim_ready;
        aim_yaw_error = outcome.aim_yaw_error;
        aim_pitch_error = outcome.aim_pitch_error;
        if (outcome.fired_now) {
            std::cout << cv::format(
                "FIRE mode=2 armor=%d lead=%.3fs yaw_err=%+.1fdeg pitch_err=%+.1fdeg",
                outcome.decision.armor_id, outcome.decision.lead_time,
                outcome.aim_yaw_error * 180.0 / CV_PI,
                outcome.aim_pitch_error * 180.0 / CV_PI)
                      << std::endl;
        }

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
            const auto gimbal_stats = gimbal.stats();
            const uint64_t sent_frames_last_second =
                gimbal_stats.sent_frames - sent_frames_previous_;
            sent_frames_previous_ = gimbal_stats.sent_frames;
            const auto now_state = gimbal.state();
            std::cout << "Processed FPS: " << processed_frames / elapsed
                      << ", armors: " << armors.size()
                      << ", PnP: " << pnp_count
                      << ", tracker: " << trackerStateName(tracker.getState())
                      << ", lost: " << tracker.getLostCount()
                      << ", omega: " << tracker.getOmega()
                      // 下位机链路：mode 是它请求的模式（1=自瞄才允许我们接管），
                      // recv 是实际收到的帧率（掉到几十 Hz 以下就要查线/USB）
                      << ", mcu: " << io::gimbalModeName(gimbal.mode())
                      << " yaw=" << now_state.yaw << " pitch=" << now_state.pitch
                      << " v=" << now_state.bullet_speed
                      << " recv=" << gimbal_stats.fps << "Hz"
                      // nn=推理速率/延迟，fresh 表示本帧有没有新观测（异步常态下会交替 0/1，
                      // 但画面不该因此丢框 —— 显示画的是估计器里的状态）
                      << " nn=" << armor_source.neuralFps() << "Hz/"
                      << armor_source.neuralLatencyMs() << "ms"
                      << " fresh=" << (last_source_fresh_ ? 1 : 0)
                      << " sent=" << sent_frames_last_second << "Hz"
                      << " crc_err=" << gimbal_stats.crc_errors
                      << ", ctrl=" << (allow_control_now ? 1 : 0)
                      << " aim_err=" << aim_yaw_error * 180.0 / CV_PI << "/"
                      << aim_pitch_error * 180.0 / CV_PI << "deg"
                      << ", fire=" << (outcome.fire ? 1 : 0) << std::endl;
            report_time = now;
            processed_frames = 0;
        }
    }

    return 0;
}
