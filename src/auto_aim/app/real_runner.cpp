#include "app/real_runner.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>

#include "common/types.hpp"
#include "common/standard_clock.hpp"
#include "config_loader.hpp"
#include "app/config_paths.hpp"
#include "app/pipeline_config.hpp"
#include "visualization/projection.hpp"
#include "perception/detector.hpp"
#include "perception/pnp_solver.hpp"
#include "Kalman/tracker.hpp"
#include "control/aim_pipeline.hpp"
#include "control/gimbal_command_frame.hpp"
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

namespace auto_aim
{
    namespace app
    {
        RealRunner::RealRunner(std::string config_dir)
            : config_dir_(std::move(config_dir))
        {
        }
    } // namespace app
} // namespace auto_aim

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

int auto_aim::app::RealRunner::run()
{
    const std::string& config_dir = config_dir_;
    const auto config_path = [&](const std::string& name) {
        return auto_aim::app::configPath(config_dir, name);
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
    auto_aim::TrackerConfig track_cfg = auto_aim::loadTrackerConfig(tracker_file);
    const auto_aim::PnpGeometry pnp_geometry =
        auto_aim::loadPnpGeometry(tracker_file);

    // ------------------- 1.5 下位机链路 + 控制链配置 -------------------
    const YAML::Node serial_file = YAML::LoadFile(config_path("serial.yaml"));
    const io::SerialConfig serial_cfg = auto_aim::loadSerialConfig(serial_file);
    const bool use_imu_world_frame =
        (serial_file["serial"] ? serial_file["serial"]["use_imu_world_frame"].as<bool>(false)
                               : false);
    // C 板回传 yaw 的符号（configs/serial.yaml 的 yaw_sign）。2026-09-28 实测这台车的
    const double pitch_sign = [&serial_file] {
        const char* v = std::getenv("ULTRA_VISION_PITCH_SIGN");
        if (v != nullptr) return std::atof(v) < 0.0 ? -1.0 : 1.0;
        const double configured = serial_file["serial"]
            ? serial_file["serial"]["pitch_sign"].as<double>(1.0) : 1.0;
        return configured < 0.0 ? -1.0 : 1.0;
    }();
    const double yaw_sign =
        (serial_file["serial"] ? serial_file["serial"]["yaw_sign"].as<double>(1.0) : 1.0) < 0.0
            ? -1.0
            : 1.0;
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

    auto_aim::AimPipelineConfig pipeline_cfg = auto_aim::app::makeAimPipelineConfig(
        track_cfg,
        auto_aim::loadSelectorConfig(tracker_file),
        auto_aim::loadGimbalConfig(tracker_file),
        auto_aim::loadShooterConfig(tracker_file),
        auto_aim::loadAimFilterConfig(tracker_file),
        auto_aim::loadGimbalCommandRateHz(tracker_file),
        auto_aim::loadMaxControlLostFrames(tracker_file),
        auto_aim::loadFireAngleToleranceDegrees(tracker_file),
        20.0);   // 真机链路必须有心跳（mode=0）
    pipeline_cfg.aim_jump_limit_deg = [] {
        const char* value = std::getenv("ULTRA_VISION_AIM_JUMP_LIMIT_DEG");
        return value != nullptr ? std::atof(value) : 30.0;
    }();
    // 没合格板时：0=瞄最后观测到的板（现行为），1=放弃（保持上一拍瞄点）。
    pipeline_cfg.no_valid_armor_policy = auto_aim::loadNoValidArmorPolicy(tracker_file);

    // 下位机链路 + 控制权：只有 C 板把 mode 置成"自瞄"且链路在线，才允许我们接管。
    const bool fire_enabled = std::getenv("ULTRA_VISION_DISABLE_FIRE") == nullptr;
    // 只跑识别、不接管云台（命令坐标系还没量准时用；见下发处注释）。
    const bool control_disabled = std::getenv("ULTRA_VISION_DISABLE_CONTROL") != nullptr;
    // 图像闭环的速度倍率（1.0 = 中等；大了更快但延迟回路可能振荡）
    const double servo_speed_env = std::getenv("ULTRA_VISION_SERVO_SPEED")
        ? std::atof(std::getenv("ULTRA_VISION_SERVO_SPEED")) : 1.0;
    // 图像闭环的符号（两轴都是 +1）；需要时不用重编译：
    const double servo_sign_env_yaw = std::getenv("ULTRA_VISION_SERVO_SIGN_YAW")
        ? std::atof(std::getenv("ULTRA_VISION_SERVO_SIGN_YAW")) : 1.0;
    const double servo_sign_env_pitch = std::getenv("ULTRA_VISION_SERVO_SIGN_PITCH")
        ? std::atof(std::getenv("ULTRA_VISION_SERVO_SIGN_PITCH")) : 1.0;
    // 开火闸门：图像残差（板相对视轴 + 弹道补偿后的残差）必须小于它、且足够新鲜。
    const double fire_err_tol_rad = (std::getenv("ULTRA_VISION_FIRE_ERR_DEG")
            ? std::atof(std::getenv("ULTRA_VISION_FIRE_ERR_DEG")) : 1.0) * M_PI / 180.0;
    const double fire_err_max_age_s = 0.25;
    // 图像闭环的积分增益（1/s，作用在"误差的积分"上，等效偏置上限 ±10°）。
    const double servo_ki_env = std::getenv("ULTRA_VISION_SERVO_KI")
        ? std::atof(std::getenv("ULTRA_VISION_SERVO_KI")) : 1.2;
    // 误差死区（deg）：小于它就把偏置清零（"保持"），不再每拍蹭一点 —— 治"点头"。
    const double servo_deadband_rad = (std::getenv("ULTRA_VISION_SERVO_DEADBAND_DEG")
            ? std::atof(std::getenv("ULTRA_VISION_SERVO_DEADBAND_DEG")) : 0.7) * M_PI / 180.0;
    // 微分增益（s）：误差变化率的反向修正，压"上下点头"的过冲。
    const double servo_kd_env = std::getenv("ULTRA_VISION_SERVO_KD")
        ? std::atof(std::getenv("ULTRA_VISION_SERVO_KD")) : 0.03;
    // 下发命令角的**速率限幅**（deg/s）。下位机视觉角环是 Ki=0 + Kd=2100/1200，
    const double cmd_rate_yaw_deg_s = (std::getenv("ULTRA_VISION_CMD_RATE_DEG")
            ? std::atof(std::getenv("ULTRA_VISION_CMD_RATE_DEG")) : 150.0);
    const double cmd_rate_pitch_deg_s = (std::getenv("ULTRA_VISION_CMD_RATE_PITCH_DEG")
            ? std::atof(std::getenv("ULTRA_VISION_CMD_RATE_PITCH_DEG")) : 110.0);
    // 速率限幅随误差变化：下界（精跟踪时用）与斜率（deg/s per deg）
    const double cmd_rate_yaw_min_deg_s = 45.0;
    const double cmd_rate_pitch_min_deg_s = 35.0;
    const double cmd_rate_gain_ = 8.0;
    // 速度前馈：下位机把它直接加到速度环设定值上（`Speed_Pid(gyro, Apid_Out + CV_EV.yaw_vel)`），
    const bool ff_enabled = [] {
        const char* v = std::getenv("ULTRA_VISION_FF");
        return v == nullptr ? true : std::string(v) != "0";
    }();
    const double ff_sign =
        std::getenv("ULTRA_VISION_FF_SIGN") ? std::atof(std::getenv("ULTRA_VISION_FF_SIGN")) : -1.0;
    io::Gimbal gimbal(serial_cfg);
    std::cout << "[io] 下位机链路: device=" << serial_cfg.device
              << " connected=" << (gimbal.connected() ? "yes" : "no")
              << " imu_world_frame=" << (use_imu_world_frame ? "on" : "off")
              // 相机 pitch 符号（world↔camera 约定）。画模型"前后反了"就查这一项：
              << " pitch_sign=" << pitch_sign << std::endl;

    // ---- 下发坐标系转换（唯一一处；瞄准 / 心跳 / 速度前馈都从这里出去）----
    std::atomic<double> sent_yaw_now{0.0};
    std::atomic<double> sent_pitch_now{0.0};
    std::atomic<int> sent_mode_now{0};
    // 图像闭环的命令偏置（叠加在"当前回传角"上）与自适应增益符号。
    std::atomic<double> aim_off_yaw{0.0};
    std::atomic<double> aim_off_pitch{0.0};
    // 图像闭环的**误差本身**（rad，板相对视轴 + 弹道下坠补偿后的残差）与它的时间戳。
    std::atomic<double> image_err_yaw{0.0};
    std::atomic<double> image_err_pitch{0.0};
    std::atomic<double> image_err_time{-1.0};   // steady_clock 秒
    std::atomic<int> fire_gate_blocked{0};      // 统计：因为图像残差大被拦下的开火请求
    std::atomic<double> servo_sign_yaw{servo_sign_env_yaw};
    std::atomic<double> servo_sign_pitch{servo_sign_env_pitch};
    // 绝对命令（采样-保持）：只在**有新观测**时锚定一次，其余帧保持不变。
    std::atomic<double> cmd_abs_yaw{0.0};
    std::atomic<double> cmd_abs_pitch{0.0};
    std::atomic<int> cmd_abs_valid{0};
    // 速度前馈状态（见 pipe_send）：需要我们命令自身的角速率，所以放在 lambda 之前。
    double ff_last_send_t = 0.0;
    double ff_last_out_yaw = 0.0;
    double ff_last_out_pitch = 0.0;
    double ff_yaw_vel = 0.0;
    const bool ff_enabled_flag_ = ff_enabled;
    (void)ff_enabled_flag_;
    (void)ff_sign;
    double ff_pitch_vel = 0.0;
    // 命令角的速率限幅状态（见 pipe_send 里的注释）
    double cmd_rate_last_t = 0.0;
    double cmd_rate_last_yaw = 0.0;
    double cmd_rate_last_pitch = 0.0;
    auto pipe_send = [&gimbal, serial_cfg, yaw_sign, &sent_yaw_now, &sent_pitch_now,
                      &sent_mode_now, &aim_off_yaw, &aim_off_pitch,
                      &cmd_abs_yaw, &cmd_abs_pitch, &cmd_abs_valid, &ff_last_send_t,
                      &ff_last_out_yaw, &ff_last_out_pitch, &ff_yaw_vel, &ff_pitch_vel,
                      &image_err_yaw, &image_err_pitch, &image_err_time, &fire_gate_blocked,
                      &cmd_rate_last_t, &cmd_rate_last_yaw, &cmd_rate_last_pitch,
                      ff_enabled, ff_sign, fire_err_tol_rad, fire_err_max_age_s,
                      cmd_rate_yaw_deg_s, cmd_rate_pitch_deg_s, cmd_rate_yaw_min_deg_s,
                      cmd_rate_pitch_min_deg_s, cmd_rate_gain_](
                         bool control, bool fire, double yaw, double yaw_vel, double yaw_acc,
                         double pitch, double pitch_vel, double pitch_acc) {
        // ---- 命令 = 下位机当前回传角(hold) + 图像闭环积累的偏置 ----
        const io::GimbalState hold_state = gimbal.state();
        double out_yaw = hold_state.yaw;
        double out_pitch = hold_state.pitch;
        if (cmd_abs_valid.load(std::memory_order_relaxed) != 0) {
            out_yaw = cmd_abs_yaw.load(std::memory_order_relaxed);
            out_pitch = cmd_abs_pitch.load(std::memory_order_relaxed);
        }
        // ---- 速度前馈：发"命令自身的角速率" ----
        double out_yaw_vel = 0.0;
        double out_pitch_vel = 0.0;
        {
            const double now_send = std::chrono::duration<double>(
                std::chrono::steady_clock::now().time_since_epoch()).count();
            if (ff_last_send_t > 0.0) {
                const double dt_send = now_send - ff_last_send_t;
                if (dt_send > 1e-4 && dt_send < 0.05) {
                    const double raw_vy = (out_yaw - ff_last_out_yaw) / dt_send;
                    const double raw_vp = (out_pitch - ff_last_out_pitch) / dt_send;
                    // 限幅 + 轻度低通（下位机侧还有一层前馈滤波，别把噪声灌进去）
                    const double kMaxVel = 200.0 * M_PI / 180.0;
                    const double vy = std::max(-kMaxVel, std::min(kMaxVel, raw_vy));
                    const double vp = std::max(-kMaxVel, std::min(kMaxVel, raw_vp));
                    ff_yaw_vel = 0.7 * ff_yaw_vel + 0.3 * vy;
                    ff_pitch_vel = 0.7 * ff_pitch_vel + 0.3 * vp;
                    if (ff_enabled) {   // 默认关：见上面环境变量说明
                        out_yaw_vel = ff_sign * ff_yaw_vel;
                        out_pitch_vel = ff_sign * ff_pitch_vel;
                    }
                }
            }
            ff_last_send_t = now_send;
            ff_last_out_yaw = out_yaw;
            ff_last_out_pitch = out_pitch;
        }
        if (serial_cfg.command_frame == io::SerialConfig::CommandFrame::World) {
            // 对照分支：仅当换成"命令帧 ≠ 回传帧"的下位机时用（本车不用）。
            io::ImuSample imu;
            double d_az = 0.0;
            double d_el = 0.0;
            double az_now = 0.0;
            double el_now = 0.0;
            // 当前姿态（与上面四元数同一包）；desired* 是控制器给的目标绝对角。
            const io::GimbalState now_state = gimbal.state();
            const double yaw_now = yaw_sign * now_state.yaw;
            const double pitch_now = now_state.pitch;
            if (!gimbal.latestImu(imu) ||
                !auto_aim::desiredWorldAnglesDelta(yaw_now, pitch_now, yaw, pitch, imu, d_az,
                                                   d_el) ||
                !auto_aim::desiredWorldAngles(yaw_now, pitch_now, yaw_now, pitch_now, imu, az_now,
                                              el_now)) {
                return true;   // 没 IMU：这一拍不下发（云台保持上一拍），别用错口径
            }
            // ---- 全部用**实测**的量拼，不再有任何坐标系假设 ----
            const double base_yaw = az_now;
            const double base_pitch = -el_now;
            // ② 增量：探针台阶实测的 2×2 响应（命令 → 电机角，单位：度/度）
            const double d_internal_yaw =
                auto_aim::GimbalAimer::normalizeAngle(yaw - yaw_now);   // 期望 − 当前（内部）
            const double d_internal_pitch = pitch - pitch_now;
            const double d_raw_yaw = -d_internal_yaw;
            const double d_raw_pitch = d_internal_pitch;
            // 对照分支：基准仍是实测 hold，增量使用现场实测并固定的 M⁻¹。
            out_yaw = base_yaw + 1.172 * d_raw_yaw - 1.116 * d_raw_pitch;
            out_pitch = base_pitch + 0.093 * d_raw_yaw - 1.141 * d_raw_pitch;
            // 速度前馈先不发（0）：位置命令本身已经由轨迹发生器平滑，而两个通道与云台
            out_yaw_vel = 0.0;
            out_pitch_vel = 0.0;
        } else {
            // command_frame: motor —— 命令 = 回传角(采样保持) + 图像闭环偏置，
            out_pitch = std::max(-0.49, std::min(0.27, out_pitch));
        }
        // ---- 命令角速率限幅：把"台阶"变成"斜坡" ----
        {
            const double now_send = std::chrono::duration<double>(
                std::chrono::steady_clock::now().time_since_epoch()).count();
            if (control && cmd_rate_last_t > 0.0) {
                const double dt = now_send - cmd_rate_last_t;
                if (dt > 1e-4 && dt < 0.2) {
                    // 速率限幅**随误差变化**：误差大（捕获/目标跑掉）时快，误差小（精跟踪）
                    const double err_deg = std::max(
                        std::abs(image_err_yaw.load(std::memory_order_relaxed)),
                        std::abs(image_err_pitch.load(std::memory_order_relaxed))) * 180.0 / M_PI;
                    const double rate_yaw = std::max(cmd_rate_yaw_min_deg_s,
                        std::min(cmd_rate_yaw_deg_s, cmd_rate_gain_ * err_deg));
                    const double rate_pitch = std::max(cmd_rate_pitch_min_deg_s,
                        std::min(cmd_rate_pitch_deg_s, cmd_rate_gain_ * err_deg));
                    const double max_yaw_step = rate_yaw * M_PI / 180.0 * dt;
                    const double max_pitch_step = rate_pitch * M_PI / 180.0 * dt;
                    out_yaw = std::max(cmd_rate_last_yaw - max_yaw_step,
                                       std::min(cmd_rate_last_yaw + max_yaw_step, out_yaw));
                    out_pitch = std::max(cmd_rate_last_pitch - max_pitch_step,
                                         std::min(cmd_rate_last_pitch + max_pitch_step,
                                                  out_pitch));
                }
            }
            cmd_rate_last_t = now_send;
            cmd_rate_last_yaw = out_yaw;
            cmd_rate_last_pitch = out_pitch;
        }
        // ---- 开火闸门：以**图像残差**为准，而不是控制器的自比快照 ----
        bool fire_out = fire;
        if (fire) {
            const double now_s = std::chrono::duration<double>(
                std::chrono::steady_clock::now().time_since_epoch()).count();
            const double err_age = now_s - image_err_time.load(std::memory_order_relaxed);
            const double e_yaw = image_err_yaw.load(std::memory_order_relaxed);
            const double e_pitch = image_err_pitch.load(std::memory_order_relaxed);
            const double tol = fire_err_tol_rad;
            if (err_age > fire_err_max_age_s || std::abs(e_yaw) > tol ||
                std::abs(e_pitch) > tol) {
                fire_out = false;   // 这拍不发 mode=2（改成 mode=1：继续控制，不开火）
                fire_gate_blocked.fetch_add(1, std::memory_order_relaxed);
            }
        }
        // 只在**真正下发控制帧**（control=true）时更新遥测：否则心跳（mode=0）会把
        if (control) {
            sent_yaw_now.store(out_yaw, std::memory_order_relaxed);
            sent_pitch_now.store(out_pitch, std::memory_order_relaxed);
            sent_mode_now.store(fire_out ? 2 : 1, std::memory_order_relaxed);
        }
        return gimbal.send(control, fire_out, out_yaw, out_yaw_vel, yaw_acc, out_pitch,
                           out_pitch_vel, pitch_acc);
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
    auto_aim::Tracker& tracker = pipeline.tracker();   // 估计器归流水线所有
    // 装甲板安装倾角：默认取 configs/tracker.yaml 的 pnp.armor_pitch_deg（真机 15°），
    {
        const char* tilt_env = std::getenv("ULTRA_VISION_ARMOR_TILT_DEG");
        if (tilt_env != nullptr) tracker.setArmorTiltDeg(std::atof(tilt_env));
        const auto& geom0 = tracker.chassisGeometry();
        std::cout << "[geometry] armor tilt = " << tracker.getArmorTiltDeg()
                  << " deg, plate_distance = ";
        for (int plate = 0; plate < 4; ++plate)
            std::cout << (plate ? "," : "") << geom0.distance(plate);
        std::cout << " m, plate_size = " << geom0.small_half_width * 2.0 << "x"
                  << geom0.half_height * 2.0 << " m（与 PnP 同源）" << std::endl;
    }
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
    // 图像闭环（见主循环 image-servo 一段）
    bool servo_have_prev_ = false;
    int servo_samples_ = 0;
    int servo_samples_total_ = 0;
    int servo_flips_yaw_ = 0;
    int servo_flips_pitch_ = 0;
    int servo_jump_frames_yaw_ = 0;
    int servo_jump_frames_pitch_ = 0;
    double servo_acc_yaw_ = 0.0;
    double servo_acc_pitch_ = 0.0;
    double servo_prev_err_az_ = 0.0;
    double servo_prev_err_el_ = 0.0;
    double servo_prev_off_yaw_ = 0.0;
    double servo_prev_off_pitch_ = 0.0;
    double servo_int_yaw_ = 0.0;
    double servo_int_pitch_ = 0.0;
    double servo_prev_int_yaw_ = 0.0;
    double servo_prev_int_pitch_ = 0.0;
    double servo_dt_ = 1.0 / 95.0;      // 观测间隔（用实际 dt 更好，这里用帧率近似）
    double servo_last_obs_time_ = -1.0;  // 上一次新观测的 steady_clock 秒（算积分 dt 用）
    double servo_speed_ = servo_speed_env;   // ULTRA_VISION_SERVO_SPEED 倍率（现场调速用）
    const double servo_ki_ = servo_ki_env;   // 积分增益（抵消下位机静差）
    const double servo_kd_ = servo_kd_env;   // 微分增益（压点头/过冲）
    double servo_prev_az_ = 0.0;
    double servo_prev_el_ = 0.0;
    double servo_rate_az_ = 0.0;   // 测量变化率（低通后），微分项用
    double servo_rate_el_ = 0.0;
    double servo_win_max_az_ = 0.0;
    double servo_win_max_el_ = 0.0;
    double selector_speed_ = 25.0;      // 弹速，用于下坠补偿（节点里会按回传更新）
    // ---- 曲线用的"最近一次量测"（给下面的曲线面板用；没观测时保持不变）----
    double curve_az_deg = 0.0;
    double curve_el_deg = 0.0;
    double curve_off_yaw_deg = 0.0;
    double curve_off_pitch_deg = 0.0;
    double curve_drop_deg = 0.0;
    // 权限看门狗（见图像闭环里的注释）：下位机到底听不听我们的。
    int servo_auth_frames_ = 0;
    double servo_auth_best_ = 1e9;
    bool servo_auth_warned_ = false;
    bool servo_auth_ok_prev_ = false;   // 上一拍是否确认"下位机在跟随"（用于捕获接管瞬间）
    double frame_age_seconds = 0.0;   // 本帧"曝光→到手"的延迟（状态行 src_age）
    bool last_source_fresh_ = false;
    // 无目标时的心跳计时（见下发处注释）
    auto last_heartbeat = std::chrono::steady_clock::now();
    int processed_frames = 0;
    // 模型解释度：每个**检测**框中心到最近一块模型板投影中心的像素距离。
    double model_fit_sum = 0.0;
    double model_fit_max = 0.0;
    int model_fit_n = 0;
    // "最差那个检测"的细节（每条秒更新一次）：像素误差 + 三维误差(m) + 类型/PnP 距离/宽度/板号
    double res_radial_sum = 0.0;
    double res_tangent_sum = 0.0;
    double res_vertical_sum = 0.0;
    int res_n = 0;
    double trusted_fit_sum = 0.0;
    double trusted_fit_max = 0.0;
    int trusted_fit_n = 0;
    int ignored_dets = 0;
    double worst_det_dist = -1.0;
    double worst_det_err3d = -1.0;
    double worst_det_pxw = -1.0;
    int worst_det_large = -1;
    int worst_det_plate = -1;
    auto report_time = std::chrono::steady_clock::now();
    // ------------------- 3.5 实时曲线（和仿真 node_sim 同一套组件）----------
    const bool display_env_ok = [] {
#if defined(__linux__)
        return std::getenv("DISPLAY") != nullptr || std::getenv("WAYLAND_DISPLAY") != nullptr;
#else
        return true;   // macOS/Windows 上窗口一般可用
#endif
    }();
    const bool windows_wanted = std::getenv("ULTRA_VISION_NO_DISPLAY") == nullptr;
    const bool windows_usable = windows_wanted && display_env_ok;
    if (windows_wanted && !display_env_ok) {
        std::cout << "[display] 没检测到 DISPLAY/WAYLAND_DISPLAY → 自动关闭窗口"
                     "（无头运行；要用窗口请在有图形界面的终端里跑，或设 DISPLAY）"
                  << std::endl;
    }
    // 下位机回传的**陀螺角速度**（pitch 那一路正是它速度环的反馈量）。加进来是为了
    const bool show_display = windows_usable;
    int key = -1;

    while (key != 27 && key != 'e' && key != 'E') {
        // ------------------- 4. Grab a frame -------------------
        if (!camera.getImage(frame, camera_timeout)) {
            std::cerr << "Failed to get image from camera" << std::endl;
            break;
        }
        if (frame.empty()) continue;

        // IO 自检：B/G/R 通道均值应同量级，否则检查 Bayer 解码/像素格式。
        {
            static int probe_count = 0;
            if (++probe_count % 30 == 1) {
                const cv::Scalar mean = cv::mean(frame);
                std::cout << "[io] frame " << frame.cols << "x" << frame.rows << " mean B/G/R = "
                          << mean[0] << "/" << mean[1] << "/" << mean[2] << std::endl;
            }
        }

        // The frame is processed at its native resolution. Downscaling here
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
        const double observation_time =
            auto_aim::StandardClock::secondsFromUs(source_result.timestamp_us);
        {
            static double last_obs_time = 0.0;
            if (last_obs_time > 0.0 && observation_time > last_obs_time) {
                const double dt = observation_time - last_obs_time;
                if (dt > 0.001 && dt < 0.2) servo_dt_ = dt;
            }
            last_obs_time = observation_time;
        }

        // ------------------- 6. 下位机姿态 + 整车估计 -------------------
        const double now_seconds = auto_aim::StandardClock::nowSeconds();
        const io::GimbalState gimbal_state = gimbal.state();
        auto_aim::CameraPose camera_pose;
        camera_pose.camera_to_gimbal = extrinsics.rotation;
        camera_pose.camera_to_gimbal_translation = extrinsics.translation;
        // ---- 关键：姿态必须取**曝光时刻**的，不能取"处理到这一帧时"的最新姿态 ----
        double frame_age = now_seconds - observation_time;
        // 时间基不同（相机 SDK 的时钟和本地单调钟不是同一纪元）时这个值会离谱：
        if (!std::isfinite(frame_age) || frame_age < 0.0 || frame_age > 0.20) {
            frame_age = 0.0;
        }
        const auto exposure_time =
            std::chrono::steady_clock::now() -
            std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                std::chrono::duration<double>(frame_age));
        io::ImuSample frame_imu;
        if (gimbal.imuAt(exposure_time, frame_imu)) {
            camera_pose.yaw = yaw_sign * frame_imu.yaw;
            camera_pose.pitch = pitch_sign * frame_imu.pitch;
            camera_pose.valid = true;
            camera_pose.timestamp_valid = true;
            if (use_imu_world_frame) {
                // world<-base：四元数 (w,x,y,z) → 旋转矩阵（列主序存进 base_to_world）。
                const double w = frame_imu.w, x = frame_imu.x, y = frame_imu.y, z = frame_imu.z;
                camera_pose.base_to_world = {{
                    {{1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)}},
                    {{2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)}},
                    {{2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)}},
                }};
            }
        } else {
            // 没有姿态历史可查（刚启动/串口刚接上）：退回最新值，并在状态行里看 src_age
            camera_pose.yaw = yaw_sign * gimbal_state.yaw;
            camera_pose.pitch = pitch_sign * gimbal_state.pitch;
            camera_pose.valid = gimbal_state.valid;
            camera_pose.timestamp_valid = gimbal_state.valid;
        }
        camera_pose.timestamp = observation_time;
        frame_age_seconds = frame_age;
        ++processed_frames;

        // 渲染用的相机姿态：与 camera_pose 同源，但取**当前这一拍**的云台角
        auto_aim::CameraPose render_pose = camera_pose;
        {
            io::ImuSample now_imu;
            if (gimbal.imuAt(std::chrono::steady_clock::now(), now_imu)) {
                render_pose.yaw = yaw_sign * now_imu.yaw;
                render_pose.pitch = pitch_sign * now_imu.pitch;
                render_pose.valid = true;
                if (use_imu_world_frame) {
                    const double w = now_imu.w, x = now_imu.x, y = now_imu.y, z = now_imu.z;
                    render_pose.base_to_world = {{
                        {{1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)}},
                        {{2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)}},
                        {{2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)}},
                    }};
                }
            } else if (gimbal_state.valid) {
                render_pose.yaw = yaw_sign * gimbal_state.yaw;
                render_pose.pitch = pitch_sign * gimbal_state.pitch;
            }
        }

        // ------------------- 6.5 估计 → 选板 → 瞄准 → 开火 -------------------
        camera_pose.timestamp = observation_time;
        auto_aim::AimPipeline::FrameInput frame_input;
        frame_input.armors = source_result.armors;
        frame_input.fresh = source_result.fresh;
        frame_input.observation_time = observation_time;
        frame_input.now = auto_aim::StandardClock::nowSeconds();
        frame_input.pose = camera_pose;
        frame_input.bullet_speed = gimbal_state.bullet_speed;
        allow_control_now = gimbal.mode() == io::GimbalMode::AUTO_AIM && gimbal.connected();
        // 安全闸：命令坐标系/符号还没量准之前，用 ULTRA_VISION_DISABLE_CONTROL=1 只跑识别，
        if (control_disabled) allow_control_now = false;
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
        // 原始检测只留**淡色细线**：识别出的板本来就是底盘模型的一部分，统一由下面
        for (const auto& armor : armors) {
            cv::line(image, armor.left.top, armor.right.bottom, cv::Scalar(0, 140, 0), 1);
            cv::line(image, armor.left.bottom, armor.right.top, cv::Scalar(0, 140, 0), 1);
        }

        const bool has_observation = tracker.hasObservation();
        const auto_aim::Armor& observed_armor = tracker.getLastObservedArmor();
        // ---- "只认最新观测"：tracker 在没有新框的帧上会保留上一帧的 armors
        const bool observed_this_frame =
            std::abs(tracker.lastObservationTime() - observation_time) < 0.001;
        const double observation_age = now_seconds - tracker.lastObservationTime();
        // ================= 图像闭环（最终的瞄准实现）=============================
        if (has_observation && last_source_fresh_ && observed_this_frame && allow_control_now &&
            tracker.getState() == auto_aim::TrackerState::TRACKING) {
            const cv::Mat& tvec = observed_armor.tvec;
            if (observed_armor.solve_result && tvec.rows >= 3 &&
                std::abs(tvec.at<double>(2)) > 1e-6) {
                // 偏角用**像素框中心**算（内参换算），不用 PnP 的 tvec：
                const auto& bar_l = observed_armor.left;
                const auto& bar_r = observed_armor.right;
                const double u = 0.25 * (bar_l.top.x + bar_l.bottom.x + bar_r.top.x + bar_r.bottom.x);
                const double v = 0.25 * (bar_l.top.y + bar_l.bottom.y + bar_r.top.y + bar_r.bottom.y);
                double az = std::atan2(u - camera_matrix.at<double>(0, 2),
                                       camera_matrix.at<double>(0, 0));   // +右
                double el = -std::atan2(v - camera_matrix.at<double>(1, 2),
                                        camera_matrix.at<double>(1, 1));  // +上
                // **只用最新这一帧的测量，不做 EMA 平滑**（用户提的"只取最新的"）。
                const double prev_az = servo_prev_az_;   // 微分项要用"上一拍"的测量
                const double prev_el = servo_prev_el_;
                servo_prev_az_ = az;
                servo_prev_el_ = el;
                const double x = tvec.at<double>(0);
                const double y = tvec.at<double>(1);
                const double z = tvec.at<double>(2);
                const double distance = std::sqrt(x * x + y * y + z * z);
                const double flight = distance / std::max(1.0, selector_speed_);
                // 弹道下坠：让板出现在视轴**下方** drop 弧度处（子弹会落这么多）
                const double drop = 0.5 * 9.81 * flight * flight / std::max(0.5, distance);
                const double err_az = 0.0 - az;
                const double err_el = (-drop) - el;
                // ---- 几何增量（主回路，2026-10-15）------------------------------
                const bool geom_ok = last_decision.valid &&
                    std::isfinite(last_decision.target_yaw) &&
                    std::isfinite(last_decision.target_pitch) && camera_pose.valid;
                const double err_geom_az = geom_ok
                    ? -auto_aim::GimbalAimer::normalizeAngle(
                          last_decision.target_yaw - camera_pose.yaw)
                    : 0.0;
                const double err_geom_el =
                    geom_ok ? -(last_decision.target_pitch - camera_pose.pitch) : 0.0;
                // ---- 跳变门：识别到的板在物理板之间跳时，单帧偏差会突跳 ±12~16°。
                constexpr double kJumpGate = 6.0 * M_PI / 180.0;
                if (servo_have_prev_ &&
                    (std::abs(err_az - servo_prev_err_az_) > kJumpGate ||
                     std::abs(err_el - servo_prev_err_el_) > kJumpGate)) {
                    if (std::abs(err_az - servo_prev_err_az_) > kJumpGate) ++servo_jump_frames_yaw_;
                    if (std::abs(err_el - servo_prev_err_el_) > kJumpGate) ++servo_jump_frames_pitch_;
                    // 连续太多帧都"突变"说明真的换了目标（不是跳变）→ 重新锚定
                    if (servo_jump_frames_yaw_ > 190 || servo_jump_frames_pitch_ > 190) {
                        servo_have_prev_ = false;
                        servo_jump_frames_yaw_ = 0;
                        servo_jump_frames_pitch_ = 0;
                    } else {
                        servo_samples_total_++;
                        goto image_servo_done;   // 本帧不改积分，也不更新历史
                    }
                } else {
                    servo_jump_frames_yaw_ = 0;
                    servo_jump_frames_pitch_ = 0;
                }
                // ---- 权限看门狗：下位机到底听不听我们的？ --------------------
                bool authority_ok = true;
                if (cmd_abs_valid.load(std::memory_order_relaxed) != 0) {
                    constexpr double kAuthorityMinError = 3.0 * M_PI / 180.0;
                    constexpr double kPitchAuthorityMinError = 12.0 * M_PI / 180.0;
                    const io::GimbalState auth_state = gimbal.state();
                    const double auth_err_yaw =
                        std::abs(cmd_abs_yaw.load(std::memory_order_relaxed) - auth_state.yaw);
                    const double auth_err_pitch =
                        std::abs(cmd_abs_pitch.load(std::memory_order_relaxed) - auth_state.pitch);
                    if (auth_err_yaw > kAuthorityMinError ||
                        auth_err_pitch > kPitchAuthorityMinError) {
                        const double auth_err = std::max(auth_err_yaw, auth_err_pitch);
                        if (auth_err < servo_auth_best_ - 0.3 * M_PI / 180.0) {
                            servo_auth_best_ = auth_err;   // 还在收敛 → 重新计时
                            servo_auth_frames_ = 0;
                        } else if (++servo_auth_frames_ > 40) {
                            authority_ok = false;
                            if (!servo_auth_warned_) {
                                servo_auth_warned_ = true;
                                std::cout << "[authority] 下位机 0.5s 内没有跟随命令（yaw 差 "
                                          << auth_err_yaw * 180.0 / M_PI << "° / pitch 差 "
                                          << auth_err_pitch * 180.0 / M_PI
                                          << "°，且不再缩小）→ 疑似不在纯视觉接管，本段只发保持"
                                          << std::endl;
                            }
                        }
                    } else {
                        // 从"没跟随"回到"跟随"（= 操作手刚把开关拨到纯视觉接管）：
                        if (!servo_auth_ok_prev_) {
                            servo_int_yaw_ = 0.0;
                            servo_int_pitch_ = 0.0;
                            servo_prev_off_yaw_ = 0.0;
                            servo_prev_off_pitch_ = 0.0;
                            std::cout << "[authority] 重新获得跟随，积分/偏置清零重来" << std::endl;
                        }
                        servo_auth_frames_ = 0;
                        servo_auth_best_ =
                            std::max(auth_err_yaw, auth_err_pitch > 1.0 * M_PI / 180.0
                                                       ? auth_err_pitch
                                                       : 0.0);
                        if (servo_auth_warned_) {
                            servo_auth_warned_ = false;
                            std::cout << "[authority] 命令重新被跟随，恢复闭环" << std::endl;
                        }
                    }
                }
                servo_auth_ok_prev_ = authority_ok;
                const double sy = servo_sign_yaw.load(std::memory_order_relaxed);
                const double sp = servo_sign_pitch.load(std::memory_order_relaxed);
                // ---- PD + I（原来是纯积分 0.25/s，实测收敛要 1~2s，响应太慢）----
                const double kpScale = servo_speed_;
                // 0.12/0.10 → 0.20/0.16：现场反馈"收敛不够快、响应不够快"。
                const double kpYaw = 0.05 * kpScale;
                const double kpPitch = 0.05 * kpScale;
                // ---- 积分项：专门治 pitch 的结构性静差 ----------------------
                const double now_obs = std::chrono::duration<double>(
                    std::chrono::steady_clock::now().time_since_epoch()).count();
                double dt_obs = servo_dt_;
                if (servo_last_obs_time_ > 0.0) {
                    const double gap = now_obs - servo_last_obs_time_;
                    // 断了很久（换目标/丢失后重捕）就别把旧积分原样带过去：减半。
                    if (gap > 0.5) {
                        servo_int_yaw_ *= 0.5;
                        servo_int_pitch_ *= 0.5;
                    }
                    dt_obs = std::min(0.2, std::max(1e-3, gap));
                }
                servo_last_obs_time_ = now_obs;
                // 积分只做"消静差"这一件事：等效偏置上限收到 ±5°（实测下位机静差 0.4~3.4°），
                constexpr double kIntegralOffsetLimit = 5.0 * M_PI / 180.0;   // ki·I ≤ 5°
                const double kIntegralLimit =
                    kIntegralOffsetLimit / std::max(0.1, servo_ki_);
                // 阈值放到 15°：符号已经定了（两轴 sp=+1，见上面的推导），现在积分是
                const bool integrate_ok = authority_ok &&
                    std::abs(err_geom_az) < 15.0 * M_PI / 180.0 &&
                    std::abs(err_geom_el) < 15.0 * M_PI / 180.0;
                // 死区判定与积分都改用**几何误差**：图像误差在追板/守角档里本来就不是 0
                const bool dead_yaw = std::abs(err_geom_az) < servo_deadband_rad;
                const bool dead_pitch = std::abs(err_geom_el) < servo_deadband_rad;
                if (integrate_ok && !dead_yaw) {
                    servo_int_yaw_ += err_geom_az * dt_obs;
                    servo_int_yaw_ = std::max(-kIntegralLimit, std::min(kIntegralLimit, servo_int_yaw_));
                }
                if (integrate_ok && !dead_pitch) {
                    servo_int_pitch_ += err_geom_el * dt_obs;
                    servo_int_pitch_ =
                        std::max(-kIntegralLimit, std::min(kIntegralLimit, servo_int_pitch_));
                }
                // ---- 微分项：专门压"上下点头"（只对有静摩擦的 pitch 明显）---------
                double rate_az = 0.0;
                double rate_el = 0.0;
                if (servo_have_prev_ && dt_obs > 1e-3) {
                    rate_az = (az - prev_az) / dt_obs;
                    rate_el = (el - prev_el) / dt_obs;
                }
                servo_rate_az_ = 0.7 * servo_rate_az_ + 0.3 * rate_az;
                servo_rate_el_ = 0.7 * servo_rate_el_ + 0.3 * rate_el;
                // ---- 偏置 = 几何增量(1.0) + 观测残差(小权重 kpYaw/kpPitch) + 静差积分 - 阻尼
                constexpr double kGeomGain = 1.0;   // 几何增量整份下发（feedforward）
                double off_yaw_new = sy * (kGeomGain * err_geom_az + kpYaw * err_az +
                                           servo_ki_ * servo_int_yaw_ -
                                           servo_kd_ * servo_rate_az_);
                double off_pitch_new = sp * (kGeomGain * err_geom_el + kpPitch * err_el +
                                             servo_ki_ * servo_int_pitch_ -
                                             servo_kd_ * servo_rate_el_);
                // ---- 死区：进死区**保持上一拍偏置**，不清零 -------------------------
                if (dead_yaw) off_yaw_new = servo_prev_off_yaw_;
                if (dead_pitch) off_pitch_new = servo_prev_off_pitch_;
                // ---- 每拍步长限幅（抗极限环的关键，比总幅值限幅更重要）----
                const double kMaxStepYaw = 6.0 * M_PI / 180.0;
                const double kMaxStepPitch = 4.0 * M_PI / 180.0;
                off_yaw_new = std::max(servo_prev_off_yaw_ - kMaxStepYaw,
                                       std::min(servo_prev_off_yaw_ + kMaxStepYaw, off_yaw_new));
                off_pitch_new = std::max(servo_prev_off_pitch_ - kMaxStepPitch,
                                         std::min(servo_prev_off_pitch_ + kMaxStepPitch,
                                                  off_pitch_new));
                // 幅值硬限幅：正常瞄准的偏置只有几度（实测收敛段 0.2~0.4°，接管瞬间 3~4°），
                const double kMaxOffYaw = 30.0 * M_PI / 180.0;
                const double kMaxOffPitch = 20.0 * M_PI / 180.0;
                double off_yaw = std::max(-kMaxOffYaw, std::min(kMaxOffYaw, off_yaw_new));
                double off_pitch = std::max(-kMaxOffPitch, std::min(kMaxOffPitch, off_pitch_new));
                // 记录给曲线面板：板相对视轴的角、比例偏置、弹道下坠等效角。
                curve_az_deg = az * 180.0 / CV_PI;
                curve_el_deg = el * 180.0 / CV_PI;
                curve_off_yaw_deg = off_yaw * 180.0 / CV_PI;
                curve_off_pitch_deg = off_pitch * 180.0 / CV_PI;
                curve_drop_deg = drop * 180.0 / CV_PI;
                // 开火闸门用的图像残差（和曲线同一个量；见 pipe_send 里的注释）
                image_err_yaw.store(err_az, std::memory_order_relaxed);
                image_err_pitch.store(err_el, std::memory_order_relaxed);
                image_err_time.store(
                    std::chrono::duration<double>(
                        std::chrono::steady_clock::now().time_since_epoch()).count(),
                    std::memory_order_relaxed);
                // 锚定绝对命令：**当前回传角**（下位机自己的反馈，"发当前值即不动"）+ 修正。
                {
                    const double gimbal_state_now_yaw = gimbal.state().yaw;
                    const double gimbal_state_now_pitch = gimbal.state().pitch;
                    const double raw_pose_yaw = gimbal_state_now_yaw;
                    const double raw_pose_pitch = gimbal_state_now_pitch;
                    const double pitch_lo = -0.49;
                    const double pitch_hi = 0.27;
                    // 没权限时退化成"保持当前姿态"：不让偏置继续堆（否则恢复接管的瞬间会甩）。
                    const double out_off_yaw = authority_ok ? off_yaw : 0.0;
                    const double out_off_pitch = authority_ok ? off_pitch : 0.0;
                    cmd_abs_yaw.store(raw_pose_yaw + out_off_yaw, std::memory_order_relaxed);
                    cmd_abs_pitch.store(
                        std::max(pitch_lo, std::min(pitch_hi, raw_pose_pitch + out_off_pitch)),
                        std::memory_order_relaxed);
                    cmd_abs_valid.store(1, std::memory_order_relaxed);
                }
                // ---- 符号自适应：Σ(Δ偏置 × Δ误差) > 0 表示"越推误差越大" = 符号反了 ----
                const bool adapt_ok = false &&
                    (std::abs(err_az) > 4.0 * M_PI / 180.0 ||
                     std::abs(err_el) > 4.0 * M_PI / 180.0);
                if (servo_have_prev_ && adapt_ok) {
                    const double d_err_az = err_az - servo_prev_err_az_;
                    const double d_err_el = err_el - servo_prev_err_el_;
                    // 只用**积分**增量判符号：比例项会让 Δoff·Δerr 恒为正，必然误判。
                    servo_acc_yaw_ += (servo_int_yaw_ - servo_prev_int_yaw_) * d_err_az;
                    servo_acc_pitch_ += (servo_int_pitch_ - servo_prev_int_pitch_) * d_err_el;
                    if (++servo_samples_ >= 150) {
                        if (servo_flips_yaw_ < 1 && servo_acc_yaw_ > 5e-4) {
                            servo_sign_yaw.store(-sy, std::memory_order_relaxed);
                            ++servo_flips_yaw_;
                            std::cout << "[image-servo] yaw 通道推反了（ΣΔoff·Δerr="
                                      << servo_acc_yaw_ << "）→ 已翻转符号" << std::endl;
                        }
                        if (servo_flips_pitch_ < 1 && servo_acc_pitch_ > 5e-4) {
                            servo_sign_pitch.store(-sp, std::memory_order_relaxed);
                            ++servo_flips_pitch_;
                            std::cout << "[image-servo] pitch 通道推反了（ΣΔoff·Δerr="
                                      << servo_acc_pitch_ << "）→ 已翻转符号" << std::endl;
                        }
                        servo_samples_ = 0;
                        servo_acc_yaw_ = 0.0;
                        servo_acc_pitch_ = 0.0;
                    }
                }
                servo_prev_err_az_ = err_az;
                servo_prev_err_el_ = err_el;
                servo_prev_off_yaw_ = off_yaw;
                servo_prev_off_pitch_ = off_pitch;
                servo_prev_int_yaw_ = servo_int_yaw_;
                servo_prev_int_pitch_ = servo_int_pitch_;
                servo_have_prev_ = true;
                // 打印口径：每 5 个观测一行，并给出**上一窗口内的最大 |偏差|** ——
                servo_win_max_az_ = std::max(servo_win_max_az_, std::abs(err_az));
                servo_win_max_el_ = std::max(servo_win_max_el_, std::abs(err_el));
                if (++servo_samples_total_ % 5 == 0) {
                    std::cout << "[image-servo] az=" << az * 180.0 / M_PI << "° el="
                              << el * 180.0 / M_PI << "°；本窗口最大 |az|="
                              << servo_win_max_az_ * 180.0 / M_PI << "° |el|="
                              << servo_win_max_el_ * 180.0 / M_PI << "°；偏置 yaw="
                              << off_yaw * 180.0 / M_PI << "° pitch=" << off_pitch * 180.0 / M_PI
                              << "°（弹道下坠等效 " << curve_drop_deg
                              << "°，期望稳态 el≈-" << curve_drop_deg << "°）观测延迟 "
                              << observation_age * 1000.0 << "ms" << std::endl;
                    servo_win_max_az_ = 0.0;
                    servo_win_max_el_ = 0.0;
                }
            }
        }
    image_servo_done:;
        const int observed_plate =
            has_observation ? tracker.getLastObservedPlateId() : -1;
        // 尺寸不再在这里拼：每块板的半宽由底盘几何按"大板/小板"给（tracker.armorHalfWidth）。
        const bool large_armor =
            has_observation && observed_armor.armor_type == auto_aim::large;
        // 模型板的姿态/位置**只走 Tracker::modelAt()**（common/chassis_model.hpp 的唯一构造

        // 模型对**已观测那块板**的重投影误差（像素）——这是判断"估计是否自洽"最直接的量：
        double reprojection_px = -1.0;
        // 只在"正在跟踪且这一帧有观测"时画模型：没有观测时画的只是滑行外推，
        const bool model_visible = has_observation &&
                                   tracker.getState() == auto_aim::TrackerState::TRACKING;
        if (model_visible) {
            // —— 重投影的时间基 ——
            const double model_dt =
                std::max(0.0, std::min(observation_age - frame_age, 0.20));
            const auto model = tracker.modelAt(model_dt, render_pose);
            // **指标必须与检测同刻**：reproj / model_fit 是"模型 vs 本帧检测框"，
            const auto model_at_obs = tracker.modelAt(0.0, camera_pose);
            for (int plate = 0; plate < 4; ++plate) {
                if (!model.valid) continue;
                const auto_aim::ArmorPose& pose = model.poses[plate];
                const cv::Mat center_mat = (cv::Mat_<double>(3, 1) << pose.center[0],
                                            pose.center[1], pose.center[2]);
                const auto corners = auto_aim::projectArmorCameraFrame(
                    center_mat, pose.axes.tangent, pose.axes.bar,
                    tracker.armorHalfWidth(plate), tracker.armorHalfHeight(),
                    auto_aim::CAMERA_MATRIX, auto_aim::DIST_COEFFS);
                if (corners.size() != 4) continue;
                // 四块板**一视同仁**：都画成底盘模型的一部分（同一个颜色、同一套 ID）。
                const bool observed = plate == observed_plate;
                const cv::Scalar color = observed ? cv::Scalar(255, 180, 0)   // 观测板：亮青蓝
                                                  : cv::Scalar(200, 140, 0);  // 外推板：稍暗
                auto_aim::drawQuad(image, corners, color, observed ? 2 : 1);
                cv::putText(image, "E" + std::to_string(plate),
                            corners[0] + cv::Point2f(2.0f, -3.0f), cv::FONT_HERSHEY_SIMPLEX, 0.45,
                            color, 1, cv::LINE_AA);
                if (observed) {
                    // 用**与检测同刻**的模型算残差（见 model_at_obs 的注释）。
                    std::vector<cv::Point2f> corners_at_obs = corners;
                    if (model_at_obs.valid) {
                        const auto& pose_obs = model_at_obs.poses[plate];
                        const cv::Mat center_obs =
                            (cv::Mat_<double>(3, 1) << pose_obs.center[0], pose_obs.center[1],
                             pose_obs.center[2]);
                        const auto projected = auto_aim::projectArmorCameraFrame(
                            center_obs, pose_obs.axes.tangent, pose_obs.axes.bar,
                            tracker.armorHalfWidth(plate), tracker.armorHalfHeight(),
                            auto_aim::CAMERA_MATRIX, auto_aim::DIST_COEFFS);
                        if (projected.size() == 4) corners_at_obs = projected;
                    }
                    // 角点顺序：模型是 (左下,左上,右上,右下)，检测是 (左上,右上,右下,左下)
                    const std::array<cv::Point2f, 4> detected{
                        observed_armor.left.top, observed_armor.right.top,
                        observed_armor.right.bottom, observed_armor.left.bottom};
                    // 两个四边形的起点/绕向可能不同，取 4 个循环移位里最小的那个（这才是
                    double best = 1e18;
                    for (int shift = 0; shift < 4; ++shift) {
                        double sum = 0.0;
                        for (int k = 0; k < 4; ++k) {
                            sum += cv::norm(corners_at_obs[k] - detected[(k + shift) % 4]);
                        }
                        best = std::min(best, sum / 4.0);
                    }
                    reprojection_px = best;
                }
            }

            // ---- 模型解释度：模型 vs **本帧所有检测**（不只被硬钉那块）----------
            if (model_at_obs.valid && !armors.empty()) {
                std::array<cv::Point2f, 4> model_px{};
                std::array<bool, 4> model_px_ok{};
                for (int plate = 0; plate < 4; ++plate) {
                    const auto& pose = model_at_obs.poses[plate];
                    const cv::Mat plate_center =
                        (cv::Mat_<double>(3, 1) << pose.center[0], pose.center[1], pose.center[2]);
                    model_px_ok[plate] = auto_aim::projectPoint(
                        plate_center, auto_aim::CAMERA_MATRIX, auto_aim::DIST_COEFFS,
                        model_px[plate]) && pose.center[2] > 0.3;
                }
                for (const auto& armor : armors) {
                    if (!armor.solve_result || armor.tvec.rows != 3) continue;
                    const cv::Point2f detected_center =
                        0.25f * (armor.left.top + armor.left.bottom + armor.right.top +
                                 armor.right.bottom);
                    double best = 1e18;
                    int best_plate = -1;
                    for (int plate = 0; plate < 4; ++plate) {
                        if (!model_px_ok[plate]) continue;
                        const double d = cv::norm(model_px[plate] - detected_center);
                        if (d < best) { best = d; best_plate = plate; }
                    }
                    // 反推这块检测的视视角（与 Tracker 的 graze 门控同一算法）：
                    const double det_dist = std::sqrt(
                        armor.tvec.at<double>(0) * armor.tvec.at<double>(0) +
                        armor.tvec.at<double>(1) * armor.tvec.at<double>(1) +
                        armor.tvec.at<double>(2) * armor.tvec.at<double>(2));
                    const double det_width_m = (armor.armor_type == auto_aim::large)
                        ? pnp_geometry.large_width : pnp_geometry.small_width;
                    const double det_fx = CAMERA_MATRIX.at<double>(0, 0);
                    const double det_expected =
                        std::max(1.0, det_width_m * det_fx / std::max(0.3, det_dist));
                    const double det_sep = cv::norm(armor.left.center - armor.right.center);
                    const double det_view_deg = std::acos(std::clamp(
                        det_sep / det_expected, 0.0, 1.0)) * 180.0 / M_PI;
                    const bool trusted = det_view_deg <= 50.0;
                    if (best < 1e17) {
                        if (trusted) {
                            trusted_fit_sum += best;
                            ++trusted_fit_n;
                            trusted_fit_max = std::max(trusted_fit_max, best);
                            // 残差**按模型自身的三轴分解**（相机系）：这一步决定下一刀砍哪里
                            if (best_plate >= 0) {
                                const auto& ax = model_at_obs.poses[best_plate].axes;
                                const cv::Vec3d obs_p(armor.tvec.at<double>(0),
                                                      armor.tvec.at<double>(1),
                                                      armor.tvec.at<double>(2));
                                const cv::Vec3d mdl_p = model_at_obs.poses[best_plate].center;
                                const cv::Vec3d res = obs_p - mdl_p;
                                res_radial_sum += std::abs(res.dot(ax.normal));
                                res_tangent_sum += std::abs(res.dot(ax.tangent));
                                res_vertical_sum += std::abs(res.dot(ax.bar));
                                ++res_n;
                            }
                        } else {
                            ++ignored_dets;
                        }
                        model_fit_sum += best;
                        ++model_fit_n;
                        // 记录"最差那个检测"的细节：像素误差 + **三维**误差（米）+
                        if (best > model_fit_max) {
                            model_fit_max = best;
                            const double dx = armor.tvec.at<double>(0);
                            const double dy = armor.tvec.at<double>(1);
                            const double dz = armor.tvec.at<double>(2);
                            const double dist3d = std::sqrt(dx * dx + dy * dy + dz * dz);
                            double err3d = -1.0;
                            if (best_plate >= 0 && model_at_obs.poses[best_plate].center[2] > 0.3) {
                                const cv::Vec3d a(dx, dy, dz);
                                const cv::Vec3d b(model_at_obs.poses[best_plate].center);
                                err3d = cv::norm(a - b);
                            }
                            worst_det_dist = dist3d;
                            worst_det_err3d = err3d;
                            worst_det_large = (armor.armor_type == auto_aim::large) ? 1 : 0;
                            worst_det_pxw = cv::norm(armor.left.center - armor.right.center);
                            worst_det_plate = best_plate;
                        }
                    }
                }
            }

            // 红点 = 底盘中心，也画在外推后的时刻（和上面四块板同一个时刻）。
            const cv::Mat center_mat = (cv::Mat_<double>(3, 1) << model.center_camera[0],
                                        model.center_camera[1], model.center_camera[2]);
            cv::Point2f chassis_center;
            if (auto_aim::projectPoint(center_mat,
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

        if (show_display) {
            cv::Mat display = image;
            // 1440x1080 原图显示前先等比缩小，只影响显示，不影响任何标定/PnP。
            constexpr int kDisplayMaxRows = 600;
            if (display.rows > kDisplayMaxRows) {
                const double scale = static_cast<double>(kDisplayMaxRows) / display.rows;
                cv::Mat resized;
                cv::resize(display, resized, cv::Size(), scale, scale, cv::INTER_AREA);
                display = resized;
            }
            // 双保险：即便有 DISPLAY，后端也可能失败（驱动/权限），任何异常都降级无头，
            static bool display_failed = false;
            if (!display_failed) {
                try {
                    cv::imshow("Detection & Tracking", display);
                } catch (const cv::Exception& e) {
                    display_failed = true;
                    std::cerr << "[display] imshow 失败（" << e.what()
                              << "）→ 关闭窗口，继续无头运行" << std::endl;
                }
            }
        }
        key = show_display ? cv::waitKey(1) : 1;

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
                      << ", mcu: " << io::gimbalModeName(gimbal.mode())
                      << " yaw=" << now_state.yaw << " pitch=" << now_state.pitch
                      << " v=" << now_state.bullet_speed
                      << " recv=" << gimbal_stats.fps << "Hz"
                      // nn=推理速率/延迟，fresh 表示本帧有没有新观测（异步常态下会交替 0/1，
                      << " nn=" << armor_source.neuralFps() << "Hz/"
                      << armor_source.neuralLatencyMs() << "ms"
                      << " fresh=" << (last_source_fresh_ ? 1 : 0)
                      << " sent=" << sent_frames_last_second << "Hz"
                      << " crc_err=" << gimbal_stats.crc_errors
                      << ", ctrl=" << (allow_control_now ? 1 : 0)
                      << " aim_err=" << aim_yaw_error * 180.0 / CV_PI << "/"
                      << aim_pitch_error * 180.0 / CV_PI << "deg"
                      // 图像残差 = 真正决定"打不打得到"的那个量（也是开火闸门的判据）。
                      << " img_err="
                      << image_err_yaw.load(std::memory_order_relaxed) * 180.0 / CV_PI << "/"
                      << image_err_pitch.load(std::memory_order_relaxed) * 180.0 / CV_PI
                      << "deg(闸门拦下 " << fire_gate_blocked.load(std::memory_order_relaxed)
                      << ")"
                      << ", fire=" << (outcome.fire ? 1 : 0) << ", reproj=" << reprojection_px
                      // 模型口径：R 是模型用的回转半径，model_age 是重投影外推到
                      << ", model_fit="
                      << (model_fit_n > 0 ? model_fit_sum / model_fit_n : -1.0) << "/"
                      << (model_fit_n > 0 ? model_fit_max : -1.0) << "px(n=" << model_fit_n << ")"
                      // trusted = 只用"视角<=50°"的可信检测统计的 model_fit（擦边板
                      << " trusted=" << (trusted_fit_n > 0 ? trusted_fit_sum / trusted_fit_n : -1.0)
                      << "/" << (trusted_fit_n > 0 ? trusted_fit_max : -1.0)
                      << "px(n=" << trusted_fit_n << ") ig=" << ignored_dets
                      // 可信检测残差的三轴分解（米）：rad=沿板法向(R 错)
                      << " res(rad=" << (res_n > 0 ? res_radial_sum / res_n : -1.0)
                      << " tan=" << (res_n > 0 ? res_tangent_sum / res_n : -1.0)
                      << " ver=" << (res_n > 0 ? res_vertical_sum / res_n : -1.0)
                      << "m)"
                      // 最差那个检测：3D=到最近模型板的三维误差(m) type=大/小 d=PnP距离
                      << " worst(3D=" << worst_det_err3d << "m d=" << worst_det_dist
                      << "w=" << worst_det_pxw << " type=" << worst_det_large
                      << " plate=" << worst_det_plate << ")"
                      // 几何慢收敛的结果（现场就看这四个数会不会收敛成"前后一个值、
                      << ", Rgeo="
                      << cv::format("%.3f/%.3f/%.3f/%.3f", tracker.chassisGeometry().distance(0),
                                    tracker.chassisGeometry().distance(1),
                                    tracker.chassisGeometry().distance(2),
                                    tracker.chassisGeometry().distance(3))
                      // view=被选中那块板的视视角（度）：>graze_gate_deg 时该板的
                      << " view=" << tracker.lastViewDeg()
                      // 逐板高度（cm，向上为正）——默认应恒为 0.0/0.0/0.0/0.0；
                      << ", Hgeo="
                      << cv::format("%.1f/%.1f/%.1f/%.1f",
                                    tracker.chassisGeometry().height(0) * 100.0,
                                    tracker.chassisGeometry().height(1) * 100.0,
                                    tracker.chassisGeometry().height(2) * 100.0,
                                    tracker.chassisGeometry().height(3) * 100.0)
                      << ", R=" << tracker.getArmorRadius()
                      << ", model_age=" << observation_age * 1000.0 << "ms"
                      << ", send=" << sent_yaw_now.load(std::memory_order_relaxed) * 180.0 / CV_PI
                      << "/" << sent_pitch_now.load(std::memory_order_relaxed) * 180.0 / CV_PI
                      << "deg（我们下发, mode=" << sent_mode_now.load(std::memory_order_relaxed)
                      << "）"
                      << "px, src_age=" << frame_age_seconds * 1000.0 << "ms";
            {
                const auto wd = tracker.watchdogStats();
                std::cout << " wd(reset=" << wd.resets << ", reject=" << wd.reject_ratio
                          << ", conv=" << (wd.converged ? 1 : 0) << ")"
                          << " still=" << (pipeline.targetStill() ? 1 : 0);
            }
            if (has_observation) {
                const double px_w = cv::norm(observed_armor.left.top - observed_armor.right.top);
                const double px_h = cv::norm(observed_armor.left.top - observed_armor.left.bottom);
                std::cout << "  armor=" << (large_armor ? "LARGE" : "small") << "(assumed w"
                          << (large_armor ? pnp_geometry.large_width : pnp_geometry.small_width)
                          << "/h" << pnp_geometry.height << "m, px " << px_w << "x" << px_h
                          << ", dist " << cv::norm(observed_armor.tvec) << "m)";
                // **靶板相对视轴的角度**：瞄得准不准就看这个接近 0（与任何坐标系约定无关）。
                const cv::Mat& tvec = observed_armor.tvec;
                if (tvec.rows >= 3 && std::abs(tvec.at<double>(2)) > 1e-6) {
                    const double x = tvec.at<double>(0);
                    const double y = tvec.at<double>(1);
                    const double z = tvec.at<double>(2);
                    std::cout << " target_az=" << std::atan2(x, z) * 180.0 / CV_PI
                              << "deg target_el=" << -std::atan2(y, std::hypot(x, z)) * 180.0 / CV_PI
                              << "deg（板相对视轴，0=正中）";
                }
            }
            std::cout << std::endl;
            report_time = now;
            processed_frames = 0;
            model_fit_sum = 0.0;
            model_fit_max = 0.0;
            model_fit_n = 0;
            res_radial_sum = 0.0;
            res_tangent_sum = 0.0;
            res_vertical_sum = 0.0;
            res_n = 0;
            trusted_fit_sum = 0.0;
            trusted_fit_max = 0.0;
            trusted_fit_n = 0;
            ignored_dets = 0;
            worst_det_dist = -1.0;
            worst_det_err3d = -1.0;
            worst_det_pxw = -1.0;
            worst_det_large = -1;
            worst_det_plate = -1;
        }
    }

    return 0;
}
