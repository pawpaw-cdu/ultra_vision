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
#include "app/csv_sink.hpp"
#include "app/pipeline_config.hpp"
#include "dataset.hpp"
#include "visualization/projection.hpp"
#include "visualization/curve_plotter.hpp"
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
    // P2 A/B 开关：ULTRA_VISION_LOCAL_FIT=1 打开局部拟合器（否则用 tracker.yaml 的默认）。
    if (const char* lf = std::getenv("ULTRA_VISION_LOCAL_FIT")) {
        track_cfg.local_fit.enabled = (std::atoi(lf) == 1);
    }

#ifdef ULTRA_VISION_HAVE_LOCAL_FIT
    if (track_cfg.local_fit.enabled && camera_matrix.rows == 3) {
        const auto env_d = [](const char* n, double d) {
            const char* v = std::getenv(n);
            return v != nullptr ? std::atof(v) : d;
        };
        auto_aim::LocalTrajectoryFitConfig fcfg;
        fcfg.K = { camera_matrix.at<double>(0, 0), camera_matrix.at<double>(1, 1),
                   camera_matrix.at<double>(0, 2), camera_matrix.at<double>(1, 2) };
        fcfg.lightbar_len = env_d("ULTRA_VISION_LOCAL_FIT_L", 0.060);   // 灯条/板高（m）
        fcfg.plate_width = env_d("ULTRA_VISION_LOCAL_FIT_W", 0.135);    // 灯条中心间距（m）
        fcfg.gimbal_pitch = 0.0;     // 每帧按实际云台俯仰更新（见主循环）
        fcfg.center_init = {{ 0.0, 0.0, 3.0 }};
        // 真实裁判系统灯条有 ~15° 安装倾角：α 固定 15°（慢层只在线估计 R）。
        const double alpha0 = env_d("ULTRA_VISION_LOCAL_FIT_ALPHA_DEG", 15.0) * M_PI / 180.0;
        fcfg.pitch_offset = env_d("ULTRA_VISION_LOCAL_FIT_PITCH_OFFSET_DEG", 0.0) * M_PI / 180.0;
        local_fit_pitch_sign_ = env_d("ULTRA_VISION_LOCAL_FIT_PITCH_SIGN", 1.0);
        local_fit_r_fixed_ = (std::getenv("ULTRA_VISION_LOCAL_FIT_R_FIXED") != nullptr);
        const char* rcv = std::getenv("ULTRA_VISION_LOCAL_FIT_RCV");
        local_fit_rcv_pose_ =
            !(rcv != nullptr && (std::string(rcv) == "0" || std::string(rcv) == "off"));
        local_fitter_ = std::make_unique<auto_aim::LocalTrajectoryFitter>(fcfg);
        local_fitter_->seedSlow(alpha0, fcfg.R_prior);
        std::cout << "[local_fit] enabled (real): alpha="
                  << env_d("ULTRA_VISION_LOCAL_FIT_ALPHA_DEG", 15.0) << "deg pitch_sign="
                  << local_fit_pitch_sign_ << " R_fixed=" << local_fit_r_fixed_ << std::endl;
    }
#endif
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
    // C 板回传 yaw 的符号（configs/serial.yaml 的 yaw_sign）。2026-09-28 实测这台车的
    // 编码器 yaw 正方向与本项目约定（正 yaw = 右转）相反，配置里填了 -1；手眼标定工具
    // 读的是同一个键，两边必须一致，否则瞄准会在 yaw 上镜像。
    // 相机 pitch 的符号（**实车标定项**）。world↔camera 的约定 cameraToBaseRotation()
    // 是按**仿真器（Bevy）**的 R_home_camera = R_y(-yaw)·R_x(pitch) 推的；真机下位机的
    // pitch 与它**反相**（2026-10-10 实车 A/B：pitch_sign=-1 时重投影模型才和底盘对上，
    // 被钉住那块两种符号都贴得住，只有另外三块能暴露）。默认读 configs/serial.yaml 的
    // serial.pitch_sign，现场可用 ULTRA_VISION_PITCH_SIGN=±1 覆盖，判定看 model_fit。
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
    // ---- 实车数据集录制（可选）：ULTRA_VISION_RECORD=<目录> ----
    // 录图像 + 每帧的输入（姿态/弹速/模式）+ 我们的输出，之后可以离线用
    // tools/replay_dataset 重新跑同一套流水线，改代码/改配置后对比效果。
    //   ULTRA_VISION_RECORD_EVERY_N   每 N 帧录一帧（默认 3；相机 ~95fps → ~30fps）
    //   ULTRA_VISION_RECORD_MAX_FRAMES 最多录多少帧（默认 0 = 不限，按需设防止写满盘）
    //   ULTRA_VISION_RECORD_QUALITY   JPEG 质量（默认 85）
    auto_aim::DatasetWriter dataset;
    const char* record_dir = std::getenv("ULTRA_VISION_RECORD");
    const bool recording = record_dir != nullptr && record_dir[0] != '\0';
    if (recording) {
        const int every_n = std::getenv("ULTRA_VISION_RECORD_EVERY_N")
            ? std::atoi(std::getenv("ULTRA_VISION_RECORD_EVERY_N")) : 3;
        const int max_frames = std::getenv("ULTRA_VISION_RECORD_MAX_FRAMES")
            ? std::atoi(std::getenv("ULTRA_VISION_RECORD_MAX_FRAMES")) : 0;
        const int quality = std::getenv("ULTRA_VISION_RECORD_QUALITY")
            ? std::atoi(std::getenv("ULTRA_VISION_RECORD_QUALITY")) : 85;
        if (!dataset.open(record_dir, every_n, max_frames, quality)) {
            std::cerr << "[dataset] 打不开录制目录 " << record_dir << "，本次不录" << std::endl;
        } else {
            std::cout << "[dataset] 开始录制 → " << record_dir << "/meta.csv + {i}.jpg（每 "
                      << every_n << " 帧一录，JPEG " << quality << "）" << std::endl;
        }
    }
    const auto record_start = std::chrono::steady_clock::now();

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

    // 下位机链路 + 控制权：只有 C 板把 mode 置成"自瞄"且链路在线，才允许我们接管。
    const bool fire_enabled = std::getenv("ULTRA_VISION_DISABLE_FIRE") == nullptr;
    // 只跑识别、不接管云台（命令坐标系还没量准时用；见下发处注释）。
    const bool control_disabled = std::getenv("ULTRA_VISION_DISABLE_CONTROL") != nullptr;
    // 图像闭环的速度倍率（1.0 = 中等；大了更快但延迟回路可能振荡）
    const double servo_speed_env = std::getenv("ULTRA_VISION_SERVO_SPEED")
        ? std::atof(std::getenv("ULTRA_VISION_SERVO_SPEED")) : 1.0;
    // 图像闭环的符号（两轴都是 +1）；需要时不用重编译：
    //   ULTRA_VISION_SERVO_SIGN_YAW=-1 / _PITCH=-1
    //
    // pitch 为什么是 **+1**（2026-09-30 定案，两次独立测量同号）：
    //   ① 轴符号探针（干净、闭环冻结的正台阶）：命令 pitch +3.0° → Δel = +3.54°
    //      ⇒ d(el)/d(命令) = +1.18 > 0；
    //   ② c1.csv 全量新鲜样本回归：d(el)/d(命令pitch) = +0.27 > 0。
    //   环路稳定性：off = sp·kp·err_el，err_el = −drop − el，
    //      d(err)/d(off) = −(d(el)/d(off))·sp·kp，要负反馈必须 sp > 0。
    //   所以 sp 必须 = +1。之前填 −1 是"正反馈"：误差越推越大 → 命令一路走到
    //   限幅端（实测命令 pitch 贴 −28.07° 共 33 s）→ 现场看到的就是"pitch 完全不跟、
    //   顶到限位"。yaw 的 +1 一直是对的，所以 yaw 现在稳定。
    //   注意：早先"pitch −1 能收敛"的日志是**假收敛** —— 当时用的还是 tracker 保留的
    //   旧框（见 observed_this_frame），闭环在追一个几秒前不动的误差。
    const double servo_sign_env_yaw = std::getenv("ULTRA_VISION_SERVO_SIGN_YAW")
        ? std::atof(std::getenv("ULTRA_VISION_SERVO_SIGN_YAW")) : 1.0;
    const double servo_sign_env_pitch = std::getenv("ULTRA_VISION_SERVO_SIGN_PITCH")
        ? std::atof(std::getenv("ULTRA_VISION_SERVO_SIGN_PITCH")) : 1.0;
    // 开火闸门：图像残差（板相对视轴 + 弹道补偿后的残差）必须小于它、且足够新鲜。
    const double fire_err_tol_rad = (std::getenv("ULTRA_VISION_FIRE_ERR_DEG")
            ? std::atof(std::getenv("ULTRA_VISION_FIRE_ERR_DEG")) : 1.0) * M_PI / 180.0;
    const double fire_err_max_age_s = 0.25;
    // 图像闭环的积分增益（1/s，作用在"误差的积分"上，等效偏置上限 ±10°）。
    // 用来抵消下位机 pitch 的固有静差 —— 见图像闭环里"积分项"那段注释。
    const double servo_ki_env = std::getenv("ULTRA_VISION_SERVO_KI")
        ? std::atof(std::getenv("ULTRA_VISION_SERVO_KI")) : 1.2;
    // 误差死区（deg）：小于它就把偏置清零（"保持"），不再每拍蹭一点 —— 治"点头"。
    const double servo_deadband_rad = (std::getenv("ULTRA_VISION_SERVO_DEADBAND_DEG")
            ? std::atof(std::getenv("ULTRA_VISION_SERVO_DEADBAND_DEG")) : 0.7) * M_PI / 180.0;
    // 微分增益（s）：误差变化率的反向修正，压"上下点头"的过冲。
    const double servo_kd_env = std::getenv("ULTRA_VISION_SERVO_KD")
        ? std::atof(std::getenv("ULTRA_VISION_SERVO_KD")) : 0.03;
    // 下发命令角的**速率限幅**（deg/s）。下位机视觉角环是 Ki=0 + Kd=2100/1200，
    // D 项吃的是"目标角 − 实际角"的微商，而目标角就是我们发的值：我们一发台阶，
    // 它的 D 项立刻饱和 → 摆/甩。把命令角做成"平滑地走过去"，它的误差微商就始终很小。
    const double cmd_rate_yaw_deg_s = (std::getenv("ULTRA_VISION_CMD_RATE_DEG")
            ? std::atof(std::getenv("ULTRA_VISION_CMD_RATE_DEG")) : 150.0);
    const double cmd_rate_pitch_deg_s = (std::getenv("ULTRA_VISION_CMD_RATE_PITCH_DEG")
            ? std::atof(std::getenv("ULTRA_VISION_CMD_RATE_PITCH_DEG")) : 110.0);
    // 速率限幅随误差变化：下界（精跟踪时用）与斜率（deg/s per deg）
    const double cmd_rate_yaw_min_deg_s = 45.0;
    const double cmd_rate_pitch_min_deg_s = 35.0;
    const double cmd_rate_gain_ = 8.0;
    // 速度前馈：下位机把它直接加到速度环设定值上（`Speed_Pid(gyro, Apid_Out + CV_EV.yaw_vel)`），
    // 就是 sp_vision 下发 MPC 速度前馈的那条通道。我们发的是"命令自身的变化率"。
    // **默认开、符号 −1**：2026-09-30 现场两种符号各跑一轮，−1 的无效摆动明显更少
    // （+1 时速度环上叠加成同向 → 过冲变大）。要关：ULTRA_VISION_FF=0；
    // 换车/换固件后要改符号：ULTRA_VISION_FF_SIGN=+1。
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
              // 实车 -1（与仿真器约定反相），仿真入口不用这个值（它自己就是 +1）。
              << " pitch_sign=" << pitch_sign << std::endl;

    // ---- 下发坐标系转换（唯一一处；瞄准 / 心跳 / 速度前馈都从这里出去）----
    // 2026-09-29 实测量定：下位机 yaw/pitch 闭环比较的是 **IMU 世界系角**，不是回传的
    // 电机角（`tools/gimbal_sign_probe --world`：把世界系当前值发回去云台不动 0.2~0.9°，
    // 发电机系当前值一次跑 35.7°；台阶 1:1）。这也正是 sp_vision 的做法——它下发的
    // 就是世界系绝对角，所以不需要任何符号/零点处理。
    //
    // 形式（增量加在**实测**世界系基准上，因此不需要标定任何零点常数）：
    //   cmd = world_sign·世界角(q) + gain_sign·Δ(内部角)，Δ = 本次角 − 当前姿态角
    // 没有 IMU 样本时不发（保持），绝不能退化成"发内部电机角"那个错口径。
    // 下发线程（控制器 100 Hz）里现取**最新一包**姿态当"当前角"：
    // 这样"参考基准（世界角）"和"当前角"来自同一包，不会因为主循环/控制线程
    // 采样时刻不同而差出一小段（那段差值会直接变成命令偏置）。
    // 遥测：把"我们实际下发的通道值"暴露给主循环打印（诊断用，不参与控制）。
    std::atomic<double> sent_yaw_now{0.0};
    std::atomic<double> sent_pitch_now{0.0};
    std::atomic<int> sent_mode_now{0};
    // ---- 运行时响应自标定（见下发的 world 分支）------------------------------
    // 目标：测出"命令通道 → 云台/电机角"的 2×2 响应 M，再用 M⁻¹ 发增量。
    // 这样"符号"和"通道耦合"同时被量出来，不需要任何坐标系假设。
    std::atomic<double> probe_off_yaw{0.0};     // 叠加到命令上的探针偏置（rad）
    std::atomic<double> probe_off_pitch{0.0};
    // 轴符号探针期间**冻结图像闭环**：探针要量的是"命令 → 板在画面里的位移"这一条
    // 纯开环响应，闭环在同时改命令就量不清了（以前的探针就是栽在这里）。
    std::atomic<int> probe_freeze{0};
    std::atomic<double> minv00{1.0}, minv01{0.0}, minv10{0.0}, minv11{1.0};   // M⁻¹
    std::atomic<int> probe_ready{0};            // 1 = M 已实测（仅诊断）
    // 图像闭环的命令偏置（叠加在"当前回传角"上）与自适应增益符号。
    std::atomic<double> aim_off_yaw{0.0};
    std::atomic<double> aim_off_pitch{0.0};
    // 图像闭环的**误差本身**（rad，板相对视轴 + 弹道下坠补偿后的残差）与它的时间戳。
    // 用途只有一个但很关键：**开火闸门**。实测（2026-09-29 第二轮曲线）：板在视轴下方
    // 9.4° 时，状态行里的 `aim_err` 仍然是 0.0/0.0，下位机照样收到 mode=2 —— 因为
    // `aim_err` 来自控制器自己的快照（"目标角 − 上一条命令"），是自己跟自己比，
    // 只要命令在动它就≈0，跟"枪口指没指到板"无关。真正可用的残差只有图像这一个。
    std::atomic<double> image_err_yaw{0.0};
    std::atomic<double> image_err_pitch{0.0};
    std::atomic<double> image_err_time{-1.0};   // steady_clock 秒
    std::atomic<int> fire_gate_blocked{0};      // 统计：因为图像残差大被拦下的开火请求
    std::atomic<double> servo_sign_yaw{servo_sign_env_yaw};
    std::atomic<double> servo_sign_pitch{servo_sign_env_pitch};
    // 绝对命令（采样-保持）：只在**有新观测**时锚定一次，其余帧保持不变。
    // 关键：不能每拍用"当前回传角 + 偏置"——那等于把偏置当成持续误差，下位机会以
    // 172°/s 的步长一直推（实测：LOST 段 mcu yaw 在 −34°↔53° 乱飞）。
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
                      &sent_mode_now, &probe_off_yaw, &probe_off_pitch, &minv00, &minv01,
                      &minv10, &minv11, &probe_ready, &aim_off_yaw, &aim_off_pitch,
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
        // 依据：
        //  · 源码：回传的 absolute_angle(INS 角) 就是角度环反馈，视觉值当目标 ⇒ "回传角就是命令帧"；
        //    所以"把当前值发回去"= 误差 0 = **保持不动**（由它的闭环定义保证）；
        //  · 命令通道与云台轴的映射/耦合/符号在这台车上被证明无法靠推理或单次标定确定
        //    （两个符号都试过、都朝反方向转；运行时测 2×2 又接近奇异）。所以改成
        //    **图像闭环**：直接以"板在画面里偏离视轴多少"为误差做积分，增益符号在线自适应，
        //    并做硬限幅 —— 这样无论映射是什么，误差都必须被真正消掉（见主循环 image_servo）。
        const io::GimbalState hold_state = gimbal.state();
        double out_yaw = hold_state.yaw;
        double out_pitch = hold_state.pitch;
        if (cmd_abs_valid.load(std::memory_order_relaxed) != 0) {
            out_yaw = cmd_abs_yaw.load(std::memory_order_relaxed);
            out_pitch = cmd_abs_pitch.load(std::memory_order_relaxed);
        }
        // ---- 速度前馈：发"命令自身的角速率" ----
        // 参考 sp_vision：它下发的不仅有角度，还有 MPC 轨迹的 yaw_vel/yaw_acc，下位机把它
        // 直接加到速度环设定值上（`Gimbal_Yaw_Apid_Out + CV_EV.yaw_vel`，反馈是陀螺速度）。
        // 我们之前一直发 0，所以只有位置环在推，必然"慢半拍"（实测：板在边缘才追上）。
        // 这里用**我们自己命令的变化率**当馈：与位置同坐标系、符号自动一致，不需要任何标定。
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
            // ① 基准（hold）：探针两次实测 —— 把 (+az_x, −el_x) 的当前值发回去，云台
            //    不动（0.2~0.9°）；发电机系当前值会一次跑 35.7°。所以基准就是它。
            const double base_yaw = az_now;
            const double base_pitch = -el_now;
            // ② 增量：探针台阶实测的 2×2 响应（命令 → 电机角，单位：度/度）
            //      Δmotor_yaw   =  0.92·Δcmd_yaw − 0.90·Δcmd_pitch
            //      Δmotor_pitch =  0.075·Δcmd_yaw − 0.945·Δcmd_pitch
            //    我们要的是"电机角按内部瞄点增量走"：Δraw_yaw = −Δinternal_yaw、
            //    Δraw_pitch = +Δinternal_pitch（内部 yaw = yaw_sign·raw，pitch 同相）。
            //    对上面矩阵求逆即可。
            const double d_internal_yaw =
                auto_aim::GimbalAimer::normalizeAngle(yaw - yaw_now);   // 期望 − 当前（内部）
            const double d_internal_pitch = pitch - pitch_now;
            const double d_raw_yaw = -d_internal_yaw;
            const double d_raw_pitch = d_internal_pitch;
            // 增量用**运行时实测**的 M⁻¹ 换算；M 还没量出来之前先用探针偏置让它自己动。
            // 对照分支：基准仍是实测 hold，增量用同一套实测 M⁻¹（不再用运行时自标定，
            // 那个采样窗口太短、会把上一相位的残运动算进来，实测 det≈0.066 接近奇异）。
            out_yaw = base_yaw + 1.172 * d_raw_yaw - 1.116 * d_raw_pitch +
                probe_off_yaw.load(std::memory_order_relaxed);
            out_pitch = base_pitch + 0.093 * d_raw_yaw - 1.141 * d_raw_pitch +
                probe_off_pitch.load(std::memory_order_relaxed);
            // 速度前馈先不发（0）：位置命令本身已经由轨迹发生器平滑，而两个通道与云台
            // 轴之间存在耦合，"按通道直传角速度"会引入误差。等位置环调好再按实测的
            // 雅可比补，别在方向还没稳的时候加。
            out_yaw_vel = 0.0;
            out_pitch_vel = 0.0;
        } else {
            // command_frame: motor —— 命令 = 回传角(采样保持) + 图像闭环偏置，
            // 再加**轴符号探针**偏置（默认 0；只有跑探针时非零）。
            out_yaw += probe_off_yaw.load(std::memory_order_relaxed);
            out_pitch += probe_off_pitch.load(std::memory_order_relaxed);
            // pitch 有软限位（下位机 Pitch_Max/Min_Angle = [-0.49, +0.27] rad），
            // 探针叠加后仍要夹在范围内，否则下位机自己夹会导致"命令 ≠ 预期"。
            out_pitch = std::max(-0.49, std::min(0.27, out_pitch));
        }
        // ---- 命令角速率限幅：把"台阶"变成"斜坡" ----
        // 为什么必须做：下位机视觉角环 Ki=0、Kd=2100(yaw)/1200(pitch)，
        // D 项吃的是"目标角 − 实际角"的微商，而目标角 = 我们发的值。我们一发台阶
        // （接管瞬间、限幅跳变、误差突变都算），它的 D 项立刻饱和 → 一次大角度摆动。
        // 这里把命令角本身按速率限制：任何情况下云台都不会被要求快速大角扫。
        // 只限 `control=true` 的帧（心跳 mode=0 的值下位机不看，别干扰状态）。
        {
            const double now_send = std::chrono::duration<double>(
                std::chrono::steady_clock::now().time_since_epoch()).count();
            if (control && cmd_rate_last_t > 0.0) {
                const double dt = now_send - cmd_rate_last_t;
                if (dt > 1e-4 && dt < 0.2) {
                    // 速率限幅**随误差变化**：误差大（捕获/目标跑掉）时快，误差小（精跟踪）
                    // 时慢。原来两档都是常数（90/60°/s），捕获一个 20° 外的目标要 0.25 s+，
                    // 板在画面边缘多待一会儿就容易丢 —— 现场反馈"响应慢、丢目标频繁"。
                    // 上限仍低于下位机自己的 172°/s 斜坡，且只在误差 >15° 时才用满。
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
        // 只有"板确实落在视轴附近（图像残差小）且这个残差是刚测的"才允许 mode=2。
        // 阈值取 0.0175 rad ≈ 1.0°（收敛段的实测残差是 0.3° 左右，留 3 倍余量）；
        // 残差新鲜度 0.25 s（检测 ~60 Hz，正常不会走到这个上限）。
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
        // "当前姿态"写进同一变量，日志里就分不清"命令"和"姿态"了。
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
    // 装甲板安装倾角：默认取 configs/tracker.yaml 的 pnp.armor_pitch_deg（真机 15°），
    // 现场可覆盖（换车/换支架要重标）。重投影整车轮廓与局部拟合器的 slow α 用同一个数。
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
        // 局部拟合器要按针孔重投影：把镜头畸变给 Tracker，push 灯条端点时先去畸变。
        if (dist_coeffs.rows * dist_coeffs.cols >= 4) tracker.setCameraDistortion(dist_coeffs);
    }

#ifdef ULTRA_VISION_HAVE_LOCAL_FIT
    if (local_fitter_) {
        auto* fitter = local_fitter_.get();
        const bool r_fixed = local_fit_r_fixed_;
        tracker.setLocalFitter([fitter, r_fixed](const std::vector<auto_aim::LightBarObservation>& obs,
                                                 auto_aim::LocalFitResult& out) {
            // 默认：快层 + 在线慢层（R 在线估计）；r_fixed 时退回写死 R 做 A/B。
            return r_fixed ? fitter->fit(obs, fitter->slowAlpha(), fitter->slowR(), out)
                           : fitter->fitWithSlow(obs, out);
        });
        tracker.startLocalFitWorker();
    }
#endif

    // 真机观测记录（可选）：与仿真同格式，便于离线对账"重投影残差多大、集中在哪"。
    auto observation_recorder = std::make_unique<auto_aim::app::CsvSink>(
        "ULTRA_VISION_OBSERVATION_CSV",
        "x,y,z,rx,ry,rz,tracker_x,tracker_y,tracker_z,cmd_yaw,cmd_pitch,time_us,plate_id,state,"
        "u0,v0,u1,v1,u2,v2,u3,v3,source_time_us\n");

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
    // 运行时响应自标定的状态（见主循环里 "self-cal" 一段）
    int probe_phase = 0;
    int probe_samples = 0;
    double probe_t0 = 0.0;
    double probe_az[2] = {0.0, 0.0};
    double probe_el[2] = {0.0, 0.0};
    double probe_d_az_yaw = 0.0, probe_d_el_yaw = 0.0;
    double probe_d_az_pitch = 0.0, probe_d_el_pitch = 0.0;
    // ---- 曲线用的"最近一次量测"（给下面的曲线面板用；没观测时保持不变）----
    // 这几个量都在图像闭环那一段算过一遍，这里只做记录，不再重复算像素→角度。
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
    // 被硬钉那块永远≈0，所以这个数主要反映"另外几块板对不对" —— 也就是
    // 底盘模型/姿态约定对不对（正确时典型值 ≲10 px，约定错了会几十上百 px）。
    double model_fit_sum = 0.0;
    double model_fit_max = 0.0;
    int model_fit_n = 0;
    auto report_time = std::chrono::steady_clock::now();
    // ------------------- 3.5 实时曲线（和仿真 node_sim 同一套组件）----------
    // 硬件上一直缺的正是这个：图像框只能说明"这一帧对上了"，而"命令 vs 回传、
    // 板相对视轴的角、命令偏置"是**时间序列**问题 —— 接管瞬间 pitch 抬升、
    // 采样保持导致的过冲、符号反了造成的正反馈，只有在曲线上才看得见。
    // 面板（同面板共用一条纵轴，才能看两条线的高低差）：
    //   panel0  回传 yaw  vs  我们下发的 yaw
    //   panel1  回传 pitch vs 我们下发的 pitch
    //   panel2  板相对视轴 az/el（0 = 正中；收敛就是两条线回到 0）
    //   panel3  命令偏置 off yaw/pitch（比例项；有静差 = 弹性，发散 = 一直变大）
    const char* curves_env = std::getenv("ULTRA_VISION_CURVES");
    const bool curves_enabled = curves_env != nullptr
        ? std::string(curves_env) != "0"
        : (std::getenv("ULTRA_VISION_CURVE_CSV") != nullptr ||
           std::getenv("ULTRA_VISION_NO_DISPLAY") == nullptr);
    auto_aim::visualization::CurvePlotter curves(600);
    const auto curve_start = std::chrono::steady_clock::now();
    curves.addCurve("mcu_yaw [deg]", cv::Scalar(80, 220, 255), 90.0, 0);
    curves.addCurve("cmd_yaw [deg]", cv::Scalar(80, 255, 80), 90.0, 0);
    curves.addCurve("mcu_pitch [deg]", cv::Scalar(80, 180, 255), 45.0, 1);
    curves.addCurve("cmd_pitch [deg]", cv::Scalar(255, 180, 80), 45.0, 1);
    curves.addCurve("plate_az [deg]", cv::Scalar(255, 80, 255), 30.0, 2);
    curves.addCurve("plate_el [deg]", cv::Scalar(200, 200, 200), 30.0, 2);
    curves.addCurve("off_yaw [deg]", cv::Scalar(0, 200, 255), 15.0, 3);
    curves.addCurve("off_pitch [deg]", cv::Scalar(255, 120, 120), 15.0, 3);
    // 下位机回传的**陀螺角速度**（pitch 那一路正是它速度环的反馈量）。加进来是为了
    // 量"我们发一个角增量，它到底往哪边、以多快动"：如果回传角与陀螺符号在某个模式下
    // 反号，光看角度是看不出来的。
    curves.addCurve("mcu_yaw_gyro [deg/s]", cv::Scalar(120, 255, 200), 200.0, 4);
    curves.addCurve("mcu_pitch_gyro [deg/s]", cv::Scalar(255, 200, 120), 200.0, 4);
    // 只落 CSV 的诊断列（不画图，避免不同量纲的线挤在一格里）：
    //   ctrl        我们发的是 mode=1 控制帧 还是 mode=0 心跳（"突然不收敛"最常见的原因）
    //   state       tracker 状态（0 LOST / 1 DETECTING / 2 TEMP_LOST / 3 TRACKING）
    //   obs_age_ms  最近一次被接受观测的年龄（大 = 识别不出框或关联被拒）
    //   dist_m      PnP 距离（现场反馈"距离一远就扰动"——得有距离这一列才能验证）
    //   box_h_px    识别框高度（像素，
    //               量测质量/角噪声的量度，远距离调增益时用）
    curves.addLogColumn("ctrl");
    curves.addLogColumn("state");
    curves.addLogColumn("obs_age_ms");
    curves.addLogColumn("dist_m");
    curves.addLogColumn("box_h_px");
    if (const char* curve_csv = std::getenv("ULTRA_VISION_CURVE_CSV")) {
        // 第一列写时间戳（秒）：离线要能把曲线和 1Hz 状态行、真值对齐，光有帧号对不上。
        if (curves.openCsv(curve_csv, /*with_time=*/true)) {
            std::cout << "[curve] CSV → " << curve_csv << "（第一列 t，单位 s）" << std::endl;
        }
    }
    const bool show_display = std::getenv("ULTRA_VISION_NO_DISPLAY") == nullptr;
    // ------------------- 3.6 一次性"轴符号"探针 ---------------------------
    // 命令帧/符号这台车已经反复猜错（两版符号都试过、都朝反方向转），但"哪个符号对"
    // 其实是个**可测的量**，而且不需要任何坐标系假设：冻结图像闭环，只给 pitch 通道
    // +3° 保持 1s，看板在画面里往哪边走（Δel）；再只给 yaw 通道 +3°，看 Δaz。
    //   Δel < 0（板下移） ⇒ 命令 + = 枪口上抬 ⇒ 偏置要用 off = +k·el
    // 用它把 ULTRA_VISION_SERVO_SIGN_* 定死，后面不再靠试。
    const bool axis_probe_enabled = std::getenv("ULTRA_VISION_AXIS_PROBE") != nullptr;
    struct AxisProbe
    {
        int phase = 0;
        int settle = 0;
        int n = 0;
        double t0 = 0.0;
        double sum_az = 0.0;
        double sum_el = 0.0;
        double base_az = 0.0;
        double base_el = 0.0;
        double d_el_pitch = 0.0;
        double d_az_yaw = 0.0;
    };
    AxisProbe axis_probe;
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
        {
            static double last_obs_time = 0.0;
            if (last_obs_time > 0.0 && observation_time > last_obs_time) {
                const double dt = observation_time - last_obs_time;
                if (dt > 0.001 && dt < 0.2) servo_dt_ = dt;
            }
            last_obs_time = observation_time;
        }

        // ------------------- 6. 下位机姿态 + 整车估计 -------------------
        // 观测是在**相机系**里解出来的，要放进底盘/世界系，必须知道拍照那一刻的
        // 相机姿态：yaw/pitch 用下位机回传的云台绝对角（它的 gimbal_yaw_motor.
        // absolute_angle），base_to_world 可选地用 IMU 四元数（底盘小陀螺时估计
        // 不会跟着底盘转）。这一步不做，估计就永远活在相机系里 —— 底盘一转就漂。
        const double now_seconds = auto_aim::StandardClock::nowSeconds();
        const io::GimbalState gimbal_state = gimbal.state();
        auto_aim::CameraPose camera_pose;
        camera_pose.camera_to_gimbal = extrinsics.rotation;
        camera_pose.camera_to_gimbal_translation = extrinsics.translation;
        // ---- 关键：姿态必须取**曝光时刻**的，不能取"处理到这一帧时"的最新姿态 ----
        // 观测是曝光那一刻的图像，而本函数跑在检测之后（延迟 = 曝光→到手，几十毫秒）。
        // 若用最新姿态去把观测映射到世界系，云台一转动就会产生"图像-姿态错位"
        // （错位角 ≈ 转速 × 延迟）：估计出来的世界系目标被整体旋转 → 瞄点为了追它
        // 继续转 → **自己追自己，一路跑到限位**（2026-09-29 实车现象：pitch 两个符号
        // 都跑到限位、只是方向相反，正是这种正反馈的特征）。
        // io::Gimbal 按时间戳缓存了姿态历史（slerp + 角度线性插值），这里就是为它准备的。
        double frame_age = now_seconds - observation_time;
        // 时间基不同（相机 SDK 的时钟和本地单调钟不是同一纪元）时这个值会离谱：
        // 那种情况下宁可退回"最新姿态"，也不要用一个几十秒前的姿态去映射观测。
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
        // （被显示的那张图就是刚到手的新帧）。观测姿态只用于把观测映射进世界系；
        // 画模型必须用当前姿态，否则云台在观测延迟里动过的角度会整片加到模型上。
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
        // 安全闸：命令坐标系/符号还没量准之前，用 ULTRA_VISION_DISABLE_CONTROL=1 只跑识别，
        // 云台不接管（只发 mode=0 心跳）。识别/估计/曲线照常，动云台的动作一概不发。
        if (control_disabled) allow_control_now = false;
        frame_input.allow_control = allow_control_now;
        frame_input.fire_enabled = fire_enabled;
#ifdef ULTRA_VISION_HAVE_LOCAL_FIT
        if (local_fitter_) {
            // R_cv 每帧更新：默认用 pose 推的**完整**旋转（含手眼外参 + 云台俯仰）；
            // 关掉 rcv 时退回 Rx(gimbal_pitch·sign + pitch_offset)·R_base 的旧口径做 A/B。
            auto fcfg = local_fitter_->config();
            bool changed = false;
            if (local_fit_rcv_pose_) {
                const auto R = tracker.fitterRcv();
                if (!fcfg.use_rcv || fcfg.rcv != R) { fcfg.use_rcv = true; fcfg.rcv = R; changed = true; }
            } else {
                const double gp = local_fit_pitch_sign_ * camera_pose.pitch;
                if (fcfg.use_rcv) { fcfg.use_rcv = false; changed = true; }
                if (std::abs(fcfg.gimbal_pitch - gp) > 1e-4) { fcfg.gimbal_pitch = gp; changed = true; }
            }
            if (changed) local_fitter_->setConfig(fcfg);
            for (const auto& a : frame_input.armors)
                tracker.pushArmorLightBars(a, observation_time, tracker.plateIdFor(a));
        }
#endif
        const auto outcome = pipeline.update(frame_input);
#ifdef ULTRA_VISION_HAVE_LOCAL_FIT
        if (local_fitter_) tracker.applyLatestLocalFit();
#endif
        // 观测记录（可选）：真机也落 observation.csv，离线即可对账重投影残差。
        if (*observation_recorder && tracker.hasObservation() &&
            !tracker.getLastObservedArmor().tvec.empty()) {
            const auto& observed = tracker.getLastObservedArmor();
            const cv::Mat center = tracker.getTargetCenter();
            auto& s = observation_recorder->stream();
            s << std::fixed << std::setprecision(6)
              << observed.tvec.at<double>(0) << ',' << observed.tvec.at<double>(1) << ','
              << observed.tvec.at<double>(2) << ',' << observed.rvec.at<double>(0) << ','
              << observed.rvec.at<double>(1) << ',' << observed.rvec.at<double>(2) << ','
              << center.at<double>(0) << ',' << center.at<double>(1) << ',' << center.at<double>(2)
              << ',' << gimbal_state.yaw << ',' << gimbal_state.pitch << ','
              << source_result.timestamp_us << ',' << tracker.getLastObservedPlateId() << ','
              << static_cast<int>(tracker.getState());
            if (observed.Points_2D.size() >= 4) {
                for (int i = 0; i < 4; ++i)
                    s << ',' << observed.Points_2D[static_cast<std::size_t>(i)].x << ','
                      << observed.Points_2D[static_cast<std::size_t>(i)].y;
            } else {
                for (int i = 0; i < 8; ++i) s << ",nan";
            }
            s << ',' << source_result.timestamp_us << '\n';
        }
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
        // 的模型框（同一颜色/同一 ID 风格）来表示，不再单独画一个抢眼的绿框。
        for (const auto& armor : armors) {
            cv::line(image, armor.left.top, armor.right.bottom, cv::Scalar(0, 140, 0), 1);
            cv::line(image, armor.left.bottom, armor.right.top, cv::Scalar(0, 140, 0), 1);
        }

        const bool has_observation = tracker.hasObservation();
        const auto_aim::Armor& observed_armor = tracker.getLastObservedArmor();
        // ---- "只认最新观测"：tracker 在没有新框的帧上会保留上一帧的 armors
        // （hasObservation 仍为真），拿它算图像误差 = 用旧数据一直推云台 —— 实测这正是
        // pitch 被一路推到限位的原因（mcu_pitch=−28°，板却按旧框报 el=−7°，几何上不可能，
        // 于是闭环一直往同一个方向推）。做法：要求 tracker 接受的观测就是**这一帧**的。
        const bool observed_this_frame =
            std::abs(tracker.lastObservationTime() - observation_time) < 0.001;
        const double observation_age = now_seconds - tracker.lastObservationTime();
        // ================= 图像闭环（最终的瞄准实现）=============================
        // 误差 = 板相对视轴的方位/俯仰（相机系直接测出来，与任何坐标系约定无关）。
        // 目标：把它打到 0（俯仰留一点弹道下坠的等效量，让板略低于视轴）。
        // 控制：对命令偏置做积分；积分增益的**符号**在线自适应 —— 如果推了几下误差反而变大，
        // 就把该轴符号翻过来。命令偏置与最终命令都做硬限幅，保证即使自适应判断错了也不会甩飞。
        // 关键：**必须要求本帧有新的推理结果**（last_source_fresh_）。
        // tracker.hasObservation() 在"没有新检测的帧"上不会清零，用它会导致
        // "拿旧检测算偏差 + 把命令锚定到新姿态" = 相对命令 ⇒ 下位机的误差恒等于偏置
        // ⇒ 以 172°/s 一直转（实测：画面完全没收敛，而日志里的偏差却看着收敛了）。
        if (has_observation && last_source_fresh_ && observed_this_frame && allow_control_now &&
            tracker.getState() == auto_aim::TrackerState::TRACKING) {
            const cv::Mat& tvec = observed_armor.tvec;
            if (observed_armor.solve_result && tvec.rows >= 3 &&
                std::abs(tvec.at<double>(2)) > 1e-6) {
                // 偏角用**像素框中心**算（内参换算），不用 PnP 的 tvec：
                // 平面装甲板的 PnP 有两个解，同一块板会来回跳，实测方位差 ±12°（这正是
                // "有时候左有时候右"的来源）；而框位置是稳的、且像素→角度无二义。
                const auto& bar_l = observed_armor.left;
                const auto& bar_r = observed_armor.right;
                const double u = 0.25 * (bar_l.top.x + bar_l.bottom.x + bar_r.top.x + bar_r.bottom.x);
                const double v = 0.25 * (bar_l.top.y + bar_l.bottom.y + bar_r.top.y + bar_r.bottom.y);
                double az = std::atan2(u - camera_matrix.at<double>(0, 2),
                                       camera_matrix.at<double>(0, 0));   // +右
                double el = -std::atan2(v - camera_matrix.at<double>(1, 2),
                                        camera_matrix.at<double>(1, 1));  // +上
                // **只用最新这一帧的测量，不做 EMA 平滑**（用户提的"只取最新的"）。
                // 之前用 0.6/0.4 的指数平滑，等于在本就有的曝光+推理延迟上再加 ~40 ms 滞后；
                // 环路稳定余量正是被这段滞后吃掉的（公式：等效增益 g=Kp·f，稳定要求 g·τ≲1）。
                // 噪声由后面的积分/限幅消化，不靠平滑。
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
                // ---- 跳变门：识别到的板在物理板之间跳时，单帧偏差会突跳 ±12~16°。
                // 突变先不接受（累计到连续 0.4s 才重新锚定），避免把积分推飞。
                constexpr double kJumpGate = 6.0 * M_PI / 180.0;
                if (servo_have_prev_ &&
                    (std::abs(err_az - servo_prev_err_az_) > kJumpGate ||
                     std::abs(err_el - servo_prev_err_el_) > kJumpGate)) {
                    if (std::abs(err_az - servo_prev_err_az_) > kJumpGate) ++servo_jump_frames_yaw_;
                    if (std::abs(err_el - servo_prev_err_el_) > kJumpGate) ++servo_jump_frames_pitch_;
                    // 连续太多帧都"突变"说明真的换了目标（不是跳变）→ 重新锚定
                    // 连续 2s 都是"突变"才认为是真的换了目标（而不是板跳变）。
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
                // 实测（2026-09-29 曲线 CSV）：有些时段命令离回传 22°，回传却 1 s 纹丝不动，
                // 而图像闭环还在把偏置堆到 ±8°。下位机的 `tj_t.mode` 是**硬编码 1**，我们
                // 看不到它是不是真的在"纯视觉接管"（mode 0/1 时它根本不看我们的值）。
                // 判据：连续 40 次新观测里"命令 − 回传"的最大分量都不再缩小且 > 3°，
                // 就认为这拍没权限 → 本帧只发"保持"（= 当前回传角），不再堆偏置。
                // **只对 yaw 生效**。pitch 是另一回事：电控源码里 pitch 的两个环都是
                // Ki=0 / IMaxout=0（gimbal_task.h: Gimbal_Pitch_Vision_Angle_Ki=0,
                // _IMaxout=0, Gimbal_Pitch_Speed_Ki=0），重力补偿那行调用又是注释掉的
                // （gimbal_task.c:1873），所以 pitch **结构上就存在 1~4° 的静差**
                // （实测：命令与回传差 3.43° 保持好几秒不动）。对 pitch 用"差 3° 就判定
                // 没权限"会把我们该积累的补偿量一次次清零 —— 实测那样 pitch 会卡在
                // el≈-9.4° 不动。yaw 没有重力、能收敛（实测残差 0.33°），才用它判权限。
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
                        // **必须把之前积攒的东西清零重来**。否则遥控阶段憋出来的偏置会在
                        // 接管瞬间一次性砸下去 —— 实测现场就是"遥控→纯视觉一接管就飘"。
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
                //  P：误差一大立刻给大偏置（即时响应，主要提升就来自这里）
                //  D：误差在变（靶板被手带/车在动）时给速度前馈，跟得更紧
                //  I：消静差
                // ---- sp_vision 式结构：命令 = 当前回传角 + 有界比例修正 ----
                // 关键差别（也是前几轮发散的原因）：
                //   ✗ 之前：把偏差**累积**到独立积分状态，再锚定到"曝光时刻姿态" ⇒
                //          "目标 − 当前"恒等于该累积值 ⇒ 下位机每拍 172°/s 继续推 ⇒ 发散。
                //   ✓ 现在：锚点就是**当前回传角本身**（= 下位机的反馈，"发当前值即不动"，
                //          源码与探针都验证过），修正量 = Kp×偏差，**有界、不累积** ⇒
                //          误差大就多推一点、误差小就少推，结构上不可能跑飞。
                // ---- 增益：原来 0.5/0.4 **太大**，这就是"误差一直累积到飘飞"的真因 ----
                // 每拍都把 off 加到**当前回传角**上，等于把"误差×Kp"以 60 Hz 灌给一个
                // 带延迟的执行机构：等效连续增益 g = Kp × f_obs（实测 f_obs≈57 Hz）。
                // 执行机构这边还有 ~0.06~0.1 s 的延迟 τ（曝光+推理+下位机 172°/s 斜坡），
                // 稳定条件是 g·τ ≲ 1。原来 Kp=0.5 ⇒ g≈28 /s ⇒ g·τ≈2~3 ⇒ **结构性振荡**。
                // 实测（run2，74.6 s 曲线）：我们的命令 yaw 只摆 40.9°，回传却摆 78.6°，
                // 图像 |az| 中位从前半段 1.2° 涨到后半段 9.7° —— 振荡在**增大**，
                // 直到超出门限甩飞；而"一开机就是纯视觉"误差本来就小，所以看起来能收敛。
                // 取 Kp = 0.12/0.10 ⇒ g≈6.8/5.7 /s ⇒ g·τ≈0.4~0.7，留足余量。
                const double kpScale = servo_speed_;
                // 0.12/0.10 → 0.20/0.16：现场反馈"收敛不够快、响应不够快"。
                // 闭环等效时间常数 ≈ 1/(Kp·f_obs)（f_obs≈45 Hz）：
                //   0.12 → 0.19 s，0.20 → 0.11 s。
                // 稳定性余量 g·τ（τ≈0.1 s：曝光+推理+下位机 172°/s 斜坡）：
                //   0.20 → 0.9、0.16 → 0.72，仍在 1 以内；大误差那一段另有
                //   pipe_send 的**随误差变化的速率限幅**（≤150/110°/s）兜着，
                //   小误差那一段有死区停车，所以这次才敢往上加。
                // 若现场还嫌快/有振，`ULTRA_VISION_SERVO_SPEED=0.7` 直接等比回退。
                const double kpYaw = 0.20 * kpScale;
                const double kpPitch = 0.16 * kpScale;
                // ---- 积分项：专门治 pitch 的结构性静差 ----------------------
                // 电控 pitch 的角度环/速度环**都是 Ki=0、IMaxout=0**（gimbal_task.h），
                // 重力补偿调用又是注释掉的（gimbal_task.c:1873），所以 pitch 天生有
                // pose 相关的静差 bias（实测 0.4°~3.4°：回传角总比命令高那么多）。
                // 而"纯比例 + 每拍按回传角重锚"（cmd = 回传 + off）的稳态是
                //     Δel = kp·err − bias = 0  ⇒  err = bias/kp = 2.5·bias
                // 实测 run2：bias=3.43°，于是稳态 err=8.5° ⇒ el 卡在 −9.4° 一动不动，
                // 和观测完全一致（不是符号问题，是"没有积分就消不掉静差"）。
                // 加上积分项后它会自己涨到把 bias 抵消，稳态 err → 0。
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
                // 而且误差很大（>8°）时**冻结**积分 —— 追大误差时不该继续往同一个方向攒，
                // 否则就是实测那样"偏置打满 ±20°、命令被顶到限幅、pitch 贴在限位上"。
                constexpr double kIntegralOffsetLimit = 5.0 * M_PI / 180.0;   // ki·I ≤ 5°
                const double kIntegralLimit =
                    kIntegralOffsetLimit / std::max(0.1, servo_ki_);
                // 阈值放到 15°：符号已经定了（两轴 sp=+1，见上面的推导），现在积分是
                // **负反馈**里的消静差项，可以早点介入；幅值上限仍然是 ±5°。
                const bool integrate_ok = authority_ok &&
                    std::abs(err_az) < 15.0 * M_PI / 180.0 &&
                    std::abs(err_el) < 15.0 * M_PI / 180.0;
                // 死区里不积分（见下面"误差死区"的注释）：否则积分会继续涨，涨到把云台
                // 从停住的位置又推走 —— 那正好是"点头"的成因。
                const bool dead_yaw = std::abs(err_az) < servo_deadband_rad;
                const bool dead_pitch = std::abs(err_el) < servo_deadband_rad;
                if (integrate_ok && !dead_yaw) {
                    servo_int_yaw_ += err_az * dt_obs;
                    servo_int_yaw_ = std::max(-kIntegralLimit, std::min(kIntegralLimit, servo_int_yaw_));
                }
                if (integrate_ok && !dead_pitch) {
                    servo_int_pitch_ += err_el * dt_obs;
                    servo_int_pitch_ =
                        std::max(-kIntegralLimit, std::min(kIntegralLimit, servo_int_pitch_));
                }
                // ---- 微分项：专门压"上下点头"（只对有静摩擦的 pitch 明显）---------
                // 现象：yaw 收敛好，pitch 在两三度范围内以 ~1 Hz 点头。这是"积分+静摩擦"
                // 的典型极限环：积分一点点涨到能推动轴 → 轴一动就过冲（实测传递 ~1.5~2×）
                // → 误差反向 → 积分再往回涨 …… 周期由积分增益决定（ki=2/s ⇒ ~1 s）。
                // 加一项"误差变化率"的反向修正（标准 PD 的 D），用两次**新鲜**测量算，
                // 再做一阶低通（裸微分在 57 Hz 上噪声太大）。
                double rate_az = 0.0;
                double rate_el = 0.0;
                if (servo_have_prev_ && dt_obs > 1e-3) {
                    rate_az = (az - prev_az) / dt_obs;
                    rate_el = (el - prev_el) / dt_obs;
                }
                servo_rate_az_ = 0.7 * servo_rate_az_ + 0.3 * rate_az;
                servo_rate_el_ = 0.7 * servo_rate_el_ + 0.3 * rate_el;
                // ---- 误差死区：小于它就把偏置清零（= 发"保持"，云台停在原地）------
                // 在死区里继续输出偏置 = 每拍都让下位机再走一点（本项目的结构是
                // cmd = 回传角 + off），那就是连续"点头"的来源。清零后它才真正停住。
                // （dead_yaw / dead_pitch 在上面算积分时已经用到。）
                double off_yaw_new = sy * (kpYaw * err_az + servo_ki_ * servo_int_yaw_ -
                                           servo_kd_ * servo_rate_az_);
                double off_pitch_new = sp * (kpPitch * err_el + servo_ki_ * servo_int_pitch_ -
                                             servo_kd_ * servo_rate_el_);
                if (dead_yaw) off_yaw_new = 0.0;
                if (dead_pitch) off_pitch_new = 0.0;
                // ---- 每拍步长限幅（抗极限环的关键，比总幅值限幅更重要）----
                // 下位机是 172°/s 的斜坡 + 我们这边 ~0.1 s 的总延迟；如果**一拍**的偏置
                // 跳变就超过"它在延迟时间内能走完的量"，它必然过冲 → 误差反向 → 再跳回来，
                // 就是实测那个 2~4 s 周期、幅度越来越大的摆动。按 57 Hz×172°/s 折算，
                // 一拍最多允许它走 3.0°，我们限 2.0°/拍（≈114°/s）留余量。
                // 步长上限只当"防呆"，真正的平滑由 pipe_send 里**随误差变化**的速率限幅
                // （≤150/110°/s，随误差线性缩放）负责 —— 之前 3°/2° 的硬步长在捕获段成了
                // 瓶颈（0.12 增益时 20° 误差的合理需求就有 2.4°，被砍到 2°）。
                const double kMaxStepYaw = 6.0 * M_PI / 180.0;
                const double kMaxStepPitch = 4.0 * M_PI / 180.0;
                off_yaw_new = std::max(servo_prev_off_yaw_ - kMaxStepYaw,
                                       std::min(servo_prev_off_yaw_ + kMaxStepYaw, off_yaw_new));
                off_pitch_new = std::max(servo_prev_off_pitch_ - kMaxStepPitch,
                                         std::min(servo_prev_off_pitch_ + kMaxStepPitch,
                                                  off_pitch_new));
                // 幅值硬限幅：正常瞄准的偏置只有几度（实测收敛段 0.2~0.4°，接管瞬间 3~4°），
                // 收紧到 ±30°/±20° 只是防呆，保证任何 bug 都不会变成甩飞。
                const double kMaxOffYaw = 30.0 * M_PI / 180.0;
                const double kMaxOffPitch = 20.0 * M_PI / 180.0;
                double off_yaw = std::max(-kMaxOffYaw, std::min(kMaxOffYaw, off_yaw_new));
                double off_pitch = std::max(-kMaxOffPitch, std::min(kMaxOffPitch, off_pitch_new));
                // 记录给曲线面板：板相对视轴的角、比例偏置、弹道下坠等效角。
                // 这三条放一起就能看出"接管瞬间 pitch 抬升"到底是偏置的方向错了，
                // 还是板本来就在视轴上方需要抬。
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
                // 之后直到下一次有新观测都保持不变（采样-保持），云台收敛到该角度并停住。
                {
                    const double gimbal_state_now_yaw = gimbal.state().yaw;
                    const double gimbal_state_now_pitch = gimbal.state().pitch;
                    const double raw_pose_yaw = gimbal_state_now_yaw;
                    const double raw_pose_pitch = gimbal_state_now_pitch;
                    const double pitch_lo = -0.49;
                    const double pitch_hi = 0.27;
                    // 轴符号探针期间不改闭环命令（见 probe_freeze 注释）。
                    if (probe_freeze.load(std::memory_order_relaxed) == 0) {
                        // 没权限时退化成"保持当前姿态"：不让偏置继续堆（否则恢复接管的瞬间会甩）。
                        const double out_off_yaw = authority_ok ? off_yaw : 0.0;
                        const double out_off_pitch = authority_ok ? off_pitch : 0.0;
                        cmd_abs_yaw.store(raw_pose_yaw + out_off_yaw, std::memory_order_relaxed);
                        cmd_abs_pitch.store(
                            std::max(pitch_lo, std::min(pitch_hi, raw_pose_pitch + out_off_pitch)),
                            std::memory_order_relaxed);
                        cmd_abs_valid.store(1, std::memory_order_relaxed);
                    }
                }
                // ---- 符号自适应：Σ(Δ偏置 × Δ误差) > 0 表示"越推误差越大" = 符号反了 ----
                // 自适应只在"误差还大(>4°) + 偏置在动"时累计：板被手移动造成的 Δerr 不会
                // 稳定同号，窗口拉长到 150 个观测、门限提高，且每轴最多翻一次（实测：窗口
                // 60 帧时出现过一次误翻转，越翻越偏）。
                // 自动翻转**已停用**：实测它会误触发（pitch 明明在收敛却判"推反了"）。
                // 符号改为启动时由 ULTRA_VISION_SERVO_SIGN_YAW/PITCH 指定（默认 +1，见上）。
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
                // 之前每 50 个观测打一行，间隔 ≈1~2s，正好和摆动周期混叠，把 ±14° 的
                // 摆动采样成"稳定在 −1°"（所以日志和画面看着矛盾）。极值能藏不住摆动。
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
        // ------------------- 7.1 轴符号探针（ULTRA_VISION_AXIS_PROBE=1）--------
        // 只在稳定跟踪后跑一次；期间 probe_freeze=1（图像闭环不改命令），
        // 探针偏置经 pipe_send 叠加到"保持中的命令"上 → 纯开环台阶。
        if (axis_probe_enabled && axis_probe.phase < 5) {
            AxisProbe& pb = axis_probe;
            const double pb_now = std::chrono::duration<double>(
                std::chrono::steady_clock::now().time_since_epoch()).count();
            const bool usable = has_observation && last_source_fresh_ && allow_control_now &&
                tracker.getState() == auto_aim::TrackerState::TRACKING;
            const auto accumulate = [&]() {
                if (usable) {
                    pb.sum_az += curve_az_deg;
                    pb.sum_el += curve_el_deg;
                    ++pb.n;
                }
            };
            const auto mean_az = [&]() { return pb.n > 0 ? pb.sum_az / pb.n : pb.base_az; };
            const auto mean_el = [&]() { return pb.n > 0 ? pb.sum_el / pb.n : pb.base_el; };
            const auto restart = [&](int phase, double hold) {
                pb.phase = phase;
                pb.t0 = pb_now;
                pb.n = 0;
                pb.sum_az = 0.0;
                pb.sum_el = 0.0;
                (void)hold;
            };
            constexpr double kStep = 3.0 * M_PI / 180.0;
            switch (pb.phase) {
            case 0:   // 等"60 次可用观测"（不能要求连续：异步推理下 fresh 会在 0/1 之间交替，
                // 上一版要求"连续 60 帧"实测永远等不到，探针根本没跑起来）
                if (usable) {
                    pb.settle = pb.settle + 1;
                    pb.t0 = pb_now;   // 复用 t0 记"最近一次可用时刻"
                } else if (pb_now - pb.t0 > 0.5) {
                    pb.settle = 0;    // 超过 0.5s 没观测才算断
                }
                if (pb.settle >= 60) {
                    probe_freeze.store(1, std::memory_order_relaxed);
                    std::cout << "[probe] 开始轴符号测量：先 0.4s 基准，再单独给 pitch/yaw "
                                 "各 +3°（图像闭环已冻结）" << std::endl;
                    restart(1, 0.4);
                }
                break;
            case 1:   // 基准 0.4s
                accumulate();
                if (pb_now - pb.t0 >= 0.4) {
                    pb.base_az = mean_az();
                    pb.base_el = mean_el();
                    probe_off_pitch.store(kStep, std::memory_order_relaxed);
                    std::cout << "[probe] 基准 az=" << pb.base_az << "° el=" << pb.base_el
                              << "°（" << pb.n << " 帧）→ pitch 通道 +3° 保持 1.0s" << std::endl;
                    restart(2, 1.0);
                }
                break;
            case 2:   // pitch +3°，1.0s
                accumulate();
                if (pb_now - pb.t0 >= 1.0) {
                    pb.d_el_pitch = mean_el() - pb.base_el;
                    probe_off_pitch.store(0.0, std::memory_order_relaxed);
                    std::cout << "[probe] pitch +3° → el " << pb.base_el << "° ⇒ " << mean_el()
                              << "°（Δ=" << pb.d_el_pitch << "°，" << pb.n << " 帧）" << std::endl;
                    restart(3, 0.4);   // 回零稳定
                }
                break;
            case 3:   // 回零 0.4s
                accumulate();
                if (pb_now - pb.t0 >= 0.4) {
                    pb.base_az = mean_az();
                    probe_off_yaw.store(kStep, std::memory_order_relaxed);
                    std::cout << "[probe] 回到基准 az=" << pb.base_az << "° → yaw 通道 +3° 保持 1.0s"
                              << std::endl;
                    restart(4, 1.0);
                }
                break;
            case 4:   // yaw +3°，1.0s
                accumulate();
                if (pb_now - pb.t0 >= 1.0) {
                    pb.d_az_yaw = mean_az() - pb.base_az;
                    probe_off_yaw.store(0.0, std::memory_order_relaxed);
                    probe_freeze.store(0, std::memory_order_relaxed);
                    const double sp_now = servo_sign_pitch.load(std::memory_order_relaxed);
                    const double sy_now = servo_sign_yaw.load(std::memory_order_relaxed);
                    const char* sp_rec = pb.d_el_pitch < -0.5
                        ? "-1（命令+ = 枪口上抬；用 off=+k·el）"
                        : (pb.d_el_pitch > 0.5 ? "+1（命令+ = 枪口下压）" : "测不出来（Δ 太小）");
                    const char* sy_rec = pb.d_az_yaw < -0.5
                        ? "-1（命令+ = 相机右转；用 off=+k·az）"
                        : (pb.d_az_yaw > 0.5 ? "+1（命令+ = 相机左转）" : "测不出来（Δ 太小）");
                    std::cout << "[probe] yaw +3° → az " << pb.base_az << "° ⇒ " << mean_az()
                              << "°（Δ=" << pb.d_az_yaw << "°）" << std::endl
                              << "[probe] ===== 结论 =====" << std::endl
                              << "[probe] pitch: Δel=+3°→" << pb.d_el_pitch
                              << "°，建议 SERVO_SIGN_PITCH=" << sp_rec << "（当前 " << sp_now << "）"
                              << std::endl
                              << "[probe] yaw  : Δaz=+3°→" << pb.d_az_yaw
                              << "°，建议 SERVO_SIGN_YAW=" << sy_rec << "（当前 " << sy_now << "）"
                              << std::endl
                              << "[probe] 图像闭环已恢复；把上面两个建议值写进环境变量即可。" << std::endl;
                    pb.phase = 5;
                }
                break;
            default: break;
            }
        }
        // ---- 运行时响应自标定：量出"命令通道 → 云台角"的 2×2 矩阵 -----------------
        // 为什么必须实测：这台车的下位机视觉支路既不是"对角"也不是"同坐标系"
        // （符号两版都试过、都朝反方向转；探针台阶又量到 pitch 通道会把 yaw 带走 0.9 倍）。
        // 做法：稳定跟踪后，先记"板在画面里的 az/el"当基准，再单独给 yaw 通道 +2.9°、
        // 再单独给 pitch 通道 +2.9°，看板在画面里往哪边移动 —— 这就是含符号、含耦合的响应。
        // 换算到电机/INS 角：raw_yaw 与 az_板 同号；相机抬头 → 板在画面里下移，故
        // raw_pitch 与 el_板 反号。最后用 M⁻¹ 发增量（见下发分支）。
        // **默认关闭**：这段自标定会主动把云台推 2.9°×2（"云台会小幅动两下"），而它的结果
        // 自己都注明"仅诊断、不用于下发"。实测中它每次启动都会动一次，就是现场反馈的
        // "一开控制云台就莫名转一点"。要看就显式开：ULTRA_VISION_SELFCAL=1。
        const bool selfcal_enabled = std::getenv("ULTRA_VISION_SELFCAL") != nullptr;
        if (selfcal_enabled && probe_ready.load(std::memory_order_relaxed) == 0) {
            const auto probe_now_clock = std::chrono::steady_clock::now();
            constexpr double probe_step_rad = 0.05;   // ≈2.9° 的探针台阶
            const bool probe_ok = has_observation && allow_control_now &&
                tracker.getState() == auto_aim::TrackerState::TRACKING &&
                tracker.watchdogStats().converged;
            const bool valid_tvec = observed_armor.tvec.rows >= 3 &&
                std::abs(observed_armor.tvec.at<double>(2)) > 1e-6;
            if (!probe_ok || !valid_tvec) {
                probe_phase = 0;   // 条件不满足就重新开始（比如目标丢了）
                probe_samples = 0;
            } else {
                const double z = observed_armor.tvec.at<double>(2);
                const double x = observed_armor.tvec.at<double>(0);
                const double y = observed_armor.tvec.at<double>(1);
                const double az_now = std::atan2(x, z);
                const double el_now = -std::atan2(y, std::hypot(x, z));
                if (probe_phase == 0) {
                    probe_phase = 1;
                    probe_t0 = std::chrono::duration<double>(probe_now_clock.time_since_epoch()).count();
                    probe_samples = 0;
                    probe_az[0] = probe_el[0] = 0.0;
                    probe_off_yaw.store(0.0, std::memory_order_relaxed);
                    probe_off_pitch.store(0.0, std::memory_order_relaxed);
                    std::cout << "[self-cal] 开始响应自标定：云台会小幅动两下（约 1s），"
                                 "请让靶板保持在画面内" << std::endl;
                } else if (probe_phase == 1) {          // 基准
                    probe_az[0] += az_now;
                    probe_el[0] += el_now;
                    if (++probe_samples >= 8) {
                        probe_az[0] /= probe_samples;
                        probe_el[0] /= probe_samples;
                        probe_phase = 2;
                        probe_t0 = std::chrono::duration<double>(probe_now_clock.time_since_epoch()).count();
                        probe_samples = 0;
                        probe_off_yaw.store(probe_step_rad, std::memory_order_relaxed);
                    }
                } else if (probe_phase == 2) {          // yaw 通道 +step
                    probe_az[1] += az_now;
                    probe_el[1] += el_now;
                    if (++probe_samples >= 8) {
                        probe_az[1] /= probe_samples;
                        probe_el[1] /= probe_samples;
                        probe_d_az_yaw = (probe_az[1] - probe_az[0]) / probe_step_rad;
                        probe_d_el_yaw = (probe_el[1] - probe_el[0]) / probe_step_rad;
                        probe_phase = 3;
                        probe_t0 = std::chrono::duration<double>(probe_now_clock.time_since_epoch()).count();
                        probe_samples = 0;
                        probe_az[1] = probe_el[1] = 0.0;
                        probe_off_yaw.store(0.0, std::memory_order_relaxed);
                        probe_off_pitch.store(probe_step_rad, std::memory_order_relaxed);
                    }
                } else if (probe_phase == 3) {          // pitch 通道 +step
                    probe_az[1] += az_now;
                    probe_el[1] += el_now;
                    if (++probe_samples >= 8) {
                        probe_az[1] /= probe_samples;
                        probe_el[1] /= probe_samples;
                        probe_d_az_pitch = (probe_az[1] - probe_az[0]) / probe_step_rad;
                        probe_d_el_pitch = (probe_el[1] - probe_el[0]) / probe_step_rad;
                        probe_off_yaw.store(0.0, std::memory_order_relaxed);
                        probe_off_pitch.store(0.0, std::memory_order_relaxed);
                        // 命令通道 → 电机(INS)角：
                        //   raw_yaw   = +az_板   （同号）
                        //   raw_pitch = -el_板   （相机抬头 → 板在画面里下移）
                        const double m00 = probe_d_az_yaw;
                        const double m01 = probe_d_az_pitch;
                        const double m10 = -probe_d_el_yaw;
                        const double m11 = -probe_d_el_pitch;
                        const double det = m00 * m11 - m01 * m10;
                        std::cout << "[self-cal] 实测响应 M = [[" << m00 << "," << m01 << "],["
                                  << m10 << "," << m11 << "]]（命令→电机角），det=" << det
                                  << "（仅诊断，不用于下发：这个窗口太短，容易把上一相位残运动"
                                     "算进来；下发固定用独立探针在稳定状态下实测的 M⁻¹）"
                                  << std::endl;
                        probe_ready.store(1, std::memory_order_relaxed);
                        probe_phase = 4;
                    }
                }
            }
        }
        const int observed_plate =
            has_observation ? tracker.getLastObservedPlateId() : -1;
        // 尺寸不再在这里拼：每块板的半宽由底盘几何按"大板/小板"给（tracker.armorHalfWidth）。
        const bool large_armor =
            has_observation && observed_armor.armor_type == auto_aim::large;
        // 模型板的姿态/位置**只走 Tracker::modelAt()**（common/chassis_model.hpp 的唯一构造
        // 入口）：底盘慢层几何（逐板距离/同高/15° 倾角/板面尺寸）+ 快层状态（车心/yaw/ω），
        // 外推到被显示那一帧、并用**那一帧的相机姿态**变换。以前三处调用点各写一套，
        // 还混用世界系/相机系向量，真机上就表现成"四块板高度不统一、底盘不对"。

        // 模型对**已观测那块板**的重投影误差（像素）——这是判断"估计是否自洽"最直接的量：
        // 如果模型连刚刚观测到的那块板都投影不回识别框上，那问题在估计/坐标系，而不是板尺寸。
        double reprojection_px = -1.0;
        // 只在"正在跟踪且这一帧有观测"时画模型：没有观测时画的只是滑行外推，
        // 很容易被误读成"模型飘了"（实际是没数据）。
        const bool model_visible = has_observation &&
                                   tracker.getState() == auto_aim::TrackerState::TRACKING;
        if (model_visible) {
            // —— 重投影的时间基 ——
            // 估计器状态停在**最后一次观测时刻**（曝光），而画面是"现在"。异步检测
            // "曝光→到手"几十~一百多 ms，在 3 rad/s 自旋上就是 12~28° 相位差，
            // 于是整车模型的四块板整体拖在真车后面。这里按 (底速, ω) 把**输出口径**
            // 的状态外推到被显示那一帧，并用**那一帧的相机姿态**做世界→相机变换
            // （真机云台在延迟期间会动/漂，用观测时刻的姿态画整片模型都是歪的）。
            // 控制/选板那条路已由选择器按 estimate.age 补过，不要重复补。
            const double model_dt = std::max(0.0, std::min(observation_age, 0.20));
            const auto model = tracker.modelAt(model_dt, render_pose);
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
                // 已观测那块额外算一次"模型 vs 检测"的像素距离（reproj），用来判断估计
                // 是否自洽；两者差得多时会看到同一块板上有两个框，那就是不自洽的直接证据。
                const bool observed = plate == observed_plate;
                const cv::Scalar color = observed ? cv::Scalar(255, 180, 0)   // 观测板：亮青蓝
                                                  : cv::Scalar(200, 140, 0);  // 外推板：稍暗
                auto_aim::drawQuad(image, corners, color, observed ? 2 : 1);
                cv::putText(image, "E" + std::to_string(plate),
                            corners[0] + cv::Point2f(2.0f, -3.0f), cv::FONT_HERSHEY_SIMPLEX, 0.45,
                            color, 1, cv::LINE_AA);
                if (observed) {
                    // 角点顺序：模型是 (左下,左上,右上,右下)，检测是 (左上,右上,右下,左下)
                    // —— 必须循环移位配对，否则等于把一个"板高"当成误差（实测虚报 ~90 px）。
                    const std::array<cv::Point2f, 4> detected{
                        observed_armor.left.top, observed_armor.right.top,
                        observed_armor.right.bottom, observed_armor.left.bottom};
                    // 两个四边形的起点/绕向可能不同，取 4 个循环移位里最小的那个（这才是
                    // "形状是否吻合"的度量；固定配对会把方向差异误算成误差）
                    double best = 1e18;
                    for (int shift = 0; shift < 4; ++shift) {
                        double sum = 0.0;
                        for (int k = 0; k < 4; ++k) {
                            sum += cv::norm(corners[k] - detected[(k + shift) % 4]);
                        }
                        best = std::min(best, sum / 4.0);
                    }
                    reprojection_px = best;
                }
            }

            // ---- 模型解释度：模型 vs **本帧所有检测**（不只被硬钉那块）----------
            // 被钉住的那块板永远重合（pin 之后又用同一姿态反变换回来），所以"底盘模型
            // 和姿态约定对不对"只能看另外几块：把四个模型板心投到图上，再跟每个检测框
            // 中心比最近距离。正确时典型 ≲10 px；姿态约定（pitch 符号 / 世界系约定）
            // 错了会几十~上百 px，而 reproj 仍然很小 —— 这正是"被观测板贴得住、
            // 另外三块板飞了"的直接度量。
            if (model.valid && !armors.empty()) {
                std::array<cv::Point2f, 4> model_px{};
                std::array<bool, 4> model_px_ok{};
                for (int plate = 0; plate < 4; ++plate) {
                    const auto& pose = model.poses[plate];
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
                    for (int plate = 0; plate < 4; ++plate) {
                        if (!model_px_ok[plate]) continue;
                        best = std::min(best, static_cast<double>(cv::norm(model_px[plate] -
                                                                          detected_center)));
                    }
                    if (best < 1e17) {
                        model_fit_sum += best;
                        model_fit_max = std::max(model_fit_max, best);
                        ++model_fit_n;
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

        // ---- 曲线采样 + 合成显示 --------------------------------------------
        // 曲线画在**同一幅图的下面**（竖直组合，和仿真一致的排法）：同宽展开，
        // 命令/回传两条线的高低差一眼可见，不用再来回切窗口。
        if (curves_enabled) {
            const double curve_t = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - curve_start).count();
            curves.pushAll({gimbal_state.yaw * 180.0 / CV_PI,
                            sent_yaw_now.load(std::memory_order_relaxed) * 180.0 / CV_PI,
                            gimbal_state.pitch * 180.0 / CV_PI,
                            sent_pitch_now.load(std::memory_order_relaxed) * 180.0 / CV_PI,
                            curve_az_deg, curve_el_deg,
                            curve_off_yaw_deg, curve_off_pitch_deg,
                            gimbal_state.yaw_vel * 180.0 / CV_PI,
                            gimbal_state.pitch_vel * 180.0 / CV_PI},
                           curve_t);
            // 诊断列（见 addLogColumn 的注释）
            const double box_h_px = has_observation
                ? cv::norm(observed_armor.left.top - observed_armor.left.bottom)
                : 0.0;
            const double dist_m = has_observation ? cv::norm(observed_armor.tvec) : 0.0;
            curves.pushLog({allow_control_now ? 1.0 : 0.0,
                            static_cast<double>(static_cast<int>(tracker.getState())),
                            observation_age * 1000.0, dist_m, box_h_px});
        }
        if (show_display) {
            cv::Mat display = image;
            // 1440x1080 原图 + 4 条曲线会高过笔记本屏幕，显示前先等比缩小（只影响显示，
            // 不影响任何标定/PnP —— 那用的是原始帧和内参）。
            constexpr int kDisplayMaxRows = 600;
            if (display.rows > kDisplayMaxRows) {
                const double scale = static_cast<double>(kDisplayMaxRows) / display.rows;
                cv::Mat resized;
                cv::resize(display, resized, cv::Size(), scale, scale, cv::INTER_AREA);
                display = resized;
            }
            if (curves_enabled) {
                const cv::Mat merged = curves.renderStacked(display, 1000);
                if (!merged.empty()) display = merged;
            }
            cv::imshow("Detection & Tracking", display);
        }
        key = show_display ? cv::waitKey(1) : 1;

        // ------------------- 7.5 数据集录制 -------------------
        // 每帧都调用，真正的抽帧交给 DatasetWriter（every_n），别放在 1Hz 的状态块里
        if (recording) {
            auto_aim::DatasetRow row;
            row.frame = frame_sequence;
            row.t = std::chrono::duration<double>(std::chrono::steady_clock::now() - record_start)
                        .count();
            row.yaw = gimbal_state.yaw;   // **原始**回传角（没做 yaw_sign 修正）
            row.pitch = gimbal_state.pitch;
            row.yaw_vel = gimbal_state.yaw_vel;
            row.pitch_vel = gimbal_state.pitch_vel;
            row.bullet_speed = gimbal_state.bullet_speed;
            row.mode = static_cast<int>(gimbal.mode());
            row.armors = static_cast<int>(armors.size());
            row.pnp = pnp_count;
            row.tracker_state = static_cast<int>(tracker.getState());
            row.plate_id = outcome.decision.armor_id;
            row.aim_yaw = outcome.decision.target_yaw;
            row.aim_pitch = outcome.decision.target_pitch;
            row.aim_yaw_error = outcome.aim_yaw_error;
            row.aim_pitch_error = outcome.aim_pitch_error;
            row.fire = outcome.fire ? 1 : 0;
            dataset.write(row, image);
        }

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
                      // 图像残差 = 真正决定"打不打得到"的那个量（也是开火闸门的判据）。
                      // 状态行里过去只有上面的 aim_err，而它是控制器自己跟自己比（恒 ≈0），
                      // 板偏轴 9° 时它照样显示 0.0 —— 别再看它判断瞄没瞄准。
                      << " img_err="
                      << image_err_yaw.load(std::memory_order_relaxed) * 180.0 / CV_PI << "/"
                      << image_err_pitch.load(std::memory_order_relaxed) * 180.0 / CV_PI
                      << "deg(闸门拦下 " << fire_gate_blocked.load(std::memory_order_relaxed)
                      << ")"
                      << ", fire=" << (outcome.fire ? 1 : 0) << ", reproj=" << reprojection_px
                      // 模型口径：R 是模型用的回转半径（online=被拟合器慢层改过），
                      // model_age 是重投影外推到"现在"用的时间（=观测延迟）。
                      // 模型解释度（对**所有**检测，不只被钉那块）：med/max px。
                      // 调 ULTRA_VISION_PITCH_SIGN 时就看这个数（取小者），
                      // 照片上表现为"背面那块板被画高还是画低"。
                      << ", model_fit="
                      << (model_fit_n > 0 ? model_fit_sum / model_fit_n : -1.0) << "/"
                      << (model_fit_n > 0 ? model_fit_max : -1.0) << "px(n=" << model_fit_n << ")"
                      // 几何慢收敛的结果（现场就看这四个数会不会收敛成"前后一个值、
                      // 左右另一个值"）；np_flip/motion 是另外两个机制的诊断。
                      << ", Rgeo="
                      << cv::format("%.3f/%.3f/%.3f/%.3f", tracker.chassisGeometry().distance(0),
                                    tracker.chassisGeometry().distance(1),
                                    tracker.chassisGeometry().distance(2),
                                    tracker.chassisGeometry().distance(3))
                      // 局部拟合器统计：attempts/valids + 最近一次 rmse/R/灯条数。
                      // 实车若一直 R=(cfg)、motion=0，就看这里的 valids 与 rmse。
                      << ", fit=" << tracker.localFitValids() << "/" << tracker.localFitAttempts()
                      << " rmse=" << tracker.localFitLastRmse() << "px Rfit="
                      << tracker.localFitLastR() << " nbars=" << tracker.localFitLastBars()
                      // push=推给拟合器的灯条累计、arcs=当前弧数、apply=结果真正写进几何的次数。
                      // 若 push 在涨而 arcs 一直是 0/1 → 门控把它挡了；apply=0 → 时间基/老化。
                      << " push=" << tracker.localFitPushed()
                      << " arcs=" << tracker.localFitArcs()
                      << " pub=" << tracker.localFitApplied()
                      // 关键三个：calls=进入次数 nofit=没拿到有效结果 stale=太老被丢
                      // geometry=真正写进几何的次数；ufr=use_fit_radius 开关；age=本次结果年龄
                      << " calls=" << tracker.localFitApplyCalls()
                      << " nofit=" << tracker.localFitApplyNoFit()
                      << " stale=" << tracker.localFitApplyStale()
                      << " geometry=" << tracker.localFitApplyGeometry()
                      << " age=" << tracker.localFitApplyAgeMs() << "ms"
                      << " ufr=" << (tracker.fitRadiusEnabled() ? 1 : 0)
                      << " lf=" << (tracker.localFitEnabled() ? 1 : 0)
                      << ", np_flip=" << tracker.nearestPlateFlips()
                      << ", motion=" << static_cast<int>(tracker.lastMotion())
                      << "(0静/1转/2移/3转+移)"
                      << ", R=" << tracker.getArmorRadius()
                      << (tracker.armorRadiusOnline() ? "(online)" : "(cfg)")
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
        }
    }

    if (recording) {
        dataset.close();
        std::cout << "[dataset] 录制结束：" << dataset.written() << " 帧已写入 "
                  << dataset.directory() << "（共处理 " << dataset.seen() << " 帧）" << std::endl;
        std::cout << "[dataset] 离线回放：./build/replay_dataset " << dataset.directory()
                  << " --config-dir configs" << std::endl;
    }
    return 0;
}
