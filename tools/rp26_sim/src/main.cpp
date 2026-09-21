// 深大 RP-26Rune 能量机关算法的仿真接入 harness。
//
// 数据流（与用户要求一致）：直接接 simulator_system 仿真器
//   daedalus --(7666 TCP 帧)--> 本程序 --(RP-26 检测+算法)--> 目标 yaw/pitch
//            <--(7667 GIMBAL/FIRE/RESET)--  本程序
//            --(7668 遥测 RUNE 事件)--> 打分
//
// 关键点：
//   * 检测用的是 RP-26Rune 自己的 NNDetector（同一份预处理/letterbox/解码/NMS）；
//   * 坐标变换用他们的 TFTree：我们把"当前云台姿态"按他们的约定填进去，
//     约定推导见 reportPoseRotation()；
//   * 逐帧 CSV 记录他们的下发角、像素残差与自洽性检查，便于和 Ultra_Vision 的
//     tools/rune_curve.py 输出对齐比较。

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <glog/logging.h>
#include <opencv2/opencv.hpp>

#include "function.hpp"
#include "PnPVariable.hpp"
#include "power_rune_interface.hpp"
#include "transform_tools/transform_tools.h"

#include "rp26_detector.hpp"
#include "rp26_dump_context.hpp"
#include "sim_link.hpp"

namespace
{
using Matrix3d = Eigen::Matrix3d;

constexpr double kPi = 3.14159265358979323846;

double deg(double angle_rad) { return angle_rad * 180.0 / kPi; }
double rad(double angle_deg) { return angle_deg * kPi / 180.0; }

double normalizeAngle(double angle)
{
    while (angle > kPi)
        angle -= 2.0 * kPi;
    while (angle < -kPi)
        angle += 2.0 * kPi;
    return angle;
}

std::string telemetryField(const std::string &line, const std::string &key)
{
    const std::string needle = key + "=";
    const auto position = line.find(needle);
    if (position == std::string::npos)
        return {};
    const auto start = position + needle.size();
    const auto end = line.find(' ', start);
    return line.substr(start, end == std::string::npos ? std::string::npos : end - start);
}

double simSeconds(const std::string &line)
{
    const auto position = line.find(" t=");
    if (position == std::string::npos)
        return -1.0;
    return std::atof(line.c_str() + position + 3);
}

struct Options
{
    std::string mode = "small";
    std::string host = "127.0.0.1";
    std::string csv_path;
    double seconds = 20.0;
    bool open_loop = false;
    double yaw_sign = 1.0;
    double pitch_sign = 1.0;
    double yaw_offset_deg = 0.0;
    double pitch_offset_deg = 0.0;
    bool reset_on_start = true;
    double fire_interval_s = 0.15;
    double fov_deg = 45.0; // 仿真器相机垂直 FOV（tools/sim/rune_camera_*.yaml）
    // 他们的火控是"视觉线程 + 高频火控线程"：get_rune_data() 按 100~200 Hz 调用，
    // 每次都用当前时间重解弹道并推进开火状态机。默认按 100 Hz 跑，0 = 逐帧调用。
    double control_hz = 100.0;
};

Options parseArgs(int argc, char **argv)
{
    Options options;
    for (int index = 1; index < argc; ++index)
    {
        const std::string arg = argv[index];
        const auto next = [&](const char *name) -> std::string {
            if (index + 1 >= argc)
            {
                std::cerr << "missing value for " << name << std::endl;
                std::exit(2);
            }
            return argv[++index];
        };
        if (arg == "--mode")
            options.mode = next("--mode");
        else if (arg == "--host")
            options.host = next("--host");
        else if (arg == "--seconds")
            options.seconds = std::atof(next("--seconds").c_str());
        else if (arg == "--csv")
            options.csv_path = next("--csv");
        else if (arg == "--open-loop")
            options.open_loop = true;
        else if (arg == "--yaw-sign")
            options.yaw_sign = std::atof(next("--yaw-sign").c_str());
        else if (arg == "--pitch-sign")
            options.pitch_sign = std::atof(next("--pitch-sign").c_str());
        else if (arg == "--yaw-offset-deg")
            options.yaw_offset_deg = std::atof(next("--yaw-offset-deg").c_str());
        else if (arg == "--pitch-offset-deg")
            options.pitch_offset_deg = std::atof(next("--pitch-offset-deg").c_str());
        else if (arg == "--no-reset")
            options.reset_on_start = false;
        else if (arg == "--fire-interval")
            options.fire_interval_s = std::atof(next("--fire-interval").c_str());
        else if (arg == "--fov-deg")
            options.fov_deg = std::atof(next("--fov-deg").c_str());
        else if (arg == "--control-hz")
            options.control_hz = std::atof(next("--control-hz").c_str());
        else if (arg == "--help" || arg == "-h")
        {
            std::cout << "usage: rp26_sim [--mode small|large] [--seconds N] [--csv path]\n"
                         "                [--open-loop] [--yaw-sign s] [--pitch-sign s]\n"
                         "                [--yaw-offset-deg d] [--pitch-offset-deg d]\n"
                         "                [--fire-interval s] [--no-reset]\n";
            std::exit(0);
        }
        else
        {
            std::cerr << "unknown option: " << arg << std::endl;
            std::exit(2);
        }
    }
    return options;
}

/// @brief 构造"当前云台姿态"对应的旋转，写进他们的 TFTree。
///
/// 约定推导（全部相对仿真器坐标系）：
///   * 仿真器相机位姿 = home_pose * Ry(-yaw_sim) * Rx(pitch_sim)，位置固定，
///     即云台只有转动、没有平移（daedalus 的 remote_camera_gimbal）。
///   * 把 home 朝向当作参考系（"车系"），OpenCV 相机约定 x 右 / y 下 / z 前，
///     则相机->车系的旋转 R_car_from_camera = Ry(yaw_sim) * Rx(pitch_sim)。
///   * V = vehicle_from_ecs、G = gimbal_from_camera（RuneTrackerManager 里硬编码），
///     二者满足 G = V^T，于是 V*G = I。
///   * 实测（见 main() 里的 TFTree 自检）：把 gimbal 节点设成 M 时，
///     他们 calculate_relative_TF(car_frame, camera_frame) 真实返回的是 M*G，
///     而不是字面上的 V*M*G。所以取 M = Ry(yaw)*Rx(pitch)*V 时，
///     他们拿到的 R_car_from_camera 恰好等于 Ry(yaw)*Rx(pitch)：
///         M*G = Ry(yaw)Rx(pitch)*V*G = Ry(yaw)Rx(pitch)
///     这正是"相机 home 朝向为车系参考"的定义，于是他们的输出角
///     （atan2(x,z) / atan2(-y,·)）直接就是仿真器的 yaw/pitch 指令。
Matrix3d reportPoseRotation(double yaw, double pitch)
{
    static const Matrix3d vehicle_from_ecs = [] {
        Matrix3d matrix;
        matrix << 0.0, 1.0, 0.0,
            0.0, 0.0, -1.0,
            -1.0, 0.0, 0.0;
        return matrix;
    }();

    const Matrix3d car_from_camera =
        Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitY()).toRotationMatrix() *
        Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitX()).toRotationMatrix();

    return car_from_camera * vehicle_from_ecs;
}

/// @brief 由像素点反解"要把它对准画面中心所需的云台角"（车系/仿真器约定）。
void requiredAnglesFromPixel(const cv::Matx33d &camera_matrix, const cv::Point2f &pixel,
                             double pose_yaw, double pose_pitch, double &yaw_required,
                             double &pitch_required)
{
    const double fx = camera_matrix(0, 0);
    const double fy = camera_matrix(1, 1);
    const double cx = camera_matrix(0, 2);
    const double cy = camera_matrix(1, 2);

    const Eigen::Vector3d camera_direction((pixel.x - cx) / fx, (pixel.y - cy) / fy, 1.0);
    const Matrix3d car_from_camera =
        Eigen::AngleAxisd(pose_yaw, Eigen::Vector3d::UnitY()).toRotationMatrix() *
        Eigen::AngleAxisd(pose_pitch, Eigen::Vector3d::UnitX()).toRotationMatrix();
    const Eigen::Vector3d car_direction = car_from_camera * camera_direction;

    yaw_required = std::atan2(car_direction.x(), car_direction.z());
    pitch_required =
        std::atan2(-car_direction.y(), std::hypot(car_direction.x(), car_direction.z()));
}

struct BladeGeometry
{
    bool valid = false;
    cv::Point2f blade_center{0.0F, 0.0F};
    cv::Point2f r_center{0.0F, 0.0F};
    int model_class = -1;
    float confidence = 0.0F;
};

/// @brief 选一片"参考扇叶"用于残差统计：优先未击打（class 0），其次任意。
BladeGeometry pickReferenceBlade(const std::vector<rp26_sim::Rp26Detection> &detections)
{
    BladeGeometry geometry;
    double best_score = -1.0;
    for (const auto &detection : detections)
    {
        double score = detection.confidence;
        if (detection.model_class_id == 0)
            score += 1.0;
        if (score <= best_score)
            continue;
        best_score = score;

        geometry.valid = true;
        geometry.model_class = detection.model_class_id;
        geometry.confidence = detection.confidence;
        geometry.blade_center = 0.25F * (detection.keypoints[0] + detection.keypoints[1] +
                                        detection.keypoints[3] + detection.keypoints[4]);
        geometry.r_center = detection.keypoints[2];
    }
    return geometry;
}
} // namespace

int main(int argc, char **argv)
{
    google::InitGoogleLogging(argv[0]);
    FLAGS_logtostderr = true;
    FLAGS_minloglevel = google::GLOG_INFO;

    const Options options = parseArgs(argc, argv);
    const bool is_large = options.mode == "large";
    const rp26_sim::Rp26RuneMode rune_mode =
        is_large ? rp26_sim::Rp26RuneMode::Large : rp26_sim::Rp26RuneMode::Small;

    std::cout << "[rp26_sim] mode=" << options.mode << " seconds=" << options.seconds
              << " open_loop=" << (options.open_loop ? 1 : 0)
              << " yaw_sign=" << options.yaw_sign << " pitch_sign=" << options.pitch_sign
              << " yaw_offset_deg=" << options.yaw_offset_deg
              << " pitch_offset_deg=" << options.pitch_offset_deg << std::endl;

    rp26_sim::Rp26RuneDetector detector;
    rp26_sim::SimLink link(options.host, 7666, 7667, 7668);

    // 相机内参：与仿真器一致。仿真器用 Bevy 透视投影（垂直 FOV 由 yaml 给，
    // 主点在画面中心），所以按实际帧尺寸现算；同时覆盖他们全局的 CAM/DIS，
    // 避免"配置文件里的实车内参 / 640x480 内参"和实际帧不匹配。
    cv::Matx33d camera_matrix(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
    bool camera_matrix_ready = false;
    std::cout << "[rp26_sim] config CAM(loaded) = [";
    for (int row = 0; row < 3; ++row)
        for (int col = 0; col < 3; ++col)
            std::cout << CAM(row, col) << (row == 2 && col == 2 ? "" : ", ");
    std::cout << "] (将被按实际帧尺寸覆盖)" << std::endl;

    // 他们的 TFTree：car_frame 为根；仿真器云台只有转动，所以三个坐标系同点。
    transform_tools::TFTree tf_tree;
    tf_tree.add_TF(transform_tools::TF(), car_frame, ecs_world_frame);
    tf_tree[ecs_world_frame].set_rotation(Eigen::Matrix3d::Identity());
    tf_tree.add_TF(transform_tools::TF(), ecs_world_frame, gimbal_frame);
    tf_tree.add_TF(transform_tools::TF(), gimbal_frame, camera_frame);
    {
        Eigen::Matrix3d gimbal_from_camera;
        gimbal_from_camera << 0.0, 0.0, -1.0,
            1.0, 0.0, 0.0,
            0.0, -1.0, 0.0;
        tf_tree[camera_frame].set_rotation(gimbal_from_camera);
    }
    tf_tree.add_TF(transform_tools::TF(), car_frame, unbiased_camera_frame);

    // 自检：多组姿态验证"我们填进去的云台姿态" == "他们读到的 R_car_from_camera"。
    // 只有这里误差接近 0，后面讨论他们的瞄准效果才有意义。
    {
        const std::array<std::pair<double, double>, 5> probes{{
            {0.0, 0.0},
            {0.20, -0.10},
            {-0.35, 0.22},
            {1.20, 0.40},
            {-2.40, -0.55},
        }};
        double worst_error = 0.0;
        for (const auto &probe : probes)
        {
            const double probe_yaw = probe.first;
            const double probe_pitch = probe.second;
            tf_tree[gimbal_frame].set_rotation(reportPoseRotation(probe_yaw, probe_pitch));
            const Matrix3d actual =
                tf_tree.calculate_relative_TF(car_frame, camera_frame).get_rotation();
            const Matrix3d expected =
                Eigen::AngleAxisd(probe_yaw, Eigen::Vector3d::UnitY()).toRotationMatrix() *
                Eigen::AngleAxisd(probe_pitch, Eigen::Vector3d::UnitX()).toRotationMatrix();
            worst_error = std::max(worst_error, (actual - expected).norm());
        }
        std::cout << "[rp26_sim] TFTree 姿态映射自检: max ||actual-expected|| = " << worst_error
                  << (worst_error < 1e-9 ? "  (PASS)" : "  (FAIL)") << std::endl;
    }

    std::ofstream csv;
    if (!options.csv_path.empty())
    {
        csv.open(options.csv_path, std::ios::out | std::ios::trunc);
        if (csv)
        {
            csv << "frame,seq,local_t,sim_t,pose_yaw_deg,pose_pitch_deg,algo_yaw_deg,"
                   "algo_pitch_deg,cmd_yaw_deg,cmd_pitch_deg,is_find,is_fire,det_n,det_classes,"
                   "ref_model_class,ref_conf,blade_px,blade_py,r_px,r_py,need_yaw_deg,"
                   "need_pitch_deg,ang_err_yaw_deg,ang_err_pitch_deg,blade_r_px,aim_px,aim_py,"
                   "fire_sent,activated_bits\n";
        }
    }

    std::ofstream ground_truth;
    if (!options.csv_path.empty())
        ground_truth.open(options.csv_path + ".ground_truth", std::ios::out | std::ios::trunc);

    // 把云台压在初始位（home），并清一次机关状态。
    if (options.reset_on_start)
        link.sendReset();
    double pose_yaw = 0.0;
    double pose_pitch = 0.0;
    link.sendGimbal(pose_yaw, pose_pitch);

    const auto run_start = std::chrono::steady_clock::now();
    const auto wallSeconds = [&run_start] {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - run_start).count();
    };

    // 共享状态：视觉线程写入"这一帧曝光时的云台姿态"给 TF/CSV，火控线程写入最新下发角。
    std::mutex pose_mutex;
    std::atomic<bool> control_stop{false};
    std::atomic<double> last_cmd_yaw{0.0};
    std::atomic<double> last_cmd_pitch{0.0};
    std::atomic<int> control_find_frames{0};
    std::atomic<int> control_fire_frames{0};
    std::atomic<int> fire_sent_atomic{0};
    std::atomic<bool> last_cmd_valid{false};
    int control_fire_frames_seen = 0;

    // 他们的火控线程：按 control_hz 调用 get_rune_data，推进开火状态机并下发指令。
    std::thread control_thread;
    if (options.control_hz > 0.0 && !options.open_loop)
    {
        control_thread = std::thread([&] {
            const auto period = std::chrono::duration<double>(1.0 / options.control_hz);
            auto next = std::chrono::steady_clock::now();
            double fire_limiter = -1e9;
            while (!control_stop.load())
            {
                const power_rune::RuneSendData send_data = power_rune::get_rune_data(is_large);
                const double cmd_yaw =
                    options.yaw_sign * send_data.yaw + rad(options.yaw_offset_deg);
                const double cmd_pitch =
                    options.pitch_sign * send_data.pitch + rad(options.pitch_offset_deg);
                if (send_data.is_find_buff)
                {
                    link.sendGimbal(cmd_yaw, cmd_pitch);
                    last_cmd_yaw.store(cmd_yaw);
                    last_cmd_pitch.store(cmd_pitch);
                    last_cmd_valid.store(true);
                    ++control_find_frames;
                }
                else
                {
                    last_cmd_valid.store(false);
                }
                if (send_data.is_enable_fire)
                {
                    ++control_fire_frames;
                    const double now = wallSeconds();
                    if (now - fire_limiter >= options.fire_interval_s)
                    {
                        link.sendFire();
                        ++fire_sent_atomic;
                        fire_limiter = now;
                    }
                }
                next += std::chrono::duration_cast<std::chrono::steady_clock::duration>(period);
                std::this_thread::sleep_until(next);
            }
        });
    }

    int hits = 0;
    int scored = 0;
    int activations = 0;
    int fire_sent = 0;
    int activated_bits = 0;
    std::optional<double> first_activation_s;
    double fire_limiter = -1e9;
    std::uint64_t frame_index = 0;
    int det_stats[3] = {0, 0, 0};
    int frames_with_detection = 0;
    int frames_total = 0;

    while (wallSeconds() < options.seconds)
    {
        // 遥测：命中/激活/机关状态（打分用）
        for (const std::string &line : link.pollTelemetry())
        {
            if (ground_truth)
                ground_truth << line << '\n';

            const std::string mode = telemetryField(line, "mode");
            const std::string expect_mode = is_large ? "large" : "small";
            if (mode.empty() || mode != expect_mode)
                continue;

            const std::string event = telemetryField(line, "event");
            if (event == "hit")
            {
                ++hits;
                if (telemetryField(line, "scored") == "1")
                    ++scored;
            }
            if (event == "activated")
            {
                ++activations;
                if (!first_activation_s.has_value())
                    first_activation_s = simSeconds(line);
            }
            const std::string activated = telemetryField(line, "activated");
            if (!activated.empty())
            {
                int bits = 0;
                for (std::size_t index = 0; index < activated.size(); ++index)
                    if (activated[index] == '1')
                        bits |= (1 << static_cast<int>(index));
                activated_bits = bits;
            }
            std::cout << "  gt " << line << std::endl;
        }

        rp26_sim::SimFrame frame;
        if (!link.readFrame(frame) || frame.bgr.empty())
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
            continue;
        }
        ++frames_total;
        rp26_sim::g_current_frame_index.store(static_cast<long long>(frame_index));

        if (!camera_matrix_ready)
        {
            // 垂直 FOV -> f = (height/2) / tan(fov/2)；主点在画面中心。
            const double focal =
                (static_cast<double>(frame.bgr.rows) * 0.5) / std::tan(rad(options.fov_deg) * 0.5);
            const double cx = static_cast<double>(frame.bgr.cols) * 0.5;
            const double cy = static_cast<double>(frame.bgr.rows) * 0.5;
            camera_matrix = cv::Matx33d(focal, 0.0, cx, 0.0, focal, cy, 0.0, 0.0, 1.0);

            CAM(0, 0) = focal;
            CAM(0, 1) = 0.0;
            CAM(0, 2) = cx;
            CAM(1, 0) = 0.0;
            CAM(1, 1) = focal;
            CAM(1, 2) = cy;
            CAM(2, 0) = 0.0;
            CAM(2, 1) = 0.0;
            CAM(2, 2) = 1.0;
            for (int col = 0; col < 5; ++col)
                DIS(0, col) = 0.0;

            camera_matrix_ready = true;
            std::cout << "[rp26_sim] frame " << frame.bgr.cols << 'x' << frame.bgr.rows
                      << ", f=" << focal << ", cx=" << cx << ", cy=" << cy
                      << "（已写入他们的 CAM/DIS）" << std::endl;
        }

        // 原始帧落盘（离线几何对照用）：RP26_DUMP_DIR + RP26_DUMP_STRIDE
        if (const char *dump_dir = std::getenv("RP26_DUMP_DIR"))
        {
            if (dump_dir[0] != '\0')
            {
                static const int raw_stride = [] {
                    const char *value = std::getenv("RP26_DUMP_STRIDE");
                    const int parsed = value != nullptr ? std::atoi(value) : 0;
                    return parsed > 0 ? parsed : 10;
                }();
                if (static_cast<long long>(frame_index) % raw_stride == 0)
                {
                    char name[256];
                    std::snprintf(name, sizeof(name), "%s/raw_f%05llu_%u_%llu.png", dump_dir,
                                  static_cast<unsigned long long>(frame_index), frame.seq,
                                  static_cast<unsigned long long>(frame.source_timestamp_us));
                    cv::imwrite(name, frame.bgr);
                }
            }
        }

        const auto inference_start = std::chrono::steady_clock::now();
        std::vector<rp26_sim::Rp26Detection> detections;
        try
        {
            detections = detector.infer(frame.bgr);
        }
        catch (const std::exception &error)
        {
            std::cerr << "[rp26_sim] inference failed: " << error.what() << std::endl;
            continue;
        }
        const double inference_ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() -
                                                     inference_start)
                .count();

        for (const auto &detection : detections)
            if (detection.model_class_id >= 0 && detection.model_class_id < 3)
                ++det_stats[detection.model_class_id];

        const auto rune_infos = rp26_sim::Rp26RuneDetector::toRuneInfos(detections, rune_mode);
        if (!rune_infos.empty())
            ++frames_with_detection;

        if (options.control_hz > 0.0 && !options.open_loop)
        {
            // 火控线程在跑：这一帧曝光时的云台姿态 = 它最近一次下发的角度。
            pose_yaw = last_cmd_yaw.load();
            pose_pitch = last_cmd_pitch.load();
        }

        // 用"这一帧曝光时刻的云台姿态"（= 最近一次下发的姿态）填 TF。
        const double frame_timestamp_s = static_cast<double>(frame.source_timestamp_us) * 1e-6;
        // 他们的 RuneInput::timestamp 是 timetool::Timestamp（system_clock 时间点）；
        // 仿真器的曝光时间戳就是 unix epoch 微秒，直接构造即可。
        const timetool::Timestamp capture_time(
            std::chrono::duration_cast<std::chrono::system_clock::duration>(
                std::chrono::microseconds(frame.source_timestamp_us)));

        tf_tree[gimbal_frame].set_rotation(reportPoseRotation(pose_yaw, pose_pitch));

        power_rune::RuneInput rune_input;
        rune_input.is_big_rune = is_large;
        rune_input.ori_mat = frame.bgr;
        rune_input.tf_tree = tf_tree;
        rune_input.timestamp = capture_time;
        rune_input.cd_my_color = 0; // 0 = 红（仿真里我们的机器人是红方）
        rune_input.nn_rune_infos = rune_infos;

        if (!rune_infos.empty())
            power_rune::process_power_rune(rune_input);

        power_rune::RuneSendData send_data{};
        double algo_yaw = 0.0;
        double algo_pitch = 0.0;
        double cmd_yaw = 0.0;
        double cmd_pitch = 0.0;
        if (options.control_hz > 0.0 && !options.open_loop)
        {
            // 火控线程负责 get_rune_data / 下发 / 开火；主线程只记录它的最新输出。
            send_data.is_find_buff = last_cmd_valid.load();
            send_data.is_enable_fire = control_fire_frames.load() > control_fire_frames_seen;
            control_fire_frames_seen = control_fire_frames.load();
            cmd_yaw = last_cmd_yaw.load();
            cmd_pitch = last_cmd_pitch.load();
            algo_yaw = (cmd_yaw - rad(options.yaw_offset_deg)) * options.yaw_sign;
            algo_pitch = (cmd_pitch - rad(options.pitch_offset_deg)) * options.pitch_sign;
            fire_sent = fire_sent_atomic.load();
        }
        else
        {
            send_data = power_rune::get_rune_data(is_large);
            algo_yaw = send_data.yaw;
            algo_pitch = send_data.pitch;
            cmd_yaw = options.yaw_sign * algo_yaw + rad(options.yaw_offset_deg);
            cmd_pitch = options.pitch_sign * algo_pitch + rad(options.pitch_offset_deg);

            if (send_data.is_enable_fire && !options.open_loop)
            {
                const double now = wallSeconds();
                if (now - fire_limiter >= options.fire_interval_s)
                {
                    link.sendFire();
                    ++fire_sent;
                    fire_limiter = now;
                }
            }

            if (!options.open_loop && send_data.is_find_buff)
            {
                pose_yaw = cmd_yaw;
                pose_pitch = cmd_pitch;
                link.sendGimbal(pose_yaw, pose_pitch);
            }
        }

        const BladeGeometry blade = pickReferenceBlade(detections);
        double need_yaw = 0.0;
        double need_pitch = 0.0;
        double blade_r_px = 0.0;
        if (blade.valid)
        {
            requiredAnglesFromPixel(camera_matrix, blade.blade_center, pose_yaw, pose_pitch,
                                    need_yaw, need_pitch);
            blade_r_px = std::hypot(blade.blade_center.x - blade.r_center.x,
                                    blade.blade_center.y - blade.r_center.y);
        }

        // 算法下发角对应的"瞄准点"像素（相机转到该角后画面中心落在哪）。
        const double aim_px =
            camera_matrix(0, 2) + camera_matrix(0, 0) * std::tan(cmd_yaw - pose_yaw);
        const double aim_py =
            camera_matrix(1, 2) - camera_matrix(1, 1) * std::tan(cmd_pitch - pose_pitch);

        if (csv)
        {
            csv << frame_index << ',' << frame.seq << ',' << std::fixed << std::setprecision(6)
                << wallSeconds() << ',' << frame_timestamp_s << ',' << std::setprecision(3)
                << deg(pose_yaw) << ',' << deg(pose_pitch) << ',' << deg(algo_yaw) << ','
                << deg(algo_pitch) << ',' << deg(cmd_yaw) << ',' << deg(cmd_pitch) << ','
                << static_cast<int>(send_data.is_find_buff) << ','
                << static_cast<int>(send_data.is_enable_fire) << ',' << detections.size() << ',';
            for (std::size_t index = 0; index < detections.size(); ++index)
                csv << (index == 0 ? "" : "|") << detections[index].model_class_id;
            csv << ',' << blade.model_class << ',' << blade.confidence << ','
                << blade.blade_center.x << ',' << blade.blade_center.y << ','
                << blade.r_center.x << ',' << blade.r_center.y << ',' << deg(need_yaw) << ','
                << deg(need_pitch) << ',' << deg(normalizeAngle(algo_yaw - need_yaw)) << ','
                << deg(normalizeAngle(algo_pitch - need_pitch)) << ',' << blade_r_px << ','
                << aim_px << ',' << aim_py << ',' << fire_sent << ',' << activated_bits << '\n';
        }

        if (frame_index % 30 == 0)
        {
            std::cout << "[rp26_sim] f" << frame_index << " seq=" << frame.seq
                      << " det=" << detections.size() << " infer=" << std::setprecision(1)
                      << inference_ms << "ms pose=(" << deg(pose_yaw) << ", " << deg(pose_pitch)
                      << ") algo=(" << deg(algo_yaw) << ", " << deg(algo_pitch)
                      << ") find=" << static_cast<int>(send_data.is_find_buff)
                      << " fire=" << static_cast<int>(send_data.is_enable_fire) << " hits=" << hits
                      << " scored=" << scored << " act=" << activations << std::endl;
        }
        ++frame_index;
    }

    control_stop.store(true);
    if (control_thread.joinable())
        control_thread.join();

    std::cout << "[rp26_sim] control thread: find_frames=" << control_find_frames.load()
              << " fire_frames=" << control_fire_frames.load()
              << " fire_sent=" << fire_sent_atomic.load() << std::endl;
    std::cout << "[rp26_sim] done: frames=" << frames_total
              << " frames_with_detection=" << frames_with_detection << " det_class_counts="
              << det_stats[0] << "/" << det_stats[1] << "/" << det_stats[2]
              << " fire_sent=" << fire_sent << " hits=" << hits << " scored=" << scored
              << " activations=" << activations << " first_activation_s="
              << (first_activation_s.has_value() ? std::to_string(*first_activation_s) : "none")
              << std::endl;
    return 0;
}
