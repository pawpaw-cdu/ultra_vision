// 相机 → 云台外参标定（hand-eye），针对「C 板回传**编码器绝对角**」的情况重写。
//
// 与上一版的区别（上一版是为了应付"陀螺积分 yaw 一直漂"的临时状态）：
//   1) 不再需要任何漂移补偿（ZUPT / settle / 拟合 β）——编码器角不漂；
//   2) 编码器绝对角会在 ±180° 折返，所以样本先**解缠**，跨度判据按解缠后的值算；
//   3) 新增 yaw/pitch **标度自标定**（--fit-yaw-scale / --fit-pitch-scale）：C 板那边的角
//      如果有减速比/单位/符号问题，解出来的 κ 会直接告诉你差多少倍（κ=0.5 就是差 2 倍，
//      κ<0 就是符号反了）；
//   4) 新增**半分交叉验证**：一半样本拟合、另一半检验，检验残差必须和拟合残差同量级才允许
//      写盘（防"残差很小但换一半样本就崩"的过拟合解）；
//   5) 新增**平移可辨识度 σt**：由残差 + 雅可比算平移三轴 1σ，>0.1 m 说明该方向没有信息量
//      （转动范围不够），直接提示，而不是给一个乱跑的 |t|。
//
// 用法：
//   hand_eye_calibrate --auto                 自动采样（推荐）
//   hand_eye_calibrate <样本目录>              离线复算 + 交叉验证
//   hand_eye_calibrate --selftest             合成数据自检（含 ±180 折返与标度错误）
//
// 选项：--config-dir configs --out hand_eye.yaml --min-samples 20 --min-step 3
//       --yaw-span 25 --pitch-span 12 --save-frames <目录>
//       --fit-yaw-scale --fit-pitch-scale
//
// 模型（与 tracker 的 cameraToBaseRotation 完全一致）：
//   p_base = Ry(yaw)·Rx(pitch) · (R_camera2gimbal · p_cam + t_camera2gimbal)
// 标定物静止时，各组样本的 p_base 应重合成一点 → 最小化这个"离散度"。
// 残差 = 板心投回底盘系后到中位点的平均距离（mm），<10 mm 可用。

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/core/ocl.hpp>
#include <opencv2/opencv.hpp>

#include "config_loader.hpp"
#include "io/gimbal/gimbal.hpp"
#include "tools/calibration/board.hpp"
#include "tools/calibration/handeye_solver.hpp"
#include "tools/calibration/pose.hpp"

#if defined(ULTRA_VISION_USE_HIK_CAMERA)
#include "io/camera/HikCamera.hpp"
using CameraType = rm_ultra::HikCamera;
#elif defined(ULTRA_VISION_USE_GALAXY_CAMERA)
#include "io/camera/GalaxyCamera.hpp"
using CameraType = rm_ultra::GalaxyCamera;
#endif

namespace
{
    using FrameTime = std::chrono::steady_clock::time_point;
    using calibration::BoardConfig;
    using calibration::BoardObservation;
    using calibration::Fit;
    using calibration::FitOptions;
    using calibration::Sample;
    using calibration::boardObjectPoints;
    using calibration::crossValidate;
    using calibration::detectBoard;
    using calibration::gimbalToBase;
    using calibration::loadBoardConfig;
    using calibration::printFit;
    using calibration::quaternionToRotation;
    using calibration::sanityProblem;
    using calibration::saveIfGood;
    using calibration::scanYawScale;
    using calibration::solve;
    using calibration::unwrapSamples;
    using calibration::unwrapYaw;
    using calibration::wrapPi;

#if defined(ULTRA_VISION_USE_HIK_CAMERA) || defined(ULTRA_VISION_USE_GALAXY_CAMERA)
    /// @brief 带时间戳取图。海康驱动有该重载（时间戳在采集线程取图返回时打）；
    ///        其它驱动没有，就退化成"取图后就地打戳"（仍比处理完再打准）。
    bool getImageTimed(CameraType& camera, cv::Mat& image, FrameTime& timestamp, int timeout_ms)
    {
#if defined(ULTRA_VISION_USE_HIK_CAMERA)
        return camera.getImage(image, timestamp, timeout_ms);
#else
        if (!camera.getImage(image, timeout_ms)) return false;
        timestamp = std::chrono::steady_clock::now();
        return true;
#endif
    }
#endif

    struct Options
    {
        int min_samples = 20;
        double min_step_deg = 3.0;
        double yaw_span_target = 25.0;
        double pitch_span_target = 12.0;
        std::string frame_dir;
        double yaw_scale_fixed = 1.0;   // --yaw-scale：给回传 yaw 乘一个固定标度（含符号）
        /// @brief --pitch-scale：给回传 pitch 乘一个固定标度（含符号）。
        ///        用来**验证 pitch 符号**：+1 与 −1 都跑一遍比残差，残差小的那个才是
        ///        与观测一致的约定（pitch 跨度不够大时两者可能都拟合得不错，那种情况
        ///        残差不会相差数量级，就别用它下结论）。
        double pitch_scale_fixed = 1.0;
        double max_rms = 0.8;           // --max-rms：单帧板重投影 rms 上限（px）
        FitOptions fit;
    };

    /// @brief 从 configs/serial.yaml 读 yaw 符号（+1/-1）。C 板编码器 yaw 的正方向如果和
    ///        tracker 约定（正 yaw = 右转）相反，就在这里填 -1 —— 工具和 tracker 共用这一项，
    ///        免得两边各拍一个符号、谁也说不清。
    double loadYawSign(const std::string& config_dir)
    {
        std::ifstream probe(config_dir + "/serial.yaml");
        if (!probe) return 1.0;
        const YAML::Node file = YAML::LoadFile(config_dir + "/serial.yaml");
        const YAML::Node serial = file["serial"] ? file["serial"] : file;
        return serial["yaw_sign"].as<double>(1.0) < 0.0 ? -1.0 : 1.0;
    }

#if defined(ULTRA_VISION_USE_HIK_CAMERA) || defined(ULTRA_VISION_USE_GALAXY_CAMERA)
    /// @brief 相机 + 串口 + 配置，各模式共用。
    struct Hardware
    {
        CameraType camera;
        std::unique_ptr<io::Gimbal> gimbal;
        cv::Mat camera_matrix;
        cv::Mat dist_coeffs;
        BoardConfig board;
        double yaw_sign = 1.0;
        bool ok = false;
    };

    /// @brief 打开相机 + 串口。相机/串口都不可拷贝，所以用出参就地构造。
    bool openHardware(Hardware& hardware, const std::string& config_dir)
    {
        const YAML::Node camera_file = YAML::LoadFile(config_dir + "/camera.yaml");
        const YAML::Node serial_file = YAML::LoadFile(config_dir + "/serial.yaml");
        const std::string camera_name = camera_file["camera"]["name"].as<std::string>("hikcamera");
        hardware.camera_matrix = auto_aim::readMatFromYaml(camera_file[camera_name]["camera_matrix"]);
        hardware.dist_coeffs = auto_aim::readMatFromYaml(camera_file[camera_name]["dist_coeffs"]);
        hardware.board = loadBoardConfig(config_dir);
        hardware.yaw_sign = loadYawSign(config_dir);
        if (!hardware.camera.init("", camera_file["camera"]["device_index"].as<int>(1))) {
            std::cerr << "[hand_eye] 打不开相机（被 MVS 客户端/别的进程占着？）" << std::endl;
            return false;
        }
        if (const YAML::Node hik = camera_file[camera_name]) {
            hardware.camera.setAutoExposure(hik["auto_exposure"].as<bool>(false));
            hardware.camera.setExposureTime(hik["exposure_ms"].as<double>(6.0) * 1000.0);
            hardware.camera.setGain(hik["gain"].as<double>(12.0));
        }
        hardware.gimbal = std::make_unique<io::Gimbal>(auto_aim::loadSerialConfig(serial_file));
        if (!hardware.gimbal->connected()) {
            std::cerr << "[hand_eye] 下位机链路没通：标定需要 C 板回传的 yaw/pitch" << std::endl;
            return false;
        }
        hardware.ok = true;
        return true;
    }

    /// @brief 一帧的观测量。
    struct Observation
    {
        bool found = false;
        bool full = false;
        double rms = -1.0;
        double tilt_deg = 0.0;
        double dist_m = 0.0;
        std::vector<cv::Point2f> corners;
        cv::Mat rvec;
        cv::Mat tvec;
        double yaw = 0.0;       // 解缠 + 符号修正后的 yaw（rad，用于求解）
        double yaw_raw = 0.0;   // **原始**回传 yaw（解缠后，未做符号修正）——存盘用这个，
                                // 离线复算再按 configs/serial.yaml 的 yaw_sign 应用一次，
                                // 避免"采集时取了反、存盘又取反"的双重取反
        double pitch = 0.0;
        bool have_pose = false;
    };

    /// @brief 检测标定板 + 取该帧时刻的云台角（yaw 已解缠）。
    Observation observeAt(Hardware& hardware, const cv::Mat& frame, FrameTime frame_time,
                          double yaw_reference)
    {
        Observation observation;
        // 姿态先取：没找到板的时候也要能显示"这一帧的云台角"（监控模式靠它看比值）
        io::ImuSample pose;
        if (hardware.gimbal->imuAt(frame_time, pose)) {
            observation.yaw_raw = unwrapYaw(pose.yaw, yaw_reference);
            observation.yaw = hardware.yaw_sign * observation.yaw_raw;
            observation.pitch = pose.pitch;
            observation.have_pose = true;
        }
        BoardObservation board_observation;
        if (!detectBoard(frame, hardware.board, hardware.camera_matrix, hardware.dist_coeffs,
                         board_observation)) {
            return observation;
        }
        observation.corners = std::move(board_observation.corners);
        observation.rvec = std::move(board_observation.rvec);
        observation.tvec = std::move(board_observation.tvec);
        observation.rms = board_observation.reprojection_rms_px;
        observation.found = true;
        observation.full = true;
        for (const auto& corner : observation.corners) {
            if (corner.x < 20.0 || corner.y < 20.0 || corner.x > frame.cols - 20.0 ||
                corner.y > frame.rows - 20.0) {
                observation.full = false;
                break;
            }
        }
        cv::Mat rotation;
        cv::Rodrigues(observation.rvec, rotation);
        const cv::Mat normal = rotation * (cv::Mat_<double>(3, 1) << 0.0, 0.0, 1.0);
        observation.tilt_deg =
            std::acos(std::min(1.0, std::abs(normal.at<double>(2)))) * 180.0 / CV_PI;
        observation.dist_m = cv::norm(observation.tvec);
        return observation;
    }

    void spansOf(const std::vector<Sample>& samples, double& yaw_span, double& pitch_span)
    {
        yaw_span = 0.0;
        pitch_span = 0.0;
        if (samples.empty()) return;
        double yaw_min = 1e9, yaw_max = -1e9, pitch_min = 1e9, pitch_max = -1e9;
        for (const auto& sample : samples) {
            yaw_min = std::min(yaw_min, sample.yaw);
            yaw_max = std::max(yaw_max, sample.yaw);
            pitch_min = std::min(pitch_min, sample.pitch);
            pitch_max = std::max(pitch_max, sample.pitch);
        }
        yaw_span = (yaw_max - yaw_min) * 180.0 / CV_PI;
        pitch_span = (pitch_max - pitch_min) * 180.0 / CV_PI;
    }

    /// @brief 自动采样：板子静止、云台转动，满足判据就记一组；采够覆盖自动解算+写盘。
    int collectAuto(const std::string& config_dir, const std::string& output, const Options& options)
    {
        Hardware hardware;
        if (!openHardware(hardware, config_dir)) return 1;
        // 默认也把样本帧存下来：万一中途退出/结果不达标，数据不会丢（可以用目录模式离线复算）。
        const std::string frame_dir = options.frame_dir.empty() ? "hand_eye_samples"
                                                                : options.frame_dir;
        std::system(("mkdir -p " + frame_dir).c_str());
        std::cout << "[hand_eye] 样本帧会存到 " << frame_dir << "/（{i}.jpg + {i}.yaml）"
                  << std::endl;

        std::cout << "[hand_eye] yaw 符号（configs/serial.yaml 的 yaw_sign）= " << hardware.yaw_sign
                  << std::endl;
        std::cout << "[hand_eye] 自动采样：标定板（" << hardware.board.pattern << " "
                  << hardware.board.cols << "x" << hardware.board.rows
                  << "）摆在画面里**别动**，然后慢慢转云台：yaw 扫一大段（跨度 ≥"
                  << options.yaw_span_target << "°）、pitch 分几档（≥" << options.pitch_span_target
                  << "°）。采够 " << options.min_samples << " 组自动收工。"
                  << std::endl;

        std::vector<Sample> samples;
        std::deque<cv::Vec3d> recent;
        constexpr int kRecent = 4;
        const auto session_start = std::chrono::steady_clock::now();
        double yaw_reference = 0.0;
        bool have_reference = false;
        auto last_status = std::chrono::steady_clock::now();

        while (true) {
            cv::Mat frame;
            FrameTime frame_time;
            if (!getImageTimed(hardware.camera, frame, frame_time, 500) || frame.empty()) continue;
            if (!have_reference) {
                io::ImuSample pose;
                if (hardware.gimbal->latestImu(pose)) {
                    yaw_reference = pose.yaw;
                    have_reference = true;
                }
                continue;
            }
            const Observation observation =
                observeAt(hardware, frame, frame_time, yaw_reference);

            // 静止判据：最近 4 帧板心抖动 <8 mm（板的位姿变化直接反映相机在不在动）
            double jitter_mm = 1e9;
            if (observation.found && observation.full) {
                recent.push_back(cv::Vec3d(observation.tvec));
                while (static_cast<int>(recent.size()) > kRecent) recent.pop_front();
                if (static_cast<int>(recent.size()) == kRecent) {
                    jitter_mm = 0.0;
                    for (const auto& point : recent) {
                        jitter_mm = std::max(jitter_mm, cv::norm(point - recent.back()) * 1000.0);
                    }
                }
            } else {
                recent.clear();
            }
            const bool still = jitter_mm < 8.0;
            const bool slow = observation.have_pose;

            // 新鲜度：和已采样本的 yaw/pitch 都差得够远
            double nearest_deg = 1e9;
            for (const auto& sample : samples) {
                const double dy = (sample.yaw - observation.yaw) * 180.0 / CV_PI;
                const double dp = (sample.pitch - observation.pitch) * 180.0 / CV_PI;
                nearest_deg = std::min(nearest_deg, std::hypot(dy, dp));
            }
            const bool novel = nearest_deg > options.min_step_deg;

            double yaw_span = 0.0, pitch_span = 0.0;
            spansOf(samples, yaw_span, pitch_span);
            const char* status = !observation.found    ? "NO BOARD"
                                  : !observation.full  ? "BOARD CUT"
                                  : observation.rms > options.max_rms ? "RMS HIGH"
                                  : !still             ? "MOVING"
                                  : !novel             ? "SAME POSE"
                                                       : "OK";
            const bool accept = observation.found && observation.full && slow &&
                                observation.rms < options.max_rms && novel && still;
            if (accept) {
                Sample sample;
                sample.yaw = observation.yaw;
                sample.pitch = observation.pitch;
                sample.rvec = observation.rvec.clone();
                sample.tvec = observation.tvec.clone();
                sample.time_s = std::chrono::duration<double>(frame_time - session_start).count();
                samples.push_back(sample);
                std::cout << "[hand_eye] 自动采第 " << samples.size()
                          << " 组：yaw="
                          << sample.yaw * 180.0 / CV_PI << "° pitch="
                          << sample.pitch * 180.0 / CV_PI << "° rms=" << observation.rms
                          << "px dist=" << cv::norm(sample.tvec) << "m 覆盖(yaw " << yaw_span
                          << "° / pitch " << pitch_span << "°)" << std::endl;
                {
                    const std::string stem = frame_dir + "/" + std::to_string(samples.size());
                    cv::imwrite(stem + ".jpg", frame);
                    std::ofstream pose_out(stem + ".yaml");
                    pose_out << std::setprecision(10) << observation.yaw_raw << " "
                             << observation.pitch << " " << sample.time_s << "\n";
                }
                if (samples.size() >= 8) {
                    Fit fit;
                    if (solve(samples, options.fit, fit)) {
                        double in_sample = 0.0;
                        fit.cross_validate_mm = crossValidate(samples, options.fit, &in_sample);
                        printFit(samples, fit, "[hand_eye] 即时解算：");
                        if (samples.size() >= 12 && samples.size() % 4 == 0) {
                            scanYawScale(samples, options.fit, false);
                        }
                        if (sanityProblem(samples, fit).empty() &&
                            samples.size() >= static_cast<std::size_t>(options.min_samples) &&
                            yaw_span >= options.yaw_span_target &&
                            pitch_span >= options.pitch_span_target && saveIfGood(output, samples, fit)) {
                            std::cout << "[hand_eye] 覆盖够 + 体检通过，标定结束。" << std::endl;
                            break;
                        }
                        if (samples.size() >= static_cast<std::size_t>(options.min_samples) &&
                            yaw_span >= options.yaw_span_target &&
                            pitch_span >= options.pitch_span_target) {
                            std::cout << "[hand_eye] 覆盖够了但这个解没过质检（见上面一行），"
                                         "继续采样。" << std::endl;
                        }
                    }
                }
            } else {
                const auto now = std::chrono::steady_clock::now();
                if (std::chrono::duration<double>(now - last_status).count() >= 3.0) {
                    last_status = now;
                    std::cout << "[hand_eye] " << status << "  yaw="
                              << observation.yaw * 180.0 / CV_PI << "° pitch="
                              << observation.pitch * 180.0 / CV_PI << "° rms=" << observation.rms
                              << "px 抖动=" << jitter_mm << "mm 已采 " << samples.size() << " 组"
                              << std::endl;
                }
            }
        }
        return 0;
    }

    /// @brief 手动采样：空格采一组、s 存盘、u 撤销、q 退出。
#endif

    /// @brief 合成数据自检：(a) 精确恢复；(b) yaw ±180 折返；(c) yaw 标度错误可被解出来。
    int selftest()
    {
        // 真实设置
        const cv::Matx33d R_true = [] {
            cv::Matx33d R;
            cv::Rodrigues(cv::Vec3d(0.03, -0.02, 0.15), R);
            return R;
        }();
        const cv::Vec3d t_true(0.03, -0.01, 0.05);
        const cv::Vec3d board(1.2, 0.35, -1.8);      // 靶标要离开 yaw 轴，否则绕 z 转不可观测

        auto makeSamples = [&](double yaw_scale, double yaw_offset, bool wrap) {
            std::vector<Sample> samples;
            for (int i = 0; i < 16; ++i) {
                const double yaw = (-14.0 + 28.0 * i / 15.0) * CV_PI / 180.0;
                const double pitch = (-8.0 + 16.0 * (i % 4) / 3.0) * CV_PI / 180.0;
                const cv::Matx33d attitude = gimbalToBase(yaw, pitch);
                Sample sample;
                sample.yaw = yaw_scale * yaw + yaw_offset;   // C 板回传的角（可能带标度/零偏）
                sample.pitch = pitch;
                if (wrap) sample.yaw = wrapPi(sample.yaw);   // 编码器在 ±180° 折返
                sample.tvec = (cv::Mat_<double>(3, 1) <<
                    (R_true.t() * (attitude.t() * board - t_true))[0],
                    (R_true.t() * (attitude.t() * board - t_true))[1],
                    (R_true.t() * (attitude.t() * board - t_true))[2]);
                samples.push_back(sample);
            }
            unwrapSamples(samples);
            return samples;
        };

        // (a) 基本恢复
        {
            auto samples = makeSamples(1.0, 0.0, false);
            Fit fit;
            if (!solve(samples, {}, fit)) return 1;
            const double translation_error = cv::norm(fit.t - t_true) * 1000.0;
            cv::Mat delta;
            cv::Rodrigues(fit.R * R_true.t(), delta);
            std::printf("自检(a) 基本恢复：残差 %.2f mm，平移误差 %.1f mm，转动误差 %.2f°，|t|=%.3f m\n",
                        fit.residual_mm, translation_error,
                        cv::norm(cv::Vec3d(delta)) * 180.0 / CV_PI, cv::norm(fit.t));
            if (fit.residual_mm > 1.0 || translation_error > 10.0) {
                std::printf("FAILED: 基本恢复没过\n");
                return 1;
            }
        }
        // (b) yaw ±180 折返
        {
            auto samples = makeSamples(1.0, 176.0 * CV_PI / 180.0, true);   // 让 yaw 跨越 ±180
            Fit fit;
            if (!solve(samples, {}, fit)) return 1;
            std::printf("自检(b) yaw 折返(偏移 176°)：残差 %.2f mm，|t|=%.3f m\n", fit.residual_mm,
                        cv::norm(fit.t));
            if (fit.residual_mm > 1.0) {
                std::printf("FAILED: 折返解缠没处理对\n");
                return 1;
            }
        }
        // (c) yaw 标度错误（0.8 倍）+ 把它解出来
        {
            auto samples = makeSamples(0.8, 0.0, false);
            Fit fit;
            FitOptions options;
            options.fit_yaw_scale = true;
            if (!solve(samples, options, fit)) return 1;
            // 注意语义：解出来的是**修正系数** κ（gimbalToBase(κ·回传角)）。回传侧少乘了 0.8，
            // 所以修正系数应当 ≈ 1/0.8 = 1.25；换成"回传侧实际标度"就是 1/κ ≈ 0.80。
            std::printf("自检(c) 回传 yaw 标度 0.8：修正系数解出 %.3f（=回传标度 %.3f），残差 %.2f mm\n",
                        fit.yaw_scale, 1.0 / fit.yaw_scale, fit.residual_mm);
            if (std::abs(fit.yaw_scale - 1.25) > 0.05 || fit.residual_mm > 1.0) {
                std::printf("FAILED: 标度没解出来\n");
                return 1;
            }
            // 不放开标度时应当解不出来（残差大）→ 说明这条诊断是有意义的
            Fit fixed;
            if (!solve(samples, {}, fixed)) return 1;
            std::printf("        不放开标度时残差 %.1f mm（应当明显变大）\n", fixed.residual_mm);
            if (fixed.residual_mm < 5.0) {
                std::printf("FAILED: 标度错误居然没被发现\n");
                return 1;
            }
        }
        std::printf("自检全部通过\n");
        return 0;
    }

}  // namespace

int main(int argc, char* argv[])
{
    cv::ocl::setUseOpenCL(false);   // macOS 上 OpenCL 缓存会刷屏；标定用不到

    std::string folder;
    std::string config_dir = "configs";
    std::string output = "hand_eye.yaml";
    Options options;
    bool auto_mode = false;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            std::cout << "用法:\n"
                         "  hand_eye_calibrate --auto                 自动采样（板子静止、云台转动，"
                         "采够自动解算+写盘）\n"
                         "  hand_eye_calibrate <样本目录>              离线复算 + 交叉验证\n"
                         "  hand_eye_calibrate --selftest             合成数据自检\n"
                         "选项：\n"
                         "  --config-dir configs  --out hand_eye.yaml\n"
                         "  --min-samples 20 --min-step 3 --yaw-span 25 --pitch-span 12\n"
                         "  --save-frames <目录>      自动模式把样本帧存下来\n"
                         "  --yaw-scale <k>           给回传 yaw 乘固定标度 k（含符号；标度扫描会给出建议）\n"
                         "  --fit-yaw-scale           把 yaw 标度当未知量解（**噪声下会退化，"
                         "只建议在无噪合成数据上用**；查标度请用扫描）\n"
                         "  --fit-pitch-scale         同上（pitch）\n";
            return 0;
        }
        if (arg == "--selftest") return selftest();
        else if (arg == "--auto") auto_mode = true;
        else if (arg == "--yaw-scale" && i + 1 < argc) options.yaw_scale_fixed = std::atof(argv[++i]);
        else if (arg == "--pitch-scale" && i + 1 < argc)
            options.pitch_scale_fixed = std::atof(argv[++i]);
        else if (arg == "--max-rms" && i + 1 < argc) options.max_rms = std::atof(argv[++i]);
        else if (arg == "--fit-yaw-scale") options.fit.fit_yaw_scale = true;
        else if (arg == "--fit-pitch-scale") options.fit.fit_pitch_scale = true;
        else if (arg == "--min-samples" && i + 1 < argc) options.min_samples = std::atoi(argv[++i]);
        else if (arg == "--min-step" && i + 1 < argc) options.min_step_deg = std::atof(argv[++i]);
        else if (arg == "--yaw-span" && i + 1 < argc) options.yaw_span_target = std::atof(argv[++i]);
        else if (arg == "--pitch-span" && i + 1 < argc)
            options.pitch_span_target = std::atof(argv[++i]);
        else if (arg == "--save-frames" && i + 1 < argc) options.frame_dir = argv[++i];
        else if (arg == "--config-dir" && i + 1 < argc) config_dir = argv[++i];
        else if (arg == "--out" && i + 1 < argc) output = argv[++i];
        else if (folder.empty()) folder = arg;
    }

    if (auto_mode) {
#if defined(ULTRA_VISION_USE_HIK_CAMERA) || defined(ULTRA_VISION_USE_GALAXY_CAMERA)
        return collectAuto(config_dir, output, options);
#else
        std::cerr << "--auto 需要带相机 SDK 的构建" << std::endl;
        return 2;
#endif
    }
    if (folder.empty()) {
        std::cerr << "给个样本目录，或 --auto / --selftest（--help 看用法）"
                  << std::endl;
        return 2;
    }

    // ---- 离线复算 ----
    const BoardConfig board = loadBoardConfig(config_dir);
    const YAML::Node camera_file = YAML::LoadFile(config_dir + "/camera.yaml");
    const std::string camera_name = camera_file["camera"]["name"].as<std::string>("hikcamera");
    const cv::Mat camera_matrix = auto_aim::readMatFromYaml(camera_file[camera_name]["camera_matrix"]);
    const cv::Mat dist_coeffs = auto_aim::readMatFromYaml(camera_file[camera_name]["dist_coeffs"]);

    std::vector<Sample> samples;
    for (int index = 1;; ++index) {
        const std::string image_path = folder + "/" + std::to_string(index) + ".jpg";
        const std::string pose_path = folder + "/" + std::to_string(index) + ".yaml";
        if (!std::ifstream(image_path).good()) break;
        const cv::Mat image = cv::imread(image_path);
        if (image.empty()) break;
        std::ifstream pose_in(pose_path);
        if (!pose_in) {
            std::cerr << "[hand_eye] 缺 " << pose_path << "（自动模式会一起存）" << std::endl;
            break;
        }
        Sample sample;
        pose_in >> sample.yaw >> sample.pitch;
        if (!(pose_in >> sample.time_s)) sample.time_s = 0.0;
        BoardObservation board_observation;
        if (!detectBoard(image, board, camera_matrix, dist_coeffs, board_observation)) {
            std::cout << "[hand_eye] 第 " << index << " 张没找到标定板，跳过" << std::endl;
            continue;
        }
        sample.rvec = std::move(board_observation.rvec);
        sample.tvec = std::move(board_observation.tvec);
        std::cout << "[hand_eye] 第 " << index << " 张：板 "
                  << board_observation.corners.size() << " 点，重投影 rms="
                  << board_observation.reprojection_rms_px << " px，距离 "
                  << cv::norm(sample.tvec) << " m"
                  << std::endl;
        samples.push_back(sample);
    }
    if (samples.size() < 8) {
        std::cerr << "[hand_eye] 有效样本 " << samples.size() << " 组（<8 组解不稳）" << std::endl;
        return 1;
    }
    unwrapSamples(samples);
    const double yaw_sign = loadYawSign(config_dir);
    if (yaw_sign < 0.0) {
        for (auto& sample : samples) sample.yaw = -sample.yaw;
        std::cout << "[hand_eye] 已按 configs/serial.yaml 的 yaw_sign = -1 取反 yaw" << std::endl;
    }
    if (options.yaw_scale_fixed != 1.0) {
        for (auto& sample : samples) sample.yaw *= options.yaw_scale_fixed;
        std::cout << "[hand_eye] 已按 --yaw-scale " << options.yaw_scale_fixed
                  << " 缩放 yaw（符号/标度修正）" << std::endl;
    }
    {
        // 样本的姿态跨度：pitch 跨度太小的话，"pitch 符号"这件事在标定里是**测不出来**的，
        // 必须明说，免得拿一版拟合很好的外参去现场发现竖直方向是镜像的。
        double yaw_min = 1e9, yaw_max = -1e9, pitch_min = 1e9, pitch_max = -1e9;
        for (const auto& sample : samples) {
            yaw_min = std::min(yaw_min, sample.yaw);
            yaw_max = std::max(yaw_max, sample.yaw);
            pitch_min = std::min(pitch_min, sample.pitch);
            pitch_max = std::max(pitch_max, sample.pitch);
        }
        const double yaw_span = (yaw_max - yaw_min) * 180.0 / CV_PI;
        const double pitch_span = (pitch_max - pitch_min) * 180.0 / CV_PI;
        std::cout << "[hand_eye] 样本姿态跨度：yaw " << yaw_span << "°，pitch " << pitch_span
                  << "°" << (pitch_span < 15.0
                                 ? "（pitch 偏小：这一版外参**测不出** pitch 符号，"
                                   "竖直方向要用 --q 的数据另外验）"
                                 : "")
                  << std::endl;
    }
    if (options.pitch_scale_fixed != 1.0) {
        for (auto& sample : samples) sample.pitch *= options.pitch_scale_fixed;
        std::cout << "[hand_eye] 已按 --pitch-scale " << options.pitch_scale_fixed
                  << " 缩放 pitch（符号/标度修正）" << std::endl;
    }

    const double best_scale = scanYawScale(samples, options.fit, true);
    if (std::abs(best_scale - 1.0) > 0.05) {
        std::cout << "[hand_eye] 注意：yaw 标度最优值 " << best_scale
                  << "（不是 1.0）→ C 板回传的 yaw 有 ≈" << 1.0 / best_scale
                  << " 倍的标度问题；下面先用 κ=1 解，要按最优值解就再跑一次并乘上它" << std::endl;
    }

    Fit fit;
    if (!solve(samples, options.fit, fit)) return 1;
    double in_sample = 0.0;
    fit.cross_validate_mm = crossValidate(samples, options.fit, &in_sample);
    printFit(samples, fit, "[hand_eye] 解算：");
    std::cout << "[hand_eye] R_camera2gimbal = [";
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            std::cout << (row || column ? "," : "") << fit.R(row, column);
        }
    }
    std::cout << "]" << std::endl;
    std::cout << "[hand_eye] t_camera2gimbal = [" << fit.t[0] << "," << fit.t[1] << "," << fit.t[2]
              << "] m" << std::endl;
    saveIfGood(output, samples, fit);
    return 0;
}
