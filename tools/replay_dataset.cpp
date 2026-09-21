// 离线回放实车数据集（由 node.cpp 的 ULTRA_VISION_RECORD 录制）：
// 把录制的图像 + 该帧的姿态/弹速重新喂进**和真机完全相同**的流水线
// （ArmorSource + AimPipeline），用于"改代码/改配置之后，离线确认效果"。
//
// 用法：
//   replay_dataset <数据集目录> [--config-dir configs] [--yaw-sign ±1]
//                  [--max-frames N] [--compare] [--video out.mp4]
//
//   --compare  逐帧和录制时的输出对比（检测数、跟踪状态、瞄点 yaw/pitch）
//   --video    把叠加了检测/模型/瞄点的画面写成 mp4，肉眼对比
#include <opencv2/core/ocl.hpp>
#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "config_loader.hpp"
#include "control/aim_pipeline.hpp"
#include "dataset.hpp"
#include "perception/armor_source.hpp"
#include "visualization/projection.hpp"

namespace
{
    double yawSignFromConfig(const std::string& config_dir)
    {
        std::ifstream probe(config_dir + "/serial.yaml");
        if (!probe) return 1.0;
        const YAML::Node file = YAML::LoadFile(config_dir + "/serial.yaml");
        const YAML::Node serial = file["serial"] ? file["serial"] : file;
        return serial["yaw_sign"].as<double>(1.0) < 0.0 ? -1.0 : 1.0;
    }
}

int main(int argc, char* argv[])
{
    cv::ocl::setUseOpenCL(false);
    std::string folder;
    std::string config_dir = "configs";
    std::string video_path;
    double yaw_sign = 0.0;      // 0 = 用 configs/serial.yaml 里的
    int max_frames = 0;
    bool compare = false;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--config-dir" && i + 1 < argc) config_dir = argv[++i];
        else if (arg == "--yaw-sign" && i + 1 < argc) yaw_sign = std::atof(argv[++i]);
        else if (arg == "--max-frames" && i + 1 < argc) max_frames = std::atoi(argv[++i]);
        else if (arg == "--compare") compare = true;
        else if (arg == "--video" && i + 1 < argc) video_path = argv[++i];
        else if (arg == "--help" || arg == "-h") {
            std::cout << "用法: replay_dataset <数据集目录> [--config-dir configs] [--yaw-sign ±1]"
                         " [--max-frames N] [--compare] [--video out.mp4]\n";
            return 0;
        } else if (folder.empty()) {
            folder = arg;
        }
    }
    if (folder.empty()) {
        std::cerr << "给一个数据集目录（node.cpp 用 ULTRA_VISION_RECORD 录出来的那个）"
                  << std::endl;
        return 2;
    }
    if (yaw_sign == 0.0) yaw_sign = yawSignFromConfig(config_dir);

    std::vector<auto_aim::DatasetRow> rows;
    if (!auto_aim::readDatasetRows(folder, rows)) {
        std::cerr << "[replay] 读不到 " << folder << "/meta.csv" << std::endl;
        return 1;
    }
    if (max_frames > 0 && static_cast<int>(rows.size()) > max_frames) rows.resize(max_frames);
    std::cout << "[replay] " << rows.size() << " 帧，yaw_sign = " << yaw_sign
              << "，config-dir = " << config_dir << std::endl;

    // ---- 和 node.cpp 完全相同的一套配置与流水线 ----
    const YAML::Node camera_file = YAML::LoadFile(config_dir + "/camera.yaml");
    const YAML::Node detector_file = YAML::LoadFile(config_dir + "/detector.yaml");
    const YAML::Node tracker_file = YAML::LoadFile(config_dir + "/tracker.yaml");
    const std::string camera_name = camera_file["camera"]["name"].as<std::string>("hikcamera");
    const cv::Mat camera_matrix =
        auto_aim::readMatFromYaml(camera_file[camera_name]["camera_matrix"]);
    const cv::Mat dist_coeffs = auto_aim::readMatFromYaml(camera_file[camera_name]["dist_coeffs"]);
    auto_aim::CAMERA_MATRIX = camera_matrix;   // 投影叠加用的全局内参
    auto_aim::DIST_COEFFS = dist_coeffs;
    const auto extrinsics = auto_aim::loadCameraExtrinsics(camera_file, camera_name);

    auto_aim::ArmorSourceConfig source_cfg;
    source_cfg.classical = auto_aim::loadDetectorConfig(detector_file);
    source_cfg.pnp = auto_aim::loadPnpGeometry(tracker_file);
#ifdef ULTRA_VISION_USE_OPENVINO
    source_cfg.use_neural = auto_aim::neuralDetectorEnabled(detector_file);
    source_cfg.neural = auto_aim::loadNeuralDetectorConfig(detector_file, config_dir);
#endif
    auto_aim::ArmorSource source(source_cfg, camera_matrix, dist_coeffs);

    auto_aim::AimPipelineConfig pipeline_cfg;
    pipeline_cfg.tracker = auto_aim::loadTrackerConfig(tracker_file);
    pipeline_cfg.selector = auto_aim::loadSelectorConfig(tracker_file);
    pipeline_cfg.gimbal = auto_aim::loadGimbalConfig(tracker_file);
    pipeline_cfg.shooter = auto_aim::loadShooterConfig(tracker_file);
    pipeline_cfg.aim_filter = auto_aim::loadAimFilterConfig(tracker_file);
    pipeline_cfg.command_rate_hz = auto_aim::loadGimbalCommandRateHz(tracker_file);
    pipeline_cfg.max_control_lost_frames = auto_aim::loadMaxControlLostFrames(tracker_file);
    pipeline_cfg.heartbeat_hz = 20.0;
    auto_aim::AimPipeline pipeline(pipeline_cfg,
                                   [](const auto&, bool, double, double, double, double, double,
                                      double) { return true; });

    cv::VideoWriter video;
    if (!video_path.empty()) {
        video.open(video_path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), 30.0,
                   cv::Size(camera_matrix.at<double>(0, 2) * 2.0, camera_matrix.at<double>(1, 2) * 2.0));
    }

    int detections = 0;
    int pnp_ok = 0;
    int tracking = 0;
    int fire = 0;
    double sum_abs_yaw_error = 0.0;
    double sum_abs_pitch_error = 0.0;
    int compared = 0;
    double max_aim_diff = 0.0;
    for (std::size_t i = 0; i < rows.size(); ++i) {
        const auto& row = rows[i];
        std::ostringstream name;
        name << folder << "/" << std::setw(6) << std::setfill('0') << i << ".jpg";
        const cv::Mat image = cv::imread(name.str());
        if (image.empty()) {
            std::cerr << "[replay] 缺图像 " << name.str() << "，停止" << std::endl;
            break;
        }
        auto_aim::CameraPose pose;
        pose.valid = true;
        pose.timestamp_valid = true;
        pose.timestamp = row.t;
        pose.yaw = yaw_sign * row.yaw;      // 录制时存的是**原始**回传角，这里按配置应用
        pose.pitch = row.pitch;
        pose.camera_to_gimbal = extrinsics.rotation;
        pose.camera_to_gimbal_translation = extrinsics.translation;

        const auto result = source.update(image, static_cast<uint64_t>(i), 0, {},
                                          auto_aim::TrackerState::LOST, false);
        auto_aim::AimPipeline::FrameInput input;
        input.armors = result.armors;
        input.fresh = result.fresh;
        input.observation_time = row.t;
        input.now = row.t;
        input.pose = pose;
        input.bullet_speed = row.bullet_speed;
        input.allow_control = false;    // 回放不控制任何东西
        input.fire_enabled = false;
        const auto outcome = pipeline.update(input);

        detections += result.armors.empty() ? 0 : 1;
        for (const auto& armor : result.armors) {
            if (armor.solve_result) ++pnp_ok;
        }
        const int state = static_cast<int>(pipeline.tracker().getState());
        if (state == 2) ++tracking;
        if (outcome.fire) ++fire;
        sum_abs_yaw_error += std::abs(outcome.aim_yaw_error);
        sum_abs_pitch_error += std::abs(outcome.aim_pitch_error);
        if (compare) {
            const double diff_yaw = std::abs(outcome.decision.target_yaw - row.aim_yaw);
            const double diff_pitch = std::abs(outcome.decision.target_pitch - row.aim_pitch);
            max_aim_diff = std::max({max_aim_diff, diff_yaw, diff_pitch});
            ++compared;
        }
        if (video.isOpened()) {
            cv::Mat canvas = image.clone();
            for (const auto& armor : result.armors) {
                auto_aim::drawQuad(canvas,
                                   {armor.left.top, armor.right.top, armor.right.bottom,
                                    armor.left.bottom},
                                   armor.solve_result ? cv::Scalar(0, 255, 0)
                                                      : cv::Scalar(0, 165, 255),
                                   2);
            }
            cv::putText(canvas, cv::format("frame %zu/%zu  armors=%zu  state=%d", i + 1, rows.size(),
                                           result.armors.size(), state),
                        cv::Point(16, 40), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 255),
                        2);
            video.write(canvas);
        }
    }

    const double n = static_cast<double>(rows.size());
    std::cout << "[replay] 检测到板的帧 " << detections << "/" << rows.size() << "（"
              << 100.0 * detections / n << "%），PnP 成功 " << pnp_ok
              << "，TRACKING 帧 " << tracking << "（" << 100.0 * tracking / n << "%）"
              << "，fire 帧 " << fire << std::endl;
    std::cout << "[replay] 瞄点误差：平均 |yaw| = " << sum_abs_yaw_error / n * 180.0 / CV_PI
              << "°，平均 |pitch| = " << sum_abs_pitch_error / n * 180.0 / CV_PI << "°" << std::endl;
    if (compare) {
        std::cout << "[replay] 与录制时输出对比：" << compared << " 帧，瞄点最大差 "
                  << max_aim_diff * 180.0 / CV_PI << "°（>1° 说明这次改动确实改变了行为）"
                  << std::endl;
    }
    return 0;
}
