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
#include <deque>
#include <fstream>
#include <numeric>
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
    struct AimSample {
        double t = 0.0;
        double yaw = 0.0;
        double pitch = 0.0;
    };

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
    std::string dump_path;
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
        else if (arg == "--dump" && i + 1 < argc) dump_path = argv[++i];
        else if (arg == "--help" || arg == "-h") {
            std::cout << "用法: replay_dataset <数据集目录> [--config-dir configs] [--yaw-sign ±1]"
                         " [--max-frames N] [--compare] [--video out.mp4] [--dump aim.csv]\n";
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

    // ---- --dump：逐帧把"估计状态 → 决策 → 指令"整条链落成 CSV ----
    // 用途：静止靶下所有非零的量**都**是噪声，看这一张表就能定位"抖在哪一段"
    // （模型中心/速度抖 → 提前量抖 → 喂给云台的速度抖 → 云台抖）。
    std::ofstream dump;
    if (!dump_path.empty()) {
        dump.open(dump_path, std::ios::out | std::ios::trunc);
        dump << "frame,t,state,obs_id,aim_id,valid,model_yaw_deg,omega_deg_s,"
                "cx_mm,cy_mm,cz_mm,vx_mm_s,vy_mm_s,vz_mm_s,dist_m,lead_ms,"
                "aim_yaw_deg,aim_pitch_deg,aim_yaw_vel_deg_s,aim_pitch_vel_deg_s,"
                "obs_x_mm,obs_y_mm,obs_z_mm,reproj_px,still,"
                "raw_yaw_deg,raw_pitch_deg,pose_yaw_deg,pose_pitch_deg,"
                "cmd_yaw_deg,cmd_pitch_deg\n";
    }

    int detections = 0;
    int pnp_ok = 0;
    int tracking = 0;
    int fire = 0;
    double sum_abs_yaw_error = 0.0;
    double sum_abs_pitch_error = 0.0;
    int compared = 0;
    double max_aim_diff = 0.0;
    int reproj_frames = 0;
    int dist_frames = 0;
    int update_frames = 0;
    int axis_frames = 0;
    int size_frames = 0;
    double sum_size_ratio_w = 0.0;
    double sum_size_ratio_h = 0.0;
    double sum_r = 0.0, sum_t = 0.0, sum_u = 0.0;
    double sum_rr = 0.0, sum_tt = 0.0, sum_uu = 0.0;
    double max_r = 0.0, max_t = 0.0, max_u = 0.0;
    int update_stuck_frames = 0;
    double sum_update_move_mm = 0.0;
    double sum_update_residual_mm = 0.0;
    int debug_printed = 0;
    double sum_estimate_distance_mm = 0.0;
    double max_estimate_distance_mm = 0.0;
    double sum_estimate_bearing_deg = 0.0;
    double sum_reproj = 0.0;
    double max_reproj = 0.0;
    // 看门狗诊断：整段录制里出现过的**最大**拒绝比例（末尾那一帧的瞬时值说明不了
    // "这条判据平时离门限多远"，而门限就是按它定的）。
    double max_reject_ratio_seen = 0.0;
    std::size_t max_reject_ratio_frame = 0;
    // 瞄点指令抖动（静止靶下这就是云台看到的抖动量）：只统计**连续两帧都有有效瞄点**
    // 的相邻帧，避免把"重新入靶"当成抖动。角度按 dt 归一化成等效角速度另给一份。
    std::vector<double> aim_jump_yaw_deg;
    std::vector<double> aim_jump_pitch_deg;
    std::vector<double> aim_jump_yaw_rate;
    std::vector<double> aim_jump_pitch_rate;
    std::vector<double> rec_aim_jump_yaw_deg;
    std::vector<double> rec_aim_jump_pitch_deg;
    bool have_prev_aim = false;
    double prev_aim_yaw = 0.0, prev_aim_pitch = 0.0, prev_aim_t = 0.0;
    bool have_prev_rec_aim = false;
    double prev_rec_aim_yaw = 0.0, prev_rec_aim_pitch = 0.0, prev_rec_aim_t = 0.0;
    int plate_id_switches = 0;
    int plate_id_samples = 0;
    int prev_plate_id = -1;
    // 1 秒滑窗峰值-峰值（用户眼睛看到的"幅度"）
    std::deque<AimSample> aim_window;
    std::vector<double> aim_ptp_yaw_deg;
    std::vector<double> aim_ptp_pitch_deg;
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
        // update **前**先记下模型里各板的位置（相机系），用来判断这一帧的观测有没有被吸收
        std::array<cv::Vec3d, 4> before_update;
        bool have_before = false;
        if (pipeline.tracker().hasObservation()) {
            const auto positions_before = pipeline.tracker().getEstimatedArmorPositions();
            for (int k = 0; k < 4; ++k) {
                if (positions_before[k].rows == 3) {
                    before_update[k] = cv::Vec3d(positions_before[k]);
                    have_before = true;
                }
            }
        }
        const auto outcome = pipeline.update(input);
        if (have_before && pipeline.tracker().hasObservation()) {
            const int plate = pipeline.tracker().getLastObservedPlateId();
            const auto after = pipeline.tracker().getEstimatedArmorPositions()[plate];
            if (after.rows == 3) {
                const double moved = cv::norm(cv::Vec3d(after) - before_update[plate]) * 1000.0;
                const cv::Vec3d obs(pipeline.tracker().getLastObservedArmor().tvec);
                const double residual = cv::norm(cv::Vec3d(after) - obs) * 1000.0;
                ++update_frames;
                sum_update_move_mm += moved;
                if (moved < 1.0) ++update_stuck_frames;
                sum_update_residual_mm += residual;
                if (std::getenv("REPLAY_DEBUG") != nullptr && debug_printed < 10) {
                    std::printf("[dbg] f=%zu id=%d 更新使模型移动 %.1fmm，更新后残差 %.1fmm\n", i,
                                plate, moved, residual);
                    ++debug_printed;
                }
            }
        }
        if (std::getenv("REPLAY_DEBUG") != nullptr &&
            pipeline.tracker().hasObservation() && debug_printed < 12) {
            const int plate = pipeline.tracker().getLastObservedPlateId();
            const auto& armor = pipeline.tracker().getLastObservedArmor();
            const auto predicted = pipeline.tracker().getEstimatedArmorPositions()[plate];
            const cv::Vec3d obs(armor.tvec);
            const cv::Vec3d pred(predicted);
            std::printf(
                "[dbg] f=%zu obs_id=%d d_obs=%.0fmm 模型该板=%.0fmm 差=%.0fmm 方位差=%.1f° "
                "预测中心距=%.0fmm\n",
                i, plate, cv::norm(obs) * 1000.0, cv::norm(pred) * 1000.0,
                cv::norm(pred - obs) * 1000.0,
                std::acos(std::clamp(obs.dot(pred) / (cv::norm(obs) * cv::norm(pred)), -1.0, 1.0)) *
                    180.0 / CV_PI,
                cv::norm(cv::Vec3d(pipeline.tracker().getTargetCenter())));
            ++debug_printed;
        }

        // 直接量"估计 vs 观测"的差（都在相机系里，和投影/画框的细节无关）：
        //   · 观测 = 这一帧检测到的那块板中心（PnP tvec）
        //   · 估计 = EKF 里**同一个 id** 那块板的位置
        // 两者差多少米，就说明"底盘模型把这块板放错了多少"。
        double estimate_observed_mm = -1.0;
        double estimate_observed_bearing_deg = -1.0;
        if (pipeline.tracker().hasObservation()) {
            const int plate = pipeline.tracker().getLastObservedPlateId();
            const auto& armor = pipeline.tracker().getLastObservedArmor();
            const auto predicted = pipeline.tracker().getEstimatedArmorPositions()[plate];
            if (!predicted.empty() && !armor.tvec.empty()) {
                const cv::Vec3d observed_vec(armor.tvec);
                const cv::Vec3d predicted_vec(predicted);
                estimate_observed_mm = cv::norm(predicted_vec - observed_vec) * 1000.0;
                const double n1 = cv::norm(observed_vec);
                const double n2 = cv::norm(predicted_vec);
                if (n1 > 1e-6 && n2 > 1e-6) {
                    const double cosine = std::clamp(observed_vec.dot(predicted_vec) / (n1 * n2),
                                                     -1.0, 1.0);
                    estimate_observed_bearing_deg = std::acos(cosine) * 180.0 / CV_PI;
                }
                ++dist_frames;
                sum_estimate_distance_mm += estimate_observed_mm;
                max_estimate_distance_mm = std::max(max_estimate_distance_mm, estimate_observed_mm);
                sum_estimate_bearing_deg += estimate_observed_bearing_deg;
            }
        }

        // 把"模型板位置 − 观测板位置"分解到**径向 / 切向 / 竖直**三轴。
        // 全部在**相机系**里做，基准也从模型自身几何取（别用 getYaw()：它返回的是相机系 yaw，
        // 配世界系的 angleForPlate 会得到错误的基准 —— 之前就这么错过一次）：
        //   径向 = normalize(模型板位置 − 模型底盘中心)
        //   竖直 = worldDirectionToCamera((0,-1,0))
        //   切向 = normalize(竖直 × 径向)
        if (pipeline.tracker().hasObservation()) {
            const int plate = pipeline.tracker().getLastObservedPlateId();
            const auto& armor = pipeline.tracker().getLastObservedArmor();
            const auto predicted = pipeline.tracker().getEstimatedArmorPositions()[plate];
            const cv::Mat center_mat = pipeline.tracker().getTargetCenter();
            if (predicted.rows == 3 && !armor.tvec.empty() && center_mat.total() >= 3) {
                const cv::Vec3d plate_cam(predicted);
                const cv::Vec3d center_cam(center_mat.at<double>(0), center_mat.at<double>(1),
                                           center_mat.at<double>(2));
                cv::Vec3d radial = plate_cam - center_cam;
                const double rn = cv::norm(radial);
                if (rn > 1e-6) {
                    radial /= rn;
                    cv::Vec3d vertical =
                        pipeline.tracker().worldDirectionToCamera(cv::Vec3d(0.0, -1.0, 0.0));
                    const double vn = cv::norm(vertical);
                    if (vn > 1e-6) {
                        vertical /= vn;
                        cv::Vec3d tangent(radial[1] * vertical[2] - radial[2] * vertical[1],
                                          radial[2] * vertical[0] - radial[0] * vertical[2],
                                          radial[0] * vertical[1] - radial[1] * vertical[0]);
                        const double tn = cv::norm(tangent);
                        if (tn > 1e-6) {
                            tangent /= tn;
                            // 米 → 毫米：下面打印的口径是 mm，别把米当 mm 读。
                            const cv::Vec3d offset(
                                (cv::Vec3d(predicted) - cv::Vec3d(armor.tvec)) * 1000.0);
                            const double r = offset.dot(radial);
                            const double t = offset.dot(tangent);
                            const double u = offset.dot(vertical);
                            ++axis_frames;
                            sum_r += r; sum_t += t; sum_u += u;
                            sum_rr += r * r; sum_tt += t * t; sum_uu += u * u;
                            max_r = std::max(max_r, std::abs(r));
                            max_t = std::max(max_t, std::abs(t));
                            max_u = std::max(max_u, std::abs(u));
                        }
                    }
                }
            }
        }

        // 和真机同口径的重投影诊断：模型对"刚观测到的那块板"投影回来，离识别框多远
        double reproj_px = -1.0;
        {
            const auto state_now = pipeline.tracker().getState();
            if (state_now != auto_aim::TrackerState::LOST && pipeline.tracker().hasObservation()) {
                const int plate = pipeline.tracker().getLastObservedPlateId();
                const auto& armor = pipeline.tracker().getLastObservedArmor();
                const double half_w =
                    (armor.armor_type == auto_aim::large ? source_cfg.pnp.large_width
                                                        : source_cfg.pnp.small_width) * 0.5;
                const double half_h = source_cfg.pnp.height * 0.5;
                const auto positions = pipeline.tracker().getEstimatedArmorPositions();
                // 和 node.cpp 同一口径：世界系的切向/竖直先转到相机系再画（世界 y 向下，
                // "向上" = (0,-1,0)）。不做这一步就会把板掰一个≈相机-板 pitch 夹角。
                                const auto to_camera = [&](const cv::Vec3d& v) {
                    return pipeline.tracker().worldDirectionToCamera(v);
                };
                // 竖直用**世界系**竖直（和 node.cpp 同口径；观测板的 roll 不该传染给模型）
                const cv::Vec3d vertical_cam = to_camera(cv::Vec3d(0.0, 1.0, 0.0));
                // 和 node.cpp 同一套画法：切向 = 竖直 × 径向（都在相机系里用模型几何算，
                // 不用 getYaw()——那是相机系 yaw，拿去造世界系切向会把板画扁）
                const cv::Mat center_mat2 = pipeline.tracker().getTargetCenter();
                const cv::Vec3d center_cam2(center_mat2.at<double>(0), center_mat2.at<double>(1),
                                            center_mat2.at<double>(2));
                cv::Vec3d radial = cv::Vec3d(positions[plate]) - center_cam2;
                const double radial_norm = cv::norm(radial);
                if (radial_norm < 1e-6) return {};
                radial /= radial_norm;
                cv::Vec3d tangent_cam = vertical_cam.cross(radial);
                if (cv::norm(tangent_cam) < 1e-6) return {};
                tangent_cam /= cv::norm(tangent_cam);
                // 竖直用**世界系**竖直（和 node.cpp 同口径；观测板的 roll 不该传染给模型）
                const auto corners = auto_aim::projectArmorCameraFrame(
                    positions[plate], tangent_cam, vertical_cam, half_w, half_h, camera_matrix,
                    dist_coeffs);
                if (corners.size() == 4) {
                    // 顺便量"模型四角 vs 识别四角"的**尺寸比**：模型用的是配置里的板尺寸，
                    // 识别框是图像实测。比值明显≠1 就说明 configs/tracker.yaml 的
                    // small_width/large_width/height 与真板不符（PnP 距离会按同一比例偏）。
                    const double detected_w = cv::norm(armor.left.top - armor.right.top);
                    const double detected_h = cv::norm(armor.left.top - armor.left.bottom);
                    const double model_w = cv::norm(corners[1] - corners[2]);
                    const double model_h = cv::norm(corners[0] - corners[1]);
                    if (model_w > 1.0 && model_h > 1.0) {
                        ++size_frames;
                        sum_size_ratio_w += detected_w / model_w;
                        sum_size_ratio_h += detected_h / model_h;
                    }
                    // 角点顺序：模型 (左下,左上,右上,右下) ↔ 检测 (左上,右上,右下,左下)，循环移位配对
                    const std::array<cv::Point2f, 4> detected{armor.left.top, armor.right.top,
                                                              armor.right.bottom, armor.left.bottom};
                    // 取 4 个循环移位里最小的（形状吻合度；起点/绕向差异不该算成误差）
                    double best = 1e18;
                    for (int shift = 0; shift < 4; ++shift) {
                        double sum = 0.0;
                        for (int k = 0; k < 4; ++k) {
                            sum += cv::norm(corners[k] - detected[(k + shift) % 4]);
                        }
                        best = std::min(best, sum / 4.0);
                    }
                    reproj_px = best;
                    if (reproj_px >= 0.0) {
                        ++reproj_frames;
                        sum_reproj += reproj_px;
                        max_reproj = std::max(max_reproj, reproj_px);
                    }
                }
            }
        }

        detections += result.armors.empty() ? 0 : 1;
        for (const auto& armor : result.armors) {
            if (armor.solve_result) ++pnp_ok;
        }
        const int state = static_cast<int>(pipeline.tracker().getState());
        if (state == 2) ++tracking;
        if (outcome.fire) ++fire;
        {
            const auto wd_now = pipeline.tracker().watchdogStats();
            if (wd_now.reject_ratio > max_reject_ratio_seen) {
                max_reject_ratio_seen = wd_now.reject_ratio;
                max_reject_ratio_frame = i;
            }
        }
        // ---- 瞄点指令抖动（本次跑出来的决策 vs 录制时的决策）----
        {
            const int pid = outcome.decision.valid ? outcome.decision.armor_id : -1;
            if (pid >= 0) {
                ++plate_id_samples;
                if (prev_plate_id >= 0 && pid != prev_plate_id) ++plate_id_switches;
                prev_plate_id = pid;
            }
            if (outcome.decision.valid) {
                if (have_prev_aim) {
                    const double dt = row.t - prev_aim_t;
                    if (dt > 1e-6 && dt < 0.2) {
                        const double jump_yaw = std::abs(auto_aim::GimbalAimer::normalizeAngle(
                            outcome.decision.target_yaw - prev_aim_yaw)) * 180.0 / CV_PI;
                        const double jump_pitch =
                            std::abs(outcome.decision.target_pitch - prev_aim_pitch) * 180.0 / CV_PI;
                        aim_jump_yaw_deg.push_back(jump_yaw);
                        aim_jump_pitch_deg.push_back(jump_pitch);
                        aim_jump_yaw_rate.push_back(jump_yaw / dt);
                        aim_jump_pitch_rate.push_back(jump_pitch / dt);
                    }
                }
                prev_aim_yaw = outcome.decision.target_yaw;
                prev_aim_pitch = outcome.decision.target_pitch;
                prev_aim_t = row.t;
                have_prev_aim = true;
                aim_window.push_back({row.t, outcome.decision.target_yaw * 180.0 / CV_PI,
                                      outcome.decision.target_pitch * 180.0 / CV_PI});
                    // 1 秒滑窗峰值-峰值：用户眼睛看到的"幅度"
                while (!aim_window.empty() && row.t - aim_window.front().t > 1.0) {
                    aim_window.pop_front();
                }
                if (aim_window.size() >= 20) {
                    double min_y = 1e9, max_y = -1e9; double min_p = 1e9, max_p = -1e9;
                    for (const auto& w : aim_window) {
                        min_y = std::min(min_y, w.yaw); max_y = std::max(max_y, w.yaw);
                        min_p = std::min(min_p, w.pitch); max_p = std::max(max_p, w.pitch);
                    }
                    aim_ptp_yaw_deg.push_back(max_y - min_y);
                    aim_ptp_pitch_deg.push_back(max_p - min_p);
                }
            } else {
                have_prev_aim = false;
                aim_window.clear();
            }
            if (row.aim_yaw != 0.0) {
                if (have_prev_rec_aim) {
                    const double dt = row.t - prev_rec_aim_t;
                    if (dt > 1e-6 && dt < 0.2) {
                        rec_aim_jump_yaw_deg.push_back(std::abs(row.aim_yaw - prev_rec_aim_yaw) *
                                                       180.0 / CV_PI);
                        rec_aim_jump_pitch_deg.push_back(std::abs(row.aim_pitch - prev_rec_aim_pitch) *
                                                         180.0 / CV_PI);
                    }
                }
                prev_rec_aim_yaw = row.aim_yaw;
                prev_rec_aim_pitch = row.aim_pitch;
                prev_rec_aim_t = row.t;
                have_prev_rec_aim = true;
            } else {
                have_prev_rec_aim = false;
            }
        }
        if (dump.is_open()) {
            const auto& tracker = pipeline.tracker();
            const auto center = tracker.getTargetCenterWorldArray();
            const auto velocity = tracker.getTargetVelocityWorld();
            const auto obs = tracker.hasObservation()
                ? cv::Vec3d(tracker.getLastObservedArmor().tvec) : cv::Vec3d(0, 0, 0);
            const auto d = outcome.decision;
            dump << i << ',' << row.t << ',' << state << ','
                 << (tracker.hasObservation() ? tracker.getLastObservedPlateId() : -1) << ','
                 << (d.valid ? d.armor_id : -1) << ',' << (d.valid ? 1 : 0) << ','
                 << tracker.getYaw() * 180.0 / CV_PI << ',' << tracker.getOmega() * 180.0 / CV_PI
                 << ',' << center[0] * 1000.0 << ',' << center[1] * 1000.0 << ','
                 << center[2] * 1000.0 << ',' << velocity[0] * 1000.0 << ','
                 << velocity[1] * 1000.0 << ',' << velocity[2] * 1000.0 << ','
                 << (d.valid ? d.distance : 0.0) << ',' << (d.valid ? d.lead_time * 1000.0 : 0.0)
                 << ',' << (d.valid ? d.target_yaw * 180.0 / CV_PI : 0.0) << ','
                 << (d.valid ? d.target_pitch * 180.0 / CV_PI : 0.0) << ','
                 << (d.valid ? d.target_yaw_velocity * 180.0 / CV_PI : 0.0) << ','
                 << (d.valid ? d.target_pitch_velocity * 180.0 / CV_PI : 0.0) << ','
                 << obs[0] * 1000.0 << ',' << obs[1] * 1000.0 << ',' << obs[2] * 1000.0 << ','
                 << reproj_px << ',' << (pipeline.targetStill() ? 1 : 0) << ','
                 // 原始回传角（未做 yaw_sign 修正）与姿态角，用于量 pitch 的物理符号
                 << row.yaw * 180.0 / CV_PI << ',' << row.pitch * 180.0 / CV_PI << ','
                 << pose.yaw * 180.0 / CV_PI << ',' << pose.pitch * 180.0 / CV_PI << ','
                 // 最终下发的期望角（控制线程快照），验证"瞄点是否真落在观测板上"
                 << pipeline.controller().snapshot().desired_yaw * 180.0 / CV_PI << ','
                 << pipeline.controller().snapshot().desired_pitch * 180.0 / CV_PI << '\n';
        }
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
            if (reproj_px >= 0.0) {
                cv::putText(canvas, cv::format("reproj=%.0fpx", reproj_px), cv::Point(16, 80),
                            cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 0, 255), 2);
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
    {
        const auto wd = pipeline.tracker().watchdogStats();
        std::cout << "[replay] 收敛看门狗：触发重初始化 " << wd.resets << " 次，当前拒绝比例 "
                  << wd.reject_ratio << "（窗口 " << wd.samples << " 帧）；整段最大拒绝比例 "
                  << max_reject_ratio_seen << "（第 " << max_reject_ratio_frame << " 帧）"
                  << std::endl;
    }
    if (size_frames > 0) {
        std::cout << "[replay] 板尺寸核对（识别四角 / 模型四角）：宽比 平均 "
                  << sum_size_ratio_w / size_frames << "，高比 平均 " << sum_size_ratio_h / size_frames
                  << "（=1 才说明 configs/tracker.yaml 里的板尺寸和真板一致；比值 k 表示应把尺寸乘以 k）"
                  << std::endl;
    }
    if (axis_frames > 0) {
        const double n = axis_frames;
        auto std_of = [&n](double s2, double s) { return std::sqrt(std::max(0.0, s2 / n - (s / n) * (s / n))); };
        std::cout << "[replay] 偏移分解（模型-观测，相机系，mm）："
                  << "径向 均值 " << sum_r / n << " / 标准差 " << std_of(sum_rr, sum_r)
                  << " / 最大 " << max_r << "；切向 均值 " << sum_t / n << " / 标准差 "
                  << std_of(sum_tt, sum_t) << " / 最大 " << max_t << "；竖直 均值 " << sum_u / n
                  << " / 标准差 " << std_of(sum_uu, sum_u) << " / 最大 " << max_u << "（" << n
                  << " 帧；hard_plate_constraint 打开时该差值被硬约束压到毫米级，"
                  << "剩下的残差来自 refineArmorPose 与原始 PnP 的差异）" << std::endl;
    }
    if (update_frames > 0) {
        std::cout << "[replay] 观测吸收情况：" << update_frames << " 帧中 " << update_stuck_frames
                  << " 帧模型几乎没动（<1 mm，说明观测被门限拒/没吸收）；平均移动 "
                  << sum_update_move_mm / update_frames << " mm，平均更新后残差 "
                  << sum_update_residual_mm / update_frames << " mm" << std::endl;
    }
    if (dist_frames > 0) {
        std::cout << "[replay] 估计 vs 观测（同一块板，相机系 3D）：平均距差 "
                  << sum_estimate_distance_mm / dist_frames << " mm，最大 "
                  << max_estimate_distance_mm << " mm；平均方向差 "
                  << sum_estimate_bearing_deg / dist_frames << "°（" << dist_frames << " 帧）"
                  << std::endl;
    }
    if (reproj_frames > 0) {
        std::cout << "[replay] 模型对已观测板的重投影：平均 " << sum_reproj / reproj_frames
                  << " px，最大 " << max_reproj << " px（" << reproj_frames
                  << " 帧；>10 px 说明估计与观测不自洽）" << std::endl;
    }
    if (compare) {
        std::cout << "[replay] 与录制时输出对比：" << compared << " 帧，瞄点最大差 "
                  << max_aim_diff * 180.0 / CV_PI << "°（>1° 说明这次改动确实改变了行为）"
                  << std::endl;
    }
    {
        auto stats = [](std::vector<double>& v) {
            std::sort(v.begin(), v.end());
            auto at = [&v](double p) {
                return v.empty() ? 0.0 : v[std::min(v.size() - 1, static_cast<std::size_t>(p * v.size()))];
            };
            const double mean = v.empty() ? 0.0
                : std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());
            return std::array<double, 5>{{mean, at(0.5), at(0.9), at(0.99), v.empty() ? 0.0 : v.back()}};
        };
        const auto jy = stats(aim_jump_yaw_deg);
        const auto jp = stats(aim_jump_pitch_deg);
        const auto ry = stats(aim_jump_yaw_rate);
        const auto rp = stats(aim_jump_pitch_rate);
        std::cout << "[replay] 瞄点指令逐帧抖动（本次代码，连续两帧都有有效瞄点，"
                  << aim_jump_yaw_deg.size() << " 对）：yaw Δ 均值 " << jy[0] << " p50 " << jy[1]
                  << " p90 " << jy[2] << " p99 " << jy[3] << " max " << jy[4]
                  << "°；pitch Δ 均值 " << jp[0] << " p50 " << jp[1] << " p90 " << jp[2]
                  << " p99 " << jp[3] << " max " << jp[4] << "°" << std::endl;
        std::cout << "[replay] 等效角速度：yaw 均值 " << ry[0] << " p90 " << ry[2] << " p99 " << ry[3]
                  << " max " << ry[4] << "°/s；pitch 均值 " << rp[0] << " p90 " << rp[2]
                  << " p99 " << rp[3] << " max " << rp[4] << "°/s" << std::endl;
        const auto py = stats(aim_ptp_yaw_deg);
        const auto pp = stats(aim_ptp_pitch_deg);
        std::cout << "[replay] 瞄点 1 秒滑窗峰值-峰值（" << aim_ptp_yaw_deg.size()
                  << " 个窗，静止靶下应≈0）：yaw 均值 " << py[0] << " p90 " << py[2] << " max "
                  << py[4] << "°；pitch 均值 " << pp[0] << " p90 " << pp[2] << " max " << pp[4]
                  << "°" << std::endl;
        const auto qy = stats(rec_aim_jump_yaw_deg);
        const auto qp = stats(rec_aim_jump_pitch_deg);
        std::cout << "[replay] 对照：录制时那条瞄点指令的逐帧抖动 yaw 均值 " << qy[0] << " p99 "
                  << qy[3] << " max " << qy[4] << "°；pitch 均值 " << qp[0] << " p99 " << qp[3]
                  << " max " << qp[4] << "°" << std::endl;
        std::cout << "[replay] 板号（瞄的那块）切换 " << plate_id_switches << " 次 / "
                  << plate_id_samples << " 帧" << std::endl;
    }
    return 0;
}
