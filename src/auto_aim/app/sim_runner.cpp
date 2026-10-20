#include "app/sim_runner.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
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
#include "app/config_paths.hpp"
#include "app/csv_sink.hpp"
#include "app/pipeline_config.hpp"
#include "app/perf_stats.hpp"
#include "app/sim_detail.hpp"
#include "visualization/projection.hpp"
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

using namespace auto_aim::app::sim_detail;

namespace auto_aim
{
    namespace app
    {
        SimRunner::SimRunner(std::string config_dir)
            : config_dir_(std::move(config_dir))
        {
        }

        bool SimRunner::initialize()
        {
            const auto config_path = [this](const std::string& name) {
                return auto_aim::app::configPath(config_dir_, name);
            };

            const YAML::Node camera_cfg = YAML::LoadFile(config_path("camera.yaml"));
            const YAML::Node detector_cfg = YAML::LoadFile(config_path("detector.yaml"));
            const YAML::Node tracker_cfg = YAML::LoadFile(config_path("tracker.yaml"));

            if (camera_cfg["simulator"]) {
                use_simulator_camera_ = true;
                simulator_fov_degrees_ = camera_cfg["simulator"]["fov_degrees"].as<double>(45.0);
                max_processing_width_ = camera_cfg["simulator"]["max_processing_width"].as<int>(960);
            }

            const auto_aim::DetectorConfig det_cfg = auto_aim::loadDetectorConfig(detector_cfg);
            auto_aim::TrackerConfig track_cfg = auto_aim::loadTrackerConfig(tracker_cfg);
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
                    auto_aim::loadNeuralDetectorConfig(detector_cfg, config_dir_);
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
            fire_enabled_ = std::getenv("ULTRA_VISION_DISABLE_FIRE") == nullptr;
            show_display_ = std::getenv("ULTRA_VISION_NO_DISPLAY") == nullptr;
            // Restrict inference to the projected chassis region while tracking.
            // OFF by default: it was measured not to speed anything up, because the
            // detector letterboxes the crop back to the network's fixed 640x640 input.
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
            (void)shooter_min_fire_interval;
            const int max_control_lost_frames =
                auto_aim::loadMaxControlLostFrames(tracker_cfg);
            const double fire_angle_tolerance =
                auto_aim::loadFireAngleToleranceDegrees(tracker_cfg) * CV_PI / 180.0;

            receiver_ = std::make_unique<sim_receiver::VisionDateReceiver>(
                config_path("simulator.yaml"));
            const char* simulator_host = std::getenv("ULTRA_VISION_SIM_HOST");
            const char* simulator_port = std::getenv("ULTRA_VISION_SIM_PORT");
            if (simulator_host || simulator_port) {
                const std::string host = simulator_host ? simulator_host : "127.0.0.1";
                const uint16_t port = static_cast<uint16_t>(std::max(
                    1, std::min(65535, simulator_port ? std::atoi(simulator_port) : 7666)));
                receiver_->overrideEndpoint(host, port);
            }
            if (const char* command_port = std::getenv("ULTRA_VISION_SIM_COMMAND_PORT")) {
                receiver_->overrideCommandEndpoint(static_cast<uint16_t>(std::max(
                    1, std::min(65535, std::atoi(command_port)))));
            }
            receiver_->connect();
            // A previous run can leave the camera pointing anywhere, which makes the
            // next run see nothing and makes any A/B comparison meaningless. Ask for a
            // clean scene unless the caller wants to keep the manual state.
            if (const char* reset = std::getenv("ULTRA_VISION_SIM_RESET")) {
                if (std::string(reset) != "0") {
                    receiver_->sendResetCommand();
                    std::cout << "Requested a simulator RESET for a reproducible run"
                              << std::endl;
                }
            }
            receiver_->sendGimbalCommand(0.0, 0.0);

            // 真机与仿真共用同一套"检测前端 + 瞄准流水线"（见 armor_source / aim_pipeline）。
            // 这个入口只保留仿真专属的东西：TCP 帧源、曝光→到手的时间基、曲线/CSV/显示。
            // 仿真相机内参：分辨率固定时由 makeCameraMatrix() 按 FOV 重建，先占位声明。
            dist_coeffs_ = cv::Mat::zeros(5, 1, CV_64F);

            auto_aim::ArmorSourceConfig source_cfg;
            source_cfg.classical = det_cfg;
            source_cfg.pnp = pnp_geometry;
            source_cfg.dynamic_roi = dynamic_roi;
#ifdef ULTRA_VISION_USE_OPENVINO
            source_cfg.use_neural = use_neural_detector;
            source_cfg.neural = auto_aim::loadNeuralDetectorConfig(detector_cfg, config_dir_);
#endif
            armor_source_ = std::make_unique<auto_aim::ArmorSource>(
                source_cfg, camera_matrix_, dist_coeffs_);

            auto_aim::AimPipelineConfig pipeline_cfg = auto_aim::app::makeAimPipelineConfig(
                track_cfg, selector_cfg, gimbal_cfg, shooter_cfg, aim_filter_cfg,
                gimbal_command_rate_hz, max_control_lost_frames,
                fire_angle_tolerance * 180.0 / CV_PI,
                0.0);   // 仿真链路不需要心跳
            // 没合格板时的兜底策略：0=瞄最后观测到的板（现行为），1=放弃（保持上一拍
            // 瞄点，不开火）。见 AimPipelineConfig；YAML 给默认值，环境变量 A/B 覆盖。
            pipeline_cfg.no_valid_armor_policy =
                auto_aim::loadNoValidArmorPolicy(tracker_cfg);
            if (const char* policy_env = std::getenv("ULTRA_VISION_NO_VALID_ARMOR_POLICY")) {
                pipeline_cfg.no_valid_armor_policy = std::atoi(policy_env) != 0 ? 1 : 0;
            }
            pipeline_ = std::make_unique<auto_aim::AimPipeline>(
                pipeline_cfg, [this](bool /*control*/, bool /*fire*/, double yaw, double yaw_vel,
                                     double yaw_acc, double pitch, double pitch_vel,
                                     double pitch_acc) {
                    (void)yaw_vel; (void)yaw_acc; (void)pitch_vel; (void)pitch_acc;
                    // 诊断/离线录制用：冻结云台（不仿真器的云台发指令），相机保持不动。
                    if (std::getenv("ULTRA_VISION_DISABLE_CONTROL") != nullptr) {
                        return true;
                    }
                    return receiver_->sendGimbalCommand(yaw, pitch);
                });

            estimate_recorder_ = std::make_unique<auto_aim::app::CsvSink>(
                // 这是**重投影模型**（画面上那四个 E 框）的口径：输出状态按观测延迟外推
                // 到当前帧时刻后的四块板心（相机系）。model_dt/yaw/omega/radius 是为了
                // 让每行自描述 —— 想回放"不外推"的滤波器原始状态时用得到。
                "ULTRA_VISION_ESTIMATE_CSV",
                "x,y,z,color,distance,time_us,plate_id,model_dt,yaw,omega,radius\n");

            // 模型构造（重投影整车轮廓）用的装甲板安装倾角：仿真靶没有 15° 支架
            // （灯条在板坐标系里绝对竖直），所以默认 0。
            {
                const char* tilt_env = std::getenv("ULTRA_VISION_ARMOR_TILT_DEG");
                const double tilt = tilt_env != nullptr ? std::atof(tilt_env) : 0.0;
                pipeline_->tracker().setArmorTiltDeg(tilt);
            }

            observation_recorder_ = std::make_unique<auto_aim::app::CsvSink>(
                "ULTRA_VISION_OBSERVATION_CSV",
                "x,y,z,rx,ry,rz,tracker_x,tracker_y,tracker_z,"
                "cmd_yaw,cmd_pitch,time_us,plate_id,state,"
                // 检测到的四个角点像素（lb, lt, rt, rb）。为了能**离线**重解 PnP /
                // 重投影做标定（不必每次改尺寸都重跑仿真）。
                "u0,v0,u1,v1,u2,v2,u3,v3,source_time_us\n");
            selector_recorder_ = std::make_unique<auto_aim::app::CsvSink>(
                "ULTRA_VISION_SELECTOR_CSV",
                "time_us,center_x,center_y,center_z,yaw,omega,armor_id,"
                "armor_x,armor_y,armor_z,target_yaw,target_pitch,lead,"
                "target_yaw_velocity,target_pitch_velocity,gimbal_time,handoff,in_fire,"
                // 提前量用的**平移速度**状态（世界系）。实测自旋时这一项抖到 ±3 m/s
                // （真值 0），乘提前量就是瞄点甩飞的直接来源，所以单独落盘。
                "vx,vy,vz\n");
            detections_recorder_ = std::make_unique<auto_aim::app::CsvSink>(
                "ULTRA_VISION_DETECTIONS_CSV",
                // time_us = 本机到手时刻（主链路原来用的"观测时间"）；
                // source_time_us = 仿真端曝光戳。两者之差就是网络+编码延迟，
                // 用来离线判断"时间戳该用哪个"。
                "time_us,source_time_us,idx,u_lt,v_lt,u_lb,v_lb,u_rt,v_rt,u_rb,v_rb,x,y,z\n");

            // 反"云台跑飞"守卫：瞄点相对当前下发角超过这个角度就不甩（见下发处的注释）。
            // ULTRA_VISION_AIM_JUMP_LIMIT_DEG 可调，0 = 关。
            aim_jump_limit_ = [] {
                const char* value = std::getenv("ULTRA_VISION_AIM_JUMP_LIMIT_DEG");
                const double degrees = value != nullptr ? std::atof(value) : 30.0;
                return degrees > 0.0 ? degrees * CV_PI / 180.0 : 1e9;
            }();
            // 曝光→收到的延迟：每秒取中位，喂给选择器当"观测年龄"的固定附加项。
            // ULTRA_VISION_SOURCE_DELAY=0 可关掉（A/B 用）。
            source_delay_enabled_ = [] {
                const char* value = std::getenv("ULTRA_VISION_SOURCE_DELAY");
                return value == nullptr || std::string(value) != "0";
            }();
            prof_stages_ = std::getenv("ULTRA_VISION_PROF") != nullptr;
            // 定时退出（ULTRA_VISION_TEST_SECONDS）：仿真回归脚本要"跑 N 秒然后干净退出"，
            // 这样 CSV 的 ofstream 正常析构、最后几行不会因为 SIGTERM 丢掉。
            const char* test_seconds_env = std::getenv("ULTRA_VISION_TEST_SECONDS");
            test_seconds_ = test_seconds_env != nullptr ? std::atof(test_seconds_env) : 0.0;
            loop_start_time_ = std::chrono::steady_clock::now();
            fps_start_ = std::chrono::steady_clock::now();

            return true;
        }

        bool SimRunner::fetchFrame(std::chrono::steady_clock::time_point loop_start)
        {
            frame_sequence_ = 0;
            frame_local_timestamp_us_ = 0;
            // 仿真器给的**曝光时刻**（sim 时钟）。与 local 时间戳相差"网络+处理+节流"
            // 的整条延迟（实测 ~57 ms），用真值做回归时必须按它对齐，否则那点延迟在
            // 云台转动时会以"相机系位移"的形式混进观测误差里。
            frame_source_timestamp_us_ = 0;
            frame_gimbal_yaw_ = 0.0f;
            frame_gimbal_pitch_ = 0.0f;
            frame_ = receiver_->getFrame(&frame_sequence_, &frame_source_timestamp_us_,
                                         &frame_local_timestamp_us_, &frame_gimbal_yaw_,
                                         &frame_gimbal_pitch_);
            after_frame_ = std::chrono::steady_clock::now();
            if (frame_.empty()) {
                ++stats_.missing_frames;
                ++stats_.net_empty;
                if (stats_.missing_frames == 1 || stats_.missing_frames % 30 == 0) {
                    std::cerr << "Waiting for simulator frames..." << std::endl;
                }
                key_ = cv::waitKey(1);
                return false;
            }
            stats_.missing_frames = 0;
            // 曝光→到手 的延迟采样（两边都是 Unix 时钟；Mac/NUC 有 NTP，
            // 常数级的钟差不影响"这段延迟有多大"的结论，只把绝对值平移）。
            // 这一项就是"观测年龄"里**原来被漏掉**的一段：检测用的是仿真端
            // 曝光时刻的像，而我们原来只用"本机收到的时刻"当观测时间。
            if (frame_source_timestamp_us_ > 0 &&
                frame_local_timestamp_us_ > frame_source_timestamp_us_) {
                stats_.source_delay_samples_us.push_back(
                    static_cast<double>(frame_local_timestamp_us_ - frame_source_timestamp_us_));
            }
            // 进帧速率统计（本机时钟，跨机器时间戳不可比）
            ++stats_.net_frames;
            stats_.net_recv_ms += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - loop_start).count();
            if (stats_.net_first_local_us == 0) {
                stats_.net_first_local_us = frame_local_timestamp_us_;
            } else {
                stats_.net_prev_local_us = frame_local_timestamp_us_;
            }
            return true;
        }

        void SimRunner::prepareFrame()
        {
            // All estimator, control, and command-history timestamps use the same
            // local monotonic clock. The simulator source timestamp is metadata
            // only and is never compared with local wall-clock time.
            last_gimbal_command_ = pipeline_->controller().snapshotAt(frame_local_timestamp_us_);
            if (last_gimbal_command_.valid) {
                commanded_yaw_ = last_gimbal_command_.yaw;
                commanded_pitch_ = last_gimbal_command_.pitch;
            }

            if (max_processing_width_ > 0 && frame_.cols > max_processing_width_) {
                const double scale = static_cast<double>(max_processing_width_) / frame_.cols;
                cv::resize(frame_, image_, cv::Size(), scale, scale, cv::INTER_AREA);
            } else {
                image_ = frame_;
            }

            auto_aim::Tracker& tracker = pipeline_->tracker();
            if (use_simulator_camera_ && image_.rows != processing_height_) {
                camera_matrix_ = makeCameraMatrix(image_.rows, image_.cols, simulator_fov_degrees_);
                processing_height_ = image_.rows;
                // 进图尺寸一变就把"实际处理尺寸 + 现算的内参"打出来：仿真器给 640x480
                // 还是 1440x1080，PnP 的尺度和检出率完全是两回事，必须看一眼就知道。
                // 期望值来自 configs/camera.yaml 的 simulator.max_processing_width，
                // 它和实车海康的 1440x1080 / 41°(f≈1447) 是同一套标定。
                const bool size_mismatch =
                    max_processing_width_ > 0 && image_.cols < max_processing_width_;
                std::cout << "[input] " << image_.cols << "x" << image_.rows
                          << " fov=" << simulator_fov_degrees_
                          << " f=" << camera_matrix_.at<double>(0, 0)
                          << " cx=" << camera_matrix_.at<double>(0, 2)
                          << " cy=" << camera_matrix_.at<double>(1, 2)
                          << " (标定尺寸 " << max_processing_width_ << " 宽)"
                          << (size_mismatch ? "  ← 偏小！" : "") << std::endl;
                if (size_mismatch) {
                    std::cerr
                        << "[input] 警告：仿真帧只有 " << image_.cols << "x" << image_.rows
                        << "，比 configs/camera.yaml 标定的 "
                        << max_processing_width_
                        << " 宽小 —— 靶面像素按面积缩水 "
                        << (static_cast<double>(max_processing_width_) / image_.cols) *
                               (static_cast<double>(max_processing_width_) / image_.cols)
                        << " 倍，检测率/PnP 尺度都会和标定对不上。\n"
                        << "[input]       仿真器要用匹配的采集尺寸启动：\n"
                        << "[input]         simulator_system/simulator/run_host.sh"
                           "（默认已是 1440x1080）\n"
                        << "[input]        或 DAEDALUS_CAPTURE_WIDTH=1440"
                           " DAEDALUS_CAPTURE_HEIGHT=1080，"
                           "或按 1440x1080 采集后再喂进来"
                        << std::endl;
                }
            }
            if (camera_matrix_.rows == 3 && camera_matrix_.cols == 3) {
                tracker.setCameraIntrinsics(camera_matrix_.at<double>(0, 0),
                                            camera_matrix_.at<double>(1, 1),
                                            camera_matrix_.at<double>(0, 2),
                                            camera_matrix_.at<double>(1, 2));
            }

            // ------------------- 检测（共用组件：传统/神经网络 + 动态 ROI + PnP）----
            armor_source_->setIntrinsics(camera_matrix_, dist_coeffs_);
            source_result_ = armor_source_->update(
                image_, frame_sequence_, frame_local_timestamp_us_,
                tracker.getEstimatedArmorPositions(), tracker.getState(),
                /*image_will_be_modified=*/show_display_);
            // Sizes captured before the armors are moved into the pipeline below;
            // the per-second stats line must not read a moved-from vector.
            light_count_ = static_cast<int>(source_result_.lights.size());
            armor_count_ = static_cast<int>(source_result_.armors.size());
            pnp_count_ = source_result_.pnp_count;

            // 可选：把本帧**全部**检测（每块板的左右两根灯条端点 + PnP 位置）落盘。
            if (*detections_recorder_) {
                detections_recorder_->stream() << std::fixed << std::setprecision(4);
                int idx = 0;
                for (const auto& a : source_result_.armors) {
                    const bool has_tvec = a.solve_result && a.tvec.rows == 3;
                    detections_recorder_->stream()
                        << source_result_.timestamp_us << ','
                        << frame_source_timestamp_us_ << ',' << idx++ << ','
                        << a.left.top.x << ',' << a.left.top.y << ','
                        << a.left.bottom.x << ',' << a.left.bottom.y << ','
                        << a.right.top.x << ',' << a.right.top.y << ','
                        << a.right.bottom.x << ',' << a.right.bottom.y << ','
                        << (has_tvec ? a.tvec.at<double>(0) : std::nan("")) << ','
                        << (has_tvec ? a.tvec.at<double>(1) : std::nan("")) << ','
                        << (has_tvec ? a.tvec.at<double>(2) : std::nan("")) << '\n';
                }
            }

            // 可选：把带检测框的原图落盘，供离线核对"检测角点 vs 真值投影"。
            // ULTRA_VISION_DUMP_DIR=<dir> 时每 25 帧存一张（最多 24 张）。
            if (const char* dump_dir = std::getenv("ULTRA_VISION_DUMP_DIR")) {
                if (dump_counter_ % 25 == 0 && dump_counter_ < 600) {
                    cv::Mat dbg = image_.clone();
                    char name[512];
                    std::snprintf(name, sizeof(name), "%s/frame_%04d_%lld.png", dump_dir,
                                  dump_counter_,
                                  static_cast<long long>(source_result_.timestamp_us));
                    for (const auto& a : source_result_.armors) {
                        const cv::Point2f pts[4] = {a.left.top, a.left.bottom, a.right.top,
                                                    a.right.bottom};
                        for (int i = 0; i < 4; ++i)
                            cv::circle(dbg, pts[i], 4, cv::Scalar(0, 255, 0), 2);
                        cv::line(dbg, a.left.top, a.left.bottom, cv::Scalar(0, 255, 0), 1);
                        cv::line(dbg, a.right.top, a.right.bottom, cv::Scalar(0, 128, 255), 1);
                    }
                    cv::putText(dbg, "t_us=" + std::to_string(source_result_.timestamp_us),
                                {10, 30},
                                cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);
                    cv::imwrite(name, dbg);
                }
                ++dump_counter_;
            }
        }

        void SimRunner::recordDetections()
        {
            auto_aim::Tracker& tracker = pipeline_->tracker();

            if (*observation_recorder_ && tracker.hasObservation() &&
                !tracker.getLastObservedArmor().tvec.empty()) {
                const auto& observed = tracker.getLastObservedArmor();
                const cv::Mat center = tracker.getTargetCenter();
                observation_recorder_->stream() << std::fixed << std::setprecision(6)
                    << observed.tvec.at<double>(0) << ','
                    << observed.tvec.at<double>(1) << ','
                    << observed.tvec.at<double>(2) << ','
                    << observed.rvec.at<double>(0) << ','
                    << observed.rvec.at<double>(1) << ','
                    << observed.rvec.at<double>(2) << ','
                    << center.at<double>(0) << ','
                    << center.at<double>(1) << ','
                    << center.at<double>(2) << ','
                    << commanded_yaw_ << ',' << commanded_pitch_ << ','
                    << frame_local_timestamp_us_ << ','
                    << tracker.getLastObservedPlateId() << ','
                    << static_cast<int>(tracker.getState());
                if (observed.Points_2D.size() >= 4) {
                    for (int i = 0; i < 4; ++i) {
                        observation_recorder_->stream() << ','
                            << observed.Points_2D[static_cast<std::size_t>(i)].x
                            << ',' << observed.Points_2D[static_cast<std::size_t>(i)].y;
                    }
                } else {
                    for (int i = 0; i < 8; ++i) observation_recorder_->stream() << ",nan";
                }
                observation_recorder_->stream() << ',' << frame_source_timestamp_us_ << '\n';
            }

            if (*estimate_recorder_ &&
                tracker.getState() != auto_aim::TrackerState::LOST &&
                tracker.getState() != auto_aim::TrackerState::DETECTING) {
                // 落盘的是**重投影模型**（= 画面上看到的整车轮廓），所以要和画图同口径：
                // 输出口径的状态 + 按观测延迟外推到当前帧时刻。这样"estimate.csv vs truth"
                // 量出来的就是"操作员看到的模型离真车多远"，而不是"滤波器状态停留在哪个
                // 过去的时刻"—— 后者会把几十~一百多 ms 的观测延迟算成模型误差。
                // 外推到**被显示那一帧的曝光时刻**（不是"到手时刻"）：dt 与渲染姿态
                // 都必须对应那张图，否则云台/目标在那段延迟里的运动会被算成模型误差。
                const double model_dt = std::max(
                    0.0, std::min(auto_aim::StandardClock::secondsFromUs(
                                      frame_source_timestamp_us_) -
                                      tracker.lastObservationTime(), 0.20));
                auto_aim::CameraPose frame_pose;
                frame_pose.valid = true;
                frame_pose.yaw = frame_gimbal_yaw_;
                frame_pose.pitch = frame_gimbal_pitch_;
                const auto model = tracker.modelAt(model_dt, frame_pose);
                if (model.valid) {
                    estimate_recorder_->stream() << std::fixed << std::setprecision(6);
                    for (int plate = 0; plate < 4; ++plate) {
                        const auto& pose = model.poses[plate];
                        const double x = pose.center[0];
                        const double y = pose.center[1];
                        const double z = pose.center[2];
                        estimate_recorder_->stream() << x << ',' << y << ',' << z << ",0,"
                                          << std::sqrt(x * x + y * y + z * z) << ','
                                          << frame_local_timestamp_us_ << ',' << plate
                                          // 自描述：外推用的 dt、当时的 yaw/ω、模型半径
                                          << ',' << model_dt << ',' << model.world_yaw << ','
                                          << model.omega << ',' << model.geometry.meanDistance()
                                          << '\n';
                    }
                }
            }
        }

        void SimRunner::runAim()
        {
            auto_aim::Tracker& tracker = pipeline_->tracker();

            // 观测时刻：仿真里检测用的是**曝光那一刻**的像，所以要把
            // "曝光→到手"的延迟从收到帧的时刻里减掉（真机串口链路是同步的，直接用帧时刻）。
            const bool source_delay_usable = source_delay_enabled_ &&
                stats_.observation_source_delay > 0.0005 && stats_.observation_source_delay < 0.120;
            const uint64_t source_delay_us = source_delay_usable
                ? static_cast<uint64_t>(stats_.observation_source_delay * 1e6) : 0;
            const double observation_time =
                auto_aim::StandardClock::secondsFromUs(source_result_.timestamp_us) -
                source_delay_us / 1e6;

            auto_aim::CameraPose detection_pose;
            // 用仿真器回传的**真实云台姿态**（执行之后），等价实车的 IMU 反馈。
            // 以前用"指令快照"（gimbal_controller.snapshotAt）——只有仿真云台瞬时
            // 到位（Direct）时才成立；现在仿真云台是限速跟随的，必须用真实姿态，
            // 闭环才和实车同构（否则会以为相机在指令角、实际在别处）。
            detection_pose.valid = true;
            detection_pose.timestamp_valid = true;
            detection_pose.timestamp = observation_time;
            detection_pose.yaw = static_cast<double>(frame_gimbal_yaw_);
            detection_pose.pitch = static_cast<double>(frame_gimbal_pitch_);

            auto_aim::AimPipeline::FrameInput frame;
            frame.fresh = source_result_.fresh;
            frame.observation_time = observation_time;
            frame.now = auto_aim::StandardClock::secondsFromUs(frame_local_timestamp_us_);
            frame.pose = detection_pose;
            frame.allow_control = true;         // 仿真里没有下位机，始终由我们控制
            frame.fire_enabled = fire_enabled_;
            // 关联层统计：从 tracker 读（组件内部统计，见 assoc: 日志）
            if (source_result_.fresh) {
                stats_.assoc_detections += armor_count_;
                stats_.assoc_nomatch += tracker.getDroppedNoMatch();
                stats_.assoc_ambiguous += tracker.getDroppedAmbiguous();
                stats_.assoc_scale += tracker.getDroppedByScale();
                stats_.assoc_accepted += tracker.getAcceptedObservations();
            }
            // 移动而不是拷贝：检出结果只进这一条流水线，搬走 vector 就够
            //（每帧省掉一次 Armor 列表（含嵌套 vector/cv::Mat 头）的深拷贝）。
            frame.armors = std::move(source_result_.armors);
            const auto outcome = pipeline_->update(frame);

            last_decision_ = outcome.decision;
            aim_ready_ = outcome.aim_ready;
            aim_yaw_error_ = outcome.aim_yaw_error;
            aim_pitch_error_ = outcome.aim_pitch_error;
            if (outcome.decision.valid) ++stats_.aim_valid_updates;
            if (outcome.no_valid_plate) ++stats_.no_valid_plate_updates;
            if (outcome.gave_up) ++stats_.give_up_updates;
            if (outcome.aim_ready) ++stats_.aim_settled_updates;
            if (outcome.held_for_jump) ++aim_jump_rejections_;
            if (outcome.fired_now) {
                receiver_->sendFireCommand();
                std::cout << cv::format(
                    "FIRE state=FIRING omega=%+.2f armor=%d lead=%.3fs "
                    "yaw_err=%+.1fdeg pitch_err=%+.1fdeg",
                    tracker.getOmega(), outcome.decision.armor_id, outcome.decision.lead_time,
                    outcome.aim_yaw_error * 180.0 / CV_PI,
                    outcome.aim_pitch_error * 180.0 / CV_PI)
                          << std::endl;
            }
            if (outcome.decision.valid && previous_armor_id_ >= 0 &&
                outcome.decision.armor_id != previous_armor_id_) {
                ++stats_.plate_switches;
            }
            previous_armor_id_ = outcome.decision.valid ? outcome.decision.armor_id : -1;
            shooter_state_ = outcome.fire ? auto_aim::ShooterState::FIRING
                                          : auto_aim::ShooterState::IDLE;
            shooter_error_ = auto_aim::ShooterErrorReason::NONE;
            last_detection_time_ = observation_time;

            // selector.csv：保留原口径（时间戳同样是**观测时刻**）
            if (*selector_recorder_ && outcome.decision.valid) {
                selector_recorder_->stream() << std::fixed << std::setprecision(6)
                    << frame_local_timestamp_us_ << ','
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
                    << static_cast<int>(outcome.decision.in_fire_window) << ','
                    << tracker.getTargetVelocityWorld()[0] << ','
                    << tracker.getTargetVelocityWorld()[1] << ','
                    << tracker.getTargetVelocityWorld()[2] << '\n';
            }
        }

        std::string SimRunner::shooterStatus() const
        {
            return shooterStatusName(shooter_state_, shooter_error_);
        }

        bool SimRunner::testSecondsExpired() const
        {
            if (test_seconds_ <= 0.0) return false;
            return std::chrono::duration<double>(
                       std::chrono::steady_clock::now() - loop_start_time_).count() >= test_seconds_;
        }

        bool SimRunner::exitRequested() const
        {
            return key_ == 27 || key_ == 'e' || key_ == 'E';
        }
    } // namespace app
} // namespace auto_aim
