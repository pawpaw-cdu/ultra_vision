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
#include "app/pipeline_config.hpp"
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

            // 模型构造（重投影整车轮廓）用的装甲板安装倾角：仿真靶没有 15° 支架
            // （灯条在板坐标系里绝对竖直），所以默认 0。
            {
                const char* tilt_env = std::getenv("ULTRA_VISION_ARMOR_TILT_DEG");
                const double tilt = tilt_env != nullptr ? std::atof(tilt_env) : 0.0;
                pipeline_->tracker().setArmorTiltDeg(tilt);
            }

            // 曝光→收到的延迟：每秒取中位，喂给选择器当"观测年龄"的固定附加项。
            // ULTRA_VISION_SOURCE_DELAY=0 可关掉（A/B 用）。
            source_delay_enabled_ = [] {
                const char* value = std::getenv("ULTRA_VISION_SOURCE_DELAY");
                return value == nullptr || std::string(value) != "0";
            }();
            // 定时退出（ULTRA_VISION_TEST_SECONDS）：仿真回归脚本要"跑 N 秒然后干净退出"，
            // 这样 CSV 的 ofstream 正常析构、最后几行不会因为 SIGTERM 丢掉。
            const char* test_seconds_env = std::getenv("ULTRA_VISION_TEST_SECONDS");
            test_seconds_ = test_seconds_env != nullptr ? std::atof(test_seconds_env) : 0.0;
            loop_start_time_ = std::chrono::steady_clock::now();
            return true;
        }

        bool SimRunner::fetchFrame()
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
            if (frame_.empty()) {
                ++missing_frames_;
                if (missing_frames_ == 1 || missing_frames_ % 30 == 0) {
                    std::cerr << "Waiting for simulator frames..." << std::endl;
                }
                key_ = cv::waitKey(1);
                return false;
            }
            missing_frames_ = 0;
            // 曝光→到手 的延迟采样（两边都是 Unix 时钟；Mac/NUC 有 NTP，
            // 常数级的钟差不影响"这段延迟有多大"的结论，只把绝对值平移）。
            // 这一项就是"观测年龄"里**原来被漏掉**的一段：检测用的是仿真端
            // 曝光时刻的像，而我们原来只用"本机收到的时刻"当观测时间。
            if (frame_source_timestamp_us_ > 0 &&
                frame_local_timestamp_us_ > frame_source_timestamp_us_) {
                source_delay_samples_us_.push_back(
                    static_cast<double>(frame_local_timestamp_us_ - frame_source_timestamp_us_));
                if (source_delay_samples_us_.size() >= 5) {
                    std::vector<double> sorted = source_delay_samples_us_;
                    std::sort(sorted.begin(), sorted.end());
                    observation_source_delay_ = sorted[sorted.size() / 2] / 1e6;
                }
                if (source_delay_samples_us_.size() > 60) {
                    source_delay_samples_us_.erase(source_delay_samples_us_.begin(),
                                                   source_delay_samples_us_.begin() + 10);
                }
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

        }

        void SimRunner::runAim()
        {
            auto_aim::Tracker& tracker = pipeline_->tracker();

            // 观测时刻：仿真里检测用的是**曝光那一刻**的像，所以要把
            // "曝光→到手"的延迟从收到帧的时刻里减掉（真机串口链路是同步的，直接用帧时刻）。
            const bool source_delay_usable = source_delay_enabled_ &&
                observation_source_delay_ > 0.0005 && observation_source_delay_ < 0.120;
            const uint64_t source_delay_us = source_delay_usable
                ? static_cast<uint64_t>(observation_source_delay_ * 1e6) : 0;
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
            // 移动而不是拷贝：检出结果只进这一条流水线，搬走 vector 就够
            //（每帧省掉一次 Armor 列表（含嵌套 vector/cv::Mat 头）的深拷贝）。
            frame.armors = std::move(source_result_.armors);
            const auto outcome = pipeline_->update(frame);

            last_decision_ = outcome.decision;
            aim_ready_ = outcome.aim_ready;
            aim_yaw_error_ = outcome.aim_yaw_error;
            aim_pitch_error_ = outcome.aim_pitch_error;
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
            previous_armor_id_ = outcome.decision.valid ? outcome.decision.armor_id : -1;
            shooter_state_ = outcome.fire ? auto_aim::ShooterState::FIRING
                                          : auto_aim::ShooterState::IDLE;
            shooter_error_ = auto_aim::ShooterErrorReason::NONE;
            last_detection_time_ = observation_time;

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
