#ifndef ULTRA_VISION_AUTO_AIM_CONFIG_LOADER_HPP
#define ULTRA_VISION_AUTO_AIM_CONFIG_LOADER_HPP

// Shared YAML loaders for the two auto_aim entry points.
//
// The hardware entry (node.cpp) and the simulator entry (node_sim.cpp) used to
// parse the same detector/tracker sections independently, which is how the two
// pipelines drifted apart: the simulator grew the whole-chassis estimator while
// the hardware entry kept its own copy of the older settings. Both entries now
// read their configuration through these helpers so a change reaches both.

#include <algorithm>
#include <array>
#include <cstddef>
#include <iostream>
#include <vector>

#include <opencv2/core.hpp>
#include <yaml-cpp/yaml.h>

#include "Kalman/tracker.hpp"
#include "common/types.hpp"
#include "control/aim_signal_filter.hpp"
#include "control/gimbal_aimer.hpp"
#include "control/shooter.hpp"
#include "control/target_selector.hpp"
#include "perception/detector.hpp"
#include "perception/pnp_solver.hpp"
#include "io/gimbal/gimbal.hpp"

#ifdef ULTRA_VISION_USE_OPENVINO
#include <filesystem>

#include "perception/nn_detector.hpp"
#endif

namespace auto_aim
{
    // A matrix stored in YAML as an explicit rows/cols/data triplet.
    inline cv::Mat readMatFromYaml(const YAML::Node& node)
    {
        const int rows = node["rows"].as<int>();
        const int cols = node["cols"].as<int>();
        const std::vector<double> data = node["data"].as<std::vector<double>>();
        cv::Mat matrix(rows, cols, CV_64F);
        for (int row = 0; row < rows; ++row) {
            for (int column = 0; column < cols; ++column) {
                matrix.at<double>(row, column) =
                    data[static_cast<std::size_t>(row) * cols + column];
            }
        }
        return matrix;
    }

    /// @brief 相机→云台外参（手眼标定结果，见 tools/hand_eye_calibrate）。
    ///        没写就是单位阵 + 零平移（仿真口径，等价于"相机在云台光心"）。
    struct CameraExtrinsics
    {
        std::array<std::array<double, 3>, 3> rotation{{
            {{1.0, 0.0, 0.0}},
            {{0.0, 1.0, 0.0}},
            {{0.0, 0.0, 1.0}}
        }};
        std::array<double, 3> translation{{0.0, 0.0, 0.0}};
    };

    // `camera_file` 是 camera.yaml 根节点，`section` 是相机段名（hikcamera/galaxy）。
    inline CameraExtrinsics loadCameraExtrinsics(const YAML::Node& camera_file,
                                                 const std::string& section)
    {
        CameraExtrinsics extrinsics;
        const YAML::Node node = camera_file[section];
        if (!node) return extrinsics;
        if (const YAML::Node rotation = node["R_camera2gimbal"]) {
            const std::vector<double> values = rotation.as<std::vector<double>>();
            if (values.size() == 9) {
                for (int row = 0; row < 3; ++row) {
                    for (int column = 0; column < 3; ++column) {
                        extrinsics.rotation[row][column] =
                            values[static_cast<std::size_t>(row) * 3 + column];
                    }
                }
            } else {
                std::cerr << "R_camera2gimbal 需要 9 个数，忽略" << std::endl;
            }
        }
        if (const YAML::Node translation = node["t_camera2gimbal"]) {
            const std::vector<double> values = translation.as<std::vector<double>>();
            if (values.size() == 3) {
                std::copy(values.begin(), values.end(), extrinsics.translation.begin());
            } else {
                std::cerr << "t_camera2gimbal 需要 3 个数，忽略" << std::endl;
            }
        }
        return extrinsics;
    }

    // `detector_file` is the parsed detector.yaml root node.
    inline DetectorConfig loadDetectorConfig(const YAML::Node& detector_file)
    {
        const YAML::Node detector = detector_file["detector"];
        DetectorConfig cfg{};

        cfg.enemy_color = detector["enemy_color"].as<int>(cfg.enemy_color);
        cfg.binary_threshold =
            detector["binary_threshold"].as<int>(cfg.binary_threshold);

        // 下面每一段都可能整个不写（默认值已经在结构体里），
        // yaml-cpp 对空节点取下标会抛 InvalidNode，所以必须先判存在。
        if (const YAML::Node light = detector["light"]) {
            cfg.light_min_ratio = light["min_ratio"].as<double>(cfg.light_min_ratio);
            cfg.light_max_ratio = light["max_ratio"].as<double>(cfg.light_max_ratio);
            cfg.light_max_angle = light["max_angle"].as<double>(cfg.light_max_angle);
            cfg.light_min_contour_points =
                light["min_contour_points"].as<int>(cfg.light_min_contour_points);
        }

        if (const YAML::Node armor = detector["armor"]) {
            cfg.armor_height_ratio_min =
                armor["height_ratio_min"].as<double>(cfg.armor_height_ratio_min);
            cfg.armor_height_ratio_max =
                armor["height_ratio_max"].as<double>(cfg.armor_height_ratio_max);
            cfg.armor_angle_diff_max =
                armor["angle_diff_max"].as<double>(cfg.armor_angle_diff_max);
            cfg.armor_width_to_height_min =
                armor["width_to_height_min"].as<double>(cfg.armor_width_to_height_min);
            cfg.armor_width_to_height_max =
                armor["width_to_height_max"].as<double>(cfg.armor_width_to_height_max);
            cfg.armor_large_ratio_thresh =
                armor["large_armor_ratio_thresh"].as<double>(
                    cfg.armor_large_ratio_thresh);
        }

        if (const YAML::Node color = detector["color"]) {
            cfg.color_use_detect =
                color["use_color_detect"].as<bool>(cfg.color_use_detect);
            cfg.use_hsv = color["method"].as<std::string>(
                cfg.use_hsv ? "hsv" : "bgr") == "hsv";
            if (cfg.use_hsv) {
                if (const YAML::Node hsv = color["hsv"]) {
                    cfg.hue_red_low1 = hsv["red_low1"].as<int>(cfg.hue_red_low1);
                    cfg.hue_red_high1 = hsv["red_high1"].as<int>(cfg.hue_red_high1);
                    cfg.hue_red_low2 = hsv["red_low2"].as<int>(cfg.hue_red_low2);
                    cfg.hue_red_high2 = hsv["red_high2"].as<int>(cfg.hue_red_high2);
                    cfg.hue_blue_low = hsv["blue_low"].as<int>(cfg.hue_blue_low);
                    cfg.hue_blue_high = hsv["blue_high"].as<int>(cfg.hue_blue_high);
                    cfg.sat_min = hsv["sat_min"].as<int>(cfg.sat_min);
                    cfg.val_min = hsv["val_min"].as<int>(cfg.val_min);
                    cfg.color_area_ratio =
                        hsv["area_ratio"].as<double>(cfg.color_area_ratio);
                }
            } else if (const YAML::Node bgr = color["bgr"]) {
                cfg.color_red_threshold =
                    bgr["red_threshold"].as<double>(cfg.color_red_threshold);
                cfg.color_blue_threshold =
                    bgr["blue_threshold"].as<double>(cfg.color_blue_threshold);
            }
        }
        return cfg;
    }

    // `tracker_file` is the parsed tracker.yaml root node.
    inline TrackerConfig loadTrackerConfig(const YAML::Node& tracker_file)
    {
        const YAML::Node tracker = tracker_file["tracker"];
        TrackerConfig cfg;

        cfg.min_detect_frames =
            tracker["min_detect_frames"].as<int>(cfg.min_detect_frames);
        cfg.max_lost_frames =
            tracker["max_lost_frames"].as<int>(cfg.max_lost_frames);
        cfg.reacquire_frames =
            tracker["reacquire_frames"].as<int>(cfg.reacquire_frames);
        cfg.max_match_distance =
            tracker["max_match_distance"].as<double>(cfg.max_match_distance);
        cfg.reacquire_center_error =
            tracker["reacquire_center_error"].as<double>(cfg.reacquire_center_error);
        cfg.reacquire_max_dt =
            tracker["reacquire_max_dt"].as<double>(cfg.reacquire_max_dt);
        cfg.max_camera_pose_dt =
            tracker["max_camera_pose_dt"].as<double>(cfg.max_camera_pose_dt);

        if (const YAML::Node rotation = tracker["rotation_rate"]) {
            cfg.rotation_rate.window =
                rotation["window"].as<double>(cfg.rotation_rate.window);
            cfg.rotation_rate.min_span =
                rotation["min_span"].as<double>(cfg.rotation_rate.min_span);
            cfg.rotation_rate.max_sample_gap =
                rotation["max_sample_gap"].as<double>(
                    cfg.rotation_rate.max_sample_gap);
            cfg.rotation_rate.max_rate =
                rotation["max_rate"].as<double>(cfg.rotation_rate.max_rate);
            cfg.rotation_rate.min_rate =
                rotation["min_rate"].as<double>(cfg.rotation_rate.min_rate);
            cfg.rotation_rate.max_residual =
                rotation["max_residual"].as<double>(
                    cfg.rotation_rate.max_residual);
            cfg.rotation_rate.min_slopes =
                rotation["min_slopes"].as<int>(cfg.rotation_rate.min_slopes);
            cfg.omega_measure_noise =
                rotation["measure_noise"].as<double>(cfg.omega_measure_noise);
            cfg.omega_nis_threshold =
                rotation["nis_threshold"].as<double>(cfg.omega_nis_threshold);
        }

        if (const YAML::Node kalman = tracker["kalman"]) {
            cfg.process_noise_pos =
                kalman["process_noise_pos"].as<double>(cfg.process_noise_pos);
            cfg.process_noise_pos_speed_ref =
                kalman["process_noise_pos_speed_ref"].as<double>(
                    cfg.process_noise_pos_speed_ref);
            cfg.process_noise_pos_max =
                kalman["process_noise_pos_max"].as<double>(
                    cfg.process_noise_pos_max);
            cfg.process_noise_vel =
                kalman["process_noise_vel"].as<double>(cfg.process_noise_vel);
            cfg.measure_noise =
                kalman["measure_noise"].as<double>(cfg.measure_noise);
            cfg.default_dt = kalman["default_dt"].as<double>(cfg.default_dt);
            cfg.armor_radius =
                kalman["armor_radius"].as<double>(cfg.armor_radius);
            cfg.y_noise_mult =
                kalman["y_noise_mult"].as<double>(cfg.y_noise_mult);
            cfg.p_cov_min = kalman["p_cov_min"].as<double>(cfg.p_cov_min);
            cfg.angle_measure_noise =
                kalman["angle_measure_noise"].as<double>(cfg.angle_measure_noise);
            cfg.bearing_measure_noise =
                kalman["bearing_measure_noise"].as<double>(
                    cfg.bearing_measure_noise);
            cfg.pitch_measure_noise =
                kalman["pitch_measure_noise"].as<double>(
                    cfg.pitch_measure_noise);
            cfg.range_noise_distance_scale =
                kalman["range_noise_distance_scale"].as<double>(
                    cfg.range_noise_distance_scale);
            cfg.ypd_nis_threshold =
                kalman["ypd_nis_threshold"].as<double>(cfg.ypd_nis_threshold);
            cfg.max_angle_error =
                kalman["max_angle_error"].as<double>(cfg.max_angle_error);
            cfg.angle_match_weight =
                kalman["angle_match_weight"].as<double>(cfg.angle_match_weight);
            cfg.position_nis_threshold =
                kalman["position_nis_threshold"].as<double>(
                    cfg.position_nis_threshold);
            cfg.angle_nis_threshold =
                kalman["angle_nis_threshold"].as<double>(
                    cfg.angle_nis_threshold);

            // UV（像素重投影）观测：见 docs/uv_observation.md。
            cfg.uv_observation =
                kalman["uv_observation"].as<bool>(cfg.uv_observation);
            cfg.uv_sigma_px =
                kalman["uv_sigma_px"].as<double>(cfg.uv_sigma_px);
            cfg.uv_nis_threshold =
                kalman["uv_nis_threshold"].as<double>(cfg.uv_nis_threshold);
            cfg.uv_compact = kalman["uv_compact"].as<bool>(cfg.uv_compact);
            cfg.uv_compact_sigma_px =
                kalman["uv_compact_sigma_px"].as<double>(cfg.uv_compact_sigma_px);
            cfg.uv_compact_sigma_angle =
                kalman["uv_compact_sigma_angle"].as<double>(cfg.uv_compact_sigma_angle);
            // UV 用的物理板面尺寸（与 pnp 段的"等效尺寸"分开，见 tracker.hpp 注释）。
            cfg.uv_armor_small_width =
                kalman["uv_armor_small_width"].as<double>(cfg.uv_armor_small_width);
            cfg.uv_armor_large_width =
                kalman["uv_armor_large_width"].as<double>(cfg.uv_armor_large_width);
            cfg.uv_armor_height =
                kalman["uv_armor_height"].as<double>(cfg.uv_armor_height);

            if (const YAML::Node offsets = kalman["armor_y_offsets"]) {
                const std::vector<double> values =
                    offsets.as<std::vector<double>>();
                if (values.size() == cfg.armor_y_offsets.size()) {
                    std::copy(values.begin(), values.end(),
                              cfg.armor_y_offsets.begin());
                } else {
                    std::cerr << "Ignoring armor_y_offsets: expected four values"
                              << std::endl;
                }
            }
        }
        // UV 观测的板面尺寸必须与 PnP 完全一致（否则"重投影"和"观测"不是同一个
        // 几何量，残差会带上模型偏差），所以直接复用 pnp 段，不在 kalman 段重复配置。
        if (const YAML::Node pnp = tracker["pnp"]) {
            cfg.armor_small_width =
                pnp["small_width"].as<double>(cfg.armor_small_width);
            cfg.armor_large_width =
                pnp["large_width"].as<double>(cfg.armor_large_width);
            cfg.armor_height = pnp["height"].as<double>(cfg.armor_height);
            // UV 的板高没单独配就直接跟 PnP 走：这两处必须是同一个几何量，
            // 否则"重投影的模型"和"观测的角点"不是一回事，残差会带模型偏差。
            if (!tracker["kalman"]["uv_armor_height"]) {
                cfg.uv_armor_height = cfg.armor_height;
            }
        }
        return cfg;
    }

    // `tracker_file` is the parsed tracker.yaml root node.
    inline PnpGeometry loadPnpGeometry(const YAML::Node& tracker_file)
    {
        PnpGeometry geometry;
        if (const YAML::Node pnp = tracker_file["tracker"]["pnp"]) {
            geometry.small_width = pnp["small_width"].as<float>(geometry.small_width);
            geometry.large_width = pnp["large_width"].as<float>(geometry.large_width);
            geometry.height = pnp["height"].as<float>(geometry.height);
        }
        return geometry;
    }

    // ---- 控制层的三个配置 -------------------------------------------------
    // 这几段原来是在 node_sim.cpp 里就地解析的（默认值也写死在那一大段里），
    // 结果"结构体默认值"和"加载器默认值"两处各有一套数，改一处就会漂。
    // 现在统一到这三个加载函数：**默认值只存在于结构体**，
    // YAML 里没写的项一律回落到结构体默认值（和 sp_vision 的 config 一个路子）。
    // 角度类参数在 YAML 里仍写成"度"，在这里换成弧度 —— 只有配置文件里用度，
    // 代码里全是弧度。
    inline constexpr double kDegreesToRadians = 3.14159265358979323846 / 180.0;

    inline GimbalAimConfig loadGimbalConfig(const YAML::Node& tracker_file)
    {
        GimbalAimConfig cfg;
        const YAML::Node gimbal = tracker_file["tracker"]["gimbal"];
        if (!gimbal) return cfg;

        cfg.pitch_min = gimbal["pitch_min_degrees"].as<double>(
            cfg.pitch_min / kDegreesToRadians) * kDegreesToRadians;
        cfg.pitch_max = gimbal["pitch_max_degrees"].as<double>(
            cfg.pitch_max / kDegreesToRadians) * kDegreesToRadians;
        cfg.max_yaw_velocity = gimbal["max_yaw_velocity_degrees_per_sec"].as<double>(
            cfg.max_yaw_velocity / kDegreesToRadians) * kDegreesToRadians;
        cfg.max_pitch_velocity = gimbal["max_pitch_velocity_degrees_per_sec"].as<double>(
            cfg.max_pitch_velocity / kDegreesToRadians) * kDegreesToRadians;
        cfg.max_yaw_acceleration =
            gimbal["max_yaw_acceleration_degrees_per_sec2"].as<double>(
                cfg.max_yaw_acceleration / kDegreesToRadians) * kDegreesToRadians;
        cfg.max_pitch_acceleration =
            gimbal["max_pitch_acceleration_degrees_per_sec2"].as<double>(
                cfg.max_pitch_acceleration / kDegreesToRadians) * kDegreesToRadians;
        cfg.max_yaw_jerk = gimbal["max_yaw_jerk_degrees_per_sec3"].as<double>(
            cfg.max_yaw_jerk / kDegreesToRadians) * kDegreesToRadians;
        cfg.max_pitch_jerk = gimbal["max_pitch_jerk_degrees_per_sec3"].as<double>(
            cfg.max_pitch_jerk / kDegreesToRadians) * kDegreesToRadians;
        cfg.yaw_response_gain =
            gimbal["yaw_response_gain"].as<double>(cfg.yaw_response_gain);
        cfg.pitch_response_gain =
            gimbal["pitch_response_gain"].as<double>(cfg.pitch_response_gain);
        cfg.feedforward_gain =
            gimbal["feedforward_gain"].as<double>(cfg.feedforward_gain);
        cfg.feedforward_time_constant = gimbal["feedforward_time_constant"].as<double>(
            cfg.feedforward_time_constant);
        cfg.settle_angle = gimbal["settle_angle_degrees"].as<double>(
            cfg.settle_angle / kDegreesToRadians) * kDegreesToRadians;
        cfg.settle_velocity = gimbal["settle_velocity_degrees_per_sec"].as<double>(
            cfg.settle_velocity / kDegreesToRadians) * kDegreesToRadians;
        cfg.max_dt = gimbal["max_control_dt"].as<double>(cfg.max_dt);
        return cfg;
    }

    // 云台指令下发频率不是 GimbalAimConfig 的字段（它属于 entry 层），单独取。
    inline double loadGimbalCommandRateHz(const YAML::Node& tracker_file)
    {
        const YAML::Node gimbal = tracker_file["tracker"]["gimbal"];
        if (!gimbal) return 100.0;
        return gimbal["command_rate_hz"].as<double>(100.0);
    }

    inline TargetSelectorConfig loadSelectorConfig(const YAML::Node& tracker_file)
    {
        TargetSelectorConfig cfg;
        // 选择器算"云台转过去要多久"用的角速度/角加速度，默认直接取云台段的
        // 物理限制 —— 这两处以前在 YAML 里各写一遍（数值还一样），改一个漏一个
        // 就会让选板的时间估计和云台实际能力不一致。
        const GimbalAimConfig gimbal = loadGimbalConfig(tracker_file);
        cfg.yaw_velocity = gimbal.max_yaw_velocity;
        cfg.pitch_velocity = gimbal.max_pitch_velocity;
        cfg.yaw_acceleration = gimbal.max_yaw_acceleration;
        cfg.pitch_acceleration = gimbal.max_pitch_acceleration;
        // 旧配置把这一段叫 predictive_aim，保留兼容。
        const YAML::Node selector = tracker_file["tracker"]["selector"]
            ? tracker_file["tracker"]["selector"]
            : tracker_file["tracker"]["predictive_aim"];
        if (!selector) return cfg;

        cfg.enabled = selector["enabled"].as<bool>(cfg.enabled);
        cfg.projectile_speed =
            selector["projectile_speed"].as<double>(cfg.projectile_speed);
        cfg.gravity = selector["gravity"].as<double>(cfg.gravity);
        cfg.command_latency =
            selector["command_latency"].as<double>(cfg.command_latency);
        cfg.max_lead_time = selector["max_lead_time"].as<double>(cfg.max_lead_time);
        cfg.spin_omega_threshold =
            selector["spin_omega_threshold"].as<double>(cfg.spin_omega_threshold);
        cfg.coming_angle = selector["coming_angle_degrees"].as<double>(
            cfg.coming_angle / kDegreesToRadians) * kDegreesToRadians;
        cfg.leaving_angle = selector["leaving_angle_degrees"].as<double>(
            cfg.leaving_angle / kDegreesToRadians) * kDegreesToRadians;
        cfg.yaw_velocity = selector["gimbal_yaw_velocity_degrees_per_sec"].as<double>(
            cfg.yaw_velocity / kDegreesToRadians) * kDegreesToRadians;
        cfg.pitch_velocity =
            selector["gimbal_pitch_velocity_degrees_per_sec"].as<double>(
                cfg.pitch_velocity / kDegreesToRadians) * kDegreesToRadians;
        cfg.yaw_acceleration =
            selector["gimbal_yaw_acceleration_degrees_per_sec2"].as<double>(
                cfg.yaw_acceleration / kDegreesToRadians) * kDegreesToRadians;
        cfg.pitch_acceleration =
            selector["gimbal_pitch_acceleration_degrees_per_sec2"].as<double>(
                cfg.pitch_acceleration / kDegreesToRadians) * kDegreesToRadians;
        cfg.handoff_start_angle =
            selector["handoff_start_angle_degrees"].as<double>(
                cfg.handoff_start_angle / kDegreesToRadians) * kDegreesToRadians;
        cfg.handoff_duration =
            selector["handoff_duration"].as<double>(cfg.handoff_duration);
        cfg.decide_speed = selector["decide_speed"].as<double>(cfg.decide_speed);
        cfg.low_speed_latency =
            selector["low_speed_latency"].as<double>(cfg.low_speed_latency);
        cfg.high_speed_latency =
            selector["high_speed_latency"].as<double>(cfg.high_speed_latency);
        cfg.lock_switch_margin =
            selector["lock_switch_margin"].as<double>(cfg.lock_switch_margin);
        return cfg;
    }

    inline ShooterConfig loadShooterConfig(const YAML::Node& tracker_file)
    {
        ShooterConfig cfg;
        const YAML::Node shooter = tracker_file["tracker"]["shooter"];
        if (!shooter) return cfg;
        cfg.min_fire_interval =
            shooter["min_fire_interval"].as<double>(cfg.min_fire_interval);
        cfg.ready_handoff_gain =
            shooter["ready_handoff_gain"].as<double>(cfg.ready_handoff_gain);
        cfg.end_handoff_gain =
            shooter["end_handoff_gain"].as<double>(cfg.end_handoff_gain);
        return cfg;
    }

    inline AimSignalFilterConfig loadAimFilterConfig(const YAML::Node& tracker_file)
    {
        AimSignalFilterConfig cfg;
        const YAML::Node aim = tracker_file["tracker"]["aim_filter"];
        if (!aim) return cfg;
        cfg.enabled = aim["enabled"].as<bool>(cfg.enabled);
        cfg.process_noise_acceleration =
            aim["process_noise_acceleration"].as<double>(
                cfg.process_noise_acceleration);
        cfg.measurement_noise =
            aim["measurement_noise"].as<double>(cfg.measurement_noise);
        cfg.reset_innovation = aim["reset_innovation_degrees"].as<double>(
            cfg.reset_innovation / kDegreesToRadians) * kDegreesToRadians;
        cfg.reset_timeout = aim["reset_timeout"].as<double>(cfg.reset_timeout);
        cfg.max_dt = aim["max_dt"].as<double>(cfg.max_dt);
        return cfg;
    }

    // 这两个既不属于 TargetSelectorConfig 也不属于 ShooterConfig，
    // 但同属"选择器/射手"段的战术参数，放在这里一起取。
    inline int loadMaxControlLostFrames(const YAML::Node& tracker_file)
    {
        const YAML::Node selector = tracker_file["tracker"]["selector"]
            ? tracker_file["tracker"]["selector"]
            : tracker_file["tracker"]["predictive_aim"];
        if (!selector) return 5;
        return selector["max_control_lost_frames"].as<int>(5);
    }

    inline double loadFireAngleToleranceDegrees(const YAML::Node& tracker_file)
    {
        const YAML::Node selector = tracker_file["tracker"]["selector"]
            ? tracker_file["tracker"]["selector"]
            : tracker_file["tracker"]["predictive_aim"];
        if (const YAML::Node shooter = tracker_file["tracker"]["shooter"]) {
            return shooter["fire_angle_tolerance_degrees"].as<double>(3.0);
        }
        if (!selector) return 3.0;
        return selector["fire_angle_tolerance_degrees"].as<double>(3.0);
    }

    // `serial_file` 是 configs/serial.yaml 的根节点（实车链路）。
    // device 支持 "auto"：按 /dev/ttyACM* → /dev/ttyUSB* 找第一个存在的设备
    // （STM32 的 USB CDC 通常是 ttyACM0；CP210x/CH340 是 ttyUSB*）。
    inline io::SerialConfig loadSerialConfig(const YAML::Node& serial_file)
    {
        io::SerialConfig cfg;
        const YAML::Node serial = serial_file["serial"] ? serial_file["serial"] : serial_file;
        if (!serial) return cfg;
        cfg.device = serial["device"].as<std::string>(cfg.device);
        cfg.baud = serial["baud"].as<int>(cfg.baud);
        cfg.read_timeout_ms = serial["read_timeout_ms"].as<int>(cfg.read_timeout_ms);
        cfg.reconnect_after_errors =
            serial["reconnect_after_errors"].as<int>(cfg.reconnect_after_errors);
        cfg.require_connection =
            serial["require_connection"].as<bool>(cfg.require_connection);
        return cfg;
    }

#ifdef ULTRA_VISION_USE_OPENVINO
    // `detector_file` is the parsed detector.yaml root node. Relative model
    // paths are resolved against the directory that contains the config
    // directory, so "models/openvino/yolo11.xml" works from any working
    // directory as long as the config lives in <repo>/configs.
    inline NnDetectorConfig loadNeuralDetectorConfig(const YAML::Node& detector_file,
                                                     const std::string& config_dir)
    {
        NnDetectorConfig cfg;
        const YAML::Node neural = detector_file["detector"]["neural"];
        if (!neural) return cfg;

        cfg.model_path = neural["model_path"].as<std::string>("");
        cfg.device = neural["device"].as<std::string>("CPU");
        cfg.output_format =
            neural["output_format"].as<std::string>(cfg.output_format);
        cfg.input_size = neural["input_size"].as<int>(cfg.input_size);
        cfg.score_threshold =
            neural["score_threshold"].as<double>(cfg.score_threshold);
        cfg.nms_threshold = neural["nms_threshold"].as<double>(cfg.nms_threshold);
        cfg.min_confidence =
            neural["min_confidence"].as<double>(cfg.min_confidence);
        cfg.filter_by_color = neural["filter_by_color"].as<bool>(true);
        cfg.enemy_color = detector_file["detector"]["enemy_color"].as<int>(0);
        cfg.performance_mode =
            neural["performance_mode"].as<std::string>(cfg.performance_mode);
        cfg.num_threads = neural["num_threads"].as<int>(cfg.num_threads);
        cfg.num_streams = neural["num_streams"].as<int>(cfg.num_streams);
        if (const YAML::Node refine = neural["refine"]) {
            cfg.refiner.enabled = refine["enabled"].as<bool>(cfg.refiner.enabled);
            cfg.refiner.binary_threshold =
                refine["binary_threshold"].as<int>(cfg.refiner.binary_threshold);
            cfg.refiner.margin_ratio =
                refine["margin_ratio"].as<double>(cfg.refiner.margin_ratio);
            cfg.refiner.max_center_shift_ratio = refine["max_center_shift_ratio"]
                .as<double>(cfg.refiner.max_center_shift_ratio);
        }
        cfg.use_roi = neural["use_roi"].as<bool>(false);
        if (const YAML::Node roi = neural["roi"]) {
            cfg.roi = cv::Rect(roi["x"].as<int>(0), roi["y"].as<int>(0),
                               roi["width"].as<int>(-1), roi["height"].as<int>(-1));
        }

        if (!cfg.model_path.empty() && cfg.model_path.front() != '/') {
            const std::string candidate = config_dir + "/../" + cfg.model_path;
            if (std::filesystem::exists(candidate)) cfg.model_path = candidate;
        }
        return cfg;
    }

    inline bool neuralDetectorEnabled(const YAML::Node& detector_file)
    {
        const YAML::Node neural = detector_file["detector"]["neural"];
        return neural && neural["enabled"].as<bool>(false);
    }
#endif
} // namespace auto_aim

#endif // ULTRA_VISION_AUTO_AIM_CONFIG_LOADER_HPP
