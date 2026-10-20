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
#include "config/tracker_config_loader.hpp"
#include "config/control_config_loader.hpp"
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
        // 下发命令的坐标系与符号（见 SerialConfig 的注释；2026-09-29 实测量定）。
        cfg.yaw_sign = serial["yaw_sign"].as<double>(1.0) < 0.0 ? -1.0 : 1.0;
        const std::string frame = serial["command_frame"].as<std::string>("world");
        cfg.command_frame = frame == "motor" ? io::SerialConfig::CommandFrame::Motor
                                            : io::SerialConfig::CommandFrame::World;
        cfg.command_world_yaw_sign =
            serial["command_world_yaw_sign"].as<double>(cfg.command_world_yaw_sign);
        cfg.command_world_pitch_sign =
            serial["command_world_pitch_sign"].as<double>(cfg.command_world_pitch_sign);
        cfg.command_yaw_sign = serial["command_yaw_sign"].as<double>(cfg.command_yaw_sign);
        cfg.command_pitch_sign =
            serial["command_pitch_sign"].as<double>(cfg.command_pitch_sign);
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
