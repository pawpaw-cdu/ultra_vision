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
#include <cstddef>
#include <iostream>
#include <vector>

#include <opencv2/core.hpp>
#include <yaml-cpp/yaml.h>

#include "Kalman/tracker.hpp"
#include "common/types.hpp"
#include "perception/detector.hpp"

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

    // Physical armor plate dimensions, used by PnP and by the estimate overlay.
    struct PnpGeometry {
        float small_width = 0.135f;
        float large_width = 0.225f;
        float height = 0.055f;
    };

    // `detector_file` is the parsed detector.yaml root node.
    inline DetectorConfig loadDetectorConfig(const YAML::Node& detector_file)
    {
        const YAML::Node detector = detector_file["detector"];
        DetectorConfig cfg{};

        cfg.enemy_color = detector["enemy_color"].as<int>(cfg.enemy_color);
        cfg.binary_threshold =
            detector["binary_threshold"].as<int>(cfg.binary_threshold);

        const YAML::Node light = detector["light"];
        cfg.light_min_ratio = light["min_ratio"].as<double>(0.01);
        cfg.light_max_ratio = light["max_ratio"].as<double>(0.15);
        cfg.light_max_angle = light["max_angle"].as<double>(10.0);
        cfg.light_min_contour_points = light["min_contour_points"].as<int>(5);

        const YAML::Node armor = detector["armor"];
        cfg.armor_height_ratio_min = armor["height_ratio_min"].as<double>(0.70);
        cfg.armor_height_ratio_max = armor["height_ratio_max"].as<double>(1.25);
        cfg.armor_angle_diff_max = armor["angle_diff_max"].as<double>(30.0);
        cfg.armor_width_to_height_min = armor["width_to_height_min"].as<double>(0.8);
        cfg.armor_width_to_height_max = armor["width_to_height_max"].as<double>(3.0);
        cfg.armor_large_ratio_thresh =
            armor["large_armor_ratio_thresh"].as<double>(3.0);

        const YAML::Node color = detector["color"];
        cfg.color_use_detect = color["use_color_detect"].as<bool>(true);
        cfg.use_hsv = color["method"].as<std::string>("bgr") == "hsv";
        if (cfg.use_hsv) {
            const YAML::Node hsv = color["hsv"];
            cfg.hue_red_low1 = hsv["red_low1"].as<int>(0);
            cfg.hue_red_high1 = hsv["red_high1"].as<int>(35);
            cfg.hue_red_low2 = hsv["red_low2"].as<int>(135);
            cfg.hue_red_high2 = hsv["red_high2"].as<int>(180);
            cfg.hue_blue_low = hsv["blue_low"].as<int>(90);
            cfg.hue_blue_high = hsv["blue_high"].as<int>(135);
            cfg.sat_min = hsv["sat_min"].as<int>(80);
            cfg.val_min = hsv["val_min"].as<int>(80);
            cfg.color_area_ratio = hsv["area_ratio"].as<double>(0.3);
        } else {
            const YAML::Node bgr = color["bgr"];
            cfg.color_red_threshold = bgr["red_threshold"].as<double>(1.0);
            cfg.color_blue_threshold = bgr["blue_threshold"].as<double>(1.0);
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
            // Kept for compatibility with older configuration files.
            cfg.state_dim = kalman["state_dim"].as<int>(cfg.state_dim);
            cfg.measure_dim = kalman["measure_dim"].as<int>(cfg.measure_dim);
            cfg.init_error_cov =
                kalman["init_error_cov"].as<double>(cfg.init_error_cov);

            cfg.process_noise_pos =
                kalman["process_noise_pos"].as<double>(cfg.process_noise_pos);
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
