// YAML → RuneConfig 的实现（结构定义在 rune_config.hpp，这里只放"怎么读"）。

#include "auto_buff/rune_config.hpp"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace auto_aim::energy
{
    namespace
    {
        std::vector<double> readDoubleList(const YAML::Node& node)
        {
            std::vector<double> values;
            values.reserve(node.size());
            for (const auto& value : node) values.push_back(value.as<double>());
            return values;
        }

        cv::Mat readMatrix(const YAML::Node& node, int rows, int columns)
        {
            const std::vector<double> values = readDoubleList(node);
            if (values.size() != static_cast<std::size_t>(rows * columns)) {
                throw std::runtime_error("RuneConfig: unexpected matrix size");
            }
            cv::Mat matrix(rows, columns, CV_64F);
            for (int row = 0; row < rows; ++row) {
                for (int column = 0; column < columns; ++column) {
                    matrix.at<double>(row, column) =
                        values[static_cast<std::size_t>(row) * columns + column];
                }
            }
            return matrix;
        }
    } // namespace

    RuneConfig loadRuneConfig(const std::string& path)
    {
        const YAML::Node file = YAML::LoadFile(path);
        const YAML::Node rune = file["rune"] ? file["rune"] : file;
        RuneConfig config;

        config.enabled = rune["enabled"].as<bool>(true);
        const std::string mode = rune["mode"].as<std::string>("small");
        config.mode = (mode == "large") ? RuneMode::Large : RuneMode::Small;

        // ---- camera ---------------------------------------------------------
        const YAML::Node camera = rune["camera"];
        if (camera) {
            config.solver.camera_matrix = readMatrix(camera["camera_matrix"], 3, 3);
            config.solver.distort_coeffs = readMatrix(camera["distort_coeffs"], 5, 1);
            config.solver.calibration_width = camera["calibration_width"].as<int>(640);
            config.solver.calibration_height = camera["calibration_height"].as<int>(480);
            config.solver.auto_scale_intrinsics =
                camera["auto_scale_intrinsics"].as<bool>(true);
        } else {
            throw std::runtime_error("RuneConfig: camera intrinsics are required");
        }

        // ---- network --------------------------------------------------------
        const YAML::Node model = rune["model"];
        config.detector.model.model_path = model["path"].as<std::string>("");
        config.detector.model.device = model["device"].as<std::string>("CPU");
        config.detector.model.input_size = model["input_size"].as<int>(640);
        config.detector.model.confidence_threshold =
            model["score_threshold"].as<float>(0.7f);
        config.detector.model.nms_threshold = model["nms_threshold"].as<float>(0.4f);
        config.detector.model.pad_value = model["pad_value"].as<int>(114);
        config.detector.model.output_layout =
            model["output_layout"].as<std::string>("sp_vision_6kpt");
        config.detector.model.num_classes = model["num_classes"].as<int>(1);
        config.detector.model.num_keypoints = model["num_keypoints"].as<int>(6);
        config.detector.model.keypoint_confidence_threshold =
            model["keypoint_confidence_threshold"].as<float>(0.8f);
        config.detector.model.min_valid_keypoints =
            model["min_valid_keypoints"].as<int>(3);
        config.detector.model.class_scores_are_logits =
            model["class_scores_are_logits"].as<bool>(false);
        config.detector.model.keypoints_have_confidence =
            model["keypoints_have_confidence"].as<bool>(false);
        config.detector.model.input_height = model["input_height"].as<int>(0);
        config.detector.model.input_pad_scale =
            model["input_pad_scale"].as<double>(1.0);
        // 逐帧自适应画布缩放（把机关钉在网络的训练尺度上，见 rune_detector.hpp）。
        config.detector.canvas_adaptive =
            model["input_pad_adaptive"].as<bool>(false);
        config.detector.canvas_target_orbit_px =
            model["input_pad_target_orbit_px"].as<double>(58.0);
        config.detector.canvas_scale_min = model["input_pad_scale_min"].as<double>(0.4);
        config.detector.canvas_scale_max = model["input_pad_scale_max"].as<double>(2.6);
        config.detector.canvas_sweep_after_lost =
            model["input_pad_sweep_after_lost"].as<int>(6);
        if (const YAML::Node sweep = model["input_pad_sweep"]) {
            std::vector<double> scales;
            for (const auto& value : sweep) scales.push_back(value.as<double>());
            if (!scales.empty()) config.detector.canvas_sweep_scales = std::move(scales);
        }
        config.detector.model.input_scale = model["input_scale"].as<double>(1.0 / 255.0);
        config.detector.model.nms_center_dist_px =
            model["nms_center_dist_px"].as<float>(30.0f);
        config.detector.model.min_plate_radius_px =
            model["min_plate_radius_px"].as<float>(3.0f);
        config.detector.model.num_threads = model["num_threads"].as<int>(4);
        config.model_path = config.detector.model.model_path;

        // ---- detector -------------------------------------------------------
        const YAML::Node detector = rune["detector"];
        config.detector.max_lost_frames = detector["max_lost_frames"].as<int>(20);
        config.detector.gray_threshold = detector["gray_threshold"].as<int>(100);
        config.detector.dilate_size = detector["dilate_size"].as<int>(5);
        config.detector.center_mask_ratio = detector["center_mask_ratio"].as<double>(0.8);
        config.detector.center_extrapolation =
            detector["center_extrapolation"].as<double>(1.4);
        config.detector.multi_candidate = detector["multi_candidate"].as<bool>(false);
        config.detector.fallback_single_blade =
            detector["fallback_single_blade"].as<bool>(true);
        config.detector.prefer_bright_blade = detector["prefer_bright_blade"].as<bool>(true);
        // 我们自己的颜色：0=红（颜色差掩膜看 R−B）1=蓝（看 B−R）。
        config.detector.our_color = detector["our_color"].as<int>(0);
        // 观测精修（轮廓几何 + 描述子闸门，见 rune_refiner.hpp）。默认关闭。
        if (const YAML::Node refiner = detector["refiner"]) {
            auto& refine = config.detector.refiner;
            refine.enabled = refiner["enabled"].as<bool>(false);
            refine.roi_scale = refiner["roi_scale"].as<double>(1.4);
            refine.red_minus_blue_threshold =
                refiner["red_minus_blue_threshold"].as<double>(30.0);
            refine.blue_minus_red_threshold =
                refiner["blue_minus_red_threshold"].as<double>(62.0);
            refine.armor_area_relative_error =
                refiner["armor_area_relative_error"].as<double>(0.35);
            refine.light_arm_line_samples = refiner["light_arm_line_samples"].as<int>(15);
            refine.armor_min_solidity = refiner["armor_min_solidity"].as<double>(0.80);
            refine.arm_min_solidity = refiner["arm_min_solidity"].as<double>(0.66);
            refine.arm_expect_aspect = refiner["arm_expect_aspect"].as<double>(5.0);
            refine.arm_aspect_tolerance = refiner["arm_aspect_tolerance"].as<double>(0.42);
            refine.require_usable = refiner["require_usable"].as<bool>(false);
            refine.fuse_center = refiner["fuse_center"].as<bool>(true);
            refine.fuse_requires_armor_usable =
                refiner["fuse_requires_armor_usable"].as<bool>(true);
            refine.fuse_weight_contour = refiner["fuse_weight_contour"].as<double>(0.5);
            refine.max_center_gap_px = refiner["max_center_gap_px"].as<double>(4.0);
            refine.border_margin_px = refiner["border_margin_px"].as<int>(2);
        }
        if (const YAML::Node plate = detector["plate_refiner"]) {
            config.detector.plate_refiner.enabled = plate["enabled"].as<bool>(true);
            config.detector.plate_refiner.margin_ratio = plate["margin_ratio"].as<double>(0.25);
            config.detector.plate_refiner.use_otsu = plate["use_otsu"].as<bool>(true);
            config.detector.plate_refiner.morph_size = plate["morph_size"].as<int>(3);
            config.detector.plate_refiner.min_area_ratio =
                plate["min_area_ratio"].as<double>(0.25);
            config.detector.plate_refiner.max_area_ratio =
                plate["max_area_ratio"].as<double>(3.0);
            config.detector.plate_refiner.max_center_shift_ratio =
                plate["max_center_shift_ratio"].as<double>(0.25);
            config.detector.plate_refiner.min_size_ratio =
                plate["min_size_ratio"].as<double>(0.6);
            config.detector.plate_refiner.max_size_ratio =
                plate["max_size_ratio"].as<double>(1.6);
            config.detector.plate_refiner.preserve_network_scale =
                plate["preserve_network_scale"].as<bool>(true);
            config.detector.plate_refiner.center_weight =
                plate["center_weight"].as<double>(0.5);
            config.detector.plate_refiner.rotation_weight =
                plate["rotation_weight"].as<double>(0.0);
            config.detector.plate_refiner.max_angle_diff_deg =
                plate["max_angle_diff_deg"].as<double>(25.0);
        }
        config.detector.require_inactive_class =
            detector["require_inactive_class"].as<bool>(true);
        // 扇叶状态分类（经典特征，华南虎/SCUT 的面积区间思路；见 rune_blade_state.hpp）。
        // 阈值必须按实拍标注帧标定 —— 用 tools/buff_state_probe 同一条代码路径量。
        if (const YAML::Node state = detector["blade_state"]) {
            auto& blade_state = config.detector.blade_state;
            blade_state.red_minus_blue = state["red_minus_blue"].as<double>(60.0);
            blade_state.roi_half_ratio = state["roi_half_ratio"].as<double>(0.35);
            blade_state.plate_orbit_ratio = state["plate_orbit_ratio"].as<double>(0.214);
            blade_state.unlit_below = state["unlit_below"].as<double>(0.06);
            blade_state.active_above = state["active_above"].as<double>(0.22);
            blade_state.arm_samples = state["arm_samples"].as<int>(9);
            blade_state.arm_lit_fraction = state["arm_lit_fraction"].as<double>(0.5);
        }
        config.detector.inactive_class_margin =
            detector["inactive_class_margin"].as<double>(1.3);
        config.detector.rescue_missing_inactive =
            detector["rescue_missing_inactive"].as<bool>(false);
        config.detector.engageable_latch_s =
            detector["engageable_latch_s"].as<double>(0.8);
        config.detector.blade_roi_ratio = detector["blade_roi_ratio"].as<double>(0.6);
        config.detector.lock_switch_margin = detector["lock_switch_margin"].as<double>(1.35);
        config.detector.lock_switch_delta = detector["lock_switch_delta"].as<double>(12.0);
        config.detector.lock_switch_frames = detector["lock_switch_frames"].as<int>(3);
        config.detector.lock_max_miss = detector["lock_max_miss"].as<int>(3);
        config.detector.lock_break_px = detector["lock_break_px"].as<double>(60.0);
        config.detector.center_orbit_ratio =
            detector["center_orbit_ratio"].as<double>(4.4);
        config.detector.hub_brightness_margin =
            detector["hub_brightness_margin"].as<double>(60.0);
        if (detector["plate_point_order"]) {
            std::vector<int> order;
            std::stringstream stream(detector["plate_point_order"].as<std::string>());
            std::string token;
            while (std::getline(stream, token, ',')) {
                try {
                    order.push_back(std::stoi(token));
                } catch (const std::exception&) {
                    order.clear();
                    break;
                }
            }
            if (order.size() == 4) {
                config.detector.plate_point_indices = order;
            } else {
                std::cerr << "rune detector: plate_point_order needs 4 indices, keeping "
                          << "the default" << std::endl;
            }
        }
        config.recenter_after_frames = detector["recenter_after_frames"].as<int>(15);
        config.max_target_velocity_deg_s =
            detector["max_target_velocity_deg_s"].as<double>(200.0);
        config.zero_feedforward_on_step =
            detector["zero_feedforward_on_step"].as<bool>(true);
        config.fire_block_on_hit_slot =
            detector["fire_block_on_hit_slot"].as<bool>(true);
        // 帧差探针（差分法）：诊断/补充观测源。环境变量 ULTRA_VISION_RUNE_DIFF=1
        // 可以免改配置直接打开（A/B 用）。
        config.diff_probe.enabled =
            detector["diff_probe_enable"].as<bool>(false) ||
            (std::getenv("ULTRA_VISION_RUNE_DIFF") != nullptr &&
             std::string(std::getenv("ULTRA_VISION_RUNE_DIFF")) != "0");
        config.diff_probe.threshold = detector["diff_threshold"].as<double>(20.0);
        config.diff_probe.min_area = detector["diff_min_area"].as<double>(6.0);
        config.diff_probe.compensate_motion = detector["diff_compensate"].as<bool>(true);

        // 相位关联选片（见 RuneDetectorConfig::PhaseAssocConfig）。
        auto& assoc = config.detector.phase_assoc;
        assoc.enabled = detector["phase_assoc_enable"].as<bool>(true);
        assoc.tolerance_rad =
            detector["phase_assoc_tolerance_deg"].as<double>(34.0) * kPi / 180.0;
        assoc.rate_min_rad_s = detector["phase_assoc_rate_min"].as<double>(0.30);
        assoc.rate_max_rad_s = detector["phase_assoc_rate_max"].as<double>(2.09);
        assoc.nominal_rate_rad_s = detector["phase_assoc_nominal_rate"].as<double>(1.1775);
        assoc.max_gap_s = detector["phase_assoc_max_gap_s"].as<double>(0.6);
        assoc.rate_smooth = detector["phase_assoc_rate_smooth"].as<double>(0.35);
        assoc.rate_accept_margin = detector["phase_assoc_rate_margin"].as<double>(0.25);
        config.park_on_center_max_age_s =
            detector["park_on_center_max_age_s"].as<double>(1.5);
        config.max_coast_frames = detector["max_coast_frames"].as<int>(30);
        config.max_distance_jump_ratio = detector["max_distance_jump_ratio"].as<double>(0.35);
        config.max_center_jump_deg = detector["max_center_jump_deg"].as<double>(20.0);

        // ---- geometry -------------------------------------------------------
        const YAML::Node geometry = rune["geometry"];
        config.solver.target_radius_m = geometry["target_radius_m"].as<double>(0.700);
        config.solver.target_half_width_m = geometry["target_half_width_m"].as<double>(0.127);
        config.solver.arm_point_radius_m = geometry["arm_point_radius_m"].as<double>(0.220);
        config.solver.correct_depth_with_radius =
            geometry["correct_depth_with_radius"].as<bool>(true);
        config.solver.correct_depth_with_orbit_fit =
            geometry["correct_depth_with_orbit_fit"].as<bool>(true);
        config.solver.orbit_fit_window_s = geometry["orbit_fit_window_s"].as<double>(10.0);
        config.solver.orbit_fit_min_samples = geometry["orbit_fit_min_samples"].as<int>(15);
        config.solver.min_plate_offset_px = geometry["min_plate_offset_px"].as<double>(4.0);
        config.solver.min_distance_m = geometry["min_distance_m"].as<double>(1.5);
        config.solver.max_distance_m = geometry["max_distance_m"].as<double>(15.0);
        if (geometry["R_camera2gimbal"]) {
            const std::vector<double> values = readDoubleList(geometry["R_camera2gimbal"]);
            if (values.size() == 9) {
                for (int row = 0; row < 3; ++row) {
                    for (int column = 0; column < 3; ++column) {
                        config.solver.R_camera2gimbal(row, column) =
                            values[static_cast<std::size_t>(row) * 3 + column];
                    }
                }
            }
        }
        if (geometry["t_camera2gimbal"]) {
            const std::vector<double> values = readDoubleList(geometry["t_camera2gimbal"]);
            if (values.size() == 3) {
                config.solver.t_camera2gimbal = Eigen::Vector3d(values[0], values[1], values[2]);
            }
        }

        // ---- aiming / fire control -----------------------------------------
        const YAML::Node aimer = rune["aimer"];
        config.aimer.yaw_offset_deg = aimer["yaw_offset_deg"].as<double>(0.0);
        config.aimer.pitch_offset_deg = aimer["pitch_offset_deg"].as<double>(0.0);
        config.aimer.fire_gap_time = aimer["fire_gap_time"].as<double>(0.7);
        config.aimer.predict_time = aimer["predict_time"].as<double>(0.12);
        config.aimer.observation_life_s = aimer["observation_life_s"].as<double>(0.2);
        config.aimer.frame_time_from_exposure =
            aimer["frame_time_from_exposure"].as<bool>(true);
        config.aimer.smooth_lead_latency = aimer["smooth_lead_latency"].as<bool>(true);
        config.aimer.bullet_speed = aimer["bullet_speed"].as<double>(25.0);
        config.aimer.gravity = aimer["gravity"].as<double>(9.81);
        config.aimer.target_radius_m = config.solver.target_radius_m;
        config.aimer.switch_angle_deg = aimer["switch_angle_deg"].as<double>(5.0);
        config.aimer.max_mistakes = aimer["max_mistakes"].as<int>(3);
        config.aimer.max_flight_time_error =
            aimer["max_flight_time_error"].as<double>(0.03);
        config.aimer.decision_speed_rad_s =
            aimer["decision_speed_degrees_per_sec"].as<double>(90.0) * kPi / 180.0;
        config.aimer.low_speed_delay = aimer["low_speed_delay"].as<double>(0.06);
        config.aimer.high_speed_delay = aimer["high_speed_delay"].as<double>(0.13);
        config.fire_thresh_deg = aimer["fire_thresh_degrees"].as<double>(1.5);
        config.max_slew_deg_per_frame = aimer["max_slew_deg_per_frame"].as<double>(8.0);
        config.aim_yaw_limit_deg = aimer["aim_yaw_limit_deg"].as<double>(50.0);
        config.aim_pitch_limit_deg = aimer["aim_pitch_limit_deg"].as<double>(35.0);
        config.round_window_s = aimer["round_window_s"].as<double>(2.5);
        config.fire_start_delay_s = aimer["fire_start_delay_s"].as<double>(0.15);
        config.round_end_frames = aimer["round_end_frames"].as<int>(5);
        config.blade_slot_jump_px = aimer["blade_slot_jump_px"].as<double>(25.0);
        config.slot_lattice_reset_silence_s =
            aimer["slot_lattice_reset_silence_s"].as<double>(5.0);
        config.slot_lattice_round_end_s =
            aimer["slot_lattice_round_end_s"].as<double>(0.8);
        config.slot_activate_votes = aimer["slot_activate_votes"].as<int>(4);
        config.slot_book_retired_on_switch =
            aimer["slot_book_retired_on_switch"].as<bool>(true);
        config.slot_rotate_booking_on_switch =
            aimer["slot_rotate_booking_on_switch"].as<bool>(false);
        config.post_hit_switch_phase_rad =
            aimer["post_hit_switch_phase_rad"].as<double>(0.30);
        config.post_hit_hold_max_s = aimer["post_hit_hold_max_s"].as<double>(1.2);
        config.post_hit_require_active_witness =
            aimer["post_hit_require_active_witness"].as<bool>(false);

        // 云台轨迹参数：与 configs/tracker.yaml 的 tracker.gimbal 同义，默认取
        // 自瞄那套（左右 360°/s、俯仰 180°/s、加加速度受限、100 Hz 下发）。
        if (const YAML::Node gimbal = rune["gimbal"]) {
            auto read_deg = [&](const char* key, double fallback_deg) {
                return gimbal[key].as<double>(fallback_deg) * kPi / 180.0;
            };
            config.gimbal.pitch_min = read_deg("pitch_min_degrees", -70.0);
            config.gimbal.pitch_max = read_deg("pitch_max_degrees", 70.0);
            config.gimbal.max_yaw_velocity =
                read_deg("max_yaw_velocity_degrees_per_sec", 360.0);
            config.gimbal.max_pitch_velocity =
                read_deg("max_pitch_velocity_degrees_per_sec", 180.0);
            config.gimbal.max_yaw_acceleration =
                read_deg("max_yaw_acceleration_degrees_per_sec2", 1800.0);
            config.gimbal.max_pitch_acceleration =
                read_deg("max_pitch_acceleration_degrees_per_sec2", 1200.0);
            config.gimbal.max_yaw_jerk =
                read_deg("max_yaw_jerk_degrees_per_sec3", 20000.0);
            config.gimbal.max_pitch_jerk =
                read_deg("max_pitch_jerk_degrees_per_sec3", 12000.0);
            config.gimbal.yaw_response_gain =
                gimbal["yaw_response_gain"].as<double>(config.gimbal.yaw_response_gain);
            config.gimbal.pitch_response_gain =
                gimbal["pitch_response_gain"].as<double>(config.gimbal.pitch_response_gain);
            config.gimbal.feedforward_gain =
                gimbal["feedforward_gain"].as<double>(config.gimbal.feedforward_gain);
            config.gimbal.feedforward_time_constant =
                gimbal["feedforward_time_constant"].as<double>(
                    config.gimbal.feedforward_time_constant);
            config.gimbal.settle_angle = read_deg("settle_angle_degrees", 0.6);
            config.gimbal.settle_velocity = read_deg("settle_velocity_degrees_per_sec", 5.0);
            config.gimbal.max_dt = gimbal["max_control_dt"].as<double>(config.gimbal.max_dt);
            config.gimbal_command_rate_hz =
                gimbal["command_rate_hz"].as<double>(100.0);
        }

        // 瞄准角递归滤波：抑制单帧抖动、并在换靶时给出干净的过渡，
        // 参数含义与 configs/tracker.yaml 的 tracker.aim_filter 一致。
        if (const YAML::Node filter = rune["aim_filter"]) {
            config.aim_filter.enabled = filter["enabled"].as<bool>(true);
            config.aim_filter.process_noise_acceleration =
                filter["process_noise_acceleration"].as<double>(
                    config.aim_filter.process_noise_acceleration);
            config.aim_filter.measurement_noise =
                filter["measurement_noise"].as<double>(config.aim_filter.measurement_noise);
            config.aim_filter.reset_innovation =
                filter["reset_innovation_degrees"].as<double>(20.0) * kPi / 180.0;
            config.aim_filter.reset_timeout =
                filter["reset_timeout"].as<double>(config.aim_filter.reset_timeout);
            config.aim_filter.max_dt = filter["max_dt"].as<double>(config.aim_filter.max_dt);
        }

        config.simulator_config = rune["simulator_config"].as<std::string>("simulator.yaml");
        config.visualize = rune["visualize"].as<bool>(true);
        return config;
    }
} // namespace auto_aim::energy
