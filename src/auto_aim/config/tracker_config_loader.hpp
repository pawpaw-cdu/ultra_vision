#ifndef AUTO_AIM_CONFIG_TRACKER_CONFIG_LOADER_HPP
#define AUTO_AIM_CONFIG_TRACKER_CONFIG_LOADER_HPP

// 从 config_loader.hpp 抽出的 tracker.yaml 解析（最大的一个 loader）。

#include <algorithm>
#include <array>
#include <cstdlib>

#include <yaml-cpp/yaml.h>

#include "Kalman/tracker.hpp"

namespace auto_aim
{
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
        // 收敛看门狗
        cfg.watchdog_enabled = tracker["watchdog_enabled"].as<bool>(cfg.watchdog_enabled);
        cfg.watchdog_window = tracker["watchdog_window"].as<int>(cfg.watchdog_window);
        cfg.watchdog_reject_ratio =
            tracker["watchdog_reject_ratio"].as<double>(cfg.watchdog_reject_ratio);
        cfg.watchdog_min_samples =
            tracker["watchdog_min_samples"].as<int>(cfg.watchdog_min_samples);
        cfg.watchdog_max_speed = tracker["watchdog_max_speed"].as<double>(cfg.watchdog_max_speed);
        cfg.watchdog_max_yaw_rate =
            tracker["watchdog_max_yaw_rate"].as<double>(cfg.watchdog_max_yaw_rate);
        cfg.watchdog_max_range = tracker["watchdog_max_range"].as<double>(cfg.watchdog_max_range);
        cfg.watchdog_max_position_cov =
            tracker["watchdog_max_position_cov"].as<double>(cfg.watchdog_max_position_cov);
        cfg.watchdog_max_frame_gap =
            tracker["watchdog_max_frame_gap"].as<double>(cfg.watchdog_max_frame_gap);
        cfg.watchdog_min_converge_updates = tracker["watchdog_min_converge_updates"]
                                                .as<int>(cfg.watchdog_min_converge_updates);

        // 把识别到的装甲板当硬约束（默认开）
        cfg.hard_plate_constraint =
            tracker["hard_plate_constraint"].as<bool>(cfg.hard_plate_constraint);
        cfg.plate_lock_margin =
            tracker["plate_lock_margin"].as<double>(cfg.plate_lock_margin);

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
            // env 覆盖（仿真/实车切换用）：仿真车的装甲板回转半径真值是 0.2101，
            // 实车实测约 0.29，两者不能共用同一个配置值。
            if (const char * r = std::getenv("ULTRA_VISION_ARMOR_RADIUS")) {
                const double v = std::atof(r);
                if (v > 0.05 && v < 1.0) cfg.armor_radius = v;
            }
            // 重投影模型外推（时间基）是否连中心一起按底速平移；见 TrackerConfig 注释。
            cfg.model_lead_center =
                kalman["model_lead_center"].as<bool>(cfg.model_lead_center);
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
            cfg.adaptive_association_gate = kalman["adaptive_association_gate"]
                                                .as<bool>(cfg.adaptive_association_gate);
            cfg.association_gate_sigma = kalman["association_gate_sigma"]
                                             .as<double>(cfg.association_gate_sigma);
            cfg.association_gate_max = kalman["association_gate_max"]
                                           .as<double>(cfg.association_gate_max);
            cfg.range_gate_sigma =
                kalman["range_gate_sigma"].as<double>(cfg.range_gate_sigma);
            cfg.range_gate_max =
                kalman["range_gate_max"].as<double>(cfg.range_gate_max);
            cfg.mahalanobis_matching =
                kalman["mahalanobis_matching"].as<bool>(cfg.mahalanobis_matching);
            cfg.association_chi2 =
                kalman["association_chi2"].as<double>(cfg.association_chi2);
            cfg.ypd_distance_sigma_coeff = kalman["ypd_distance_sigma_coeff"]
                                               .as<double>(cfg.ypd_distance_sigma_coeff);
            cfg.soft_nis_gate =
                kalman["soft_nis_gate"].as<bool>(cfg.soft_nis_gate);
            cfg.angle_match_weight =
                kalman["angle_match_weight"].as<double>(cfg.angle_match_weight);
            // PnP 平面解 yaw 的可信门限（度）；0 = 关闭门控（见 TrackerConfig 注释）。
            cfg.pnp_yaw_trust_deg =
                kalman["pnp_yaw_trust_deg"].as<double>(cfg.pnp_yaw_trust_deg);
            cfg.joint_solve = kalman["joint_solve"].as<bool>(cfg.joint_solve);
            cfg.joint_solve_max_residual =
                kalman["joint_solve_max_residual"].as<double>(cfg.joint_solve_max_residual);
            cfg.joint_solve_yaw_noise =
                kalman["joint_solve_yaw_noise"].as<double>(cfg.joint_solve_yaw_noise);
            cfg.joint_solve_center_noise =
                kalman["joint_solve_center_noise"].as<double>(cfg.joint_solve_center_noise);
            cfg.reanchor_yaw_on_plate_switch =
                kalman["reanchor_yaw_on_plate_switch"].as<bool>(
                    cfg.reanchor_yaw_on_plate_switch);
            cfg.distance_noise_scale =
                kalman["distance_noise_scale"].as<double>(cfg.distance_noise_scale);
            cfg.armor_angle_noise_base =
                kalman["armor_angle_noise_base"].as<double>(cfg.armor_angle_noise_base);
            cfg.graze_gate_deg = kalman["graze_gate_deg"].as<double>(cfg.graze_gate_deg);
            cfg.graze_noise_scale =
                kalman["graze_noise_scale"].as<double>(cfg.graze_noise_scale);
            cfg.nearest_plate_constraint =
                kalman["nearest_plate_constraint"].as<bool>(cfg.nearest_plate_constraint);
            cfg.nearest_plate_margin_m =
                kalman["nearest_plate_margin_m"].as<double>(cfg.nearest_plate_margin_m);
            cfg.geometry_learn = kalman["geometry_learn"].as<bool>(cfg.geometry_learn);
            cfg.geometry_learn_gain =
                kalman["geometry_learn_gain"].as<double>(cfg.geometry_learn_gain);
            cfg.geometry_learn_max_step =
                kalman["geometry_learn_max_step"].as<double>(cfg.geometry_learn_max_step);
            cfg.geometry_learn_max_error =
                kalman["geometry_learn_max_error"].as<double>(cfg.geometry_learn_max_error);
            cfg.geometry_learn_range =
                kalman["geometry_learn_range"].as<double>(cfg.geometry_learn_range);
            cfg.geometry_learn_max_omega =
                kalman["geometry_learn_max_omega"].as<double>(cfg.geometry_learn_max_omega);
            cfg.geometry_learn_height =
                kalman["geometry_learn_height"].as<bool>(cfg.geometry_learn_height);
            cfg.position_nis_threshold =
                kalman["position_nis_threshold"].as<double>(
                    cfg.position_nis_threshold);
            cfg.angle_nis_threshold =
                kalman["angle_nis_threshold"].as<double>(
                    cfg.angle_nis_threshold);

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
            // 装甲板安装倾角（真机 15° 支架；仿真靶 0°）。模型构造/重投影用它。
            cfg.armor_tilt_deg =
                pnp["armor_pitch_deg"].as<double>(cfg.armor_tilt_deg);
            // 逐板回转距离 + 大/小板标记（真车是**矩形**底盘：前后 ≠ 左右）。
            // 不给就用 kalman.armor_radius 当圆（仿真靶）。见 common/chassis_model.hpp。
            if (const YAML::Node dist = pnp["plate_distance"]) {
                if (dist.size() == 0 || dist.IsNull()) {
                    // 空列表 = 没配（用 kalman.armor_radius 当圆），不是错误
                } else if (dist.size() == 4) {
                    for (std::size_t i = 0; i < 4; ++i)
                        cfg.plate_distance[i] = dist[i].as<double>(0.0);
                } else {
                    std::cerr << "Ignoring pnp.plate_distance: 需要 4 个数" << std::endl;
                }
            }
            if (const YAML::Node large = pnp["plate_large"]) {
                if (large.size() == 4) {
                    for (std::size_t i = 0; i < 4; ++i)
                        cfg.plate_large[i] = large[i].as<bool>(false);
                } else {
                    std::cerr << "Ignoring pnp.plate_large: 需要 4 个 bool" << std::endl;
                }
            }
            // UV 的板高没单独配就直接跟 PnP 走：这两处必须是同一个几何量，
            // 否则"重投影的模型"和"观测的角点"不是一回事，残差会带模型偏差。
            if (!tracker["kalman"]["uv_armor_height"]) {
                cfg.uv_armor_height = cfg.armor_height;
            }
        }
        return cfg;
    }

    // `tracker_file` is the parsed tracker.yaml root node.
}

#endif // AUTO_AIM_CONFIG_TRACKER_CONFIG_LOADER_HPP
