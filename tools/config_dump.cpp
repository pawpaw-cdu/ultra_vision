// 把 configs/ 里真正**加载进结构体**的每一个值打印出来（一行一个，`名字 值`）。
//
// 为什么要有它：配置文件里写的东西和代码里的默认值这两处都可能是"真值"，
// 改配置/搬默认值时很容易改错一边，而跑起来只看行为是看不出来的
// （比如把 yaml 里的 1.5 删掉了、hpp 默认却是 0.5，仿真看着"还行"）。
// 用法：
//   config_dump > /tmp/cfg_before.txt      # 改之前
//   ...改配置/代码...
//   config_dump > /tmp/cfg_after.txt       # 改之后
//   diff /tmp/cfg_before.txt /tmp/cfg_after.txt    # 有效的值必须一行不差
//
// 默认读 <repo>/configs，可用 ULTRA_VISION_CONFIG_DIR 指定别的目录。

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>

#include <yaml-cpp/yaml.h>

#include "config_loader.hpp"

namespace
{
    std::string configDir()
    {
        if (const char* env = std::getenv("ULTRA_VISION_CONFIG_DIR")) return env;
        return "../configs";
    }

    void dumpTracker(const auto_aim::TrackerConfig& cfg)
    {
        std::cout << std::boolalpha << std::setprecision(10);
        std::cout << "tracker.min_detect_frames " << cfg.min_detect_frames << '\n';
        std::cout << "tracker.max_lost_frames " << cfg.max_lost_frames << '\n';
        std::cout << "tracker.reacquire_frames " << cfg.reacquire_frames << '\n';
        std::cout << "tracker.max_match_distance " << cfg.max_match_distance << '\n';
        std::cout << "tracker.association_margin " << cfg.association_margin << '\n';
        std::cout << "tracker.reacquire_center_error " << cfg.reacquire_center_error << '\n';
        std::cout << "tracker.reacquire_max_dt " << cfg.reacquire_max_dt << '\n';
        std::cout << "tracker.max_camera_pose_dt " << cfg.max_camera_pose_dt << '\n';
        std::cout << "tracker.omega_measure_noise " << cfg.omega_measure_noise << '\n';
        std::cout << "tracker.omega_nis_threshold " << cfg.omega_nis_threshold << '\n';
        std::cout << "tracker.process_noise_pos " << cfg.process_noise_pos << '\n';
        std::cout << "tracker.process_noise_pos_speed_ref "
                  << cfg.process_noise_pos_speed_ref << '\n';
        std::cout << "tracker.process_noise_pos_max " << cfg.process_noise_pos_max << '\n';
        std::cout << "tracker.process_noise_vel " << cfg.process_noise_vel << '\n';
        std::cout << "tracker.measure_noise " << cfg.measure_noise << '\n';
        std::cout << "tracker.default_dt " << cfg.default_dt << '\n';
        std::cout << "tracker.armor_radius " << cfg.armor_radius << '\n';
        std::cout << "tracker.y_noise_mult " << cfg.y_noise_mult << '\n';
        std::cout << "tracker.p_cov_min " << cfg.p_cov_min << '\n';
        std::cout << "tracker.uv_armor_small_width " << cfg.uv_armor_small_width << '\n';
        std::cout << "tracker.uv_armor_large_width " << cfg.uv_armor_large_width << '\n';
        std::cout << "tracker.uv_armor_height " << cfg.uv_armor_height << '\n';
        std::cout << "tracker.scale_gate_ratio " << cfg.scale_gate_ratio << '\n';
        std::cout << "tracker.armor_small_width " << cfg.armor_small_width << '\n';
        std::cout << "tracker.armor_large_width " << cfg.armor_large_width << '\n';
        std::cout << "tracker.armor_height " << cfg.armor_height << '\n';
        std::cout << "tracker.angle_measure_noise " << cfg.angle_measure_noise << '\n';
        std::cout << "tracker.bearing_measure_noise " << cfg.bearing_measure_noise << '\n';
        std::cout << "tracker.pitch_measure_noise " << cfg.pitch_measure_noise << '\n';
        std::cout << "tracker.range_noise_distance_scale "
                  << cfg.range_noise_distance_scale << '\n';
        std::cout << "tracker.ypd_nis_threshold " << cfg.ypd_nis_threshold << '\n';
        std::cout << "tracker.max_angle_error " << cfg.max_angle_error << '\n';
        std::cout << "tracker.adaptive_association_gate "
                  << cfg.adaptive_association_gate << '\n';
        std::cout << "tracker.association_gate_sigma "
                  << cfg.association_gate_sigma << '\n';
        std::cout << "tracker.association_gate_max "
                  << cfg.association_gate_max << '\n';
        std::cout << "tracker.range_gate_sigma "
                  << cfg.range_gate_sigma << '\n';
        std::cout << "tracker.range_gate_max "
                  << cfg.range_gate_max << '\n';
        std::cout << "tracker.mahalanobis_matching "
                  << cfg.mahalanobis_matching << '\n';
        std::cout << "tracker.association_chi2 "
                  << cfg.association_chi2 << '\n';
        std::cout << "tracker.ypd_distance_sigma_coeff "
                  << cfg.ypd_distance_sigma_coeff << '\n';
        std::cout << "tracker.soft_nis_gate " << cfg.soft_nis_gate << '\n';
        std::cout << "tracker.plate_model_enabled " << cfg.plate_model_enabled << '\n';
        std::cout << "tracker.plate_model_process_acc "
                  << cfg.plate_model_process_acc << '\n';
        std::cout << "tracker.plate_model_process_yaw_acc "
                  << cfg.plate_model_process_yaw_acc << '\n';
        std::cout << "tracker.plate_model_enter_omega "
                  << cfg.plate_model_enter_omega << '\n';
        std::cout << "tracker.plate_model_exit_omega "
                  << cfg.plate_model_exit_omega << '\n';
        std::cout << "tracker.angle_match_weight " << cfg.angle_match_weight << '\n';
        std::cout << "tracker.position_nis_threshold " << cfg.position_nis_threshold << '\n';
        std::cout << "tracker.angle_nis_threshold " << cfg.angle_nis_threshold << '\n';
        // 收敛看门狗（configs/tracker.yaml 的 watchdog_* 一组）
        std::cout << "tracker.watchdog_enabled " << cfg.watchdog_enabled << '\n';
        std::cout << "tracker.watchdog_window " << cfg.watchdog_window << '\n';
        std::cout << "tracker.watchdog_reject_ratio " << cfg.watchdog_reject_ratio << '\n';
        std::cout << "tracker.watchdog_min_samples " << cfg.watchdog_min_samples << '\n';
        std::cout << "tracker.watchdog_max_speed " << cfg.watchdog_max_speed << '\n';
        std::cout << "tracker.watchdog_max_yaw_rate " << cfg.watchdog_max_yaw_rate << '\n';
        std::cout << "tracker.watchdog_max_range " << cfg.watchdog_max_range << '\n';
        std::cout << "tracker.watchdog_max_position_cov " << cfg.watchdog_max_position_cov << '\n';
        std::cout << "tracker.watchdog_max_frame_gap " << cfg.watchdog_max_frame_gap << '\n';
        std::cout << "tracker.watchdog_min_converge_updates "
                  << cfg.watchdog_min_converge_updates << '\n';
        std::cout << "tracker.hard_plate_constraint " << cfg.hard_plate_constraint << '\n';
        for (std::size_t i = 0; i < cfg.armor_y_offsets.size(); ++i) {
            std::cout << "tracker.armor_y_offsets[" << i << "] "
                      << cfg.armor_y_offsets[i] << '\n';
        }
        std::cout << "tracker.rotation_rate.window " << cfg.rotation_rate.window << '\n';
        std::cout << "tracker.rotation_rate.min_span " << cfg.rotation_rate.min_span << '\n';
        std::cout << "tracker.rotation_rate.max_sample_gap "
                  << cfg.rotation_rate.max_sample_gap << '\n';
        std::cout << "tracker.rotation_rate.max_rate " << cfg.rotation_rate.max_rate << '\n';
        std::cout << "tracker.rotation_rate.min_rate " << cfg.rotation_rate.min_rate << '\n';
        std::cout << "tracker.rotation_rate.max_residual "
                  << cfg.rotation_rate.max_residual << '\n';
        std::cout << "tracker.rotation_rate.min_slopes "
                  << cfg.rotation_rate.min_slopes << '\n';
    }

    void dumpDetector(const auto_aim::DetectorConfig& cfg)
    {
        std::cout << std::boolalpha << std::setprecision(10);
        std::cout << "detector.enemy_color " << cfg.enemy_color << '\n';
        std::cout << "detector.binary_threshold " << cfg.binary_threshold << '\n';
        std::cout << "detector.light_min_ratio " << cfg.light_min_ratio << '\n';
        std::cout << "detector.light_max_ratio " << cfg.light_max_ratio << '\n';
        std::cout << "detector.light_max_angle " << cfg.light_max_angle << '\n';
        std::cout << "detector.light_min_contour_points "
                  << cfg.light_min_contour_points << '\n';
        std::cout << "detector.armor_height_ratio_min " << cfg.armor_height_ratio_min << '\n';
        std::cout << "detector.armor_height_ratio_max " << cfg.armor_height_ratio_max << '\n';
        std::cout << "detector.armor_angle_diff_max " << cfg.armor_angle_diff_max << '\n';
        std::cout << "detector.armor_width_to_height_min "
                  << cfg.armor_width_to_height_min << '\n';
        std::cout << "detector.armor_width_to_height_max "
                  << cfg.armor_width_to_height_max << '\n';
        std::cout << "detector.armor_large_ratio_thresh "
                  << cfg.armor_large_ratio_thresh << '\n';
        std::cout << "detector.color_use_detect " << cfg.color_use_detect << '\n';
        std::cout << "detector.color_red_threshold " << cfg.color_red_threshold << '\n';
        std::cout << "detector.color_blue_threshold " << cfg.color_blue_threshold << '\n';
        std::cout << "detector.use_hsv " << cfg.use_hsv << '\n';
        std::cout << "detector.hue_red_low1 " << cfg.hue_red_low1 << '\n';
        std::cout << "detector.hue_red_high1 " << cfg.hue_red_high1 << '\n';
        std::cout << "detector.hue_red_low2 " << cfg.hue_red_low2 << '\n';
        std::cout << "detector.hue_red_high2 " << cfg.hue_red_high2 << '\n';
        std::cout << "detector.hue_blue_low " << cfg.hue_blue_low << '\n';
        std::cout << "detector.hue_blue_high " << cfg.hue_blue_high << '\n';
        std::cout << "detector.sat_min " << cfg.sat_min << '\n';
        std::cout << "detector.val_min " << cfg.val_min << '\n';
        std::cout << "detector.color_area_ratio " << cfg.color_area_ratio << '\n';
    }

    void dumpControl(const YAML::Node& tracker_file)
    {
        const auto_aim::GimbalAimConfig gimbal =
            auto_aim::loadGimbalConfig(tracker_file);
        const auto_aim::TargetSelectorConfig selector =
            auto_aim::loadSelectorConfig(tracker_file);
        const auto_aim::ShooterConfig shooter =
            auto_aim::loadShooterConfig(tracker_file);
        const auto_aim::AimSignalFilterConfig aim =
            auto_aim::loadAimFilterConfig(tracker_file);
        std::cout << std::setprecision(10);
        std::cout << "gimbal.pitch_min " << gimbal.pitch_min << '\n';
        std::cout << "gimbal.pitch_max " << gimbal.pitch_max << '\n';
        std::cout << "gimbal.max_yaw_velocity " << gimbal.max_yaw_velocity << '\n';
        std::cout << "gimbal.max_pitch_velocity " << gimbal.max_pitch_velocity << '\n';
        std::cout << "gimbal.max_yaw_acceleration " << gimbal.max_yaw_acceleration << '\n';
        std::cout << "gimbal.max_pitch_acceleration "
                  << gimbal.max_pitch_acceleration << '\n';
        std::cout << "gimbal.max_yaw_jerk " << gimbal.max_yaw_jerk << '\n';
        std::cout << "gimbal.max_pitch_jerk " << gimbal.max_pitch_jerk << '\n';
        std::cout << "gimbal.yaw_response_gain " << gimbal.yaw_response_gain << '\n';
        std::cout << "gimbal.pitch_response_gain " << gimbal.pitch_response_gain << '\n';
        std::cout << "gimbal.feedforward_gain " << gimbal.feedforward_gain << '\n';
        std::cout << "gimbal.feedforward_time_constant "
                  << gimbal.feedforward_time_constant << '\n';
        std::cout << "gimbal.settle_angle " << gimbal.settle_angle << '\n';
        std::cout << "gimbal.settle_velocity " << gimbal.settle_velocity << '\n';
        std::cout << "gimbal.max_dt " << gimbal.max_dt << '\n';
        std::cout << "gimbal.command_rate_hz "
                  << auto_aim::loadGimbalCommandRateHz(tracker_file) << '\n';
        std::cout << "selector.enabled " << selector.enabled << '\n';
        std::cout << "selector.projectile_speed " << selector.projectile_speed << '\n';
        std::cout << "selector.gravity " << selector.gravity << '\n';
        std::cout << "selector.command_latency " << selector.command_latency << '\n';
        std::cout << "selector.max_lead_time " << selector.max_lead_time << '\n';
        std::cout << "selector.spin_omega_threshold "
                  << selector.spin_omega_threshold << '\n';
        std::cout << "selector.chase_omega_threshold "
                  << selector.chase_omega_threshold << '\n';
        std::cout << "selector.spin_hold_enabled "
                  << selector.spin_hold_enabled << '\n';
        std::cout << "selector.no_valid_armor_policy "
                  << auto_aim::loadNoValidArmorPolicy(tracker_file) << '\n';
        std::cout << "selector.lead_velocity_scale "
                  << selector.lead_velocity_scale << '\n';
        std::cout << "selector.spin_velocity_scale "
                  << selector.spin_velocity_scale << '\n';
        std::cout << "selector.spin_velocity_gate_omega "
                  << selector.spin_velocity_gate_omega << '\n';
        std::cout << "selector.spin_fire_tolerance "
                  << selector.spin_fire_tolerance << '\n';
        std::cout << "selector.coming_angle " << selector.coming_angle << '\n';
        std::cout << "selector.leaving_angle " << selector.leaving_angle << '\n';
        std::cout << "selector.yaw_velocity " << selector.yaw_velocity << '\n';
        std::cout << "selector.pitch_velocity " << selector.pitch_velocity << '\n';
        std::cout << "selector.yaw_acceleration " << selector.yaw_acceleration << '\n';
        std::cout << "selector.pitch_acceleration "
                  << selector.pitch_acceleration << '\n';
        std::cout << "selector.handoff_start_angle "
                  << selector.handoff_start_angle << '\n';
        std::cout << "selector.handoff_duration " << selector.handoff_duration << '\n';
        std::cout << "selector.decide_speed " << selector.decide_speed << '\n';
        std::cout << "selector.low_speed_latency " << selector.low_speed_latency << '\n';
        std::cout << "selector.high_speed_latency " << selector.high_speed_latency << '\n';
        std::cout << "selector.lock_switch_margin " << selector.lock_switch_margin << '\n';
        std::cout << "selector.motion_gate_enabled " << selector.motion_gate_enabled << '\n';
        std::cout << "selector.motion_gate_enter_speed "
                  << selector.motion_gate_enter_speed << '\n';
        std::cout << "selector.motion_gate_exit_speed "
                  << selector.motion_gate_exit_speed << '\n';
        std::cout << "selector.motion_gate_enter_omega "
                  << selector.motion_gate_enter_omega << '\n';
        std::cout << "selector.motion_gate_exit_omega "
                  << selector.motion_gate_exit_omega << '\n';
        std::cout << "selector.max_control_lost_frames "
                  << auto_aim::loadMaxControlLostFrames(tracker_file) << '\n';
        std::cout << "selector.fire_angle_tolerance_degrees "
                  << auto_aim::loadFireAngleToleranceDegrees(tracker_file) << '\n';
        std::cout << "shooter.min_fire_interval " << shooter.min_fire_interval << '\n';
        std::cout << "shooter.ready_handoff_gain " << shooter.ready_handoff_gain << '\n';
        std::cout << "shooter.end_handoff_gain " << shooter.end_handoff_gain << '\n';
        std::cout << "aim_filter.enabled " << aim.enabled << '\n';
        std::cout << "aim_filter.process_noise_acceleration "
                  << aim.process_noise_acceleration << '\n';
        std::cout << "aim_filter.measurement_noise " << aim.measurement_noise << '\n';
        std::cout << "aim_filter.reset_innovation " << aim.reset_innovation << '\n';
        std::cout << "aim_filter.reset_timeout " << aim.reset_timeout << '\n';
        std::cout << "aim_filter.max_dt " << aim.max_dt << '\n';
    }

#ifdef ULTRA_VISION_USE_OPENVINO
    void dumpNeural(const auto_aim::NnDetectorConfig& cfg)
    {
        std::cout << std::boolalpha << std::setprecision(10);
        std::cout << "neural.model_path " << cfg.model_path << '\n';
        std::cout << "neural.device " << cfg.device << '\n';
        std::cout << "neural.output_format " << cfg.output_format << '\n';
        std::cout << "neural.input_size " << cfg.input_size << '\n';
        std::cout << "neural.score_threshold " << cfg.score_threshold << '\n';
        std::cout << "neural.nms_threshold " << cfg.nms_threshold << '\n';
        std::cout << "neural.min_confidence " << cfg.min_confidence << '\n';
        std::cout << "neural.filter_by_color " << cfg.filter_by_color << '\n';
        std::cout << "neural.enemy_color " << cfg.enemy_color << '\n';
        std::cout << "neural.performance_mode " << cfg.performance_mode << '\n';
        std::cout << "neural.num_threads " << cfg.num_threads << '\n';
        std::cout << "neural.num_streams " << cfg.num_streams << '\n';
        std::cout << "neural.refiner.enabled " << cfg.refiner.enabled << '\n';
        std::cout << "neural.refiner.binary_threshold "
                  << cfg.refiner.binary_threshold << '\n';
        std::cout << "neural.refiner.margin_ratio " << cfg.refiner.margin_ratio << '\n';
        std::cout << "neural.refiner.max_center_shift_ratio "
                  << cfg.refiner.max_center_shift_ratio << '\n';
        std::cout << "neural.use_roi " << cfg.use_roi << '\n';
        std::cout << "neural.roi.x " << cfg.roi.x << '\n';
        std::cout << "neural.roi.y " << cfg.roi.y << '\n';
        std::cout << "neural.roi.width " << cfg.roi.width << '\n';
        std::cout << "neural.roi.height " << cfg.roi.height << '\n';
    }
#endif

    void dumpPnp(const auto_aim::PnpGeometry& geometry)
    {
        std::cout << std::setprecision(10);
        std::cout << "pnp.small_width " << geometry.small_width << '\n';
        std::cout << "pnp.large_width " << geometry.large_width << '\n';
        std::cout << "pnp.height " << geometry.height << '\n';
        std::cout << "pnp.yaw_refine " << geometry.yaw_refine << '\n';
        std::cout << "pnp.yaw_refine_span_deg "
                  << geometry.yaw_refine_span_deg << '\n';
        std::cout << "pnp.yaw_refine_coarse_deg "
                  << geometry.yaw_refine_coarse_deg << '\n';
        std::cout << "pnp.armor_pitch_deg " << geometry.armor_pitch_deg << '\n';
    }

    void dumpCamera(const YAML::Node& camera_file)
    {
        const YAML::Node camera = camera_file["camera"];
        std::cout << std::setprecision(10);
        std::cout << "camera.name " << camera["name"].as<std::string>("galaxy") << '\n';
        std::cout << "camera.device_index " << camera["device_index"].as<int>(1) << '\n';
        std::cout << "camera.timeout_ms " << camera["timeout_ms"].as<int>(1000) << '\n';
        if (const YAML::Node sim = camera_file["simulator"]) {
            std::cout << "simulator.fov_degrees " << sim["fov_degrees"].as<double>(45.0)
                      << '\n';
            std::cout << "simulator.max_processing_width "
                      << sim["max_processing_width"].as<int>(960) << '\n';
        }
        for (const char* name : {"simulator", "galaxy", "hikcamera"}) {
            if (!camera_file[name]) continue;
            const cv::Mat matrix =
                auto_aim::readMatFromYaml(camera_file[name]["camera_matrix"]);
            for (int row = 0; row < matrix.rows; ++row) {
                for (int column = 0; column < matrix.cols; ++column) {
                    std::cout << name << ".camera_matrix[" << row << "," << column
                              << "] " << matrix.at<double>(row, column) << '\n';
                }
            }
        }
    }
} // namespace

int main()
{
    const std::string dir = configDir();
    try {
        const YAML::Node tracker_file = YAML::LoadFile(dir + "/tracker.yaml");
        const YAML::Node detector_file = YAML::LoadFile(dir + "/detector.yaml");
        const YAML::Node camera_file = YAML::LoadFile(dir + "/camera.yaml");
        dumpTracker(auto_aim::loadTrackerConfig(tracker_file));
        dumpControl(tracker_file);
        dumpDetector(auto_aim::loadDetectorConfig(detector_file));
        dumpPnp(auto_aim::loadPnpGeometry(tracker_file));
        dumpCamera(camera_file);
        // 串口/下发符号（真机闭环符号靠这两项，见 docs/auto_aim_aim_loop.md §9）
        {
            const YAML::Node serial_file = YAML::LoadFile(dir + "/serial.yaml");
            const io::SerialConfig serial = auto_aim::loadSerialConfig(serial_file);
            const YAML::Node node = serial_file["serial"] ? serial_file["serial"] : serial_file;
            std::cout << "serial.yaw_sign（回传） "
                      << (node["yaw_sign"].as<double>(1.0) < 0.0 ? -1 : 1) << '\n';
            std::cout << "serial.command_frame（下发坐标系） "
                      << (serial.command_frame == io::SerialConfig::CommandFrame::World ? "world"
                                                                                        : "motor")
                      << '\n';
            std::cout << "serial.command_yaw_sign（motor 模式） " << serial.command_yaw_sign << '\n';
            std::cout << "serial.command_pitch_sign（motor 模式） " << serial.command_pitch_sign << '\n';
            std::cout << "serial.command_world_yaw_sign（world 模式） "
                      << serial.command_world_yaw_sign << '\n';
            std::cout << "serial.command_world_pitch_sign（pitch 通道） "
                      << serial.command_world_pitch_sign << '\n';
        }
#ifdef ULTRA_VISION_USE_OPENVINO
        dumpNeural(auto_aim::loadNeuralDetectorConfig(detector_file, dir));
#endif
    } catch (const std::exception& error) {
        std::cerr << "config_dump: " << error.what() << std::endl;
        return 1;
    }
    return 0;
}
