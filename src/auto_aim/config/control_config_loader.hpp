#ifndef AUTO_AIM_CONFIG_CONTROL_CONFIG_LOADER_HPP
#define AUTO_AIM_CONFIG_CONTROL_CONFIG_LOADER_HPP

// 从 config_loader.hpp 抽出的控制层配置解析（云台/选择器/火控/滤波）。

#include <algorithm>
#include <array>

#include <yaml-cpp/yaml.h>

#include "control/aim_signal_filter.hpp"
#include "control/gimbal_aimer.hpp"
#include "control/shooter.hpp"
#include "control/target_selector.hpp"

namespace auto_aim
{
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
        cfg.chase_omega_threshold =
            selector["chase_omega_threshold"].as<double>(cfg.chase_omega_threshold);
        cfg.spin_hold_enabled =
            selector["spin_hold_enabled"].as<bool>(cfg.spin_hold_enabled);
        if (const YAML::Node tolerance = selector["spin_fire_tolerance_degrees"]) {
            cfg.spin_fire_tolerance =
                tolerance.as<double>(cfg.spin_fire_tolerance * 180.0 / 3.14159265358979323846) *
                3.14159265358979323846 / 180.0;
        }
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
        cfg.lead_velocity_scale =
            selector["lead_velocity_scale"].as<double>(cfg.lead_velocity_scale);
        cfg.spin_velocity_scale =
            selector["spin_velocity_scale"].as<double>(cfg.spin_velocity_scale);
        cfg.spin_velocity_gate_omega =
            selector["spin_velocity_gate_omega"].as<double>(cfg.spin_velocity_gate_omega);
        cfg.lock_switch_margin =
            selector["lock_switch_margin"].as<double>(cfg.lock_switch_margin);
        cfg.motion_gate_enabled =
            selector["motion_gate_enabled"].as<bool>(cfg.motion_gate_enabled);
        cfg.motion_gate_enter_speed =
            selector["motion_gate_enter_speed"].as<double>(cfg.motion_gate_enter_speed);
        cfg.motion_gate_exit_speed =
            selector["motion_gate_exit_speed"].as<double>(cfg.motion_gate_exit_speed);
        cfg.motion_gate_enter_omega =
            selector["motion_gate_enter_omega"].as<double>(cfg.motion_gate_enter_omega);
        cfg.motion_gate_exit_omega =
            selector["motion_gate_exit_omega"].as<double>(cfg.motion_gate_exit_omega);
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

    /// 没合格板时的兜底策略（0=瞄最后观测到的板，1=放弃/保持上一拍瞄点）。
    /// 见 AimPipelineConfig::no_valid_armor_policy。
    inline int loadNoValidArmorPolicy(const YAML::Node& tracker_file)
    {
        const YAML::Node selector = tracker_file["tracker"]["selector"]
            ? tracker_file["tracker"]["selector"]
            : tracker_file["tracker"]["predictive_aim"];
        if (!selector) return 0;
        const int policy = selector["no_valid_armor_policy"].as<int>(0);
        return policy != 0 ? 1 : 0;
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
}

#endif // AUTO_AIM_CONFIG_CONTROL_CONFIG_LOADER_HPP
