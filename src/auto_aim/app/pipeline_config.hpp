#ifndef AUTO_AIM_APP_PIPELINE_CONFIG_HPP
#define AUTO_AIM_APP_PIPELINE_CONFIG_HPP

// 组合层：把各层的配置对象拼成 AimPipelineConfig。
// 真机入口从 YAML 现读，仿真入口用已经读好的对象；拼装字段完全一样，
// 所以统一到这里，两个入口各留一行调用。

#include "control/aim_pipeline.hpp"

namespace auto_aim
{
    namespace app
    {
        inline AimPipelineConfig makeAimPipelineConfig(
            const TrackerConfig& tracker,
            const TargetSelectorConfig& selector,
            const GimbalAimConfig& gimbal,
            const ShooterConfig& shooter,
            const AimSignalFilterConfig& aim_filter,
            double command_rate_hz,
            int max_control_lost_frames,
            double fire_angle_tolerance_deg,
            double heartbeat_hz)
        {
            AimPipelineConfig config;
            config.tracker = tracker;
            config.selector = selector;
            config.gimbal = gimbal;
            config.shooter = shooter;
            config.aim_filter = aim_filter;
            config.command_rate_hz = command_rate_hz;
            config.max_control_lost_frames = max_control_lost_frames;
            config.fire_angle_tolerance_deg = fire_angle_tolerance_deg;
            config.heartbeat_hz = heartbeat_hz;
            return config;
        }
    }
}

#endif // AUTO_AIM_APP_PIPELINE_CONFIG_HPP
