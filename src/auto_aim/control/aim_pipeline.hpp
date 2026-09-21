#ifndef AUTO_AIM_CONTROL_AIM_PIPELINE_HPP
#define AUTO_AIM_CONTROL_AIM_PIPELINE_HPP

// 瞄准流水线：**真机与仿真共用的"估计 → 选板 → 提前量 → 云台 → 开火"整条链**。
//
// 抽出来的原因：这条链原来在 node_sim.cpp 里写了一遍（100 多行），真机 node.cpp
// 又得再写一遍 —— 于是识别/控制上的每一个修复都要抄两次，迟早抄漏（比如反跑飞
// 守卫就是只加在了仿真那边）。现在两个入口只负责"喂帧 + 显示 + 各自的时间基"，
// 决策全在这里，行为天然一致。
//
// 输入：一帧的观测（已 PnP）+ 观测时刻 + 该时刻的相机姿态 + 弹速 + 是否允许控制。
// 输出：目标决策、是否瞄稳、是否开火、瞄准误差、以及"守卫是否按住了云台"。
//
// 不含：IO（串口/TCP 由调用方在 SendFn 里接）、显示、CSV（调用方按需记录）。

#include <atomic>
#include <cstdint>
#include <chrono>
#include <functional>
#include <vector>

#include "control/aim_signal_filter.hpp"
#include "control/gimbal_aimer.hpp"
#include "control/gimbal_controller.hpp"
#include "control/shooter.hpp"
#include "control/target_selector.hpp"
#include "Kalman/tracker.hpp"

namespace auto_aim
{
    struct AimPipelineConfig
    {
        TrackerConfig tracker;
        TargetSelectorConfig selector;
        GimbalAimConfig gimbal;
        ShooterConfig shooter;
        AimSignalFilterConfig aim_filter;
        double command_rate_hz = 100.0;
        int max_control_lost_frames = 5;
        double fire_angle_tolerance_deg = 3.0;
        /// 反"云台跑飞"：瞄点相对当前下发角跳变超过该角度就按住不甩（0 = 关）。
        double aim_jump_limit_deg = 30.0;
        /// 没有可用目标时的心跳频率（真机链路必须有：下位机靠收包超时区分
        /// "视觉活着但没目标"和"视觉掉线"；仿真链路不需要，填 0 关掉）。
        double heartbeat_hz = 0.0;
    };

    class AimPipeline
    {
    public:
        /// 真机：把指令写进串口帧；仿真：把指令发给仿真器。返回值只用于统计。
        using SendFn = std::function<bool(bool control, bool fire,
                                          double yaw, double yaw_vel, double yaw_acc,
                                          double pitch, double pitch_vel, double pitch_acc)>;

        AimPipeline(const AimPipelineConfig& config, SendFn send);

        struct FrameInput
        {
            std::vector<Armor> armors;       // 本帧观测（已 PnP）
            /// 本帧有没有**新**观测。异步检测器（神经网络）在推理没跟上时会给 false：
            /// 这时估计器保持状态、选择器继续按 age 外推 —— 但控制/显示照常跑。
            bool fresh = true;
            double observation_time = 0.0;   // 观测时刻（fresh 时用于估计器时间基）
            double now = 0.0;                // 控制侧当前时刻（每帧都更新，用于观测年龄与开火判据）
            CameraPose pose;                 // 观测时刻的相机姿态（反旋到世界/底盘系用）
            double bullet_speed = 0.0;       // <=0 表示用配置里的值
            bool allow_control = true;       // 下位机是否把控制权交给我们
            bool fire_enabled = true;
        };

        struct Outcome
        {
            TargetDecision decision;
            bool track_usable = false;
            bool aim_ready = false;
            bool fire = false;
            bool fired_now = false;          // 上升沿（真机用来打日志）
            bool held_for_jump = false;      // 反跑飞守卫按住了这一帧
            double aim_yaw_error = 0.0;
            double aim_pitch_error = 0.0;
        };

        Outcome update(const FrameInput& input);

        Tracker& tracker() { return tracker_; }
        GimbalController& controller() { return controller_; }
        /// 决策结果（供显示/遥测读取）
        const Outcome& lastOutcome() const { return last_outcome_; }
        const Tracker& tracker() const { return tracker_; }
        /// 当前用的弹速（实车由下位机回传；只接受合理值）
        double projectileSpeed() const { return selector_.projectileSpeed(); }

    private:
        AimPipelineConfig config_;
        SendFn send_;
        Tracker tracker_;
        TargetSelector selector_;
        GimbalAimer bootstrap_aimer_;
        AimSignalFilter aim_filter_;
        GimbalController controller_;
        Shooter shooter_;
        // 控制线程（100 Hz）读的原子量：主循环只写
        std::atomic<bool> control_enabled_{false};
        std::atomic<bool> fire_request_{false};
        std::atomic<double> desired_yaw_vel_{0.0};
        std::atomic<double> desired_pitch_vel_{0.0};
        Outcome last_outcome_;
        double last_detection_time_ = 0.0;
        double aim_jump_limit_ = 1e9;
        bool last_fire_ = false;
        std::chrono::steady_clock::time_point last_heartbeat_;
    };
} // namespace auto_aim

#endif // AUTO_AIM_CONTROL_AIM_PIPELINE_HPP
