#ifndef AUTO_AIM_ENERGY_RUNE_AIM_BRIDGE_HPP
#define AUTO_AIM_ENERGY_RUNE_AIM_BRIDGE_HPP

// 瞄准桥：帧循环（~20-30 Hz）与云台控制线程（100 Hz）之间的接口。
//
// 结构照搬 sp_vision 的 multithread/commandgener：
//   * 帧循环只 **发布** 最新目标状态（RuneTarget 副本 + 该帧时刻 + 平滑延迟 +
//     帧级闸门），不直接算瞄准；
//   * 控制线程每个周期调用本类的 provider：用"当前时刻"重算瞄准/弹道，
//     并把指令回写（command + 开火请求）供帧循环记录与发火。
//
// 这样拆的好处：瞄准频率、开火判据都在一个明确的类里（可单测）；帧循环只管
// 感知与状态机；谁在什么时候能开火有唯一的判定点。

#include <atomic>
#include <functional>
#include <mutex>
#include <optional>

#include <opencv2/core.hpp>

#include "control/aim_signal_filter.hpp"
#include "control/gimbal_controller.hpp"
#include "auto_buff/rune_aimer.hpp"
#include "auto_buff/rune_config.hpp"
#include "auto_buff/rune_target.hpp"

namespace auto_aim::energy
{
    class RuneAimBridge
    {
    public:
        /// @brief 帧循环每帧发布的状态。
        struct Input
        {
            /// 构造一个"空"输入（target 是占位估计器，valid=false）。
            Input(RuneMode mode, int max_coast_frames, double max_distance_jump_ratio,
                  double max_center_jump_deg)
                : target(mode, max_coast_frames, max_distance_jump_ratio, max_center_jump_deg)
            {
            }

            RuneTarget target;          // 当前状态估计（拷贝）
            double frame_time = 0.0;    // 该帧曝光时刻（StandardClock 秒）
            double latency = -1.0;      // 平滑后的帧→瞄准延迟；<0 表示由瞄准器自算
            bool valid = false;         // 估计器可用（可以解算瞄准）
            bool center_only = false;   // 观测过期 → 只瞄符心（不外推扇叶）
            bool engageable = false;    // 帧级闸门：锁定的那片现在是可打的（class 0）
            bool hold = false;          // 命中后保持（换靶确认前不打）
            // 瞄"跟踪片往前 k 个槽位"的那片（命中后按 sp_vision 的判据转到新亮的片）。
            int slot_offset = 0;
            bool park = false;          // 没有可用目标 → 停在给定角
            double park_yaw = 0.0;
            double park_pitch = 0.0;
        };

        RuneAimBridge(const RuneConfig& config, RuneAimer& aimer);

        /// @brief 把 provider 注册到控制器（安装后 setTargetAngles 不再生效）。
        ///        fire 由控制线程在"瞄准器许可 + 帧级闸门 + 云台到位"同时满足时调用。
        void install(GimbalController& controller, AimSignalFilter& filter, bool fire_enabled,
                     std::function<void()> fire);

        /// @brief 帧循环发布最新状态（线程安全）。
        void publish(const Input& input);

        /// @brief 帧循环取回控制线程最近一次算出的指令（用于日志/发火）。
        RuneCommand lastCommand() const;

        /// @brief 取走并清除"控制线程判定的开火请求"。
        bool consumeFireRequest();

        /// @brief 丢弃当前发布的目标（例如回标定位姿/重置时）。
        void clear();

    private:
        bool provide(GimbalTargetAngles& out, double now_seconds);

        struct Shared
        {
            std::optional<Input> input;
            RuneCommand command;
            bool command_valid = false;
            bool shoot_latched = false;
            bool fire_request = false;
        };

        const RuneConfig& config_;
        RuneAimer& aimer_;
        RuneTarget aim_buffer_;
        mutable std::mutex mutex_;
        Shared shared_;
        GimbalController* controller_ = nullptr;
        AimSignalFilter* filter_ = nullptr;
        bool fire_enabled_ = true;
        std::function<void()> fire_;
    };
} // namespace auto_aim::energy

#endif // AUTO_AIM_ENERGY_RUNE_AIM_BRIDGE_HPP
