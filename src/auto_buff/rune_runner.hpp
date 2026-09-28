#ifndef AUTO_AIM_ENERGY_RUNE_RUNNER_HPP
#define AUTO_AIM_ENERGY_RUNE_RUNNER_HPP

// 能量机关主循环（收帧 → 感知 → 估计 → 瞄准发布 → 开火闸门）。
// 诊断/落盘在 rune_diag，控制下发在 control/GimbalController（100 Hz 线程）。

#include <string>

#include "auto_buff/rune_config.hpp"

namespace auto_aim::energy
{
    /// @brief 跑一轮能量机关链路；video_path 非空时用视频回放代替仿真器。
    /// @return 进程退出码（0 正常）。
    int runRune(const RuneConfig& config, const std::string& config_dir,
                const std::string& video_path);
} // namespace auto_aim::energy

#endif // AUTO_AIM_ENERGY_RUNE_RUNNER_HPP
