#pragma once
//
// Harness 专用的 json.hpp —— 影子头文件，优先于 RP-26Rune-main 自带的两个
// 同名头文件（src/core/algorithm/power_rune/config/json.hpp 与
// src/app_plugin/detector/config/json.hpp）被包含。
//
// 为什么不直接用他们那份：
//   * 他们用 std::filesystem::canonical("/proc/self/exe") 定位配置目录，macOS 上
//     这个路径不存在，canonical 会抛异常（静态初始化期直接 terminate）。
//   * 他们的 J_DETECT 用 __FILE__ 定位 detect.json，换目录就失效。
// 这里只改"配置从哪来"，配置的语义与他们的完全一致：
//   J_POWER_RUNE -> <config_dir>/power_rune.json （能量机关算法）
//   J_DETECT     -> <config_dir>/rp26_sim.json   （五点模型，格式同 detect.json）

#include <cstdlib>
#include <filesystem>
#include <string>

#include "json/ReJson.hpp"

namespace rp26_sim
{
/// @brief harness 配置目录：优先环境变量，其次编译期给的默认目录（源码树里的 config）。
inline const std::filesystem::path &config_dir()
{
    static const std::filesystem::path dir = [] {
        if (const char *env = std::getenv("RP26_CONFIG_DIR"))
            return std::filesystem::path(env);
        // 他们原来用 std::filesystem::canonical("/proc/self/exe") 定位 <root>/config，
        // 这在 macOS 上不可用；harness 改成编译期常量 + 环境变量覆盖。
        return std::filesystem::path(RP26_DEFAULT_CONFIG_DIR);
    }();
    return dir;
}
} // namespace rp26_sim

/// @brief 能量机关算法配置（与 RP-26Rune 的 J_POWER_RUNE 同义）。
inline ReJson J_POWER_RUNE((rp26_sim::config_dir() / "power_rune.json").string());

/// @brief 五点检测模型配置（与 RP-26Rune 的 J_DETECT 同义，格式同 detect.json）。
inline const std::filesystem::path DETECTOR_CONFIG_DIR = rp26_sim::config_dir();
inline ReJson J_DETECT((DETECTOR_CONFIG_DIR / "rp26_sim.json").string());
