// 能量机关（大小符）入口：只负责"参数/环境 → 配置 → 跑 runner"。
//
// 分层（对照 sp_vision 的 tasks/auto_buff + 独立小入口）：
//   rune_node.cpp   —— 本文件：参数解析、配置加载、调用 runner；
//   rune_runner.cpp —— 主循环：收帧 → 感知 → 估计 → 瞄准发布 → 开火闸门（控制侧
//                      由 control/GimbalController 的 100 Hz 线程执行）；
//   rune_diag.cpp   —— 诊断层：CSV/遥测真值/录帧/静态试射/汇总打印；
//   rune_detector/solver/target/aimer/model/refiner —— 算法本体。
//
// 常用环境变量（详见 README 与 docs/energy_rune_issue_audit.md）：
//   ULTRA_VISION_CONFIG_DIR          配置目录（默认 configs）
//   ULTRA_VISION_RUNE_MODE           small|large，覆盖配置里的 mode
//   ULTRA_VISION_RUNE_VIDEO          用视频回放代替仿真器
//   ULTRA_VISION_RUNE_TEST_SECONDS   跑多少秒后退出
//   ULTRA_VISION_RUNE_CSV            逐帧诊断 CSV 输出路径
//   ULTRA_VISION_RUNE_GT_CSV         仿真真值遥测原文输出路径
//   ULTRA_VISION_RUNE_AIM_IN_CONTROL 0 = 关掉"瞄准解算在控制线程"（默认开）
//   ULTRA_VISION_RUNE_BORESIGHT      "yaw_deg,pitch_deg" 静态试射模式
//   ULTRA_VISION_RUNE_DIFF           1 = 打开帧差探针（诊断列 diff_*）
//   ULTRA_VISION_RUNE_NO_DIAG        1 = 关掉诊断层（不写 CSV/不落帧）

#include <fstream>
#include <iostream>
#include <string>

#include "auto_buff/rune_config.hpp"
#include "auto_buff/rune_runner.hpp"

#ifndef ULTRA_VISION_CONFIG_DIR
#define ULTRA_VISION_CONFIG_DIR "configs"
#endif

int main(int argc, char* argv[])
{
    std::string config_dir = ULTRA_VISION_CONFIG_DIR;
    if (argc > 1) config_dir = argv[1];
    if (const char* env = std::getenv("ULTRA_VISION_CONFIG_DIR")) config_dir = env;

    std::string video_path;
    if (argc > 2) video_path = argv[2];
    if (const char* env = std::getenv("ULTRA_VISION_RUNE_VIDEO")) video_path = env;

    auto_aim::energy::RuneConfig config;
    try {
        config = auto_aim::energy::loadRuneConfig(config_dir + "/buff.yaml");
    } catch (const std::exception& error) {
        std::cerr << "Failed to load rune config: " << error.what() << std::endl;
        return -1;
    }
    if (const char* env = std::getenv("ULTRA_VISION_RUNE_MODE")) {
        config.mode = (std::string(env) == "large") ? auto_aim::energy::RuneMode::Large
                                                   : auto_aim::energy::RuneMode::Small;
    }

    // 相对模型路径按"配置目录的上一级"解析，这样在任意工作目录下都能找到模型。
    if (!config.detector.model.model_path.empty() &&
        config.detector.model.model_path.front() != '/') {
        const std::string candidate = config_dir + "/../" + config.detector.model.model_path;
        if (std::ifstream(candidate).good()) {
            config.detector.model.model_path = candidate;
        }
    }

    return auto_aim::energy::runRune(config, config_dir, video_path);
}
