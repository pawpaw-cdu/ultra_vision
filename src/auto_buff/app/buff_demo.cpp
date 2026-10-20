// 能量机关（大小符）入口：只负责"参数/环境 → 配置 → 跑 runner"。

#include <fstream>
#include <iostream>
#include <string>

#include "auto_buff/config/rune_config.hpp"
#include "auto_buff/app/rune_runner.hpp"

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
