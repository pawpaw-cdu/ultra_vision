#ifndef AUTO_AIM_APP_CONFIG_PATHS_HPP
#define AUTO_AIM_APP_CONFIG_PATHS_HPP

// 组合层（app/）的第一块：配置目录解析 + 路径拼接。
// 真机入口 node.cpp 与仿真入口 node_sim.cpp 原本各自抄了一遍同样的 3 行解析
// 和 config_path lambda；抽到这里，入口只留 IO 差异。

#include <cstdlib>
#include <string>

namespace auto_aim
{
    namespace app
    {
        /// @brief 配置目录：命令行参数 > 环境变量 > 编译期默认（ULTRA_VISION_CONFIG_DIR）。
        inline std::string resolveConfigDir(int argc, char** argv)
        {
            std::string dir = ULTRA_VISION_CONFIG_DIR;
            if (argc > 1) dir = argv[1];
            if (const char* env = std::getenv("ULTRA_VISION_CONFIG_DIR")) dir = env;
            return dir;
        }

        inline std::string configPath(const std::string& config_dir,
                                      const std::string& name)
        {
            return config_dir + "/" + name;
        }
    }
}

#endif // AUTO_AIM_APP_CONFIG_PATHS_HPP
