#ifndef AUTO_AIM_APP_CSV_SINK_HPP
#define AUTO_AIM_APP_CSV_SINK_HPP

// 组合层：CSV 记录器。环境变量给路径 + 打开 + 写表头，三件事原本在两个入口里
// 各抄了一遍；抽成一个类后，入口只写"路径环境变量 + 表头"这一行声明。
//
// 用法（保持原来的流式写法，调用点只多一个 .stream()）：
//   app::CsvSink rec("ULTRA_VISION_ESTIMATE_CSV", "x,y,z,...\n");
//   if (rec && ...) rec.stream() << ... << '\n';

#include <cstdlib>
#include <fstream>
#include <iostream>

namespace auto_aim
{
    namespace app
    {
        class CsvSink
        {
        public:
            /// @param env_name 环境变量名；未设置 → 不打开（记录关闭，行为与旧代码一致）。
            /// @param header   表头（含结尾换行）。
            CsvSink(const char* env_name, const char* header)
            {
                const char* path = std::getenv(env_name);
                if (path == nullptr) return;
                out_.open(path, std::ios::out | std::ios::trunc);
                if (out_) {
                    out_ << header;
                } else {
                    std::cerr << "Failed to open CSV: " << path << std::endl;
                }
            }

            std::ofstream& stream() { return out_; }
            explicit operator bool() const { return static_cast<bool>(out_); }

        private:
            std::ofstream out_;
        };
    }
}

#endif // AUTO_AIM_APP_CSV_SINK_HPP
