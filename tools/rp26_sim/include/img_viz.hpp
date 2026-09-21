#pragma once
//
// 影子头文件：替换 RP-26Rune 的 HighGUI 可视化管理器（src/core/utility/img_viz）。
// 仿真 harness 没有窗口，但保留"落盘"能力：设置 RP26_DUMP_DIR 后，
// 他们代码里每个命名窗口的图像会按 RP26_DUMP_STRIDE（默认 10 帧）写一份 PNG，
// 这样"网络结果 -> 二值图 -> 轮廓 -> 重投影"每一级都能直接看，
// 方便定位他们的传统链路在哪一级失败。

#include <cstdlib>
#include <map>
#include <string>
#include <string_view>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include "rp26_dump_context.hpp"

class ImgViz
{
public:
    static void init(bool) {}
    [[nodiscard]] static bool enabled() { return false; }
    static void enqueue_image_copy(std::string_view name, const cv::Mat &image)
    {
        dump(name, image);
    }
    static void enqueue_image_zero_copy(std::string_view name, const cv::Mat &image)
    {
        dump(name, image);
    }

private:
    static void dump(std::string_view name, const cv::Mat &image)
    {
        const char *dir = std::getenv("RP26_DUMP_DIR");
        if (dir == nullptr || dir[0] == '\0' || image.empty())
            return;

        // 只按固定节奏抽样，避免一次测试写出几万个文件。
        static int stride = [] {
            const char *value = std::getenv("RP26_DUMP_STRIDE");
            const int parsed = value != nullptr ? std::atoi(value) : 0;
            return parsed > 0 ? parsed : 10;
        }();

        std::string safe_name(name);
        for (char &character : safe_name)
            if (character == '/' || character == ' ')
                character = '_';

        static std::map<std::string, int> counters;
        const int index = counters[safe_name]++;
        if (index % stride != 0)
            return;

        const std::string path = std::string(dir) + "/" + safe_name + "_f" +
                                 std::to_string(rp26_sim::g_current_frame_index.load()) + "_" +
                                 std::to_string(index) + ".png";
        cv::imwrite(path, image);
    }
};
