// 扇叶状态分类的**离线标定探针**。
//
// 为什么需要它：RuneBladeState 的 lit_ratio 是"ROI 内点亮面积 / 轨道半径²"，
// 是个**比值**。比值量只有在"同一份代码 + 同一种 hub/靶心几何来源"下才能互相
// 比较；之前用手工估的圆心/半径在两张图上各量一次，量出来的尺度差了 24 倍，
// 那种数据不能拿来定阈值（见 docs/energy_rune_issue_audit.md §46）。
//
// 这个探针直接调用**流水线里的同一个 RuneBladeState::classify**，只把
// "hub、轨道半径、各片靶心"这三样换成人工标注（网络检不到的"已激活"外观只能
// 靠人标），于是两张标注帧的数就在同一条路径上，可以直接定阈值。
//
// 用法：
//   buff_state_probe <image> --hub X,Y --orbit R \
//       [--blade X,Y]... | [--n5 phase_deg ...] [--dump prefix.png]
//
//   --hub    圆心肌像素
//   --orbit  轨道半径像素（hub→靶心）
//   --blade  一片靶心像素（可重复）
//   --n5     给定相位（度，0 = 正上方，顺时针）自动摆 5 片靶心
//   --dump   把每片的 ROI 原图与点亮掩膜拼图落盘，便于肉眼核对

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "auto_buff/rune_blade_state.hpp"

namespace
{
    bool parsePoint(const char* text, cv::Point2f& point)
    {
        double x = 0.0;
        double y = 0.0;
        if (std::sscanf(text, "%lf,%lf", &x, &y) != 2) return false;
        point = cv::Point2f(static_cast<float>(x), static_cast<float>(y));
        return true;
    }
} // namespace

int main(int argc, char** argv)
{
    if (argc < 4) {
        std::printf(
            "usage: %s <image> --hub X,Y --orbit R [--blade X,Y]... [--n5 deg...] "
            "[--dump prefix]\n",
            argv[0]);
        return 2;
    }

    const std::string image_path = argv[1];
    cv::Point2f hub(0.0f, 0.0f);
    bool have_hub = false;
    double orbit = 0.0;
    std::vector<cv::Point2f> blades;
    std::vector<double> phases_deg;
    std::string dump_prefix;

    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        const auto next = [&]() -> const char* { return (i + 1 < argc) ? argv[++i] : nullptr; };
        if (arg == "--hub") {
            const char* value = next();
            if (value == nullptr || !parsePoint(value, hub)) {
                std::printf("bad --hub value\n");
                return 2;
            }
            have_hub = true;
        } else if (arg == "--orbit") {
            const char* value = next();
            if (value == nullptr) return 2;
            orbit = std::atof(value);
        } else if (arg == "--blade") {
            const char* value = next();
            cv::Point2f point;
            if (value == nullptr || !parsePoint(value, point)) {
                std::printf("bad --blade value\n");
                return 2;
            }
            blades.push_back(point);
        } else if (arg == "--n5") {
            const char* value = next();
            if (value == nullptr) return 2;
            phases_deg.push_back(std::atof(value));
        } else if (arg == "--dump") {
            const char* value = next();
            if (value == nullptr) return 2;
            dump_prefix = value;
        } else {
            std::printf("unknown argument: %s\n", arg.c_str());
            return 2;
        }
    }

    if (!have_hub) {
        std::printf("--hub is required\n");
        return 2;
    }
    if (orbit <= 5.0) {
        std::printf("--orbit must be > 5 px\n");
        return 2;
    }

    const cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        std::printf("failed to read %s\n", image_path.c_str());
        return 2;
    }

    // --n5：按相位摆 5 片（相位 0 = 正上方；顺时针 = 屏幕坐标 y 向下为正）。
    for (const double phase_deg : phases_deg) {
        for (int k = 0; k < 5; ++k) {
            const double angle_deg = phase_deg + 72.0 * k;
            const double angle = (angle_deg - 90.0) * CV_PI / 180.0;
            blades.emplace_back(
                static_cast<float>(hub.x + orbit * std::cos(angle)),
                static_cast<float>(hub.y + orbit * std::sin(angle)));
        }
    }

    if (blades.empty()) {
        std::printf("no blade centres given (--blade / --n5)\n");
        return 2;
    }

    auto_aim::energy::RuneBladeState::Config config;
    const auto_aim::energy::RuneBladeState classifier(config);

    std::printf("%-7s %-16s %8s %8s %8s %8s %8s %-9s\n", "blade", "centre", "orbit", "roi_px",
                "lit_area", "lit_ratio", "arm", "state");

    std::vector<cv::Mat> rows;
    for (std::size_t i = 0; i < blades.size(); ++i) {
        const auto result = classifier.classify(image, hub, blades[i], orbit);
        std::printf("%-7zu (%7.1f,%6.1f) %8.1f %8d %8.0f %8.4f %8.3f %-9s\n", i, blades[i].x,
                    blades[i].y, result.orbit_px, result.roi_px, result.lit_area,
                    result.lit_ratio, result.arm_lit_ratio,
                    auto_aim::energy::RuneBladeState::name(result.state));

        if (!dump_prefix.empty()) {
            const int half =
                std::max(6, static_cast<int>(std::lround(config.roi_half_ratio * orbit)));
            cv::Rect roi(static_cast<int>(std::lround(blades[i].x)) - half,
                         static_cast<int>(std::lround(blades[i].y)) - half, 2 * half, 2 * half);
            roi &= cv::Rect(0, 0, image.cols, image.rows);
            if (roi.width < 4 || roi.height < 4) continue;
            cv::Mat patch = image(roi).clone();
            cv::Mat channels[3];
            cv::split(patch, channels);
            cv::Mat lit;
            cv::subtract(channels[2], channels[0], lit);
            cv::threshold(lit, lit, config.red_minus_blue, 255, cv::THRESH_BINARY);
            cv::morphologyEx(lit, lit, cv::MORPH_OPEN,
                             cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));
            cv::Mat mask_bgr;
            cv::cvtColor(lit, mask_bgr, cv::COLOR_GRAY2BGR);
            cv::Mat side;
            cv::hconcat(patch, mask_bgr, side);
            cv::putText(side, std::to_string(i), cv::Point(6, 22), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                        cv::Scalar(0, 255, 0), 2);
            rows.push_back(side);
        }
    }

    if (!dump_prefix.empty() && !rows.empty()) {
        cv::Mat canvas;
        cv::vconcat(rows, canvas);
        cv::imwrite(dump_prefix, canvas);
        std::printf("wrote %s (%dx%d, %zu blades, left=ROI right=mask)\n", dump_prefix.c_str(),
                    canvas.cols, canvas.rows, rows.size());
    }
    return 0;
}
