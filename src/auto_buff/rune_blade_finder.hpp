#ifndef AUTO_AIM_ENERGY_RUNE_BLADE_FINDER_HPP
#define AUTO_AIM_ENERGY_RUNE_BLADE_FINDER_HPP

// 暗片通道 / 观测精修：照搬深大 RP-26Rune `RuneObservationRefiner` 的配方
// （src/core/algorithm/power_rune/src/RuneObservationRefiner.cpp + config/power_rune.json）：
//
//   1) 颜色差图：B−R（蓝方）或 R−B（红方）→ 高斯模糊(5×5) → 阈值(bin=62/50) → findContours；
//   2) **面积门限以"网络自己的框"为期望**：|area − 网络面积|/网络面积 ≤ 0.35
//      （`armor_module_area_relative_error_threshold`）—— 不是我自己推的理论面积；
//   3) 位置门限：轮廓中心离"预测槽位"不超过 max_offset_px；
//   4) 形状：外接框长宽比相对误差 ≤ 0.42（斜视允许压缩）。
//
// 为什么要它（§26/§32 的结论）：网络只可靠检出亮片，信息不完备；这条通道把**未点亮的
// 片**也测出来，信息才完备（rm_vision_core 的 5 片配准那套才搬得动）。

#include <array>

#include <opencv2/opencv.hpp>

namespace auto_aim::energy
{
    class RuneBladeFinder
    {
    public:
        struct Config
        {
            // 特征通道：仿真里暗片是暗灰、不是红/蓝 ⇒ 用边缘强度（Sobel 幅值）而不是
            // 深大的颜色差（阈值 60：§32 探针里靶面纹理响应远高于背景）。
            double edge_threshold = 60.0;
            // ---- 形状/面积判据（实拍实测标定，见 docs §40）----
            // 网络四点 = 靶面**边中点** ⇒ 四点面积 ≈ 靶面方框面积；可见的**圆盘面**
            // 实测只有它的 ~0.25 倍（实拍：四点 15376 px²、完整圆盘 3800~4000 px²）。
            double disc_area_ratio = 0.25;   // 期望圆盘面积 = 该比例 × 网络四点面积
            double min_area_ratio = 0.35;    // 相对**期望圆盘面积**
            double max_area_ratio = 1.70;
            double min_circularity = 0.30;   // 实拍实测：真圆盘 0.41~0.46，支架 0.05、背景 0.01（§40）
            double min_aspect = 0.45;
            double max_aspect = 2.20;
            double max_offset_px_ratio = 0.6; // 位置门限 = 该比例 × 靶面半尺寸
            double min_area_px = 20.0;
        };

        struct Result
        {
            std::array<cv::Point2f, 5> centers{};
            std::array<double, 5> scores{};   // 命中的轮廓面积（0 = 没找到）
            std::array<bool, 5> found{};
            int count() const
            {
                int n = 0;
                for (bool value : found) n += value ? 1 : 0;
                return n;
            }
        };

        RuneBladeFinder() = default;
        explicit RuneBladeFinder(const Config& config) : config_(config) {}

        /// @param expected_area_px 网络检出的靶面面积（深大用"相对误差"而不是绝对阈值）
        Result find(const cv::Mat& bgr, cv::Point2f hub, double orbit_px,
                    const std::array<double, 5>& angles_rad, double expected_area_px,
                    double plate_half_px, int search_px) const;

        const Config& config() const { return config_; }

    private:
        Config config_;
    };
} // namespace auto_aim::energy

#endif // AUTO_AIM_ENERGY_RUNE_BLADE_FINDER_HPP
