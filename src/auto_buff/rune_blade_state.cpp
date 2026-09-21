#include "auto_buff/rune_blade_state.hpp"

#include <algorithm>
#include <cmath>

namespace auto_aim::energy
{
    const char* RuneBladeState::name(State state)
    {
        switch (state) {
        case State::Unlit: return "unlit";
        case State::Inactive: return "inactive";
        case State::Active: return "active";
        default: return "unknown";
        }
    }

    RuneBladeState::Result RuneBladeState::classify(const cv::Mat& bgr, cv::Point2f hub,
                                                    cv::Point2f blade, double orbit_px) const
    {
        Result result;
        if (bgr.empty() || orbit_px < 5.0) return result;
        result.orbit_px = orbit_px;

        // ROI：以该片靶心为中心、按轨道半径定尺寸（与网络关键点无关 ⇒ 尺度可控）。
        const int half = std::max(6, static_cast<int>(std::lround(config_.roi_half_ratio * orbit_px)));
        cv::Rect roi(static_cast<int>(std::lround(blade.x)) - half,
                     static_cast<int>(std::lround(blade.y)) - half, 2 * half, 2 * half);
        roi &= cv::Rect(0, 0, bgr.cols, bgr.rows);
        if (roi.width < 4 || roi.height < 4) return result;
        result.roi_px = std::min(roi.width, roi.height);

        // 点亮图案：红/橙 LED ⇒ R − B 显著为正（灰盘/白墙/天空反射都不满足）。
        std::vector<cv::Mat> channels;
        cv::split(bgr(roi), channels);
        cv::Mat lit;
        cv::subtract(channels[2], channels[0], lit);
        cv::threshold(lit, lit, config_.red_minus_blue, 255, cv::THRESH_BINARY);
        cv::morphologyEx(lit, lit, cv::MORPH_OPEN,
                         cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));
        const double lit_area = cv::countNonZero(lit);
        result.lit_area = lit_area;
        result.lit_ratio = lit_area / (orbit_px * orbit_px);

        // 灯臂：从靶心朝圆心采样，看点亮的比例（未激活是"梯状点阵臂"、已激活是"橙色框"，
        // 两者都点亮；未点亮的片这里几乎全灭）。
        const cv::Point2f direction = hub - blade;
        const double length = cv::norm(direction);
        int lit_samples = 0;
        int total = 0;
        if (length > 4.0) {
            for (int i = 0; i < std::max(1, config_.arm_samples); ++i) {
                const double t = 0.35 + 0.55 * static_cast<double>(i) /
                                            std::max(1, config_.arm_samples - 1);
                const cv::Point2f point = blade + direction * static_cast<float>(t);
                const int x = static_cast<int>(std::lround(point.x));
                const int y = static_cast<int>(std::lround(point.y));
                if (x < 1 || y < 1 || x >= bgr.cols - 1 || y >= bgr.rows - 1) continue;
                const double difference =
                    static_cast<double>(bgr.at<cv::Vec3b>(y, x)[2]) -
                    static_cast<double>(bgr.at<cv::Vec3b>(y, x)[0]);
                ++total;
                if (difference > config_.red_minus_blue) ++lit_samples;
            }
        }
        result.arm_lit_ratio = total > 0 ? static_cast<double>(lit_samples) / total : -1.0;

        // 三态判定：先看有没有点亮，再按归一化面积区分未激活/已激活（SCUT 的面积区间思路）。
        if (result.lit_ratio < config_.unlit_below) {
            result.state = State::Unlit;
        } else if (result.lit_ratio < config_.active_above) {
            result.state = State::Inactive;
        } else {
            result.state = State::Active;
        }
        result.valid = true;
        return result;
    }
} // namespace auto_aim::energy
