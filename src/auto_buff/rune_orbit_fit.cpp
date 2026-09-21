#include "auto_buff/rune_orbit_fit.hpp"

#include <algorithm>
#include <cmath>

namespace auto_aim::energy
{
    void RuneOrbitFit::add(const cv::Point2f& blade_px, double now)
    {
        if (!std::isfinite(blade_px.x) || !std::isfinite(blade_px.y)) return;
        samples_.push_back(Sample{blade_px, now});
        // 窗口外的样本直接丢掉（两侧都清），避免 deque 无限增长。
        while (!samples_.empty() && now - samples_.front().time > config_.window_s * 2.0) {
            samples_.pop_front();
        }
    }

    std::vector<cv::Point2f> RuneOrbitFit::windowPoints(double now) const
    {
        std::vector<cv::Point2f> points;
        points.reserve(samples_.size());
        for (auto it = samples_.rbegin(); it != samples_.rend(); ++it) {
            if (now - it->time > config_.window_s) break;
            points.push_back(it->point);
        }
        return points;
    }

    bool RuneOrbitFit::fitOnce(const std::vector<cv::Point2f>& points, Result& out)
    {
        if (points.size() < 5) return false;   // cv::fitEllipse 的下限
        const cv::RotatedRect ellipse = cv::fitEllipse(points);
        const double major = std::max(ellipse.size.width, ellipse.size.height) * 0.5;
        const double minor = std::min(ellipse.size.width, ellipse.size.height) * 0.5;
        if (major <= 0.0 || minor <= 0.0) return false;
        out.valid = true;
        out.center = ellipse.center;
        out.semi_major_px = major;
        out.semi_minor_px = minor;
        out.angle_deg = ellipse.angle;
        out.samples = static_cast<int>(points.size());
        return true;
    }

    RuneOrbitFit::Result RuneOrbitFit::fit(double now) const
    {
        Result result;
        const std::vector<cv::Point2f> points = windowPoints(now);
        if (static_cast<int>(points.size()) < config_.min_samples) return result;
        if (!fitOnce(points, result)) return Result{};
        // 注意：闸门不通过时**仍然把原始拟合值带回去**（valid=false），
        // 否则 CSV 里只看到 0，无法判断到底卡在哪一条（见 §13.3 的诊断）。
        const bool plausible =
            result.semi_major_px >= config_.min_semi_major_px &&
            result.semi_major_px <= config_.max_semi_major_px &&
            result.semi_minor_px / result.semi_major_px >= config_.min_axis_ratio;
        // 相对向量的椭圆中心应当在原点附近：偏得多说明样本被平移污染（例如云台
        // 大幅运动期间采的样），这种拟合不可信。
        const bool centered = cv::norm(result.center) <= config_.max_center_offset_px;

        // 一次野值剔除再拟合：单个跳点会把样本少时的椭圆拽歪。
        const double a = result.semi_major_px;
        const double b = result.semi_minor_px;
        std::vector<double> residuals;
        residuals.reserve(points.size());
        for (const cv::Point2f& point : points) {
            const cv::Point2f delta = point - result.center;
            // 椭圆在极角 φ 处的半径（极坐标形式，不是参数形式）。
            const double phi =
                std::atan2(delta.y, delta.x) - result.angle_deg * CV_PI / 180.0;
            const double expected =
                a * b / std::hypot(b * std::cos(phi), a * std::sin(phi));
            const double residual = std::abs(cv::norm(delta) - expected);
            residuals.push_back(residual);
        }
        std::vector<double> ordered = residuals;
        std::sort(ordered.begin(), ordered.end());
        const double median = ordered.empty() ? 0.0 : ordered[ordered.size() / 2];
        std::vector<cv::Point2f> filtered;
        filtered.reserve(points.size());
        for (std::size_t i = 0; i < points.size(); ++i) {
            if (residuals[i] <= std::max(config_.outlier_gate * median, 1.0)) {
                filtered.push_back(points[i]);
            }
        }
        if (static_cast<int>(filtered.size()) >= config_.min_samples &&
            filtered.size() < points.size()) {
            Result refit;
            if (fitOnce(filtered, refit)) {
                refit.valid = refit.semi_major_px >= config_.min_semi_major_px &&
                              refit.semi_major_px <= config_.max_semi_major_px &&
                              refit.semi_minor_px / refit.semi_major_px >= config_.min_axis_ratio &&
                              cv::norm(refit.center) <= config_.max_center_offset_px;
                return refit;
            }
        }
        result.valid = plausible && centered;
        return result;
    }
} // namespace auto_aim::energy
