#include "auto_buff/rune_plate_pattern.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

#include <opencv2/imgproc.hpp>

namespace auto_aim::energy
{
    RunePlatePattern scoreRunePlatePattern(const cv::Mat& bgr, cv::Point2f center,
                                           float plate_span_px)
    {
        RunePlatePattern result;
        if (bgr.empty() || !std::isfinite(center.x) || !std::isfinite(center.y) ||
            !std::isfinite(plate_span_px) || plate_span_px < 6.0f) {
            return result;
        }

        const int side = std::clamp(
            static_cast<int>(std::lround(plate_span_px * 1.8f)), 16, 192);
        cv::Rect roi(static_cast<int>(std::lround(center.x)) - side / 2,
                     static_cast<int>(std::lround(center.y)) - side / 2,
                     side, side);
        roi &= cv::Rect(0, 0, bgr.cols, bgr.rows);
        if (roi.width < 12 || roi.height < 12) return result;

        cv::Mat gray;
        if (bgr.channels() == 3) {
            cv::cvtColor(bgr(roi), gray, cv::COLOR_BGR2GRAY);
        } else {
            gray = bgr(roi).clone();
        }
        if (gray.type() != CV_8U) {
            cv::normalize(gray, gray, 0, 255, cv::NORM_MINMAX, CV_8U);
        }

        constexpr int kRadialBins = 24;
        constexpr int kThetaBins = 180;
        std::vector<double> polar(static_cast<std::size_t>(kRadialBins * kThetaBins), 0.0);
        const float center_x = center.x - static_cast<float>(roi.x);
        const float center_y = center.y - static_cast<float>(roi.y);
        const double max_radius = 0.5 * std::min(gray.cols, gray.rows);
        if (max_radius < 3.0) return result;

        for (int y = 0; y < gray.rows; ++y) {
            for (int x = 0; x < gray.cols; ++x) {
                const double dx = x - center_x;
                const double dy = y - center_y;
                const double radius = std::hypot(dx, dy);
                if (radius >= max_radius) continue;
                const double theta = std::atan2(dy, dx);
                const int r_bin = std::clamp(
                    static_cast<int>(radius / max_radius * kRadialBins),
                    0, kRadialBins - 1);
                const int t_bin = std::clamp(
                    static_cast<int>((theta + CV_PI) / (2.0 * CV_PI) * kThetaBins),
                    0, kThetaBins - 1);
                polar[static_cast<std::size_t>(r_bin * kThetaBins + t_bin)] +=
                    static_cast<double>(gray.at<uchar>(y, x)) * (r_bin + 1);
            }
        }

        std::array<double, kThetaBins> angular{};
        for (int theta_bin = 0; theta_bin < kThetaBins; ++theta_bin) {
            for (int radius_bin = 0; radius_bin < kRadialBins; ++radius_bin) {
                angular[static_cast<std::size_t>(theta_bin)] +=
                    polar[static_cast<std::size_t>(radius_bin * kThetaBins + theta_bin)];
            }
        }

        double harmonic_cos = 0.0;
        double harmonic_sin = 0.0;
        double angular_sum = 0.0;
        for (int theta_bin = 0; theta_bin < kThetaBins; ++theta_bin) {
            const double theta = 2.0 * CV_PI * theta_bin / kThetaBins - CV_PI;
            angular_sum += angular[static_cast<std::size_t>(theta_bin)];
            harmonic_cos += angular[static_cast<std::size_t>(theta_bin)] *
                std::cos(4.0 * theta);
            harmonic_sin += angular[static_cast<std::size_t>(theta_bin)] *
                std::sin(4.0 * theta);
        }
        const double phase = 0.25 * std::atan2(harmonic_sin, harmonic_cos);
        // Normalized fourth-harmonic amplitude. It is exactly the color- and
        // exposure-independent part of the RMCS peak/valley idea and is much
        // less sensitive to which side of the four arms is called a peak.
        result.pvd = std::sqrt(harmonic_cos * harmonic_cos + harmonic_sin * harmonic_sin) /
            (angular_sum + 1e-9);
        result.phase_rad = phase;
        result.roi_px = std::min(roi.width, roi.height);
        result.valid = true;
        return result;
    }
} // namespace auto_aim::energy
