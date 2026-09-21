#include "auto_buff/rune_diff_probe.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace auto_aim::energy
{
    std::optional<RuneDiffCandidate> RuneDiffProbe::update(const cv::Mat& bgr,
                                                           cv::Point2f hub,
                                                           double orbit_radius_px)
    {
        if (!config_.enabled || bgr.empty()) return std::nullopt;

        cv::Mat gray;
        cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
        if (previous_gray_.empty() || previous_gray_.size() != gray.size()) {
            previous_gray_ = gray.clone();
            return std::nullopt;
        }

        cv::Mat reference = previous_gray_;
        previous_gray_ = gray.clone();

        // 运动补偿：云台在动时不做补偿的话，整幅背景都会被算成"变化"。
        if (config_.compensate_motion) {
            cv::Mat prev32, cur32;
            previous_gray_.convertTo(cur32, CV_32F);
            reference.convertTo(prev32, CV_32F);
            const cv::Point2d shift = cv::phaseCorrelate(prev32, cur32);
            if (std::abs(shift.x) <= config_.max_shift_px &&
                std::abs(shift.y) <= config_.max_shift_px) {
                const cv::Mat warp = (cv::Mat_<double>(2, 3) << 1.0, 0.0, -shift.x, 0.0, 1.0,
                                      -shift.y);
                cv::warpAffine(reference, reference, warp, gray.size(), cv::INTER_LINEAR,
                               cv::BORDER_REPLICATE);
            }
        }

        cv::Mat difference;
        cv::absdiff(previous_gray_, reference, difference);
        cv::GaussianBlur(difference, difference, cv::Size(5, 5), 0);
        cv::Mat mask;
        cv::threshold(difference, mask, config_.threshold, 255, cv::THRESH_BINARY);
        cv::morphologyEx(mask, mask, cv::MORPH_OPEN,
                         cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));

        cv::Mat labels, stats, centroids;
        const int count = cv::connectedComponentsWithStats(mask, labels, stats, centroids, 8);

        std::optional<RuneDiffCandidate> best;
        const bool have_hub = hub.x >= 0.0f && hub.y >= 0.0f && orbit_radius_px > 1.0;
        for (int index = 1; index < count; ++index) {
            const double area = static_cast<double>(stats.at<int>(index, cv::CC_STAT_AREA));
            if (area < config_.min_area) continue;
            const cv::Point2f center(static_cast<float>(centroids.at<double>(index, 0)),
                                     static_cast<float>(centroids.at<double>(index, 1)));
            double radius = 0.0;
            if (have_hub) {
                radius = cv::norm(center - hub);
                if (radius < config_.ring_low * orbit_radius_px ||
                    radius > config_.ring_high * orbit_radius_px) {
                    continue;
                }
            }
            if (!best.has_value() || area > best->area) {
                RuneDiffCandidate candidate;
                candidate.center = center;
                candidate.area = area;
                candidate.radius_px = radius;
                candidate.strength = area;
                best = candidate;
            }
        }
        return best;
    }
} // namespace auto_aim::energy
