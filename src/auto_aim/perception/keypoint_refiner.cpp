#include "keypoint_refiner.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include "common/types.hpp"

namespace auto_aim
{
    namespace
    {
        // A light-bar candidate inside the refinement ROI, in full-image pixels.
        struct Bar {
            cv::Point2f center;
            cv::Point2f top;      // upper endpoint (smaller y)
            cv::Point2f bottom;   // lower endpoint
            double height = 0.0;
            double tilt = 0.0;    // degrees from the vertical
        };

        double angleFromVertical(const cv::Point2f& direction)
        {
            // Undirected axis: (0,-1) and (0,1) must give the same angle.
            double degrees = std::atan2(direction.x, -direction.y) * 180.0 / CV_PI;
            if (degrees >= 90.0) degrees -= 180.0;
            else if (degrees < -90.0) degrees += 180.0;
            return degrees;
        }

        std::vector<Bar> findBars(const cv::Mat& roi_bgr, const cv::Rect& roi,
                                  const KeypointRefinerConfig& config)
        {
            std::vector<Bar> bars;
            cv::Mat gray, binary;
            cv::cvtColor(roi_bgr, gray, cv::COLOR_BGR2GRAY);
            if (config.use_otsu) {
                cv::threshold(gray, binary, 0, 255,
                              cv::THRESH_BINARY | cv::THRESH_OTSU);
            } else {
                cv::threshold(gray, binary, config.binary_threshold, 255,
                              cv::THRESH_BINARY);
            }
            const cv::Mat kernel = cv::getStructuringElement(
                cv::MORPH_ELLIPSE, cv::Size(config.morph_size, config.morph_size));
            cv::morphologyEx(binary, binary, cv::MORPH_CLOSE, kernel);
            cv::morphologyEx(binary, binary, cv::MORPH_OPEN, kernel);

            std::vector<std::vector<cv::Point>> contours;
            cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
            for (const auto& contour : contours) {
                const cv::RotatedRect rect = cv::minAreaRect(contour);
                if (rect.size.width < 1.0f || rect.size.height < 1.0f) continue;
                const Light light(rect);   // sets width/height/top/bottom/tilt_angle
                const double ratio = light.width / std::max(1.0, light.height);
                if (ratio < config.min_bar_ratio || ratio > config.max_bar_ratio) continue;
                if (std::abs(light.tilt_angle) > config.max_bar_angle) continue;
                // The ROI is cropped from the full image, so shift the points back.
                const cv::Point2f offset(static_cast<float>(roi.x),
                                         static_cast<float>(roi.y));
                Bar bar;
                bar.center = light.center + offset;
                bar.height = std::max(light.width, light.height);
                const cv::Point2f a = light.top + offset;
                const cv::Point2f b = light.bottom + offset;
                bar.top = a.y <= b.y ? a : b;
                bar.bottom = a.y <= b.y ? b : a;
                bar.tilt = angleFromVertical(bar.bottom - bar.top);
                bars.push_back(bar);
            }
            return bars;
        }
    } // namespace

    KeypointRefiner::KeypointRefiner(const KeypointRefinerConfig& config)
        : config_(config)
    {
        config_.binary_threshold = std::max(1, std::min(config_.binary_threshold, 254));
        config_.morph_size = std::max(1, config_.morph_size);
    }

    RefineResult KeypointRefiner::refine(
        const cv::Mat& bgr, const std::array<cv::Point2f, 4>& nn_keypoints) const
    {
        RefineResult result;
        result.keypoints = nn_keypoints;
        if (!config_.enabled || bgr.empty()) return result;

        const cv::Point2f& top_left = nn_keypoints[0];
        const cv::Point2f& top_right = nn_keypoints[1];
        const cv::Point2f& bottom_right = nn_keypoints[2];
        const cv::Point2f& bottom_left = nn_keypoints[3];

        const cv::Point2f top_mid = 0.5f * (top_left + top_right);
        const cv::Point2f bottom_mid = 0.5f * (bottom_left + bottom_right);
        const cv::Point2f left_mid = 0.5f * (top_left + bottom_left);
        const cv::Point2f right_mid = 0.5f * (top_right + bottom_right);
        const cv::Point2f center = 0.5f * (top_mid + bottom_mid);
        const double width = cv::norm(top_right - top_left);
        if (width < 2.0) return result;

        cv::Rect roi = cv::boundingRect(std::vector<cv::Point2f>{
            top_left, top_right, bottom_right, bottom_left});
        const int margin_x = static_cast<int>(config_.margin_ratio * roi.width) + 2;
        const int margin_y = static_cast<int>(config_.margin_ratio * roi.height) + 2;
        roi.x -= margin_x;
        roi.y -= margin_y;
        roi.width += 2 * margin_x;
        roi.height += 2 * margin_y;
        roi &= cv::Rect(0, 0, bgr.cols, bgr.rows);
        if (roi.width < 4 || roi.height < 4) return result;

        // Light-bar brightness varies a lot between a lit and an unlit armor and
        // between the real and the rendered scene, so a single threshold is not
        // reliable. Try progressively lower ones inside this small ROI and keep
        // the first geometrically consistent result.
        const cv::Mat roi_image = bgr(roi);
        // Otsu on the ROI first (adapts to the local brightness), then fixed
        // thresholds as a fallback.
        std::array<KeypointRefinerConfig, 4> attempts{};
        attempts[0] = config_;
        attempts[0].use_otsu = true;
        for (std::size_t i = 1; i < attempts.size(); ++i) {
            attempts[i] = config_;
            attempts[i].use_otsu = false;
            attempts[i].binary_threshold = std::max(1, std::min(
                static_cast<int>(config_.binary_threshold *
                                 (i == 1 ? 1.0 : (i == 2 ? 0.8 : 0.6))), 254));
        }
        for (const KeypointRefinerConfig& local : attempts) {
            const std::vector<Bar> bars = findBars(roi_image, roi, local);
            if (bars.size() < 2) continue;

            const Bar* left = nullptr;
            const Bar* right = nullptr;
            double left_cost = std::numeric_limits<double>::max();
            double right_cost = std::numeric_limits<double>::max();
            for (const Bar& bar : bars) {
                const double to_left = cv::norm(bar.center - left_mid);
                const double to_right = cv::norm(bar.center - right_mid);
                if (to_left < left_cost) { left_cost = to_left; left = &bar; }
                if (to_right < right_cost) { right_cost = to_right; right = &bar; }
            }
            if (left == nullptr || right == nullptr || left == right) continue;

            const double height_max = std::max(left->height, right->height);
            if (height_max <= 1.0) continue;
            if (std::abs(left->height - right->height) / height_max >
                config_.max_height_mismatch) {
                continue;
            }
            if (std::abs(left->tilt + right->tilt) > config_.max_pair_angle_sum) {
                continue;
            }

            const cv::Point2f refined_center = 0.5f * (left->center + right->center);
            const double shift = cv::norm(refined_center - center);
            if (shift > config_.max_center_shift_ratio * width) continue;

            result.refined = true;
            result.keypoints = {left->top, right->top, right->bottom, left->bottom};
            result.center_shift_px = shift;
            result.bar_height_px = height_max;
            return result;
        }
        return result;
    }
} // namespace auto_aim
