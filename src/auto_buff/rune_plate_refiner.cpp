#include "auto_buff/rune_plate_refiner.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace auto_aim::energy
{
    namespace
    {
        double quadArea(const std::array<cv::Point2f, 4>& points)
        {
            return std::abs(cv::contourArea(std::vector<cv::Point2f>(points.begin(), points.end())));
        }

        cv::Point2f quadCenter(const std::array<cv::Point2f, 4>& points)
        {
            cv::Point2f center(0.0f, 0.0f);
            for (const cv::Point2f& point : points) center += point;
            return center * 0.25f;
        }

        /// 把旋转矩形的四条**边中点**按"与网络点方向最接近"配过去，
        /// 保证输出顺序仍是 外→右→内→左（与 PnP 物点一致）。
        std::array<cv::Point2f, 4> edgeMidpoints(const cv::RotatedRect& rect,
                                                 const std::array<cv::Point2f, 4>& keypoints)
        {
            cv::Point2f corners[4];
            rect.points(corners);   // 顺序：左下、左上、右上、右下（OpenCV 约定）
            std::array<cv::Point2f, 4> midpoints{};
            for (int i = 0; i < 4; ++i) {
                midpoints[static_cast<std::size_t>(i)] =
                    0.5f * (corners[i] + corners[(i + 1) % 4]);
            }
            const cv::Point2f center = rect.center;
            std::array<cv::Point2f, 4> ordered{};
            std::array<bool, 4> used{};
            for (int k = 0; k < 4; ++k) {
                const cv::Point2f direction = keypoints[static_cast<std::size_t>(k)] - quadCenter(keypoints);
                int best = -1;
                double best_dot = -std::numeric_limits<double>::max();
                for (int i = 0; i < 4; ++i) {
                    if (used[static_cast<std::size_t>(i)]) continue;
                    const cv::Point2f edge_direction = midpoints[static_cast<std::size_t>(i)] - center;
                    const double norm = cv::norm(direction) * cv::norm(edge_direction);
                    if (norm < 1e-6) continue;
                    const double dot = direction.dot(edge_direction) / norm;
                    if (dot > best_dot) {
                        best_dot = dot;
                        best = i;
                    }
                }
                if (best >= 0) {
                    used[static_cast<std::size_t>(best)] = true;
                    ordered[static_cast<std::size_t>(k)] = midpoints[static_cast<std::size_t>(best)];
                } else {
                    return keypoints;   // 配不上就原样返回（不采信）
                }
            }
            return ordered;
        }
    } // namespace

    RunePlateRefiner::Result RunePlateRefiner::refine(
        const cv::Mat& bgr, const std::array<cv::Point2f, 4>& keypoints) const
    {
        Result result;
        result.points = keypoints;   // 默认原样返回（不采信时）
        if (bgr.empty() || !config_.enabled) return result;

        const double network_area = quadArea(keypoints);
        if (network_area < 16.0) return result;

        // ① ROI：网络四点外接框按 margin 放大（与自瞄侧同构）。
        cv::Rect box = cv::boundingRect(std::vector<cv::Point2f>(keypoints.begin(), keypoints.end()));
        const int margin_x = static_cast<int>(config_.margin_ratio * box.width);
        const int margin_y = static_cast<int>(config_.margin_ratio * box.height);
        cv::Rect roi(box.x - margin_x, box.y - margin_y,
                     box.width + 2 * margin_x, box.height + 2 * margin_y);
        roi &= cv::Rect(0, 0, bgr.cols, bgr.rows);
        if (roi.width < 6 || roi.height < 6) return result;
        result.center_shift_px = 0.0;

        // ② ROI 内经典提取：Otsu 亮区 → 形态学 → 轮廓。
        cv::Mat gray, binary;
        cv::cvtColor(bgr(roi), gray, cv::COLOR_BGR2GRAY);
        if (config_.use_otsu) {
            cv::threshold(gray, binary, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
        } else {
            cv::threshold(gray, binary, config_.binary_threshold, 255, cv::THRESH_BINARY);
        }
        const cv::Mat kernel = cv::getStructuringElement(
            cv::MORPH_ELLIPSE, cv::Size(std::max(1, config_.morph_size),
                                        std::max(1, config_.morph_size)));
        cv::morphologyEx(binary, binary, cv::MORPH_CLOSE, kernel);
        cv::morphologyEx(binary, binary, cv::MORPH_OPEN, kernel);

        const cv::Point2f offset(static_cast<float>(roi.x), static_cast<float>(roi.y));
        const cv::Point2f network_center = quadCenter(keypoints);

        // ③ 选"与网络四边形中心最重合、面积合理"的轮廓。
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        double best_score = -std::numeric_limits<double>::max();
        cv::RotatedRect best_rect;
        double best_area_ratio = -1.0;
        for (const auto& contour : contours) {
            const double area = std::abs(cv::contourArea(contour));
            if (area < 8.0) continue;
            const cv::RotatedRect rect = cv::minAreaRect(contour);
            const double ratio = area / network_area;
            if (ratio < config_.min_area_ratio || ratio > config_.max_area_ratio) continue;
            const double shift = cv::norm((rect.center + offset) - network_center);
            if (shift > config_.max_center_shift_ratio * std::sqrt(network_area)) continue;
            // 评分：中心更近、面积更接近网络面积（两者都归一化后取负距离）
            const double score = -shift / std::sqrt(network_area) - std::abs(ratio - 1.0);
            if (score > best_score) {
                best_score = score;
                best_rect = rect;
                best_area_ratio = ratio;
            }
        }
        if (best_score == -std::numeric_limits<double>::max()) return result;

        // ④-a 尺度保持模式：只借经典结果的中心（可选朝向），尺寸保留网络原值。
        //     这样"经典 ≠ 网络标注特征"的尺度差就不会进 PnP 深度（§44 的教训）。
        best_rect.center += offset;
        if (config_.preserve_network_scale) {
            const cv::Point2f network_center_plate = quadCenter(keypoints);
            const double shift = cv::norm(best_rect.center - network_center_plate);
            if (shift > config_.max_center_shift_ratio * std::sqrt(network_area)) return result;
            const double center_weight = std::clamp(config_.center_weight, 0.0, 1.0);
            const cv::Point2f target_center =
                network_center_plate * static_cast<float>(1.0 - center_weight) +
                best_rect.center * static_cast<float>(center_weight);

            // 朝向：minAreaRect 的角度对近方形有 90° 歧义，差太大就不修。
            double angle_diff = 0.0;
            if (config_.rotation_weight > 0.0) {
                const cv::RotatedRect network_rect = cv::minAreaRect(
                    std::vector<cv::Point2f>(keypoints.begin(), keypoints.end()));
                angle_diff = best_rect.angle - network_rect.angle;
                while (angle_diff > 45.0) angle_diff -= 90.0;
                while (angle_diff < -45.0) angle_diff += 90.0;
                if (std::abs(angle_diff) > config_.max_angle_diff_deg) angle_diff = 0.0;
            }
            const double theta = config_.rotation_weight * angle_diff * CV_PI / 180.0;
            const double cos_theta = std::cos(theta);
            const double sin_theta = std::sin(theta);
            std::array<cv::Point2f, 4> corrected{};
            for (int k = 0; k < 4; ++k) {
                const cv::Point2f d = keypoints[static_cast<std::size_t>(k)] - network_center_plate;
                const cv::Point2f rotated(static_cast<float>(cos_theta * d.x - sin_theta * d.y),
                                          static_cast<float>(sin_theta * d.x + cos_theta * d.y));
                corrected[static_cast<std::size_t>(k)] = target_center + rotated;
            }
            result.refined = true;
            result.points = corrected;
            result.area_ratio = best_area_ratio;
            result.center_shift_px = shift;
            result.size_ratio = 1.0;   // 尺度显式保持
            return result;
        }

        const std::array<cv::Point2f, 4> ordered = edgeMidpoints(best_rect, keypoints);
        const double refined_area = quadArea(ordered);
        const double size_ratio = refined_area > 1.0 ? refined_area / network_area : -1.0;
        if (size_ratio < config_.min_size_ratio || size_ratio > config_.max_size_ratio) {
            return result;
        }
        result.refined = true;
        result.points = ordered;
        result.area_ratio = best_area_ratio;
        result.center_shift_px = cv::norm(quadCenter(ordered) - network_center);
        result.size_ratio = size_ratio;
        return result;
    }
} // namespace auto_aim::energy
