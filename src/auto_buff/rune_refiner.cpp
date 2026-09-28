#include "auto_buff/rune_refiner.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace auto_aim::energy
{
    namespace
    {
        // 与深大 is_line_pass_through_contour 等价：线段 A→B 上采样若干点，
        // 只要有一个落在轮廓内部就算"穿过"。
        bool linePassesThroughContour(const cv::Point2f& a, const cv::Point2f& b,
                                      const std::vector<cv::Point>& contour, int samples)
        {
            for (int i = 0; i <= samples; ++i) {
                const float t = static_cast<float>(i) / static_cast<float>(samples);
                const cv::Point2f point = a + t * (b - a);
                if (cv::pointPolygonTest(contour, point, false) > 0) return true;
            }
            return false;
        }

        double solidity(const std::vector<cv::Point>& contour)
        {
            const double area = std::abs(cv::contourArea(contour));
            std::vector<cv::Point> hull;
            cv::convexHull(contour, hull);
            const double hull_area = std::abs(cv::contourArea(hull));
            return hull_area > 1e-6 ? area / hull_area : 0.0;
        }

        bool nearBorder(const std::vector<cv::Point>& contour, const cv::Size& size, int margin)
        {
            for (const cv::Point& point : contour) {
                const int distance = std::min({point.x, point.y, size.width - 1 - point.x,
                                               size.height - 1 - point.y});
                if (distance <= margin) return true;
            }
            return false;
        }

        cv::Point2f centroid(const std::vector<cv::Point>& contour, cv::Point2f fallback)
        {
            const cv::Moments moments = cv::moments(contour);
            if (std::abs(moments.m00) < 1e-6) return fallback;
            return {static_cast<float>(moments.m10 / moments.m00),
                    static_cast<float>(moments.m01 / moments.m00)};
        }
    } // namespace

    RuneBladeRefinement RuneRefiner::refine(const cv::Mat& bgr,
                                            const std::vector<cv::Point2f>& keypoints,
                                            int our_color) const
    {
        RuneBladeRefinement result;
        if (!config_.enabled || bgr.empty() || keypoints.size() < 5) return result;

        const cv::Point2f top = keypoints[0];
        const cv::Point2f left = keypoints[1];
        const cv::Point2f center_r = keypoints[2];
        const cv::Point2f right = keypoints[3];
        const cv::Point2f bottom = keypoints[4];
        const cv::Point2f armor_center_nn = 0.25f * (top + left + right + bottom);

        // ① ROI：五点 minAreaRect × roi_scale（与他们的 convert2rune_observation 一致）
        std::vector<cv::Point2f> points{top, left, center_r, right, bottom};
        cv::RotatedRect roi_rect = cv::minAreaRect(points);
        roi_rect.size *= static_cast<float>(config_.roi_scale);
        cv::Rect view = roi_rect.boundingRect();
        view &= cv::Rect(0, 0, bgr.cols, bgr.rows);
        if (view.width < 4 || view.height < 4) return result;

        // ② 颜色差掩膜：R−B（红方）或 B−R（蓝方）→ 高斯 → 二值化 → 外轮廓
        const cv::Mat view_image = bgr(view);
        std::vector<cv::Mat> channels;
        cv::split(view_image, channels);
        cv::Mat difference = our_color == 0 ? channels[2] - channels[0] : channels[0] - channels[2];
        cv::GaussianBlur(difference, difference, cv::Size(5, 5), 0);
        const double threshold = our_color == 0 ? config_.red_minus_blue_threshold
                                                : config_.blue_minus_red_threshold;
        cv::Mat mask;
        cv::threshold(difference, mask, threshold, 255, cv::THRESH_BINARY);

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
        for (auto& contour : contours) {
            for (cv::Point& point : contour) {
                point.x += view.x;
                point.y += view.y;
            }
        }
        if (contours.empty()) return result;

        // ③ 语义分割：三个互斥判据（与深大 constrain_contours 一致）
        const double nn_ellipse_area =
            0.25 * cv::norm(top - bottom) * cv::norm(left - right) * CV_PI;
        std::vector<std::vector<cv::Point>> armor_candidates;
        std::vector<std::vector<cv::Point>> arm_candidates;
        std::vector<std::vector<cv::Point>> center_r_candidates;
        for (const auto& contour : contours) {
            if (contour.size() < 4) continue;

            // 装甲板：包含网络靶心 + 面积与网络椭圆相符
            const bool inside_armor = cv::pointPolygonTest(contour, armor_center_nn, false) > 0;
            const double relative_error =
                nn_ellipse_area > 1e-6
                    ? std::abs(std::abs(cv::contourArea(contour)) - nn_ellipse_area) / nn_ellipse_area
                    : std::numeric_limits<double>::max();
            if (inside_armor && relative_error <= config_.armor_area_relative_error) {
                armor_candidates.push_back(contour);
            }

            // 灯臂：靶心与 R 标都在轮廓外，二者连线穿过轮廓，且左右点连线不穿过
            const bool center_outside = cv::pointPolygonTest(contour, armor_center_nn, false) < 0;
            const bool r_outside = cv::pointPolygonTest(contour, center_r, false) < 0;
            if (center_outside && r_outside &&
                linePassesThroughContour(armor_center_nn, center_r, contour,
                                         config_.light_arm_line_samples) &&
                !linePassesThroughContour(left, right, contour, config_.light_arm_line_samples)) {
                arm_candidates.push_back(contour);
            }

            // R 标：包含 R 点，且其它关键点都在外
            const bool inside_r = cv::pointPolygonTest(contour, center_r, false) > 0;
            if (inside_r && cv::pointPolygonTest(contour, top, false) < 0 &&
                cv::pointPolygonTest(contour, left, false) < 0 &&
                cv::pointPolygonTest(contour, right, false) < 0 &&
                cv::pointPolygonTest(contour, bottom, false) < 0 &&
                cv::pointPolygonTest(contour, armor_center_nn, false) < 0) {
                center_r_candidates.push_back(contour);
            }
        }

        // ④ 唯一化：装甲板取面积误差最小，灯臂取面积最大，R 标取面积最小
        const std::vector<cv::Point>* armor = nullptr;
        {
            double best = std::numeric_limits<double>::max();
            for (const auto& contour : armor_candidates) {
                const double error =
                    nn_ellipse_area > 1e-6
                        ? std::abs(std::abs(cv::contourArea(contour)) - nn_ellipse_area) /
                              nn_ellipse_area
                        : 0.0;
                if (error < best) {
                    best = error;
                    armor = &contour;
                }
            }
        }
        const std::vector<cv::Point>* arm = nullptr;
        {
            double best = -1.0;
            for (const auto& contour : arm_candidates) {
                const double area = std::abs(cv::contourArea(contour));
                if (area > best) {
                    best = area;
                    arm = &contour;
                }
            }
        }
        const std::vector<cv::Point>* center_mark = nullptr;
        {
            double best = std::numeric_limits<double>::max();
            for (const auto& contour : center_r_candidates) {
                const double area = std::abs(cv::contourArea(contour));
                if (area < best) {
                    best = area;
                    center_mark = &contour;
                }
            }
        }

        // ⑤ 描述子闸门（深大的 EllipseDescriptor / RectangularDescriptor 等价物）
        if (armor != nullptr) {
            result.has_armor = true;
            result.armor_solidity = solidity(*armor);
            {
                const double area = std::abs(cv::contourArea(*armor));
                const double perimeter = cv::arcLength(*armor, true);
                if (perimeter > 1.0) {
                    result.armor_circularity = 4.0 * CV_PI * area / (perimeter * perimeter);
                }
                // 网络四点（k0,k1,k3,k4）构成的四边形面积：作为**尺度基准**，
                // 面积比与圆度都相对它来判（与 §39 的实拍标定一致）。
                double quad_area = 0.0;
                if (keypoints.size() >= 5) {
                    const cv::Point2f quad[4] = {keypoints[0], keypoints[1], keypoints[3],
                                                 keypoints[4]};
                    quad_area = std::abs(cv::contourArea(std::vector<cv::Point2f>(
                        quad, quad + 4)));
                }
                if (quad_area > 1.0) result.armor_area_ratio = area / quad_area;
            }
            result.armor_usable = result.armor_solidity > config_.armor_min_solidity;
            result.armor_center = centroid(*armor, armor_center_nn);
            if (nearBorder(*armor, bgr.size(), config_.border_margin_px)) {
                result.near_border = true;
                result.armor_usable = false;
            }
        }
        if (arm != nullptr) {
            result.has_arm = true;
            result.arm_solidity = solidity(*arm);
            const cv::RotatedRect rect = cv::minAreaRect(*arm);
            const double width = static_cast<double>(rect.size.width);
            const double height = static_cast<double>(rect.size.height);
            const double long_side = std::max(width, height);
            const double short_side = std::max(1e-3, std::min(width, height));
            result.arm_aspect = long_side / short_side;
            const double aspect_error =
                std::abs(config_.arm_expect_aspect - result.arm_aspect) / config_.arm_expect_aspect;
            result.arm_usable = result.arm_solidity > config_.arm_min_solidity &&
                                aspect_error < config_.arm_aspect_tolerance;
        }
        if (center_mark != nullptr && center_mark->size() >= 5) {
            result.has_center_r = true;
            const cv::RotatedRect ellipse = cv::fitEllipse(*center_mark);
            result.contour_center = ellipse.center;
            result.center_gap_px = cv::norm(result.contour_center - center_r);
        }

        result.usable = result.has_armor && result.has_arm && result.has_center_r &&
                        result.armor_usable && result.arm_usable;
        return result;
    }

    cv::Point2f RuneRefiner::refineCenter(const cv::Mat& bgr,
                                          const std::vector<cv::Point2f>& keypoints,
                                          cv::Point2f keypoint_center, int our_color,
                                          RuneBladeRefinement* detail) const
    {
        const RuneBladeRefinement refinement = refine(bgr, keypoints, our_color);
        if (detail != nullptr) *detail = refinement;

        cv::Point2f center = keypoint_center;
        const bool gap_ok = refinement.has_center_r &&
                            (refinement.center_gap_px < 0.0 ||
                             refinement.center_gap_px <= config_.max_center_gap_px) &&
                            (!config_.fuse_requires_armor_usable || refinement.armor_usable);
        if (config_.enabled && config_.fuse_center && gap_ok) {
            const double weight = std::clamp(config_.fuse_weight_contour, 0.0, 1.0);
            center.x = static_cast<float>((1.0 - weight) * keypoint_center.x +
                                          weight * refinement.contour_center.x);
            center.y = static_cast<float>((1.0 - weight) * keypoint_center.y +
                                          weight * refinement.contour_center.y);
        }
        if (detail != nullptr) {
            detail->center_fused = cv::norm(center - keypoint_center) > 1e-3;
            detail->refined_center = center;
        }
        return center;
    }
} // namespace auto_aim::energy
