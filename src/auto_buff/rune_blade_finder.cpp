#include "auto_buff/rune_blade_finder.hpp"

#include <algorithm>
#include <cmath>

namespace auto_aim::energy
{
    RuneBladeFinder::Result RuneBladeFinder::find(
        const cv::Mat& bgr, cv::Point2f hub, double orbit_px,
        const std::array<double, 5>& angles_rad, double expected_area_px,
        double plate_half_px, int search_px) const
    {
        Result result;
        if (bgr.empty() || hub.x <= 0.0f || orbit_px < 5.0) return result;

        // 特征通道：**边缘强度**（Sobel 幅值）。深大用颜色差（B−R/R−B，阈值 62/50）是因为
        // 他们的实车靶面是红/蓝配色；我们仿真里未点亮的靶面是暗灰 ⇒ 颜色差≈0（实测召回≈0），
        // 所以特征换成边缘/纹理（§32 探针：亮片 26818 vs 其余槽 824 vs 背景 <300）。
        // **验证结构照抄深大**：面积以"网络自己的靶面面积"为期望（相对误差 35%）、
        // 长宽比相对误差 42%、位置/包含关系。
        cv::Mat gray_features, gx, gy, magnitude;
        cv::cvtColor(bgr, gray_features, cv::COLOR_BGR2GRAY);
        cv::GaussianBlur(gray_features, gray_features, cv::Size(5, 5), 0);
        cv::Sobel(gray_features, gx, CV_32F, 1, 0, 3);
        cv::Sobel(gray_features, gy, CV_32F, 0, 1, 3);
        cv::magnitude(gx, gy, magnitude);
        cv::Mat mask;
        cv::threshold(magnitude, mask, config_.edge_threshold, 255.0, cv::THRESH_BINARY);
        mask.convertTo(mask, CV_8U);
        cv::morphologyEx(mask, mask, cv::MORPH_CLOSE,
                         cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));

        // 窗口必须**大于靶面本身**（否则轮廓被裁掉、面积/形状全失真）：
        // 半边 = 靶面半尺寸 + 搜索半径。
        const int half = std::max(8, static_cast<int>(std::lround(plate_half_px)) + search_px);
        for (int slot = 0; slot < 5; ++slot) {
            const double angle = angles_rad[static_cast<std::size_t>(slot)];
            const cv::Point2f predicted(
                hub.x + static_cast<float>(orbit_px * std::cos(angle)),
                hub.y + static_cast<float>(orbit_px * std::sin(angle)));
            cv::Rect roi(static_cast<int>(predicted.x) - half,
                         static_cast<int>(predicted.y) - half, 2 * half, 2 * half);
            roi &= cv::Rect(0, 0, bgr.cols, bgr.rows);
            if (roi.width < 8 || roi.height < 8) continue;

            std::vector<std::vector<cv::Point>> contours;
            cv::findContours(mask(roi), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

            double best_area = 0.0;
            double best_error = -1.0;
            cv::Rect best_box;
            for (const auto& contour : contours) {
                const double area = cv::contourArea(contour);
                if (area < config_.min_area_px) continue;
                // ② 面积门限：期望值 = disc_area_ratio × 网络四点面积（实拍标定）；
                //    允许区间按期望圆盘面积给（碎片化会偏小、粘连会偏大）。
                const double expected_disc = config_.disc_area_ratio * expected_area_px;
                if (expected_disc > 1.0) {
                    if (area < config_.min_area_ratio * expected_disc ||
                        area > config_.max_area_ratio * expected_disc) {
                        continue;
                    }
                }
                // ②b 圆度：圆盘面 ≈1，支架/墙边/天花板灯细长或带尖角 ⇒ 低。
                const double perimeter = cv::arcLength(contour, true);
                if (perimeter <= 1.0) continue;
                const double circularity = 4.0 * CV_PI * area / (perimeter * perimeter);
                if (circularity < config_.min_circularity) continue;
                const cv::Rect box = cv::boundingRect(contour);
                if (box.height <= 0) continue;
                // ④ 长宽比相对误差（期望 1，斜视允许压缩）
                const double aspect = static_cast<double>(box.width) / box.height;
                if (aspect < config_.min_aspect || aspect > config_.max_aspect) continue;
                const cv::Point2f center(box.x + 0.5f * box.width + roi.x,
                                         box.y + 0.5f * box.height + roi.y);
                // ③ 位置门限 + 包含关系（深大：armor_module 必须包含网络靶心）
                if (cv::norm(center - predicted) > config_.max_offset_px_ratio * plate_half_px) continue;
                const cv::Point2f within(predicted.x - roi.x, predicted.y - roi.y);
                if (within.x < box.x || within.x > box.x + box.width ||
                    within.y < box.y || within.y > box.y + box.height) {
                    continue;
                }
                // 深大 `constrain_contours`：条件全过之后取该轮廓；这里若有多个候选，
                // 取"面积最接近期望"的那个（比"面积最大"更稳）。
                if (best_area <= 0.0 || circularity > best_error) {
                    best_error = circularity;
                    best_area = area;
                    best_box = box;
                }
            }
            if (best_area > 0.0) {
                result.centers[static_cast<std::size_t>(slot)] =
                    cv::Point2f(best_box.x + 0.5f * best_box.width + roi.x,
                                best_box.y + 0.5f * best_box.height + roi.y);
                result.scores[static_cast<std::size_t>(slot)] = best_area;
                result.found[static_cast<std::size_t>(slot)] = true;
            } else {
                result.centers[static_cast<std::size_t>(slot)] = predicted;
            }
        }
        return result;
    }
} // namespace auto_aim::energy
