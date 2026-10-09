#include "keypoint_refiner.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include "common/types.hpp"

namespace auto_aim
{
    namespace
    {
        double angleFromVertical(const cv::Point2f& direction);

        // A light-bar candidate inside the refinement ROI, in full-image pixels.
        struct Bar {
            cv::Point2f center;
            cv::Point2f top;      // upper endpoint (smaller y)
            cv::Point2f bottom;   // lower endpoint
            double height = 0.0;
            double width = 0.0;   // 灯条短边（粗值，供亚像素剖面定搜索范围）
            double tilt = 0.0;    // degrees from the vertical
            bool subpixel = false;   // 端点/边缘是否已用剖面亚像素精修过
        };

        // ROI 内的一点做双线性采样；越界返回 -1。
        double bilinear(const cv::Mat& gray, const cv::Point2f& q)
        {
            if (q.x < 0.0f || q.y < 0.0f || q.x > gray.cols - 1.001f ||
                q.y > gray.rows - 1.001f) {
                return -1.0;
            }
            const int x0 = static_cast<int>(q.x), y0 = static_cast<int>(q.y);
            const float fx = q.x - x0, fy = q.y - y0;
            const uchar* r0 = gray.ptr<uchar>(y0);
            const uchar* r1 = gray.ptr<uchar>(y0 + 1);
            const double v00 = r0[x0], v01 = r0[x0 + 1];
            const double v10 = r1[x0], v11 = r1[x0 + 1];
            return (1 - fx) * (1 - fy) * v00 + fx * (1 - fy) * v01 +
                   (1 - fx) * fy * v10 + fx * fy * v11;
        }

        // 沿 direction 采样一维亮度剖面（在 origin 附近，半宽半长之外自动填 -1）。
        std::vector<double> profile1D(const cv::Mat& gray, const cv::Point2f& origin,
                                      const cv::Point2f& direction, double half_range,
                                      double step, int& zero_index)
        {
            const int n = static_cast<int>(2.0 * half_range / step) + 1;
            zero_index = n / 2;
            std::vector<double> out(static_cast<std::size_t>(n), -1.0);
            for (int i = 0; i < n; ++i) {
                const double s = (i - zero_index) * step;
                const cv::Point2f q = origin + static_cast<float>(s) * direction;
                out[static_cast<std::size_t>(i)] = bilinear(gray, q);
            }
            return out;
        }

        // 在剖面里找以峰值所在的连通段，两端用半高线性插值 → 亚像素起止下标。
        bool segmentEnds(const std::vector<double>& prof, int zero, double& lo, double& hi)
        {
            int peak = -1;
            double peak_v = -1e9;
            std::vector<double> valid;
            for (std::size_t i = 0; i < prof.size(); ++i) {
                if (prof[i] < 0) continue;
                valid.push_back(prof[i]);
                if (prof[i] > peak_v) { peak_v = prof[i]; peak = static_cast<int>(i); }
            }
            if (peak < 0 || valid.size() < 5) return false;
            std::nth_element(valid.begin(), valid.begin() + valid.size() / 10, valid.end());
            const double bg = valid[valid.size() / 10];
            if (peak_v - bg < 12.0) return false;      // 对比度太低，别乱精修
            const double thr = 0.5 * (peak_v + bg);

            auto walk = [&](int from, int to, int step) -> double {
                int prev = from;
                for (int i = from; i != to; i += step) {
                    if (prof[static_cast<std::size_t>(i)] < 0) break;
                    if (prof[static_cast<std::size_t>(i)] < thr) {
                        const double a = prof[static_cast<std::size_t>(prev)];
                        const double b = prof[static_cast<std::size_t>(i)];
                        const double w = (thr - a) / std::max(1e-6, b - a);
                        return (prev + w * (i - prev)) - zero;
                    }
                    prev = i;
                }
                return (prof[static_cast<std::size_t>(peak)] >= thr)
                           ? static_cast<double>(to - step) - zero
                           : 0.0;
            };
            lo = walk(peak, 0, -1);
            hi = walk(peak, static_cast<int>(prof.size()) - 1, 1);
            return hi > lo;
        }

        // 以 minAreaRect 的粗结果为先验，用亮度剖面把"灯条两端"和"灯条横向中心"
        // 精修到亚像素。minAreaRect 的端点会被灯条宽度污染且量化到像素，实测
        // 精修后反而比网络原值差；这里换成剖面插值。
        bool refineBarSubpixel(const cv::Mat& gray, const cv::Point2f& offset,
                               double coarse_width, Bar& bar)
        {
            cv::Point2f axis = bar.bottom - bar.top;
            const double len = cv::norm(axis);
            if (len < 4.0) return false;
            axis /= static_cast<float>(len);
            const cv::Point2f perp(-axis.y, axis.x);
            const cv::Point2f c = bar.center - offset;

            // 1) 沿长轴：半高交叉 → 亚像素两端
            int zero = 0;
            const double step = 0.5;
            const std::vector<double> prof =
                profile1D(gray, c, axis, 0.5 * len * 1.6, step, zero);
            double s_lo = 0.0, s_hi = 0.0;
            if (!segmentEnds(prof, zero, s_lo, s_hi)) return false;

            // 2) 沿短轴：在长轴中点处再取一次剖面 → 亚像素中心与半宽
            const cv::Point2f mid = c + static_cast<float>(0.5 * (s_lo + s_hi)) * axis;
            const double t_range = std::max(2.0, 0.75 * coarse_width);
            int zero_t = 0;
            const std::vector<double> prof_t = profile1D(gray, mid, perp, t_range, 0.5, zero_t);
            double t_lo = 0.0, t_hi = 0.0;
            const bool have_w = segmentEnds(prof_t, zero_t, t_lo, t_hi);

            const cv::Point2f center = mid +
                (have_w ? static_cast<float>(0.5 * (t_lo + t_hi)) : 0.0f) * perp;
            const float half_len = static_cast<float>(0.5 * (s_hi - s_lo));
            if (half_len < 2.0f) return false;
            bar.center = center + offset;
            bar.top = center + half_len * axis + offset;
            bar.bottom = center - half_len * axis + offset;
            bar.height = 2.0 * half_len;
            bar.tilt = angleFromVertical(bar.bottom - bar.top);
            bar.subpixel = true;
            return true;
        }

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
                bar.width = std::min(light.width, light.height);
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
        cv::Mat roi_gray;
        cv::cvtColor(roi_image, roi_gray, cv::COLOR_BGR2GRAY);
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
            std::vector<Bar> bars = findBars(roi_image, roi, local);
            if (bars.size() < 2) continue;

            int left_idx = -1;
            int right_idx = -1;
            double left_cost = std::numeric_limits<double>::max();
            double right_cost = std::numeric_limits<double>::max();
            for (std::size_t i = 0; i < bars.size(); ++i) {
                const Bar& bar = bars[i];
                const double to_left = cv::norm(bar.center - left_mid);
                const double to_right = cv::norm(bar.center - right_mid);
                if (to_left < left_cost) { left_cost = to_left; left_idx = static_cast<int>(i); }
                if (to_right < right_cost) { right_cost = to_right; right_idx = static_cast<int>(i); }
            }
            if (left_idx < 0 || right_idx < 0 || left_idx == right_idx) continue;

            // 亚像素精修：以 minAreaRect 的粗结果为先验，用亮度剖面重定端点与横向中心。
            // 失败就保留粗值（几何一致性检查仍然按最终值做）。
            refineBarSubpixel(roi_gray, cv::Point2f(static_cast<float>(roi.x),
                                                   static_cast<float>(roi.y)),
                              bars[left_idx].width, bars[left_idx]);
            refineBarSubpixel(roi_gray, cv::Point2f(static_cast<float>(roi.x),
                                                   static_cast<float>(roi.y)),
                              bars[right_idx].width, bars[right_idx]);

            const Bar* left = &bars[left_idx];
            const Bar* right = &bars[right_idx];

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
