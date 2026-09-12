#include "../include/detector.hpp"
#include <iostream>

namespace energy {

Detector::Detector(const DetectorConfig& cfg) : cfg_(cfg) {}

namespace {

cv::Scalar hsvLower(const HsvRange& range) {
    return cv::Scalar(range.h_min, range.s_min, range.v_min);
}

cv::Scalar hsvUpper(const HsvRange& range) {
    return cv::Scalar(range.h_max, 255, 255);
}

} // namespace

cv::Mat Detector::makeColorMask(const cv::Mat& hsv, int color, bool r_center) const {
    cv::Mat mask;
    if (color == 1) {
        const HsvRange& range1 = r_center ? cfg_.r_red_range1 : cfg_.red_range1;
        const HsvRange& range2 = r_center ? cfg_.r_red_range2 : cfg_.red_range2;
        cv::Mat mask1, mask2;
        cv::inRange(hsv, hsvLower(range1), hsvUpper(range1), mask1);
        cv::inRange(hsv, hsvLower(range2), hsvUpper(range2), mask2);
        cv::bitwise_or(mask1, mask2, mask);
    } else {
        const HsvRange& range = r_center ? cfg_.r_blue_range : cfg_.blue_range;
        cv::inRange(hsv, hsvLower(range), hsvUpper(range), mask);
    }
    return mask;
}

cv::Mat Detector::preprocessMask(const cv::Mat& bgr, int color) {
    cv::Mat hsv;
    cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);
    cv::Mat mask = makeColorMask(hsv, color, false);
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(cfg_.morph_size, cfg_.morph_size));
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);
    if (cfg_.debug) {
        cv::Mat debug_view;
        cv::resize(mask, debug_view, cv::Size(960, 540));
        cv::imshow("mask_blade", debug_view);
    }
    return mask;
}

std::vector<DetectedBlade> Detector::extractBlades(const cv::Mat& mask, const cv::Point2f& model_center) {
    std::vector<DetectedBlade> blades;
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    for (const auto& cnt : contours) {
        double area = cv::contourArea(cnt);
        if (area < cfg_.fanblade_min_area || area > cfg_.fanblade_max_area) continue;
        cv::RotatedRect rect = cv::minAreaRect(cnt);
        float w = rect.size.width, h = rect.size.height;
        float ratio = std::min(w, h) / std::max(w, h);
        if (ratio < cfg_.fanblade_ratio_min || ratio > cfg_.fanblade_ratio_max) continue;
        double perimeter = cv::arcLength(cnt, true);
        double circularity = 4 * CV_PI * area / (perimeter * perimeter);
        if (circularity > cfg_.fanblade_circularity_max) continue;
        std::vector<cv::Point> hull;
        cv::convexHull(cnt, hull);
        double hull_area = cv::contourArea(hull);
        double solidity = area / hull_area;
        if (solidity < cfg_.fanblade_solidity_min) continue;

        DetectedBlade blade;
        cv::Moments m = cv::moments(cnt);
        if (m.m00 <= 0) continue;
        blade.center = cv::Point2f(m.m10 / m.m00, m.m01 / m.m00);

        // 顶部边取两条短边中离模型圆心更远的一条，即扇叶外沿
        cv::Point2f corners[4];
        rect.points(corners);
        float l01 = cv::norm(corners[0] - corners[1]);
        float l12 = cv::norm(corners[1] - corners[2]);
        int short_edges[2];
        if (l01 <= l12) {
            short_edges[0] = 0; // 边 (0,1)
            short_edges[1] = 2; // 边 (2,3)
        } else {
            short_edges[0] = 1; // 边 (1,2)
            short_edges[1] = 3; // 边 (3,0)
        }
        int best = short_edges[0];
        float best_dist = -1.0f;
        for (int k = 0; k < 2; ++k) {
            int e = short_edges[k];
            cv::Point2f mid = (corners[e] + corners[(e + 1) % 4]) * 0.5f;
            float d = cv::norm(mid - model_center);
            if (d > best_dist) {
                best_dist = d;
                best = e;
            }
        }
        blade.top_left = corners[best];
        blade.top_right = corners[(best + 1) % 4];
        blades.push_back(blade);
    }
    return blades;
}

bool Detector::detectRCenter(const cv::Mat& bgr, cv::Point2f& center, float& radius,
                             cv::Mat* center_mask) {
    cv::Mat hsv;
    cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);
    cv::Mat mask = makeColorMask(hsv, cfg_.color, true);
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(cfg_.morph_size, cfg_.morph_size));
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);
    if (center_mask) {
        *center_mask = mask.clone();
    }
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    for (const auto& cnt : contours) {
        double area = cv::contourArea(cnt);
        if (area < cfg_.r_min_area || area > cfg_.r_max_area) continue;
        cv::RotatedRect rect = cv::minAreaRect(cnt);
        float w = rect.size.width, h = rect.size.height;
        float ratio = std::min(w, h) / std::max(w, h);
        if (ratio < 0.7) continue;
        double perimeter = cv::arcLength(cnt, true);
        double circularity = 4 * CV_PI * area / (perimeter * perimeter);
        if (circularity < 0.3) continue;
        cv::Point2f cnt_center;
        float cnt_radius;
        cv::minEnclosingCircle(cnt, cnt_center, cnt_radius);
        if (cnt_radius >= cfg_.r_min_radius && cnt_radius <= cfg_.r_max_radius) {
            center = cnt_center;
            radius = 1.2*cnt_radius;
            return true;
        }
    }
    return false;
}

DetectorResult Detector::process(const cv::Mat& bgr) {
    DetectorResult result;
    result.r_found = false;
    cv::Mat center_mask;
    result.r_found = detectRCenter(bgr, result.r_center, result.r_radius,
                                   cfg_.debug ? &center_mask : nullptr);
    if (cfg_.debug && !center_mask.empty()) {
        cv::Mat debug_view;
        cv::resize(center_mask, debug_view, cv::Size(960, 540));
        cv::imshow("mask_center", debug_view);
    }
    cv::Mat mask = preprocessMask(bgr, cfg_.color);
    cv::Point2f model_center = result.r_found ? result.r_center
                                              : cv::Point2f(bgr.cols / 2.0f, bgr.rows / 2.0f);
    result.blades = extractBlades(mask, model_center);
    return result;
}

} // namespace energy
