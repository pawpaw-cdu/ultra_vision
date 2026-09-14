#include "ultra_vision/perception/detector.hpp"
#include <iostream>
#include <algorithm>
#include <utility>

namespace ultra_vision
{
    Detector::Detector(const cv::Mat& img, const DetectorConfig& cfg)
        : config_(cfg)
    {
        this->clone_img = img;
    }

    void Detector::gray_img(const cv::Mat& clone_img, cv::Mat& gray_img)
    {
        cv::cvtColor(clone_img, gray_img, cv::COLOR_BGR2GRAY);
    }

    void Detector::binary_img(const cv::Mat& gray_img, cv::Mat& img_, int binary_threshold)
    {
        //cv::threshold(gray_img, img_, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
        cv::threshold(gray_img, img_, binary_threshold, 255, cv::THRESH_BINARY);
    }

    void Detector::open_close_img(const cv::Mat& binary_img, cv::Mat& img_)
    {
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));  // 使用矩形核
        //cv::dilate(binary_img, img_, kernel);
        cv::morphologyEx(binary_img, binary_img, cv::MORPH_CLOSE, kernel);
        cv::morphologyEx(binary_img, img_, cv::MORPH_OPEN, kernel);
        //cv::dilate(binary_img, img_, kernel);

    }

    void Detector::find_contours(const cv::Mat& bilated_img, std::vector<std::vector<cv::Point>>& contours)
    {
        cv::findContours(bilated_img, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    }

    void Detector::find_lights(const std::vector<std::vector<cv::Point>>& contours,
                           std::vector<Light>& light_,
                           const cv::Mat& img,
                           int detector_color)
{
    //std::cout << "Total contours: " << contours.size() << std::endl;
    for (const auto& contour : contours) 
    {
    //    std::cout << "Contour size: " << contour.size() << std::endl;
    //    if (contour.size() < config_.light_min_contour_points) {
    //        std::cout << "  -> rejected: too few points (" << contour.size() << " < " << config_.light_min_contour_points << ")" << std::endl;
    //        continue;
    //    }

        cv::RotatedRect rrect = cv::minAreaRect(contour);
        Light light(rrect);
        light.min_ratio = config_.light_min_ratio;
        light.max_ratio = config_.light_max_ratio;
        light.max_light_angle = config_.light_max_angle;

        // 打印原始几何信息

        cv::Rect rect = cv::boundingRect(contour);
        cv::Rect roi_rect = rect & cv::Rect(0, 0, img.cols, img.rows);
        if (roi_rect.width <= 0 || roi_rect.height <= 0) {
            continue;
        }

        cv::Mat img_roi = img(roi_rect);
        // Build the contour mask in the small ROI. Allocating a full-frame mask
        // for every contour made detection cost proportional to image area.
        cv::Mat mask_roi = cv::Mat::zeros(roi_rect.size(), CV_8U);
        std::vector<cv::Point> local_contour;
        local_contour.reserve(contour.size());
        for (const auto& point : contour) {
            local_contour.emplace_back(point.x - roi_rect.x, point.y - roi_rect.y);
        }
        std::vector<std::vector<cv::Point>> local_contours = {std::move(local_contour)};
        cv::drawContours(mask_roi, local_contours, 0, 255, cv::FILLED);

    bool color_ok = true;
    if (config_.color_use_detect) {
        if (config_.use_hsv) {
            // ========== HSV 方法 ==========
            cv::Mat hsv;
            cv::cvtColor(img_roi, hsv, cv::COLOR_BGR2HSV);
            cv::Mat mask_target;
            if (detector_color == 1) { // 红色
                cv::Mat mask1, mask2;
                cv::inRange(hsv, cv::Scalar(config_.hue_red_low1, config_.sat_min, config_.val_min),
                                 cv::Scalar(config_.hue_red_high1, 255, 255), mask1);
                cv::inRange(hsv, cv::Scalar(config_.hue_red_low2, config_.sat_min, config_.val_min),
                                 cv::Scalar(config_.hue_red_high2, 255, 255), mask2);
                cv::bitwise_or(mask1, mask2, mask_target);
            } else { // 蓝色
                cv::inRange(hsv, cv::Scalar(config_.hue_blue_low, config_.sat_min, config_.val_min),
                                 cv::Scalar(config_.hue_blue_high, 255, 255), mask_target);
            }
            // 只保留与灯条掩码交集的部分
            cv::bitwise_and(mask_target, mask_roi, mask_target);
            int total_pixels = cv::countNonZero(mask_roi);
            int target_pixels = cv::countNonZero(mask_target);
            double ratio = (total_pixels > 0) ? (double)target_pixels / total_pixels : 0.0;
            color_ok = (ratio > config_.color_area_ratio);
        } else {
            // ========== BGR 方法（原均值法）==========
            cv::Scalar mean_bgr = cv::mean(img_roi, mask_roi);
            if (detector_color == 1) { // 红色
                if (mean_bgr[2] <= mean_bgr[0] * config_.color_red_threshold)
                    color_ok = false;
            } else {
                if (mean_bgr[0] <= mean_bgr[2] * config_.color_blue_threshold)
                    color_ok = false;
            }
        }
    }

    if (!color_ok) continue;

        // 形状筛选
        bool valid = light.isValid();
        //std::cout << "  isValid() = " << valid << std::endl;
        if (valid) {
            light_.push_back(light);
        } else {
            //float ratio = light.width / light.height;
            //std::cout << "  -> rejected by isValid: ratio=" << ratio 
            //          << " (min=" << light.min_ratio << ", max=" << light.max_ratio 
            //          << "), angle=" << light.tilt_angle 
            //          << " (max=" << light.max_light_angle << ")" << std::endl;
            continue;
        }
    }
}
    // 检查由两个灯条构成的矩形区域内是否包含其他灯条
    bool Detector::containLight(const Light& light1, const Light& light2, const std::vector<Light>& lights)
    {
        std::vector<cv::Point2f> points = {light1.top, light2.top, light1.bottom, light2.bottom};
        cv::Rect bounding_rect = cv::boundingRect(points);
        for (const auto& test_light : lights) {
            if (test_light.center == light1.center || test_light.center == light2.center) continue;
            if (bounding_rect.contains(test_light.top) ||
                bounding_rect.contains(test_light.bottom) ||
                bounding_rect.contains(test_light.center)) {
                return true;
            }
        }
        return false;
    }

    // 判断两个灯条是否能构成装甲板，并返回类型
    bool Detector::isArmor(const Light& light1, const Light& light2, ArmorType& type)
    {
        // 1. 高度比约束
        float h1 = light1.height, h2 = light2.height;
        float height_ratio = (h1 > h2) ? (h2 / h1) : (h1 / h2);
        if (height_ratio < config_.armor_height_ratio_min || height_ratio > config_.armor_height_ratio_max)
            return false;

        // 2. 左右灯条倾斜角和约束（应接近0，因为方向相反）
        float angle_sum = std::abs(light1.tilt_angle + light2.tilt_angle);
        if (angle_sum > config_.armor_angle_diff_max)
            return false;

        // 3. 水平距离与灯条高度之比
        float center_dist = std::abs(light1.center.x - light2.center.x);
        float avg_height = (h1 + h2) / 2.0f;
        float width_ratio = center_dist / avg_height;
        if (width_ratio < config_.armor_width_to_height_min || width_ratio > config_.armor_width_to_height_max)
            return false;

        // 判断大小
        if (width_ratio > config_.armor_large_ratio_thresh)
            type = large;
        else
            type = small;
        return true;
    }

    void Detector::find_armors(const std::vector<Light>& lights, std::vector<Armor>& armors)
    {
        armors.clear();
        std::size_t n = lights.size();
        if (n < 2) return;

        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = i + 1; j < n; ++j) {
                const Light& l1 = lights[i];
                const Light& l2 = lights[j]; 

                // 颜色必须与敌方一致
                if (l1.color != config_.enemy_color || l2.color != config_.enemy_color)
                    continue;

                // 确保左右顺序（左灯条中心 x 更小）
                //if (l1.center.x > l2.center.x) continue;

                // 检查区域内是否包含其他灯条
                 if (containLight(l1, l2, lights))
                     continue;

                ArmorType type;
                if (isArmor(l1, l2, type)) {
                    Armor armor(l1, l2);
                    armor.armor_type = type;
                    armors.push_back(armor);
                }
            }
        }
    }
}
