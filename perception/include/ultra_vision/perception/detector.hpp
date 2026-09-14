#ifndef DETECTOR_HPP
#define DETECTOR_HPP

#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include "ultra_vision/core/types.hpp"

namespace ultra_vision
{
    struct DetectorConfig {
        int enemy_color;
        int binary_threshold;
        double light_min_ratio;
        double light_max_ratio;
        double light_max_angle;
        int light_min_contour_points;
        double armor_height_ratio_min;
        double armor_height_ratio_max;
        double armor_angle_diff_max;       // 角度和阈值（左右灯条角度和应接近0）
        double armor_width_to_height_min;
        double armor_width_to_height_max;
        double armor_large_ratio_thresh;
        bool color_use_detect;
        double color_red_threshold;
        double color_blue_threshold;

        bool use_hsv;               // true: 使用 HSV, false: 使用 BGR
        int hue_red_low1;           // 红色下限1 (0~10)
        int hue_red_high1;
        int hue_red_low2;           // 红色下限2 (160~180)
        int hue_red_high2;
        int hue_blue_low;           // 蓝色下限 (100~130)
        int hue_blue_high;
        int sat_min;                // 最小饱和度（过滤低饱和区域）
        int val_min;                // 最小明度（过滤过暗区域）
        double color_area_ratio; 
    };

    class Detector
    {
    private:
        DetectorConfig config_;
        std::vector<Light> lights_;
        std::vector<Armor> armors_;
    public:
        cv::Mat clone_img;
        Detector(const cv::Mat& img, const DetectorConfig& cfg);
        ~Detector() = default;

        void gray_img(const cv::Mat& clone_img, cv::Mat& gray_img);

        void binary_img(const cv::Mat& gray_img, cv::Mat& img_, int binary_threshold);
        
        void open_close_img(const cv::Mat& binary_img, cv::Mat& dilated_img);

        void find_contours(const cv::Mat& thres_img, std::vector<std::vector<cv::Point>>& contours);
        
        void find_lights(const std::vector<std::vector<cv::Point>>& contours,
                         std::vector<Light>& light_,
                         const cv::Mat& img,
                         int detector_color);
        void find_armors(const std::vector<Light>& lights, std::vector<Armor>& armors);

        bool containLight(const Light& light1, const Light& light2, const std::vector<Light>& lights);
        
        bool isArmor(const Light& light1, const Light& light2, ArmorType& type);
    };
}

#endif
