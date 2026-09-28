#ifndef DETECTOR_HPP
#define DETECTOR_HPP

#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include "common/types.hpp"

namespace auto_aim
{
    struct DetectorConfig {
        // 传统灯条检测器的参数。默认值 = 本工程一直在用的那一组
        // （原 configs/detector.yaml 的 light/armor/color 段，已经搬到这里）。
        // 神经网络开着的时候（detector.neural.enabled: true）这一整套都不参与，
        // 所以不再往 YAML 里放；要调传统路径就改这里，或在 detector.yaml 里
        // 重新写回同名的 light/armor/color 段（加载器仍会读）。
        int enemy_color = 0;
        int binary_threshold = 121;
        double light_min_ratio = 0.01;
        double light_max_ratio = 0.15;
        double light_max_angle = 30.0;
        int light_min_contour_points = 5;
        double armor_height_ratio_min = 0.70;
        double armor_height_ratio_max = 1.25;
        double armor_angle_diff_max = 30.0; // 角度和阈值（左右灯条角度和应接近0）
        double armor_width_to_height_min = 0.8;
        double armor_width_to_height_max = 3.0;
        double armor_large_ratio_thresh = 3.0;
        bool color_use_detect = true;
        double color_red_threshold = 1.0;
        double color_blue_threshold = 1.0;

        bool use_hsv = false;       // true: 使用 HSV, false: 使用 BGR
        int hue_red_low1 = 0;       // 红色下限1 (0~10)
        int hue_red_high1 = 35;
        int hue_red_low2 = 135;     // 红色下限2 (160~180)
        int hue_red_high2 = 180;
        int hue_blue_low = 90;      // 蓝色下限 (100~130)
        int hue_blue_high = 135;
        int sat_min = 80;           // 最小饱和度（过滤低饱和区域）
        int val_min = 80;           // 最小明度（过滤过暗区域）
        double color_area_ratio = 0.3;
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
