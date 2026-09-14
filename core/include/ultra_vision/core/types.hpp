#ifndef TYPES_HPP
#define TYPES_HPP

#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <opencv2/opencv.hpp>

namespace ultra_vision
{
    enum ArmorType { small, large };

    struct Light : public cv::RotatedRect
    {
        int color;
        double min_ratio = 0.1;
        double max_ratio = 0.4;
        double max_light_angle = 40.0;

        double tilt_angle;
        double height, width;
        cv::Point2f center, top, bottom;
        cv::RotatedRect rect;

            explicit Light(cv::RotatedRect rect) : cv::RotatedRect(rect)
    {
        cv::Point2f p[4];
        rect.points(p);
        this->center = rect.center;
        float w = rect.size.width;
        float h = rect.size.height;
        float angle_deg = rect.angle;

        // 确定长边、短边和长轴方向角
        float long_axis_rad;
        if (w >= h) {
            this->height = w;
            this->width = h;
            long_axis_rad = angle_deg * CV_PI / 180.0f;
        } else {
            this->height = h;
            this->width = w;
            long_axis_rad = (angle_deg + 90.0f) * CV_PI / 180.0f;
        }

        // 长轴单位向量
        cv::Point2f dir(cos(long_axis_rad), sin(long_axis_rad));
        float half_len = this->height / 2.0f;
        this->top    = this->center - dir * half_len;
        this->bottom = this->center + dir * half_len;

        // 计算有符号倾斜角（与垂直轴的夹角）
        cv::Point2f vertical(0, 1);
        float dot = dir.x * vertical.x + dir.y * vertical.y;
        float angle_rad = std::acos(std::clamp(dot, -1.0f, 1.0f));

        //float angle_rad = std::atan2(std::abs(p[1].x - p[2].x),std::abs(p[1].y - p[2].y));
        //float sign = 1.0f;
        
        float sign = (dir.x > 0) ? 1.0f : -1.0f;
        this->tilt_angle = sign * angle_rad * 180.0f / CV_PI;
    
        }

        bool isValid() const {
            float ratio = width / height;
            bool ratio_ok = (min_ratio < ratio && ratio < max_ratio);
            bool angle_ok = (std::abs(tilt_angle) < max_light_angle);
            return ratio_ok && angle_ok;
        }

        Light() = default;
    };

    struct Armor
    {
        Light left, right;
        double center;
        int number;
        bool solve_result;
        ArmorType armor_type;
        std::vector<cv::Point2f> Points_2D;
        std::vector<cv::Point3f> Points_3D;
        cv::Mat rvec, tvec;

        Armor(const Light& light1, const Light& light2) {
            if (light1.center.x > light2.center.x) {
                left = light2;
                right = light1;
            } else {
                left = light1;
                right = light2;
            }
        }
        Armor() = default;
    };
}

#endif