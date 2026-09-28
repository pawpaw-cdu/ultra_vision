#ifndef TYPES_HPP
#define TYPES_HPP

#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <opencv2/opencv.hpp>

namespace auto_aim
{
    enum ArmorType { small, large };

    struct Light : public cv::RotatedRect
    {
        int color = 0;
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

        // 灯条长轴与图像竖直方向的夹角。灯条是一条无向线段，长轴方向
        // (0,-1) 和 (0,1) 必须给出同一个倾角，所以先用 atan2 求方向角，
        // 再折叠到 [-90, 90)。若改用 acos(dir·vertical)，竖直灯条会被算成
        // 0 或 180 度，使 |tilt| < max_angle 的筛选随机丢弃一半竖直灯条，
        // 导致同一块装甲板的两根灯条经常只剩一根、无法配对。
        float tilt_deg = std::atan2(dir.x, -dir.y) * 180.0f / CV_PI;
        if (tilt_deg >= 90.0f) tilt_deg -= 180.0f;
        else if (tilt_deg < -90.0f) tilt_deg += 180.0f;
        this->tilt_angle = tilt_deg;
    
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
