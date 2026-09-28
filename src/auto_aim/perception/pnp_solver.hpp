#ifndef PNP_SOLVER_HPP
#define PNP_SOLVER_HPP

#include <opencv2/opencv.hpp>
#include "common/types.hpp"

namespace auto_aim
{
    // 装甲板等效物点尺寸（PnP 与显示共用）。数值是**等效尺寸**，不是标准板尺寸：
    // 网络的四个关键点不是 135x125 板的四角，见 configs/tracker.yaml 的 pnp 段注释。
    struct PnpGeometry
    {
        float small_width = 0.135f;
        float large_width = 0.225f;
        float height = 0.055f;
    };

    extern cv::Mat CAMERA_MATRIX;
    extern cv::Mat DIST_COEFFS;

    std::vector<cv::Point3f> getArmor3DPoints(int type, float small_width, float large_width, float height);
    bool solveArmorPnP(auto_aim::Armor& armor,
                       const cv::Mat& camera_matrix,
                       const cv::Mat& dist_coeffs,
                       float small_width = 0.135f,
                       float large_width = 0.225f,
                       float height = 0.055f);
}

#endif
