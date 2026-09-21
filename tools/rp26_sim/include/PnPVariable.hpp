#pragma once
//
// 影子头文件：RP-26Rune-main/src/core/algorithm/power_rune/config/PnPVariable.hpp 的
// 等价实现。原文件所在目录不能被加到 include 路径里（否则它同目录的 json.hpp 会被
// 优先命中，见 include/json.hpp 的说明），所以这里重新声明同样的全局量。
//
// 语义完全一致：CAM=相机内参 3x3，DIS=畸变系数 1x5，均在静态初始化期从
// J_POWER_RUNE 的 camera 段读取。

#include <opencv2/opencv.hpp>

#include "json.hpp"

inline cv::Matx<double, 3, 3> CAM;
inline cv::Matx<double, 1, 5> DIS;

inline auto initCamAndDis = []()
{
    const cv::FileNode cam_node = J_POWER_RUNE.config_["camera"]["cam"];
    const cv::FileNode dis_node = J_POWER_RUNE.config_["camera"]["dis"];

    cv::Mat cam;
    cam_node >> cam;
    if (cam.rows == 3 && cam.cols == 3)
    {
        cam.convertTo(cam, CV_64F);
        for (int row = 0; row < 3; ++row)
            for (int col = 0; col < 3; ++col)
                CAM(row, col) = cam.at<double>(row, col);
    }

    cv::Mat dis;
    dis_node >> dis;
    if (!dis.empty())
    {
        cv::Mat dis_row = dis.reshape(1, 1);
        dis_row.convertTo(dis_row, CV_64F);
        for (int col = 0; col < 5 && col < dis_row.cols; ++col)
            DIS(0, col) = dis_row.at<double>(0, col);
    }
    return 0;
};

inline auto INIT_CAM_AND_DIS = initCamAndDis();
