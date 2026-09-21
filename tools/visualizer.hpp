#ifndef VISUALIZER_HPP
#define VISUALIZER_HPP

#include <iostream>
#include <vector>
#include <string>

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>

namespace tools{

    inline void draw_points(
    cv::Mat & img,const std::vector<std::vector<cv::Point>> & contours,
    const cv::Scalar &color = cv::Scalar(0, 0, 255), int thickness = 3)
    {
        cv::drawContours(img, contours, -1, color, thickness);
    }

    inline void draw_points(
    cv::Mat &img, const std::vector<cv::Point> & points,
    const cv::Scalar &color = cv::Scalar(0, 0, 255), int thickness = 2)
{
    std::vector<std::vector<cv::Point>> contours;
    contours.emplace_back(points);
    cv::drawContours(img, contours, -1, color, thickness);
}

    inline void draw_points(
    cv::Mat &img, const std::vector<cv::Point2f> &points,
    const cv::Scalar &color = cv::Scalar(0, 0, 255), int thickness = 2)
{
    std::vector<cv::Point> int_points(points.begin(), points.end());
    draw_points(img, int_points, color, thickness);
}

};      //namespace tools


#endif