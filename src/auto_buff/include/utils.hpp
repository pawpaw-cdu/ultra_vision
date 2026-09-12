#ifndef ENERGY_DETECTOR_UTILS_HPP
#define ENERGY_DETECTOR_UTILS_HPP

#include <opencv2/opencv.hpp>
#include <vector>
#include "types.hpp"

namespace energy {

float normalizeAngle(float angle);
void showBladesROI(const cv::Mat& src, const std::vector<cv::Point2f>& blade_centers, int roi_size = 80);
void drawModel(cv::Mat& img, const TrackerResult& result);
} // namespace energy

#endif
