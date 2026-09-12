#ifndef ENERGY_DETECTOR_TYPES_HPP
#define ENERGY_DETECTOR_TYPES_HPP

#include <opencv2/opencv.hpp>
#include <vector>

namespace energy {

enum class BladeStatus { ACTIVE, NOACTIVE, NOTHING };

struct Blade {
    std::vector<cv::Point2f> centers;
    std::vector<BladeStatus> status;
};

struct EnergyBlade {
    cv::Point2f center;
    int id;
    float angle_rad;
    cv::Point2f object_points[4];
    cv::Point2f transforme_points[4];
};

struct DetectedBlade {
    cv::Point2f center;
    cv::Point2f top_left;
    cv::Point2f top_right;
};

struct SolvePnPresult {
    cv::Mat tvec;
    cv::Mat rvec;
    double distance;
};

struct DetectorResult {
    std::vector<DetectedBlade> blades;
    cv::Point2f r_center;
    float r_radius;
    bool r_found;
};

struct TrackerResult {
    std::vector<EnergyBlade> blades;
    cv::Point2f model_center;
    float model_radius;
    bool valid;
};

} // namespace energy

#endif
