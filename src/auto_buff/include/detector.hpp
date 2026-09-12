#ifndef ENERGY_DETECTOR_DETECTOR_HPP
#define ENERGY_DETECTOR_DETECTOR_HPP

#include "types.hpp"
#include "config.hpp"

namespace energy {

class Detector {
public:
    explicit Detector(const DetectorConfig& cfg);
    DetectorResult process(const cv::Mat& bgr);

private:
    DetectorConfig cfg_;
    cv::Mat makeColorMask(const cv::Mat& hsv, int color, bool r_center) const;
    cv::Mat preprocessMask(const cv::Mat& bgr, int color);
    std::vector<DetectedBlade> extractBlades(const cv::Mat& mask, const cv::Point2f& model_center);
    bool detectRCenter(const cv::Mat& bgr, cv::Point2f& center, float& radius,
                       cv::Mat* center_mask = nullptr);
};

} // namespace energy

#endif
