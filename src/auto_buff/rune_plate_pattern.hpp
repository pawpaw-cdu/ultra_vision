#ifndef AUTO_AIM_ENERGY_RUNE_PLATE_PATTERN_HPP
#define AUTO_AIM_ENERGY_RUNE_PLATE_PATTERN_HPP

#include <opencv2/core.hpp>

namespace auto_aim::energy
{
    struct RunePlatePattern
    {
        bool valid = false;
        double pvd = 0.0;
        double phase_rad = 0.0;
        int roi_px = 0;
    };

    /// @brief Color-independent four-fold peak/valley score for a rune plate.
    ///
    /// A ring-like inactive face and a four-arm activated pattern have different
    /// angular energy distributions. PVD is the normalized difference between
    /// the four strongest angular peaks and the valleys between them. The score
    /// is deliberately returned as evidence; callers decide the threshold.
    RunePlatePattern scoreRunePlatePattern(const cv::Mat& bgr, cv::Point2f center,
                                           float plate_span_px);
} // namespace auto_aim::energy

#endif // AUTO_AIM_ENERGY_RUNE_PLATE_PATTERN_HPP
