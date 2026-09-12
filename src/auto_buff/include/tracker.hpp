#ifndef ENERGY_DETECTOR_TRACKER_HPP
#define ENERGY_DETECTOR_TRACKER_HPP

#include "types.hpp"
#include "config.hpp"
#include <vector>

namespace energy {

class Tracker {
public:
    explicit Tracker(const TrackerConfig& cfg);
    TrackerResult update(const DetectorResult& det, double dt);
    void reset();

private:
    struct BladeTracker {
        int id;
        cv::Point2f center;
        cv::Point2f top_left;
        cv::Point2f top_right;
        float top_extent;
        float half_width;
        float angle;
        bool lost;
        int lost_count;
    };
    TrackerConfig cfg_;
    std::vector<BladeTracker> trackers_;
    std::vector<float> smoothed_angles_;
    std::vector<bool> angle_initialized_;
    cv::Point2f smoothed_center_;
    float smoothed_radius_;
    bool first_frame_;
    double last_timestamp_;
    double last_ref_angle_;
    bool has_ref_angle_;
    double predicted_ref_angle_;

    void applyDetection(BladeTracker& tr, const DetectedBlade& det);
    std::vector<int> hungarianMatch(const std::vector<DetectedBlade>& detected);
    void updateTrackers(const std::vector<DetectedBlade>& detected, double dt);
    void predictMissing(double dt);
    TrackerResult generateResult(const cv::Point2f& center, float radius);
};

} // namespace energy

#endif
