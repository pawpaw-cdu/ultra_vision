#ifndef ENERGY_DETECTOR_CONFIG_HPP
#define ENERGY_DETECTOR_CONFIG_HPP

#include <string>
#include <yaml-cpp/yaml.h>

namespace energy {

struct HsvRange {
    int h_min;
    int h_max;
    int s_min;
    int v_min;
};

struct SolvePnPconfig {
    std::array<double, 9> camera_matrix{};
    std::vector<double> distortion_coefficients;
};

struct DetectorConfig {
    int color;                   // 1:red, 0:blue
    HsvRange red_range1;
    HsvRange red_range2;
    HsvRange blue_range;
    HsvRange r_red_range1;
    HsvRange r_red_range2;
    HsvRange r_blue_range;
    int morph_size;
    double fanblade_min_area, fanblade_max_area;
    double fanblade_ratio_min, fanblade_ratio_max;
    double fanblade_circularity_max;
    double fanblade_solidity_min;
    double r_min_area, r_max_area, r_min_radius, r_max_radius;
    int debug;
};

struct TrackerConfig {
    double center_smooth;
    double radius_smooth;
    double angle_smooth;
    double blade_angle_smooth;
    double angular_velocity_deg;
    int max_lost_frames;
    double match_distance_threshold;
    int num_blades;
    double jump_threshold_px;
};

class ConfigManager {
public:
    bool load(const std::string& detector_yaml, const std::string& tracker_yaml, const std::string& solvePnP_yaml);
    const DetectorConfig& getDetector() const { return detector_cfg_; }
    const TrackerConfig& getTracker() const { return tracker_cfg_; }
    const SolvePnPconfig& getSolvePnP() const {return solvePnP_cfg_;}
private:
    SolvePnPconfig solvePnP_cfg_;
    DetectorConfig detector_cfg_;
    TrackerConfig tracker_cfg_;
};

} // namespace energy

#endif
