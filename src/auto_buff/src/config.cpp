#include "../include/config.hpp"
#include <iostream>

namespace energy {

bool ConfigManager::load(const std::string& detector_yaml, const std::string& tracker_yaml,
                         const std::string& solvePnP_yaml) {
    try {
        // 加载检测器配置
        YAML::Node det_node = YAML::LoadFile(detector_yaml);
        auto load_range = [](const YAML::Node& node) {
            HsvRange range;
            range.h_min = node["h_min"].as<int>();
            range.h_max = node["h_max"].as<int>();
            range.s_min = node["s_min"].as<int>();
            range.v_min = node["v_min"].as<int>();
            return range;
        };
        detector_cfg_.color = det_node["color"].as<int>();
        detector_cfg_.red_range1 = load_range(det_node["red_range1"]);
        detector_cfg_.red_range2 = load_range(det_node["red_range2"]);
        detector_cfg_.blue_range = load_range(det_node["blue_range"]);
        detector_cfg_.r_red_range1 = load_range(det_node["r_red_range1"]);
        detector_cfg_.r_red_range2 = load_range(det_node["r_red_range2"]);
        detector_cfg_.r_blue_range = load_range(det_node["r_blue_range"]);
        detector_cfg_.morph_size = det_node["morph_size"].as<int>();
        detector_cfg_.fanblade_min_area = det_node["fanblade_min_area"].as<double>();
        detector_cfg_.fanblade_max_area = det_node["fanblade_max_area"].as<double>();
        detector_cfg_.fanblade_ratio_min = det_node["fanblade_ratio_min"].as<double>();
        detector_cfg_.fanblade_ratio_max = det_node["fanblade_ratio_max"].as<double>();
        detector_cfg_.fanblade_circularity_max = det_node["fanblade_circularity_max"].as<double>();
        detector_cfg_.fanblade_solidity_min = det_node["fanblade_solidity_min"].as<double>();
        detector_cfg_.r_min_area = det_node["r_min_area"].as<double>();
        detector_cfg_.r_max_area = det_node["r_max_area"].as<double>();
        detector_cfg_.r_min_radius = det_node["r_min_radius"].as<double>();
        detector_cfg_.r_max_radius = det_node["r_max_radius"].as<double>();
        detector_cfg_.debug = det_node["debug"].as<int>();

        // 加载跟踪器配置
        YAML::Node tr_node = YAML::LoadFile(tracker_yaml);
        tracker_cfg_.center_smooth = tr_node["center_smooth"].as<double>();
        tracker_cfg_.radius_smooth = tr_node["radius_smooth"].as<double>();
        tracker_cfg_.angle_smooth = tr_node["angle_smooth"].as<double>();
        tracker_cfg_.blade_angle_smooth = tr_node["blade_angle_smooth"].as<double>();
        tracker_cfg_.angular_velocity_deg = tr_node["angular_velocity_deg"].as<double>();
        tracker_cfg_.max_lost_frames = tr_node["max_lost_frames"].as<int>();
        tracker_cfg_.match_distance_threshold = tr_node["match_distance_threshold"].as<double>();
        tracker_cfg_.num_blades = tr_node["num_blades"].as<int>();
        tracker_cfg_.jump_threshold_px = tr_node["jump_threshold_px"].as<double>();
        
        // 加载解算配置
        YAML::Node solve_node = YAML::LoadFile(solvePnP_yaml);
        auto cam_mat_vec = solve_node["camera_matrix"].as<std::vector<double>>();
        if (cam_mat_vec.size() == 9) {
            std::copy(cam_mat_vec.begin(), cam_mat_vec.end(), solvePnP_cfg_.camera_matrix.begin());
        }
        solvePnP_cfg_.distortion_coefficients = solve_node["dist_coeffs"].as<std::vector<double>>();

        return true;
    } catch (const std::exception& e) {
        std::cerr << "Failed to load config: " << e.what() << std::endl;
        return false;
    }
}

} // namespace energy
