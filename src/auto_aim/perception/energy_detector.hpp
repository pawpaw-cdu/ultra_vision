#ifndef ENERGY_DETECTOR_HPP
#define ENERGY_DETECTOR_HPP

#include <opencv2/opencv.hpp>
#include <vector>

namespace auto_aim {

struct EnergyBlade {
    cv::Point2f center;
    int id;
    float angle_rad;
};

struct EnergyDetectorConfig {
    float angular_velocity_deg = 60.0f;   // 恒定角速度（度/秒）
    int num_blades = 5;                   // 扇叶数量
    float jump_threshold_px = 10.0f;      // 圆心跳变阈值（像素）
    float min_radius_px = 20.0f;          // 最小半径（像素）
    float max_radius_px = 300.0f;         // 最大半径
    int max_track_lost_frames = 5;        // 最大丢失帧数
    float max_match_distance_ratio = 0.3f; // 匹配距离系数
};

class EnergyDetector {
public:
    EnergyDetector(const EnergyDetectorConfig& cfg = EnergyDetectorConfig());
    ~EnergyDetector() = default;

    bool process(const std::vector<cv::Point2f>& blade_centers,
                 std::vector<EnergyBlade>& predicted_blades);

    cv::Point2f getCircleCenter() const { return circle_center_; }
    float getRadius() const { return radius_px_; }
    bool isModelInitialized() const { return model_initialized_; }
    void reset();

private:
    EnergyDetectorConfig cfg_;
    cv::Point2f prev_center_;
    cv::Point2f current_center_;
    cv::Point2f circle_center_;
    float radius_px_ = 0.0f;
    float blade0_angle_rad_ = 0.0f;
    double last_timestamp_ = 0.0;
    int lost_frame_count_ = 0;
    bool model_initialized_ = false;

    bool findSameBlade(const std::vector<cv::Point2f>& centers, cv::Point2f& matched);
    bool computeCircleFromTwoPoints(const cv::Point2f& p1, const cv::Point2f& p2,
                                    float delta_angle_rad, cv::Point2f& center, float& radius);
    bool isCenterJump(const cv::Point2f& old_center, const cv::Point2f& new_center);
    void predictAllBlades(std::vector<EnergyBlade>& blades);
    double getTimestamp();
};

} // namespace auto_aim

#endif