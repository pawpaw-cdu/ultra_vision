#include "energy_detector.hpp"
#include "common/standard_clock.hpp"
#include <cmath>
#include <chrono>
#include <algorithm>

namespace auto_aim {

EnergyDetector::EnergyDetector(const EnergyDetectorConfig& cfg) : cfg_(cfg) {
    reset();
}

void EnergyDetector::reset() {
    prev_center_ = cv::Point2f(0, 0);
    current_center_ = cv::Point2f(0, 0);
    circle_center_ = cv::Point2f(0, 0);
    radius_px_ = 0;
    blade0_angle_rad_ = 0;
    last_timestamp_ = 0;
    lost_frame_count_ = 0;
    model_initialized_ = false;
}

double EnergyDetector::getTimestamp() {
    return StandardClock::nowSeconds();
}

bool EnergyDetector::findSameBlade(const std::vector<cv::Point2f>& centers, cv::Point2f& matched) {
    if (centers.empty() || !model_initialized_) return false;
    float min_dist = 1e6;
    int best_idx = -1;
    for (size_t i = 0; i < centers.size(); ++i) {
        float dist = cv::norm(centers[i] - prev_center_);
        if (dist < min_dist) {
            min_dist = dist;
            best_idx = i;
        }
    }
    float max_match_dist = (radius_px_ > 0) ? radius_px_ * cfg_.max_match_distance_ratio : 50.0f;
    if (best_idx != -1 && min_dist < max_match_dist) {
        matched = centers[best_idx];
        return true;
    }
    return false;
}

bool EnergyDetector::computeCircleFromTwoPoints(const cv::Point2f& p1, const cv::Point2f& p2,
                                                float delta_angle_rad,
                                                cv::Point2f& center, float& radius) {
    float cos_a = std::cos(delta_angle_rad);
    float sin_a = std::sin(delta_angle_rad);
    float a = 1 - cos_a;
    float b = sin_a;

    float c = p2.x - p1.x * cos_a + p1.y * sin_a;
    float d = p2.y - p1.x * sin_a - p1.y * cos_a;

    float denom = a * a + b * b;
    if (std::fabs(denom) < 1e-6) return false;

    center.x = (a * c - b * d) / denom;
    center.y = (a * d + b * c) / denom;

    radius = cv::norm(p1 - center);
    return (radius > cfg_.min_radius_px && radius < cfg_.max_radius_px);
}

bool EnergyDetector::isCenterJump(const cv::Point2f& old_center, const cv::Point2f& new_center) {
    return cv::norm(old_center - new_center) > cfg_.jump_threshold_px;
}

void EnergyDetector::predictAllBlades(std::vector<EnergyBlade>& blades) {
    blades.clear();
    if (!model_initialized_ || radius_px_ <= 0) return;

    float angle_step = 2 * M_PI / cfg_.num_blades;
    for (int i = 0; i < cfg_.num_blades; ++i) {
        float angle = blade0_angle_rad_ + i * angle_step;
        EnergyBlade blade;
        blade.center = cv::Point2f(circle_center_.x + radius_px_ * std::cos(angle),
                                   circle_center_.y + radius_px_ * std::sin(angle));
        blade.id = i;
        blade.angle_rad = angle;
        blades.push_back(blade);
    }
}

bool EnergyDetector::process(const std::vector<cv::Point2f>& blade_centers,
                             std::vector<EnergyBlade>& predicted_blades) {
    double now = getTimestamp();

    // 未初始化模型
    if (!model_initialized_) {
        if (last_timestamp_ == 0 && !blade_centers.empty()) {
            prev_center_ = blade_centers[0];
            last_timestamp_ = now;
            return false;
        }
        else if (last_timestamp_ != 0 && !blade_centers.empty()) {
            cv::Point2f matched;
            if (findSameBlade(blade_centers, matched)) {
                current_center_ = matched;
                double dt = now - last_timestamp_;
                if (dt < 1e-6) return false;
                float delta_angle = cfg_.angular_velocity_deg * M_PI / 180.0f * dt;
                if (computeCircleFromTwoPoints(prev_center_, current_center_,
                                               delta_angle, circle_center_, radius_px_)) {
                    blade0_angle_rad_ = std::atan2(prev_center_.y - circle_center_.y,
                                                   prev_center_.x - circle_center_.x);
                    model_initialized_ = true;
                    lost_frame_count_ = 0;
                    predictAllBlades(predicted_blades);
                    return true;
                }
            }
            // 失败：将当前帧作为新的第一帧
            last_timestamp_ = now;
            if (!blade_centers.empty()) prev_center_ = blade_centers[0];
            return false;
        }
        return false;
    }

    // 模型已初始化，跟踪同一扇叶
    cv::Point2f matched;
    bool found = findSameBlade(blade_centers, matched);
    if (found) {
        current_center_ = matched;
        double dt = now - last_timestamp_;
        if (dt > 0) {
            float delta_angle = cfg_.angular_velocity_deg * M_PI / 180.0f * dt;
            cv::Point2f new_center;
            float new_radius;
            if (computeCircleFromTwoPoints(prev_center_, current_center_,
                                           delta_angle, new_center, new_radius)) {
                if (!isCenterJump(circle_center_, new_center)) {
                    circle_center_ = new_center;
                    radius_px_ = new_radius;
                    blade0_angle_rad_ = std::atan2(current_center_.y - circle_center_.y,
                                                   current_center_.x - circle_center_.x);
                    lost_frame_count_ = 0;
                } else {
                    reset();
                    prev_center_ = current_center_;
                    last_timestamp_ = now;
                    return false;
                }
            }
        }
        prev_center_ = current_center_;
        last_timestamp_ = now;
        predictAllBlades(predicted_blades);
        return true;
    } else {
        lost_frame_count_++;
        if (lost_frame_count_ > cfg_.max_track_lost_frames) {
            reset();
            return false;
        }
        predictAllBlades(predicted_blades);
        return true;
    }
}

} // namespace auto_aim
