#include "tracker.hpp"
#include "Kalman/tracker_utils.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <optional>

using namespace auto_aim::tracker_detail;

namespace auto_aim
{
    int Tracker::selectPlateId(const Armor& armor) const
    {
        int best_plate_id = plate_id_;
        const double previous_cost = matchCost(armor, best_plate_id);
        double best_cost = previous_cost;

        for (int plate = 0; plate < 4; ++plate) {
            const double cost = matchCost(armor, plate);
            if (cost < best_cost && cost < previous_cost * 0.65) {
                best_cost = cost;
                best_plate_id = plate;
            }
        }
        return best_plate_id;
    }

    double Tracker::matchCost(const Armor& armor, int plate_id) const
    {
        // 关联门限跟着滤波器自身的 yaw 不确定度走
        double angle_gate = cfg_.max_angle_error;
        if (cfg_.adaptive_association_gate && ekf_.isInit()) {
            angle_gate = std::min(
                cfg_.association_gate_max,
                cfg_.max_angle_error + cfg_.association_gate_sigma * ekf_.yawStd());
        }
        const auto observation = armorYpdObservation(armor, plate_id, cfg_);
        if (observation) {
            // 马氏距离关联：代价 = d²，门限 = χ²(4,95%)。
            // S 已按逐维度噪声归一化，所以不需要再手调"角度多少度算换板"。
            if (cfg_.mahalanobis_matching && ekf_.isInit()) {
                const double squared = ekf_.ypdMahalanobisSquared(*observation);
                if (!std::isfinite(squared) || squared > cfg_.association_chi2) {
                    return std::numeric_limits<double>::infinity();
                }
                return std::sqrt(squared);
            }
            const YpdObservation predicted = ekf_.predictYpd(plate_id);
            const double yaw_error = ArmorEKF::normalizeAngle(
                observation->yaw - predicted.yaw);
            const double pitch_error = ArmorEKF::normalizeAngle(
                observation->pitch - predicted.pitch);
            const double range_error = observation->distance - predicted.distance;
            const double armor_angle_error = std::abs(ArmorEKF::normalizeAngle(
                observation->armor_yaw - predicted.armor_yaw));
            const double yaw_trust = cfg_.pnp_yaw_trust_deg * M_PI / 180.0;
            const bool yaw_usable = (cfg_.pnp_yaw_trust_deg <= 0.0)
                ? true
                : (armor_angle_error <= yaw_trust);
            double range_gate = cfg_.max_match_distance;
            if (cfg_.adaptive_association_gate && ekf_.isInit()) {
                const double range = std::max(0.5, predicted.distance);
                const double sigma_range = std::sqrt(std::max(1e-6, cfg_.measure_noise) *
                    (1.0 + cfg_.range_noise_distance_scale * range * range));
                range_gate = std::min(cfg_.range_gate_max,
                                      cfg_.max_match_distance +
                                          cfg_.range_gate_sigma * sigma_range);
            }
            const double bearing_distance = predicted.distance *
                std::hypot(yaw_error, pitch_error);
            double bearing_gate = cfg_.max_match_distance;
            if (cfg_.adaptive_association_gate && ekf_.isInit()) {
                // 滤波器自己的 yaw 不确定度 → 预测距离处的横向位置不确定度（米）。
                bearing_gate = std::min(
                    cfg_.range_gate_max,
                    cfg_.max_match_distance +
                        cfg_.association_gate_sigma * ekf_.yawStd() * predicted.distance);
            }
            if (std::abs(range_error) > range_gate) {
                return std::numeric_limits<double>::infinity();
            }
            if (cfg_.pnp_yaw_trust_deg <= 0.0) {
                if (armor_angle_error > angle_gate) {
                    return std::numeric_limits<double>::infinity();
                }
            } else if (bearing_distance > bearing_gate) {
                // 只用**位置**（bearing 距离）当门限；yaw 只做弱项（见上）。
                return std::numeric_limits<double>::infinity();
            }

            return bearing_distance + std::abs(range_error) +
                (yaw_usable ? cfg_.angle_match_weight * armor_angle_error : 0.0);
        }

        const double position_distance = distanceToPlate(armor.tvec, plate_id);
        if (position_distance > cfg_.max_match_distance) {
            return std::numeric_limits<double>::infinity();
        }

        const auto radial_angle = armorRadialAngle(armor);
        if (!radial_angle) return position_distance;

        const double angle_error = std::abs(ArmorEKF::normalizeAngle(
            *radial_angle - ekf_.getPlateAngle(plate_id)));
        if (angle_error > angle_gate) {
            return std::numeric_limits<double>::infinity();
        }

        // Convert angular mismatch to an equivalent center error so it can be
        // compared directly with the existing metre-based association cost.
        return position_distance + cfg_.angle_match_weight * angle_error;
    }

    double Tracker::distanceToPlate(const cv::Mat& tvec, int plate_id) const
    {
        const Vec<3> observed = toVec(tvec);
        const Vec<3> predicted = ekf_.armorPosition(plate_id);
        const double dx = observed(0, 0) - predicted(0, 0);
        const double dy = observed(1, 0) - predicted(1, 0);
        const double dz = observed(2, 0) - predicted(2, 0);
        return std::sqrt(dx * dx + dy * dy + dz * dz);
    }

}
