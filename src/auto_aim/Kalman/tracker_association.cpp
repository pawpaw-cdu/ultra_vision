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
    int Tracker::plateIdFromBearing(const Armor& armor) const
    {
        if (!ekf_.isInit()) return -1;
        const Armor home = toWorld(armor);               // 注意：toWorld 把板心写回 tvec
        if (!home.solve_result || home.tvec.rows != 3) return -1;
        const auto c = getTargetCenterWorldArray();      // 世界系 {x,y,z}（水平面 = x-z）
        const double dx = home.tvec.at<double>(0) - c[0];
        const double dz = home.tvec.at<double>(2) - c[2];
        const double r_obs = std::hypot(dx, dz);
        // 板心离旋转中心太近/太远 → 位置本身不可信（近处杂检、深度退化）
        if (!(r_obs > 0.05 && r_obs < 0.60)) return -1;
        const double bearing = std::atan2(dx, dz);        // 与 ArmorEKF::getYaw() 同约定
        const double yaw = ekf_.getYaw();
        int best = -1;
        double best_res = 1e9;
        for (int k = 0; k < 4; ++k) {
            const double res = std::abs(ArmorEKF::normalizeAngle(
                bearing - yaw - armorPhaseOffset(k)));
            if (res < best_res) { best_res = res; best = k; }
        }
        const double gate = cfg_.plate_bearing_gate_deg * M_PI / 180.0;
        return (gate > 0.0 && best_res > gate) ? -1 : best;
    }

    int Tracker::selectPlateId(const Armor& armor) const
    {
        // 2026-10-12：**几何（位置）优先**。原因见 plateIdFromBearing 的注释 ——
        // 原路径的代价函数含"预测的 armor_yaw"，斜视时它病态，会把板号挑到 90° 外。
        const int geo = plateIdFromBearing(armor);
        if (geo >= 0) return geo;

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
        // 关联门限跟着滤波器自身的 yaw 不确定度走：初始化 / 丢帧 / 刚重捕时
        // σ_yaw 大 → 门限宽（否则第一个预测还没建立，观测全被判 nomatch）；
        // 收敛后 σ_yaw 小 → 门限回到 max_angle_error 的紧度。上限 < 90°，
        // 保证相邻物理板始终被挡在外面。见 TrackerConfig 的注释。
        double angle_gate = cfg_.max_angle_error;
        if (cfg_.adaptive_association_gate && ekf_.isInit()) {
            angle_gate = std::min(
                cfg_.association_gate_max,
                cfg_.max_angle_error + cfg_.association_gate_sigma * ekf_.yawStd());
        }
        const auto observation = armorYpdObservation(armor, plate_id, cfg_);
        if (observation) {
            // 马氏距离关联（上科大 2026 §5B.4）：代价 = d²，门限 = χ²(4,95%)。
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
            // PnP 的 yaw（板法向 = 径向）是**双解**量：翻解时它一跳就是十几度甚至镜像。
            // 所以它只当"弱观测量"：超过可信门限就不参与代价（也不参与下面的硬门限），
            // 关联完全由位置（bearing 距离 + 距离误差）决定 —— 位置是良态的。
            const double yaw_trust = cfg_.pnp_yaw_trust_deg * M_PI / 180.0;
            const bool yaw_usable = (cfg_.pnp_yaw_trust_deg <= 0.0)
                ? true
                : (armor_angle_error <= yaw_trust);
            // 距离门限跟着同一套"距离噪声随距离增长"的模型放大：远距 PnP 的
            // 距离抖动本来就大（6.5 m 处 1 px≈0.5 m），固定门限会把真板挡掉。
            double range_gate = cfg_.max_match_distance;
            if (cfg_.adaptive_association_gate && ekf_.isInit()) {
                const double range = std::max(0.5, predicted.distance);
                const double sigma_range = std::sqrt(std::max(1e-6, cfg_.measure_noise) *
                    (1.0 + cfg_.range_noise_distance_scale * range * range));
                range_gate = std::min(cfg_.range_gate_max,
                                      cfg_.max_match_distance +
                                          cfg_.range_gate_sigma * sigma_range);
            }
            // 角度硬门限改成落在**bearing 距离**上（米，= 预测距离 × 角度偏差），
            // 与 max_match_distance 同量纲：位置差得远才是"够不着"；
            // yaw 翻解不该把真板挡掉（那正是"选板跳/模型跟着转"的来源）。
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
                // 老行为（A/B 用）：直接拿 PnP 的 yaw 当硬门限。
                // 平面解一翻，真板就会被挡在外面 —— 这就是"选板跳、模型跟着转"的老毛病。
                if (armor_angle_error > angle_gate) {
                    return std::numeric_limits<double>::infinity();
                }
            } else if (bearing_distance > bearing_gate) {
                // 新行为：只用**位置**（bearing 距离）当门限；yaw 只做弱项（见上）。
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
