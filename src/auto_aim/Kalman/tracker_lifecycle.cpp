#include "tracker.hpp"
#include "Kalman/tracker_utils.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <optional>

using namespace auto_aim::tracker_detail;

namespace auto_aim
{
    void Tracker::init(const Armor& armor, double timestamp)
    {
        Armor refined_armor = armor;
        refineArmorPose(refined_armor);
        const Armor home_armor = toWorld(refined_armor);
        latest_armor_ = armor;
        last_timestamp_ = timestamp;

        const auto radial_angle = armorRadialAngle(home_armor);
        if (radial_angle) {
            // A single plate is always the visible near-side plate. Anchor it
            // to plate 2 and derive yaw from PnP instead of inventing yaw=0.
            plate_id_ = 2;
            const Vec<3> observation = toVec(home_armor.tvec);
            const Vec<3> center = centerFromRadialAngle(
                observation, *radial_angle, cfg_.armor_radius,
                cfg_.armor_y_offsets[plate_id_]);
            const double yaw = ArmorEKF::normalizeAngle(
                *radial_angle - ArmorEKF::angleForPlate(plate_id_));
            ekf_.initAt(center, yaw);
        } else {
            plate_id_ = 0;
            ekf_.init(toVec(home_armor.tvec), plate_id_);
        }
        latest_armor_.tvec = worldToCamera(ekf_.armorPosition(plate_id_));

        state_ = TrackerState::DETECTING;
        accepted_since_init_ = 0;   // 刚初始化：ω 还不可信
        detect_count_ = 1;
        lost_count_ = 0;
        confidence_ = 0.3;
        last_measurement_weight_ = 1.0;
        last_observed_armor_ = armor;
        last_observed_plate_id_ = plate_id_;
        last_observation_time_ = timestamp;
        has_observation_ = true;
    }

    void Tracker::init(const std::vector<Armor>& armors, double timestamp)
    {
        struct ArmorPair {
            Armor home;
            Armor camera;
        };

        std::vector<ArmorPair> observations;
        observations.reserve(armors.size());
        for (const auto& armor : armors) {
            if (!armor.solve_result || armor.tvec.empty()) continue;
            observations.push_back({toWorld(armor), armor});
        }

        if (observations.empty()) return;
        if (observations.size() == 1) {
            init(observations.front().camera, timestamp);
            return;
        }

        if (observations.size() > 4) {
            std::sort(observations.begin(), observations.end(),
                [](const ArmorPair& lhs, const ArmorPair& rhs) {
                    return cv::norm(lhs.home.tvec) < cv::norm(rhs.home.tvec);
                });
            observations.resize(4);
        }

        struct Initialization {
            Vec<3> center;
            double yaw = 0.0;
            double error = std::numeric_limits<double>::max();
            std::array<int, 4> plate_ids{{0, 0, 0, 0}};
            bool valid = false;
        };

        Initialization best;
        std::array<int, 4> assignment{{0, 0, 0, 0}};
        std::array<bool, 4> used_plate{{false, false, false, false}};

        const auto evaluate = [&](std::size_t count) {
            double mean_y = 0.0;
            for (std::size_t i = 0; i < count; ++i) {
                const int plate_id = assignment[i];
                mean_y += observations[i].home.tvec.at<double>(1) -
                    cfg_.armor_y_offsets[plate_id];
            }
            mean_y /= count;

            // 径向角**与 yaw 无关**，先算一次存下来。它内部要做一次旋转
            // （armorRadialAngle → 位姿矩阵），原来放在 1440 步的扫描里反复调，
            // 一次 init 要白算上万次（实测每次重捕 20~40 ms 的大头就是它）。
            std::array<std::optional<double>, 4> radial_angles;
            for (std::size_t i = 0; i < count; ++i) {
                radial_angles[i] = armorRadialAngle(observations[i].home);
            }

            double center_x = 0.0;
            double center_z = 0.0;
            std::size_t center_count = 0;
            for (std::size_t i = 0; i < count; ++i) {
                const auto& radial_angle = radial_angles[i];
                if (radial_angle) {
                    const Vec<3> center = centerFromRadialAngle(
                        toVec(observations[i].home.tvec), *radial_angle,
                        cfg_.armor_radius,
                        cfg_.armor_y_offsets[assignment[i]]);
                    center_x += center(0, 0);
                    center_z += center(2, 0);
                    ++center_count;
                }
            }
            if (center_count > 0) {
                center_x /= center_count;
                center_z /= center_count;
            }

            // 粗→细两段扫描 yaw。原来是 1440 步（0.25°）定步长暴力扫，
            // 而 evaluate 对**每一种板号排列**都要跑一遍（4 块板最多 24 种），
            // 于是每次 tracker 重捕要 20~40 ms（实测 pnp+ekf 段 31 ms，且
            // CPU 时间≈墙钟，是真在算）—— 偏偏重捕是最该快的时候。
            // 代价函数对 yaw 是"每 90° 重复 + 单瓣内平滑"的，所以先 4° 粗扫，
            // 再在粗最优点附近 0.5°、0.0625° 各细扫一圈：122 次评估取到
            // 同一条极小值（约 12× 少算），精度反而更高（0.0625° vs 0.25°）。
            // 本次排列自己的最优（细扫要围着它转，而不是围着全局最优转）
            double local_error = std::numeric_limits<double>::max();
            double local_yaw = 0.0;
            const auto scan = [&](double begin, double end, double step) {
                for (double yaw = begin; yaw < end; yaw += step) {
                if (center_count == 0) {
                    double fallback_x = 0.0;
                    double fallback_z = 0.0;
                    for (std::size_t i = 0; i < count; ++i) {
                        const double angle = yaw + ArmorEKF::angleForPlate(assignment[i]);
                        fallback_x += observations[i].home.tvec.at<double>(0) -
                            cfg_.armor_radius * std::cos(angle);
                        fallback_z += observations[i].home.tvec.at<double>(2) -
                            cfg_.armor_radius * std::sin(angle);
                    }
                    center_x = fallback_x / count;
                    center_z = fallback_z / count;
                }

                double error = 0.0;
                for (std::size_t i = 0; i < count; ++i) {
                    const double angle = yaw + ArmorEKF::angleForPlate(assignment[i]);
                    const double dx = observations[i].home.tvec.at<double>(0) -
                        (center_x + cfg_.armor_radius * std::cos(angle));
                    const double dy = observations[i].home.tvec.at<double>(1) -
                        (mean_y + cfg_.armor_y_offsets[assignment[i]]);
                    const double dz = observations[i].home.tvec.at<double>(2) -
                        (center_z + cfg_.armor_radius * std::sin(angle));
                    error += dx * dx + dy * dy + dz * dz;

                    const auto& radial_angle = radial_angles[i];
                    if (radial_angle) {
                        const double angle_error = ArmorEKF::normalizeAngle(
                            *radial_angle - angle);
                        error += cfg_.armor_radius * cfg_.armor_radius *
                            angle_error * angle_error;
                    }
                }

                if (error < local_error) {
                    local_error = error;
                    local_yaw = yaw;
                }
                if (error < best.error) {
                    best.center(0, 0) = center_x;
                    best.center(1, 0) = mean_y;
                    best.center(2, 0) = center_z;
                    best.yaw = yaw;
                    best.error = error;
                    best.plate_ids = assignment;
                    best.valid = true;
                }
                }
            };
            const double coarse_step = 4.0 * CV_PI / 180.0;
            scan(-CV_PI, CV_PI, coarse_step);
            const double coarse_yaw = local_yaw;
            scan(coarse_yaw - coarse_step, coarse_yaw + coarse_step, 0.5 * CV_PI / 180.0);
            const double medium_step = 0.5 * CV_PI / 180.0;
            scan(local_yaw - medium_step, local_yaw + medium_step, 0.0625 * CV_PI / 180.0);
        };

        const auto search = [&](auto&& self, std::size_t index) -> void {
            if (index == observations.size()) {
                evaluate(observations.size());
                return;
            }
            for (int plate = 0; plate < 4; ++plate) {
                if (used_plate[plate]) continue;
                used_plate[plate] = true;
                assignment[index] = plate;
                self(self, index + 1);
                used_plate[plate] = false;
            }
        };
        search(search, 0);

        if (!best.valid) return;

        ekf_.initAt(best.center, best.yaw);
        plate_id_ = best.plate_ids[0];
        latest_armor_ = observations.front().camera;
        latest_armor_.tvec = worldToCamera(ekf_.armorPosition(plate_id_));
        last_observed_armor_ = observations.front().camera;
        last_observed_plate_id_ = plate_id_;
        last_observation_time_ = timestamp;
        has_observation_ = true;
        last_timestamp_ = timestamp;
        state_ = TrackerState::DETECTING;
        accepted_since_init_ = 0;   // 重捕/重初始化：ω 还不可信
        detect_count_ = 1;
        lost_count_ = 0;
        confidence_ = 0.3;
        last_measurement_weight_ = 1.0;
    }

    Armor Tracker::predict(double timestamp)
    {
        if (state_ == TrackerState::LOST)
            return latest_armor_;

        predictStateTo(timestamp);

        Armor predicted_armor = latest_armor_;
        predicted_armor.tvec = worldToCamera(ekf_.armorPosition(plate_id_));
        return predicted_armor;
    }

    void Tracker::predictTo(double timestamp)
    {
        has_observation_ = false;
        predictStateTo(timestamp);
    }

    void Tracker::predictStateTo(double timestamp)
    {
        if (state_ == TrackerState::LOST || !ekf_.isInit()) return;
        double dt = timestamp - last_timestamp_;
        if (dt <= 1e-6) return;
        if (dt > 0.5)
            dt = cfg_.default_dt;

        ekf_.setDt(dt);
        ekf_.predict();
        last_timestamp_ = timestamp;
    }

    bool Tracker::tryReacquire(
        const std::vector<const Armor*>& observations,
        double timestamp)
    {
        if (!ekf_.isInit() || observations.empty()) return false;
        if (lost_count_ > cfg_.max_lost_frames + cfg_.reacquire_frames) {
            return false;
        }

        double dt = timestamp - last_timestamp_;
        if (dt <= 1e-6 || dt > cfg_.reacquire_max_dt) return false;

        ekf_.setDt(dt);
        ekf_.predict();
        last_timestamp_ = timestamp;

        const Vec<3> predicted_center = ekf_.getCenter();
        double best_error = std::numeric_limits<double>::max();
        for (const Armor* armor : observations) {
            if (!armor || !armor->solve_result || armor->tvec.empty()) continue;
            const auto radial_angle = armorRadialAngle(*armor);
            if (!radial_angle) continue;

            const Vec<3> observed_center = centerFromRadialAngle(
                toVec(armor->tvec), *radial_angle, cfg_.armor_radius);
            const double error = std::sqrt(
                std::pow(observed_center(0, 0) - predicted_center(0, 0), 2) +
                std::pow(observed_center(1, 0) - predicted_center(1, 0), 2) +
                std::pow(observed_center(2, 0) - predicted_center(2, 0), 2));
            best_error = std::min(best_error, error);
        }

        return best_error <= cfg_.reacquire_center_error;
    }

}
