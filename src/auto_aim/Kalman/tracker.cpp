#include "tracker.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>

namespace auto_aim
{
    namespace
    {
        Vec<3> toVec(const cv::Mat& tvec)
        {
            Vec<3> result;
            result(0, 0) = tvec.at<double>(0);
            result(1, 0) = tvec.at<double>(1);
            result(2, 0) = tvec.at<double>(2);
            return result;
        }

        cv::Mat toMat(const Vec<3>& value)
        {
            return (cv::Mat_<double>(3, 1) <<
                value(0, 0), value(1, 0), value(2, 0));
        }

        std::optional<double> armorRadialAngle(const Armor& armor)
        {
            if (!armor.solve_result || armor.rvec.empty() || armor.tvec.empty()) {
                return std::nullopt;
            }

            cv::Mat rotation;
            cv::Rodrigues(armor.rvec, rotation);
            cv::Vec3d normal(
                rotation.at<double>(0, 0),
                rotation.at<double>(1, 0),
                rotation.at<double>(2, 0));
            const cv::Vec3d position(
                armor.tvec.at<double>(0),
                armor.tvec.at<double>(1),
                armor.tvec.at<double>(2));

            // The visible armor face points toward the camera. Its horizontal
            // normal is also the radial direction from chassis center to plate.
            if (normal.dot(position) > 0.0) normal *= -1.0;
            const double norm = cv::norm(normal);
            if (norm < 1e-6) return std::nullopt;

            normal /= norm;
            return std::atan2(normal[2], normal[0]);
        }

        Vec<3> centerFromRadialAngle(
            const Vec<3>& observation, double radial_angle, double radius)
        {
            Vec<3> center;
            center(0, 0) = observation(0, 0) - radius * std::cos(radial_angle);
            center(1, 0) = observation(1, 0);
            center(2, 0) = observation(2, 0) - radius * std::sin(radial_angle);
            return center;
        }
    }

    Tracker::Tracker(const TrackerConfig& cfg)
        : cfg_(cfg), last_timestamp_(0.0), state_(TrackerState::LOST),
          detect_count_(0), lost_count_(0), confidence_(0.0), plate_id_(0),
          last_measurement_weight_(0.0), last_observed_plate_id_(-1),
          has_observation_(false)
    {
        ArmorEKF::Config ekf_cfg;
        ekf_cfg.R = cfg_.armor_radius;
        ekf_cfg.process_acc = cfg_.process_noise_pos;
        ekf_cfg.process_omega = cfg_.process_noise_vel;
        ekf_cfg.meas_noise = cfg_.measure_noise;
        ekf_cfg.y_noise_mult = cfg_.y_noise_mult;
        ekf_cfg.P_min = cfg_.p_cov_min;
        ekf_.setConfig(ekf_cfg);
    }

    void Tracker::init(const Armor& armor, double timestamp)
    {
        latest_armor_ = armor;
        last_timestamp_ = timestamp;

        const auto radial_angle = armorRadialAngle(armor);
        if (radial_angle) {
            // A single plate is always the visible near-side plate. Anchor it
            // to plate 2 and derive yaw from PnP instead of inventing yaw=0.
            plate_id_ = 2;
            const Vec<3> observation = toVec(armor.tvec);
            const Vec<3> center = centerFromRadialAngle(
                observation, *radial_angle, cfg_.armor_radius);
            const double yaw = ArmorEKF::normalizeAngle(
                *radial_angle - ArmorEKF::angleForPlate(plate_id_));
            ekf_.initAt(center, yaw);
        } else {
            plate_id_ = 0;
            ekf_.init(toVec(armor.tvec), plate_id_);
        }
        latest_armor_.tvec = toMat(ekf_.armorPosition(plate_id_));

        state_ = TrackerState::DETECTING;
        detect_count_ = 1;
        lost_count_ = 0;
        confidence_ = 0.3;
        last_measurement_weight_ = 1.0;
        last_observed_armor_ = armor;
        last_observed_plate_id_ = plate_id_;
        has_observation_ = true;
    }

    void Tracker::init(const std::vector<Armor>& armors, double timestamp)
    {
        std::vector<const Armor*> observations;
        observations.reserve(armors.size());
        for (const auto& armor : armors) {
            if (!armor.solve_result || armor.tvec.empty()) continue;
            observations.push_back(&armor);
        }

        if (observations.empty()) return;
        if (observations.size() == 1) {
            init(*observations.front(), timestamp);
            return;
        }

        if (observations.size() > 4) {
            std::sort(observations.begin(), observations.end(),
                [](const Armor* lhs, const Armor* rhs) {
                    return cv::norm(lhs->tvec) < cv::norm(rhs->tvec);
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
                mean_y += observations[i]->tvec.at<double>(1);
            }
            mean_y /= count;

            double center_x = 0.0;
            double center_z = 0.0;
            std::size_t center_count = 0;
            for (std::size_t i = 0; i < count; ++i) {
                const auto radial_angle = armorRadialAngle(*observations[i]);
                if (radial_angle) {
                    const Vec<3> center = centerFromRadialAngle(
                        toVec(observations[i]->tvec), *radial_angle, cfg_.armor_radius);
                    center_x += center(0, 0);
                    center_z += center(2, 0);
                    ++center_count;
                }
            }
            if (center_count > 0) {
                center_x /= center_count;
                center_z /= center_count;
            }

            for (int step = 0; step < 1440; ++step) {
                const double yaw = -CV_PI + step * CV_PI / 720.0;
                if (center_count == 0) {
                    double fallback_x = 0.0;
                    double fallback_z = 0.0;
                    for (std::size_t i = 0; i < count; ++i) {
                        const double angle = yaw + ArmorEKF::angleForPlate(assignment[i]);
                        fallback_x += observations[i]->tvec.at<double>(0) -
                            cfg_.armor_radius * std::cos(angle);
                        fallback_z += observations[i]->tvec.at<double>(2) -
                            cfg_.armor_radius * std::sin(angle);
                    }
                    center_x = fallback_x / count;
                    center_z = fallback_z / count;
                }

                double error = 0.0;
                for (std::size_t i = 0; i < count; ++i) {
                    const double angle = yaw + ArmorEKF::angleForPlate(assignment[i]);
                    const double dx = observations[i]->tvec.at<double>(0) -
                        (center_x + cfg_.armor_radius * std::cos(angle));
                    const double dy = observations[i]->tvec.at<double>(1) - mean_y;
                    const double dz = observations[i]->tvec.at<double>(2) -
                        (center_z + cfg_.armor_radius * std::sin(angle));
                    error += dx * dx + dy * dy + dz * dz;

                    const auto radial_angle = armorRadialAngle(*observations[i]);
                    if (radial_angle) {
                        const double angle_error = ArmorEKF::normalizeAngle(
                            *radial_angle - angle);
                        error += cfg_.armor_radius * cfg_.armor_radius *
                            angle_error * angle_error;
                    }
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
        latest_armor_ = *observations.front();
        latest_armor_.tvec = toMat(ekf_.armorPosition(plate_id_));
        last_observed_armor_ = *observations.front();
        last_observed_plate_id_ = plate_id_;
        has_observation_ = true;
        last_timestamp_ = timestamp;
        state_ = TrackerState::DETECTING;
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
        predicted_armor.tvec = toMat(ekf_.armorPosition(plate_id_));
        return predicted_armor;
    }

    void Tracker::predictStateTo(double timestamp)
    {
        if (state_ == TrackerState::LOST || !ekf_.isInit()) return;
        double dt = timestamp - last_timestamp_;
        if (dt < 1e-6 || dt > 0.5)
            dt = cfg_.default_dt;

        ekf_.setDt(dt);
        ekf_.predict();
        last_timestamp_ = timestamp;
    }

    bool Tracker::update(const Armor& armor, double timestamp)
    {
        has_observation_ = false;
        if (!armor.solve_result || armor.tvec.empty()) {
            predictStateTo(timestamp);
            lostUpdate();
            return false;
        }

        plate_id_ = selectPlateId(armor);
        const double cost = matchCost(armor, plate_id_);
        if (!std::isfinite(cost) && state_ == TrackerState::TRACKING) {
            lostUpdate();
            return false;
        }

        const auto radial_angle = armorRadialAngle(armor);
        double angle_weight = 1.0;
        if (radial_angle) {
            angle_weight = ekf_.updateAngle(
                *radial_angle, plate_id_, cfg_.angle_measure_noise,
                cfg_.angle_nis_threshold);
        }

        const Vec<3> observation = toVec(armor.tvec);
        const Vec<3> center_observation = radial_angle
            ? centerFromRadialAngle(observation, *radial_angle, cfg_.armor_radius)
            : ekf_.centerFromPlate(observation, plate_id_);
        const double center_weight = ekf_.updateCenter(
            center_observation, cfg_.measure_noise, cfg_.y_noise_mult,
            cfg_.position_nis_threshold);
        if (center_weight <= 0.0 && state_ == TrackerState::TRACKING) {
            lostUpdate();
            return false;
        }

        ekf_.dampMotion(1.0, radial_angle ? 0.5 : 0.3);
        last_measurement_weight_ = angle_weight * center_weight;
        latest_armor_.tvec = toMat(ekf_.armorPosition(plate_id_));
        latest_armor_.armor_type = armor.armor_type;
        latest_armor_.left = armor.left;
        latest_armor_.right = armor.right;

        last_timestamp_ = timestamp;
        lost_count_ = 0;

        if (state_ == TrackerState::DETECTING) {
            detect_count_++;
            if (detect_count_ >= cfg_.min_detect_frames) {
                state_ = TrackerState::TRACKING;
                confidence_ = 0.8;
            } else {
                confidence_ = 0.3 + 0.5 * detect_count_ / cfg_.min_detect_frames;
            }
        } else if (state_ == TrackerState::TRACKING) {
            confidence_ = std::min(1.0, confidence_ + 0.05);
        } else if (state_ == TrackerState::TEMP_LOST) {
            state_ = TrackerState::TRACKING;
            confidence_ = 0.7;
        }
        last_observed_armor_ = armor;
        last_observed_plate_id_ = plate_id_;
        has_observation_ = true;
        return true;
    }

    bool Tracker::update(const std::vector<Armor>& armors, double timestamp)
    {
        std::vector<const Armor*> valid_armors;
        valid_armors.reserve(armors.size());
        for (const auto& armor : armors) {
            if (armor.solve_result && !armor.tvec.empty()) {
                valid_armors.push_back(&armor);
            }
        }
        if (valid_armors.size() > 4) {
            std::sort(valid_armors.begin(), valid_armors.end(),
                [](const Armor* lhs, const Armor* rhs) {
                    return cv::norm(lhs->tvec) < cv::norm(rhs->tvec);
                });
            valid_armors.resize(4);
        }

        has_observation_ = false;

        if (state_ == TrackerState::LOST) {
            if (valid_armors.empty()) return false;
            init(armors, timestamp);
        }
        if (!ekf_.isInit()) return false;

        if (valid_armors.empty()) {
            // During loss yaw and omega are unobservable. Damping prevents a
            // noisy angular velocity estimate from rotating the chassis model.
            ekf_.dampMotion(0.7, 0.2);
            predictStateTo(timestamp);
            lostUpdate();
            return false;
        }

        predictStateTo(timestamp);

        int best_armor_index = -1;
        int best_plate_id = -1;
        bool updated_from_geometry = false;

        if (valid_armors.size() >= 2) {
            ChassisFit fit;
            if (fitChassis(valid_armors, fit)) {
                // The fitted circle center is a low-noise aggregate measurement.
                // Plate updates below then recover yaw, omega and the four-plate layout.
                ekf_.updateCenter(fit.center, cfg_.measure_noise / valid_armors.size(), 1.0);
                updated_from_geometry = updateFromFit(
                    valid_armors, fit, best_armor_index, best_plate_id);
            }
        }

        struct Candidate {
            double distance;
            std::size_t armor_index;
            int plate_id;
        };

        std::vector<Candidate> candidates;
        candidates.reserve(valid_armors.size() * 4);
        for (std::size_t i = 0; i < valid_armors.size(); ++i) {
            for (int plate = 0; plate < 4; ++plate) {
                const double cost = matchCost(*valid_armors[i], plate);
                if (std::isfinite(cost)) {
                    candidates.push_back({cost, i, plate});
                }
            }
        }
        std::sort(candidates.begin(), candidates.end(),
            [](const Candidate& lhs, const Candidate& rhs) {
                return lhs.distance < rhs.distance;
            });

        if (!updated_from_geometry && valid_armors.size() == 1) {
            plate_id_ = selectPlateId(*valid_armors.front());
            const double cost = matchCost(*valid_armors.front(), plate_id_);
            if (!std::isfinite(cost) && state_ == TrackerState::TRACKING) {
                has_observation_ = false;
                lostUpdate();
                return false;
            }

            // One plate constrains the chassis translation but not its yaw.
            // Updating only the inferred center avoids fitting arbitrary yaw and
            // omega changes to each new single-plate observation.
            const auto radial_angle = armorRadialAngle(*valid_armors.front());
            double angle_weight = 1.0;
            if (radial_angle) {
                angle_weight = ekf_.updateAngle(
                    *radial_angle, plate_id_, cfg_.angle_measure_noise,
                    cfg_.angle_nis_threshold);
            }
            const Vec<3> observation = toVec(valid_armors.front()->tvec);
            const Vec<3> center_observation = radial_angle
                ? centerFromRadialAngle(observation, *radial_angle, cfg_.armor_radius)
                : ekf_.centerFromPlate(observation, plate_id_);
            const double center_weight = ekf_.updateCenter(
                center_observation, cfg_.measure_noise, cfg_.y_noise_mult,
                cfg_.position_nis_threshold);
            if (center_weight <= 0.0) {
                has_observation_ = false;
                lostUpdate();
                return false;
            }
            ekf_.dampMotion(1.0, radial_angle ? 0.5 : 0.3);
            last_measurement_weight_ = angle_weight * center_weight;
            best_armor_index = 0;
            best_plate_id = plate_id_;
        }

        if (!updated_from_geometry && best_armor_index < 0) {
            std::vector<bool> armor_used(valid_armors.size(), false);
            std::array<bool, 4> plate_used{false, false, false, false};
            double best_distance = std::numeric_limits<double>::max();

            for (const auto& candidate : candidates) {
                if (armor_used[candidate.armor_index] || plate_used[candidate.plate_id]) continue;
                armor_used[candidate.armor_index] = true;
                plate_used[candidate.plate_id] = true;
                const Armor& armor = *valid_armors[candidate.armor_index];
                const double position_weight = ekf_.update(
                    toVec(armor.tvec), candidate.plate_id,
                    cfg_.position_nis_threshold);
                if (position_weight <= 0.0) continue;

                double angle_weight = 1.0;
                const auto radial_angle = armorRadialAngle(armor);
                if (radial_angle) {
                    angle_weight = ekf_.updateAngle(
                        *radial_angle, candidate.plate_id,
                        cfg_.angle_measure_noise, cfg_.angle_nis_threshold);
                }
                last_measurement_weight_ = position_weight * angle_weight;
                if (candidate.distance < best_distance) {
                    best_distance = candidate.distance;
                    best_armor_index = static_cast<int>(candidate.armor_index);
                    best_plate_id = candidate.plate_id;
                }
            }
        }

        if (best_armor_index < 0) {
            has_observation_ = false;
            lostUpdate();
            return false;
        }

        plate_id_ = best_plate_id;
        latest_armor_ = *valid_armors[best_armor_index];
        latest_armor_.tvec = toMat(ekf_.armorPosition(best_plate_id));
        last_observed_armor_ = *valid_armors[best_armor_index];
        last_observed_plate_id_ = best_plate_id;
        has_observation_ = true;
        last_timestamp_ = timestamp;
        lost_count_ = 0;

        if (state_ == TrackerState::DETECTING) {
            detect_count_++;
            if (detect_count_ >= cfg_.min_detect_frames) {
                state_ = TrackerState::TRACKING;
                confidence_ = 0.8;
            } else {
                confidence_ = 0.3 + 0.5 * detect_count_ / cfg_.min_detect_frames;
            }
        } else if (state_ == TrackerState::TRACKING) {
            confidence_ = std::min(1.0, confidence_ + 0.05);
        } else if (state_ == TrackerState::TEMP_LOST) {
            state_ = TrackerState::TRACKING;
            confidence_ = 0.7;
        }

        return true;
    }

    bool Tracker::updateFromFit(const std::vector<const Armor*>& observations,
                                const ChassisFit& fit, int& best_armor_index,
                                int& best_plate_id)
    {
        if (observations.size() < 2 || observations.size() > 4 || !fit.valid) return false;

        double min_weight = 1.0;
        double best_residual = std::numeric_limits<double>::max();
        for (std::size_t i = 0; i < observations.size(); ++i) {
            const int plate_id = fit.plate_ids[i];
            if (plate_id < 0 || plate_id >= 4) return false;

            const Vec<3> observed = toVec(observations[i]->tvec);
            const Vec<3> fitted = ekf_.armorPosition(plate_id);
            const double residual_x = observed(0, 0) - fitted(0, 0);
            const double residual_y = observed(1, 0) - fitted(1, 0);
            const double residual_z = observed(2, 0) - fitted(2, 0);
            const double residual = std::sqrt(
                residual_x * residual_x + residual_y * residual_y + residual_z * residual_z);

            const auto radial_angle = armorRadialAngle(*observations[i]);
            if (radial_angle) {
                // The aggregate center was already applied in Tracker::update.
                // Updating every plate position again would let an ambiguous
                // plate assignment pull the shared center away from that fit.
                const double angle_weight = ekf_.updateAngle(
                    *radial_angle, plate_id, cfg_.angle_measure_noise,
                    cfg_.angle_nis_threshold);
                min_weight = std::min(min_weight, angle_weight);
            } else {
                const double position_weight = ekf_.update(
                    observed, plate_id, cfg_.position_nis_threshold);
                if (position_weight <= 0.0) continue;
                min_weight = std::min(min_weight, position_weight);
            }
            if (residual < best_residual) {
                best_residual = residual;
                best_armor_index = static_cast<int>(i);
                best_plate_id = plate_id;
            }
        }

        last_measurement_weight_ = min_weight;
        return best_armor_index >= 0;
    }

    bool Tracker::fitChassis(const std::vector<const Armor*>& observations,
                             ChassisFit& result) const
    {
        if (observations.size() < 2 || observations.size() > 4) return false;

        result = ChassisFit{};

        struct Fit {
            Vec<3> center;
            double yaw = 0.0;
            double error = std::numeric_limits<double>::max();
            std::array<int, 4> plate_ids{{-1, -1, -1, -1}};
            bool valid = false;
        };

        Fit best;
        std::array<int, 4> assignment{{0, 0, 0, 0}};
        std::array<bool, 4> used_plate{{false, false, false, false}};

        const auto evaluateYaw = [&](double candidate_yaw, double mean_y,
                                     std::size_t count) {
            double center_x = 0.0;
            double center_z = 0.0;
            std::size_t center_count = 0;
            for (std::size_t i = 0; i < count; ++i) {
                const auto radial_angle = armorRadialAngle(*observations[i]);
                if (radial_angle) {
                    const Vec<3> center = centerFromRadialAngle(
                        toVec(observations[i]->tvec), *radial_angle, cfg_.armor_radius);
                    center_x += center(0, 0);
                    center_z += center(2, 0);
                    ++center_count;
                }
            }

            if (center_count > 0) {
                center_x /= center_count;
                center_z /= center_count;
            } else {
                for (std::size_t i = 0; i < count; ++i) {
                    const double angle = candidate_yaw + ArmorEKF::angleForPlate(assignment[i]);
                    center_x += observations[i]->tvec.at<double>(0) -
                        cfg_.armor_radius * std::cos(angle);
                    center_z += observations[i]->tvec.at<double>(2) -
                        cfg_.armor_radius * std::sin(angle);
                }
                center_x /= count;
                center_z /= count;
            }

            double error = 0.0;
            for (std::size_t i = 0; i < count; ++i) {
                const double angle = candidate_yaw + ArmorEKF::angleForPlate(assignment[i]);
                const double dx = observations[i]->tvec.at<double>(0) -
                    (center_x + cfg_.armor_radius * std::cos(angle));
                const double dy = observations[i]->tvec.at<double>(1) - mean_y;
                const double dz = observations[i]->tvec.at<double>(2) -
                    (center_z + cfg_.armor_radius * std::sin(angle));
                error += dx * dx + dy * dy + dz * dz;

                const auto radial_angle = armorRadialAngle(*observations[i]);
                if (radial_angle) {
                    const double angle_error = ArmorEKF::normalizeAngle(
                        *radial_angle - angle);
                    error += cfg_.armor_radius * cfg_.armor_radius *
                        angle_error * angle_error;
                }
            }

            double score = error;
            if (ekf_.isInit()) {
                const double yaw_error = std::atan2(
                    std::sin(candidate_yaw - ekf_.getYaw()),
                    std::cos(candidate_yaw - ekf_.getYaw()));
                const Vec<3> predicted = ekf_.getCenter();
                const double center_error =
                    std::pow(center_x - predicted(0, 0), 2) +
                    std::pow(mean_y - predicted(1, 0), 2) +
                    std::pow(center_z - predicted(2, 0), 2);
                score += 0.05 * yaw_error * yaw_error + 0.02 * center_error;
            }

            if (score < best.error) {
                best.center(0, 0) = center_x;
                best.center(1, 0) = mean_y;
                best.center(2, 0) = center_z;
                best.yaw = candidate_yaw;
                best.error = score;
                best.plate_ids = assignment;
                best.valid = true;
            }
        };

        const auto evaluate = [&](std::size_t count) {
            double mean_y = 0.0;
            for (std::size_t i = 0; i < count; ++i) {
                mean_y += observations[i]->tvec.at<double>(1);
            }
            mean_y /= count;

            // For a fixed plate assignment, the direction between any two plate
            // centers determines yaw up to a half-turn. Evaluating every pair
            // avoids the previous 1440-step yaw scan while remaining noise-robust.
            for (std::size_t first = 0; first < count; ++first) {
                for (std::size_t second = first + 1; second < count; ++second) {
                    const double dx = observations[second]->tvec.at<double>(0) -
                        observations[first]->tvec.at<double>(0);
                    const double dz = observations[second]->tvec.at<double>(2) -
                        observations[first]->tvec.at<double>(2);
                    if (dx * dx + dz * dz < 1e-8) continue;

                    const double midpoint = std::atan2(dz, dx);
                    const double plate_midpoint = 0.5 * (
                        ArmorEKF::angleForPlate(assignment[first]) +
                        ArmorEKF::angleForPlate(assignment[second]));
                    evaluateYaw(midpoint - CV_PI / 2.0 - plate_midpoint, mean_y, count);
                    evaluateYaw(midpoint + CV_PI / 2.0 - plate_midpoint, mean_y, count);
                }
            }
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

        if (!best.valid || best.error / observations.size() > 0.05) return false;
        result.center = best.center;
        result.yaw = best.yaw;
        result.plate_ids = best.plate_ids;
        result.error = best.error / observations.size();
        result.valid = true;
        return true;
    }

    void Tracker::lostUpdate()
    {
        lost_count_++;
        if (state_ == TrackerState::TRACKING) {
            if (lost_count_ >= 1) {
                state_ = TrackerState::TEMP_LOST;
                confidence_ -= 0.1;
            }
        } else if (state_ == TrackerState::TEMP_LOST) {
            if (lost_count_ >= cfg_.max_lost_frames) {
                state_ = TrackerState::LOST;
                confidence_ = 0.0;
            } else {
                confidence_ -= 0.05;
            }
        } else if (state_ == TrackerState::DETECTING) {
            if (lost_count_ >= 2) {
                state_ = TrackerState::LOST;
                confidence_ = 0.0;
            }
        }
    }

    cv::Mat Tracker::getTargetCenter() const
    {
        const Vec<3> center = ekf_.getCenter();
        return (cv::Mat_<double>(3, 1) << center(0, 0), center(1, 0), center(2, 0));
    }

    std::array<cv::Mat, 4> Tracker::getEstimatedArmorPositions() const
    {
        std::array<cv::Mat, 4> positions;
        for (int plate = 0; plate < 4; ++plate) {
            positions[plate] = toMat(ekf_.armorPosition(plate));
        }
        return positions;
    }

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
        const double position_distance = distanceToPlate(armor.tvec, plate_id);
        if (position_distance > cfg_.max_match_distance) {
            return std::numeric_limits<double>::infinity();
        }

        const auto radial_angle = armorRadialAngle(armor);
        if (!radial_angle) return position_distance;

        const double angle_error = std::abs(ArmorEKF::normalizeAngle(
            *radial_angle - ekf_.getPlateAngle(plate_id)));
        if (angle_error > cfg_.max_angle_error) {
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
