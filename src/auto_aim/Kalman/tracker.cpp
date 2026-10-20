#include "tracker.hpp"
#include "Kalman/tracker_utils.hpp"

#include "perception/pnp_solver.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>

using namespace auto_aim::tracker_detail;

namespace auto_aim
{

    Tracker::Tracker(const TrackerConfig& cfg)
        : cfg_(cfg), rotation_rate_estimator_(cfg.rotation_rate),
          last_timestamp_(0.0), state_(TrackerState::LOST),
          detect_count_(0), lost_count_(0), confidence_(0.0), plate_id_(0),
          last_measurement_weight_(0.0), last_observed_plate_id_(-1),
          has_observation_(false)
    {
        ArmorEKF::Config ekf_cfg;
        ekf_cfg.R = cfg_.armor_radius;
        ekf_cfg.process_acc = cfg_.process_noise_pos;
        ekf_cfg.process_acc_speed_ref = cfg_.process_noise_pos_speed_ref;
        ekf_cfg.process_acc_max = cfg_.process_noise_pos_max;
        ekf_cfg.process_omega = cfg_.process_noise_vel;
        ekf_cfg.meas_noise = cfg_.measure_noise;
        ekf_cfg.y_noise_mult = cfg_.y_noise_mult;
        ekf_cfg.P_min = cfg_.p_cov_min;
        ekf_cfg.armor_y_offsets = cfg_.armor_y_offsets;
        ekf_.setConfig(ekf_cfg);

        // 底盘慢层几何的初值
        syncGeometryFromConfig();

        // 双模型的板模型（并行运行，见 TrackerConfig::plate_model_enabled）
        PlateEKF::Config plate_cfg;
        plate_cfg.process_acc = cfg_.plate_model_process_acc;
        plate_cfg.process_yaw_acc = cfg_.plate_model_process_yaw_acc;
        plate_ekf_.setConfig(plate_cfg);
    }

    bool Tracker::update(const Armor& armor, double timestamp)
    {
        has_observation_ = false;
        if (camera_pose_.timestamp_valid &&
            std::abs(camera_pose_.timestamp - timestamp) > cfg_.max_camera_pose_dt) {
            lostUpdate();
            return false;
        }
        if (!armor.solve_result || armor.tvec.empty()) {
            predictStateTo(timestamp);
            lostUpdate();
            return false;
        }

        const Armor home_armor = toWorld(armor);
        plate_id_ = selectPlateId(home_armor);
        const double cost = matchCost(home_armor, plate_id_);
        if (!std::isfinite(cost) && state_ == TrackerState::TRACKING) {
            lostUpdate();
            return false;
        }
        const auto radial_angle = armorRadialAngle(home_armor);
        double measurement_weight = 1.0;
        const auto ypd_observation = armorYpdObservation(
            home_armor, plate_id_, cfg_);
        if (ypd_observation && !scaleConsistent(armor, ypd_observation->distance)) {
            // 尺度不自洽（塌缩/错解）：不要这条观测，让滤波器滑行。
            predictStateTo(timestamp);
            lostUpdate();
            return false;
        }
        bool yaw_trusted = true;
        if (ypd_observation) {
            YpdObservation observation = *ypd_observation;
            if (cfg_.pnp_yaw_trust_deg > 0.0 && ekf_.isInit()) {
                const double error = std::abs(ArmorEKF::normalizeAngle(
                    observation.armor_yaw - ekf_.getPlateAngle(plate_id_)));
                if (error > cfg_.pnp_yaw_trust_deg * M_PI / 180.0) {
                    observation.armor_yaw_noise = 1e3;
                    yaw_trusted = false;
                }
            }
            measurement_weight = ekf_.updateYpd(
                observation,
                effectiveNisThreshold(cfg_.soft_nis_gate, cfg_.ypd_nis_threshold));
        } else {
            if (radial_angle) {
                measurement_weight = ekf_.updateAngle(
                    *radial_angle, plate_id_, cfg_.angle_measure_noise,
                    effectiveNisThreshold(cfg_.soft_nis_gate, cfg_.angle_nis_threshold));
            }
            const Vec<3> observation = toVec(home_armor.tvec);
            const Vec<3> center_observation = radial_angle
                ? centerFromRadialAngle(observation, *radial_angle, cfg_.armor_radius)
                : ekf_.centerFromPlate(observation, plate_id_);
            measurement_weight *= ekf_.updateCenter(
                center_observation, cfg_.measure_noise, cfg_.y_noise_mult,
                effectiveNisThreshold(cfg_.soft_nis_gate, cfg_.position_nis_threshold));
        }
        if (measurement_weight <= 0.0 && state_ == TrackerState::TRACKING) {
            lostUpdate();
            return false;
        }
        updateRotationRate(home_armor, timestamp, yaw_trusted, plate_id_);

        last_measurement_weight_ = measurement_weight;
        latest_armor_ = armor;
        latest_armor_.tvec = worldToCamera(ekf_.armorPosition(plate_id_));
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
        last_observation_time_ = timestamp;
        has_observation_ = true;
        return true;
    }

    bool Tracker::update(const std::vector<Armor>& armors, double timestamp)
    {
        struct ArmorPair {
            Armor home;
            Armor camera;
        };

        // 本帧的关联统计（供 node_sim 打印 / 诊断用）
        dropped_ambiguous_ = 0;
        dropped_nomatch_ = 0;
        dropped_scale_ = 0;
        accepted_observations_ = 0;

        std::vector<ArmorPair> valid_pairs;
        valid_pairs.reserve(armors.size());
        for (const auto& armor : armors) {
            if (armor.solve_result && !armor.tvec.empty()) {
                Armor refined_armor = armor;
                refineArmorPose(refined_armor);
                valid_pairs.push_back({toWorld(refined_armor), refined_armor});
            }
        }
        if (valid_pairs.size() > 4) {
            std::sort(valid_pairs.begin(), valid_pairs.end(),
                [](const ArmorPair& lhs, const ArmorPair& rhs) {
                    return cv::norm(lhs.home.tvec) < cv::norm(rhs.home.tvec);
                });
            valid_pairs.resize(4);
        }

        std::vector<const Armor*> valid_armors;
        valid_armors.reserve(valid_pairs.size());
        for (const auto& pair : valid_pairs) {
            valid_armors.push_back(&pair.home);
        }

        has_observation_ = false;
        if (camera_pose_.timestamp_valid &&
            std::abs(camera_pose_.timestamp - timestamp) > cfg_.max_camera_pose_dt) {
            lostUpdate();
            return false;
        }

        if (cfg_.watchdog_enabled && ekf_.isInit() && state_ != TrackerState::LOST &&
            last_timestamp_ > 0.0) {
            const double gap = timestamp - last_timestamp_;
            if (std::isfinite(gap) && gap > cfg_.watchdog_max_frame_gap) {
                watchdogTrip("帧间隔过长（相机离线 / 时间戳跳变）");
            }
        }

        bool reacquired = false;
        if (state_ == TrackerState::LOST) {
            if (valid_armors.empty()) {
                ++lost_count_;
                return false;
            }
            // 看门狗触发的重初始化：不要用已经发散的旧状态去尝试重捕，直接用观测初始化
            if (force_reinit_) {
                force_reinit_ = false;
                init(armors, timestamp);
                reacquired = true;
                state_ = TrackerState::TRACKING;
                detect_count_ = std::max(detect_count_, cfg_.min_detect_frames);
                lost_count_ = 0;
                confidence_ = 0.7;
            } else if (tryReacquire(valid_armors, timestamp)) {
                state_ = TrackerState::TRACKING;
                detect_count_ = std::max(detect_count_, cfg_.min_detect_frames);
                lost_count_ = 0;
                confidence_ = 0.7;
                reacquired = true;
            } else {
                init(armors, timestamp);
            }
        }
        if (!ekf_.isInit()) return false;

        if (valid_armors.empty()) {
            predictStateTo(timestamp);
            lostUpdate();
            return false;
        }

        if (!reacquired) predictStateTo(timestamp);

        int best_armor_index = -1;
        int best_plate_id = -1;

        struct Candidate {
            double distance;
            std::size_t armor_index;
            int plate_id;
        };

        std::vector<Candidate> candidates;
        candidates.reserve(valid_armors.size() * 4);
        for (std::size_t i = 0; i < valid_armors.size(); ++i) {
            double best_cost = std::numeric_limits<double>::max();
            double second_cost = std::numeric_limits<double>::max();
            int best_plate = -1;
            for (int plate = 0; plate < 4; ++plate) {
                const double cost = matchCost(*valid_armors[i], plate);
                if (!std::isfinite(cost)) continue;
                if (cost < best_cost) {
                    second_cost = best_cost;
                    best_cost = cost;
                    best_plate = plate;
                } else if (cost < second_cost) {
                    second_cost = cost;
                }
            }
            if (best_plate < 0) {
                ++dropped_nomatch_;
                continue;
            }
            if (cfg_.plate_lock_margin > 1.0 && plate_id_ >= 0 && plate_id_ < 4 &&
                plate_id_ != best_plate) {
                const double locked_cost = matchCost(*valid_armors[i], plate_id_);
                if (std::isfinite(locked_cost) &&
                    locked_cost <= best_cost * cfg_.plate_lock_margin) {
                    best_plate = plate_id_;
                    best_cost = locked_cost;
                }
            }
            if (cfg_.association_margin > 1.0 &&
                best_cost * cfg_.association_margin > second_cost) {
                ++dropped_ambiguous_;
                continue;   // 模棱两可：不拿它更新，交给滑行
            }
            candidates.push_back({best_cost, i, best_plate});
        }
        std::sort(candidates.begin(), candidates.end(),
            [](const Candidate& lhs, const Candidate& rhs) {
                return lhs.distance < rhs.distance;
            });

        std::vector<bool> armor_used(valid_armors.size(), false);
        std::array<bool, 4> plate_used{{false, false, false, false}};
        std::vector<YpdObservation> ypd_observations;
        ypd_observations.reserve(candidates.size());
        std::vector<std::pair<int, std::size_t>> accepted_pairs;   // 几何慢收敛
        // 被选中的那块板的 PnP yaw 是否可信（见 pnp_yaw_trust_deg）
        bool best_yaw_trusted = true;
        for (const auto& candidate : candidates) {
            if (armor_used[candidate.armor_index] || plate_used[candidate.plate_id]) continue;
            const Armor& armor = *valid_armors[candidate.armor_index];
            const auto observation = armorYpdObservation(armor, candidate.plate_id, cfg_);
            if (!observation) continue;
            if (!scaleConsistent(armor, observation->distance)) {
                ++dropped_scale_;
                continue;
            }

            YpdObservation accepted = *observation;
            bool yaw_trusted = true;
            if (cfg_.distance_noise_scale > 0.0 && camera_fx_ > 1.0 && !armor.tvec.empty()) {
                const double depth = std::max(0.3, std::abs(armor.tvec.at<double>(2)));
                const double width = (armor.armor_type == auto_aim::large)
                    ? cfg_.armor_large_width : cfg_.armor_small_width;
                const double expected = std::max(1.0, width * camera_fx_ / depth);
                const double sep = std::abs(armor.right.center.x - armor.left.center.x);
                const double view_rad = std::acos(std::clamp(sep / expected, 0.0, 1.0));
                accepted.distance_noise =
                    cfg_.distance_noise_scale * (std::log(std::abs(view_rad) + 1.0) + 1.0);
                accepted.armor_yaw_noise =
                    std::log(accepted.distance + 1.0) / 200.0 + cfg_.armor_angle_noise_base;
                last_view_deg_ = view_rad * 180.0 / M_PI;
                // 极端擦边（>graze_gate_deg，默认 75°）仍直接丢弃：那时连 bearing 都不可靠
                if (cfg_.graze_gate_deg > 0.0 && last_view_deg_ > cfg_.graze_gate_deg) {
                    accepted.distance_noise *= cfg_.graze_noise_scale;
                    accepted.armor_yaw_noise = 1e3;
                    yaw_trusted = false;
                }
            }
            if (cfg_.pnp_yaw_trust_deg > 0.0 && ekf_.isInit()) {
                const double error = std::abs(ArmorEKF::normalizeAngle(
                    accepted.armor_yaw - ekf_.getPlateAngle(candidate.plate_id)));
                if (error > cfg_.pnp_yaw_trust_deg * M_PI / 180.0) {
                    accepted.armor_yaw_noise = 1e3;
                    yaw_trusted = false;
                }
            }
            armor_used[candidate.armor_index] = true;
            plate_used[candidate.plate_id] = true;
            ypd_observations.push_back(accepted);
            accepted_pairs.emplace_back(candidate.plate_id, candidate.armor_index);
            ++accepted_observations_;
            if (best_armor_index < 0) {
                best_armor_index = static_cast<int>(candidate.armor_index);
                best_plate_id = candidate.plate_id;
                best_yaw_trusted = yaw_trusted;
            }
        }

        if (!ypd_observations.empty()) {
            last_measurement_weight_ = ekf_.updateYpdBatch(
                ypd_observations,
                effectiveNisThreshold(cfg_.soft_nis_gate, cfg_.ypd_nis_threshold));
            if (last_measurement_weight_ <= 0.0) best_armor_index = -1;
        } else if (valid_armors.size() == 1) {
            plate_id_ = selectPlateId(*valid_armors.front());
            const double cost = matchCost(*valid_armors.front(), plate_id_);
            if (!std::isfinite(cost) && state_ == TrackerState::TRACKING) {
                // 唯一的观测连预测板都够不着：同样是"模型解释不了观测"
                noteMeasurement(false);
                has_observation_ = false;
                lostUpdate();
                return false;
            }
            const Vec<3> observation = toVec(valid_armors.front()->tvec);
            last_measurement_weight_ = ekf_.updateCenter(
                ekf_.centerFromPlate(observation, plate_id_), cfg_.measure_noise,
                cfg_.y_noise_mult,
                effectiveNisThreshold(cfg_.soft_nis_gate, cfg_.position_nis_threshold));
            if (last_measurement_weight_ > 0.0) {
                best_armor_index = 0;
                best_plate_id = plate_id_;
            }
        }

        if (best_armor_index < 0) {
            noteMeasurement(false);
            has_observation_ = false;
            lostUpdate();
            return false;
        }

        plate_id_ = best_plate_id;
        if (cfg_.geometry_learn && ekf_.isInit() && !accepted_pairs.empty() &&
            !(cfg_.geometry_learn_max_omega > 0.0 &&
              cfg_.geometry_learn_max_omega < 3.0 &&
              std::abs(getOmega()) > cfg_.geometry_learn_max_omega)) {
            double sum_radial[2] = {0.0, 0.0};
            double sum_vertical[2] = {0.0, 0.0};
            int count[2] = {0, 0};
            for (const auto& [plate, armor_index] : accepted_pairs) {
                const int group = (plate == 1 || plate == 3) ? 1 : 0;   // 1/3=前后组
                const Vec<3> observed = toVec(valid_pairs[armor_index].home.tvec);
                const Vec<3> modeled = ekf_.armorPosition(plate);
                const double phase = ekf_.getYaw() + ArmorEKF::angleForPlate(plate);
                const double radial_error =
                    (observed(0, 0) - modeled(0, 0)) * std::cos(phase) +
                    (observed(2, 0) - modeled(2, 0)) * std::sin(phase);
                const double vertical_error = observed(1, 0) - modeled(1, 0);
                if (std::abs(radial_error) < cfg_.geometry_learn_max_error) {
                    sum_radial[group] += radial_error;
                    ++count[group];
                }
                if (cfg_.geometry_learn_height &&
                    std::abs(vertical_error) < cfg_.geometry_learn_max_error) {
                    sum_vertical[group] += -vertical_error;   // 模型里向上为正
                }
            }
            bool geometry_changed = false;
            for (int g = 0; g < 2; ++g) {
                if (count[g] == 0) continue;
                const double step = std::clamp(
                    cfg_.geometry_learn_gain * (sum_radial[g] / count[g]),
                    -cfg_.geometry_learn_max_step, cfg_.geometry_learn_max_step);
                if (std::abs(step) > 1e-7) {
                    for (int plate = g; plate < 4; plate += 2) {   // g=0→{0,2}, g=1→{1,3}
                        const double base = cfg_.plate_distance[plate] > 0.05
                            ? cfg_.plate_distance[plate] : cfg_.armor_radius;
                        geometry_.plate_distance[plate] = std::clamp(
                            geometry_.plate_distance[plate] + step,
                            base - cfg_.geometry_learn_range, base + cfg_.geometry_learn_range);
                    }
                    geometry_changed = true;
                }
                if (cfg_.geometry_learn_height && count[g] > 0) {
                    const double hstep = std::clamp(
                        cfg_.geometry_learn_gain * (sum_vertical[g] / count[g]),
                        -cfg_.geometry_learn_max_step, cfg_.geometry_learn_max_step);
                    if (std::abs(hstep) > 1e-7) {
                        for (int plate = g; plate < 4; plate += 2)
                            geometry_.plate_height[plate] = std::clamp(
                                geometry_.plate_height[plate] + hstep, -0.10, 0.10);
                        geometry_changed = true;
                    }
                }
            }
            if (geometry_changed) {
                // **尺度锚定**：只让学习器改"组间差"，总尺度不动（尺度由拟合器的弧段解给）。
                double mean = 0.0;
                for (double d : geometry_.plate_distance) mean += d;
                mean /= 4.0;
                const double anchor = armor_radius_online_ > 0.0 ? armor_radius_online_
                                                                  : cfg_.armor_radius;
                for (std::size_t i = 0; i < geometry_.plate_distance.size(); ++i)
                    geometry_.plate_distance[i] = std::clamp(
                        geometry_.plate_distance[i] + (anchor - mean),
                        anchor - cfg_.geometry_learn_range, anchor + cfg_.geometry_learn_range);
                syncGeometryToEkf();
            }
        }
        if (cfg_.nearest_plate_constraint && ekf_.isInit() && camera_pose_.valid) {
            const Matrix<3, 3> camera_to_home =
                tracker_detail::cameraToWorldRotation(camera_pose_);
            Vec<3> forward;                      // 相机光轴（+z）在 home 系里的方向
            for (int i = 0; i < 3; ++i) forward(i, 0) = camera_to_home(i, 2);
            const Vec<3> camera_home = tracker_detail::pointToWorld(camera_pose_, Vec<3>());
            const Vec<3> observed = toVec(valid_pairs[best_armor_index].home.tvec);
            const Vec<3> center = ekf_.getCenter();
            const auto depth = [&](const Vec<3>& point) {
                return (point(0, 0) - camera_home(0, 0)) * forward(0, 0) +
                    (point(1, 0) - camera_home(1, 0)) * forward(1, 0) +
                    (point(2, 0) - camera_home(2, 0)) * forward(2, 0);
            };
            if (depth(observed) > depth(center) + cfg_.nearest_plate_margin_m) {
                if (++nearest_plate_violation_frames_ >= 2) {
                    ekf_.flipChassisYaw();
                    ++nearest_plate_flips_;
                    nearest_plate_violation_frames_ = 0;
                }
            } else {
                nearest_plate_violation_frames_ = 0;
            }
        } else {
            nearest_plate_violation_frames_ = 0;
        }
        if (cfg_.joint_solve && ekf_.isInit() && accepted_pairs.size() >= 2) {
            std::vector<std::pair<int, Vec<3>>> plates;
            plates.reserve(accepted_pairs.size());
            for (const auto& entry : accepted_pairs)
                plates.emplace_back(entry.first, toVec(valid_pairs[entry.second].home.tvec));
            Vec<3> center_solved;
            double yaw_solved = 0.0;
            double residual = 0.0;
            if (solveChassisFromPlates(plates, center_solved, yaw_solved, residual) &&
                residual <= cfg_.joint_solve_max_residual) {
                ekf_.updateAngle(
                    ArmorEKF::normalizeAngle(yaw_solved + ArmorEKF::angleForPlate(0)), 0,
                    cfg_.joint_solve_yaw_noise,
                    std::numeric_limits<double>::infinity());
                ekf_.updateCenter(center_solved, cfg_.joint_solve_center_noise,
                                  cfg_.y_noise_mult,
                                  std::numeric_limits<double>::infinity());
                ++joint_solves_;
                joint_solve_residual_ = residual;
                joint_solve_yaw_ = yaw_solved;
            }
        }

        if (cfg_.hard_plate_constraint) {
            const Vec<3> observed = toVec(valid_pairs[best_armor_index].home.tvec);
            const Vec<3> predicted = ekf_.armorPosition(plate_id_);
            if (std::getenv("ULTRA_VISION_CONSTRAINT_DEBUG") != nullptr) {
                const Vec<3> error = observed - predicted;
                const double mm = std::sqrt(error(0, 0) * error(0, 0) + error(1, 0) * error(1, 0) +
                                            error(2, 0) * error(2, 0)) * 1000.0;
                std::printf("[constraint] id=%d 约束前 |obs-pred| = %.1f mm（约束后按定义 = 0）\n",
                            plate_id_, mm);
            }
            ekf_.shiftCenter(observed - predicted);
        }
        updateRotationRate(valid_pairs[best_armor_index].home, timestamp, best_yaw_trusted,
                           plate_id_);

        if (cfg_.plate_model_enabled) {
            const Armor& observed = valid_pairs[best_armor_index].home;
            const auto plate_radial = armorRadialAngle(observed);
            if (plate_radial) {
                double dt = cfg_.default_dt;
                if (plate_model_last_timestamp_ > 0.0) {
                    const double gap = timestamp - plate_model_last_timestamp_;
                    if (gap > 1e-4 && gap < 0.2) dt = gap;
                }
                plate_ekf_.setDt(dt);
                plate_ekf_.predict();
                plate_ekf_.update(toVec(observed.tvec), *plate_radial,
                                  cfg_.ypd_nis_threshold);
                plate_model_last_timestamp_ = timestamp;
            }
        }
        {
            const double omega_abs = std::abs(ekf_.getOmega());
            if (!cfg_.plate_model_enabled || !plate_ekf_.isInit()) {
                plate_model_weight_ = 0.0;
            } else {
                const double span = std::max(1e-3, cfg_.plate_model_exit_omega -
                                                       cfg_.plate_model_enter_omega);
                const double weight = (cfg_.plate_model_exit_omega - omega_abs) / span;
                plate_model_weight_ = std::max(0.0, std::min(1.0, weight));
            }
        }
        latest_armor_ = valid_pairs[best_armor_index].camera;
        latest_armor_.tvec = worldToCamera(ekf_.armorPosition(best_plate_id));
        last_observed_armor_ = valid_pairs[best_armor_index].camera;
        last_observed_plate_id_ = best_plate_id;
        last_observation_time_ = timestamp;
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

       // ---- 收敛看门狗 ----
       noteMeasurement(true);

        if (std::getenv("ULTRA_VISION_OMEGA_DEBUG") != nullptr &&
            std::abs(ekf_.getOmega()) > 0.35) {
            std::printf("[omega] t=%.3f state=%d id=%d ekf=%.1f°/s est=%.1f°/s "
                        "est_valid=%d residual=%.4f dt_since_last=%.3f\n",
                        timestamp, static_cast<int>(state_), plate_id_,
                        ekf_.getOmega() * 180.0 / CV_PI,
                        rotation_rate_estimator_.omega() * 180.0 / CV_PI,
                        rotation_rate_estimator_.valid() ? 1 : 0,
                        rotation_rate_estimator_.residual(),
                        timestamp - debug_last_timestamp_);
            debug_last_timestamp_ = timestamp;
        }
        const int samples = static_cast<int>(recent_measurements_.size());
        if (!cfg_.soft_nis_gate && samples >= cfg_.watchdog_min_samples &&
            last_reject_ratio_ >= cfg_.watchdog_reject_ratio) {
            watchdogTrip("量测长期被拒（不收敛）");
        } else if (stateDiverged()) {
            watchdogTrip("状态发散");
        }

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

}
