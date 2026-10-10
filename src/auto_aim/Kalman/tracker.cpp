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

        // 底盘慢层几何的初值（真机 15° 支架 / 逐板半径 / 板面尺寸），
        // 之后由局部拟合器慢层或在线慢收敛修改。见 common/chassis_model.hpp。
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
            // PnP 平面解 yaw 只在和预测一致时才喂给 EKF（见 pnp_yaw_trust_deg）：
            // 不一致 → 把它的 yaw 噪声放到无穷大，位置/距离照用，yaw 交给 ω 外推。
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
        updateRotationRate(home_armor, timestamp, yaw_trusted);

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

        // ---- 判据三：帧间隔过长（相机离线 / 时间戳跳变）----
        // 只对"本来在跟踪"的状态判：LOST 本来就会重新初始化，DETECTING 还没建立状态。
        // 对齐 sp_vision tracker.cpp：`state != lost && dt > 0.1s → lost`。
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
            // Motion blur commonly drops a few frames during a small-gyro
            // maneuver. Keep the constant-velocity estimate through the short
            // loss window so the next plate can be associated at the correct
            // phase. The state is reinitialized after the full loss timeout.
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
            // 四块预测板各算一次代价：留最优，并且要求最优**明显优于**次优，
            // 否则这块观测在两块板之间模棱两可 → 直接不要（见 TrackerConfig 注释）。
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
                // 四个预测板全都"够不着"（超出 max_match_distance / max_angle_error）
                // —— 这类框不是"模棱两可"，是跟当前预测完全对不上（假框，
                // 或者预测已经飘了）。和 ambiguous / scale 分开计数才好定位。
                ++dropped_nomatch_;
                continue;
            }
            // ---- 板号锁定：优先沿用上一帧定的那块板 ----
            // 实测（2026-09-29，图像闭环日志）：识别到的板会在物理板之间跳动，视轴上的
            // 偏差一帧内跳 ±12~16°（正好是相邻/对角板的视差），闭环追着跳变永远不收敛。
            // 已关联的那块只要代价不比最优差太多（< lock_margin 倍）就继续保持。
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
        std::vector<std::pair<int, std::size_t>> accepted_pairs;   // 几何慢收敛要用
        // 被选中的那块板的 PnP yaw 是否可信（见 pnp_yaw_trust_deg）——
        // 不可信时不能喂差分测速器（差分正是放大量）。
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

            // PnP 平面解 yaw 的双解保护（见 pnp_yaw_trust_deg 注释）：只有当它和滤波器
            // 预测的板相位一致（≤ 门限）时才让 EKF 用它的 yaw 分量；否则这一帧只信位置
            // （bearing/距离），yaw 由 ω 外推 + 拟合器慢层提供。
            YpdObservation accepted = *observation;
            bool yaw_trusted = true;
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
            // PnP can theoretically succeed without a usable rotation. Keep one
            // Cartesian fallback only for that degenerate case.
            plate_id_ = selectPlateId(*valid_armors.front());
            const double cost = matchCost(*valid_armors.front(), plate_id_);
            if (!std::isfinite(cost) && state_ == TrackerState::TRACKING) {
                // 唯一的观测连预测板都够不着：同样是"模型解释不了观测"，
                // 计入看门狗窗口（否则这一类不收敛完全不入账）。
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
            // 有观测但量测被门限拒：计入看门狗窗口（这是"不收敛"的主要信号）
            noteMeasurement(false);
            has_observation_ = false;
            lostUpdate();
            return false;
        }

        plate_id_ = best_plate_id;
        // ---- 硬约束：构造用的那块板一定是**离相机最近**的那块 ----------------
        // 车心必须比它更远（画面上更靠上）。违反 = 模型的径向/相位翻了 180°
        // （PnP 法向翻号或关联到对面板）→ 把 yaw 翻回来再摆。
        // ---- 观测慢收敛：底盘几何自己学（必须在硬钉之前，硬钉会清零残差）----
        if (cfg_.geometry_learn && ekf_.isInit() && !accepted_pairs.empty()) {
            bool geometry_changed = false;
            for (const auto& [plate, armor_index] : accepted_pairs) {
                const Vec<3> observed = toVec(valid_pairs[armor_index].home.tvec);
                const Vec<3> modeled = ekf_.armorPosition(plate);
                const double phase = ekf_.getYaw() + ArmorEKF::angleForPlate(plate);
                // 径向（水平）上的残差 → 该板的回转距离
                const double radial_error =
                    (observed(0, 0) - modeled(0, 0)) * std::cos(phase) +
                    (observed(2, 0) - modeled(2, 0)) * std::sin(phase);
                // 竖直残差（world y 向下）→ 该板的高度（模型里向上为正 → 取负）
                const double vertical_error = observed(1, 0) - modeled(1, 0);
                if (std::abs(radial_error) < cfg_.geometry_learn_max_error) {
                    const double step = std::clamp(cfg_.geometry_learn_gain * radial_error,
                                                   -cfg_.geometry_learn_max_step,
                                                   cfg_.geometry_learn_max_step);
                    if (std::abs(step) > 1e-7) {
                        const double base = cfg_.plate_distance[plate] > 0.05
                            ? cfg_.plate_distance[plate] : cfg_.armor_radius;
                        geometry_.plate_distance[plate] = std::clamp(
                            geometry_.plate_distance[plate] + step,
                            base - cfg_.geometry_learn_range, base + cfg_.geometry_learn_range);
                        geometry_changed = true;
                    }
                }
                if (std::abs(vertical_error) < cfg_.geometry_learn_max_error) {
                    const double step = std::clamp(-cfg_.geometry_learn_gain * vertical_error,
                                                   -cfg_.geometry_learn_max_step,
                                                   cfg_.geometry_learn_max_step);
                    if (std::abs(step) > 1e-7) {
                        geometry_.plate_height[plate] = std::clamp(
                            geometry_.plate_height[plate] + step, -0.10, 0.10);
                        geometry_changed = true;
                    }
                }
            }
            if (geometry_changed) {
                // **尺度锚定**：只让学习器改"逐板差异"（矩形底盘），总尺度不动 ——
                // 单板帧里"半径偏大"与"车心沿径向偏"不可分（实车 log 里 Rgeo 一路
                // 涨到 clamp 就是这个耦合），尺度交给拟合器的弧段解（可观）。
                double mean = 0.0;
                for (double d : geometry_.plate_distance) mean += d;
                mean /= 4.0;
                double anchor = cfg_.armor_radius;
                if (armor_radius_online_ > 0.0) anchor = armor_radius_online_;
                for (std::size_t i = 0; i < geometry_.plate_distance.size(); ++i) {
                    geometry_.plate_distance[i] = std::clamp(
                        geometry_.plate_distance[i] + (anchor - mean),
                        anchor - cfg_.geometry_learn_range, anchor + cfg_.geometry_learn_range);
                }
                syncGeometryToEkf();
            }
        }

        // 判据必须用**沿视轴的深度**，不能用"到相机的距离"：
        //   ±90° 那两块侧板到相机的距离本来就 > 车心（√(d²+R²) > d），
        //   用距离判会在"看到侧板"时误触发 → yaw 每帧被翻 180° → 模型直接崩（实车踩过）。
        // 只有观测板明确落在**远半侧**（沿视轴比车心还远 ⇒ 径向指反了）才翻，
        // 且连续两帧确认（防单帧异常）。
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
        // ---- 硬约束：让"模型算出的这块板"精确落在观测上（只平移底盘中心）----
        // 单板工况下，滤波器对"中心沿板面法向/切向"这两个方向本就不可辨识，靠软量测
        // 会稳定偏出板心十几厘米（离线实测：径向 140 mm）。把观测当硬约束后，模型必定
        // 穿过这块板，其余三块随之平移 —— 这正是"以识别到的板为锚"的物理含义。
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
        updateRotationRate(valid_pairs[best_armor_index].home, timestamp, best_yaw_trusted);

        // ---- 双模型：同步喂板模型（上科大 2026 §9.4.4）-----------------------
        // 观测 = 可见板的**世界系中心 + 板朝向（径向角）**。只并行运行，不参与输出；
        // 输出切换（按 |ω| 带滞回）是下一片的事。
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
        // 双模型权重（文档 §9.4.4）：|ω| ≤ enter → 全用板模型；≥ exit → 全用整车；
        // 中间是 BOTH 档，按 |ω| 线性融合。权重连续，所以不需要额外滞回。
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
        // soft_nis_gate 打开后，"被拒"只剩关联层的 nomatch —— 那恰恰是高速自旋
        // 刚开始锁定/换板时的正常现象，用它触发重初始化会把正在收敛的状态反复
        // 打掉（实测 5 rad/s：只关掉这一条，重置就从 7 次降到 0~2 次）。
        // 真正的发散交给下面的 stateDiverged()，相机断流交给帧间隔判据。
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
