#include "tracker.hpp"

#include "perception/pnp_solver.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>

namespace auto_aim
{
    namespace
    {
        /// @brief 装甲板四角像素 → UV 观测（顺序与 getArmor3DPoints 一致：
        ///        left_bottom, left_top, right_top, right_bottom）。
        ///        PnP 的 Points_2D 就是这个顺序，所以这里原样搬过来即可；
        ///        半尺寸必须与 PnP 用的同一套（configs/tracker.yaml 的 pnp 段）。
        std::optional<UvObservation> armorUvObservation(const Armor& armor, int plate_id,
                                                        const TrackerConfig& config)
        {
            if (armor.Points_2D.size() < 4) return std::nullopt;
            UvObservation observation;
            observation.plate_id = plate_id;
            const double width = armor.armor_type == auto_aim::small
                                     ? config.uv_armor_small_width
                                     : config.uv_armor_large_width;
            observation.half_width = 0.5 * width;
            observation.half_height = 0.5 * config.uv_armor_height;
            observation.sigma_px = config.uv_sigma_px;
            for (int i = 0; i < 4; ++i) {
                const cv::Point2f& corner = armor.Points_2D[static_cast<std::size_t>(i)];
                if (!std::isfinite(corner.x) || !std::isfinite(corner.y)) return std::nullopt;
                observation.corners[static_cast<std::size_t>(i)](0, 0) = corner.x;
                observation.corners[static_cast<std::size_t>(i)](1, 0) = corner.y;
            }
            return observation;
        }

        /// @brief 装甲板的**一条灯条**端点像素 → 紧凑 UV 观测。
        ///        端点顺序按图像上下取（v 小的当 top），与 awakening-main 一致。
        std::optional<UvCompactObservation> armorUvCompactObservation(
            const Armor& armor, int plate_id, const TrackerConfig& config)
        {
            const cv::Point2f top = armor.left.top;
            const cv::Point2f bottom = armor.left.bottom;
            const cv::Point2f right_top = armor.right.top;
            const cv::Point2f right_bottom = armor.right.bottom;
            for (const cv::Point2f& point : {top, bottom, right_top, right_bottom}) {
                if (!std::isfinite(point.x) || !std::isfinite(point.y)) return std::nullopt;
            }
            UvCompactObservation observation;
            observation.plate_id = plate_id;
            const double width = armor.armor_type == auto_aim::small
                                     ? config.uv_armor_small_width
                                     : config.uv_armor_large_width;
            observation.half_width = 0.5 * width;
            observation.half_height = 0.5 * config.uv_armor_height;
            observation.sigma_px = config.uv_compact_sigma_px;
            observation.sigma_angle = config.uv_compact_sigma_angle;
            // 探测器给的"左灯条"：图像左侧那条（上下端点可能颠倒，这里统一。
            cv::Point2f first = top;
            cv::Point2f second = bottom;
            if (first.y > second.y) std::swap(first, second);
            const double dx = first.x - second.x;
            const double dy = first.y - second.y;
            observation.angle = std::atan2(dx, dy);
            observation.center_x = 0.5 * (first.x + second.x);
            observation.center_y = 0.5 * (first.y + second.y);
            observation.length = std::sqrt(dx * dx + dy * dy);
            return observation;
        }

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

        Matrix<3, 3> cameraToBaseRotation(double yaw, double pitch)
        {
            const double cy = std::cos(yaw);
            const double sy = std::sin(yaw);
            const double cp = std::cos(pitch);
            const double sp = std::sin(pitch);

            // Simulator camera pose in Bevy coordinates:
            // R_home_camera = R_y(-yaw) * R_x(pitch). OpenCV optical axes are
            // converted with diag(1, -1, -1), so the camera-to-home transform
            // is S * R_home_camera * S.
            Matrix<3, 3> bevy_rotation;
            bevy_rotation(0, 0) = cy;
            bevy_rotation(0, 1) = -sy * sp;
            bevy_rotation(0, 2) = -sy * cp;
            bevy_rotation(1, 0) = 0.0;
            bevy_rotation(1, 1) = cp;
            bevy_rotation(1, 2) = -sp;
            bevy_rotation(2, 0) = sy;
            bevy_rotation(2, 1) = cy * sp;
            bevy_rotation(2, 2) = cy * cp;

            const std::array<double, 3> optical_sign{{1.0, -1.0, -1.0}};
            Matrix<3, 3> result;
            for (int row = 0; row < 3; ++row) {
                for (int column = 0; column < 3; ++column) {
                    result(row, column) = optical_sign[row] *
                        bevy_rotation(row, column) * optical_sign[column];
                }
            }
            return result;
        }

        // camera←gimbal 的旋转（手眼外参），默认单位阵
        Matrix<3, 3> cameraToGimbal(const CameraPose& pose)
        {
            Matrix<3, 3> rotation;
            for (int row = 0; row < 3; ++row) {
                for (int column = 0; column < 3; ++column) {
                    rotation(row, column) = pose.camera_to_gimbal[row][column];
                }
            }
            return rotation;
        }

        Vec<3> cameraToGimbalTranslation(const CameraPose& pose)
        {
            Vec<3> translation;
            for (int row = 0; row < 3; ++row) {
                translation(row, 0) = pose.camera_to_gimbal_translation[row];
            }
            return translation;
        }

        // 相机系点 → 世界系：先按手眼外参到云台系（含平移），再由 yaw/pitch 与
        // base_to_world 转到世界系。外参为单位阵/零时与旧行为完全一致。
        Matrix<3, 3> cameraToWorldRotation(const CameraPose& pose)
        {
            Matrix<3, 3> base_to_world;
            for (int row = 0; row < 3; ++row) {
                for (int column = 0; column < 3; ++column) {
                    base_to_world(row, column) = pose.base_to_world[row][column];
                }
            }
            return base_to_world * cameraToBaseRotation(pose.yaw, pose.pitch) *
                cameraToGimbal(pose);
        }

        Vec<3> pointToWorld(const CameraPose& pose, const Vec<3>& camera_point)
        {
            const Vec<3> gimbal_point =
                cameraToGimbal(pose) * camera_point + cameraToGimbalTranslation(pose);
            const Matrix<3, 3> base_to_world = [&pose] {
                Matrix<3, 3> matrix;
                for (int row = 0; row < 3; ++row) {
                    for (int column = 0; column < 3; ++column) {
                        matrix(row, column) = pose.base_to_world[row][column];
                    }
                }
                return matrix;
            }();
            return base_to_world * (cameraToBaseRotation(pose.yaw, pose.pitch) * gimbal_point);
        }

        Vec<3> pointToCamera(const CameraPose& pose, const Vec<3>& world_point)
        {
            const Matrix<3, 3> base_to_world = [&pose] {
                Matrix<3, 3> matrix;
                for (int row = 0; row < 3; ++row) {
                    for (int column = 0; column < 3; ++column) {
                        matrix(row, column) = pose.base_to_world[row][column];
                    }
                }
                return matrix;
            }();
            const Vec<3> gimbal_point =
                cameraToBaseRotation(pose.yaw, pose.pitch).transpose() *
                (base_to_world.transpose() * world_point);
            return cameraToGimbal(pose).transpose() *
                (gimbal_point - cameraToGimbalTranslation(pose));
        }

        Vec<3> transformVector(const Matrix<3, 3>& rotation, const Vec<3>& value)
        {
            Vec<3> result;
            for (int row = 0; row < 3; ++row) {
                for (int column = 0; column < 3; ++column) {
                    result(row, 0) += rotation(row, column) * value(column, 0);
                }
            }
            return result;
        }

        Vec<3> transformCvVector(const Matrix<3, 3>& rotation, const cv::Mat& value)
        {
            Vec<3> input;
            input(0, 0) = value.at<double>(0);
            input(1, 0) = value.at<double>(1);
            input(2, 0) = value.at<double>(2);
            return transformVector(rotation, input);
        }

        std::optional<cv::Mat> transformRotation(
            const Matrix<3, 3>& frame_rotation, const cv::Mat& rvec)
        {
            if (rvec.empty()) return std::nullopt;

            cv::Mat camera_rotation;
            cv::Rodrigues(rvec, camera_rotation);
            Matrix<3, 3> object_to_home;
            for (int row = 0; row < 3; ++row) {
                for (int column = 0; column < 3; ++column) {
                    for (int index = 0; index < 3; ++index) {
                        object_to_home(row, column) += frame_rotation(row, index) *
                            camera_rotation.at<double>(index, column);
                    }
                }
            }

            cv::Mat home_rotation(3, 3, CV_64F);
            for (int row = 0; row < 3; ++row) {
                for (int column = 0; column < 3; ++column) {
                    home_rotation.at<double>(row, column) = object_to_home(row, column);
                }
            }

            cv::Mat home_rvec;
            cv::Rodrigues(home_rotation, home_rvec);
            return home_rvec;
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

        std::optional<YpdObservation> armorYpdObservation(
            const Armor& armor, int plate_id, const TrackerConfig& config)
        {
            if (!armor.solve_result || armor.tvec.empty()) return std::nullopt;
            const auto radial_angle = armorRadialAngle(armor);
            if (!radial_angle) return std::nullopt;

            const double x = armor.tvec.at<double>(0);
            const double y = armor.tvec.at<double>(1);
            const double z = armor.tvec.at<double>(2);
            const double horizontal = std::hypot(x, z);
            const double distance = std::sqrt(x * x + y * y + z * z);
            if (!std::isfinite(distance) || distance <= 1e-4) return std::nullopt;

            YpdObservation observation;
            observation.plate_id = plate_id;
            observation.yaw = std::atan2(x, z);
            observation.pitch = std::atan2(y, horizontal);
            observation.distance = distance;
            observation.armor_yaw = *radial_angle;
            observation.yaw_noise = config.bearing_measure_noise;
            observation.pitch_noise = config.pitch_measure_noise;
            observation.distance_noise = config.measure_noise *
                (1.0 + config.range_noise_distance_scale * distance * distance);
            observation.armor_yaw_noise = config.angle_measure_noise;
            return observation;
        }

        Vec<3> centerFromRadialAngle(
            const Vec<3>& observation, double radial_angle, double radius,
            double armor_y_offset = 0.0)
        {
            Vec<3> center;
            center(0, 0) = observation(0, 0) - radius * std::cos(radial_angle);
            center(1, 0) = observation(1, 0) - armor_y_offset;
            center(2, 0) = observation(2, 0) - radius * std::sin(radial_angle);
            return center;
        }
    }

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
    }

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
        // UV 观测（像素重投影）优先：直接把四角像素喂进 EKF，跳过 PnP 那一层
        // 噪声放大。cfg.uv_observation 关掉时行为与以前完全一致（YPD 路径）。
        const auto uv_observation =
            (cfg_.uv_observation && camera_intrinsics_valid_)
                ? armorUvObservation(armor, plate_id_, cfg_)
                : std::nullopt;
        const auto uv_compact_observation =
            (cfg_.uv_observation && cfg_.uv_compact && camera_intrinsics_valid_)
                ? armorUvCompactObservation(armor, plate_id_, cfg_)
                : std::nullopt;
        const auto ypd_observation = armorYpdObservation(
            home_armor, plate_id_, cfg_);
        if (ypd_observation && !scaleConsistent(armor, ypd_observation->distance)) {
            // 尺度不自洽（塌缩/错解）：不要这条观测，让滤波器滑行。
            predictStateTo(timestamp);
            lostUpdate();
            return false;
        }
        if (uv_compact_observation) {
            measurement_weight = ekf_.updateUvCompact(
                *uv_compact_observation, uvCamera(), cfg_.uv_nis_threshold);
        } else if (uv_observation) {
            measurement_weight = ekf_.updateUv(
                *uv_observation, uvCamera(), cfg_.uv_nis_threshold);
        } else if (ypd_observation) {
            measurement_weight = ekf_.updateYpd(
                *ypd_observation, cfg_.ypd_nis_threshold);
        } else {
            if (radial_angle) {
                measurement_weight = ekf_.updateAngle(
                    *radial_angle, plate_id_, cfg_.angle_measure_noise,
                    cfg_.angle_nis_threshold);
            }
            const Vec<3> observation = toVec(home_armor.tvec);
            const Vec<3> center_observation = radial_angle
                ? centerFromRadialAngle(observation, *radial_angle, cfg_.armor_radius)
                : ekf_.centerFromPlate(observation, plate_id_);
            measurement_weight *= ekf_.updateCenter(
                center_observation, cfg_.measure_noise, cfg_.y_noise_mult,
                cfg_.position_nis_threshold);
        }
        if (measurement_weight <= 0.0 && state_ == TrackerState::TRACKING) {
            lostUpdate();
            return false;
        }
        updateRotationRate(home_armor, timestamp);

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

        bool reacquired = false;
        if (state_ == TrackerState::LOST) {
            if (valid_armors.empty()) {
                ++lost_count_;
                return false;
            }
            if (tryReacquire(valid_armors, timestamp)) {
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
        std::vector<UvObservation> uv_observations;
        std::vector<UvCompactObservation> uv_compact_observations;
        const bool use_uv = cfg_.uv_observation && camera_intrinsics_valid_;
        if (use_uv) {
            uv_observations.reserve(candidates.size());
            uv_compact_observations.reserve(candidates.size());
        }
        for (const auto& candidate : candidates) {
            if (armor_used[candidate.armor_index] || plate_used[candidate.plate_id]) continue;
            const Armor& armor = *valid_armors[candidate.armor_index];
            if (use_uv) {
                // 多块板同帧：同一个刚体，每多一块就多 8 个像素约束。
                if (cfg_.uv_compact) {
                    const auto compact = armorUvCompactObservation(armor, candidate.plate_id, cfg_);
                    if (!compact) continue;
                    armor_used[candidate.armor_index] = true;
                    plate_used[candidate.plate_id] = true;
                    uv_compact_observations.push_back(*compact);
                    if (best_armor_index < 0) {
                        best_armor_index = static_cast<int>(candidate.armor_index);
                        best_plate_id = candidate.plate_id;
                    }
                    continue;
                }
                const auto uv = armorUvObservation(armor, candidate.plate_id, cfg_);
                if (!uv) continue;
                armor_used[candidate.armor_index] = true;
                plate_used[candidate.plate_id] = true;
                uv_observations.push_back(*uv);
                if (best_armor_index < 0) {
                    best_armor_index = static_cast<int>(candidate.armor_index);
                    best_plate_id = candidate.plate_id;
                }
                continue;
            }
            const auto observation = armorYpdObservation(armor, candidate.plate_id, cfg_);
            if (!observation) continue;
            if (!scaleConsistent(armor, observation->distance)) {
                ++dropped_scale_;
                continue;
            }

            armor_used[candidate.armor_index] = true;
            plate_used[candidate.plate_id] = true;
            ypd_observations.push_back(*observation);
            ++accepted_observations_;
            if (best_armor_index < 0) {
                best_armor_index = static_cast<int>(candidate.armor_index);
                best_plate_id = candidate.plate_id;
            }
        }

        if (!uv_compact_observations.empty()) {
            const UvCamera camera = uvCamera();
            last_measurement_weight_ = 0.0;
            for (const auto& observation : uv_compact_observations) {
                last_measurement_weight_ = std::max(
                    last_measurement_weight_,
                    ekf_.updateUvCompact(observation, camera, cfg_.uv_nis_threshold));
            }
            if (last_measurement_weight_ <= 0.0) best_armor_index = -1;
        } else if (!uv_observations.empty()) {
            // 逐条更新（rmcs_auto_aim_v2 的做法）：每喂一条就用**更新后的状态**重新
            // 线性化下一条的预测，比"一次批处理、只在先验处线性化一次"更准，
            // 也省掉 8N×8N 的矩阵求逆（每条只需求 8×8 或 4×4）。
            const UvCamera camera = uvCamera();
            last_measurement_weight_ = 0.0;
            for (const auto& observation : uv_observations) {
                last_measurement_weight_ = std::max(
                    last_measurement_weight_,
                    ekf_.updateUv(observation, camera, cfg_.uv_nis_threshold));
            }
            if (last_measurement_weight_ <= 0.0) best_armor_index = -1;
        } else if (!ypd_observations.empty()) {
            last_measurement_weight_ = ekf_.updateYpdBatch(
                ypd_observations, cfg_.ypd_nis_threshold);
            if (last_measurement_weight_ <= 0.0) best_armor_index = -1;
        } else if (valid_armors.size() == 1) {
            // PnP can theoretically succeed without a usable rotation. Keep one
            // Cartesian fallback only for that degenerate case.
            plate_id_ = selectPlateId(*valid_armors.front());
            const double cost = matchCost(*valid_armors.front(), plate_id_);
            if (!std::isfinite(cost) && state_ == TrackerState::TRACKING) {
                has_observation_ = false;
                lostUpdate();
                return false;
            }
            const Vec<3> observation = toVec(valid_armors.front()->tvec);
            last_measurement_weight_ = ekf_.updateCenter(
                ekf_.centerFromPlate(observation, plate_id_), cfg_.measure_noise,
                cfg_.y_noise_mult, cfg_.position_nis_threshold);
            if (last_measurement_weight_ > 0.0) {
                best_armor_index = 0;
                best_plate_id = plate_id_;
            }
        }

        if (best_armor_index < 0) {
            has_observation_ = false;
            lostUpdate();
            return false;
        }

        plate_id_ = best_plate_id;
        updateRotationRate(valid_pairs[best_armor_index].home, timestamp);
        latest_armor_ = valid_pairs[best_armor_index].camera;
        latest_armor_.tvec = worldToCamera(ekf_.armorPosition(best_plate_id));
        last_observed_armor_ = valid_pairs[best_armor_index].camera;
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
        return worldToCamera(ekf_.getCenter());
    }

    std::array<double, 3> Tracker::getTargetCenterArray() const
    {
        const Vec<3> center = worldToCameraVector(ekf_.getCenter());
        return {center(0, 0), center(1, 0), center(2, 0)};
    }

    std::array<double, 3> Tracker::getTargetVelocity() const
    {
        const Vec<9> state = ekf_.getState();
        Vec<3> velocity;
        velocity(0, 0) = state(1, 0);
        velocity(1, 0) = state(3, 0);
        velocity(2, 0) = state(8, 0);
        velocity = worldToCameraVector(velocity);
        return {velocity(0, 0), velocity(1, 0), velocity(2, 0)};
    }

    std::array<double, 3> Tracker::getTargetCenterWorldArray() const
    {
        const Vec<3> center = ekf_.getCenter();
        return {center(0, 0), center(1, 0), center(2, 0)};
    }

    std::array<double, 3> Tracker::getTargetVelocityWorld() const
    {
        const Vec<9> state = ekf_.getState();
        return {state(1, 0), state(3, 0), state(8, 0)};
    }

    std::array<double, 3> Tracker::worldToCameraPoint(
        const std::array<double, 3>& point) const
    {
        Vec<3> home;
        home(0, 0) = point[0];
        home(1, 0) = point[1];
        home(2, 0) = point[2];
        const Vec<3> camera = pointToCamera(camera_pose_, home);
        return {camera(0, 0), camera(1, 0), camera(2, 0)};
    }

    double Tracker::worldToCameraYaw(double yaw) const
    {
        Vec<3> radial;
        radial(0, 0) = std::cos(yaw);
        radial(1, 0) = 0.0;
        radial(2, 0) = std::sin(yaw);
        const Vec<3> camera = worldToCameraVector(radial);
        return std::atan2(camera(2, 0), camera(0, 0));
    }

    std::array<cv::Mat, 4> Tracker::getEstimatedArmorPositions() const
    {
        std::array<cv::Mat, 4> positions;
        for (int plate = 0; plate < 4; ++plate) {
            positions[plate] = worldToCamera(ekf_.armorPosition(plate));
        }
        return positions;
    }

    double Tracker::getYaw() const
    {
        Vec<3> radial;
        radial(0, 0) = std::cos(ekf_.getYaw());
        radial(1, 0) = 0.0;
        radial(2, 0) = std::sin(ekf_.getYaw());
        radial = worldToCameraVector(radial);
        return std::atan2(radial(2, 0), radial(0, 0));
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
        const auto observation = armorYpdObservation(armor, plate_id, cfg_);
        if (observation) {
            const YpdObservation predicted = ekf_.predictYpd(plate_id);
            const double yaw_error = ArmorEKF::normalizeAngle(
                observation->yaw - predicted.yaw);
            const double pitch_error = ArmorEKF::normalizeAngle(
                observation->pitch - predicted.pitch);
            const double range_error = observation->distance - predicted.distance;
            const double armor_angle_error = std::abs(ArmorEKF::normalizeAngle(
                observation->armor_yaw - predicted.armor_yaw));
            if (std::abs(range_error) > cfg_.max_match_distance ||
                armor_angle_error > cfg_.max_angle_error) {
                return std::numeric_limits<double>::infinity();
            }

            const double bearing_distance = predicted.distance *
                std::hypot(yaw_error, pitch_error);
            return bearing_distance + std::abs(range_error) +
                cfg_.angle_match_weight * armor_angle_error;
        }

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

    void Tracker::updateRotationRate(const Armor& armor, double timestamp)
    {
        const auto radial_angle = armorRadialAngle(armor);
        if (!radial_angle ||
            !rotation_rate_estimator_.update(*radial_angle, timestamp)) {
            // One unusable sample must not erase a rate the filter already
            // holds. Halving omega here used to make a single dropped frame
            // (or a plate switch) look like the chassis had stopped spinning.
            return;
        }
        ekf_.updateOmega(rotation_rate_estimator_.omega(),
                         cfg_.omega_measure_noise,
                         cfg_.omega_nis_threshold);
    }

    void Tracker::setCameraPose(const CameraPose& pose)
    {
        if (!pose.valid || !std::isfinite(pose.yaw) || !std::isfinite(pose.pitch)) {
            return;
        }
        camera_pose_ = pose;
    }

    void Tracker::setCameraAngles(double yaw, double pitch)
    {
        if (std::isfinite(yaw) && std::isfinite(pitch)) {
            camera_pose_.valid = true;
            camera_pose_.timestamp_valid = false;
            camera_pose_.yaw = yaw;
            camera_pose_.pitch = pitch;
        }
    }

    void Tracker::setCameraIntrinsics(double fx, double fy, double cx, double cy)
    {
        if (!(fx > 1.0) || !(fy > 1.0)) return;
        camera_fx_ = fx;
        camera_fy_ = fy;
        camera_cx_ = cx;
        camera_cy_ = cy;
        camera_intrinsics_valid_ = true;
    }

    Armor Tracker::toWorld(const Armor& armor) const
    {
        if (!armor.solve_result || armor.tvec.empty()) return armor;

        if (!camera_pose_.valid) return armor;
        const Matrix<3, 3> camera_to_world = cameraToWorldRotation(camera_pose_);
        Armor result = armor;
        const Vec<3> position = pointToWorld(camera_pose_, toVec(armor.tvec));
        result.tvec = toMat(position);
        const auto world_rotation = transformRotation(camera_to_world, armor.rvec);
        if (world_rotation) result.rvec = *world_rotation;
        return result;
    }

    Vec<3> Tracker::worldToCameraVector(const Vec<3>& value) const
    {
        const Matrix<3, 3> world_to_camera = cameraToWorldRotation(
            camera_pose_).transpose();
        return transformVector(world_to_camera, value);
    }



    bool Tracker::scaleConsistent(const Armor& camera_armor, double distance) const
    {
        if (cfg_.scale_gate_ratio <= 0.0 || !camera_intrinsics_valid_) return true;
        if (camera_armor.Points_2D.size() < 4 || !(distance > 0.1)) return true;
        const double width_m = camera_armor.armor_type == auto_aim::small
                                   ? cfg_.uv_armor_small_width
                                   : cfg_.uv_armor_large_width;
        const double height_m = cfg_.uv_armor_height;
        const auto& p = camera_armor.Points_2D;   // lb, lt, rt, rb
        const double px_width = 0.5 * (cv::norm(p[1] - p[2]) + cv::norm(p[0] - p[3]));
        const double px_height = 0.5 * (cv::norm(p[1] - p[0]) + cv::norm(p[2] - p[3]));
        if (px_width < 1.0 || px_height < 1.0) return false;   // 关键点塌缩
        const double implied_width = camera_fx_ * width_m / px_width;
        const double implied_height = camera_fy_ * height_m / px_height;
        const auto ratio_of = [](double lhs, double rhs) {
            return std::max(lhs, rhs) / std::max(1e-3, std::min(lhs, rhs));
        };
        const double tolerance = 1.0 + cfg_.scale_gate_ratio;
        // 宽、高各自跟 PnP 距离比一次：**任一维度自洽就放行**。
        // 为什么不能像原来那样"两个隐含距离取平均再比"：
        //   · 宽度那条用标定过的等效宽度（0.1315 m），远近都准；
        //   · 高度那条硬套板高 0.06 m，而近距离大靶面下网络关键点的竖直跨度
        //     并不服从这个模型 —— 实测 1 m 工况：宽度隐含 ≈1.02 m（真值 1.00 m ✓）、
        //     高度隐含 ≈0.46 m ✗，两者一平均就把**好观测**判成不一致。
        // 这正是 NUC 上 1 m 工况 scale 拒收 ~70%、跟踪一直 TEMP_LOST 的原因。
        // 远距那种荒唐解（7.5 m 处 PnP 给 0.34 m）两个维度会同时不符，
        // 照样拦得住，所以闸门原本的目的没有丢。
        return ratio_of(distance, implied_width) <= tolerance ||
               ratio_of(distance, implied_height) <= tolerance;
    }

    void Tracker::refineArmorPose(Armor& armor) const
    {
        // **默认关**：实测它没有收益 —— 4 个角点 + 6 自由度时 IPPE 已经是精确最小二乘
        // （重投影残差中位 0.09 px），LM 精修"无残差可降"；A/B 里 observation error
        // 0.087 → 0.095 m（差异在噪声内）。UV 的价值不在"精修同一块板的位姿"，
        // 而在更多约束/更好的噪声模型，见 docs/uv_observation.md。
        static const bool enabled = [] {
            const char* value = std::getenv("ULTRA_VISION_UV_POSE_REFINE");
            return value != nullptr && std::string(value) != "0";
        }();
        if (!enabled || !camera_intrinsics_valid_) return;
        if (!armor.solve_result || armor.tvec.empty() || armor.Points_2D.size() < 4) return;
        const auto object_points = getArmor3DPoints(
            armor.armor_type == auto_aim::small ? 0 : 1,
            static_cast<float>(cfg_.uv_armor_small_width),
            static_cast<float>(cfg_.uv_armor_large_width),
            static_cast<float>(cfg_.uv_armor_height));
        if (object_points.size() != 4) return;
        const std::vector<cv::Point2f> image_points = {
            armor.left.bottom, armor.left.top, armor.right.top, armor.right.bottom};
        cv::Mat camera_matrix = (cv::Mat_<double>(3, 3) << camera_fx_, 0.0, camera_cx_,
                                 0.0, camera_fy_, camera_cy_, 0.0, 0.0, 1.0);
        cv::Mat rvec = armor.rvec.clone();
        cv::Mat tvec = armor.tvec.clone();
        try {
            // LM 迭代重投影：初值来自 PnP，只优化这一块板的位姿（无关联/分支问题）。
            cv::solvePnPRefineLM(object_points, image_points, camera_matrix, cv::Mat(), rvec, tvec);
        } catch (const cv::Exception&) {
            return;   // 精修失败就保留 PnP 结果
        }
        if (cv::checkRange(tvec) && cv::checkRange(rvec)) {
            armor.rvec = rvec;
            armor.tvec = tvec;
        }
    }

    UvCamera Tracker::uvCamera() const
    {
        UvCamera camera;
        camera.fx = camera_fx_;
        camera.fy = camera_fy_;
        camera.cx = camera_cx_;
        camera.cy = camera_cy_;
        camera.world_to_camera = cameraToWorldRotation(camera_pose_).transpose();
        return camera;
    }

    cv::Mat Tracker::worldToCamera(const Vec<3>& value) const
    {
        return toMat(worldToCameraVector(value));
    }
}
