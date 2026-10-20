#ifndef TRACKER_HPP
#define TRACKER_HPP

#include <array>
#include <deque>
#include <chrono>
#include <cmath>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/video/tracking.hpp>

#include "estimation/rotation_rate_estimator.hpp"
#include "armor_ekf.hpp"
#include "common/armor_geometry.hpp"
#include "common/chassis_model.hpp"
#include "common/types.hpp"

namespace auto_aim
{
    enum class TrackerState { DETECTING, TRACKING, TEMP_LOST, LOST };

    struct CameraPose {
        bool valid = true;
        bool timestamp_valid = false;
        double timestamp = 0.0;
        double yaw = 0.0;
        double pitch = 0.0;
        std::array<std::array<double, 3>, 3> base_to_world{{
            {{1.0, 0.0, 0.0}},
            {{0.0, 1.0, 0.0}},
            {{0.0, 0.0, 1.0}}
        }};
        std::array<std::array<double, 3>, 3> camera_to_gimbal{{
            {{1.0, 0.0, 0.0}},
            {{0.0, 1.0, 0.0}},
            {{0.0, 0.0, 1.0}}
        }};
        std::array<double, 3> camera_to_gimbal_translation{{0.0, 0.0, 0.0}};
    };

    struct TrackerConfig {
        bool watchdog_enabled = true;
        int watchdog_window = 20;
        double watchdog_reject_ratio = 0.4;
        int watchdog_min_samples = 8;
        double watchdog_max_speed = 12.0;          // m/s
        double watchdog_max_yaw_rate = 20.0;       // rad/s
        double watchdog_max_range = 30.0;          // 位置模长 m
        double watchdog_max_position_cov = 25.0;   // P 的位置对角 m²
        double watchdog_max_frame_gap = 0.20;      // 帧间隔上限 s
        int watchdog_min_converge_updates = 3;
        double plate_lock_margin = 1.35;
        int min_detect_frames = 3;
        int max_lost_frames = 15;
        int reacquire_frames = 3;
        double max_match_distance = 0.35;
        double association_margin = 1.4;
        double reacquire_center_error = 0.25;
        double reacquire_max_dt = 0.15;
        double max_camera_pose_dt = 0.05;

        RotationRateEstimatorConfig rotation_rate;
        double omega_measure_noise = 1.0;
        double omega_nis_threshold = 9.0;

        double process_noise_pos = 1.0;
        // 平移自适应过程噪声：静止时用 process_noise_pos
        double process_noise_pos_speed_ref = 0.28;
        double process_noise_pos_max = 40.0;
        double process_noise_vel = 400.0;
        double measure_noise = 0.01;
        double default_dt = 0.033;

        // Parameters of the circular armor EKF.
        double armor_radius = 0.21;
        double armor_tilt_deg = 15.0;
        bool model_lead_center = true;
        double y_noise_mult = 30.0;
        double p_cov_min = 1e-3;
        double uv_armor_small_width = 0.134;
        double uv_armor_large_width = 0.225;
        double uv_armor_height = 0.057;
        double scale_gate_ratio = 0.35;
        double armor_small_width = 0.1315;
        double armor_large_width = 0.221;
        double armor_height = 0.060;
        std::array<double, 4> plate_distance{{0.0, 0.0, 0.0, 0.0}};
        std::array<bool, 4> plate_large{{false, false, false, false}};
        double angle_measure_noise = 0.04;
        double bearing_measure_noise = 0.0025;
        double pitch_measure_noise = 0.005;
        double range_noise_distance_scale = 0.3;
        double ypd_distance_sigma_coeff = 0.0;
        double ypd_nis_threshold = 9.488;
        double max_angle_error = 0.35;
        bool adaptive_association_gate = true;
        double association_gate_sigma = 1.5;
        double association_gate_max = 1.30;   // rad ≈ 74.5°，必须 < 90°
        double range_gate_sigma = 2.0;
        double range_gate_max = 1.00;         // m
        double angle_match_weight = 0.08;
        bool geometry_learn = true;
        double geometry_learn_gain = 0.05;        // 每帧最多修 5%
        double geometry_learn_max_step = 0.004;   // 单帧最大 4 mm
        double geometry_learn_max_error = 0.05;   // 残差 > 5 cm 当异常，不学
        double geometry_learn_range = 0.08;       // 相对初值最多学 ±8 cm
        double geometry_learn_max_omega = 1.0;    // rad/s；<=0 表示不限
        bool geometry_learn_height = false;

        bool reanchor_yaw_on_plate_switch = true;

        double distance_noise_scale = 1.0;
        double armor_angle_noise_base = 0.09;
        double graze_gate_deg = 75.0;
        double graze_noise_scale = 100.0;  // 距离噪声放大倍数
        double position_nis_threshold = 7.815;
        double angle_nis_threshold = 3.841;
        bool soft_nis_gate = true;
        std::array<double, 4> armor_y_offsets{{0.0, 0.0, 0.0, 0.0}};

    };

    class Tracker
    {
    public:
        explicit Tracker(const TrackerConfig& cfg);
        ~Tracker() = default;

        void init(const Armor& armor, double timestamp);
        void init(const std::vector<Armor>& armors, double timestamp);
        Armor predict(double timestamp);
        bool update(const Armor& armor, double timestamp);
        bool update(const std::vector<Armor>& armors, double timestamp);
        void predictTo(double timestamp);
        void lostUpdate();

        TrackerState getState() const { return state_; }
        const Armor& getLatestArmor() const { return latest_armor_; }
        const Armor& getLastObservedArmor() const { return last_observed_armor_; }
        int getLastObservedPlateId() const { return last_observed_plate_id_; }
        bool hasObservation() const { return has_observation_; }
        double lastObservationTime() const { return last_observation_time_; }
        double getConfidence() const { return confidence_; }
        int getLostCount() const { return lost_count_; }
        int getDroppedAmbiguous() const { return dropped_ambiguous_; }
        /// @brief 四块预测板全都够不着（超出 max_match_distance/max_angle_error）的框数。
        int getDroppedNoMatch() const { return dropped_nomatch_; }
        int getDroppedByScale() const { return dropped_scale_; }
        int getAcceptedObservations() const { return accepted_observations_; }

        cv::Mat getTargetCenter() const;
        std::array<double, 3> getTargetCenterArray() const;
        std::array<double, 3> getTargetVelocity() const;
        std::array<double, 3> getTargetCenterWorldArray() const;
        std::array<double, 3> getTargetVelocityWorld() const;
        std::array<cv::Mat, 4> getEstimatedArmorPositions() const;
        const ChassisGeometry& chassisGeometry() const { return geometry_; }
        /// @brief 直接写入慢层几何（会同步到 EKF 的逐板半径）。
        void setChassisGeometry(const ChassisGeometry& geometry);
        void syncGeometryFromConfig();
        void syncGeometryToEkf();

        ArmorPose armorPose(int plate) const;
        ChassisGeometry effectiveChassisGeometry() const;
        double groupRadiusDelta() const { return ekf_.groupRadiusDelta(); }
        double groupHeightDelta() const { return ekf_.groupHeightDelta(); }
        double armorHalfWidth(int plate) const { return geometry_.halfWidth(plate); }
        double armorHalfHeight() const { return geometry_.half_height; }

        struct ModelSnapshot
        {
            bool valid = false;
            double world_yaw = 0.0;                       // 外推后的底盘 yaw（home 系）
            double omega = 0.0;                           // rad/s
            ChassisGeometry geometry;                     // 用的慢层几何
            cv::Vec3d center_camera{0.0, 0.0, 0.0};       // 车心（render 相机系）
            std::array<ArmorPose, 4> poses{};             // 四块板（render 相机系）
        };
        ModelSnapshot modelAt(double dt, const CameraPose& render_pose) const;
        ModelSnapshot modelAt(double dt) const { return modelAt(dt, camera_pose_); }

        void setArmorTiltDeg(double degrees)
        {
            if (std::isfinite(degrees) && std::abs(degrees) < 60.0)
            {
                cfg_.armor_tilt_deg = degrees;
                geometry_.armor_pitch = degrees * CV_PI / 180.0;
            }
        }
        double getArmorTiltDeg() const { return cfg_.armor_tilt_deg; }
        double getYaw() const;
        double getYawWorld() const { return ekf_.getYaw(); }
        double getOmega() const { return ekf_.getOmega(); }
        double getRotationRateOmega() const {
            return rotation_rate_estimator_.omega();
        }
        double getEkfOmega() const { return ekf_.getOmega(); }
        double ekfCovarianceDiag(int i) const { return ekf_.covarianceDiag(i); }
        bool hasRotationRate() const { return rotation_rate_estimator_.valid(); }
        double getArmorRadius() const { return cfg_.armor_radius; }
        double getLastMeasurementWeight() const { return last_measurement_weight_; }
        double lastViewDeg() const { return last_view_deg_; }

        /// @brief 收敛看门狗统计：触发次数 / 当前拒绝比例 / 是否收敛。
        struct WatchdogStats
        {
            int resets = 0;
            double reject_ratio = 0.0;
            int samples = 0;
            bool converged = true;
        };
        WatchdogStats watchdogStats() const;

        std::array<double, 3> worldToCameraPoint(
            const std::array<double, 3>& point) const;
        double worldToCameraYaw(double yaw) const;

        void setCameraPose(const CameraPose& pose);
        void setCameraAngles(double yaw, double pitch);

        void setCameraIntrinsics(double fx, double fy, double cx, double cy);
        bool hasCameraIntrinsics() const { return camera_intrinsics_valid_; }

        int plateIdFor(const Armor& armor) const { return selectPlateId(toWorld(armor)); }

    private:

        void noteMeasurement(bool accepted);
        /// @brief 看门狗判定；命中则强制重初始化并返回 true。
        bool watchdogTrip(const char* reason);
        /// @brief 状态发散检查（速度/角速度/位置/位置协方差越界）。
        bool stateDiverged() const;

        TrackerConfig cfg_;
        ArmorEKF ekf_;
        /// 底盘慢层几何（唯一模型来源；由 config / 观测慢收敛更新）。
        ChassisGeometry geometry_;
        int previous_plate_id_ = -1;
        double last_view_deg_ = -1.0;

        std::deque<bool> recent_measurements_;   // 最近若干帧的量测接受情况
        int watchdog_resets_ = 0;
        int accepted_since_init_ = 0;
        bool force_reinit_ = false;              // 下一帧跳过 tryReacquire，直接 init
        double last_reject_ratio_ = 0.0;
        RotationRateEstimator rotation_rate_estimator_;
        double last_timestamp_;
        double debug_last_timestamp_ = 0.0;   // ULTRA_VISION_OMEGA_DEBUG 用
        TrackerState state_;
        int detect_count_;
        int lost_count_;
        double confidence_;
        int plate_id_;
        double last_measurement_weight_;
        Armor latest_armor_;
        Armor last_observed_armor_;
        int last_observed_plate_id_;
        double last_observation_time_ = 0.0;   // 只在真的接受观测时更新（lastObservationTime）
        bool has_observation_;
        int dropped_ambiguous_ = 0;
        int dropped_nomatch_ = 0;
        int dropped_scale_ = 0;
        int accepted_observations_ = 0;
        CameraPose camera_pose_;
        bool camera_intrinsics_valid_ = false;
        double camera_fx_ = 0.0;
        double camera_fy_ = 0.0;
        double camera_cx_ = 0.0;
        double camera_cy_ = 0.0;

        Armor toWorld(const Armor& armor) const;
        Vec<3> worldToCameraVector(const Vec<3>& value) const;

    public:
        cv::Vec3d worldDirectionToCamera(const cv::Vec3d& world) const
        {
            Vec<3> home;
            home(0, 0) = world[0];
            home(1, 0) = world[1];
            home(2, 0) = world[2];
            const Vec<3> camera = worldToCameraVector(home);
            return cv::Vec3d(camera(0, 0), camera(1, 0), camera(2, 0));
        }

    private:
        cv::Mat worldToCamera(const Vec<3>& value) const;

        void refineArmorPose(Armor& armor) const;
        bool scaleConsistent(const Armor& camera_armor, double distance) const;
        void predictStateTo(double timestamp);
        bool tryReacquire(const std::vector<const Armor*>& observations,
                          double timestamp);
        int selectPlateId(const Armor& armor) const;
        double distanceToPlate(const cv::Mat& tvec, int plate_id) const;
        double matchCost(const Armor& armor, int plate_id) const;
        void updateRotationRate(const Armor& armor, double timestamp, bool yaw_valid = true,
                                int plate_id = -1);

    };
}

#endif
