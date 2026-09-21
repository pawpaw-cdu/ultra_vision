#ifndef TRACKER_HPP
#define TRACKER_HPP

#include <array>
#include <chrono>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/video/tracking.hpp>

#include "estimation/rotation_rate_estimator.hpp"
#include "armor_ekf.hpp"
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
        // world <- base. Simulator uses identity; a real robot can fill this
        // from IMU so the target EKF stays in the world frame.
        std::array<std::array<double, 3>, 3> base_to_world{{
            {{1.0, 0.0, 0.0}},
            {{0.0, 1.0, 0.0}},
            {{0.0, 0.0, 1.0}}
        }};
        // 相机 → 云台的外参（**手眼标定**结果，见 tools/hand_eye_calibrate）。
        // 相机光心相对云台旋转中心有平移和偏转：不补的话，瞄点会随距离/偏角系统性
        // 偏（1° 在 7 m 处 = 12 cm），底盘或云台转动时偏差还会跟着转。
        // 仿真里相机就在光心上 → 单位阵 + 零平移（等价于旧行为）。
        std::array<std::array<double, 3>, 3> camera_to_gimbal{{
            {{1.0, 0.0, 0.0}},
            {{0.0, 1.0, 0.0}},
            {{0.0, 0.0, 1.0}}
        }};
        std::array<double, 3> camera_to_gimbal_translation{{0.0, 0.0, 0.0}};
    };

    struct TrackerConfig {
        int min_detect_frames = 3;
        int max_lost_frames = 15;
        int reacquire_frames = 3;
        double max_match_distance = 1.5;
        // 关联的"最优 vs 次优"判据：最优代价必须**明显**小于次优（best*margin ≤ second）
        // 才接受这块观测。四块板相隔 90°，一块观测常常"两块板都能解释"，按最近选
        // 就是误关联（实测板号一致率只有 79%，误关联帧的观测误差 0.13 m vs
        // 一致帧 0.07 m，而底盘中心误差 0.11 m 正好卡在中间）。模棱两可的帧宁可
        // 不给观测，让滤波器滑行。1.0 = 关闭该判据。
        double association_margin = 1.4;
        double reacquire_center_error = 0.25;
        double reacquire_max_dt = 0.15;
        double max_camera_pose_dt = 0.05;

        RotationRateEstimatorConfig rotation_rate;
        // Fusion weights for the external radial-angle rate estimate. It is
        // applied as a gated scalar measurement of the EKF rate state.
        //
        // The external estimate is deliberately down-weighted: on a 3 rad/s
        // small-gyro run it proved biased low (|omega| median 1.48 vs a true
        // 3.0), while the EKF's own estimate from the armor_yaw sequence
        // reached |omega| median 2.49. Treating it as a strong measurement made
        // the fused rate worse, so it is kept as a weak prior only.
        double omega_measure_noise = 1.0;
        double omega_nis_threshold = 9.0;

        double process_noise_pos = 1.0;
        // 平移自适应过程噪声：静止时用 process_noise_pos，速度越大按
        // (1+(|v|/speed_ref)²) 放大，上限 process_noise_pos_max。
        // 见 ArmorEKF::Config 的注释与 configs/tracker.yaml 里的实测数据。
        double process_noise_pos_speed_ref = 0.28;
        double process_noise_pos_max = 40.0;
        double process_noise_vel = 400.0;
        double measure_noise = 0.01;
        double default_dt = 0.033;

        // Parameters of the circular armor EKF.
        double armor_radius = 0.21;
        double y_noise_mult = 30.0;
        double p_cov_min = 1e-3;
        // ---- UV（像素重投影）观测 -------------------------------------------
        // 打开后，估计器的更新改用"装甲板四角像素 + 重投影"（ArmorEKF::updateUvBatch），
        // 不再用 PnP 解出的 yaw/pitch/distance/armor_yaw 当量测：PnP 会把像素噪声放大
        // 而且相关（距离尤其病态），UV 只做一次最小二乘、量测噪声就是像素噪声。
        // PnP 仍然用于初始化、板号匹配与自适应（这些需要三维位姿），所以这一项是
        // **观测路径**的 A/B，不是"删掉 PnP"。见 docs/uv_observation.md。
        bool uv_observation = false;
        double uv_sigma_px = 6.0;
        double uv_nis_threshold = 40.0;   // 8 维卡方门限（单块板 4 角点）
        // 紧凑 UV：一个灯条 → (角度, 中心 x, 中心 y, 长度) 4 维（借 awakening-main
        // 的 points_to_observation）。比四角点少一半维度、没有角点配对二义。
        bool uv_compact = false;
        double uv_compact_sigma_px = 6.0;
        double uv_compact_sigma_angle = 0.05;
        // UV 重投影用的板面尺寸。**必须与探测器的点语义匹配**，而不是"标准装甲板
        // 的物理尺寸"：网络的 4 个关键点并不是 135×125 板的四个角（实测：按
        // 0.135×0.125 重投影，残差中位 66 px、状态直接发散；按 pnp 段标定出来的
        // 0.134×0.057 重投影，静止时残差 0.5~1.2 px）。所以这里默认取与 pnp 段
        // 相同的"等效尺寸"，并留成可标定量：换检测器/换模型都要重新量残差。
        double uv_armor_small_width = 0.134;
        double uv_armor_large_width = 0.225;
        double uv_armor_height = 0.057;
        // 尺度一致性闸门：PnP 距离与像素尺寸反推距离的允许相对偏差（0 = 关）。
        double scale_gate_ratio = 0.35;
        // 与 PnP 同一套几何尺寸（configs/tracker.yaml 的 pnp 段）。
        double armor_small_width = 0.1315;
        double armor_large_width = 0.221;
        double armor_height = 0.060;
        double angle_measure_noise = 0.04;
        double bearing_measure_noise = 0.0025;
        double pitch_measure_noise = 0.005;
        double range_noise_distance_scale = 0.3;
        double ypd_nis_threshold = 9.488;
        double max_angle_error = 0.70;
        double angle_match_weight = 0.08;
        double position_nis_threshold = 7.815;
        double angle_nis_threshold = 3.841;
        // Per-armor vertical center offsets. They remain zero for a symmetric
        // real robot model and can be calibrated for a simulator vehicle.
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
        // Advance the estimate to `timestamp` without registering a missed
        // detection. The asynchronous detector simply has no new result for
        // some frames, and those must not count towards the loss timeout.
        void predictTo(double timestamp);
        void lostUpdate();

        TrackerState getState() const { return state_; }
        const Armor& getLatestArmor() const { return latest_armor_; }
        const Armor& getLastObservedArmor() const { return last_observed_armor_; }
        int getLastObservedPlateId() const { return last_observed_plate_id_; }
        bool hasObservation() const { return has_observation_; }
        double getConfidence() const { return confidence_; }
        int getLostCount() const { return lost_count_; }
        /// @brief 上一次 update() 里各阶段的观测计数，用来回答"为什么没更新"：
        ///        检测框 → 被判定模棱两可丢掉（association_margin）/ 被尺度闸门
        ///        丢掉（scale_gate）/ 真正接受的条数。三个数一起看就知道该修
        ///        关联层还是修检测器（见 node_sim 的 `assoc:` 日志）。
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
        double getYaw() const;
        double getYawWorld() const { return ekf_.getYaw(); }
        // The EKF owns the rate state: the external estimator is only one of
        // its measurements, so both the selector and the telemetry read the
        // fused value from the filter.
        double getOmega() const { return ekf_.getOmega(); }
        double getRotationRateOmega() const {
            return rotation_rate_estimator_.omega();
        }
        double getEkfOmega() const { return ekf_.getOmega(); }
        bool hasRotationRate() const { return rotation_rate_estimator_.valid(); }
        double getArmorRadius() const { return cfg_.armor_radius; }
        double getLastMeasurementWeight() const { return last_measurement_weight_; }

        std::array<double, 3> worldToCameraPoint(
            const std::array<double, 3>& point) const;
        double worldToCameraYaw(double yaw) const;

        // The EKF state is maintained in the world frame. Set the timestamped
        // camera pose used by the next observation before update().
        void setCameraPose(const CameraPose& pose);
        void setCameraAngles(double yaw, double pitch);

        /// @brief UV 观测用的相机内参（帧尺寸变化时由上层重新调用）。
        ///        只有 cfg.uv_observation 打开时才需要；world→camera 的旋转仍由
        ///        setCameraPose() 提供，两处必须同一时刻。
        void setCameraIntrinsics(double fx, double fy, double cx, double cy);
        bool hasCameraIntrinsics() const { return camera_intrinsics_valid_; }

    private:
        TrackerConfig cfg_;
        ArmorEKF ekf_;
        RotationRateEstimator rotation_rate_estimator_;
        double last_timestamp_;
        TrackerState state_;
        int detect_count_;
        int lost_count_;
        double confidence_;
        int plate_id_;
        double last_measurement_weight_;
        Armor latest_armor_;
        Armor last_observed_armor_;
        int last_observed_plate_id_;
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
        cv::Mat worldToCamera(const Vec<3>& value) const;
        /// @brief 用检测到的角点像素**精修这一块装甲板的位姿**（PnP 初值 + LM 迭代）。
        ///        这是"UV 的思路"用在观测层：重投影只约束**这一块板自己的 6 自由度位姿**
        ///        （良态、没有四块板的分支歧义），再把精修后的 yaw/pitch/distance/armor_yaw
        ///        喂给整车 EKF —— 架构不变，但拿掉了 PnP 对像素噪声的放大。
        ///        ULTRA_VISION_UV_POSE_REFINE=0 可关（默认开，见 docs/uv_observation.md）。
        void refineArmorPose(Armor& armor) const;
        /// @brief 尺度一致性闸门：PnP 的距离必须与"实测靶面像素尺寸反推的距离"相符。
        ///        远距（7.5 m）靶面只有 ~10 px，关键点一旦塌缩，PnP 会给出 0.3 m 这种
        ///        荒唐解并且**完美拟合塌缩的点**（实测 observation error 7.36 m）；
        ///        用同一套物点尺寸反推距离做交叉校验就能把它挡掉。
        bool scaleConsistent(const Armor& camera_armor, double distance) const;
        /// @brief UV 观测用的相机模型（内参 + 当前位姿的 world→camera）。
        UvCamera uvCamera() const;

        void predictStateTo(double timestamp);
        bool tryReacquire(const std::vector<const Armor*>& observations,
                          double timestamp);
        int selectPlateId(const Armor& armor) const;
        double distanceToPlate(const cv::Mat& tvec, int plate_id) const;
        double matchCost(const Armor& armor, int plate_id) const;
        void updateRotationRate(const Armor& armor, double timestamp);
    };
}

#endif
