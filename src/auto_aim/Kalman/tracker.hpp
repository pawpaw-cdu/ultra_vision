#ifndef TRACKER_HPP
#define TRACKER_HPP

#include <array>
#include <deque>
#include <chrono>
#include <cmath>
#include <atomic>
#include <functional>
#include <mutex>
#include <optional>
#include <thread>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/video/tracking.hpp>

#include "estimation/rotation_rate_estimator.hpp"
#include "armor_ekf.hpp"
#include "plate_ekf.hpp"
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

    struct LocalFitConfig
    {
        bool enabled = false;
        int window_frames = 24;         // 滑窗帧数（覆盖相位 ≥ π/3 才有 ω 可辨识性）
        int min_bars = 8;               // 窗口内至少这么多条灯条观测才拟合
        int period_frames = 10;         // 每多少帧触发一次拟合（低频异步）
        double max_result_age = 0.20;   // s：拟合结果比当前帧老过它就外推/丢弃
        int max_window_gap_frames = 6;  // 连续丢帧超过它就清空窗口（避免跨运动段）
        double front_gate = 0.5;
        double assoc_gate_px = 45.0;    // 帧间板心关联门限（px），配合常速预测
        double plate_width = 0.135;
        double lightbar_len = 0.060;
        double min_depth = 0.8;
        double max_depth = 9.0;         // 上限（m）
        bool joint_plates = false;
        int joint_max_arcs = 4;         // 联合时最多用几条板弧
        double yaw_noise = 0.02;        // 拟合 yaw 的观测噪声（rad²，质量再缩放）
        double omega_noise = 0.05;      // 拟合 ω 的观测噪声
        bool use_yaw = false;
        double yaw_use_omega_min = 1.0;    // rad/s
        double yaw_use_omega_full = 2.5;   // rad/s
        double yaw_max_rmse_px = 3.0;          // yaw 可注入的最大拟合 rmse
        double yaw_min_phase_span_deg = 20.0;  // yaw 可注入的最小弧段相位跨度
        double omega_max_rmse_px = 3.0;
        double yaw_bias_alpha = 0.02;      // 学习率
        double yaw_bias_limit_deg = 30.0;  // 限幅（防跑飞）
        double ref_yaw_gate_deg = 40.0;
        double yaw_rate_sign = -1.0;
        bool also_update_center = false;
        double center_noise = 0.01;
        bool use_fit_radius = false;
        double radius_alpha = 0.25;      // 一阶低通系数（每次有效拟合最多跟 25%）
        double radius_max_step = 0.02;   // 单次最大变化（m）
        bool center_anchor = true;
        double center_anchor_spin_min = 1.0;      // rad/s
        double center_anchor_move_max = 0.15;     // m/s
        double center_anchor_gain_rotating = 0.35;
        double center_anchor_gain_other = 0.10;
        double center_anchor_max_error = 0.60;    // 差太多（m）说明拟合不可信
        double max_slow_result_age = 2.0;
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
        bool hard_plate_constraint = true;
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
        double pnp_yaw_trust_deg = 25.0;
        bool geometry_learn = true;
        double geometry_learn_gain = 0.05;        // 每帧最多修 5%
        double geometry_learn_max_step = 0.004;   // 单帧最大 4 mm
        double geometry_learn_max_error = 0.05;   // 残差 > 5 cm 当异常，不学
        double geometry_learn_range = 0.08;       // 相对初值最多学 ±8 cm
        double geometry_learn_max_omega = 1.0;    // rad/s；<=0 表示不限
        bool geometry_learn_height = false;

        bool nearest_plate_constraint = true;
        bool reanchor_yaw_on_plate_switch = true;


        bool joint_solve = true;
        double joint_solve_max_residual = 0.06;   // 联立残差 > 6cm 视为不可信，不喂
        double joint_solve_yaw_noise = 0.0004;    // (rad²) ≈ 1.1°
        double joint_solve_center_noise = 0.0004; // (m²) ≈ 2cm
        double distance_noise_scale = 1.0;
        double armor_angle_noise_base = 0.09;
        double graze_gate_deg = 75.0;
        double graze_noise_scale = 100.0;  // 距离噪声放大倍数
        double nearest_plate_margin_m = 0.05;   // 明显违反才翻（防抖）
        double position_nis_threshold = 7.815;
        double angle_nis_threshold = 3.841;
        bool soft_nis_gate = true;
        bool plate_model_enabled = false;
        double plate_model_process_acc = 20.0;
        double plate_model_process_yaw_acc = 50.0;
        double plate_model_enter_omega = 0.7;   // |ω| 低于此值 → 板模型可信
        double plate_model_exit_omega = 2.7;    // |ω| 高于此值 → 只信整车模型
        bool mahalanobis_matching = false;
        double association_chi2 = 9.488;   // χ²(4, 95%)
        std::array<double, 4> armor_y_offsets{{0.0, 0.0, 0.0, 0.0}};

        LocalFitConfig local_fit;
    };

    class Tracker
    {
    public:
        explicit Tracker(const TrackerConfig& cfg);
        ~Tracker();

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
        void updateChassisFromFit(double radius, double armor_pitch_rad, double height_up);
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
        double getOmega() const {
            if (plate_model_weight_ <= 0.0) return ekf_.getOmega();
            return (1.0 - plate_model_weight_) * ekf_.getOmega() +
                plate_model_weight_ * plate_ekf_.getOmega();
        }
        double getPlateModelOmega() const {
            return plate_ekf_.isInit() ? plate_ekf_.getOmega() : ekf_.getOmega();
        }
        bool hasPlateModel() const { return plate_ekf_.isInit(); }
        bool plateModelActive() const { return plate_model_weight_ > 0.5; }
        double plateModelWeight() const { return plate_model_weight_; }
        double getRotationRateOmega() const {
            return rotation_rate_estimator_.omega();
        }
        double getEkfOmega() const { return ekf_.getOmega(); }
        double ekfCovarianceDiag(int i) const { return ekf_.covarianceDiag(i); }
        bool hasRotationRate() const { return rotation_rate_estimator_.valid(); }
        double getArmorRadius() const { return cfg_.armor_radius; }
        void setArmorRadius(double radius)
        {
            if (!(radius > 0.05 && radius < 1.0)) return;
            cfg_.armor_radius = radius;
            ekf_.setArmorRadius(radius);
            for (auto& d : geometry_.plate_distance) d = radius;
            ekf_.setPlateDistance(geometry_.plate_distance);
            if (std::isfinite(radius)) armor_radius_online_ = radius;
        }
        bool armorRadiusOnline() const { return armor_radius_online_ > 0.0; }
        double getLastMeasurementWeight() const { return last_measurement_weight_; }
        /// @brief "构造板一定最近"硬约束把 yaw 翻回来的累计次数（诊断用）。
        int nearestPlateFlips() const { return nearest_plate_flips_; }
        /// @brief 最近一次拟合判出的运动状态（诊断用）。
        ChassisMotion lastMotion() const { return last_motion_; }
        /// @brief 局部拟合器运行统计（诊断：实车"R=(cfg)/motion 一直 0"就是它没出过有效解）。
        int localFitAttempts() const { return local_fit_attempts_; }
        int localFitValids() const { return local_fit_valids_; }
        double localFitLastRmse() const { return local_fit_last_rmse_; }
        double localFitLastR() const { return local_fit_last_R_; }
        int localFitLastBars() const { return local_fit_last_bars_; }
        /// 推给拟合器的灯条总数 / 拟合结果真正写进几何的次数（诊断用）。
        int localFitPushed() const { return local_fit_pushed_; }
        int localFitApplied() const { return local_fit_applied_; }
        bool fitRadiusEnabled() const { return cfg_.local_fit.use_fit_radius; }
        bool localFitEnabled() const { return cfg_.local_fit.enabled; }
        int localFitLinks() const { return local_fit_links_; }
        int localFitNewArcs() const { return local_fit_new_arcs_; }
        int localFitObsMax() const { return local_fit_obs_max_; }
        int localFitCandEmpty() const { return local_fit_cand_empty_; }
        int localFitApplyCalls() const { return local_fit_apply_calls_; }
        int localFitApplyNoFit() const { return local_fit_apply_no_fit_; }
        int localFitApplyStale() const { return local_fit_apply_stale_; }
        int localFitApplyGeometry() const { return local_fit_apply_geometry_; }
        /// @brief 相位代表元被判"不可信"、退回模 90° 吸附的累计次数（诊断用）。
        ///        长期为 0 说明 ref_armor_id 一直自洽（可以放心用精确注入）。
        int localFitYawSnaps() const { return local_fit_yaw_snaps_; }
        /// @brief 最近一次"拟合 yaw − EKF yaw"的残差（rad，已做相机系→home 系换算与
        ///        延迟外推）。诊断用：它长期贴在 ±90°/180° 附近 = 参考板号或相位表
        ///        不一致；贴在几十度 = 常值偏置/拟合器自身偏差。
        double localFitYawResidual() const { return local_fit_yaw_residual_; }
        /// @brief 参考板号自洽（|残差| ≤ ref_yaw_gate_deg）的累计次数，与
        ///        localFitYawSnaps() 对照着看就是"精确注入 vs 吸附"的比例。
        int localFitYawExact() const { return local_fit_yaw_exact_; }
        /// @brief yaw 通道被**质量门**拦下的累计次数（拟合 rmse 超限 或 相位跨度不足）。
        ///        与 localFitYawExact() 一起看 = "该喂 yaw 的帧里有多少被门控挡住"。
        ///        实测口径见 estimation/local_fit_gate.hpp。
        int localFitYawGated() const { return local_fit_yaw_gated_; }
        /// @brief ω 通道被质量门拦下的累计次数（实测：放宽深度后 ω 也会害人，见 gate 头文件）。
        int localFitOmegaGated() const { return local_fit_omega_gated_; }
        /// @brief 最近一次 yaw 门控的判定（0=ok 1=bad_rmse 2=bad_span 3=invalid）。
        int localFitYawVerdict() const { return local_fit_yaw_verdict_; }
        /// @brief 最近一次拟合的相位跨度（度）——yaw 可辨识性的直接读数量。
        double localFitPhaseSpanDeg() const { return local_fit_phase_span_deg_; }
        double localFitApplyAgeMs() const { return local_fit_apply_age_ms_; }
        double localFitLastMs() const { return local_fit_last_ms_; }
        double lastViewDeg() const { return last_view_deg_; }
        int localFitArcs() const { return static_cast<int>(local_fit_arcs_.size()); }

        /// @brief 同帧多板联立解 (车心, yaw)（home 系）。返回是否成功（残差 < limit）。
        ///        几何用当前慢层的逐板半径；只解 1 个 yaw 自由度（1D 高斯-牛顿），
        ///        车心由各板反推取平均。板对接近正对/相反（信息矩阵退化）时返回 false。
        bool solveChassisFromPlates(const std::vector<std::pair<int, Vec<3>>>& plates,
                                    Vec<3>& center_out, double& yaw_out,
                                    double& residual_out) const;
        int jointSolveCount() const { return joint_solves_; }
        double jointSolveResidual() const { return joint_solve_residual_; }
        double jointSolveYaw() const { return joint_solve_yaw_; }

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
        void setCameraDistortion(const cv::Mat& dist_coeffs);

        std::array<double, 9> fitterRcv() const;

        using LocalFitterCallback =
            std::function<bool(const std::vector<LightBarObservation>&, LocalFitResult&)>;
        void setLocalFitter(LocalFitterCallback fn);

        void pushArmorLightBars(const Armor& armor, double timestamp, int armor_id);

        int plateIdFor(const Armor& armor) const { return selectPlateId(toWorld(armor)); }

        bool runLocalFitNow();

        void startLocalFitWorker();
        void stopLocalFitWorker();

        std::optional<LocalFitResult> latestLocalFit() const;

        double applyLatestLocalFit();

    private:

        void noteMeasurement(bool accepted);
        /// @brief 看门狗判定；命中则强制重初始化并返回 true。
        bool watchdogTrip(const char* reason);
        /// @brief 状态发散检查（速度/角速度/位置/位置协方差越界）。
        bool stateDiverged() const;

        TrackerConfig cfg_;
        ArmorEKF ekf_;
        /// >0 表示半径已经被在线慢层改过（诊断/状态行用）。
        double armor_radius_online_ = 0.0;
        /// 底盘慢层几何（唯一模型来源；由 config / 拟合器慢层 / 观测慢收敛更新）。
        ChassisGeometry geometry_;
        PlateEKF plate_ekf_;                       // 双模型的装甲板模型
        /// "最近板硬约束"把 yaw 翻回来的次数（诊断：正常应长期为 0，只有翻解时才动）。
        int nearest_plate_flips_ = 0;
        /// 连续多少帧处在"观测板落到远半侧"（两条确认才翻，防单帧异常）。
        int nearest_plate_violation_frames_ = 0;
        /// 最近一次拟合判出的运动状态（诊断/状态行用）。
        ChassisMotion last_motion_ = ChassisMotion::Still;
        /// 局部拟合器统计（诊断用）。
        int local_fit_attempts_ = 0;
        int local_fit_valids_ = 0;
        double local_fit_last_rmse_ = -1.0;
        double local_fit_last_R_ = 0.0;
        int local_fit_last_bars_ = 0;
        int local_fit_pushed_ = 0;
        int local_fit_links_ = 0;
        int local_fit_new_arcs_ = 0;
        int local_fit_obs_max_ = 0;
        int local_fit_cand_empty_ = 0;
        int local_fit_apply_calls_ = 0;
        int local_fit_apply_no_fit_ = 0;
        int local_fit_apply_stale_ = 0;
        int local_fit_apply_geometry_ = 0;
        int local_fit_yaw_snaps_ = 0;
        int local_fit_yaw_exact_ = 0;
        int local_fit_yaw_gated_ = 0;
        int local_fit_omega_gated_ = 0;
        int local_fit_yaw_verdict_ = 0;
        double local_fit_phase_span_deg_ = 0.0;
        double local_fit_yaw_residual_ = 0.0;
        double local_fit_apply_age_ms_ = -1.0;
        double local_fit_last_ms_ = -1.0;
        int joint_solves_ = 0;
        double joint_solve_residual_ = -1.0;
        double joint_solve_yaw_ = 0.0;
        double fit_yaw_bias_ = 0.0;
        bool fit_yaw_bias_valid_ = false;
        int previous_plate_id_ = -1;
        double last_view_deg_ = -1.0;

        double last_push_time_ = 0.0;
        int local_fit_applied_ = 0;
        double plate_model_last_timestamp_ = -1.0;
        double plate_model_weight_ = 0.0;
        Vec<3> plateDerivedCenter() const;
        Vec<3> plateDerivedVelocity() const;
        double plateDerivedYaw() const;
        Vec<3> blendedCenter() const;
        Vec<3> blendedVelocity() const;
        double blendedYawWorld() const;
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
        cv::Mat camera_dist_;            // 镜头畸变系数（可为空 = 无畸变）
        bool camera_dist_valid_ = false;
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

        mutable std::mutex local_fit_mutex_;
        struct LocalFitArc
        {
            std::vector<LightBarObservation> obs;  // 每帧 2 根灯条
            int plate_hint = -1;                   // 种子帧板号（0..3，-1=未知）
            double seed_cx = 0.0, seed_cz = 0.0, seed_t = 0.0;
            double u = 0.0, v = 0.0, t = 0.0;      // 最近一次板心(px)与时间
            double vu = 0.0, vv = 0.0;             // 板心速度(px/s)，用于关联预测
        };
        std::vector<LocalFitArc> local_fit_arcs_;
        std::optional<LocalFitResult> latest_local_fit_;
        LocalFitterCallback local_fitter_;
        std::thread local_fit_thread_;
        std::atomic<bool> local_fit_running_{false};
    };
}

#endif
