#ifndef TRACKER_HPP
#define TRACKER_HPP

#include <array>
#include <deque>
#include <chrono>
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

    /// @brief 局部拟合器（小陀螺 θ/ω 拟合）在 Tracker 内的接入配置。
    ///        默认**关闭**（与 spin_geometry_estimator 同款：可选组件，先只读不接管）。
    struct LocalFitConfig
    {
        bool enabled = false;
        int window_frames = 24;         // 滑窗帧数（覆盖相位 ≥ π/3 才有 ω 可辨识性）
        int min_bars = 8;               // 窗口内至少这么多条灯条观测才拟合
        int period_frames = 10;         // 每多少帧触发一次拟合（低频异步）
        double max_result_age = 0.20;   // s：拟合结果比当前帧老过它就外推/丢弃
        int max_window_gap_frames = 6;  // 连续丢帧超过它就清空窗口（避免跨运动段）
        // ---- A 段（单板正面弧切分）------------------------------------------
        // 正对门控：图像两灯条中心距 sep 与"该深度的最大值" sep_max=W·fx/depth 之比
        // = cos(板相对相机偏角)。只累计 sep > front_gate·sep_max 的帧（默认 0.5 →
        // |偏角|<60°），离开即清段——保证图像左右=板面切向、且相位跨度可控。
        double front_gate = 0.5;
        double assoc_gate_px = 45.0;    // 帧间板心关联门限（px），配合常速预测
        double min_depth = 1.5;         // 参与局部拟合的板深下限（m，滤掉近处杂检/道具）
        double max_depth = 9.0;         // 上限（m）
        // 多板联合（方向2 P0）：把**同时可见的多块板**堆进同一次拟合——共享 C/R/h、
        // 相位相差固定角（由 plate_yaw_offsets 给）。打破单板 R–Cz 耦合、Yaw 更准。
        bool joint_plates = false;
        int joint_max_arcs = 4;         // 联合时最多用几条板弧
        double yaw_noise = 0.02;        // 拟合 yaw 的观测噪声（rad²，质量再缩放）
        double omega_noise = 0.05;      // 拟合 ω 的观测噪声
        // 是否把拟合的 yaw 也当观测喂 EKF。实测 yaw 是最脆弱的量（板号/相位约定稍错就把
        // EKF 拉坏，A/B 里 6.5m/5rad 误差从 0.67 炸到 13.9）；ω/中心才是稳健可用的输出。
        // 默认只喂 ω（yaw 观测关）。
        bool use_yaw = false;
        // 相机系→整车系符号（标定项）：拟合器的 θ 约定（相机 X-Z 平面，从 +Z 向 +X 量）
        // 与 EKF 的 ω/yaw 约定相反 → 仿真实测 -1（gyro5 A/B：+1 会让 EKF ω 更差，-1 明显更稳）。
        double yaw_rate_sign = -1.0;
        bool also_update_center = false; // 是否把拟合中心也当观测（默认否，先只给 yaw/ω）
    };

    struct TrackerConfig {
        // ---- 收敛看门狗（对齐 sp_vision 的 recent_nis_failures + diverged 思路）----
        // 判据一：最近 window 帧（只统计"有观测"的帧）里，量测被拒的比例 ≥ reject_ratio；
        // 判据二：状态本身发散（速度/角速度/位置/位置协方差越界）。
        // 判据三：帧间隔超过 max_frame_gap（相机离线 / 跨主机时间戳跳变，对齐
        //         sp_vision tracker.cpp 里的 `dt > 0.1s → lost`）。
        // 命中任一 → 立即 LOST + **强制**下一帧用观测重新初始化（跳过用旧状态的 tryReacquire）。
        bool watchdog_enabled = true;
        int watchdog_window = 20;
        double watchdog_reject_ratio = 0.4;
        int watchdog_min_samples = 8;
        double watchdog_max_speed = 12.0;          // m/s
        double watchdog_max_yaw_rate = 20.0;       // rad/s
        double watchdog_max_range = 30.0;          // 位置模长 m
        double watchdog_max_position_cov = 25.0;   // P 的位置对角 m²
        double watchdog_max_frame_gap = 0.20;      // 帧间隔上限 s
        /// @brief "已收敛"需要连续被接受的量测条数（对齐 sp_vision 的 update_count_ > 3）。
        int watchdog_min_converge_updates = 3;

        /// @brief 把识别到的装甲板当**硬约束**：每次更新后把底盘中心平移，使模型算出的
        ///        该块板精确落在观测位置上（只动中心 3 自由度）。单板工况下这能消掉
        ///        "模型稳定偏出板心十几厘米"的稳态偏差。
        bool hard_plate_constraint = true;

        /// @brief 板号锁定：已关联的板只要代价不超过最优的该倍数，就继续沿用，避免在
        ///        相邻/对角板之间跳（实测视轴偏差一帧跳 ±12~16°，闭环追跳变不收敛）。
        ///        1.0 = 关闭锁定。
        double plate_lock_margin = 1.35;

        int min_detect_frames = 3;
        int max_lost_frames = 15;
        int reacquire_frames = 3;
        double max_match_distance = 0.35;
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
        // 等效板面尺寸。**必须与探测器的点语义匹配**，而不是"标准装甲板
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
        // ---- 物理量纲的距离噪声（上科大 2026 §5B.2）--------------------------
        // PnP 的深度不确定度近似与距离成正比：σ_dist = ypd_distance_sigma_coeff · d。
        // 文档给的系数范围 0.05~0.20（远距小靶面时取大值）。>0 时覆盖上面那套
        // `measure_noise·(1+scale·d²)` 的手调模型；0 = 用旧模型。
        // 为什么重要：马氏关联的门限就是卡方门限，R 低估 → 真板被当异常值拒掉
        // （实测 6.5 m 用旧模型只接受 5 条观测，用 0.2·d 后恢复正常）。
        double ypd_distance_sigma_coeff = 0.0;
        double ypd_nis_threshold = 9.488;
        // 关联的板角误差上限：相邻板相差 90°，所以 20°(0.35 rad) 足够严格地把
        // "另一块物理板"的观测挡掉（实测它会带来 ±12~16° 的视轴跳变）。
        double max_angle_error = 0.35;
        // ---- 高速自旋（小陀螺 ≥5 rad/s）通用的关联门限 ----------------------
        // 固定角门限在高速下必然失配：ω 估计还没收敛时，预测 yaw 每帧落后
        // ω·dt（5 rad/s、24 fps 就是 12°/帧），几帧就顶穿 20° → 关联
        // 返回 inf → 观测被拒 → ω 永远学不到（正反馈，看门狗反复重初始化）。
        // 通用做法：门限跟着**滤波器自身的 yaw 标准差**放大，收敛后自动收窄：
        //   gate = min(association_gate_max,
        //              max_angle_error + association_gate_sigma · σ_yaw)
        // 上限必须 < π/2，否则相邻物理板也会落进同一门限 → 误关联。
        bool adaptive_association_gate = true;
        double association_gate_sigma = 1.5;
        double association_gate_max = 1.30;   // rad ≈ 74.5°，必须 < 90°
        // 距离门限同理：PnP 的距离噪声随距离平方增长（configs 的
        // range_noise_distance_scale 就是在建模它）。固定 0.35 m 在 6.5 m 处
        // （1 px ≈ 0.5 m）会把**真板**也判成 nomatch。这里按同一套噪声模型放大：
        //   range_gate = max_match_distance
        //              + range_gate_sigma · sqrt(meas_noise·(1+scale·d²))，封顶。
        // 角度门限仍是主判据（< 90°），所以放宽距离不会放进相邻板。
        double range_gate_sigma = 2.0;
        double range_gate_max = 1.00;         // m
        double angle_match_weight = 0.08;
        double position_nis_threshold = 7.815;
        double angle_nis_threshold = 3.841;
        // ---- NIS：诊断量而不是硬门限（对齐 sp_vision 的 recent_nis_failures）----
        // 硬门限在高速下有正反馈：ω 没学到 → 残差大 → 拒更新 → ω 永远学不到。
        // 打开后 YPD/中心/角度更新始终应用（靠关联门限 + 尺度闸门挡假观测），
        // 只把 NIS 留给诊断；UV 路径仍保留硬门限（它用来挡矩形板的镜像二义）。
        bool soft_nis_gate = true;
        // ---- 双模型（上科大 2026 §9.4.4）--------------------------------------
        // 除整车 9 维 EKF 外，并行维护一个**装甲板模型**（直接跟踪可见板的中心+朝向）。
        // 低速时板模型少一层"半径/切向不可观测"的约束，静止漂移更小；高速时切回整车
        // 模型。第一阶段只并行运行、不参与输出（默认关），后面再接"按 |ω| 切换输出"。
        bool plate_model_enabled = false;
        double plate_model_process_acc = 20.0;
        double plate_model_process_yaw_acc = 50.0;
        double plate_model_enter_omega = 0.7;   // |ω| 低于此值 → 板模型可信
        double plate_model_exit_omega = 2.7;    // |ω| 高于此值 → 只信整车模型
        // ---- 马氏距离关联（上科大 2026 §5B.4）--------------------------------
        // 用 d² = rᵀ·S⁻¹·r（S = H·P·Hᵀ + R）做板号匹配，门限取卡方 χ²(4,95%)。
        // 它替代"手写代价 + 固定/自适应角度门限"：门限由滤波器自身的 P 和逐维度
        // 噪声自动标定，初始化/丢帧时自动放宽、收敛后自动收紧，也不需要按 ω 手调。
        // 关掉 = 回到 adaptive_association_gate 那套启发式（A/B 用）。
        bool mahalanobis_matching = false;
        double association_chi2 = 9.488;   // χ²(4, 95%)
        // Per-armor vertical center offsets. They remain zero for a symmetric
        // real robot model and can be calibrated for a simulator vehicle.
        std::array<double, 4> armor_y_offsets{{0.0, 0.0, 0.0, 0.0}};

        // 局部拟合器接入（默认关；见 docs/local_trajectory_fitter.md）
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
        /// @brief 最后一次**被接受**的观测的时间戳（只有真正收到并接受了框才会更新，
        ///        单纯的 predict/coast 不算）。用途："只认最新" —— 异步推理下 tracker 会在
        ///        没有新框的帧上保留上一帧的 armors（hasObservation 仍为真），拿它算图像
        ///        误差等于用旧数据一直推云台。注意不能用 last_timestamp_：那个在 predict()
        ///        里也会更新。
        double lastObservationTime() const { return last_observation_time_; }
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
        double getOmega() const {
            if (plate_model_weight_ <= 0.0) return ekf_.getOmega();
            return (1.0 - plate_model_weight_) * ekf_.getOmega() +
                plate_model_weight_ * plate_ekf_.getOmega();
        }
        /// @brief 板模型的角速度/中心（双模型用；未启用时返回整车模型的值）。
        double getPlateModelOmega() const {
            return plate_ekf_.isInit() ? plate_ekf_.getOmega() : ekf_.getOmega();
        }
        bool hasPlateModel() const { return plate_ekf_.isInit(); }
        /// @brief 当前是否由板模型输出（按 |ω| 带滞回切换；见 TrackerConfig）。
        bool plateModelActive() const { return plate_model_weight_ > 0.5; }
        /// @brief 板模型权重 0~1（1=只用板模型，0=只用整车模型；中间=文档 §9.4.4 的
        ///        BOTH 档融合）。
        double plateModelWeight() const { return plate_model_weight_; }
        double getRotationRateOmega() const {
            return rotation_rate_estimator_.omega();
        }
        double getEkfOmega() const { return ekf_.getOmega(); }
        bool hasRotationRate() const { return rotation_rate_estimator_.valid(); }
        double getArmorRadius() const { return cfg_.armor_radius; }
        double getLastMeasurementWeight() const { return last_measurement_weight_; }

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

        // The EKF state is maintained in the world frame. Set the timestamped
        // camera pose used by the next observation before update().
        void setCameraPose(const CameraPose& pose);
        void setCameraAngles(double yaw, double pitch);

        /// @brief UV 观测用的相机内参（帧尺寸变化时由上层重新调用）。
        ///        只有 cfg.uv_observation 打开时才需要；world→camera 的旋转仍由
        ///        setCameraPose() 提供，两处必须同一时刻。
        void setCameraIntrinsics(double fx, double fy, double cx, double cy);
        bool hasCameraIntrinsics() const { return camera_intrinsics_valid_; }
        /// @brief 镜头畸变系数（实车必填）。拟合器是**纯针孔重投影**，喂进去的灯条端点
        ///        必须先去畸变，否则实车强畸变下重投影会有系统性偏差（离光心越远越大）。
        void setCameraDistortion(const cv::Mat& dist_coeffs);

        // ---- 局部拟合器接入（小陀螺 θ/ω；默认关闭）--------------------------------
        // 数据流（提示词 §十一）：每帧把灯条端点推进缓冲 → 低频异步拟合 →
        // LocalFitResult → applyLatestLocalFit() 用强 yaw/ω 观测更新 EKF。
        // 拟合算法由外部注入（app 层传入 Ceres 版 LocalTrajectoryFitter），
        // 这样 auto_aim_core 不引入 Ceres 依赖（与 spin_geometry 同款分层）。
        using LocalFitterCallback =
            std::function<bool(const std::vector<LightBarObservation>&, LocalFitResult&)>;
        void setLocalFitter(LocalFitterCallback fn);
        /// @brief 把一块装甲板的两根灯条端点推入拟合缓冲（每帧调用）。
        ///        timestamp 用**曝光时刻**（观测时间），保证 θ(t) 的时间基正确。
        ///        armor_id 必须是**板号 0..3**（不是网络类别 id）——用 plateIdFor() 取。
        void pushArmorLightBars(const Armor& armor, double timestamp, int armor_id);
        /// @brief 由几何把一块观测装甲板映射到板号 0..3（-1 = 不确定）。
        int plateIdFor(const Armor& armor) const { return selectPlateId(armor); }
        /// @brief 同步跑一次拟合（把当前缓冲喂给回调）；返回是否产出有效结果。
        bool runLocalFitNow();
        /// @brief 启动/停止内部异步拟合线程（低频）。
        void startLocalFitWorker();
        void stopLocalFitWorker();
        /// @brief 取最近一次拟合结果（线程安全）。
        std::optional<LocalFitResult> latestLocalFit() const;
        /// @brief 用最近的拟合结果更新 EKF（相机系→EKF 系换算后）。返回有效权重。
        double applyLatestLocalFit();

    private:
        /// @brief 记一帧的量测结果（true=被接受）。只统计"确实有观测"的帧。
        void noteMeasurement(bool accepted);
        /// @brief 看门狗判定；命中则强制重初始化并返回 true。
        bool watchdogTrip(const char* reason);
        /// @brief 状态发散检查（速度/角速度/位置/位置协方差越界）。
        bool stateDiverged() const;

        TrackerConfig cfg_;
        ArmorEKF ekf_;
        PlateEKF plate_ekf_;                       // 双模型的装甲板模型
        double plate_model_last_timestamp_ = -1.0;
        double plate_model_weight_ = 0.0;
        /// @brief 由板模型反推的整车状态（板心 − R·径向；yaw = 板朝向 − 板号相位）。
        ///        这样 selector/ROI 依赖的"中心 + yaw + 半径"几何依然成立。
        Vec<3> plateDerivedCenter() const;
        Vec<3> plateDerivedVelocity() const;
        double plateDerivedYaw() const;
        Vec<3> blendedCenter() const;
        Vec<3> blendedVelocity() const;
        double blendedYawWorld() const;
        std::deque<bool> recent_measurements_;   // 最近若干帧的量测接受情况
        int watchdog_resets_ = 0;
        // 自上次（重新）初始化以来被接受的量测条数：`converged` 用它（对齐
        // sp_vision 的 `update_count_ > 3`）。**不依赖当前是不是 TRACKING**：
        // 用 state 判会在每次 TEMP_LOST 翻一次，反而把提前量开关抖成震荡。
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
        /// @brief 把**世界系**的方向向量转到**相机系**。
        ///        画底盘模型时必须用这个把"切向/竖直"和"板心位置"放到同一个坐标系里：
        ///        板心来自 getEstimatedArmorPositions()（相机系），而切向/竖直是在世界系里
        ///        定义的（yaw + 板角位置、竖直 = 世界 ±y）。混用两系会让画出来的板被掰一个
        ///        ≈相机-板 pitch 夹角（"模型像在斜坡上"）——这个坑实际踩过。
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
        void predictStateTo(double timestamp);
        bool tryReacquire(const std::vector<const Armor*>& observations,
                          double timestamp);
        int selectPlateId(const Armor& armor) const;
        double distanceToPlate(const cv::Mat& tvec, int plate_id) const;
        double matchCost(const Armor& armor, int plate_id) const;
        void updateRotationRate(const Armor& armor, double timestamp);

        // ---- 局部拟合器状态 ------------------------------------------------------
        mutable std::mutex local_fit_mutex_;
        // A 段：一条"单块板正面弧"的缓冲。按**几何航迹**（图像板心最近邻+常速预测）
        // 关联，不用 plate_id 做身份——同一帧里可能有多块板被映到同一个 id，用 id
        // 分桶会把不同物理板的弧混在一起。plate_hint 只用于取拟合的 yaw 偏移。
        struct LocalFitArc
        {
            std::vector<LightBarObservation> obs;  // 每帧 2 根灯条
            int plate_hint = -1;                   // 种子帧板号（0..3，-1=未知）
            // 两段式相对身份：种子帧 PnP 板心 (x,z) 与时间。先用最长弧拟合出旋转中心 C，
            // 再用"板心相对 C 的方位角"定各弧的相位偏移（用位置量，避开病态的单帧 PnP yaw）。
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
