#ifndef AUTO_AIM_ENERGY_RUNE_AIMER_HPP
#define AUTO_AIM_ENERGY_RUNE_AIMER_HPP

// Energy-rune aimer: intercept the lit plate and decide when to fire.
//
// Ported from sp_vision_25 `tasks/auto_buff/buff_aimer.{hpp,cpp}` (the plain
// ballistic path, not the MPC variant). The reference predicts the plate for
// the detection latency plus the projectile flight time, then solves the
// ballistic pitch for the resulting intercept point.
//
// Output convention is the Ultra_Vision / simulator one: absolute gimbal
// angles with positive yaw turning right and positive pitch turning up. The
// reference works with positive pitch upwards as well, but the opposite yaw
// sign because its world frame is left-handed for yaw; `toGimbal()`
// centralises that conversion.

#include <Eigen/Dense>

namespace auto_aim::energy
{
    struct RuneAimerConfig
    {
        double yaw_offset_deg = 0.0;
        double pitch_offset_deg = 0.0;
        double fire_gap_time = 0.7;   // minimum time between two shots
        double predict_time = 0.12;   // extra lead on top of the detection age
        double bullet_speed = 25.0;
        double gravity = 9.81;
        double target_radius_m = 0.700;
        double switch_angle_deg = 5.0; // blade switch detection threshold
        int max_mistakes = 3;          // forced re-acquisition after this many
        // The two ballistic passes must agree on the flight time. The reference
        // used 0.01 s, which is too strict once the range grows (0.33 s of
        // flight at 8 m).
        double max_flight_time_error = 0.03;
        // 参考 sp_vision planner：延迟补偿按目标转速分档，而不是固定值。
        double decision_speed_rad_s = 1.6;   // ~90 deg/s
        double low_speed_delay = 0.06;
        double high_speed_delay = 0.13;
        // 观测寿命（参考实现 `temp.data_life = 0.2 s`）：超过这段时间没有新的
        // 解算结果时，不再外推扇叶位置，改为瞄圆心——圆心运动慢，云台会稳稳
        // 停在符上，而不是按 60°/s 一路把相机甩出画面（实测会甩 40°+ 再归位）。
        double observation_life_s = 0.2;
        // 时间基准：true=用仿真曝光时刻、并在推理后采样（提前量含全部延迟）；
        // false=用本地到达时刻、推理前采样（旧行为，供 A/B 对比）。
        bool frame_time_from_exposure = true;
        // 是否对"帧到瞄准延迟"做 EMA 平滑后再用于提前量（配合上面那项使用）。
        bool smooth_lead_latency = true;
        // 换叶判定的"角度兜底"确认帧数：连续这么多帧角度跳变超过 switch_angle_deg
        // 才判换叶。单帧角度跳变不再关控制（大符真实靶心运动只有 ~0.4°/帧，
        // 而抖动可达 4~16°/帧，用单帧阈值会把抖动误判成换叶）。
        int switch_confirm_frames = 2;
        // 输出角斜率限制（度/次调用）：控制线程 100 Hz 调用 ⇒ 60°/s ≈ 0.6°/call。
        // 正常跟踪靶心只要 ~13°/s，不受影响；"丢失→重捕获"那种 8~15° 的瞬间回弹
        // 会被摊成 0.15~0.25 s 的平滑过渡（实测回弹最大 15.5°、≤0.1 s 内完成，
        // 是云台"乱飘"最明显的一部分）。<=0 关闭。
        double max_aim_step_deg = 0.6;
    };

    struct RuneCommand
    {
        bool control = false;
        bool shoot = false;
        double yaw = 0.0;
        double pitch = 0.0;
        // 靶心角速度（rad/s），前馈给云台轨迹生成器用，避免纯角度跟踪的滞后与顿挫。
        double yaw_velocity = 0.0;
        double pitch_velocity = 0.0;
        double fly_time = 0.0;
        double aim_roll = 0.0;     // predicted rune roll, for visualization
        bool blade_switched = false;
    };

    class RuneTarget;

    class RuneAimer
    {
    public:
        explicit RuneAimer(const RuneAimerConfig& config);

        // `timestamp` is the capture time of the frame the estimate came from,
        // `now` is the current time; both in seconds on the same clock.
        // `detection_age_override` >= 0 时用它代替 `now - timestamp` 作为提前量：
        // 调用方可以传"平滑后的帧到瞄准延迟"，既保留延迟补偿、又不引入处理耗时抖动。
        RuneCommand aim(RuneTarget& target, double timestamp, double now,
                        double detection_age_override = -1.0, int slot_offset = 0);
        // 观测过期时的保底瞄准：瞄圆心，不外推扇叶位置，也不开火。
        RuneCommand aimCenter(RuneTarget& target, double timestamp, double now,
                              double detection_age_override = -1.0);

        double bulletSpeed() const { return config_.bullet_speed; }
        /// @brief 状态估计器报告的"本帧换叶"（观测二次确认后接管）。火控以此为准，
        ///        角度阈值只作兜底。
        void setObservedBladeSwitch(bool value) { observed_switch_ = value; }

    private:
        RuneCommand aimImpl(RuneTarget& target, double timestamp, double now, bool center_only,
                            double detection_age_override, double phase_offset);
        bool sendAngles(RuneTarget& target, double predict_time, double bullet_speed,
                        const Eigen::Vector3d& plate_in_buff, double& yaw_world,
                        double& pitch_up, double& fly_time, double phase_offset) const;

        RuneAimerConfig config_;
        double last_yaw_ = 0.0;
        double last_pitch_ = 0.0;
        bool last_angle_valid_ = false;
        int mistake_count_ = 0;
        bool switch_fanblade_ = false;
        bool observed_switch_ = false;
        bool last_fire_initialized_ = false;
        double last_fire_time_ = 0.0;
    };
} // namespace auto_aim::energy

#endif // AUTO_AIM_ENERGY_RUNE_AIMER_HPP
