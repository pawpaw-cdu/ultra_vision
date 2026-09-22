#ifndef AUTO_AIM_TARGET_SELECTOR_HPP
#define AUTO_AIM_TARGET_SELECTOR_HPP

#include <array>

namespace auto_aim
{
    struct TargetSelectorConfig {
        bool enabled = true;
        double projectile_speed = 25.0;
        double gravity = 9.81;
        double command_latency = 0.08;
        double max_lead_time = 0.50;
        double spin_omega_threshold = 2.0;

        // ---- 按估计转速分档的策略（2026-10-05）--------------------------------
        // **注意（2026-10-05 更正）**：早先"|ω|>云台最大角速度就追不上"的推理是
        // 错的。真值实测（5 rad/s）：四块板相对**相机**的方位角都挤在底盘中心方位
        // ±4°以内（median 3.4°），单块板的方位角变化率中位只有 0.21 rad/s —— 板是绕
        // 中心做深度/侧向圆周运动，方位角几乎不动，云台 172°/s 根本不是瓶颈。所以
        // 高速下的抖动是**决策层（择板/开火窗）在 ω 高时高频翻转**造成的，不是执行机构。
        // 分档用于工程兜底：
        //   |ω| < motion_gate_enter_omega  → 静止档（提前量/前馈清零，现有逻辑）
        //   ≤ chase_omega_threshold         → 追板档（现有逻辑，拦截迭代已含云台时间）
        //   >  chase_omega_threshold         → 守角档（瞄预测的底盘中心）——**极端工况的
        //     退化保证**，不是最终策略；真正的修法是让决策层在高速下不抖。
        double chase_omega_threshold = 3.0;   // rad/s（经验值，见上）
        bool spin_hold_enabled = true;        // 守角档开关（A/B 用）
        /// @brief 守角档开火窗：预测弹丸到达时刻，最近的板与守角（底盘中心方位）的
        ///        夹角上限。近侧板与中心方位天然只差 asin(R/d)≈2~3°，所以 6° 足够，
        ///        又不会把"板已转到侧面/背面"的帧放进来。
        double spin_fire_tolerance = 0.105;   // rad ≈ 6°
        double coming_angle = 1.0471975511965976;  // 60 degrees
        double leaving_angle = 0.3490658503988659; // 20 degrees
        double yaw_velocity = 6.283185307179586;   // rad/s
        double pitch_velocity = 3.141592653589793; // rad/s
        double yaw_acceleration = 31.41592653589793;   // rad/s^2
        double pitch_acceleration = 20.94395102393195; // rad/s^2
        double handoff_start_angle = 0.17453292519943295; // 10 degrees
        double handoff_duration = 0.20; // s

        // Latency is not a single number: a fast-spinning chassis is tracked
        // from staler observations and the gimbal has been slewing, so its lead
        // needs a larger allowance than a slow target. Mirrors the reference
        // implementation's decision_speed / high_speed_delay_time split.
        double decide_speed = 7.0;          // rad/s
        double low_speed_latency = 0.06;    // s
        double high_speed_latency = 0.13;   // s

        // Plate locking: keep the chosen plate while it stays eligible unless
        // another candidate is clearly cheaper. Stops the selector from
        // alternating between two plates that sit at similar angles.
        double lock_switch_margin = 0.65;

        // ---- 静止靶闸门（2026-09-29）--------------------------------------------
        // 为什么需要它：EKF 的**速度/角速度状态在静止靶上并不为 0**。实车静止靶
        // 实测（`tools/replay_dataset --dump`，2.2 m、48 fps）：|v| 均值 88 mm/s /
        // p90 227 mm/s / max 432 mm/s，|ω| p50 0.95°/s / p99 94°/s，而**观测**本身
        // 只抖 0.5 mm（x）/ 6.5 mm（深度）/帧。这些噪声原本会通过
        //   ① 提前量（|v|×386 ms ≈ 35~150 mm → 1~4°）
        //   ② 直接喂给云台的角速度前馈（|ω_yaw| 均值 0.59°/s、max 24°/s）
        // 变成云台可见的抖动 —— 静止靶上本应两者都为 0。
        //
        // 为什么用**瞬时阈值的滞回**而不是"连续 N 帧才算动"：实测静止靶的 |v| 噪声
        // 是**慢漂**（>0.12 m/s 的最长连续段 238 帧），持续判据根本区分不出来；
        // 但噪声**上界**很干净：|v| 从未超过 0.43 m/s、|ω| 的 p99 之外才有大值。
        // 所以按"噪声够不到、真实运动够得着"取阈值：
        //   真实平移 ≥0.5~1 m/s、真实小陀螺 ≥0.6 rad/s（34°/s）→ 立刻按"在动"处理；
        //   |v|<0.3 m/s 且 |ω|<0.35 rad/s → 按"静止"处理（提前量/前馈清零）。
        // 代价：0.3~0.5 m/s 的**慢速平移**会被当成静止（提前量少 0.12~0.19 m·s
        // 的一小段），换来的是静止靶上云台真正不动 —— 见 docs/auto_aim_aim_loop.md。
        // 阈值都是**量出来的**（改完 P_ω/阶跃保护之后再测）：静止靶上 |v| 的
        // p90=200 mm/s、p99=340 mm/s，|ω| 的 p90=3.7°/s、p99=33°/s、max=57°/s。
        // 提前量按 386 ms 算，ω 的噪声尾巴（20~57°/s）会让瞄点相位错 7~22°
        // → 4.8° 的瞄点甩动（实测）。所以 ω 的"在动"门限取 1.0 rad/s（57°/s）：
        // 正好把这条尾巴关掉，而 RM 小陀螺常态是 1.5~3 rad/s（86~172°/s）不受影响。
        // 代价：0.4~1.0 rad/s（23~57°/s）的慢速自转不做提前量，相位少算
        // r·ω·0.386 ≈ 36~96 mm（2.2 m 处 0.9~2.5°）。
        bool motion_gate_enabled = true;
        double motion_gate_enter_speed = 0.50;   // m/s，超过 → 立刻按"在动"处理
        double motion_gate_exit_speed = 0.30;    // m/s，低于 → 允许回到"静止"
        double motion_gate_enter_omega = 1.00;   // rad/s ≈ 57°/s
        double motion_gate_exit_omega = 0.50;    // rad/s ≈ 29°/s
    };

    // Pure estimator output consumed by the selector. All values are in the
    // world frame so the selector is independent of camera rotation.
    struct TargetEstimate {
        std::array<double, 3> center{{0.0, 0.0, 0.0}};
        std::array<double, 3> velocity{{0.0, 0.0, 0.0}};
        double yaw = 0.0;
        double omega = 0.0;
        double armor_radius = 0.21;
        // Seconds between the observation that produced this state and the
        // moment the decision is made. An asynchronous detector can be a couple
        // of hundred milliseconds behind, which at 3 rad/s is tens of degrees,
        // so the lead time has to account for it.
        double age = 0.0;
        // 滤波器是否已经收敛（`Tracker::watchdogStats().converged`：稳定跟踪 + 最近
        // 连续若干帧量测都被接受 + 状态未发散）。**未收敛时不允许用 ω 做提前量**：
        // 实测刚重捕的 1~3 帧里 ω 会冲到 12~120°/s（`initAt` 的 P_ω=100 让滤波器
        // 一帧就把 ω 拉到估计值上，而该估计此刻还不代表真实转速），乘上 386 ms 的
        // 提前量就是 5~8° 的瞄点甩动，随后又弹回来 —— 云台上就是"抽一下"。
        // sp_vision 的 `Target::convergened()` 正是这个用途（收敛前不参与决策）。
        bool converged = true;
    };

    // The selector's only output. The gimbal consumes `aim_point`; the
    // visualizer consumes `armor_position` and `predicted_yaw`; the firing
    // decision consumes `in_fire_window` plus gimbal readiness.
    struct TargetDecision {
        bool valid = false;
        int armor_id = -1;
        std::array<double, 3> armor_position{{0.0, 0.0, 0.0}};
        std::array<double, 3> aim_point{{0.0, 0.0, 0.0}};
        double predicted_yaw = 0.0;
        double target_yaw = 0.0;
        double target_pitch = 0.0;
        double target_yaw_velocity = 0.0;
        double target_pitch_velocity = 0.0;
        double distance = 0.0;
        double flight_time = 0.0;
        double aim_time = 0.0;
        double gimbal_time = 0.0;
        double lead_time = 0.0;
        double handoff_gain = 1.0;
        double handoff_alpha = 0.0;
        double time_to_handoff = 0.0;
        int next_armor_id = -1;
        bool in_fire_window = false;
    };

    class TargetSelector
    {
    public:
        explicit TargetSelector(const TargetSelectorConfig& config = {});

        TargetDecision select(const TargetEstimate& estimate,
                              double current_yaw,
                              double current_pitch);

        /// @brief 实车用：弹速由下位机每帧回传（TJ_T 的 bullet_speed），不能写死。
        ///        只接受合理值，异常值忽略（保留上一个有效弹速）。
        void setProjectileSpeed(double speed)
        {
            if (speed >= 10.0 && speed <= 45.0) config_.projectile_speed = speed;
        }
        double projectileSpeed() const { return config_.projectile_speed; }
        /// @brief 目标当前是否被判定为"静止"（静止时提前量与前馈都按 0 处理）。
        bool targetStill() const { return still_; }

    private:
        double armorDelta(const TargetEstimate& estimate,
                          int armor_id,
                          double lead_time) const;

        TargetSelectorConfig config_;
        // Plate chosen in the previous frame, or -1 when nothing was eligible.
        int lock_id_ = -1;
        // 静止靶闸门的"持续运动"计分（见 TargetSelectorConfig::motion_gate_*）。
        bool still_ = true;   // 上一次判定：目标静止（提前量/前馈已清零）
    };
}

#endif
