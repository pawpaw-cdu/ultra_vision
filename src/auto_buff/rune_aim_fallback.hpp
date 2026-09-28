#ifndef AUTO_AIM_ENERGY_RUNE_AIM_FALLBACK_HPP
#define AUTO_AIM_ENERGY_RUNE_AIM_FALLBACK_HPP

// 没有可用目标时云台该怎么办（停靠 / 回中策略）。
//
// 现场观察：丢掉目标就回标定位姿是最糟的选择 —— RuneTarget::reset() 会把相位
// 与几何全丢掉，重新捕获要 0.5~1 s，而这段时间本来可以用来打下一片。所以策略是
// **分级**的（逐条都留在数据里，见 docs/energy_rune_issue_audit.md）：
//   1) 还有"最后已知符心"且不旧于 park_on_center_max_age_s →
//      把相机停在那个方向，估计器继续滑行，网络一报出扇叶就能立刻接上；
//   2) 符心也过期了，且连续没有可用目标超过 recenter_after_frames 帧 →
//      才回标定位姿 + 复位估计器（最后的兜底）。
//
// 这里只做**判决**，不动估计器也不发指令：调用方（帧循环）拿到 Decision 之后
// 自己去 reset 估计器 / 发布角度。这样策略可以单独喂序列测试。

namespace auto_aim::energy
{
    class RuneAimFallback
    {
    public:
        struct Config
        {
            double park_on_center_max_age_s = 1.5;  // 符心最多能用多久
            int recenter_after_frames = 15;         // 连续多少帧无目标才回中
            double home_epsilon_rad = 1e-3;         // 已经接近标定位姿就不必再回中
        };

        struct Decision
        {
            enum class Kind
            {
                None,
                AimLastCenter,   // 停在最后已知符心
                Recenter,        // 回标定位姿
            };

            Kind kind = Kind::None;
            /// 回中分支被触发：调用方应复位估计器（原实现里 `target.reset()`
            /// 在回中前执行，与是否真的发了回中指令无关）。
            bool reset_estimator = false;
            double yaw = 0.0;
            double pitch = 0.0;
        };

        RuneAimFallback() = default;
        explicit RuneAimFallback(const Config& config) : config_(config) {}

        /// @brief 解算出一次符心（帧循环每次拿到解算结果时调用）。
        void noteCenter(double yaw, double pitch, double now);

        /// @brief 有可用目标（帧循环发出控制目标）时调用：清掉无目标计数。
        void noteControlAvailable() { frames_without_control_ = 0; }

        /// @brief 本帧没有可用目标：给出停靠/回中判决。
        Decision decide(double now, double commanded_yaw, double commanded_pitch);

        int parkedFrames() const { return parked_frames_; }
        int recenterCount() const { return recenter_count_; }
        bool hasCenter() const { return have_last_center_; }
        double lastCenterYaw() const { return last_center_yaw_; }
        double lastCenterPitch() const { return last_center_pitch_; }
        int framesWithoutControl() const { return frames_without_control_; }

        void reset();

    private:
        Config config_;
        int frames_without_control_ = 0;
        int parked_frames_ = 0;
        int recenter_count_ = 0;
        bool have_last_center_ = false;
        double last_center_time_ = 0.0;
        double last_center_yaw_ = 0.0;
        double last_center_pitch_ = 0.0;
    };
} // namespace auto_aim::energy

#endif // AUTO_AIM_ENERGY_RUNE_AIM_FALLBACK_HPP
