#ifndef AUTO_AIM_SHOOT_EVALUATOR_HPP
#define AUTO_AIM_SHOOT_EVALUATOR_HPP
// 开火判据（决策②）：**只回答"现在值不值得打"**，不碰状态机/节流/硬件。
// 形状照 rmcs_auto_aim_v2 的 ShootEvaluator（判据独立成类、可单测），
// 容差与判据项参照 sp_vision 的 Shooter（双容差 + 距离判据）。
#include <cmath>
#include <string>
namespace auto_aim
{
    struct ShootEvaluatorConfig
    {
        double yaw_tolerance = 0.07;         // rad
        double pitch_tolerance = 0.04;       // rad
        double tolerance_growth_per_m = 0.004; // 距离越远容差越松（弹道散布）
        double max_angular_rate = 3.2;       // rad/s，超过则提前量不可信
        double min_distance = 1.0;           // m
        double max_distance = 12.0;          // m
    };

    struct ShootInput
    {
        bool solved = false;              // 有可用目标
        bool ballistics_valid = false;    // 弹道有解
        bool cooldown_ready = true;       // 节流已到
        bool already_hit_this_target = false;
        double yaw_error = 0.0;           // |云台角 - 目标角| (rad)
        double pitch_error = 0.0;
        double target_distance = 0.0;     // m
        double target_angular_rate = 0.0; // rad/s
        double fly_time = 0.0;            // s
    };

    enum class ShootVerdict
    {
        Fire,
        NoTarget,
        BallisticsInvalid,
        AlreadyHit,
        Cooldown,
        RateTooHigh,
        OutOfRange,
        NotSettled,
    };

    inline const char* shootVerdictName(ShootVerdict verdict)
    {
        switch (verdict) {
        case ShootVerdict::Fire: return "FIRE";
        case ShootVerdict::NoTarget: return "NO_TARGET";
        case ShootVerdict::BallisticsInvalid: return "BALLISTICS";
        case ShootVerdict::AlreadyHit: return "ALREADY_HIT";
        case ShootVerdict::Cooldown: return "COOLDOWN";
        case ShootVerdict::RateTooHigh: return "RATE_TOO_HIGH";
        case ShootVerdict::OutOfRange: return "OUT_OF_RANGE";
        case ShootVerdict::NotSettled: return "NOT_SETTLED";
        }
        return "UNKNOWN";
    }

    class ShootEvaluator
    {
    public:
        explicit ShootEvaluator(const ShootEvaluatorConfig& config = {}) : config_(config) {}

        ShootVerdict evaluate(const ShootInput& in) const
        {
            if (!in.solved) return ShootVerdict::NoTarget;
            if (!in.ballistics_valid) return ShootVerdict::BallisticsInvalid;
            if (in.already_hit_this_target) return ShootVerdict::AlreadyHit;
            if (!in.cooldown_ready) return ShootVerdict::Cooldown;
            if (std::abs(in.target_angular_rate) > config_.max_angular_rate) {
                return ShootVerdict::RateTooHigh;
            }
            if (in.target_distance < config_.min_distance ||
                in.target_distance > config_.max_distance) {
                return ShootVerdict::OutOfRange;
            }
            // 距离越远、弹道越弯，允许的角度误差按距离放宽。
            const double slack =
                config_.tolerance_growth_per_m * std::max(0.0, in.target_distance);
            if (std::abs(in.yaw_error) > config_.yaw_tolerance + slack ||
                std::abs(in.pitch_error) > config_.pitch_tolerance + slack) {
                return ShootVerdict::NotSettled;
            }
            return ShootVerdict::Fire;
        }

        const ShootEvaluatorConfig& config() const { return config_; }

    private:
        ShootEvaluatorConfig config_;
    };
} // namespace auto_aim
#endif
