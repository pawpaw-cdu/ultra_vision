#ifndef AUTO_AIM_ROTATION_RATE_ESTIMATOR_HPP
#define AUTO_AIM_ROTATION_RATE_ESTIMATOR_HPP

#include <deque>

namespace auto_aim
{
    struct RotationRateEstimatorConfig {
        double window = 0.35;
        double min_span = 0.12;
        double max_sample_gap = 0.25;
        double max_rate = 10.0;
        double min_rate = 0.6;
        double max_residual = 0.16;
        // 单步"阶跃"门限（rad）：超过它就不是转速而是**量测跳变** —— 对称装甲板的
        // PnP 姿态会在 ±90° 上翻面（位置不动、朝向翻 1.57 rad）。实车实测
        // （静止靶回放）：这类跳变会让窗口内的中位斜率变成
        // 1.57/窗口 ≈ 0.8 rad/s（44°/s），经 updateOmega 灌进 EKF 后乘 386 ms
        // 提前量就是瞄点甩 5°。取 1.0 rad（57°）：真实小陀螺最快 3 rad/s，跨
        // 0.2 s 的丢失也只转 0.6 rad，够不到；而翻面 1.57 rad 一定超过。
        // 落在 [max_step, max_step_angle) 的单步按"坏样本"丢弃（保留已有转速），
        // ≥ max_step_angle 的按"阶跃"处理：整窗重来，绝不跨着它拟合。
        double max_step_angle = 1.0;
        int min_slopes = 4;
    };

    class RotationRateEstimator
    {
    public:
        explicit RotationRateEstimator(
            const RotationRateEstimatorConfig& config = {});

        void reset();
        bool update(double radial_angle, double timestamp);

        double omega() const { return valid_ ? omega_ : 0.0; }
        bool valid() const { return valid_; }
        double residual() const { return residual_; }

    private:
        struct Sample {
            double timestamp = 0.0;
            double angle = 0.0;
        };

        RotationRateEstimatorConfig config_;
        std::deque<Sample> samples_;
        // Smoothed sample interval, used to size the fitting window. A fixed
        // window cannot hold enough samples once the detector runs at a few Hz.
        double sample_interval_ = 0.0;
        double omega_ = 0.0;
        double residual_ = 0.0;
        bool valid_ = false;
    };
}

#endif
