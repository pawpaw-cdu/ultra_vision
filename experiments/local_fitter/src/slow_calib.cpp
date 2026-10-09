#include "slow_calib.h"
#include <algorithm>
#include <cmath>

namespace rm
{
    SlowCalib::SlowCalib()
        : alpha_(15.0 * M_PI / 180.0), R_(0.20),
          var_alpha_(std::pow(2.0 * M_PI / 180.0, 2)), var_R_(std::pow(0.03, 2)),
          forget_(0.99), min_info_alpha_(1e-9), min_info_R_(1e-6) {}

    void SlowCalib::update(double dalpha, double dR, double info_alpha, double info_R,
                           double obs_gate)
    {
        var_alpha_ /= forget_;
        var_R_ /= forget_;
        var_alpha_ = std::max(var_alpha_, 1e-12);
        var_R_ = std::max(var_R_, 1e-12);
        if (!(info_alpha * obs_gate > min_info_alpha_ && info_R * obs_gate > min_info_R_)) return;
        if (!std::isfinite(info_alpha) || !std::isfinite(info_R) ||
            !std::isfinite(dalpha) || !std::isfinite(dR)) return;
        var_alpha_ = 1.0 / (1.0 / var_alpha_ + info_alpha);
        var_R_ = 1.0 / (1.0 / var_R_ + info_R);
        alpha_ += dalpha * info_alpha * var_alpha_;
        R_ += dR * info_R * var_R_;
        if (!std::isfinite(alpha_) || !std::isfinite(R_)) { alpha_ = 15.0 * M_PI / 180.0; R_ = 0.20; }
        alpha_ = std::clamp(alpha_, 8.0 * M_PI / 180.0, 25.0 * M_PI / 180.0);
        R_ = std::clamp(R_, 0.10, 0.35);
    }

    void SlowCalib::resetR(double R_prior, double var_R)
    {
        R_ = std::clamp(R_prior, 0.10, 0.35);
        var_R_ = std::max(1e-6, var_R);
    }

    void SlowCalib::resetAlpha(double alpha_prior, double var_alpha)
    {
        alpha_ = std::clamp(alpha_prior, 8.0 * M_PI / 180.0, 25.0 * M_PI / 180.0);
        var_alpha_ = std::max(1e-9, var_alpha);
    }
}  // namespace rm
