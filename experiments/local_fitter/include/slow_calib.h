#pragma once
namespace rm
{
    /// 慢层：α、R 的带遗忘标量信息融合 + 可观测性门控 + 硬边界。
    class SlowCalib
    {
    public:
        SlowCalib();
        void update(double dalpha, double dR, double info_alpha, double info_R, double obs_gate);
        double alpha() const { return alpha_; }
        double R() const { return R_; }
        double varAlpha() const { return var_alpha_; }
        double varR() const { return var_R_; }
        void resetR(double R_prior, double var_R);
        void resetAlpha(double alpha_prior, double var_alpha);
    private:
        double alpha_, R_, var_alpha_, var_R_;
        double forget_, min_info_alpha_, min_info_R_;
    };
}  // namespace rm
