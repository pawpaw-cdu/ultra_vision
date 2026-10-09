// 局部拟合器"系统收益"量化：可观测性 → 单帧 PnP yaw 方差 → (θ,ω)EKF →
// 瞄点横向误差 → 开火窗占空。纯解析 + 2 状态 Kalman，可复现、不依赖仿真器。
#include "types.h"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>

namespace
{
    using rm::Vec2; using rm::Vec3; using rm::CameraIntrinsics;

    // 正面差分可观测性 s(d) = d(u_top-u_bot)/dθ（px/rad）。
    double obsSignal(double Cz, double alpha, const CameraIntrinsics& K, double R, double L)
    {
        const double h = 1e-4;
        auto delta = [&](double dth)
        {
            Vec3 t, b;
            rm::barEndpointsCamera(Vec3(0, 0, Cz), 0.10, dth, alpha, R, L, 0.0, t, b);
            const Vec2 pt = rm::project(t, K), pb = rm::project(b, K);
            return pt.x() - pb.x();
        };
        return (delta(+0.5 * h) - delta(-0.5 * h)) / h;
    }

    // 2 状态 Kalman：(θ, ω)，F=[[1,dt],[0,1]]，Q=diag(qth,qw)。H 为 0/1 行。
    struct Kalman2
    {
        double th = 0.0, w = 3.0;
        double p00 = 1.0, p01 = 0.0, p11 = 1.0;
        // 物理含义：底盘 yaw 近似匀速旋转 → ω 的过程噪声小；qth 也很小。
        double qth = 1e-6, qw = 0.05, dt = 0.01;

        void predict()
        {
            th += w * dt;
            const double n00 = p00 + 2 * dt * p01 + dt * dt * p11 + qth;
            const double n01 = p01 + dt * p11;
            const double n11 = p11 + qw;
            p00 = n00; p01 = n01; p11 = n11;
        }
        void update(double H0, double H1, double z, double r)
        {
            const double S = H0 * H0 * p00 + 2 * H0 * H1 * p01 + H1 * H1 * p11 + r;
            const double K0 = (p00 * H0 + p01 * H1) / S;
            const double K1 = (p01 * H0 + p11 * H1) / S;
            const double res = z - (H0 * th + H1 * w);
            th += K0 * res; w += K1 * res;
            // Joseph: P' = (I-KH)P(I-KH)ᵀ + K r Kᵀ
            const double m00 = 1 - K0 * H0, m01 = -K0 * H1;
            const double m10 = -K1 * H0, m11 = 1 - K1 * H1;
            const double ap00 = m00 * p00 + m01 * p01, ap01 = m00 * p01 + m01 * p11;
            const double ap10 = m10 * p00 + m11 * p01, ap11 = m10 * p01 + m11 * p11;
            const double np00 = ap00 * m00 + ap01 * m01 + K0 * K0 * r;
            const double np01 = ap00 * m10 + ap01 * m11 + K0 * K1 * r;
            const double np11 = ap10 * m10 + ap11 * m11 + K1 * K1 * r;
            p00 = np00; p01 = np01; p11 = np11;
        }
    };

    // 返回 {σθ, σω}（稳态）。
    struct SW { double sth, sw; };
    SW steady(bool use_fit, double sigma_pnp, double sigma_fit_th, double sigma_fit_w)
    {
        Kalman2 f;
        const int n = 3000;
        double acc0 = 0, acc1 = 0; int cnt = 0;
        for (int i = 0; i < n; ++i)
        {
            f.predict();
            if (use_fit) { f.update(1, 0, 0, sigma_fit_th * sigma_fit_th);
                           f.update(0, 1, 3.0, sigma_fit_w * sigma_fit_w); }
            else         { f.update(1, 0, 0, sigma_pnp * sigma_pnp); }
            if (i > n / 2) { acc0 += f.p00; acc1 += f.p11; ++cnt; }
        }
        return { std::sqrt(acc0 / cnt), std::sqrt(acc1 / cnt) };
    }
}

int main()
{
    rm::CameraIntrinsics K; K.fx = 800; K.fy = 800; K.cx = 432; K.cy = 960;
    const double alpha = 15.0 * M_PI / 180.0, R = 0.20, L = 0.10;
    const double sigma_px = 0.5, lead = 0.10;
    const double sigma_fit_th = 0.10 * M_PI / 180.0;  // 拟合 yaw 误差（保守；实验实测 ~0.01°）
    const double sigma_fit_w = 0.05;                   // 拟合 ω 误差（实验实测 ~0.006~0.2）

    std::ofstream csv("data/value_analysis.csv");
    csv << "d_m,signal_px_per_rad,sigma_yaw_pnp_deg,ekf_yaw_pnp_deg,ekf_omega_pnp,"
           "ekf_yaw_fit_deg,ekf_omega_fit,miss_pnp_cm,miss_fit_cm,hit_pnp,hit_fit\n";
    const double armor_half_w = 6.75;   // 小装甲板半宽 cm（命中判据）
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "===== 局部拟合器系统收益 (alpha=15,R=0.20,L=0.10,fx=800,sigma=0.5px,lead=0.1s) =====\n";
    std::cout << " d(m)   s(px/rad)  s_yaw_PnP(deg)  EKF_yaw_PnP  EKF_yaw_fit  miss_PnP(cm)  miss_fit(cm)  hit_PnP  hit_fit\n";
    for (double d : {1.0, 2.0, 3.0, 4.0, 5.0, 8.0})
    {
        const double s = obsSignal(d, alpha, K, R, L);
        const double sigma_th_pnp = std::abs(s) > 1e-9 ? (std::sqrt(2.0) * sigma_px) / std::abs(s) : 1e9;
        const SW pnp = steady(false, sigma_th_pnp, 0, 0);
        const SW fit = steady(true, 0, sigma_fit_th, sigma_fit_w);
        auto aim = [&](double sth, double sw) { return (R / d) * std::sqrt(sth * sth + (sw * lead) * (sw * lead)); };
        const double aim_pnp = aim(pnp.sth, pnp.sw), aim_fit = aim(fit.sth, fit.sw);
        auto hit = [&](double miss_cm) {
            if (!std::isfinite(miss_cm) || miss_cm < 1e-6) return 1.0;
            return std::erf(armor_half_w / (std::sqrt(2.0) * miss_cm));
        };
        std::cout << std::setw(5) << d << "  " << std::setw(10) << s << "  " << std::setw(13)
                  << sigma_th_pnp * 180 / M_PI << "  " << std::setw(11) << pnp.sth * 180 / M_PI
                  << "  " << std::setw(11) << fit.sth * 180 / M_PI << "  " << std::setw(11)
                  << aim_pnp * d * 100 << "  " << std::setw(11) << aim_fit * d * 100 << "  "
                  << std::setw(8) << hit(aim_pnp * d * 100) << "  "
                  << std::setw(8) << hit(aim_fit * d * 100) << "\n";
        csv << d << ',' << s << ',' << sigma_th_pnp * 180 / M_PI << ',' << pnp.sth * 180 / M_PI
            << ',' << pnp.sw << ',' << fit.sth * 180 / M_PI << ',' << fit.sw << ',' << aim_pnp * d * 100
            << ',' << aim_fit * d * 100 << ',' << hit(aim_pnp * d * 100) << ','
            << hit(aim_fit * d * 100) << '\n';
    }
    csv.close();
    std::cout << "\n口径: miss_* = 靶面横向脱靶量 = R·sqrt(σθ² + (σω·lead)²) (cm);\n"
                 "      hit_*  = 落在小装甲板半宽 6.75cm 内的概率 erf(6.75/(sqrt2·miss));\n"
                 "      EKF = 2 状态(θ,ω) Kalman 稳态（PnP 只测 yaw；fit 同时测 yaw+ω）。\n";

    // ---- 角点噪声扫描（3 m）：真实检测角点精度 ~6~13px（见 CORNER_ACCURACY.md）----
    // 拟合器是多帧估计：σθ_fit ∝ σ_px/√N_eff，实测在 σ_px=0.5 时 ~0.068°，
    // 故按线性外推 σθ_fit(σ_px) = 0.068°·(σ_px/0.5)。
    std::ofstream csv2("data/value_vs_noise.csv");
    csv2 << "sigma_px,yaw_pnp_deg,yaw_fit_deg,miss_pnp_cm,miss_fit_cm,hit_pnp,hit_fit\n";
    const double d = 3.0;
    const double s3 = std::abs(obsSignal(d, alpha, K, R, L));
    std::cout << "\n=== 角点噪声扫描 @3m ===\n";
    std::cout << " sigma_px  yaw_PnP(deg)  yaw_fit(deg)  miss_PnP(cm)  miss_fit(cm)  hit_PnP  hit_fit\n";
    for (double sp : {0.5, 2.0, 4.0, 8.0, 13.0})
    {
        const double sy_pnp = (std::sqrt(2.0) * sp) / s3 * 180 / M_PI;
        const double sy_fit = 0.068 * (sp / 0.5);
        const double m_pnp = R / d * (sy_pnp * M_PI / 180.0);
        const double m_fit = R / d * (sy_fit * M_PI / 180.0);
        auto hit = [&](double miss) { return std::erf(6.75 / (std::sqrt(2.0) * miss * d * 100)); };
        std::cout << std::setw(8) << sp << "  " << std::setw(11) << sy_pnp << "  " << std::setw(11)
                  << sy_fit << "  " << std::setw(12) << m_pnp * d * 100 << "  " << std::setw(12)
                  << m_fit * d * 100 << "  " << std::setw(7) << hit(m_pnp) << "  " << std::setw(7)
                  << hit(m_fit) << "\n";
        csv2 << sp << ',' << sy_pnp << ',' << sy_fit << ',' << m_pnp * d * 100 << ','
             << m_fit * d * 100 << ',' << hit(m_pnp) << ',' << hit(m_fit) << '\n';
    }
    csv2.close();
    return 0;
}
