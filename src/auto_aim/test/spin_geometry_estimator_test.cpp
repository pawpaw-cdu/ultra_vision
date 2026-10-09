// spin_geometry_estimator 的可复现单测：合成小陀螺灯条轨迹 → 拟合 → 校验精度。
// 覆盖：pitch=0 无噪（应≈0）；pitch≠0 + 偏轴 + 轴倾斜 + 噪声（应达标）。
#include "estimation/spin_geometry_estimator.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace
{
    using auto_aim::SpinBarObservation;
    using auto_aim::SpinGeometryConfig;
    using auto_aim::SpinGeometryResult;

    void endpoints(const Eigen::Vector3d& C, double R, double c, double L, double alpha,
                   double theta, const Eigen::Vector3d& axis_in,
                   Eigen::Vector3d& top, Eigen::Vector3d& bottom)
    {
        const Eigen::Vector3d a = axis_in.normalized();
        Eigen::Vector3d ref(1.0, 0.0, 0.0);
        if (std::abs(a.dot(ref)) > 0.9) ref = Eigen::Vector3d(0.0, 1.0, 0.0);
        const Eigen::Vector3d e1 = (ref - a * ref.dot(a)).normalized();
        const Eigen::Vector3d e2 = e1.cross(a);
        const Eigen::Vector3d rhat = std::cos(theta) * e1 + std::sin(theta) * e2;
        const Eigen::Vector3d that = -std::sin(theta) * e1 + std::cos(theta) * e2;
        const Eigen::Vector3d center = C + rhat * R + that * c;
        const Eigen::Vector3d dir = std::cos(alpha) * a + std::sin(alpha) * rhat;
        top = center + dir * (0.5 * L);
        bottom = center - dir * (0.5 * L);
    }

    std::vector<SpinBarObservation> makeObs(const SpinGeometryConfig& cfg,
                                            const Eigen::Vector3d& C, double R, double alpha,
                                            const Eigen::Vector3d& axis, double w1, double sin_amp,
                                            double sin_freq, int frames, double dt,
                                            double noise_px, double theta0)
    {
        std::mt19937 rng(7);
        std::normal_distribution<double> noise(0.0, noise_px);
        std::vector<SpinBarObservation> obs;
        for (int i = 0; i < frames; ++i)
        {
            const double t = i * dt;
            const double th = theta0 + w1 * t + sin_amp * std::sin(2.0 * M_PI * sin_freq * t);
            Eigen::Vector3d top, bot;
            endpoints(C, R, cfg.bar_offset_c, cfg.bar_length_L, alpha, th, axis, top, bot);
            SpinBarObservation o;
            o.t = t;
            o.u_top = cfg.K.fx * top.x() / top.z() + cfg.K.cx + noise(rng);
            o.v_top = cfg.K.fy * top.y() / top.z() + cfg.K.cy + noise(rng);
            o.u_bot = cfg.K.fx * bot.x() / bot.z() + cfg.K.cx + noise(rng);
            o.v_bot = cfg.K.fy * bot.y() / bot.z() + cfg.K.cy + noise(rng);
            obs.push_back(o);
        }
        return obs;
    }

    double yawRmseDeg(const std::vector<SpinBarObservation>& obs, const SpinGeometryResult& r,
                      const Eigen::Vector3d& C, double R, double alpha, const Eigen::Vector3d& axis,
                      double w1, double sin_amp, double sin_freq, double theta0)
    {
        (void)C; (void)R; (void)alpha; (void)axis;
        double sum = 0.0;
        for (const auto& o : obs)
        {
            const double th_true = theta0 + w1 * o.t + sin_amp * std::sin(2.0 * M_PI * sin_freq * o.t);
            const double d = std::remainder(r.yawAt(o.t) - th_true, 2.0 * M_PI);
            sum += d * d;
        }
        return std::sqrt(sum / obs.size()) * 180.0 / M_PI;
    }

    // 多灯条 + 真实噪声（端点噪声 + 框抖动 + 丢帧 + 外点）→ 一帧给 8 根（可见的）。
    std::vector<SpinBarObservation> makeMultiObs(const SpinGeometryConfig& cfg,
                                                 const Eigen::Vector3d& C, double R, double alpha,
                                                 const Eigen::Vector3d& axis, double w1,
                                                 double sin_amp, double sin_freq, int frames,
                                                 double dt, double sigma, double box_jitter,
                                                 double dropout, double ofrac, double opx,
                                                 double theta0)
    {
        std::mt19937 rng(11);
        std::normal_distribution<double> noise(0.0, sigma);
        std::uniform_real_distribution<double> uni(0.0, 1.0);
        std::uniform_real_distribution<double> out_off(-opx, opx);
        std::normal_distribution<double> box(0.0, box_jitter);
        std::vector<SpinBarObservation> obs;
        for (int i = 0; i < frames; ++i)
        {
            const double t = i * dt;
            const double th = theta0 + w1 * t + sin_amp * std::sin(2.0 * M_PI * sin_freq * t);
            const double du = box_jitter > 0 ? box(rng) : 0.0;
            const double dv = box_jitter > 0 ? box(rng) : 0.0;
            for (int plate = 0; plate < 4; ++plate)
            {
                const double phase = th + plate * M_PI / 2.0;
                for (int side = -1; side <= 1; side += 2)
                {
                    if (dropout > 0 && uni(rng) < dropout) continue;
                    Eigen::Vector3d top, bot;
                    endpoints(C, R, cfg.bar_offset_c * side, cfg.bar_length_L, alpha, phase,
                              axis, top, bot);
                    SpinBarObservation o;
                    o.t = t; o.plate = plate; o.side = side;
                    o.u_top = cfg.K.fx * top.x() / top.z() + cfg.K.cx + du + noise(rng);
                    o.v_top = cfg.K.fy * top.y() / top.z() + cfg.K.cy + dv + noise(rng);
                    o.u_bot = cfg.K.fx * bot.x() / bot.z() + cfg.K.cx + du + noise(rng);
                    o.v_bot = cfg.K.fy * bot.y() / bot.z() + cfg.K.cy + dv + noise(rng);
                    if (opx > 0 && uni(rng) < ofrac) { o.u_top += out_off(rng); o.v_top += out_off(rng); }
                    obs.push_back(o);
                }
            }
        }
        return obs;
    }
}

int main()
{
    bool pass = true;
    auto check = [&](bool ok, const char* msg) {
        if (!ok) { std::cerr << "FAILED: " << msg << "\n"; pass = false; }
    };

    SpinGeometryConfig cfg;
    const Eigen::Vector3d C(0.35, 0.12, 4.2);
    const double R = 0.25, alpha = 12.0 * M_PI / 180.0, w1 = 3.0, sin_amp = 0.4, sin_freq = 1.0;
    const Eigen::Vector3d axis(0.10, 1.0, 0.0);   // 轴倾斜 ~5.7°
    const int frames = 60;
    const double dt = 1.0 / 30.0;

    // --- 工况 A：pitch=0，光轴上，无噪 → 误差≈0 ---
    {
        auto obs = makeObs(cfg, Eigen::Vector3d(0.0, 0.0, 4.0), R, 0.0,
                           Eigen::Vector3d(0, 1, 0), w1, 0.0, 1.0, frames, dt, 0.0, 0.3);
        auto_aim::SpinGeometryEstimator est(cfg);
        const bool ok = est.fit(obs);
        const auto& r = est.result();
        std::cout << "[A pitch0 on-axis noiseless] valid=" << ok
                  << " center_err=" << (r.center - Eigen::Vector3d(0, 0, 4)).norm() * 1000
                  << "mm yaw_rmse=" << yawRmseDeg(obs, r, C, R, 0, axis, w1, sin_amp, sin_freq, 0.3)
                  << "deg\n";
        check(ok, "A should be valid");
        check((r.center - Eigen::Vector3d(0, 0, 4)).norm() * 1000 < 5.0, "A center err < 5mm");
        check(std::abs(r.radius - R) * 1000 < 5.0, "A radius err < 5mm");
    }

    // --- 工况 B：pitch≠0、偏轴、轴倾斜、0.7px 噪声 → yaw<2°、中心<50mm ---
    {
        auto obs = makeObs(cfg, C, R, alpha, axis, w1, sin_amp, sin_freq, frames, dt, 0.7, 0.3);
        auto_aim::SpinGeometryEstimator est(cfg);
        const bool ok = est.fit(obs);
        const auto& r = est.result();
        const double center_mm = (r.center - C).norm() * 1000.0;
        const double yaw_deg = yawRmseDeg(obs, r, C, R, alpha, axis, w1, sin_amp, sin_freq, 0.3);
        const double axis_err = std::acos(std::clamp(
            Eigen::Vector3d(r.axis_tx, 1, r.axis_tz).normalized().dot(axis.normalized()),
            -1.0, 1.0)) * 180.0 / M_PI;
        std::cout << "[B pitch12 offaxis tilt noise] valid=" << ok
                  << " arc=" << r.arc_coverage_deg << "deg center_err=" << center_mm
                  << "mm yaw_rmse=" << yaw_deg << "deg axis_err=" << axis_err
                  << "deg reproj=" << r.reproj_rmse_px << "px\n";
        check(ok, "B should be valid");
        check(center_mm < 50.0, "B center err < 50mm");
        check(yaw_deg < 2.0, "B yaw RMSE < 2deg");
        check(std::abs(r.radius - R) * 1000 < 10.0, "B radius err < 10mm");
        check(axis_err < 2.0, "B axis recovered < 2deg");
    }

    // --- 工况 C：短弧（~130°）→ 应判为不可用（门控生效） ---
    {
        auto obs = makeObs(cfg, C, R, alpha, axis, w1, sin_amp, sin_freq, 24, dt, 0.5, 0.3);
        auto_aim::SpinGeometryEstimator est(cfg);
        const bool ok = est.fit(obs);
        std::cout << "[C short-arc ~132deg] valid=" << ok
                  << " arc=" << est.result().arc_coverage_deg << "deg (expect invalid)\n";
        check(!ok, "C short arc should be gated out");
    }

    // --- 工况 D：多灯条 + 真实噪声（sigma2px + 框抖动5px + 丢帧30% + 外点5%@12px）---
    {
        auto obs = makeMultiObs(cfg, C, R, alpha, axis, w1, sin_amp, sin_freq, frames, dt,
                                /*sigma=*/2.0, /*box=*/5.0, /*dropout=*/0.30, /*ofrac=*/0.05,
                                /*opx=*/12.0, 0.3);
        auto_aim::SpinGeometryEstimator est(cfg);
        const bool ok = est.fit(obs);
        const auto& r = est.result();
        const double center_mm = (r.center - C).norm() * 1000.0;
        double ysum = 0.0;
        for (const auto& o : obs)
        {
            const double th_true = 0.3 + w1 * o.t + sin_amp * std::sin(2.0 * M_PI * sin_freq * o.t);
            const double d = std::remainder(r.yawAt(o.t) - th_true, 2.0 * M_PI);
            ysum += d * d;
        }
        const double yaw_deg = std::sqrt(ysum / obs.size()) * 180.0 / M_PI;
        std::cout << "[D multi-bar realistic noise] valid=" << ok << " n_bars=" << obs.size()
                  << " arc=" << r.arc_coverage_deg << "deg center_err=" << center_mm
                  << "mm yaw_rmse=" << yaw_deg << "deg reproj=" << r.reproj_rmse_px << "px\n";
        check(ok, "D should be valid");
        check(yaw_deg < 3.0, "D yaw RMSE < 3deg (multi-bar)");
        check(center_mm < 120.0, "D center err < 120mm (multi-bar)");
    }

    std::cout << (pass ? "ALL PASSED\n" : "SOME FAILED\n");
    return pass ? 0 : 1;
}
