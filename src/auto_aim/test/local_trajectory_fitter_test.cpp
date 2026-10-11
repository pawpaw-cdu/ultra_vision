// local_trajectory_fitter 的可复现单测：合成四板小陀螺灯条观测 → 拟合 → 校验。
// 覆盖：快层 θ/ω（正面 + 俯仰）、慢层 α/R（单步 refine 与四板联合结论）。
#include "estimation/local_trajectory_fitter.hpp"
#include "common/chassis_model.hpp"   // armorPhaseOffset：相位表的唯一来源

#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

namespace
{
    using auto_aim::LightBarObservation;
    using auto_aim::LocalTrajectoryFitConfig;
    using auto_aim::LocalFitResult;
    using auto_aim::LocalTrajectoryFitter;

    struct Model
    {
        LocalTrajectoryFitConfig cfg;
        double alpha = 15.0 * M_PI / 180.0;
        double R = 0.20;
        double h = 0.10;
        double omega = 3.0;
        std::array<double, 3> center{{0.0, 0.0, 3.0}};
    };

    // 车辆系角点 → 相机系像素（与 fitter 同模型）。
    void project(const Model& m, double theta, double a, double b, double& u, double& v)
    {
        const double rho = m.R - b * std::sin(m.alpha);
        const double z = m.h + b * std::cos(m.alpha);
        const double pvx = rho * std::cos(theta) - a * std::sin(theta);
        const double pvy = rho * std::sin(theta) + a * std::cos(theta);
        const double gp = m.cfg.gimbal_pitch;
        const double s = std::sin(gp), c = std::cos(gp);
        const double X = pvy + m.center[0];
        const double Y = -s * pvx - c * z + m.center[1];
        const double Z = c * pvx - s * z + m.center[2];
        u = m.cfg.K.fx * X / Z + m.cfg.K.cx;
        v = m.cfg.K.fy * Y / Z + m.cfg.K.cy;
    }

    std::vector<LightBarObservation> synth(const Model& m, int n_frames, int n_plates, double dt,
                                           double sigma, unsigned seed)
    {
        std::mt19937 rng(seed);
        std::normal_distribution<double> noise(0.0, sigma);
        std::vector<LightBarObservation> obs;
        const double W = m.cfg.plate_width, L = m.cfg.lightbar_len;
        for (int k = 0; k < n_frames; ++k)
        {
            const double t = k * dt;
            for (int p = 0; p < n_plates; ++p)
            {
                // 板相位 = 底盘相位 + 该板的**配置**偏移。必须走配置里的表（而不是
                // 这里写死 p·90°），否则"改相位表"这类改动不会被这个测试发现 ——
                // 而相位表正是拟合器 ↔ EKF 之间最容易错的地方。
                const double theta = m.omega * t +
                    (p < static_cast<int>(m.cfg.plate_yaw_offsets.size())
                         ? m.cfg.plate_yaw_offsets[static_cast<std::size_t>(p)]
                         : p * 2.0 * M_PI / n_plates);
                for (double a : { +0.5 * W, -0.5 * W })
                {
                    LightBarObservation o;
                    o.timestamp = t;
                    o.armor_id = p;
                    o.bar_offset = a;
                    double u, v;
                    project(m, theta, a, +0.5 * L, u, v);
                    o.u_top = u + noise(rng); o.v_top = v + noise(rng);
                    project(m, theta, a, -0.5 * L, u, v);
                    o.u_bot = u + noise(rng); o.v_bot = v + noise(rng);
                    obs.push_back(o);
                }
            }
        }
        return obs;
    }

    int failures = 0;
    void check(bool ok, const char* name, double value, double limit)
    {
        std::cout << (ok ? "  [ok]   " : "  [FAIL] ") << name << " = " << value
                  << " (limit " << limit << ")\n";
        if (!ok) ++failures;
    }
}  // namespace

int main()
{
    Model m;
    m.cfg.lightbar_len = 0.10;
    m.cfg.plate_width = 0.135;
    m.cfg.K = {800, 800, 432, 960};

    // ---- 快层：四板联合，正面，N=40 ----
    {
        std::cout << "快层四板联合 (omega=3, Cz=3, sigma=0.5):\n";
        auto obs = synth(m, 40, 4, 0.01, 0.5, 7);
        LocalTrajectoryFitter fitter(m.cfg);
        LocalFitResult r;
        const bool ok = fitter.fit(obs, m.alpha, m.R, r);
        const double yaw_err = std::abs(std::remainder(r.yaw - m.omega * 0.39, 2 * M_PI)) * 180 / M_PI;
        check(ok, "valid", ok ? 1 : 0, 0.5);
        check(yaw_err < 3.0, "yaw_err_deg", yaw_err, 3.0);
        check(std::abs(r.omega - 3.0) < 0.2, "omega_err", std::abs(r.omega - 3.0), 0.2);
        check(r.reprojection_rmse_px > 0.2 && r.reprojection_rmse_px < 0.8, "rmse_px",
              r.reprojection_rmse_px, 0.8);
    }

    // ---- 慢层：四板联合，从错初值一步 refine α/R ----
    {
        std::cout << "慢层四板联合一步 refine (true alpha=15, R=0.20; a0=13, R0=0.18):\n";
        auto obs = synth(m, 80, 4, 0.01, 0.5, 11);
        LocalTrajectoryFitter fitter(m.cfg);
        double alpha = 13.0 * M_PI / 180.0, R = 0.18, ia = 0, iR = 0;
        LocalTrajectoryFitter::FastFit fast;
        const bool ok = fitter.fitFast(obs, alpha, R, fast);
        fitter.refineSlow(obs, fast, alpha, R, ia, iR);
        check(ok, "fast_valid", ok ? 1 : 0, 0.5);
        check(std::abs(alpha * 180 / M_PI - 15.0) < 1.6 && std::abs(R - 0.20) < 0.02,
              "alpha_R_converged", std::abs(R - 0.20), 0.02);
        std::cout << "    alpha=" << alpha * 180 / M_PI << "deg R=" << R
                  << " info_a=" << ia << " info_R=" << iR << "\n";
    }

    // ---- 相位表绑定：拟合器的 plate_yaw_offsets 必须与 EKF/chassis_model 同一张表 ----
    // 两张表不一致时，"拟合 yaw"和"EKF yaw"会整体差 ±90°/180°：四块板对称时被
    // "模 90° 吸附"掩盖，启用逐板几何（plate_distance / l / h）后就变成模型分叉。
    {
        std::cout << "相位表绑定 (fitter plate_yaw_offsets vs chassis_model::armorPhaseOffset):\n";
        LocalTrajectoryFitConfig cfg;
        for (int k = 0; k < 4; ++k)
        {
            const double expected = auto_aim::armorPhaseOffset(k);
            const double actual = k < static_cast<int>(cfg.plate_yaw_offsets.size())
                ? cfg.plate_yaw_offsets[static_cast<std::size_t>(k)]
                : std::numeric_limits<double>::quiet_NaN();
            check(std::isfinite(actual) &&
                      std::abs(std::remainder(actual - expected, 2 * M_PI)) < 1e-9,
                  "plate_offset_binding", actual, expected);
        }
    }

    std::cout << (failures == 0 ? "ALL PASS\n" : "FAILURES\n");
    return failures == 0 ? 0 : 1;
}
