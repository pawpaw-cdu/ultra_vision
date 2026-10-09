#include "simulator.h"
#include "trajectory_fitter.h"
#include "slow_calib.h"
#include "state_solver.h"
#include "visualizer.h"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>

namespace
{
    double yawErrDeg(const rm::FitResult& f, const std::vector<rm::SimulationFrame>& fr)
    {
        const double d = std::remainder(f.yawAt(fr.back().timestamp) - fr.back().yaw_gt, 2 * M_PI);
        return std::abs(d) * 180.0 / M_PI;
    }

    struct Metrics { double yaw_deg, omega_err, center_cm, rmse_px; };

    Metrics evaluate(const std::vector<rm::SimulationFrame>& fr, const rm::FitResult& fit)
    {
        Metrics m;
        m.yaw_deg = yawErrDeg(fit, fr);
        m.omega_err = std::abs(fit.omega - fr.back().omega_gt);
        m.center_cm = (fit.centerAt(fr.back().timestamp, fr.front().timestamp)
                       - fr.back().center_cam_gt).norm() * 100.0;
        m.rmse_px = fit.reprojection_rmse;
        return m;
    }

    std::vector<rm::LightBarObservation> asObs(const std::vector<rm::SimulationFrame>& fr)
    {
        std::vector<rm::LightBarObservation> obs;
        for (const auto& f : fr)
        {
            rm::LightBarObservation o;
            o.timestamp = f.timestamp;
            o.top_px = f.top_px; o.bottom_px = f.bottom_px;
            obs.push_back(o);
        }
        return obs;
    }

    void runCase(const char* name, rm::SimulationConfig cfg, rm::FitConfig fc,
                 double alpha_fit, double R_fit, std::ofstream& out)
    {
        const auto fr = rm::runSimulation(cfg, "");
        const auto obs = asObs(fr);
        const rm::FitResult fit = rm::fitTrajectory(obs, cfg.K, cfg.gimbal_pitch, alpha_fit, R_fit,
                                                    cfg.lightbar_len, fc);
        const Metrics m = evaluate(fr, fit);
        std::cout << std::left << std::setw(26) << name << " yaw=" << std::setw(7) << m.yaw_deg
                  << " omega_err=" << std::setw(7) << m.omega_err
                  << " center_cm=" << std::setw(7) << m.center_cm
                  << " rmse_px=" << std::setw(7) << m.rmse_px << "\n";
        out << name << ',' << m.yaw_deg << ',' << m.omega_err << ',' << m.center_cm << ','
            << m.rmse_px << '\n';
    }
}

int main()
{
    using namespace rm;
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "=== RM local fitter validation ===\n";

    std::ofstream out("data/summary.csv");
    out << "case,yaw_deg,omega_err,center_cm,rmse_px\n";

    SimulationConfig base;
    base.K.fx = 800; base.K.fy = 800; base.K.cx = 432; base.K.cy = 960;
    base.radius = 0.20;
    base.pitch_install = 15.0 * M_PI / 180.0;
    base.lightbar_len = 0.10; base.lightbar_h = 0.10;
    base.omega0 = 3.0; base.num_frames = 20; base.dt = 0.01;
    base.pixel_noise_std = 0.5;

    FitConfig fc;
    fc.center_init = Vec3(0, 0, 3);
    fc.h_init = 0.10;
    fc.omega_init = 3.0;

    const double a15 = 15.0 * M_PI / 180.0;

    runCase("base w3 Cz3 s0.5 p0", base, fc, a15, 0.20, out);
    { auto c = base; c.num_frames = 40; runCase("base w3 N40 (cover>pi/3)", c, fc, a15, 0.20, out); }
    { auto c = base; c.omega0 = 1.0; runCase("omega=1", c, fc, a15, 0.20, out); }
    { auto c = base; c.omega0 = 6.0; runCase("omega=6", c, fc, a15, 0.20, out); }
    { auto c = base; c.center_cam = Vec3(0, 0, 2); runCase("Cz=2", c, fc, a15, 0.20, out); }
    { auto c = base; c.center_cam = Vec3(0, 0, 5); runCase("Cz=5", c, fc, a15, 0.20, out); }
    { auto c = base; c.pixel_noise_std = 0.1; runCase("sigma=0.1", c, fc, a15, 0.20, out); }
    { auto c = base; c.pixel_noise_std = 1.0; runCase("sigma=1.0", c, fc, a15, 0.20, out); }
    { auto c = base; c.gimbal_pitch = 10.0 * M_PI / 180.0; runCase("gimbal_pitch=+10", c, fc, a15, 0.20, out); }
    { auto c = base; c.gimbal_pitch = -10.0 * M_PI / 180.0; runCase("gimbal_pitch=-10", c, fc, a15, 0.20, out); }
    { auto c = base; c.drift_velocity = Vec3(0.2, 0, 0); runCase("drift=0.2", c, fc, a15, 0.20, out); }
    { auto c = base; c.drift_velocity = Vec3(0.5, 0, 0); runCase("drift=0.5", c, fc, a15, 0.20, out); }
    { auto c = base; c.num_frames = 40; c.pitch_install = 0.0;
      runCase("alpha=0 (yaw unobservable?)", c, fc, 0.0, 0.20, out); }

    for (double a0_deg : {13.0, 17.0})
    {
        SimulationConfig c = base;
        c.num_frames = 800;
        const auto fr = runSimulation(c, "");
        SlowCalib slow;
        slow.resetAlpha(a0_deg * M_PI / 180.0, std::pow(2 * M_PI / 180.0, 2));
        slow.resetR(0.20, std::pow(0.005, 2));
        const int N = 20;
        for (int w0 = 0; w0 + N <= static_cast<int>(fr.size()); w0 += N)
        {
            std::vector<LightBarObservation> obs;
            for (int i = w0; i < w0 + N; ++i) { LightBarObservation o;
                o.timestamp = fr[i].timestamp; o.top_px = fr[i].top_px; o.bottom_px = fr[i].bottom_px;
                obs.push_back(o); }
            const FitResult f = fitTrajectory(obs, c.K, c.gimbal_pitch, slow.alpha(), slow.R(),
                                              c.lightbar_len, fc);
            double a = slow.alpha(), R = slow.R(), ia = 0, iR = 0;
            refineSlowParams(obs, c.K, c.gimbal_pitch, c.lightbar_len, f, a, R, ia, iR);
            slow.update(a - slow.alpha(), R - slow.R(), ia, iR, 1.0);
        }
        std::cout << "slow alpha from " << a0_deg << "deg -> " << slow.alpha() * 180 / M_PI
                  << "deg (target 15)\n";
    }

    for (double R0 : {0.18, 0.22})
    {
        SimulationConfig c = base;
        c.num_frames = 800;
        const auto fr = runSimulation(c, "");
        SlowCalib slow;
        slow.resetAlpha(a15, std::pow(0.5 * M_PI / 180.0, 2));
        slow.resetR(R0, std::pow(0.02, 2));
        const int N = 20;
        for (int w0 = 0; w0 + N <= static_cast<int>(fr.size()); w0 += N)
        {
            std::vector<LightBarObservation> obs;
            for (int i = w0; i < w0 + N; ++i) { LightBarObservation o;
                o.timestamp = fr[i].timestamp; o.top_px = fr[i].top_px; o.bottom_px = fr[i].bottom_px;
                obs.push_back(o); }
            const FitResult f = fitTrajectory(obs, c.K, c.gimbal_pitch, slow.alpha(), slow.R(),
                                              c.lightbar_len, fc);
            double a = slow.alpha(), R = slow.R(), ia = 0, iR = 0;
            refineSlowParams(obs, c.K, c.gimbal_pitch, c.lightbar_len, f, a, R, ia, iR);
            slow.update(a - slow.alpha(), R - slow.R(), ia, iR, 1.0);
        }
        std::cout << "slow R from " << R0 << " -> " << slow.R() << " (target 0.20)\n";
    }

    {
        auto c = base; c.num_frames = 60;
        const auto fr = runSimulation(c, "data/sim.csv");
        const auto obs = asObs(fr);
        const FitResult f = fitTrajectory(obs, c.K, c.gimbal_pitch, a15, 0.20, c.lightbar_len, fc);
        visualizeResults(fr, f, c.K, c.gimbal_pitch, a15, 0.20, c.lightbar_len, "data/results.png");
    }

    out.close();
    std::cout << "wrote data/summary.csv + data/sim.csv + data/results.png\n";
    return 0;
}
