#include "experiments.h"

#include "simulator.h"
#include "slow_calib.h"
#include "trajectory_fitter.h"
#include "types.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

#ifdef RM_HAVE_OPENCV
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#endif

namespace rm
{
    namespace
    {
        std::vector<double> plateOffsets(int np)
        {
            std::vector<double> o(np);
            for (int k = 0; k < np; ++k) o[k] = k * 2.0 * M_PI / np;
            return o;
        }

        // 从多板帧序列取 [i0,i1) 帧的全部灯条观测（每板两根，a=±W/2）。
        // only_plate >= 0 时只取该板（用于"单板 vs 多板"公平对照，模型都带正确 a）。
        std::vector<LightBarObservation> barsFromFrames(const std::vector<MultiFrame>& fr,
                                                        int i0, int i1, double W, int only_plate = -1)
        {
            std::vector<LightBarObservation> out;
            for (int i = i0; i < i1 && i < static_cast<int>(fr.size()); ++i)
            {
                for (const auto& p : fr[i].plates)
                {
                    if (only_plate >= 0 && p.armor_id != only_plate) continue;
                    LightBarObservation l, r;
                    l.timestamp = r.timestamp = fr[i].timestamp;
                    l.armor_id = r.armor_id = p.armor_id;
                    l.top_px = p.corners_px[0]; l.bottom_px = p.corners_px[1]; l.bar_offset = +0.5 * W;
                    r.top_px = p.corners_px[2]; r.bottom_px = p.corners_px[3]; r.bar_offset = -0.5 * W;
                    l.confidence = r.confidence = p.confidence;
                    out.push_back(l); out.push_back(r);
                }
            }
            return out;
        }

        double wrapDeg(double rad) { return std::remainder(rad, 2.0 * M_PI) * 180.0 / M_PI; }

        // 板局部物点（yaw=0，法线 +X）；顺序 [LT,LB,RT,RB]。
        std::array<Vec3, 4> plateObject(double alpha, double R, double h, double L, double W)
        {
            const double rho_top = R - 0.5 * L * std::sin(alpha);
            const double rho_bot = R + 0.5 * L * std::sin(alpha);
            const double z_top = h + 0.5 * L * std::cos(alpha);
            const double z_bot = h - 0.5 * L * std::cos(alpha);
            return { Vec3(rho_top, +0.5 * W, z_top), Vec3(rho_bot, +0.5 * W, z_bot),
                     Vec3(rho_top, -0.5 * W, z_top), Vec3(rho_bot, -0.5 * W, z_bot) };
        }

        double stdDeg(const std::vector<double>& v)
        {
            if (v.size() < 2) return 0.0;
            double m = 0.0; for (double x : v) m += x; m /= v.size();
            double s = 0.0; for (double x : v) s += (x - m) * (x - m);
            return std::sqrt(s / v.size());
        }

        double meanAbs(const std::vector<double>& v)
        {
            if (v.empty()) return 0.0;
            double s = 0.0; for (double x : v) s += std::abs(x);
            return s / v.size();
        }

        // 单帧 PnP：已知装甲板几何(α,R,h,W,L)，用一帧 4 角点解整体 yaw θ。
        bool pnpTheta(const Vec2 px[4], const CameraIntrinsics& K, double gp, double alpha,
                      double R, double h, double L, double W, double plate_yaw_off,
                      double& theta_out)
        {
#ifdef RM_HAVE_OPENCV
            std::vector<cv::Point3f> obj;
            std::vector<cv::Point2f> img;
            const auto O = plateObject(alpha, R, h, L, W);
            for (int i = 0; i < 4; ++i)
            {
                obj.emplace_back(static_cast<float>(O[i].x()), static_cast<float>(O[i].y()),
                                 static_cast<float>(O[i].z()));
                img.emplace_back(static_cast<float>(px[i].x()), static_cast<float>(px[i].y()));
            }
            const cv::Matx33d Kc(K.fx, 0, K.cx, 0, K.fy, K.cy, 0, 0, 1);
            std::vector<cv::Mat> rvecs, tvecs;
            int nsol = 0;
            // 我们的物点不在 Z=0 平面（α 倾角），IPPE 会错；用通用解算器。
            try { nsol = cv::solvePnPGeneric(obj, img, Kc, cv::noArray(), rvecs, tvecs, false,
                                             cv::SOLVEPNP_SQPNP); }
            catch (const cv::Exception&) { nsol = 0; }
            if (nsol <= 0)
            {
                cv::Mat r0 = cv::Mat::zeros(3, 1, CV_64F), t0 = cv::Mat::zeros(3, 1, CV_64F);
                if (!cv::solvePnP(obj, img, Kc, cv::noArray(), r0, t0, false, cv::SOLVEPNP_ITERATIVE))
                    return false;
                rvecs = {r0}; tvecs = {t0}; nsol = 1;
            }
            // 平面 PnP 有两个解（正/反面）；取"板面法线朝向相机"的那个。
            cv::Mat Rpo = cv::Mat::eye(3, 3, CV_64F);
            bool found = false;
            for (int s = 0; s < nsol; ++s)
            {
                cv::Mat Rc; cv::Rodrigues(rvecs[s], Rc);
                const double nz = Rc.at<double>(2, 0);   // 物体 X 轴(板法线)在相机系 Z 分量
                if (nz < 0.0) { Rpo = Rc; found = true; break; }
            }
            if (!found) cv::Rodrigues(rvecs[0], Rpo);
            const Mat3 Rcv = rotationVehicleToCamera(gp);
            Eigen::Matrix3d Rpo_e;
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j) Rpo_e(i, j) = Rpo.at<double>(i, j);
            const Mat3 Rz = Rcv.transpose() * Rpo_e;   // = Rz(θ + plate_yaw_off)
            theta_out = std::atan2(Rz(1, 0), Rz(0, 0)) - plate_yaw_off;
            return true;
#else
            (void)px; (void)K; (void)gp; (void)alpha; (void)R; (void)h; (void)L; (void)W;
            (void)plate_yaw_off; (void)theta_out;
            return false;
#endif
        }

        // 慢层一次收敛跑的记录
        struct SlowTrace { std::vector<std::array<double, 5>> rows; };  // window,t,alpha,R,info_R

        // 慢层递归收敛。mode: 0=四板联合, 1=单板(0号, 两块灯条), 2=提示词单灯条(a=0)。
        // use_hypothesis=true 时 R 用离散假设检验（提示词 4.6），否则用连续 refine。
        double runSlow(const SimulationConfig& cfg, const std::vector<double>& offs,
                       double a0_deg, double R0, int N, int mode, bool use_hypothesis,
                       const std::vector<double>& R_cands, double* a_out, SlowTrace* trace,
                       std::ofstream* csv)
        {
            FitConfig fc; fc.center_init = Vec3(0, 0, 3); fc.h_init = 0.10; fc.omega_init = 3.0;
            SlowCalib slow;
            slow.resetAlpha(a0_deg * M_PI / 180.0, std::pow(3.0 * M_PI / 180.0, 2));
            slow.resetR(R0, std::pow(0.03, 2));
            const char* mode_name = mode == 0 ? "multi" : (mode == 1 ? "plate" : "bar1");
            double prev_w = cfg.omega0, R_prior = R0;

            // 单灯条(a=0)走原始模拟器/拟合器；其余走多板路径。
            std::vector<SimulationFrame> fr1;
            std::vector<MultiFrame> frM;
            int nframes = 0;
            if (mode == 2) { fr1 = runSimulation(cfg, ""); nframes = static_cast<int>(fr1.size()); }
            else { frM = runMultiSimulation(cfg, ""); nframes = static_cast<int>(frM.size()); }

            for (int w0 = 0; w0 + N <= nframes; w0 += N)
            {
                FitConfig fcw = fc; fcw.omega_init = prev_w;
                FitResult fit; double a = slow.alpha(), R = slow.R(), ia = 0, iR = 0;
                double R_step = 0.0, t_mid = 0.0;
                if (mode == 2)
                {
                    std::vector<LightBarObservation> win;
                    for (int i = w0; i < w0 + N; ++i)
                    {
                        LightBarObservation o; o.timestamp = fr1[i].timestamp;
                        o.top_px = fr1[i].top_px; o.bottom_px = fr1[i].bottom_px;
                        win.push_back(o);
                    }
                    fit = fitTrajectory(win, cfg.K, cfg.gimbal_pitch, slow.alpha(), slow.R(),
                                        cfg.lightbar_len, fcw);
                    if (!fit.success) continue;
                    prev_w = fit.omega;
                    refineSlowParams(win, cfg.K, cfg.gimbal_pitch, cfg.lightbar_len, fit, a, R, ia, iR);
                    R_step = R - slow.R();
                    t_mid = fr1[std::min<int>(w0 + N, nframes - 1)].timestamp;
                }
                else
                {
                    auto win = barsFromFrames(frM, w0, w0 + N, cfg.plate_width, mode == 1 ? 0 : -1);
                    fit = fitTrajectoryJoint(win, cfg.K, cfg.gimbal_pitch, slow.alpha(), slow.R(),
                                             cfg.lightbar_len, offs, fcw);
                    if (!fit.success) continue;
                    prev_w = fit.omega;
                    FitResult fit_best = fit; double R_target = fit.R_used;
                    if (use_hypothesis)
                    {
                        double best_cost = std::numeric_limits<double>::infinity();
                        for (double cand : R_cands)
                        {
                            FitResult fcnd = fitTrajectoryJoint(win, cfg.K, cfg.gimbal_pitch,
                                                                slow.alpha(), cand, cfg.lightbar_len,
                                                                offs, fcw);
                            if (!fcnd.success) continue;
                            const double cost = fcnd.reprojection_rmse * fcnd.reprojection_rmse *
                                                    win.size() + std::pow((cand - R_prior) / 0.04, 2);
                            if (cost < best_cost) { best_cost = cost; R_target = cand; fit_best = fcnd; }
                        }
                    }
                    refineSlowParamsJoint(win, cfg.K, cfg.gimbal_pitch, cfg.lightbar_len, offs,
                                          fit_best, a, R, ia, iR);
                    R_step = use_hypothesis ? (R_target - slow.R()) : (R - slow.R());
                    t_mid = frM[std::min<int>(w0 + N, nframes - 1)].timestamp;
                }
                slow.update(a - slow.alpha(), R_step, ia, iR, 1.0);
                if (use_hypothesis && cfg.radius_step_time >= 0.0 && t_mid > cfg.radius_step_time &&
                    std::abs(slow.R() - R_prior) > 1.5 * 0.03)
                {
                    R_prior = slow.R();
                    slow.resetR(slow.R(), std::pow(0.03, 2));   // 车型切换：重新放大 R 不确定性
                }
                if (trace) trace->rows.push_back({double(w0 / N), t_mid, slow.alpha() * 180 / M_PI,
                                                  slow.R(), iR});
                if (csv) *csv << a0_deg << ',' << R0 << ',' << N << ',' << mode_name << ','
                              << w0 / N << ',' << t_mid << ',' << slow.alpha() * 180 / M_PI << ','
                              << slow.R() << ',' << ia << ',' << iR << '\n';
            }
            if (a_out) *a_out = slow.alpha() * 180 / M_PI;
            return slow.R();
        }
    }  // namespace

    void runMultiPlateExperiments()
    {
        std::cout << std::fixed << std::setprecision(4);
        std::cout << "\n=== 多装甲板联合 / 慢层收敛 / 车型切换 / PnP 对照 ===\n";

        SimulationConfig base;
        base.K.fx = 800; base.K.fy = 800; base.K.cx = 432; base.K.cy = 960;
        base.radius = 0.20;
        base.pitch_install = 15.0 * M_PI / 180.0;
        base.lightbar_len = 0.10; base.lightbar_h = 0.10;
        base.plate_width = 0.135;
        base.omega0 = 3.0; base.dt = 0.01;
        base.pixel_noise_std = 0.5;
        base.num_plates = 4;

        const double a15 = 15.0 * M_PI / 180.0;
        const std::vector<double> offs = plateOffsets(base.num_plates);

        FitConfig fc; fc.center_init = Vec3(0, 0, 3); fc.h_init = 0.10; fc.omega_init = 3.0;

        // ---------- E1: 同一联合拟合器，单板(0号) vs 四板 ----------
        {
            SimulationConfig c = base; c.num_frames = 40;
            auto fr = runMultiSimulation(c, "");
            auto all = barsFromFrames(fr, 0, static_cast<int>(fr.size()), c.plate_width, -1);
            auto one = barsFromFrames(fr, 0, static_cast<int>(fr.size()), c.plate_width, 0);
            auto f_all = fitTrajectoryJoint(all, c.K, c.gimbal_pitch, a15, 0.20, c.lightbar_len, offs, fc);
            auto f_one = fitTrajectoryJoint(one, c.K, c.gimbal_pitch, a15, 0.20, c.lightbar_len, offs, fc);
            const double t_end = fr.back().timestamp;
            auto rep = [&](const char* tag, const FitResult& f)
            {
                std::cout << "E1 " << tag << " yaw="
                          << std::abs(wrapDeg(f.yawAt(t_end) - fr.back().yaw_gt))
                          << "deg omega_err=" << std::abs(f.omega - fr.back().omega_gt)
                          << " center_cm=" << (f.centerAt(t_end, fr.front().timestamp) -
                                               fr.back().center_cam_gt).norm() * 100
                          << " rmse=" << f.reprojection_rmse << "\n";
            };
            rep("plate0-only ", f_one);
            rep("4-plate joint", f_all);
        }

        // ---------- E2a: 慢层收敛 vs 窗口长度 & 观测模型（bar1 / plate / multi）----------
        {
            std::ofstream csv("data/identifiability.csv");
            csv << "N,mode,alpha_end_deg,R_end\n";
            std::cout << "\nE2a 慢层递归收敛 (a0=13, R0=0.18, 1400帧, true 15/0.20):\n";
            for (int N : {20, 40, 80})
            {
                SimulationConfig c = base; c.num_frames = 1400;
                for (int mode = 0; mode < 3; ++mode)
                {
                    double a_end = 0;
                    const double R_end = runSlow(c, offs, 13.0, 0.18, N, mode, false, {}, &a_end,
                                                 nullptr, nullptr);
                    const char* mn = mode == 0 ? "multi" : (mode == 1 ? "plate" : "bar1");
                    std::cout << "  N=" << std::setw(3) << N << " " << std::setw(5) << mn
                              << " a0=13 R0=0.18 -> alpha="
                              << std::setw(7) << a_end << " R=" << std::setw(7) << R_end
                              << (std::abs(a_end - 15.0) < 1.0 && std::abs(R_end - 0.20) < 0.02 ? "  OK"
                                                                                                : "  --")
                              << "\n";
                    csv << N << ',' << mn << ',' << a_end << ',' << R_end << '\n';
                }
            }
            csv.close();
        }

        // ---------- E2b: 慢层递归收敛轨迹（多板联合）----------
        {
            std::ofstream csv("data/multi_slow.csv");
            csv << "init_alpha_deg,init_R,N,mode,window,t,alpha_deg,R,info_alpha,info_R\n";
            std::cout << "\nE2b 慢层递归收敛 (true alpha=15, R=0.20):\n";
            const std::vector<std::pair<double, double>> inits = {
                {13.0, 0.20}, {17.0, 0.20}, {15.0, 0.18}, {15.0, 0.22}};
            for (int N : {40, 80})
            {
                SimulationConfig c = base; c.num_frames = 1400;
                for (auto [a0, R0] : inits)
                {
                    double a_end = 0;
                    const double R_end = runSlow(c, offs, a0, R0, N, /*mode=*/0,
                                                 /*hyp=*/false, {}, &a_end, nullptr, &csv);
                    std::cout << "  N=" << std::setw(3) << N << " a0=" << a0 << " R0=" << R0
                              << " -> alpha=" << std::setw(7) << a_end << "deg R=" << std::setw(7)
                              << R_end << (std::abs(a_end - 15.0) < 1.0 && std::abs(R_end - 0.20) < 0.02
                                           ? "  OK" : "  --") << "\n";
                }
            }
            {
                SimulationConfig c = base; c.num_frames = 1400;
                double a_end = 0;
                const double R_end = runSlow(c, offs, 13.0, 0.20, 40, /*mode=*/2,
                                             /*hyp=*/false, {}, &a_end, nullptr, &csv);
                std::cout << "  单灯条(bar1) N=40 a0=13 R0=0.20 -> alpha=" << a_end << "deg R=" << R_end
                          << "\n";
            }
            csv.close();
        }

        // ---------- E3: 车型切换 R 0.20 -> 0.28（t>=5s），离散假设检验 ----------
        {
            std::ofstream csv("data/model_switch.csv");
            csv << "a0_deg,R0,N,mode,window,t,alpha_deg,R,info_alpha,info_R\n";
            std::vector<double> cands;
            for (double r = 0.16; r <= 0.321; r += 0.02) cands.push_back(r);
            std::cout << "\nE3 车型切换 R 0.20->0.28 @5s (离散假设检验):\n";
            for (int N : {40, 80})
            {
                SimulationConfig c = base; c.num_frames = 1400;
                c.radius_step_time = 5.0; c.radius_after = 0.28;
                SlowTrace tr; double a_end = 0;
                const double R_end = runSlow(c, offs, 15.0, 0.20, N, 0, true, cands, &a_end, &tr, &csv);
                // 切换前后各取平均
                double R_pre = 0, R_post = 0; int np = 0, nq = 0;
                for (auto& r : tr.rows)
                {
                    if (r[1] < 4.5) { R_pre += r[3]; ++np; }
                    if (r[1] > 7.0) { R_post += r[3]; ++nq; }
                }
                std::cout << "  N=" << N << " R_pre=" << (np ? R_pre / np : 0)
                          << " R_post=" << (nq ? R_post / nq : 0) << " final=" << R_end
                          << " (target 0.20 -> 0.28)\n";
            }
            // 诊断：切换后某窗口的 RMSE-vs-R（看 0.28 是否真的更优）
            {
                SimulationConfig c = base; c.num_frames = 1200;
                c.radius_step_time = 5.0; c.radius_after = 0.28;
                auto fr = runMultiSimulation(c, "");
                auto win = barsFromFrames(fr, 800, 840, c.plate_width, -1);
                std::cout << "  [诊断] t≈8s 窗口 RMSE-vs-R:";
                for (double cand : {0.20, 0.24, 0.28, 0.32})
                {
                    auto f = fitTrajectoryJoint(win, c.K, c.gimbal_pitch, a15, cand,
                                                c.lightbar_len, offs, fc);
                    std::cout << " R=" << cand << "->" << f.reprojection_rmse;
                }
                std::cout << " (真实 R=0.28, 重投影应最小)\n";
            }
            csv.close();
        }

        // ---------- E4: 正面可观测性退化 d(u_top-u_bot)/dθ 随 Cz（提示词 4.4）----------
        {
            std::ofstream csv("data/observability.csv");
            csv << "Cz,alpha_deg,d_deltaU_dtheta_px_per_rad\n";
            std::cout << "\nE4 正面差分可观测性 d(u_top-u_bot)/dθ 随 Cz (R=0.20, L=0.10):\n";
            const double dth = 1e-4;
            auto dDelta = [&](double Cz, double alpha)
            {
                Vec3 t0, b0, t1, b1;
                barEndpointsCamera(Vec3(0, 0, Cz), 0.10, -0.5 * dth, alpha, 0.20, 0.10, 0.0, t0, b0);
                barEndpointsCamera(Vec3(0, 0, Cz), 0.10, +0.5 * dth, alpha, 0.20, 0.10, 0.0, t1, b1);
                const Vec2 e0 = project(t0, base.K), e1 = project(b0, base.K);
                const Vec2 f0 = project(t1, base.K), f1 = project(b1, base.K);
                return ((f0.x() - f1.x()) - (e0.x() - e1.x())) / dth;
            };
            for (double Cz : {0.30, 0.40, 0.50, 0.60, 0.80, 1.0, 2.0, 3.0, 5.0})
            {
                const double d15 = dDelta(Cz, a15), d0 = dDelta(Cz, 0.0);
                std::cout << "  Cz=" << Cz << (std::abs(Cz - 0.40) < 1e-9 ? "(=2R)" : "     ")
                          << " alpha=15deg: " << std::setw(10) << d15 << " | alpha=0: " << std::setw(10)
                          << d0 << "\n";
                csv << Cz << ",15," << d15 << '\n' << Cz << ",0," << d0 << '\n';
            }
            csv.close();
        }

        // ---------- E5: 单帧 PnP vs 多帧拟合 Yaw 标准差 ----------
        {
            std::ofstream csv("data/pnp_compare.csv");
            csv << "t,plate,cosang,yaw_pnp_err_deg,yaw_pnp_branch_deg,yaw_fit_err_deg\n";
            SimulationConfig c = base; c.num_frames = 300;
            auto fr = runMultiSimulation(c, "");
            auto win = barsFromFrames(fr, 0, static_cast<int>(fr.size()), c.plate_width, -1);
            auto fit = fitTrajectoryJoint(win, c.K, c.gimbal_pitch, a15, 0.20, c.lightbar_len, offs, fc);
            std::vector<double> pe, peb, pf, pe_f, peb_f, pf_f;
            int flips = 0;
            for (const auto& mf : fr)
            {
                const double yaw_fit = fit.yawAt(mf.timestamp);
                for (const auto& p : mf.plates)
                {
                    const double theta_k = mf.yaw_gt + p.armor_id * 2.0 * M_PI / c.num_plates;
                    const Vec3 n_v(std::cos(theta_k), std::sin(theta_k), 0.0);
                    const Mat3 rcv = rotationVehicleToCamera(mf.gimbal_pitch_gt);
                    const Vec3 n_c = rcv * n_v;
                    const Vec3 pc = rcv * (mf.radius_gt * n_v) + mf.center_cam_gt;
                    const double cosang = n_c.dot(-pc) / (n_c.norm() * pc.norm());
                    if (cosang < std::cos(70.0 * M_PI / 180.0)) continue;  // 只统计朝向相机的板
                    double theta_pnp = 0.0;
                    if (!pnpTheta(p.corners_px, c.K, mf.gimbal_pitch_gt, a15, 0.20, c.lightbar_h,
                                  c.lightbar_len, c.plate_width, p.armor_id * 2.0 * M_PI / c.num_plates,
                                  theta_pnp))
                        continue;
                    const double e_pnp = std::abs(wrapDeg(theta_pnp - mf.yaw_gt));
                    const double e_pnp_b = std::min(e_pnp, 180.0 - e_pnp);  // 解掉 ±180° 分支歧义
                    const double e_fit = std::abs(wrapDeg(yaw_fit - mf.yaw_gt));
                    if (e_pnp > 90.0) ++flips;
                    pe.push_back(e_pnp); peb.push_back(e_pnp_b); pf.push_back(e_fit);
                    if (cosang > std::cos(25.0 * M_PI / 180.0))
                    { pe_f.push_back(e_pnp); peb_f.push_back(e_pnp_b); pf_f.push_back(e_fit); }
                    csv << mf.timestamp << ',' << p.armor_id << ',' << cosang << ',' << e_pnp << ','
                        << e_pnp_b << ',' << e_fit << '\n';
                }
            }
            std::cout << "\nE5 单帧 PnP vs 多帧拟合 yaw 误差 (deg):\n";
            std::cout << "  PnP raw        std=" << stdDeg(pe) << " meanAbs=" << meanAbs(pe)
                      << " | frontal std=" << stdDeg(pe_f) << " meanAbs=" << meanAbs(pe_f)
                      << " | 分支翻转 " << (pe.empty() ? 0 : 100 * flips / pe.size()) << "%\n";
            std::cout << "  PnP 解分支后   std=" << stdDeg(peb) << " meanAbs=" << meanAbs(peb)
                      << " | frontal std=" << stdDeg(peb_f) << " meanAbs=" << meanAbs(peb_f) << "\n";
            std::cout << "  Fit   std=" << stdDeg(pf) << " meanAbs=" << meanAbs(pf)
                      << " | frontal std=" << stdDeg(pf_f) << " meanAbs=" << meanAbs(pf_f) << "\n";
            csv.close();
        }
    }
}  // namespace rm
