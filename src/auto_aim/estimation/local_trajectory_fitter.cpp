#include "estimation/local_trajectory_fitter.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>

#include <ceres/ceres.h>

namespace auto_aim
{
    namespace
    {
        /// @brief 这条拟合的**参考板号**：观测里出现最多的那个 armor_id
        ///        （多板联合时 = 参考弧那条；单板时 = 那条弧的板号；没有合法 id → -1）。
        ///        只用于诊断与调用方的相位门控，不参与拟合本身。
        inline int referenceArmorId(const std::vector<LightBarObservation>& obs)
        {
            int count[4] = {0, 0, 0, 0};
            for (const auto& o : obs)
                if (o.armor_id >= 0 && o.armor_id < 4) ++count[o.armor_id];
            int best = -1;
            int best_n = 0;
            for (int k = 0; k < 4; ++k)
                if (count[k] > best_n) { best_n = count[k]; best = k; }
            return best_n > 0 ? best : -1;
        }

        // 车辆系→相机系。默认 R_cv = Rx(gimbal_pitch + pitch_offset)·R_base，
        // R_base: X_v→Z_c, Y_v→X_c, Z_v→-Y_c（与 experiments/local_fitter 一致）。
        // use_rcv 时直接用给定的完整 3×3 R_cv（含"相机→云台安装旋转"，实车必须）。
        template <typename T>
        inline void vehicleToCamera(const T* pv, T gp, bool use_rcv, const double* R,
                                    T& X, T& Y, T& Z)
        {
            if (use_rcv)
            {
                X = T(R[0]) * pv[0] + T(R[1]) * pv[1] + T(R[2]) * pv[2];
                Y = T(R[3]) * pv[0] + T(R[4]) * pv[1] + T(R[5]) * pv[2];
                Z = T(R[6]) * pv[0] + T(R[7]) * pv[1] + T(R[8]) * pv[2];
                return;
            }
            const T s = ceres::sin(gp), c = ceres::cos(gp);
            X = pv[1];
            Y = -s * pv[0] - c * pv[2];
            Z = c * pv[0] - s * pv[2];
        }

        // 灯条端点（车辆系）：板面 n=(cosθ,sinθ,0)、切向 t=(-sinθ,cosθ,0)；
        // 角点 = ρ·n + a·t + z·ẑ，ρ = R − b·sinα，z = h + b·cosα。
        template <typename T>
        inline void barEndpointV(T h, T theta, T alpha, T R, T L, T a, T b, T* pv)
        {
            const T rho = R - b * ceres::sin(alpha);
            const T z = h + b * ceres::cos(alpha);
            const T ct = ceres::cos(theta), st = ceres::sin(theta);
            pv[0] = rho * ct - a * st;
            pv[1] = rho * st + a * ct;
            pv[2] = z;
        }

        struct FastResidual
        {
            FastResidual(const LocalFitIntrinsics& K, double gp, double alpha, double R, double L,
                         double a, double yaw_off, double t, double t0, double tu, double tv,
                         double bu, double bv, bool use_beta, bool use_rcv,
                         const std::array<double, 9>& rcv)
                : K_(K), gp_(gp), alpha_(alpha), R_(R), L_(L), a_(a), yaw_off_(yaw_off), t_(t),
                  t0_(t0), tu_(tu), tv_(tv), bu_(bu), bv_(bv), use_beta_(use_beta),
                  use_rcv_(use_rcv), rcv_(rcv) {}

            template <typename T>
            bool operator()(const T* const center, const T* const h_ptr, const T* const phase,
                            const T* const beta, const T* const vel, T* residual) const
            {
                const T dt = T(t_ - t0_);
                const T theta = phase[0] + phase[1] * dt +
                    (use_beta_ ? T(0.5) * beta[0] * dt * dt : T(0.0)) + T(yaw_off_);
                T pv[3], pc[3];
                barEndpointV<T>(h_ptr[0], theta, T(alpha_), T(R_), T(L_), T(a_), T(0.5) * T(L_),
                                pv);
                vehicleToCamera<T>(pv, T(gp_), use_rcv_, rcv_.data(), pc[0], pc[1], pc[2]);
                residual[0] = T(K_.fx) * (pc[0] + center[0] + vel[0] * dt) /
                                  (pc[2] + center[2] + vel[2] * dt) + T(K_.cx) - T(tu_);
                residual[1] = T(K_.fy) * (pc[1] + center[1] + vel[1] * dt) /
                                  (pc[2] + center[2] + vel[2] * dt) + T(K_.cy) - T(tv_);
                barEndpointV<T>(h_ptr[0], theta, T(alpha_), T(R_), T(L_), T(a_), T(-0.5) * T(L_),
                                pv);
                vehicleToCamera<T>(pv, T(gp_), use_rcv_, rcv_.data(), pc[0], pc[1], pc[2]);
                residual[2] = T(K_.fx) * (pc[0] + center[0] + vel[0] * dt) /
                                  (pc[2] + center[2] + vel[2] * dt) + T(K_.cx) - T(bu_);
                residual[3] = T(K_.fy) * (pc[1] + center[1] + vel[1] * dt) /
                                  (pc[2] + center[2] + vel[2] * dt) + T(K_.cy) - T(bv_);
                return true;
            }
            LocalFitIntrinsics K_;
            double gp_, alpha_, R_, L_, a_, yaw_off_, t_, t0_, tu_, tv_, bu_, bv_;
            bool use_beta_, use_rcv_;
            std::array<double, 9> rcv_;
        };

        // h 的慢层先验：r = (h − h_prior)/σ。防止快层把 Cz 的误差吸进 h。
        struct HeightPrior
        {
            explicit HeightPrior(double prior, double sigma) : prior_(prior), inv_(1.0 / sigma) {}
            template <typename T>
            bool operator()(const T* const h, T* residual) const
            {
                residual[0] = (h[0] - T(prior_)) * T(inv_);
                return true;
            }
            double prior_, inv_;
        };

        struct VelPrior
        {
            explicit VelPrior(double sigma) : inv_(1.0 / sigma) {}
            template <typename T>
            bool operator()(const T* const vel, T* residual) const
            {
                residual[0] = vel[0] * T(inv_);
                residual[1] = vel[1] * T(inv_);
                residual[2] = vel[2] * T(inv_);
                return true;
            }
            double inv_;
        };

        inline double projectU(const LocalFitIntrinsics& K, const double* pc) { return K.fx * pc[0] / pc[2] + K.cx; }
        inline double projectV(const LocalFitIntrinsics& K, const double* pc) { return K.fy * pc[1] / pc[2] + K.cy; }

        // 有效性门限（像素）：绝对像素保底 + 随焦距放大的角度门限。
        // 同一几何误差下像素残差 ∝ f：仿真 f≈1444、实车 f≈2390，只用像素门限会把
        // 实车的好拟合全判无效。角度门限让它与分辨率无关。
        inline double rmseGatePx(const LocalTrajectoryFitConfig& cfg, double /*rmse_px*/)
        {
            const double angle_gate = cfg.K.fx > 1.0
                ? cfg.K.fx * std::tan(cfg.max_rmse_deg * 3.14159265358979323846 / 180.0)
                : 0.0;
            return std::max(cfg.max_rmse_px, angle_gate);
        }

        // 质量（0~1）：用同一个"有效尺度"归一，避免实车因像素残差天然更大而被压成 0。
        inline double qualityFromRmse(const LocalTrajectoryFitConfig& cfg, double rmse_px)
        {
            const double scale = cfg.huber_delta * (rmseGatePx(cfg, rmse_px) /
                                                    std::max(1e-3, cfg.max_rmse_px));
            return std::exp(-rmse_px / std::max(0.1, scale));
        }

        /// 这条弧在窗口内覆盖的相位跨度（rad）。yaw 能不能被定住看的是它，而不是 rmse。
        inline double phaseSpanRad(
            const LocalTrajectoryFitter::FastFit& f,
            const std::vector<LightBarObservation>& obs)
        {
            if (obs.size() < 2) return 0.0;
            const double dt = obs.back().timestamp - obs.front().timestamp;
            return std::abs(f.omega * dt + 0.5 * f.beta * dt * dt);
        }

        void endpointPc(const LocalTrajectoryFitConfig& cfg, double alpha, double R, double h, double theta,
                        double a, double b, const double* center, const double* drift, double dt,
                        double* pc)
        {
            double pv[3];
            barEndpointV<double>(h, theta, alpha, R, cfg.lightbar_len, a, b, pv);
            double x, y, z;
            vehicleToCamera<double>(pv, cfg.gimbal_pitch + cfg.pitch_offset, cfg.use_rcv,
                                    cfg.rcv.data(), x, y, z);
            pc[0] = x + center[0] + drift[0] * dt;
            pc[1] = y + center[1] + drift[1] * dt;
            pc[2] = z + center[2] + drift[2] * dt;
        }

        // 固定快层参数、对 (α,R) 的残差向量（每观测 4 个）。
        void slowResiduals(const std::vector<LightBarObservation>& obs, const LocalTrajectoryFitConfig& cfg,
                           const LocalTrajectoryFitter::FastFit& f, double alpha, double R,
                           std::vector<double>& out)
        {
            out.clear();
            out.reserve(obs.size() * 4);
            for (const auto& o : obs)
            {
                if (o.armor_id < 0 || o.armor_id >= static_cast<int>(cfg.plate_yaw_offsets.size()))
                    continue;
                const double yaw_off = cfg.plate_yaw_offsets[o.armor_id];
                const double theta = f.yawAt(o.timestamp) + yaw_off;
                const double dt = o.timestamp - f.t0;
                double pc[3];
                endpointPc(cfg, alpha, R, f.h, theta, o.bar_offset, +0.5 * cfg.lightbar_len,
                           f.center, f.drift, dt, pc);
                out.push_back(projectU(cfg.K, pc) - o.u_top);
                out.push_back(projectV(cfg.K, pc) - o.v_top);
                endpointPc(cfg, alpha, R, f.h, theta, o.bar_offset, -0.5 * cfg.lightbar_len,
                           f.center, f.drift, dt, pc);
                out.push_back(projectU(cfg.K, pc) - o.u_bot);
                out.push_back(projectV(cfg.K, pc) - o.v_bot);
            }
        }
    }  // namespace

    bool LocalTrajectoryFitter::fitFast(const std::vector<LightBarObservation>& observations,
                                        double alpha_fixed, double R_fixed, FastFit& out) const
    {
        out = FastFit{};
        if (observations.size() < 4) return false;
        const auto& cfg = config_;
        const bool use_beta = cfg.omega_model_order >= 2;
        const double t0 = observations.front().timestamp;

        double best_cost = std::numeric_limits<double>::infinity();
        const auto multi_start_begin = std::chrono::steady_clock::now();
        // 多起点：ω 的符号/倍率 + 初相相位（相位初值若差 ~90° 会掉进"镜像/旋转"坏局部解）。
        for (double ph0 : { 0.0, 0.5 * M_PI, M_PI, -0.5 * M_PI })
        for (double w0 : { cfg.omega_init, -cfg.omega_init, 2.0 * cfg.omega_init,
                           -2.0 * cfg.omega_init })
        {
            double center[3] = { cfg.center_init[0], cfg.center_init[1], cfg.center_init[2] };
            double h = cfg.h_init;
            double phase[2] = { ph0, w0 };
            double beta = 0.0;
            double vel[3] = { 0.0, 0.0, 0.0 };
            ceres::Problem problem;
            for (const auto& o : observations)
            {
                if (o.armor_id < 0 || o.armor_id >= static_cast<int>(cfg.plate_yaw_offsets.size()))
                    continue;
                auto* cost = new ceres::AutoDiffCostFunction<FastResidual, 4, 3, 1, 2, 1, 3>(
                    new FastResidual(cfg.K, cfg.gimbal_pitch + cfg.pitch_offset, alpha_fixed, R_fixed,
                                     cfg.lightbar_len, o.bar_offset,
                                     cfg.plate_yaw_offsets[o.armor_id], o.timestamp, t0, o.u_top,
                                     o.v_top, o.u_bot, o.v_bot, use_beta, cfg.use_rcv, cfg.rcv));
                problem.AddResidualBlock(cost,
                    cfg.use_huber ? new ceres::HuberLoss(cfg.huber_delta) : nullptr, center, &h,
                    phase, &beta, vel);
            }
            if (problem.NumResidualBlocks() < 2) return false;
            if (!use_beta) problem.SetParameterBlockConstant(&beta);
            problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<VelPrior, 3, 3>(new VelPrior(0.6)), nullptr, vel);
            // h 的慢层先验（见 HeightPrior 注释）：把"板心高度是机械常数"显式加进残差。
            if (cfg.h_prior_sigma > 1e-6) {
                problem.AddResidualBlock(
                    new ceres::AutoDiffCostFunction<HeightPrior, 1, 1>(
                        new HeightPrior(cfg.h_prior, cfg.h_prior_sigma)),
                    nullptr, &h);
            }
            problem.SetParameterLowerBound(&h, 0, -2.0);
            problem.SetParameterUpperBound(&h, 0, 2.0);
            // θ0 只在 cos/sin 里出现 → 模 2π 等价，限定到 [-π,π] 即可（防优化器沿
            // 无穷方向漂走）。yaw_off 是常数，会被 θ0 吸收。
            problem.SetParameterLowerBound(phase, 0, -M_PI);
            problem.SetParameterUpperBound(phase, 0, M_PI);
            problem.SetParameterLowerBound(phase, 1, -30.0);
            problem.SetParameterUpperBound(phase, 1, 30.0);
            for (int i = 0; i < 3; ++i)
            {
                problem.SetParameterLowerBound(vel, i, -3.0);
                problem.SetParameterUpperBound(vel, i, 3.0);
            }
            // 总时间预算：超了就不再试新初值（NUC 上多起点会把单次拟合拖到几秒）。
            if (std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                              multi_start_begin).count() >
                    cfg.multi_start_budget_s &&
                std::isfinite(best_cost)) {
                break;
            }
            ceres::Solver::Options opt;
            opt.linear_solver_type = ceres::DENSE_QR;
            opt.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
            opt.max_num_iterations = cfg.max_iterations;
            opt.max_solver_time_in_seconds = std::max(0.005, cfg.max_solver_time_s);
            opt.minimizer_progress_to_stdout = false;
            ceres::Solver::Summary summary;
            ceres::Solve(opt, &problem, &summary);
            if (summary.final_cost < best_cost)
            {
                best_cost = summary.final_cost;
                out.center[0] = center[0]; out.center[1] = center[1]; out.center[2] = center[2];
                out.h = h;
                out.theta0 = phase[0];
                out.omega = phase[1];
                out.beta = use_beta ? beta : 0.0;
                out.drift[0] = vel[0]; out.drift[1] = vel[1]; out.drift[2] = vel[2];
                out.t0 = t0;
            }
        }
        // RMSE
        std::vector<double> res;
        slowResiduals(observations, cfg, out, alpha_fixed, R_fixed, res);
        double sse = 0.0;
        for (double r : res) sse += r * r;
        out.rmse_px = res.empty() ? std::numeric_limits<double>::infinity()
                                  : std::sqrt(sse / res.size());
        out.valid = std::isfinite(out.rmse_px);
        return out.valid;
    }

    bool LocalTrajectoryFitter::fit(const std::vector<LightBarObservation>& observations,
                                    double alpha_fixed, double R_fixed, LocalFitResult& out) const
    {
        out = LocalFitResult{};
        FastFit f;
        if (!fitFast(observations, alpha_fixed, R_fixed, f)) return false;
        const double rmse_gate = rmseGatePx(config_, f.rmse_px);
        out.valid = f.valid && f.rmse_px <= rmse_gate;
        out.timestamp = observations.back().timestamp;
        out.center[0] = f.center[0]; out.center[1] = f.center[1]; out.center[2] = f.center[2];
        out.yaw = f.yawAt(observations.back().timestamp);
        out.omega = f.omega;
        out.ref_armor_id = referenceArmorId(observations);
        out.alpha = alpha_fixed;
        out.R = R_fixed;
        out.h = f.h;
        for (int i = 0; i < 3; ++i) out.drift[i] = f.drift[i];
        out.reprojection_rmse_px = f.rmse_px;
        out.phase_span_rad = phaseSpanRad(f, observations);
        out.quality = qualityFromRmse(config_, f.rmse_px);
        out.num_bars = static_cast<int>(observations.size());
        return out.valid;
    }

    void LocalTrajectoryFitter::refineSlow(const std::vector<LightBarObservation>& observations,
                                           const FastFit& fast, double& alpha, double& R,
                                           double& info_alpha, double& info_R) const
    {
        info_alpha = 0.0; info_R = 0.0;
        if (observations.empty() || !fast.valid) return;
        const auto& cfg = config_;
        const double hd = std::max(0.5, cfg.huber_delta);
        double a = std::clamp(alpha, cfg.alpha_lo, cfg.alpha_hi);
        double r = std::clamp(R, cfg.R_lo, cfg.R_hi);
        std::vector<double> res, res_a, res_r;
        for (int it = 0; it < 8; ++it)
        {
            slowResiduals(observations, cfg, fast, a, r, res);
            if (res.empty()) return;
            const double ha = 1e-6, hr = 1e-6;
            slowResiduals(observations, cfg, fast, std::min(a + ha, cfg.alpha_hi), r, res_a);
            slowResiduals(observations, cfg, fast, a, std::min(r + hr, cfg.R_hi), res_r);
            if (res_a.size() != res.size() || res_r.size() != res.size()) return;
            double Haa = 0, Hrr = 0, Har = 0, ga = 0, gr = 0;
            for (size_t i = 0; i < res.size(); ++i)
            {
                const double rr = res[i];
                const double w = std::abs(rr) > hd ? hd / std::abs(rr) : 1.0;
                const double ja = (res_a[i] - rr) / ha, jr = (res_r[i] - rr) / hr;
                Haa += w * ja * ja; Hrr += w * jr * jr; Har += w * ja * jr;
                ga += w * ja * rr; gr += w * jr * rr;
            }
            info_alpha = Haa; info_R = Hrr;
            const double lam = 1e-9 * (Haa + Hrr) + 1e-12;
            const double det = (Haa + lam) * (Hrr + lam) - Har * Har;
            if (!std::isfinite(det) || std::abs(det) < 1e-18) break;
            double da = (-ga * (Hrr + lam) + gr * Har) / det;
            double dr = (-gr * (Haa + lam) + ga * Har) / det;
            const double step = std::hypot(da, dr);
            if (!std::isfinite(step)) break;
            if (step > 0.2) { da *= 0.2 / step; dr *= 0.2 / step; }
            if (std::abs(da) < 1e-12 && std::abs(dr) < 1e-12) break;
            a = std::clamp(a + da, cfg.alpha_lo, cfg.alpha_hi);
            r = std::clamp(r + dr, cfg.R_lo, cfg.R_hi);
        }
        alpha = a; R = r;
    }

    bool LocalTrajectoryFitter::fitWithSlow(const std::vector<LightBarObservation>& observations,
                                            LocalFitResult& out)
    {
        out = LocalFitResult{};
        if (observations.size() < 4) return false;
        FastFit fast;
        if (!fitFast(observations, slow_alpha_, slow_R_, fast)) return false;

        // 慢层一步：α 固定，只放开 R（确定性 Gauss-Newton）。
        double a = slow_alpha_, r = slow_R_, info_a = 0.0, info_r = 0.0;
        refineSlow(observations, fast, a, r, info_a, info_r);
        const double Cz = fast.center[2];
        const bool observable = std::isfinite(r) && fast.rmse_px <= rmseGatePx(config_, fast.rmse_px) &&
                                std::abs(Cz - 2.0 * slow_R_) > config_.R_freeze_band &&
                                info_r > config_.R_info_min;
        if (observable)
        {
            const double step = std::clamp(r - slow_R_, -config_.R_step_max, config_.R_step_max);
            // GN 步 + 弱先验回拉：R 只在"可观测且拟合可信"时小步走，并被先验轻轻拉住，
            // 避免 R–Cz 耦合导致 R 慢漂（实测 6.5m 无回拉会漂到 0.28）。
            slow_R_ += config_.R_learn * step;
            slow_R_ += config_.R_prior_gain * (config_.R_prior - slow_R_);
            slow_R_ = std::clamp(slow_R_, config_.R_lo, config_.R_hi);
        }

        // 用更新后的 (α,R) 重算快层，作为输出（外推到窗口末端时间）。
        FastFit f;
        if (!fitFast(observations, slow_alpha_, slow_R_, f)) return false;
        out.valid = f.valid && f.rmse_px <= rmseGatePx(config_, f.rmse_px);
        out.timestamp = observations.back().timestamp;
        out.center[0] = f.center[0]; out.center[1] = f.center[1]; out.center[2] = f.center[2];
        out.yaw = f.yawAt(observations.back().timestamp);
        out.omega = f.omega;
        out.ref_armor_id = referenceArmorId(observations);
        out.alpha = slow_alpha_;
        out.R = slow_R_;
        out.h = f.h;
        for (int i = 0; i < 3; ++i) out.drift[i] = f.drift[i];
        out.reprojection_rmse_px = f.rmse_px;
        out.phase_span_rad = phaseSpanRad(f, observations);
        out.quality = qualityFromRmse(config_, f.rmse_px);
        out.num_bars = static_cast<int>(observations.size());
        return out.valid;
    }
}  // namespace auto_aim
