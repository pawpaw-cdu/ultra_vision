#include "trajectory_fitter.h"

#include <ceres/ceres.h>
#include <algorithm>
#include <cmath>
#include <iostream>

namespace rm
{
    namespace
    {
        // 参数：center(3), h(1), phase(2)=[theta0, omega], beta(1), vel(3)；
        // α,R,L,gimbal_pitch 为常量。C(t)=center+vel·(t-t0) 处理中心漂移。
        struct FastResidual
        {
            FastResidual(const CameraIntrinsics& K, double gimbal_pitch, double alpha, double R,
                         double L, double t, double t0, const Vec2& top, const Vec2& bottom,
                         bool use_beta)
                : K_(K), gp_(gimbal_pitch), alpha_(alpha), R_(R), L_(L), t_(t), t0_(t0),
                  top_(top), bottom_(bottom), use_beta_(use_beta) {}

            template <typename T>
            bool operator()(const T* const center, const T* const h_ptr, const T* const phase,
                            const T* const beta, const T* const vel, T* residual) const
            {
                const T theta = phase[0] + phase[1] * T(t_) +
                    (use_beta_ ? T(0.5) * beta[0] * T(t_) * T(t_) : T(0.0));
                const T dt = T(t_ - t0_);
                const T cx = center[0] + vel[0] * dt;
                const T cy = center[1] + vel[1] * dt;
                const T cz = center[2] + vel[2] * dt;
                const T sa = ceres::sin(T(alpha_)), ca = ceres::cos(T(alpha_));
                const T r_top = T(R_) - T(0.5) * T(L_) * sa;
                const T r_bot = T(R_) + T(0.5) * T(L_) * sa;
                const T h_top = h_ptr[0] + T(0.5) * T(L_) * ca;
                const T h_bot = h_ptr[0] - T(0.5) * T(L_) * ca;
                const T ct = ceres::cos(theta), st = ceres::sin(theta);
                const T pv_top[3] = { r_top * ct, r_top * st, h_top };
                const T pv_bot[3] = { r_bot * ct, r_bot * st, h_bot };
                // Rcv = [[0,1,0],[-s,0,-c],[c,0,-s]]
                const T s = ceres::sin(T(gp_)), c = ceres::cos(T(gp_));
                auto pc = [&](const T* pv, T* out) {
                    out[0] = pv[1] + cx;
                    out[1] = -s * pv[0] - c * pv[2] + cy;
                    out[2] = c * pv[0] - s * pv[2] + cz;
                };
                T pt[3], pb[3];
                pc(pv_top, pt);
                pc(pv_bot, pb);
                residual[0] = T(K_.fx) * pt[0] / pt[2] + T(K_.cx) - T(top_.x());
                residual[1] = T(K_.fy) * pt[1] / pt[2] + T(K_.cy) - T(top_.y());
                residual[2] = T(K_.fx) * pb[0] / pb[2] + T(K_.cx) - T(bottom_.x());
                residual[3] = T(K_.fy) * pb[1] / pb[2] + T(K_.cy) - T(bottom_.y());
                return true;
            }

            CameraIntrinsics K_;
            double gp_, alpha_, R_, L_, t_, t0_;
            Vec2 top_, bottom_;
            bool use_beta_;
        };

        // 慢层：只有 α、R 是参数；快层最优参数作常量。
        // 速度弱先验：无漂移时把 vel 拉回 0，避免它去"吃掉" ω（R-Cz/ω-漂移 耦合）。
        struct VelPrior
        {
            explicit VelPrior(double sigma) : inv_sigma_(1.0 / sigma) {}
            template <typename T>
            bool operator()(const T* const vel, T* residual) const
            {
                residual[0] = vel[0] * T(inv_sigma_);
                residual[1] = vel[1] * T(inv_sigma_);
                residual[2] = vel[2] * T(inv_sigma_);
                return true;
            }
            double inv_sigma_;
        };
        struct SlowResidual
        {
            SlowResidual(const CameraIntrinsics& K, double gp, double L, double t,
                         const Vec3& C, double h, double theta0, double omega, double beta,
                         const Vec2& top, const Vec2& bottom)
                : K_(K), gp_(gp), L_(L), t_(t), C_(C), h_(h), theta0_(theta0),
                  omega_(omega), beta_(beta), top_(top), bottom_(bottom) {}

            template <typename T>
            bool operator()(const T* const alpha, const T* const R, T* residual) const
            {
                const T theta = T(theta0_) + T(omega_) * T(t_) + T(0.5) * T(beta_) * T(t_) * T(t_);
                const T sa = ceres::sin(alpha[0]), ca = ceres::cos(alpha[0]);
                const T r_top = R[0] - T(0.5) * T(L_) * sa;
                const T r_bot = R[0] + T(0.5) * T(L_) * sa;
                const T h_top = T(h_) + T(0.5) * T(L_) * ca;
                const T h_bot = T(h_) - T(0.5) * T(L_) * ca;
                const T ct = ceres::cos(theta), st = ceres::sin(theta);
                const T pv_top[3] = { r_top * ct, r_top * st, h_top };
                const T pv_bot[3] = { r_bot * ct, r_bot * st, h_bot };
                const T s = ceres::sin(T(gp_)), c = ceres::cos(T(gp_));
                auto pc = [&](const T* pv, T* out) {
                    out[0] = pv[1] + T(C_.x());
                    out[1] = -s * pv[0] - c * pv[2] + T(C_.y());
                    out[2] = c * pv[0] - s * pv[2] + T(C_.z());
                };
                T pt[3], pb[3];
                pc(pv_top, pt);
                pc(pv_bot, pb);
                residual[0] = T(K_.fx) * pt[0] / pt[2] + T(K_.cx) - T(top_.x());
                residual[1] = T(K_.fy) * pt[1] / pt[2] + T(K_.cy) - T(top_.y());
                residual[2] = T(K_.fx) * pb[0] / pb[2] + T(K_.cx) - T(bottom_.x());
                residual[3] = T(K_.fy) * pb[1] / pb[2] + T(K_.cy) - T(bottom_.y());
                return true;
            }
            CameraIntrinsics K_;
            double gp_, L_, t_;
            Vec3 C_;
            double h_, theta0_, omega_, beta_;
            Vec2 top_, bottom_;
        };

        double rmseOf(const std::vector<LightBarObservation>& obs, const CameraIntrinsics& K,
                      double gp, double alpha, double R, double L, const FitResult& f)
        {
            double sum = 0.0; int n = 0;
            const double t0 = obs.empty() ? 0.0 : obs.front().timestamp;
            for (const auto& o : obs)
            {
                Vec3 top, bot;
                barEndpointsCamera(f.centerAt(o.timestamp, t0), f.h, f.yawAt(o.timestamp), alpha, R,
                                   L, gp, top, bot);
                const Vec2 pt = project(top, K), pb = project(bot, K);
                sum += (pt - o.top_px).squaredNorm() + (pb - o.bottom_px).squaredNorm();
                n += 4;
            }
            return n > 0 ? std::sqrt(sum / n) : 0.0;
        }

        // 通用灯条残差：含板 yaw 偏移 yaw_off 与灯条切向偏移 a。
        // a=0 且 yaw_off=0 时与 FastResidual 等价。
        struct FastResidualJoint
        {
            FastResidualJoint(const CameraIntrinsics& K, double gp, double alpha, double R,
                              double L, double a, double yaw_off, double t, double t0,
                              const Vec2& top, const Vec2& bottom, bool use_beta)
                : K_(K), gp_(gp), alpha_(alpha), R_(R), L_(L), a_(a), yaw_off_(yaw_off),
                  t_(t), t0_(t0), top_(top), bottom_(bottom), use_beta_(use_beta) {}

            template <typename T>
            bool operator()(const T* const center, const T* const h_ptr, const T* const phase,
                            const T* const beta, const T* const vel, T* residual) const
            {
                const T theta = phase[0] + phase[1] * T(t_) +
                    (use_beta_ ? T(0.5) * beta[0] * T(t_) * T(t_) : T(0.0)) + T(yaw_off_);
                const T dt = T(t_ - t0_);
                const T cx = center[0] + vel[0] * dt;
                const T cy = center[1] + vel[1] * dt;
                const T cz = center[2] + vel[2] * dt;
                const T sa = ceres::sin(T(alpha_)), ca = ceres::cos(T(alpha_));
                const T r_top = T(R_) - T(0.5) * T(L_) * sa;
                const T r_bot = T(R_) + T(0.5) * T(L_) * sa;
                const T h_top = h_ptr[0] + T(0.5) * T(L_) * ca;
                const T h_bot = h_ptr[0] - T(0.5) * T(L_) * ca;
                const T ct = ceres::cos(theta), st = ceres::sin(theta);
                const T aa = T(a_);
                const T pv_top[3] = { r_top * ct - aa * st, r_top * st + aa * ct, h_top };
                const T pv_bot[3] = { r_bot * ct - aa * st, r_bot * st + aa * ct, h_bot };
                const T s = ceres::sin(T(gp_)), c = ceres::cos(T(gp_));
                auto pc = [&](const T* pv, T* out) {
                    out[0] = pv[1] + cx;
                    out[1] = -s * pv[0] - c * pv[2] + cy;
                    out[2] = c * pv[0] - s * pv[2] + cz;
                };
                T pt[3], pb[3];
                pc(pv_top, pt);
                pc(pv_bot, pb);
                residual[0] = T(K_.fx) * pt[0] / pt[2] + T(K_.cx) - T(top_.x());
                residual[1] = T(K_.fy) * pt[1] / pt[2] + T(K_.cy) - T(top_.y());
                residual[2] = T(K_.fx) * pb[0] / pb[2] + T(K_.cx) - T(bottom_.x());
                residual[3] = T(K_.fy) * pb[1] / pb[2] + T(K_.cy) - T(bottom_.y());
                return true;
            }

            CameraIntrinsics K_;
            double gp_, alpha_, R_, L_, a_, yaw_off_, t_, t0_;
            Vec2 top_, bottom_;
            bool use_beta_;
        };

        struct SlowResidualJoint
        {
            SlowResidualJoint(const CameraIntrinsics& K, double gp, double L, double a,
                              double yaw_off, double t, const Vec3& C, double h, double theta0,
                              double omega, double beta, const Vec2& top, const Vec2& bottom)
                : K_(K), gp_(gp), L_(L), a_(a), yaw_off_(yaw_off), t_(t), C_(C), h_(h),
                  theta0_(theta0), omega_(omega), beta_(beta), top_(top), bottom_(bottom) {}

            template <typename T>
            bool operator()(const T* const alpha, const T* const R, T* residual) const
            {
                const T theta = T(theta0_) + T(omega_) * T(t_) + T(0.5) * T(beta_) * T(t_) * T(t_)
                                + T(yaw_off_);
                const T sa = ceres::sin(alpha[0]), ca = ceres::cos(alpha[0]);
                const T r_top = R[0] - T(0.5) * T(L_) * sa;
                const T r_bot = R[0] + T(0.5) * T(L_) * sa;
                const T h_top = T(h_) + T(0.5) * T(L_) * ca;
                const T h_bot = T(h_) - T(0.5) * T(L_) * ca;
                const T ct = ceres::cos(theta), st = ceres::sin(theta);
                const T aa = T(a_);
                const T pv_top[3] = { r_top * ct - aa * st, r_top * st + aa * ct, h_top };
                const T pv_bot[3] = { r_bot * ct - aa * st, r_bot * st + aa * ct, h_bot };
                const T s = ceres::sin(T(gp_)), c = ceres::cos(T(gp_));
                auto pc = [&](const T* pv, T* out) {
                    out[0] = pv[1] + T(C_.x());
                    out[1] = -s * pv[0] - c * pv[2] + T(C_.y());
                    out[2] = c * pv[0] - s * pv[2] + T(C_.z());
                };
                T pt[3], pb[3];
                pc(pv_top, pt);
                pc(pv_bot, pb);
                residual[0] = T(K_.fx) * pt[0] / pt[2] + T(K_.cx) - T(top_.x());
                residual[1] = T(K_.fy) * pt[1] / pt[2] + T(K_.cy) - T(top_.y());
                residual[2] = T(K_.fx) * pb[0] / pb[2] + T(K_.cx) - T(bottom_.x());
                residual[3] = T(K_.fy) * pb[1] / pb[2] + T(K_.cy) - T(bottom_.y());
                return true;
            }
            CameraIntrinsics K_;
            double gp_, L_, a_, yaw_off_, t_;
            Vec3 C_;
            double h_, theta0_, omega_, beta_;
            Vec2 top_, bottom_;
        };
    }  // namespace

    FitResult fitTrajectory(const std::vector<LightBarObservation>& observations,
                            const CameraIntrinsics& K, double gimbal_pitch,
                            double alpha_fixed, double R_fixed, double L, const FitConfig& cfg)
    {
        FitResult out;
        if (observations.size() < 4) return out;
        const bool use_beta = cfg.omega_model_order >= 2;
        const double t0 = observations.front().timestamp;

        // 初值：像素质心粗略反推 + 多起点（ω 方向）
        double best_cost = 1e300;
        for (double w0 : { cfg.omega_init, -cfg.omega_init, 2.0 * cfg.omega_init, -2.0 * cfg.omega_init })
        {
            double center[3] = { cfg.center_init.x(), cfg.center_init.y(), cfg.center_init.z() };
            double h = cfg.h_init;
            double phase[2] = { cfg.theta0_init, w0 };
            double beta = 0.0;
            double vel[3] = { 0.0, 0.0, 0.0 };
            ceres::Problem problem;
            for (const auto& o : observations)
            {
                auto* cost = new ceres::AutoDiffCostFunction<FastResidual, 4, 3, 1, 2, 1, 3>(
                    new FastResidual(K, gimbal_pitch, alpha_fixed, R_fixed, L, o.timestamp,
                                     t0, o.top_px, o.bottom_px, use_beta));
                problem.AddResidualBlock(cost,
                    cfg.use_huber ? new ceres::HuberLoss(cfg.huber_delta) : nullptr,
                    center, &h, phase, &beta, vel);
            }
            if (!use_beta) problem.SetParameterBlockConstant(&beta);
            {
                auto* prior = new ceres::AutoDiffCostFunction<VelPrior, 3, 3>(
                    new VelPrior(/*sigma=*/0.6));
                problem.AddResidualBlock(prior, nullptr, vel);
            }
            problem.SetParameterLowerBound(&h, 0, -2.0);
            problem.SetParameterUpperBound(&h, 0, 2.0);
            problem.SetParameterLowerBound(phase, 1, -30.0);
            problem.SetParameterUpperBound(phase, 1, 30.0);
            for (int i = 0; i < 3; ++i)
            {
                problem.SetParameterLowerBound(vel, i, -3.0);
                problem.SetParameterUpperBound(vel, i, 3.0);
            }
            ceres::Solver::Options opt;
            opt.linear_solver_type = ceres::DENSE_QR;
            opt.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
            opt.max_num_iterations = cfg.max_iterations;
            opt.function_tolerance = cfg.function_tolerance;
            opt.gradient_tolerance = cfg.gradient_tolerance;
            opt.minimizer_progress_to_stdout = false;
            ceres::Solver::Summary summary;
            ceres::Solve(opt, &problem, &summary);
            if (summary.final_cost < best_cost)
            {
                best_cost = summary.final_cost;
                out.center_cam = Vec3(center[0], center[1], center[2]);
                out.drift_velocity = Vec3(vel[0], vel[1], vel[2]);
                out.h = h;
                out.theta0 = phase[0];
                out.omega = phase[1];
                out.beta = use_beta ? beta : 0.0;
                out.num_iterations = static_cast<int>(summary.iterations.size());
            }
        }
        out.alpha_used = alpha_fixed;
        out.R_used = R_fixed;
        out.reprojection_rmse = rmseOf(observations, K, gimbal_pitch, alpha_fixed, R_fixed, L, out);
        out.success = std::isfinite(out.reprojection_rmse);
        return out;
    }

    namespace
    {
        // 固定快层参数，对 (alpha,R) 求残差向量（每观测 4 个）。
        // offs==nullptr 时退化为单灯条模型（无板 yaw 偏移、a=0）。
        void slowResiduals(const std::vector<LightBarObservation>& obs, const CameraIntrinsics& K,
                           double gp, double L, const std::vector<double>* offs,
                           const FitResult& f, double alpha, double R, std::vector<double>& out)
        {
            out.clear();
            const double t0 = obs.empty() ? 0.0 : obs.front().timestamp;
            for (const auto& o : obs)
            {
                double yaw_off = 0.0, a = 0.0;
                if (offs)
                {
                    if (o.armor_id < 0 || o.armor_id >= static_cast<int>(offs->size())) continue;
                    yaw_off = (*offs)[o.armor_id];
                    a = o.bar_offset;
                }
                const Vec3 c = f.centerAt(o.timestamp, t0);
                const double theta = f.yawAt(o.timestamp) + yaw_off;
                Vec3 top, bot;
                barEndpointsCameraOffset(c, f.h, theta, alpha, R, L, a, gp, top, bot);
                const Vec2 pt = project(top, K), pb = project(bot, K);
                out.push_back(pt.x() - o.top_px.x());
                out.push_back(pt.y() - o.top_px.y());
                out.push_back(pb.x() - o.bottom_px.x());
                out.push_back(pb.y() - o.bottom_px.y());
            }
        }

        // 确定性慢层：对 (alpha,R) 做 Huber 加权 Gauss-Newton（数值 J），
        // 输出更新后的 alpha,R 与 JᵀJ 对角信息量。规避 Ceres Evaluate 的浮点退化/非确定性。
        void slowRefine(const std::vector<LightBarObservation>& obs, const CameraIntrinsics& K,
                        double gp, double L, const std::vector<double>* offs, const FitResult& f,
                        double& alpha, double& R, double& info_alpha, double& info_R)
        {
            info_alpha = 0.0; info_R = 0.0;
            if (obs.empty() || !f.success) return;
            const double lo_a = 8.0 * M_PI / 180.0, hi_a = 25.0 * M_PI / 180.0;
            const double lo_r = 0.10, hi_r = 0.35, hd = 2.0;
            double a = std::clamp(alpha, lo_a, hi_a), r = std::clamp(R, lo_r, hi_r);
            std::vector<double> res, res_a, res_r;
            for (int it = 0; it < 8; ++it)
            {
                slowResiduals(obs, K, gp, L, offs, f, a, r, res);
                if (res.empty()) return;
                const double ha = 1e-6, hr = 1e-6;
                slowResiduals(obs, K, gp, L, offs, f, std::min(a + ha, hi_a), r, res_a);
                slowResiduals(obs, K, gp, L, offs, f, a, std::min(r + hr, hi_r), res_r);
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
                a = std::clamp(a + da, lo_a, hi_a);
                r = std::clamp(r + dr, lo_r, hi_r);
            }
            alpha = a; R = r;
        }
    }  // namespace

    void refineSlowParams(const std::vector<LightBarObservation>& observations,
                          const CameraIntrinsics& K, double gimbal_pitch, double L,
                          const FitResult& fast, double& alpha, double& R,
                          double& info_alpha, double& info_R)
    {
        slowRefine(observations, K, gimbal_pitch, L, nullptr, fast, alpha, R, info_alpha, info_R);
    }

    namespace
    {
        double rmseJoint(const std::vector<LightBarObservation>& obs, const CameraIntrinsics& K,
                         double gp, double alpha, double R, double L,
                         const std::vector<double>& offs, const FitResult& f)
        {
            double sum = 0.0; int n = 0;
            const double t0 = obs.empty() ? 0.0 : obs.front().timestamp;
            for (const auto& o : obs)
            {
                if (o.armor_id < 0 || o.armor_id >= static_cast<int>(offs.size())) continue;
                Vec3 top, bot;
                barEndpointsCameraOffset(f.centerAt(o.timestamp, t0), f.h,
                                         f.yawAt(o.timestamp) + offs[o.armor_id], alpha, R, L,
                                         o.bar_offset, gp, top, bot);
                const Vec2 pt = project(top, K), pb = project(bot, K);
                sum += (pt - o.top_px).squaredNorm() + (pb - o.bottom_px).squaredNorm();
                n += 4;
            }
            return n > 0 ? std::sqrt(sum / n) : 0.0;
        }
    }  // namespace

    FitResult fitTrajectoryJoint(const std::vector<LightBarObservation>& observations,
                                 const CameraIntrinsics& K, double gimbal_pitch,
                                 double alpha_fixed, double R_fixed, double L,
                                 const std::vector<double>& plate_yaw_offsets,
                                 const FitConfig& cfg)
    {
        FitResult out;
        if (observations.size() < 4 || plate_yaw_offsets.empty()) return out;
        const bool use_beta = cfg.omega_model_order >= 2;
        const double t0 = observations.front().timestamp;

        double best_cost = 1e300;
        for (double w0 : { cfg.omega_init, -cfg.omega_init, 2.0 * cfg.omega_init, -2.0 * cfg.omega_init })
        {
            double center[3] = { cfg.center_init.x(), cfg.center_init.y(), cfg.center_init.z() };
            double h = cfg.h_init;
            double phase[2] = { cfg.theta0_init, w0 };
            double beta = 0.0;
            double vel[3] = { 0.0, 0.0, 0.0 };
            ceres::Problem problem;
            for (const auto& o : observations)
            {
                if (o.armor_id < 0 || o.armor_id >= static_cast<int>(plate_yaw_offsets.size())) continue;
                auto* cost = new ceres::AutoDiffCostFunction<FastResidualJoint, 4, 3, 1, 2, 1, 3>(
                    new FastResidualJoint(K, gimbal_pitch, alpha_fixed, R_fixed, L, o.bar_offset,
                                          plate_yaw_offsets[o.armor_id], o.timestamp, t0,
                                          o.top_px, o.bottom_px, use_beta));
                problem.AddResidualBlock(cost,
                    cfg.use_huber ? new ceres::HuberLoss(cfg.huber_delta) : nullptr,
                    center, &h, phase, &beta, vel);
            }
            if (problem.NumResidualBlocks() < 2) return out;
            if (!use_beta) problem.SetParameterBlockConstant(&beta);
            {
                auto* prior = new ceres::AutoDiffCostFunction<VelPrior, 3, 3>(
                    new VelPrior(/*sigma=*/0.6));
                problem.AddResidualBlock(prior, nullptr, vel);
            }
            problem.SetParameterLowerBound(&h, 0, -2.0);
            problem.SetParameterUpperBound(&h, 0, 2.0);
            problem.SetParameterLowerBound(phase, 1, -30.0);
            problem.SetParameterUpperBound(phase, 1, 30.0);
            for (int i = 0; i < 3; ++i)
            {
                problem.SetParameterLowerBound(vel, i, -3.0);
                problem.SetParameterUpperBound(vel, i, 3.0);
            }
            ceres::Solver::Options opt;
            opt.linear_solver_type = ceres::DENSE_QR;
            opt.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
            opt.max_num_iterations = cfg.max_iterations;
            opt.function_tolerance = cfg.function_tolerance;
            opt.gradient_tolerance = cfg.gradient_tolerance;
            opt.minimizer_progress_to_stdout = false;
            ceres::Solver::Summary summary;
            ceres::Solve(opt, &problem, &summary);
            if (summary.final_cost < best_cost)
            {
                best_cost = summary.final_cost;
                out.center_cam = Vec3(center[0], center[1], center[2]);
                out.drift_velocity = Vec3(vel[0], vel[1], vel[2]);
                out.h = h;
                out.theta0 = phase[0];
                out.omega = phase[1];
                out.beta = use_beta ? beta : 0.0;
                out.num_iterations = static_cast<int>(summary.iterations.size());
            }
        }
        out.alpha_used = alpha_fixed;
        out.R_used = R_fixed;
        out.reprojection_rmse = rmseJoint(observations, K, gimbal_pitch, alpha_fixed, R_fixed, L,
                                          plate_yaw_offsets, out);
        out.success = std::isfinite(out.reprojection_rmse) && out.reprojection_rmse < 1e4;
        return out;
    }

    void refineSlowParamsJoint(const std::vector<LightBarObservation>& observations,
                               const CameraIntrinsics& K, double gimbal_pitch, double L,
                               const std::vector<double>& plate_yaw_offsets,
                               const FitResult& fast, double& alpha, double& R,
                               double& info_alpha, double& info_R)
    {
        if (plate_yaw_offsets.empty()) { info_alpha = 0.0; info_R = 0.0; return; }
        slowRefine(observations, K, gimbal_pitch, L, &plate_yaw_offsets, fast, alpha, R,
                   info_alpha, info_R);
    }
}  // namespace rm
