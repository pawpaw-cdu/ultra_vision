#include "trajectory_fitter.h"

#include <ceres/ceres.h>
#include <algorithm>
#include <cmath>

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

    void refineSlowParams(const std::vector<LightBarObservation>& observations,
                          const CameraIntrinsics& K, double gimbal_pitch, double L,
                          const FitResult& fast, double& alpha, double& R,
                          double& info_alpha, double& info_R)
    {
        info_alpha = 0.0; info_R = 0.0;
        if (observations.empty()) return;
        double a = alpha, r = R;
        ceres::Problem problem;
        for (const auto& o : observations)
        {
            auto* cost = new ceres::AutoDiffCostFunction<SlowResidual, 4, 1, 1>(
                new SlowResidual(K, gimbal_pitch, L, o.timestamp, fast.center_cam, fast.h,
                                 fast.theta0, fast.omega, fast.beta, o.top_px, o.bottom_px));
            problem.AddResidualBlock(cost, new ceres::HuberLoss(2.0), &a, &r);
        }
        problem.SetParameterLowerBound(&a, 0, 8.0 * M_PI / 180.0);
        problem.SetParameterUpperBound(&a, 0, 25.0 * M_PI / 180.0);
        problem.SetParameterLowerBound(&r, 0, 0.10);
        problem.SetParameterUpperBound(&r, 0, 0.35);
        ceres::Solver::Options opt;
        opt.linear_solver_type = ceres::DENSE_QR;
        opt.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
        opt.max_num_iterations = 10;
        opt.minimizer_progress_to_stdout = false;
        ceres::Solver::Summary summary;
        ceres::Solve(opt, &problem, &summary);

        // 信息量：JᵀJ 对角（α、R 两列）。
        double cost = 0.0;
        std::vector<double> residuals;
        std::vector<double> jacobian;
        std::vector<double*> params{ &a, &r };
        if (problem.Evaluate(ceres::Problem::EvaluateOptions(), &cost, &residuals, &jacobian, nullptr))
        {
            const int nres = static_cast<int>(residuals.size());
            for (int i = 0; i < nres; ++i)
            {
                info_alpha += jacobian[i * 2 + 0] * jacobian[i * 2 + 0];
                info_R += jacobian[i * 2 + 1] * jacobian[i * 2 + 1];
            }
        }
        alpha = a; R = r;
    }
}  // namespace rm
