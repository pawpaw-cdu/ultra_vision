#include "estimation/spin_geometry_estimator.hpp"

#include <ceres/ceres.h>
#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace auto_aim
{
namespace
{
    constexpr int kKnots = 10;   // yaw(t) 三次 B 样条控制点数（编译期固定，供 AutoDiff 使用）

    // 均匀三次 B 样条基（给定 t、控制点数 K、域 [t0,t1] → 4 个非零 (index,weight)）。
    void splineBasis(double t, double t0, double t1, int K, int idx[4], double w[4])
    {
        const double h = (t1 - t0) / std::max(1, K - 3);
        double s = (t - t0) / h + 3.0;
        int j = static_cast<int>(std::floor(s));
        if (j < 3) j = 3;
        if (j > K - 1) j = K - 1;
        const double u = s - j;
        const double u2 = u * u, u3 = u2 * u;
        idx[0] = j - 3; idx[1] = j - 2; idx[2] = j - 1; idx[3] = j;
        w[0] = (1.0 - u) * (1.0 - u) * (1.0 - u) / 6.0;
        w[1] = (3.0 * u3 - 6.0 * u2 + 4.0) / 6.0;
        w[2] = (-3.0 * u3 + 3.0 * u2 + 3.0 * u + 1.0) / 6.0;
        w[3] = u3 / 6.0;
    }

    double evalSpline(const std::vector<double>& c, double t, double t0, double t1)
    {
        if (c.size() < 4) return 0.0;
        int idx[4]; double w[4];
        splineBasis(t, t0, t1, static_cast<int>(c.size()), idx, w);
        double s = 0.0;
        for (int k = 0; k < 4; ++k) s += c[idx[k]] * w[k];
        return s;
    }

    std::vector<double> solveSplineCoeffs(const std::vector<double>& ts,
                                          const std::vector<double>& ys, int K,
                                          double t0, double t1)
    {
        const int n = static_cast<int>(ts.size());
        Eigen::MatrixXd A(n, K);
        Eigen::VectorXd y(n);
        for (int i = 0; i < n; ++i)
        {
            int idx[4]; double w[4];
            splineBasis(ts[i], t0, t1, K, idx, w);
            A.row(i).setZero();
            for (int k = 0; k < 4; ++k) A(i, idx[k]) += w[k];
            y(i) = ys[i];
        }
        const Eigen::VectorXd c = A.colPivHouseholderQr().solve(y);
        return std::vector<double>(c.data(), c.data() + c.size());
    }

    // 通用正演：旋转轴 â=normalize((tx,1,tz))，相位 th，灯条 side=+1 的两端点像素。
    template <typename T>
    void endpointsGeneric(const T* c3, const T R, const T alpha, const T th, const T tx,
                          const T tz, int plate, int side, double c_off, double L,
                          const double* Kd, T* up_t, T* vp_t, T* up_b, T* vp_b)
    {
        const T n = ceres::sqrt(tx * tx + T(1.0) + tz * tz);
        const T ax = tx / n, ay = T(1.0) / n, az = tz / n;
        T e1x = T(1.0) - ax * ax, e1y = -ax * ay, e1z = -ax * az;
        const T e1n = ceres::sqrt(e1x * e1x + e1y * e1y + e1z * e1z);
        e1x /= e1n; e1y /= e1n; e1z /= e1n;
        const T e2x = e1y * az - e1z * ay;
        const T e2y = e1z * ax - e1x * az;
        const T e2z = e1x * ay - e1y * ax;
        const T phase = th + T(plate) * T(M_PI / 2.0);
        const T ct = ceres::cos(phase), st = ceres::sin(phase);
        const T rx = ct * e1x + st * e2x, ry = ct * e1y + st * e2y, rz = ct * e1z + st * e2z;
        const T hx = -st * e1x + ct * e2x, hy = -st * e1y + ct * e2y, hz = -st * e1z + ct * e2z;
        const T sc = T(side * c_off);
        const T bx = c3[0] + R * rx + sc * hx;
        const T by = c3[1] + R * ry + sc * hy;
        const T bz = c3[2] + R * rz + sc * hz;
        const T ca = ceres::cos(alpha), sa = ceres::sin(alpha);
        const T dx = ca * ax + sa * rx, dy = ca * ay + sa * ry, dz = ca * az + sa * rz;
        const T h = T(0.5) * T(L);
        const T tX = bx + h * dx, tY = by + h * dy, tZ = bz + h * dz;
        const T bX = bx - h * dx, bY = by - h * dy, bZ = bz - h * dz;
        *up_t = T(Kd[0]) * tX / tZ + T(Kd[2]);
        *vp_t = T(Kd[1]) * tY / tZ + T(Kd[3]);
        *up_b = T(Kd[0]) * bX / bZ + T(Kd[2]);
        *vp_b = T(Kd[1]) * bY / bZ + T(Kd[3]);
    }

    template <int K>
    struct ReprojResidualSpline
    {
        ReprojResidualSpline(const SpinIntrinsics& K_, double c, double L, double t, int plate,
                             int side, double u_t, double v_t, double u_b, double v_b,
                             double t0, double t1)
            : c_(c), L_(L), plate_(plate), side_(side),
              u_t_(u_t), v_t_(v_t), u_b_(u_b), v_b_(v_b)
        {
            Kd_[0] = K_.fx; Kd_[1] = K_.fy; Kd_[2] = K_.cx; Kd_[3] = K_.cy;
            splineBasis(t, t0, t1, K, idx_, w_);
        }

        template <typename T>
        bool operator()(const T* const center, const T* const R, const T* const alpha,
                        const T* const tx, const T* const tz, const T* const coeff,
                        T* residual) const
        {
            T th = T(0.0);
            for (int k = 0; k < 4; ++k) th += coeff[idx_[k]] * T(w_[k]);
            T ut, vt, ub, vb;
            endpointsGeneric<T>(center, R[0], alpha[0], th, tx[0], tz[0], plate_, side_, c_, L_,
                                Kd_, &ut, &vt, &ub, &vb);
            residual[0] = ut - T(u_t_);
            residual[1] = vt - T(v_t_);
            residual[2] = ub - T(u_b_);
            residual[3] = vb - T(v_b_);
            return true;
        }

        double c_, L_;
        int plate_, side_;
        double u_t_, v_t_, u_b_, v_b_;
        double Kd_[4];
        int idx_[4];
        double w_[4];
    };

    double reprojRmse(const std::vector<SpinBarObservation>& obs, const SpinIntrinsics& K,
                      double c, double L, const Eigen::Vector3d& C, double R, double alpha,
                      double tx, double tz, const std::vector<double>& spline,
                      double t0, double t1)
    {
        double sum = 0.0;
        int n = 0;
        const double Kd[4] = {K.fx, K.fy, K.cx, K.cy};
        const double c3[3] = {C.x(), C.y(), C.z()};
        for (const auto& o : obs)
        {
            const double th = evalSpline(spline, o.t, t0, t1);
            double ut, vt, ub, vb;
            endpointsGeneric<double>(c3, R, alpha, th, tx, tz, o.plate, o.side, c, L, Kd,
                                     &ut, &vt, &ub, &vb);
            sum += (ut - o.u_top) * (ut - o.u_top) + (vt - o.v_top) * (vt - o.v_top)
                 + (ub - o.u_bot) * (ub - o.u_bot) + (vb - o.v_bot) * (vb - o.v_bot);
            n += 4;
        }
        return n > 0 ? std::sqrt(sum / n) : 0.0;
    }
}  // namespace

    double SpinGeometryResult::yawAt(double t) const
    {
        return evalSpline(spline, t, t_begin, t_end);
    }

    SpinGeometryEstimator::SpinGeometryEstimator(SpinGeometryConfig config)
        : config_(std::move(config))
    {
        config_.spline_knots = kKnots;
    }

    std::vector<std::array<Eigen::Vector3d, 2>> SpinGeometryEstimator::allBars(double theta) const
    {
        std::vector<std::array<Eigen::Vector3d, 2>> bars;
        bars.reserve(8);
        const Eigen::Vector3d axis =
            Eigen::Vector3d(result_.axis_tx, 1.0, result_.axis_tz).normalized();
        for (int plate = 0; plate < 4; ++plate)
        {
            const double phase = theta + plate * M_PI / 2.0;
            for (int side = -1; side <= 1; side += 2)
            {
                const double ct = std::cos(phase), st = std::sin(phase);
                Eigen::Vector3d ref(1.0, 0.0, 0.0);
                if (std::abs(axis.dot(ref)) > 0.9) ref = Eigen::Vector3d(0.0, 1.0, 0.0);
                const Eigen::Vector3d e1 = (ref - axis * ref.dot(axis)).normalized();
                const Eigen::Vector3d e2 = e1.cross(axis);
                const Eigen::Vector3d rhat = ct * e1 + st * e2;
                const Eigen::Vector3d that = -st * e1 + ct * e2;
                const Eigen::Vector3d center =
                    result_.center + rhat * result_.radius + that * (side * config_.bar_offset_c);
                const Eigen::Vector3d dir =
                    std::cos(result_.pitch) * axis + std::sin(result_.pitch) * rhat;
                bars.push_back({center + dir * (0.5 * config_.bar_length_L),
                                center - dir * (0.5 * config_.bar_length_L)});
            }
        }
        return bars;
    }

    bool SpinGeometryEstimator::fit(const std::vector<SpinBarObservation>& observations)
    {
        result_ = SpinGeometryResult{};
        const int n = static_cast<int>(observations.size());
        if (n < 8) return false;

        const SpinIntrinsics& K = config_.K;
        const double c = config_.bar_offset_c;
        const double L = config_.bar_length_L;
        const double t0 = observations.front().t;
        const double t1 = observations.back().t;
        if (t1 - t0 < 1e-3) return false;

        // 像素统计 → 粗初值
        double u_min = 1e18, u_max = -1e18, us = 0.0, vs = 0.0;
        for (const auto& o : observations)
        {
            const double mu = 0.5 * (o.u_top + o.u_bot);
            const double mv = 0.5 * (o.v_top + o.v_bot);
            u_min = std::min(u_min, mu); u_max = std::max(u_max, mu);
            us += mu; vs += mv;
        }
        const double Z0 = std::max(0.5, config_.distance_hint);
        const Eigen::Vector3d C0((us / n - K.cx) / K.fx * Z0,
                                 (vs / n - K.cy) / K.fy * Z0, Z0);
        const double R0 = std::clamp((u_max - u_min) * Z0 / (2.0 * K.fx), 0.05, 0.6);

        std::vector<double> ts(n);
        for (int i = 0; i < n; ++i) ts[i] = observations[i].t;

        double best_cost = std::numeric_limits<double>::infinity();
        bool ok = false;
        for (double th0 : {0.0, M_PI})
        {
            for (double w1 : {3.0, -3.0})
            {
                std::vector<double> ys(n);
                for (int i = 0; i < n; ++i) ys[i] = th0 + w1 * (ts[i] - t0);
                std::vector<double> coeff =
                    solveSplineCoeffs(ts, ys, config_.spline_knots, t0, t1);

                double center[3] = {C0.x(), C0.y(), C0.z()};
                double R = R0, alpha = 0.0, tx = 0.0, tz = 0.0;
                ceres::Solver::Summary summary;
                for (int stage = 0; stage < 2; ++stage)
                {
                    ceres::Problem problem;
                    const double huber = (stage == 0) ? 20.0 : 2.0;
                    for (const auto& o : observations)
                    {
                        auto* cost = new ceres::AutoDiffCostFunction<
                            ReprojResidualSpline<kKnots>, 4, 3, 1, 1, 1, 1, kKnots>(
                            new ReprojResidualSpline<kKnots>(
                                K, c, L, o.t, o.plate, o.side, o.u_top, o.v_top, o.u_bot,
                                o.v_bot, t0, t1));
                        problem.AddResidualBlock(cost, new ceres::HuberLoss(huber),
                                                 center, &R, &alpha, &tx, &tz, coeff.data());
                    }
                    problem.SetParameterLowerBound(&R, 0, 0.02);
                    problem.SetParameterUpperBound(&R, 0, 1.0);
                    problem.SetParameterLowerBound(&alpha, 0, -1.45);
                    problem.SetParameterUpperBound(&alpha, 0, 1.45);
                    problem.SetParameterLowerBound(&tx, 0, -0.35);
                    problem.SetParameterUpperBound(&tx, 0, 0.35);
                    problem.SetParameterLowerBound(&tz, 0, -0.35);
                    problem.SetParameterUpperBound(&tz, 0, 0.35);
                    if (!config_.estimate_axis)
                    {
                        problem.SetParameterBlockConstant(&tx);
                        problem.SetParameterBlockConstant(&tz);
                    }
                    ceres::Solver::Options options;
                    options.linear_solver_type = ceres::DENSE_QR;
                    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
                    options.max_num_iterations = 400;
                    options.function_tolerance = 1e-13;
                    options.gradient_tolerance = 1e-14;
                    options.parameter_tolerance = 1e-12;
                    options.minimizer_progress_to_stdout = false;
                    ceres::Solve(options, &problem, &summary);
                }

                const Eigen::Vector3d C(center[0], center[1], center[2]);
                const double cost = reprojRmse(observations, K, c, L, C, R, alpha, tx, tz,
                                               coeff, t0, t1);
                if (cost < best_cost)
                {
                    best_cost = cost;
                    result_.center = C;
                    result_.radius = R;
                    result_.pitch = alpha;
                    result_.axis_tx = tx;
                    result_.axis_tz = tz;
                    result_.t_begin = t0;
                    result_.t_end = t1;
                    result_.spline = coeff;
                    result_.reproj_rmse_px = cost;
                    result_.message_cost = summary.final_cost;
                    ok = true;
                }
            }
        }
        if (!ok) return false;

        // 弧覆盖（净转角，单位：度）—— 有效性门控的主要判据。
        // 注意：不能用 Σ|Δyaw|（噪声会把拟合 yaw 抖大 → 弧覆盖被虚高）。
        // 样条 yaw 本身连续（不取模），>360° 也能正确计数；不要用 remainder。
        result_.arc_coverage_deg =
            std::abs(result_.yawAt(t1) - result_.yawAt(t0)) * 180.0 / M_PI;
        result_.valid = result_.arc_coverage_deg >= 190.0;   // 实测门限 ~200°
        return result_.valid;
    }
}  // namespace auto_aim
