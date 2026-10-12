#include "pose_2d_solver.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include "common/armor_geometry.hpp"
#include "common/chassis_model.hpp"   // armorPhaseOffset（与 EKF 的 angleForPlate 同表）

namespace auto_aim
{
    namespace
    {
        constexpr double kPi = 3.14159265358979323846;

        /// 一次投影 + 残差。返回 false 表示有点跑到相机背面。
        /// 公式与 `pnp_solver.cpp` 的 `yawReprojectionCost` 同源：
        ///     X_i = tx + y_i·T_x + z_i·V_x
        ///     Y_i = ty + z_i·V_y          （T_y ≡ 0）
        ///     Z_i = tz + y_i·T_z + z_i·V_z （**与 tx,ty 无关** ← 线性性的来源）
        struct Projected
        {
            double x = 0.0, y = 0.0, z = 0.0;
        };

        inline bool projectAll(
            const Pose2DInput& in, const cv::Vec3d& tangent, const cv::Vec3d& vertical, double tx,
            double ty, double tz, std::array<Projected, 4>& out)
        {
            for (int i = 0; i < 4; ++i)
            {
                const double oy = in.object[static_cast<std::size_t>(i)].y;
                const double oz = in.object[static_cast<std::size_t>(i)].z;
                Projected p;
                p.x = tx + oy * tangent[0] + oz * vertical[0];
                p.y = ty + oy * tangent[1] + oz * vertical[1];
                p.z = tz + oy * tangent[2] + oz * vertical[2];
                if (!(p.z > 1e-3)) return false;
                out[static_cast<std::size_t>(i)] = p;
            }
            return true;
        }

        inline double rmseOf(
            const Pose2DInput& in, const std::array<Projected, 4>& p)
        {
            double sse = 0.0;
            for (int i = 0; i < 4; ++i)
            {
                const Projected& q = p[static_cast<std::size_t>(i)];
                const cv::Point2f& uv = in.corners[static_cast<std::size_t>(i)];
                const double du = in.fx * q.x / q.z + in.cx - uv.x;
                const double dv = in.fy * q.y / q.z + in.cy - uv.y;
                sse += du * du + dv * dv;
            }
            return std::sqrt(sse / 8.0);
        }
    }  // namespace

    Pose2DOutput solvePose2D(const Pose2DInput& input)
    {
        Pose2DOutput out;
        if (!(input.fx > 1.0) || !(input.fy > 1.0)) return out;
        for (const cv::Point2f& p : input.corners)
            if (!std::isfinite(p.x) || !std::isfinite(p.y)) return out;

        cv::Vec3d normal, tangent, vertical;
        armorKeypointBasis(input.psi, input.pitch, normal, tangent, vertical);

        // ---- 闭式解 (t_x, t_y) ---------------------------------------------------
        // 残差 u_i = fx·X_i/Z_i + cx − u_obs = A_i·tx + B_i，其中 Z_i 与 tx,ty 无关
        // （T_y ≡ 0，且 Z 只用 T_z、V_z）。所以最小化 Σres² 直接给
        //     tx = −Σ A_i B_i / Σ A_i²,  A_i = fx/Z_i, B_i = fx·K_i/Z_i + cx − u_i
        // （K_i = y_i·T_x + z_i·V_x 是常数项）。t_y 同理。
        double tz = input.tz;
        if (!std::isfinite(tz) || tz <= 1e-3) return out;

        double tx = 0.0, ty = 0.0;
        int iterations = 0;
        for (int iter = 0; iter < (input.solve_tz ? 20 : 1); ++iter)
        {
            ++iterations;
            double a_num = 0.0, a_den = 0.0, b_num = 0.0, b_den = 0.0;
            double min_depth = std::numeric_limits<double>::max();
            bool ok = true;
            for (int i = 0; i < 4; ++i)
            {
                const double oy = input.object[static_cast<std::size_t>(i)].y;
                const double oz = input.object[static_cast<std::size_t>(i)].z;
                const double kx = oy * tangent[0] + oz * vertical[0];
                const double ky = oy * tangent[1] + oz * vertical[1];
                const double z = tz + oy * tangent[2] + oz * vertical[2];
                if (!(z > 1e-3)) { ok = false; break; }
                min_depth = std::min(min_depth, z);
                const cv::Point2f& uv = input.corners[static_cast<std::size_t>(i)];
                const double ax = input.fx / z;
                const double bx = ax * kx + input.cx - uv.x;
                a_num += ax * bx;
                a_den += ax * ax;
                const double ay = input.fy / z;
                const double by = ay * ky + input.cy - uv.y;
                b_num += ay * by;
                b_den += ay * ay;
            }
            if (!ok || a_den < 1e-12 || b_den < 1e-12) return out;
            tx = -a_num / a_den;
            ty = -b_num / b_den;
            out.min_depth = min_depth;

            if (!input.solve_tz) break;

            // ---- 放开 t_z：对 t_z 做一步 GN（∂u/∂t_z = −fx·X/Z²，∂v/∂t_z = −fy·Y/Z²）
            std::array<Projected, 4> p{};
            if (!projectAll(input, tangent, vertical, tx, ty, tz, p)) { out.valid = false; return out; }
            double g = 0.0, h = 0.0;
            for (int i = 0; i < 4; ++i)
            {
                const Projected& q = p[static_cast<std::size_t>(i)];
                const cv::Point2f& uv = input.corners[static_cast<std::size_t>(i)];
                const double ru = input.fx * q.x / q.z + input.cx - uv.x;
                const double rv = input.fy * q.y / q.z + input.cy - uv.y;
                const double ju = -input.fx * q.x / (q.z * q.z);
                const double jv = -input.fy * q.y / (q.z * q.z);
                g += ju * ru + jv * rv;
                h += ju * ju + jv * jv;
            }
            if (h < 1e-18) break;
            const double step = -g / h;
            const double next = tz + std::clamp(step, -0.5, 0.5);
            if (!(next > 1e-3)) break;
            if (std::abs(next - tz) < 1e-9) { tz = next; break; }
            tz = next;
        }

        std::array<Projected, 4> p{};
        if (!projectAll(input, tangent, vertical, tx, ty, tz, p)) return out;
        out.valid = true;
        out.tx = tx;
        out.ty = ty;
        out.tz = tz;
        out.rmse_px = rmseOf(input, p);
        out.iterations = iterations;
        return out;
    }

    YawScanResult scanYaw(const Pose2DInput& input, double psi_center, double span, double step)
    {
        YawScanResult best;
        best.rmse_px = std::numeric_limits<double>::infinity();
        if (!(step > 1e-6) || span < 0.0) return best;

        Pose2DInput probe = input;
        double best_cost = std::numeric_limits<double>::infinity();
        const auto sse = [](const Pose2DOutput& r) { return r.rmse_px * r.rmse_px * 8.0; };
        for (double d = -span; d <= span + 1e-12; d += step)
        {
            const double psi = psi_center + d;
            probe.psi = psi;
            const Pose2DOutput r = solvePose2D(probe);
            if (!r.valid) continue;
            const double cost = sse(r);
            if (cost < best_cost)
            {
                best_cost = cost;
                best.psi = psi;
                best.rmse_px = r.rmse_px;
            }
        }

        // 曲率（可观测性代理量）：在最优点 ±step 上做二阶差分。正对相机时这条
        // profile cost 曲线是平谷 → info≈0，正是单帧 PnP yaw 病态的本质。
        if (std::isfinite(best_cost))
        {
            Pose2DInput a = input, b = input;
            a.psi = best.psi - step;
            b.psi = best.psi + step;
            const Pose2DOutput ra = solvePose2D(a), rb = solvePose2D(b);
            if (ra.valid && rb.valid)
                best.info = std::max(0.0, (sse(ra) + sse(rb) - 2.0 * best_cost) / (step * step));
        }
        return best;
    }

    FitterArmorPose fitterArmorPose(const LocalFitResult& fit, int plate_id, const double R_cv[9])
    {
        FitterArmorPose out;
        if (!fit.valid) return out;
        if (plate_id < 0 || plate_id > 3) return out;

        // 拟合器的相位表与 EKF/底盘模型同一张（见 armor_ekf.hpp 的 angleForPlate）：
        // 板 k 的相位 = 底盘 yaw + armorPhaseOffset(k)。
        const double theta = fit.yaw + armorPhaseOffset(plate_id);
        const double ct = std::cos(theta), st = std::sin(theta);
        // 板心在**车体系**：R·n̂ + h·ẑ（b=0、a=0，与 barEndpointV 一致）
        const cv::Vec3d center_v{fit.R * ct, fit.R * st, fit.h};
        const cv::Vec3d normal_v{ct, st, 0.0};

        const auto mul = [&](const cv::Vec3d& v) {
            return cv::Vec3d(
                R_cv[0] * v[0] + R_cv[1] * v[1] + R_cv[2] * v[2],
                R_cv[3] * v[0] + R_cv[4] * v[1] + R_cv[5] * v[2],
                R_cv[6] * v[0] + R_cv[7] * v[1] + R_cv[8] * v[2]);
        };
        const cv::Vec3d pc = mul(center_v);
        out.center = cv::Vec3d(pc[0] + fit.center[0], pc[1] + fit.center[1], pc[2] + fit.center[2]);
        out.radius = fit.R;

        // 法向方位角：与 refineArmorYaw 同口径——先折到"朝向相机"的那一侧，再取 atan2(z, x)。
        cv::Vec3d n = mul(normal_v);
        if (n.dot(out.center) > 0.0) n = -n;
        out.psi = std::atan2(n[2], n[0]);
        out.valid = std::isfinite(out.psi) && out.center[2] > 1e-3;
        return out;
    }
}  // namespace auto_aim
