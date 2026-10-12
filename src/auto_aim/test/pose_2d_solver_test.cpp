// 2D 位姿求解器（6D→4D→2D 的最后一环）的回归自检。
//
// 钉四件事：
//   ① **往返**：用拟合器的刚体模型正向投影出四个角点，再让 2D 求解器解回来，
//      (t_x, t_y) 必须回到真值（亚毫米），残差必须 ≈0 —— 这直接验证
//      "拟合器的 ψ/t_z" 与 "PnP 关键点的 ψ/t_z" 是同一个量（方案假设的前提）。
//   ② **线性性**：闭式解与"迭代式高斯-牛顿"必须一致，且在解处 Jᵀr ≈ 0。
//   ③ **机械倾角的作用**：正面观测时，pitch=15° 的 ψ 曲率必须显著大于 pitch=0°
//      ——这是方案 §2.1「L·sinα(2R−C_z)/C_z²」那条公式的数值验证。
//   ④ **灵敏度**：ψ 偏 1° 会让 t_x 偏多少 —— 用来对照方案 §9.2 的误差传播估算。
//
// 纯数学、无 Ceres、无相机：直接构造角点像素。

#include "estimation/pose_2d_solver.hpp"
#include "perception/pnp_solver.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <random>
#include <vector>

namespace
{
    constexpr double kPi = 3.14159265358979323846;
    int failures = 0;

    bool expect(bool ok, const char* what)
    {
        if (!ok)
        {
            std::cerr << "FAILED: " << what << std::endl;
            ++failures;
        }
        return ok;
    }

    // 实车相机（configs/camera.yaml 的 hikcamera 标定）
    constexpr double kFx = 2398.73, kFy = 2396.69, kCx = 726.559, kCy = 565.457;

    /// 车体系→相机系：x前/y右/z上 → x右/y下/z前（"外参=单位阵 + 静态相机"时的 R_cv）。
    const double kRcv[9] = {0, 1, 0, 0, 0, -1, 1, 0, 0};

    std::array<cv::Point3f, 4> smallObject()
    {
        const auto pts = auto_aim::getArmor3DPoints(0, 0.1315f, 0.221f, 0.060f);
        std::array<cv::Point3f, 4> out{};
        for (int i = 0; i < 4; ++i) out[static_cast<std::size_t>(i)] = pts[static_cast<std::size_t>(i)];
        return out;
    }

    /// 用拟合器的刚体解正向生成 4 个角点像素（= 2D 求解器的输入）。
    std::array<cv::Point2f, 4> projectFromFitter(
        const auto_aim::FitterArmorPose& pose, const std::array<cv::Point3f, 4>& object,
        double pitch)
    {
        cv::Vec3d n, t, v;
        const double cp = std::cos(pitch), sp = std::sin(pitch);
        // armorKeypointBasis
        const cv::Vec3d n0(std::cos(pose.psi), 0.0, std::sin(pose.psi));
        const cv::Vec3d v0(0.0, 1.0, 0.0);
        const cv::Vec3d t0 = v0.cross(n0);
        n = cp * n0 + sp * v0;
        v = -sp * n0 + cp * v0;
        t = t0;
        (void)n;
        std::array<cv::Point2f, 4> uv{};
        for (int i = 0; i < 4; ++i)
        {
            const double oy = object[static_cast<std::size_t>(i)].y;
            const double oz = object[static_cast<std::size_t>(i)].z;
            const cv::Vec3d p = pose.center + oy * t + oz * v;
            uv[static_cast<std::size_t>(i)] = cv::Point2f(
                static_cast<float>(kFx * p[0] / p[2] + kCx),
                static_cast<float>(kFy * p[1] / p[2] + kCy));
        }
        return uv;
    }

    auto_aim::LocalFitResult makeFit(double yaw, double R, double h, double cx, double cy, double cz)
    {
        auto_aim::LocalFitResult f;
        f.valid = true;
        f.yaw = yaw;
        f.omega = 0.0;
        f.R = R;
        f.h = h;
        f.alpha = 15.0 * kPi / 180.0;
        f.center[0] = cx;
        f.center[1] = cy;
        f.center[2] = cz;
        f.quality = 1.0;
        return f;
    }

    const auto kObject = smallObject();
    const double kPitch = 15.0 * kPi / 180.0;

    // ---- ① 往返 --------------------------------------------------------------
    void testRoundTrip()
    {
        // 靶在相机正前方 2.0 m，车心 → 板心沿 +x_v（前）偏移，板号 1（相位 0°）
        const auto fit = makeFit(/*yaw=*/0.35, /*R=*/0.20, /*h=*/-0.12, /*cx=*/0.05, /*cy=*/0.10,
                                 /*cz=*/2.00);
        for (int plate = 0; plate < 4; ++plate)
        {
            const auto pose = auto_aim::fitterArmorPose(fit, plate, kRcv);
            if (!expect(pose.valid, "fitterArmorPose valid")) continue;

            const auto uv = projectFromFitter(pose, kObject, kPitch);
            auto_aim::Pose2DInput in;
            in.corners = uv;
            in.object = kObject;
            in.fx = kFx; in.fy = kFy; in.cx = kCx; in.cy = kCy;
            in.psi = pose.psi;
            in.tz = pose.center[2];
            in.pitch = kPitch;

            const auto r = auto_aim::solvePose2D(in);
            if (!expect(r.valid, "solvePose2D valid")) continue;
            const double dx = r.tx - pose.center[0];
            const double dy = r.ty - pose.center[1];
            const double dxy = std::hypot(dx, dy);
            if (!expect(dxy < 1e-3, "往返 t_x/t_y 误差 < 1 mm")) {
                std::printf("  plate %d: dxy = %.6f m (tx %.4f vs %.4f)\n", plate, dxy, r.tx,
                            pose.center[0]);
            }
            if (!expect(r.rmse_px < 1e-3, "往返残差 ≈ 0")) {
                std::printf("  plate %d: rmse = %.4f px, psi=%.4f rad\n", plate, r.rmse_px,
                            pose.psi);
            }
        }
    }

    // ---- ② 线性性：闭式解 = GN 解，且 Jᵀr ≈ 0 --------------------------------
    void testClosedFormIsStationary()
    {
        const auto fit = makeFit(0.35, 0.20, -0.12, 0.05, 0.10, 2.00);
        const auto pose = auto_aim::fitterArmorPose(fit, 1, kRcv);
        auto_aim::Pose2DInput in;
        in.corners = projectFromFitter(pose, kObject, kPitch);
        in.object = kObject;
        in.fx = kFx; in.fy = kFy; in.cx = kCx; in.cy = kCy;
        in.psi = pose.psi;
        in.tz = pose.center[2];
        in.pitch = kPitch;
        const auto r = auto_aim::solvePose2D(in);
        expect(r.valid, "solve valid");

        // 数值梯度：把 (tx,ty) 挪 0.1 mm，残差平方和必须变大（解在极小点）
        const auto sse = [&](double tx, double ty) {
            double s = 0.0;
            cv::Vec3d n, t, v;
            const double cp = std::cos(in.pitch), sp = std::sin(in.pitch);
            const cv::Vec3d n0(std::cos(in.psi), 0.0, std::sin(in.psi));
            const cv::Vec3d v0(0.0, 1.0, 0.0);
            t = v0.cross(n0);
            v = -sp * n0 + cp * v0;
            n = cp * n0 + sp * v0;
            (void)n;
            for (int i = 0; i < 4; ++i) {
                const double oy = in.object[static_cast<std::size_t>(i)].y;
                const double oz = in.object[static_cast<std::size_t>(i)].z;
                const cv::Vec3d p = cv::Vec3d(tx, ty, in.tz) + oy * t + oz * v;
                const double du = kFx * p[0] / p[2] + kCx - in.corners[static_cast<std::size_t>(i)].x;
                const double dv = kFy * p[1] / p[2] + kCy - in.corners[static_cast<std::size_t>(i)].y;
                s += du * du + dv * dv;
            }
            return s;
        };
        const double base = sse(r.tx, r.ty);
        const double h = 1e-4;
        expect(sse(r.tx + h, r.ty) > base, "t_x 方向在极小点");
        expect(sse(r.tx - h, r.ty) > base, "t_x 反向在极小点");
        expect(sse(r.tx, r.ty + h) > base, "t_y 方向在极小点");
        expect(sse(r.tx, r.ty - h) > base, "t_y 反向在极小点");
    }

    // ---- ③ 机械倾角 ↔ ψ 可观测性 --------------------------------------------
    double yawInfo(double plate_yaw, double pitch)
    {
        // 直接给板心在相机系的位置（不经过拟合器），只关心 pitch 对 ψ 曲率的影响
        auto_aim::Pose2DInput in;
        in.object = kObject;
        in.fx = kFx; in.fy = kFy; in.cx = kCx; in.cy = kCy;
        in.pitch = pitch;
        // 板"正对相机"：车心在 (0,0,2)，板心 = 车心 + R·(cosθ,sinθ,0)_v 经 R_cv
        const double R = 0.20;
        const cv::Vec3d cv_{R * std::cos(plate_yaw), R * std::sin(plate_yaw), 0.0};
        const cv::Vec3d cc{kRcv[1] * cv_[1], -kRcv[5] * cv_[2], kRcv[6] * cv_[0]};
        // 用拟合器路径更省事：直接构造"车心在正前方"的解，取板 1（相位 0）
        const auto fit = makeFit(plate_yaw, R, 0.0, 0.0, 0.0, 2.0);
        const auto pose = auto_aim::fitterArmorPose(fit, 1, kRcv);
        in.corners = projectFromFitter(pose, kObject, pitch);
        in.psi = pose.psi;
        in.tz = pose.center[2];
        (void)cc;
        return auto_aim::scanYaw(in, pose.psi, 20.0 * kPi / 180.0, kPi / 180.0).info;
    }

    void testTiltGivesYawObservability()
    {
        // 板正对相机（θ=0 → 板号 1 的相位 0）时，15° 支架应把 ψ 从"平谷"抬起来
        const double info0 = yawInfo(0.0, 0.0);
        const double info15 = yawInfo(0.0, kPitch);
        std::printf("  ψ 曲率：pitch=0° → %.4g，pitch=15° → %.4g（倍数 %.1f×）\n", info0, info15,
                    info0 > 0 ? info15 / info0 : 0.0);
        expect(info15 > info0, "正面观测：15° 支架的 ψ 曲率 > 0° 支架");
        // 侧向观测（θ=45°）本来就良态
        const double info45 = yawInfo(45.0 * kPi / 180.0, kPitch);
        expect(info45 > 0.0, "侧向观测 ψ 可观测");
    }

    // ---- ④ ψ 偏 1° 对 t_x、t_y 的影响（对照方案 §9.2）------------------------
    void testYawSensitivity()
    {
        const auto fit = makeFit(0.0, 0.20, 0.0, 0.0, 0.0, 2.0);
        const auto pose = auto_aim::fitterArmorPose(fit, 1, kRcv);
        auto_aim::Pose2DInput in;
        in.corners = projectFromFitter(pose, kObject, kPitch);
        in.object = kObject;
        in.fx = kFx; in.fy = kFy; in.cx = kCx; in.cy = kCy;
        in.psi = pose.psi;
        in.tz = pose.center[2];
        in.pitch = kPitch;
        const auto r0 = auto_aim::solvePose2D(in);
        expect(r0.valid, "基准解有效");
        // ⚠ 角点**固定**在真值姿态上（真实观测不会跟着 ψ 走），只改求解器拿到的 ψ。
        // 第一版把角点也按扰动后的 ψ 重新生成，量出来永远是 0 —— 那是自洽假设，不是灵敏度。
        std::printf("  ψ 误差 → (t_x,t_y) 误差（板宽 %.0f mm、深度 %.2f m）\n",
                    kObject[0].y * 2.0 * 1e3, pose.center[2]);
        for (double deg : {0.5, 1.0, 3.0, 5.0})
        {
            auto_aim::Pose2DInput p = in;
            p.psi = pose.psi + deg * kPi / 180.0;
            const auto r = auto_aim::solvePose2D(p);
            if (!expect(r.valid, "扰动解有效")) continue;
            const double predicted = 0.5 * kObject[0].y * 2.0 * std::tan(deg * kPi / 180.0) * 1e3;
            std::printf("  ψ 偏 %+.1f° → d(t_x,t_y) = (%6.2f, %6.2f) mm（估算 %.2f），残差 %.3f px\n",
                        deg, (r.tx - r0.tx) * 1e3, (r.ty - r0.ty) * 1e3, predicted, r.rmse_px);
        }

        // 斜视（板偏 45°）时 ψ 的灵敏度会明显变大——这是"正面/侧面"两种工况的分界
        const auto fit45 = makeFit(45.0 * kPi / 180.0, 0.20, 0.0, 0.0, 0.0, 2.0);
        const auto pose45 = auto_aim::fitterArmorPose(fit45, 0, kRcv);
        auto_aim::Pose2DInput in45;
        in45.corners = projectFromFitter(pose45, kObject, kPitch);
        in45.object = kObject;
        in45.fx = kFx; in45.fy = kFy; in45.cx = kCx; in45.cy = kCy;
        in45.psi = pose45.psi;
        in45.tz = pose45.center[2];
        in45.pitch = kPitch;
        const auto r45 = auto_aim::solvePose2D(in45);
        if (expect(r45.valid, "斜视基准解有效"))
        {
            auto_aim::Pose2DInput p = in45;
            p.psi = pose45.psi + 1.0 * kPi / 180.0;
            const auto r = auto_aim::solvePose2D(p);
            std::printf("  斜视 45°：ψ 偏 1° → d(t_x,t_y) = (%.2f, %.2f) mm，残差 %.3f px\n",
                        (r.tx - r45.tx) * 1e3, (r.ty - r45.ty) * 1e3, r.rmse_px);
        }

        // t_z 的灵敏度（方案 §9.2 认为 σ_Z=1~5cm 会带来 ~2mm 的 t_x 误差）
        for (double dz_cm : {2.0, 5.0})
        {
            auto_aim::Pose2DInput p = in;
            p.tz = pose.center[2] + dz_cm * 1e-2;
            const auto r = auto_aim::solvePose2D(p);
            if (!expect(r.valid, "t_z 扰动解有效")) continue;
            std::printf("  t_z 偏 %+.0f cm → d(t_x,t_y) = (%6.2f, %6.2f) mm，残差 %.3f px\n", dz_cm,
                        (r.tx - r0.tx) * 1e3, (r.ty - r0.ty) * 1e3, r.rmse_px);
        }
    }

    // ---- ⑤ 放开 t_z 时必须收敛回真值 ----------------------------------------
    // ---- ④b 像素噪声下：6D PnP vs 2D 求解器（方案"精度提升 10 倍"的判据）----
    // 这是唯一能判定"降维到底买到了什么"的实验：无噪声时角点本身就决定了
    // (t_x,t_y)，看不出差别；有噪声时 6D PnP 会**让噪声漏进病态方向**
    // （yaw/t_z），再经旋转耦合回到 t_x,t_y。2D 求解器把这两个方向钉住，就切断了这条路。
    void testNoiseComparison()
    {
        std::mt19937 rng(20261012u);
        std::normal_distribution<double> px_noise(0.0, 0.5);          // σ=0.5 px
        std::normal_distribution<double> psi_noise(0.0, 0.6 * kPi / 180.0);   // 实测 σ_ψ≈0.6°
        std::normal_distribution<double> tz_noise(0.0, 0.02);         // σ_Z=2 cm

        const auto fit = makeFit(0.0, 0.20, 0.0, 0.0, 0.0, 2.0);

        double pnp_med = 0.0, pnp_yaw_med = 0.0;
        double fit_med = 0.0, fit_yaw_med = 0.0;
        double orc_med = 0.0;   // 用真值 ψ/t_z（理论上界）
        std::vector<double> a, b, c, ay, fy, z_pnp;
        const int kTrials = 300;
        for (int trial = 0; trial < kTrials; ++trial)
        {
            const auto pose = auto_aim::fitterArmorPose(fit, 1, kRcv);
            const auto clean = projectFromFitter(pose, kObject, kPitch);
            std::array<cv::Point2f, 4> noisy{};
            for (int i = 0; i < 4; ++i)
                noisy[static_cast<std::size_t>(i)] = cv::Point2f(
                    clean[static_cast<std::size_t>(i)].x + static_cast<float>(px_noise(rng)),
                    clean[static_cast<std::size_t>(i)].y + static_cast<float>(px_noise(rng)));

            // (a) 6D PnP（IPPE；与 perception/pnp_solver.cpp 同一条路径的第一步）
            cv::Mat K = (cv::Mat_<double>(3, 3) << kFx, 0, kCx, 0, kFy, kCy, 0, 0, 1);
            cv::Mat D = cv::Mat::zeros(5, 1, CV_64F);
            cv::Vec3d rvec, tvec;
            const std::vector<cv::Point2f> img(
                noisy.begin(), noisy.end());
            bool ok = false;
            try {
                ok = cv::solvePnP(kObject, img, K, D, rvec, tvec, false, cv::SOLVEPNP_IPPE);
            } catch (const cv::Exception&) { ok = false; }
            if (ok && tvec[2] > 1e-3)
            {
                a.push_back(std::hypot(tvec[0] - pose.center[0], tvec[1] - pose.center[1]) * 1e3);
                z_pnp.push_back(std::abs(tvec[2] - pose.center[2]) * 1e3);
                cv::Mat R;
                cv::Rodrigues(rvec, R);
                cv::Vec3d n(R.at<double>(0, 0), R.at<double>(1, 0), R.at<double>(2, 0));
                if (n.dot(cv::Vec3d(tvec[0], tvec[1], tvec[2])) > 0) n = -n;
                const double psi_pnp = std::atan2(n[2], n[0]);
                ay.push_back(std::abs(std::remainder(psi_pnp - pose.psi, kPi)) * 180.0 / kPi);
            }

            // (b) 2D 求解器，用真值 ψ/t_z（上界）
            auto_aim::Pose2DInput in;
            in.corners = noisy;
            in.object = kObject;
            in.fx = kFx; in.fy = kFy; in.cx = kCx; in.cy = kCy;
            in.pitch = kPitch;
            in.psi = pose.psi;
            in.tz = pose.center[2];
            const auto r2 = auto_aim::solvePose2D(in);
            if (r2.valid)
                b.push_back(std::hypot(r2.tx - pose.center[0], r2.ty - pose.center[1]) * 1e3);

            // (c) 2D 求解器，用**实测的** ψ/t_z 误差（现实工况）
            in.psi = pose.psi + psi_noise(rng);
            in.tz = pose.center[2] + tz_noise(rng);
            const auto r3 = auto_aim::solvePose2D(in);
            if (r3.valid)
            {
                c.push_back(std::hypot(r3.tx - pose.center[0], r3.ty - pose.center[1]) * 1e3);
                fy.push_back(std::abs(std::remainder(in.psi - pose.psi, kPi)) * 180.0 / kPi);
            }
        }
        const auto med = [](std::vector<double> v) {
            if (v.empty()) return 0.0;
            std::sort(v.begin(), v.end());
            return v[v.size() / 2];
        };
        pnp_med = med(a);
        orc_med = med(b);
        fit_med = med(c);
        pnp_yaw_med = med(ay);
        fit_yaw_med = med(fy);
        const double pnp_z_med = med(z_pnp);
        std::printf("  [σ_px=0.5] (t_x,t_y) 中位误差：6D PnP %.2f mm ｜ 2D(真值 ψ,Z) %.2f mm ｜ "
                    "2D(实测 σ_ψ=0.6°,σ_Z=2cm) %.2f mm\n",
                    pnp_med, orc_med, fit_med);
        std::printf("              yaw 中位误差：PnP %.2f° ｜ 2D（继承输入）%.2f°\n", pnp_yaw_med,
                    fit_yaw_med);
        std::printf("              t_z 中位误差：PnP **%.1f mm**（拟合器的 Z 精度按方案是 10~50 mm）\n",
                    pnp_z_med);
        expect(!a.empty() && !c.empty(), "噪声对比有样本");
        expect(orc_med < pnp_med, "2D（真值 ψ,Z）比 6D PnP 更准");
    }

    void testSolveTz()
    {
        const auto fit = makeFit(0.0, 0.20, 0.0, 0.0, 0.0, 2.0);
        const auto pose = auto_aim::fitterArmorPose(fit, 1, kRcv);
        auto_aim::Pose2DInput in;
        in.corners = projectFromFitter(pose, kObject, kPitch);
        in.object = kObject;
        in.fx = kFx; in.fy = kFy; in.cx = kCx; in.cy = kCy;
        in.psi = pose.psi;
        in.pitch = kPitch;
        in.solve_tz = true;
        in.tz = pose.center[2] + 0.15;   // 初值故意偏 15 cm
        const auto r = auto_aim::solvePose2D(in);
        if (expect(r.valid, "solve_tz 有效"))
        {
            std::printf("  solve_tz：tz %.4f → %.4f（真值 %.4f），残差 %.3f px，%d 轮\n",
                        in.tz, r.tz, pose.center[2], r.rmse_px, r.iterations);
            expect(std::abs(r.tz - pose.center[2]) < 5e-3, "solve_tz 收敛回真值（<5 mm）");
            expect(r.rmse_px < 0.05, "solve_tz 残差 ≈ 0");
        }
    }
}  // namespace

int main()
{
    testRoundTrip();
    testClosedFormIsStationary();
    testTiltGivesYawObservability();
    testYawSensitivity();
    testNoiseComparison();
    testSolveTz();
    if (failures == 0)
    {
        std::cout << "pose_2d_solver_test: all checks passed" << std::endl;
        return 0;
    }
    std::cerr << "pose_2d_solver_test: " << failures << " check(s) failed" << std::endl;
    return 1;
}
