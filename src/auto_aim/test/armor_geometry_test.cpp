// 模型构造（整车轮廓重投影）的几何自检。
//
// 为什么要单独立这个测试：现场看到"绿色框贴得住、重投影的模型跑飞"，其中一部分
// 是**模型构造本身**的问题 —— 三处调用点各写一套（世界系 yaw 造切向 / 拿观测板
// PnP 的 rvec 当竖直 / 角点组装时把切向的 y 分量丢掉），而且没有一处带真机
// 装甲板的 15° 安装倾角。这里把"模型构造 = 局部拟合器的车辆系几何"钉成恒等式：
//
//   拟合器的车辆系模型（estimation/local_trajectory_fitter.cpp::barEndpointV）：
//     p_v(a,b) = (R − b·sinα)·n̂ + a·t̂ + (h + b·cosα)·ẑ,   a=±W/2, b=±L/2
//   模型构造（common/armor_geometry.hpp）：
//     p = center + a·tangent + b·bar,  center = R·n̂ + h·ẑ
//   两者必须逐点相等（这就是"灯条 15° 倾角作为模型约束"的具体含义）。
//
// 另外两条回归：
//   · 三轴正交归一（任意相机姿态下）；
//   · **旋转不变性**：把车辆系整体转到相机系后，用相机系的径向/竖直重建的角点，
//     必须等于"车辆系角点整体旋转"的结果 —— 老版本在角点组装里只把竖直轴的 y
//     分量写进 y、把切向的 y 分量丢掉，相机一有 pitch 就不满足这条。

#include <cmath>
#include <cstdio>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>

#include "common/armor_geometry.hpp"

using auto_aim::ArmorAxes;
using auto_aim::armorAxesFromRadial;
using auto_aim::armorCorners;

namespace
{
    int failures = 0;

    void expectNear(double got, double want, double tol, const char* what)
    {
        if (std::abs(got - want) > tol) {
            std::printf("FAIL %s: got %.9f want %.9f (tol %.1e)\n", what, got, want, tol);
            ++failures;
        }
    }

    void expectVec(const cv::Vec3d& got, const cv::Vec3d& want, double tol, const char* what)
    {
        for (int i = 0; i < 3; ++i) {
            if (std::abs(got[i] - want[i]) > tol) {
                std::printf("FAIL %s: got (%.6f,%.6f,%.6f) want (%.6f,%.6f,%.6f)\n", what,
                            got[0], got[1], got[2], want[0], want[1], want[2]);
                ++failures;
                return;
            }
        }
    }

    // 拟合器车辆系：板面法向 n、切向 t、竖直 ẑ（笛卡尔右手：t = ẑ × n）。
    void vehicleFrame(double theta, cv::Vec3d& n, cv::Vec3d& t, cv::Vec3d& zhat)
    {
        const double c = std::cos(theta);
        const double s = std::sin(theta);
        n = cv::Vec3d(c, s, 0.0);
        t = cv::Vec3d(-s, c, 0.0);
        zhat = cv::Vec3d(0.0, 0.0, 1.0);
    }

    // 拟合器的灯条端点公式（与 local_trajectory_fitter.cpp 的 barEndpointV 同构）。
    cv::Vec3d fitterCorner(double h, double theta, double alpha, double R, double a, double b,
                           const cv::Vec3d& n, const cv::Vec3d& t, const cv::Vec3d& zhat)
    {
        const double rho = R - b * std::sin(alpha);
        const double z = h + b * std::cos(alpha);
        return rho * n + a * t + z * zhat;
    }

    cv::Matx33d rotationFromRodrigues(const cv::Vec3d& axis, double angle)
    {
        cv::Matx33d R;
        cv::Rodrigues(axis * (angle / cv::norm(axis)), R);
        return R;
    }

    void checkCase(double theta, double alpha_deg, double R, double W, double L, double h)
    {
        const double alpha = alpha_deg * CV_PI / 180.0;
        cv::Vec3d n, t, zhat;
        vehicleFrame(theta, n, t, zhat);

        const ArmorAxes axes = armorAxesFromRadial(n, zhat, alpha);
        const cv::Vec3d center = R * n + h * zhat;

        // ① 恒等式：模型构造 == 拟合器几何（四个角点逐一相等）
        const std::vector<cv::Vec3d> model =
            armorCorners(center, axes, 0.5 * W, 0.5 * L);
        const double a[4] = {-0.5 * W, -0.5 * W, 0.5 * W, 0.5 * W};
        const double b[4] = {-0.5 * L, 0.5 * L, 0.5 * L, -0.5 * L};
        for (int i = 0; i < 4; ++i) {
            const cv::Vec3d want = fitterCorner(h, theta, alpha, R, a[i], b[i], n, t, zhat);
            expectVec(model[i], want, 1e-9, "模型角点 == 拟合器车辆系角点");
        }

        // ② 三轴正交归一
        expectNear(cv::norm(axes.normal), 1.0, 1e-12, "|normal|=1");
        expectNear(cv::norm(axes.tangent), 1.0, 1e-12, "|tangent|=1");
        expectNear(cv::norm(axes.bar), 1.0, 1e-12, "|bar|=1");
        expectNear(axes.normal.dot(axes.tangent), 0.0, 1e-12, "normal·tangent=0");
        expectNear(axes.normal.dot(axes.bar), 0.0, 1e-12, "normal·bar=0");
        expectNear(axes.tangent.dot(axes.bar), 0.0, 1e-12, "tangent·bar=0");

        // ③ 倾角方向：灯条顶端向**车内**倾 α（bar·radial = −sinα），且绕切线转 α
        expectNear(axes.bar.dot(n), -std::sin(alpha), 1e-12, "bar·radial = −sinα（顶部向车内）");
        expectNear(axes.bar.dot(zhat), std::cos(alpha), 1e-12, "bar·up = cosα");
        expectNear(axes.normal.dot(n), std::cos(alpha), 1e-12, "normal·radial = cosα");
    }

    // ④ 旋转不变性：把整台车转到任意相机姿态后，用相机系径向/竖直重建的角点，
    //    必须等于"车辆系角点整体旋转"。老版本角点组装丢 tangent.y 就过不了这条。
    void checkRotationInvariance()
    {
        const double alpha = 15.0 * CV_PI / 180.0;
        const double R = 0.21;
        const double W = 0.1315;
        const double L = 0.060;
        const double h = 0.0;
        cv::Vec3d n, t, zhat;
        vehicleFrame(0.4, n, t, zhat);
        const cv::Vec3d center_v = R * n + h * zhat;
        const std::vector<cv::Vec3d> corners_v = armorCorners(
            center_v, armorAxesFromRadial(n, zhat, alpha), 0.5 * W, 0.5 * L);

        // 相机姿态：俯仰 25° + 偏航 −40° + 一点 roll（真实云台就是这个量级）
        cv::Matx33d Rcam = rotationFromRodrigues(cv::Vec3d(0, 1, 0), -0.7) *
                          rotationFromRodrigues(cv::Vec3d(1, 0, 0), 0.44) *
                          rotationFromRodrigues(cv::Vec3d(0, 0, 1), 0.13);
        const cv::Vec3d n_cam = Rcam * n;
        const cv::Vec3d up_cam = Rcam * zhat;
        const cv::Vec3d center_cam = Rcam * center_v;
        const std::vector<cv::Vec3d> corners_cam =
            armorCorners(center_cam, armorAxesFromRadial(n_cam, up_cam, alpha), 0.5 * W, 0.5 * L);
        for (std::size_t i = 0; i < corners_v.size(); ++i) {
            expectVec(corners_cam[i], Rcam * corners_v[i], 1e-9,
                      "相机系重建的角点 == 车辆系角点整体旋转");
        }
        // 相机系里切向的 y 分量不为 0（这正是老版本丢掉的那一项）
        if (std::abs(armorAxesFromRadial(n_cam, up_cam, alpha).tangent[1]) < 1e-3) {
            std::printf("FAIL 相机姿态下 tangent.y ≈ 0 —— 用例没覆盖到丢 y 分量那个 bug\n");
            ++failures;
        }
    }
}  // namespace

int main()
{
    // 仿真靶（α=0，灯条绝对竖直）× 真机靶（α=15°）× 几个板相位
    const double angles[3] = {0.0, 0.7, -2.1};
    const double alphas[3] = {0.0, 15.0, -15.0};
    for (double th : angles) {
        for (double al : alphas) checkCase(th, al, 0.25, 0.135, 0.060, 0.0);
    }
    checkCase(1.1, 15.0, 0.21, 0.1315, 0.060, 0.03);   // 带板高偏置
    checkRotationInvariance();

    if (failures == 0) {
        std::printf("armor_geometry_test: OK\n");
        return 0;
    }
    std::printf("armor_geometry_test: %d 处失败\n", failures);
    return 1;
}
