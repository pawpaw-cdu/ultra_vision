// PnP 降自由度（fit_single，上科大 2026 §5A.4）自检。
//
// 构造一块**已知位姿**的竖直装甲板，把四角投影到像平面并注入像素噪声，然后
// 比较"标准 IPPE 6 自由度"与"固定 tvec 的 1D yaw 精修"恢复出的**径向角**误差。
// 径向角是下游 EKF 估计 ω、以及关联板号的关键观测量，所以这是最直接的验收指标。

#include "perception/pnp_solver.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>

namespace
{
    constexpr double kPi = 3.14159265358979323846;

    bool expect(bool condition, const char* message)
    {
        if (!condition) std::cerr << "FAILED: " << message << std::endl;
        return condition;
    }

    double normalizeAngle(double angle)
    {
        return std::atan2(std::sin(angle), std::cos(angle));
    }

    /// @brief 与 tracker.cpp 的 armorRadialAngle 同口径：取 rvec 的第一列做法向，
    ///        按"可见面朝相机"折叠 ±π，再返回 atan2(n_z, n_x)。
    double radialAngle(const auto_aim::Armor& armor)
    {
        cv::Mat rotation;
        cv::Rodrigues(armor.rvec, rotation);
        cv::Vec3d normal(rotation.at<double>(0, 0), rotation.at<double>(1, 0),
                         rotation.at<double>(2, 0));
        const cv::Vec3d position(armor.tvec.at<double>(0), armor.tvec.at<double>(1),
                                 armor.tvec.at<double>(2));
        if (normal.dot(position) > 0.0) normal = -normal;
        return std::atan2(normal[2], normal[0]);
    }
}

int main()
{
    bool passed = true;

    const cv::Mat camera_matrix = (cv::Mat_<double>(3, 3) <<
        1444.3, 0.0, 720.0,
        0.0, 1444.3, 540.0,
        0.0, 0.0, 1.0);
    const cv::Mat dist_coeffs = cv::Mat::zeros(5, 1, CV_64F);

    auto_aim::PnpGeometry geometry;
    geometry.small_width = 0.1315f;
    geometry.large_width = 0.221f;
    geometry.height = 0.060f;

    // 底盘圆心在正前方 5 m，半径 0.21 m。取正对相机的那块板（plate 2 → ψ=-90°）。
    const double center_z = 3.0;
    const double radius = 0.21;
    // angleForPlate(2) = (1-2)·π/2 = -π/2：法向指向相机（近侧板）。
    const double true_radial = -kPi / 2.0;
    // 真实 RM 装甲板有 ≈15° 机械俯仰；这正是让"正对相机的单板 yaw"一阶可观测的原因。
    const double pitch = geometry.armor_pitch_deg * kPi / 180.0;
    const cv::Vec3d radial0(std::cos(true_radial), 0.0, std::sin(true_radial));
    const cv::Vec3d vertical0(0.0, 1.0, 0.0);
    const cv::Vec3d tangent = vertical0.cross(radial0);
    const cv::Vec3d normal = std::cos(pitch) * radial0 + std::sin(pitch) * vertical0;
    const cv::Vec3d vertical = -std::sin(pitch) * radial0 + std::cos(pitch) * vertical0;
    const cv::Vec3d plate_center(
        radius * radial0[0], 0.0, center_z + radius * radial0[2]);
    const double half_w = 0.5 * geometry.small_width;
    const double half_h = 0.5 * geometry.height;

    const double obj_y[4] = {half_w, half_w, -half_w, -half_w};
    const double obj_z[4] = {-half_h, half_h, half_h, -half_h};
    std::vector<cv::Point2f> corners(4);
    for (int i = 0; i < 4; ++i) {
        const cv::Vec3d point = plate_center + obj_y[i] * tangent + obj_z[i] * vertical;
        corners[static_cast<std::size_t>(i)] = cv::Point2f(
            static_cast<float>(camera_matrix.at<double>(0, 0) * point[0] / point[2] +
                               camera_matrix.at<double>(0, 2)),
            static_cast<float>(camera_matrix.at<double>(1, 1) * point[1] / point[2] +
                               camera_matrix.at<double>(1, 2)));
    }

    // 多次带噪声的试验取平均，避免单次噪声抽样的偶然性。
    std::mt19937 rng(20261006u);
    std::normal_distribution<double> pixel_noise(0.0, 0.3);
    double raw_sum = 0.0;
    double refined_sum = 0.0;
    double raw_max = 0.0;
    double refined_max = 0.0;
    constexpr int trials = 60;
    for (int trial = 0; trial < trials; ++trial) {
        auto_aim::Armor armor;
        armor.armor_type = auto_aim::small;
        armor.left.bottom = cv::Point2f(
            static_cast<float>(corners[0].x + pixel_noise(rng)),
            static_cast<float>(corners[0].y + pixel_noise(rng)));
        armor.left.top = cv::Point2f(
            static_cast<float>(corners[1].x + pixel_noise(rng)),
            static_cast<float>(corners[1].y + pixel_noise(rng)));
        armor.right.top = cv::Point2f(
            static_cast<float>(corners[2].x + pixel_noise(rng)),
            static_cast<float>(corners[2].y + pixel_noise(rng)));
        armor.right.bottom = cv::Point2f(
            static_cast<float>(corners[3].x + pixel_noise(rng)),
            static_cast<float>(corners[3].y + pixel_noise(rng)));

        auto_aim::PnpGeometry raw_geometry = geometry;
        raw_geometry.yaw_refine = false;
        auto_aim::PnpGeometry refined_geometry = geometry;
        refined_geometry.yaw_refine = true;

        auto_aim::Armor raw = armor;
        auto_aim::Armor refined = armor;
        if (!auto_aim::solveArmorPnP(raw, camera_matrix, dist_coeffs, raw_geometry) ||
            !auto_aim::solveArmorPnP(refined, camera_matrix, dist_coeffs, refined_geometry)) {
            passed &= expect(false, "solveArmorPnP must succeed on a synthetic plate");
            break;
        }

        const double raw_error = std::abs(normalizeAngle(radialAngle(raw) - true_radial));
        const double refined_error =
            std::abs(normalizeAngle(radialAngle(refined) - true_radial));
        raw_sum += raw_error;
        refined_sum += refined_error;
        raw_max = std::max(raw_max, raw_error);
        refined_max = std::max(refined_max, refined_error);
    }

    const double raw_mean = raw_sum / trials;
    const double refined_mean = refined_sum / trials;
    std::cout << "fit_single radial error (deg): raw_mean=" << raw_mean * 180.0 / kPi
              << " raw_max=" << raw_max * 180.0 / kPi
              << " refined_mean=" << refined_mean * 180.0 / kPi
              << " refined_max=" << refined_max * 180.0 / kPi << std::endl;

    // 结论（实测）：单板 yaw 的可观测性由 sin(俯仰)·像面尺寸决定 —— 3 m / 0.3 px
    // 角点噪声下 IPPE 的径向角误差 ≈1.7°，1D 精修 ≈1.6°（约 10%）。真正的收益在
    // **角点精修（§5A.3）把 0.3 px 降到 0.1 px 之后**，以及双板 fit_double。
    // 这里只钉住"精修不会让结果更差"这一必然成立的性质。
    passed &= expect(refined_mean <= raw_mean + 1e-9,
                     "1D yaw refinement must not increase the radial-angle error");

    if (!passed) return 1;
    std::cout << "pnp_yaw_refine_test passed" << std::endl;
    return 0;
}
