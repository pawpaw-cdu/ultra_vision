#include "pnp_solver.hpp"
#include "common/armor_geometry.hpp"
#include <cmath>
#include <iostream>
#include <limits>

namespace auto_aim
{
    namespace
    {
        constexpr double kPi = 3.14159265358979323846;

        // 基向量已提取到 common/armor_geometry.hpp 的 armorKeypointBasis()：
        // `pose_2d_solver` 也要用**同一份** ψ 约定，不能各写一份
        // （历史上"整车模型在三个地方各写一套、各错一点"就是这么来的）。
        using auto_aim::armorKeypointBasis;

        /// @brief 给定 ψ 时四角在**归一化像平面**上的重投影误差平方和（按 fx/fy 折回像素）。
        ///        归一化坐标用 undistortPoints 预先算好，所以这里只做针孔投影，
        ///        畸变不会重复计入；对零畸变相机与直接投影等价。
        double yawReprojectionCost(double psi, const cv::Vec3d& tvec,
                                   double pitch,
                                   double half_width, double half_height,
                                   double fx, double fy,
                                   const std::vector<cv::Point2f>& normalized)
        {
            cv::Vec3d normal, tangent, vertical;
            armorKeypointBasis(psi, pitch, normal, tangent, vertical);
            // 物点顺序（getArmor3DPoints）：lb, lt, rt, rb
            //   P = tvec + obj.y·tangent + obj.z·vertical
            const double obj_y[4] = {half_width, half_width, -half_width, -half_width};
            const double obj_z[4] = {-half_height, half_height, half_height, -half_height};
            double cost = 0.0;
            for (int i = 0; i < 4; ++i) {
                const cv::Vec3d point = tvec + obj_y[i] * tangent + obj_z[i] * vertical;
                if (point[2] <= 1e-3) return 1e12;   // 跑到相机背面：非法
                const double du = fx * (point[0] / point[2] - normalized[i].x);
                const double dv = fy * (point[1] / point[2] - normalized[i].y);
                cost += du * du + dv * dv;
            }
            return cost;
        }

        cv::Mat rotationFromYaw(double psi, double pitch)
        {
            cv::Vec3d normal, tangent, vertical;
            armorKeypointBasis(psi, pitch, normal, tangent, vertical);
            // 列 = 物体 x(法向) / y(切向) / z(竖直)，与 getArmor3DPoints 对应。
            return (cv::Mat_<double>(3, 3) <<
                normal[0], tangent[0], vertical[0],
                normal[1], tangent[1], vertical[1],
                normal[2], tangent[2], vertical[2]);
        }
    }

    cv::Mat CAMERA_MATRIX = (cv::Mat_<double>(3,3) << 1286.0, 0.0, 645.0,
                                                     0.0, 1288.0, 483.0,
                                                     0.0, 0.0, 1.0);
    cv::Mat DIST_COEFFS = cv::Mat::zeros(5, 1, CV_64F);

    std::vector<cv::Point3f> getArmor3DPoints(int type, float small_width, float large_width, float height)
    {
        float half_w = (type == 0) ? small_width / 2.0f : large_width / 2.0f;
        float half_h = height / 2.0f;
        return {
            cv::Point3f(0,  half_w, -half_h), // left_bottom
            cv::Point3f(0,  half_w,  half_h), // left_top
            cv::Point3f(0, -half_w,  half_h), // right_top
            cv::Point3f(0, -half_w, -half_h)  // right_bottom
        };
    }

    namespace
    {
        /// @brief 一次 solvePnP（IPPE）+ 一次 1D yaw 精修。两个公开重载共用。
        bool solveAndRefine(auto_aim::Armor& armor, const cv::Mat& camera_matrix,
                            const cv::Mat& dist_coeffs, const PnpGeometry& geometry)
        {
            const int type = (armor.armor_type == auto_aim::small) ? 0 : 1;
            const std::vector<cv::Point3f> obj_points = getArmor3DPoints(
                type, geometry.small_width, geometry.large_width, geometry.height);
            const std::vector<cv::Point2f> img_points = {
                armor.left.bottom, armor.left.top,
                armor.right.top, armor.right.bottom};

            // ---- 平面板选解（学吉林大学 sortPnPResult + RMCS 的机械约束剪枝）-------
            // 背景：平面板的近解/远解重投影几乎一样（实车实测同一块板：1.43 m / 18.56 m，
            // 残差都在 1 px 量级），只靠像素挑不出来；但"板是装在底盘上的"这条**机械约束**
            // 能挑。参考实现的做法：
            //   · 吉林大学：先看两解的误差比，接近时用 RPY/期望 pitch 选；
            //   · RMCS：丢掉 z<=0、法向背对相机、|pitch−期望|>40° 的解，剩下取误差最小。
            // 我们这里用相机系里可算的两条等价判据：
            //   ① 尺度自洽：实测板宽像素 ≈ fx·W/z（远解会"又远又大"，比例差好几倍）；
            //   ② 板大致竖直：板平面内竖直方向（对象点 y 轴）在相机系里的 |R[1][1]| ≈ 1
            //      （远解要求板近乎倒伏，等价 RMCS 的 pitch 偏差剪枝）。
            // **两条都只作为打分偏好，不返回失败**：解失败在下游处理不完整（实测会让
            // auto_aim_sim 段错误），所以这里永远给出一个解（最差 = 残差最小的那个）。
            // reprojectionError 用 std::vector<double> 接（按 Mat 读类型不对会崩）。
            std::vector<cv::Mat> rvecs;
            std::vector<cv::Mat> tvecs;
            std::vector<double> reproj_errs;
            if (!cv::solvePnPGeneric(obj_points, img_points, camera_matrix, dist_coeffs,
                                     rvecs, tvecs, false, cv::SOLVEPNP_IPPE, cv::noArray(),
                                     cv::noArray(), reproj_errs) ||
                rvecs.empty() || tvecs.size() != rvecs.size()) {
                return false;   // 这里失败是"完全没解"，与"选不出好解"不同
            }
            constexpr double kMinScaleRatio = 0.45;
            constexpr double kMaxScaleRatio = 1.50;
            constexpr double kMinUpright = 0.707;      // cos45°
            constexpr double kBadPenalty = 1000.0;     // 只影响排序，不会导致失败
            const double plate_w = (armor.armor_type == auto_aim::small)
                ? static_cast<double>(geometry.small_width)
                : static_cast<double>(geometry.large_width);
            const double fx = camera_matrix.at<double>(0, 0);
            const double observed_px = cv::norm(img_points[1] - img_points[2]);
            std::size_t chosen = 0;
            double best_score = std::numeric_limits<double>::max();
            for (std::size_t i = 0; i < rvecs.size(); ++i) {
                const double z = tvecs[i].at<double>(2);
                const double err = i < reproj_errs.size() ? reproj_errs[i] : 0.0;
                double score = std::isfinite(err) ? err : kBadPenalty;
                if (!std::isfinite(z) || z <= 0.1) {
                    score += kBadPenalty;
                } else if (fx > 1.0 && observed_px > 1.0 && plate_w > 1e-3) {
                    const double ratio = observed_px / (fx * plate_w / z);
                    if (!std::isfinite(ratio) || ratio < kMinScaleRatio ||
                        ratio > kMaxScaleRatio) {
                        score += kBadPenalty;   // 尺度不自洽（18 m 那类远解）→ 排后面
                    }
                }
                cv::Mat rotation;
                cv::Rodrigues(rvecs[i], rotation);
                const double upright = std::abs(rotation.at<double>(1, 1));
                if (!(std::isfinite(upright) && upright >= kMinUpright)) {
                    score += kBadPenalty;       // 板不竖直（远解会倒伏）→ 排后面
                }
                if (score < best_score) { best_score = score; chosen = i; }
            }
            armor.rvec = rvecs[chosen];
            armor.tvec = tvecs[chosen];
            // 降自由度精修（fit_single，上科大 2026 §5A.4）：固定 tvec + 竖直板约束，
            // 只在方位角上做一维重投影最小化。失败时保留 IPPE 的结果。
            refineArmorYaw(armor, camera_matrix, dist_coeffs, geometry);
            return true;
        }
    }

    bool solveArmorPnP(auto_aim::Armor& armor,
                       const cv::Mat& camera_matrix,
                       const cv::Mat& dist_coeffs,
                       float small_width, float large_width, float height)
    {
        PnpGeometry geometry;
        geometry.small_width = small_width;
        geometry.large_width = large_width;
        geometry.height = height;
        return solveAndRefine(armor, camera_matrix, dist_coeffs, geometry);
    }

    bool solveArmorPnP(auto_aim::Armor& armor,
                       const cv::Mat& camera_matrix,
                       const cv::Mat& dist_coeffs,
                       const PnpGeometry& geometry)
    {
        return solveAndRefine(armor, camera_matrix, dist_coeffs, geometry);
    }

    bool refineArmorYaw(auto_aim::Armor& armor,
                        const cv::Mat& camera_matrix,
                        const cv::Mat& dist_coeffs,
                        const PnpGeometry& geometry)
    {
        if (!geometry.yaw_refine) return false;
        if (armor.tvec.empty() || armor.rvec.empty()) return false;
        if (armor.tvec.rows != 3 || camera_matrix.rows != 3) return false;

        const int type = (armor.armor_type == auto_aim::small) ? 0 : 1;
        const float width = (type == 0) ? geometry.small_width : geometry.large_width;
        const double half_width = 0.5 * static_cast<double>(width);
        const double half_height = 0.5 * static_cast<double>(geometry.height);

        const std::vector<cv::Point2f> image_points = {
            armor.left.bottom, armor.left.top, armor.right.top, armor.right.bottom};
        for (const cv::Point2f& point : image_points) {
            if (!std::isfinite(point.x) || !std::isfinite(point.y)) return false;
        }

        const cv::Vec3d tvec(
            armor.tvec.at<double>(0), armor.tvec.at<double>(1), armor.tvec.at<double>(2));
        if (!std::isfinite(tvec[0]) || !std::isfinite(tvec[1]) || !std::isfinite(tvec[2]) ||
            tvec[2] <= 1e-3) {
            return false;
        }

        // 角点 → 归一化像平面（一次），之后重投影只做针孔投影。
        std::vector<cv::Point2f> normalized;
        try {
            cv::undistortPoints(image_points, normalized, camera_matrix, dist_coeffs);
        } catch (const cv::Exception&) {
            return false;
        }
        if (normalized.size() != 4) return false;

        const double fx = camera_matrix.at<double>(0, 0);
        const double fy = camera_matrix.at<double>(1, 1);
        if (!(fx > 1.0) || !(fy > 1.0)) return false;

        // 初值：PnP 解出的法向方位角（与 tracker 的 armorRadialAngle 同口径地
        // 处理"可见面朝相机"造成的 ±π 折叠）。
        cv::Mat rotation;
        cv::Rodrigues(armor.rvec, rotation);
        cv::Vec3d normal(rotation.at<double>(0, 0), rotation.at<double>(1, 0),
                         rotation.at<double>(2, 0));
        if (normal.dot(tvec) > 0.0) normal = -normal;
        const double psi0 = std::atan2(normal[2], normal[0]);

        const double span = geometry.yaw_refine_span_deg * kPi / 180.0;
        const double coarse = std::max(0.5 * kPi / 180.0,
                                       geometry.yaw_refine_coarse_deg * kPi / 180.0);
        const double pitch = geometry.armor_pitch_deg * kPi / 180.0;
        double best_psi = psi0;
        double best_cost = std::numeric_limits<double>::max();
        for (double offset = -span; offset <= span + 1e-9; offset += coarse) {
            const double psi = psi0 + offset;
            const double cost = yawReprojectionCost(
                psi, tvec, pitch, half_width, half_height, fx, fy, normalized);
            if (cost < best_cost) {
                best_cost = cost;
                best_psi = psi;
            }
        }
        // 细扫：在最优点附近 0.1° 步长收敛（1D，成本极低）。
        const double fine = 0.1 * kPi / 180.0;
        for (double psi = best_psi - coarse; psi <= best_psi + coarse + 1e-9; psi += fine) {
            const double cost = yawReprojectionCost(
                psi, tvec, pitch, half_width, half_height, fx, fy, normalized);
            if (cost < best_cost) {
                best_cost = cost;
                best_psi = psi;
            }
        }
        if (!std::isfinite(best_cost) || best_cost >= 1e11) return false;

        cv::Rodrigues(rotationFromYaw(best_psi, pitch), armor.rvec);
        return true;
    }
}
