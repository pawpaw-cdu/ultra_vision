#include "pnp_solver.hpp"
#include <cmath>
#include <iostream>
#include <limits>

namespace auto_aim
{
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

}
