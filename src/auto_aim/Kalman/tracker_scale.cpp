#include "tracker.hpp"
#include "Kalman/tracker_utils.hpp"
#include "perception/pnp_solver.hpp"

#include <cmath>
#include <cstdlib>
#include <string>

using namespace auto_aim::tracker_detail;

namespace auto_aim
{
    bool Tracker::scaleConsistent(const Armor& camera_armor, double distance) const
    {
        if (cfg_.scale_gate_ratio <= 0.0 || !camera_intrinsics_valid_) return true;
        if (camera_armor.Points_2D.size() < 4 || !(distance > 0.1)) return true;
        const double width_m = camera_armor.armor_type == auto_aim::small
                                   ? cfg_.uv_armor_small_width
                                   : cfg_.uv_armor_large_width;
        const double height_m = cfg_.uv_armor_height;
        const auto& p = camera_armor.Points_2D;   // lb, lt, rt, rb
        const double px_width = 0.5 * (cv::norm(p[1] - p[2]) + cv::norm(p[0] - p[3]));
        const double px_height = 0.5 * (cv::norm(p[1] - p[0]) + cv::norm(p[2] - p[3]));
        if (px_width < 1.0 || px_height < 1.0) return false;   // 关键点塌缩
        const double implied_width = camera_fx_ * width_m / px_width;
        const double implied_height = camera_fy_ * height_m / px_height;
        const auto ratio_of = [](double lhs, double rhs) {
            return std::max(lhs, rhs) / std::max(1e-3, std::min(lhs, rhs));
        };
        const double tolerance = 1.0 + cfg_.scale_gate_ratio;
        return ratio_of(distance, implied_width) <= tolerance ||
               ratio_of(distance, implied_height) <= tolerance;
    }

    void Tracker::refineArmorPose(Armor& armor) const
    {

        static const bool enabled = [] {
            const char* value = std::getenv("ULTRA_VISION_UV_POSE_REFINE");
            return value != nullptr && std::string(value) != "0";
        }();
        if (!enabled || !camera_intrinsics_valid_) return;
        if (!armor.solve_result || armor.tvec.empty() || armor.Points_2D.size() < 4) return;
        const auto object_points = getArmor3DPoints(
            armor.armor_type == auto_aim::small ? 0 : 1,
            static_cast<float>(cfg_.uv_armor_small_width),
            static_cast<float>(cfg_.uv_armor_large_width),
            static_cast<float>(cfg_.uv_armor_height));
        if (object_points.size() != 4) return;
        const std::vector<cv::Point2f> image_points = {
            armor.left.bottom, armor.left.top, armor.right.top, armor.right.bottom};
        cv::Mat camera_matrix = (cv::Mat_<double>(3, 3) << camera_fx_, 0.0, camera_cx_,
                                 0.0, camera_fy_, camera_cy_, 0.0, 0.0, 1.0);
        cv::Mat rvec = armor.rvec.clone();
        cv::Mat tvec = armor.tvec.clone();
        try {
            cv::solvePnPRefineLM(object_points, image_points, camera_matrix, cv::Mat(), rvec, tvec);
        } catch (const cv::Exception&) {
            return;   // 精修失败就保留 PnP 结果
        }
        if (cv::checkRange(tvec) && cv::checkRange(rvec)) {
            armor.rvec = rvec;
            armor.tvec = tvec;
        }
    }


    cv::Mat Tracker::worldToCamera(const Vec<3>& value) const
    {
        return toMat(pointToCamera(camera_pose_, value));
    }
}
