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
        // 宽、高各自跟 PnP 距离比一次：**任一维度自洽就放行**。
        // 为什么不能像原来那样"两个隐含距离取平均再比"：
        //   · 宽度那条用标定过的等效宽度（0.1315 m），远近都准；
        //   · 高度那条硬套板高 0.06 m，而近距离大靶面下网络关键点的竖直跨度
        //     并不服从这个模型 —— 实测 1 m 工况：宽度隐含 ≈1.02 m（真值 1.00 m ✓）、
        //     高度隐含 ≈0.46 m ✗，两者一平均就把**好观测**判成不一致。
        // 这正是 NUC 上 1 m 工况 scale 拒收 ~70%、跟踪一直 TEMP_LOST 的原因。
        // 远距那种荒唐解（7.5 m 处 PnP 给 0.34 m）两个维度会同时不符，
        // 照样拦得住，所以闸门原本的目的没有丢。
        return ratio_of(distance, implied_width) <= tolerance ||
               ratio_of(distance, implied_height) <= tolerance;
    }

    void Tracker::refineArmorPose(Armor& armor) const
    {
        // **默认关**：实测它没有收益 —— 4 个角点 + 6 自由度时 IPPE 已经是精确最小二乘
        // （重投影残差中位 0.09 px），LM 精修"无残差可降"；A/B 里 observation error
        // 0.087 → 0.095 m（差异在噪声内）。UV 的价值不在"精修同一块板的位姿"，
        // 而在更多约束/更好的噪声模型，见 docs/uv_observation.md。
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
            // LM 迭代重投影：初值来自 PnP，只优化这一块板的位姿（无关联/分支问题）。
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
        // **点**变换：必须和 pointToWorld() 严格互逆（含平移）。
        // 原来这里错写成 worldToCameraVector(value)（只旋转、不减平移），于是所有"模型板
        // 投回相机系"的结果都恒定偏一个 |t|（本车外参 |t|=146 mm，实测偏差 145.4 mm 完全
        // 吻合）——显示、诊断全被它骗了；滤波器本身（世界系）是对的。
        return toMat(pointToCamera(camera_pose_, value));
    }
}
