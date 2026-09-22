#ifndef PNP_SOLVER_HPP
#define PNP_SOLVER_HPP

#include <opencv2/opencv.hpp>
#include "common/types.hpp"

namespace auto_aim
{
    // 装甲板等效物点尺寸（PnP 与显示共用）。数值是**等效尺寸**，不是标准板尺寸：
    // 网络的四个关键点不是 135x125 板的四角，见 configs/tracker.yaml 的 pnp 段注释。
    struct PnpGeometry
    {
        float small_width = 0.135f;
        float large_width = 0.225f;
        float height = 0.055f;
        // ---- PnP 降自由度：1D yaw 精修（fit_single，上科大 2026 §5A.4）--------
        // 标准 solvePnP 同时估 6 个自由度，角点噪声会在平移/朝向之间分摊，
        // yaw 的估计因此变差；而下游 EKF 的 armor_yaw（径向角）正是由法向决定的。
        // 做法：固定 PnP 的 tvec 与"板竖直、法向水平"的机械约束，只在方位角 ψ
        // 上做一维搜索，最小化四角重投影误差。1D 搜索几乎不可能发散。
        bool yaw_refine = true;
        float yaw_refine_span_deg = 70.0f;    // 在 PnP 初值两侧的搜索半宽
        float yaw_refine_coarse_deg = 5.0f;   // 粗扫步长（随后 0.1° 细扫）
        // 装甲板的机械俯仰角（RM 规范 ≈15°）。**必须与实车一致**：
        // pitch=0 时正对相机的单板 yaw 是二阶可观测（数值上几乎不可辨识），
        // 15° 俯仰才把它变成一阶可观测，1D 精修才有意义。符号由实车标定决定。
        float armor_pitch_deg = 15.0f;
    };

    extern cv::Mat CAMERA_MATRIX;
    extern cv::Mat DIST_COEFFS;

    std::vector<cv::Point3f> getArmor3DPoints(int type, float small_width, float large_width, float height);
    bool solveArmorPnP(auto_aim::Armor& armor,
                       const cv::Mat& camera_matrix,
                       const cv::Mat& dist_coeffs,
                       float small_width = 0.135f,
                       float large_width = 0.225f,
                       float height = 0.055f);

    /// @brief 同上，但直接吃完整的 PnpGeometry（含 1D yaw 精修参数）。
    bool solveArmorPnP(auto_aim::Armor& armor,
                       const cv::Mat& camera_matrix,
                       const cv::Mat& dist_coeffs,
                       const PnpGeometry& geometry);

    /// @brief 在固定 tvec 的前提下，只用四角重投影误差优化装甲板的方位角 ψ
    ///        （fit_single）。成功后把重构出的 rvec 写回 armor（法向 = 径向、竖直 =
    ///        相机 y），使 `armorRadialAngle` 直接得到精修后的径向角。
    ///        返回是否有过精修（角点/位姿不合法时返回 false 并保持原状）。
    bool refineArmorYaw(auto_aim::Armor& armor,
                        const cv::Mat& camera_matrix,
                        const cv::Mat& dist_coeffs,
                        const PnpGeometry& geometry);
}

#endif
