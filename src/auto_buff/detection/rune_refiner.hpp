#ifndef AUTO_AIM_ENERGY_RUNE_REFINER_HPP
#define AUTO_AIM_ENERGY_RUNE_REFINER_HPP

// 能量机关观测精修：把"五点关键点"升级成"轮廓几何 + 可用性判据"。
//
// 来源与取舍：思路来自深大 RP-26Rune 的 RuneObservationRefiner
// （颜色差掩膜 → 语义分割成 装甲板/灯臂/R标 → 描述子可用性），但他们整套
// 链路（chamfer 拟合 + 平面窗口 + LM-IRLS）耦合了他们自己的相机/光照参数，
// 直接搬会踩阈值坑（实测他们的 R−B 阈值 50 在我们仿真器画面上只剩 11% 可用）。
// 所以这里只引入两件"平台无关、收益可测"的东西：
//
//   1. 描述子可用性闸门：轮廓实心度/长宽比/贴边 —— 用来判定"这一帧的几何能
//      不能信"，可以直接挡掉明显误检（我们现有链路只有亮点裕度一个判据）。
//   2. 轮廓圆心与关键点圆心的融合：R 标轮廓的 fitEllipse 中心与网络 k2 是
//      两个独立估计，二者一致（差值 ≤ max_center_gap_px）时加权平均能降噪；
//      不一致时不动 k2（宁可保留原值，也不引入误检）。
//
// 全部阈值都在 configs/buff.yaml 的 refiner 段，默认 enabled=false，便于 A/B。

#include <opencv2/opencv.hpp>
#include <vector>

namespace auto_aim::energy
{
    struct RuneRefinerConfig
    {
        bool enabled = false;
        // ROI：五点 minAreaRect × roi_scale（与深大一致）
        double roi_scale = 1.4;
        // 掩膜：ROI 内 (红方) R−B / (蓝方) B−R 二值化阈值
        double red_minus_blue_threshold = 30.0;
        double blue_minus_red_threshold = 62.0;
        // 语义分割：装甲板轮廓面积与网络椭圆的相对误差上界
        double armor_area_relative_error = 0.35;
        int light_arm_line_samples = 15;
        // 描述子门槛（实心度 = 轮廓面积/凸包面积；长宽比 = minAreaRect 长短边比）
        double armor_min_solidity = 0.80;
        double arm_min_solidity = 0.66;
        double arm_expect_aspect = 5.0;
        double arm_aspect_tolerance = 0.42;
        // 灯臂不可用（或根本没找到灯臂）时是否丢弃该候选
        bool require_usable = false;
        // 圆心融合
        bool fuse_center = true;
        /// 融合的**前置条件**：是否要求装甲板轮廓先过描述子闸门（`armor_usable`）。
        /// 实测两边相反（§42）：仿真开有益（p90 35.8→25.4 px），但实拍上深大的
        /// solidity/aspect 阈值一次都过不了 ⇒ 融合会被完全关死（0/625 帧），
        /// 反而丢掉融合带来的稳定性收益。**按部署目标（实车）默认关**。
        bool fuse_requires_armor_usable = false;
        double fuse_weight_contour = 0.5;
        double max_center_gap_px = 4.0;
        int border_margin_px = 2;
    };

    /// @brief 一片扇叶的精修结果（全部是诊断/决策用的中间量）。
    struct RuneBladeRefinement
    {
        bool has_armor = false;
        bool has_arm = false;
        bool has_center_r = false;
        bool armor_usable = false;
        bool arm_usable = false;
        bool near_border = false;
        bool usable = false;      // 装甲板 + 灯臂 + R 标齐备且都可用
        bool center_fused = false; // 本帧是否真的用了融合圆心
        cv::Point2f armor_center{0.0f, 0.0f};
        cv::Point2f contour_center{0.0f, 0.0f};
        cv::Point2f refined_center{0.0f, 0.0f};
        double armor_solidity = -1.0;
        /// 装甲板轮廓面积 / 网络四点面积（实拍标定用：真实圆盘实测 0.16~0.26）
        double armor_area_ratio = -1.0;
        /// 装甲板轮廓圆度 4πA/P²（实拍实测：真实圆盘 0.41~0.46，支架 0.05，背景 0.01）
        double armor_circularity = -1.0;
        double arm_solidity = -1.0;
        double arm_aspect = -1.0;
        double center_gap_px = -1.0;
    };

    class RuneRefiner
    {
    public:
        explicit RuneRefiner(RuneRefinerConfig config = {}) : config_(config) {}

        const RuneRefinerConfig& config() const { return config_; }
        void setConfig(const RuneRefinerConfig& config) { config_ = config; }

        /// @brief 精修一片扇叶。
        /// @param bgr 原图；@param keypoints 五点（index 2 = R 标）；
        /// @param our_color 0=红（看 R−B）1=蓝（看 B−R）。
        RuneBladeRefinement refine(const cv::Mat& bgr,
                                   const std::vector<cv::Point2f>& keypoints,
                                   int our_color) const;

        /// @brief 在 refine() 基础上给出"建议使用的圆心"：融合成功用融合值，
        ///        否则返回传入的 keypoint_center（= 网络 k2）。
        cv::Point2f refineCenter(const cv::Mat& bgr,
                                 const std::vector<cv::Point2f>& keypoints,
                                 cv::Point2f keypoint_center,
                                 int our_color,
                                 RuneBladeRefinement* detail = nullptr) const;

    private:
        RuneRefinerConfig config_;
    };
} // namespace auto_aim::energy

#endif // AUTO_AIM_ENERGY_RUNE_REFINER_HPP
