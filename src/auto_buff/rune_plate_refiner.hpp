#ifndef AUTO_BUFF_RUNE_PLATE_REFINER_HPP
#define AUTO_BUFF_RUNE_PLATE_REFINER_HPP

// 靶面几何精修：**网络给 ROI，经典角点提取在 ROI 内定几何**（华南虎那套思路，
// 本仓库自瞄侧的 `perception/keypoint_refiner` 已用同一模式且有效）。
//
// 为什么：网络回归的四个点在图像里只有十几到几十像素，横向误差直接进 PnP
// （实测距离差 2~9%）；而经典"阈值 + 轮廓 + 最小外接矩形"在**给定 ROI** 内
// 又准又稳（它只是在全图里召回差 —— 召回交给网络）。
//
// 与自瞄侧的区别：能量机关的四个关键点是靶面方框的**边中点**（§14 标定），
// 所以重建时取精修四边形的**四条边的中点**，顺序按"网络点相对四边形中心的方向"
// 一一对应（外→右→内→左），保证与 PnP 物点顺序一致。
//
// 安全性：只有几何一致性检查通过才采信（中心位移 ≤ 靶面尺寸的比例、尺寸变化在
// 允许区间），否则**原样返回网络结果** —— 与自瞄侧同一约定。

#include <array>

#include <opencv2/opencv.hpp>

namespace auto_aim::energy
{
    class RunePlateRefiner
    {
    public:
        struct Config
        {
            bool enabled = false;   // 实测会系统性拉远深度，见 rune_config/README 的说明
            /// ROI = 网络四点的外接框按该比例放大
            double margin_ratio = 0.25;
            /// ROI 内二值化：Otsu（亮区=靶面图案）或固定阈值
            bool use_otsu = true;
            int binary_threshold = 150;
            int morph_size = 3;
            /// 轮廓面积相对网络四边形面积的允许区间
            double min_area_ratio = 0.25;
            double max_area_ratio = 3.0;
            /// 中心位移上界（占靶面尺寸的比例）——超过就认为 ROI 锁到别的东西了
            double max_center_shift_ratio = 0.25;
            /// 精修后尺寸相对网络尺寸的允许比例区间（尺度保持模式下不使用）
            double min_size_ratio = 0.6;
            double max_size_ratio = 1.6;
            // ---- 尺度保持模式（§45，默认）------------------------------------
            // 经典的"亮区域"与网络标注的"图案边中点"**不是同一个几何量**（实测直接
            // 替换四点会把深度拉远 +9.6%~+24%）。所以只借用经典结果的**中心**（可选
            // **朝向**），四个点到中心的**相对几何（尺寸）保持网络原值**。
            bool preserve_network_scale = true;
            double center_weight = 0.5;      // 0=只用网络中心，1=只用经典中心
            double rotation_weight = 0.0;    // 0=不修朝向（先看中心的收益）
            double max_angle_diff_deg = 25.0;  // 朝向差超过它就放弃修朝向（方形的 90° 歧义）
        };

        struct Result
        {
            bool refined = false;
            std::array<cv::Point2f, 4> points{};   // 重建后的四个边中点（全图坐标）
            double area_ratio = -1.0;              // 轮廓面积 / 网络四边形面积
            double center_shift_px = 0.0;
            double size_ratio = -1.0;
        };

        RunePlateRefiner() = default;
        explicit RunePlateRefiner(const Config& config) : config_(config) {}

        /// @param keypoints 网络的四个边中点（k0 外、k1 右、k3 内、k4 左）
        Result refine(const cv::Mat& bgr, const std::array<cv::Point2f, 4>& keypoints) const;

        const Config& config() const { return config_; }

    private:
        Config config_;
    };
} // namespace auto_aim::energy

#endif // AUTO_BUFF_RUNE_PLATE_REFINER_HPP
