#ifndef AUTO_BUFF_RUNE_BLADE_STATE_HPP
#define AUTO_BUFF_RUNE_BLADE_STATE_HPP

// 扇叶状态分类（经典特征）：**网络/几何给 ROI，ROI 内用传统图像处理判状态**
// —— 华南虎那份能量机关 demo 的做法（`enum isActive {ACTIVE, NOACTIVE, NOTHING}`，
// 用点亮图案的**轮廓面积区间**分类：NOACTIVE 7000~11000 / ACTIVE 13000~15000）。
//
// 为什么要它：我们的网络 class 输出是瓶颈（class0 可用率随激活片数掉到 23%/43%/17%），
// 而"点亮图案的面积/颜色"是**可以直接测的几何量**：实拍上两种点亮的区别很明确 ——
//   未激活（该打）：方/圆靶心 + 梯状点阵灯臂，LED 流动 ⇒ 面积随时间变化；
//   已激活（打过）：圆环+圆点+两侧"翅膀" + 橙色灯臂框 ⇒ 面积约 1.4~1.8 倍；
//   未点亮：灰色风扇盘，R−B ≈ 0。
//
// 尺度无关：面积特征用**轨道半径²**归一化（SCUT 的固定阈值是配他们的相机/距离的，
// 我们不能照抄数值，但可以照抄这个形式）。
//
// 输出三态：Unlit（未点亮）/ Inactive（未激活，该打）/ Active（已激活，打过）。

#include <opencv2/opencv.hpp>

namespace auto_aim::energy
{
    class RuneBladeState
    {
    public:
        enum class State
        {
            Unknown,
            Unlit,      // 没点亮（灰盘）
            Inactive,   // 未激活：该打的那片
            Active,     // 已激活：已经打过的片
        };

        struct Config
        {
            /// 点亮图案的颜色判据（红/橙 LED：R − B 超过它）
            double red_minus_blue = 60.0;
            /// ROI 半尺寸 = 该比例 × 轨道半径（靶面尺度 ≈ 0.21×轨道半径，见 §14）
            double roi_half_ratio = 0.35;
            /// 轨道半径与靶面半径之比（0.214 = 靶面 0.127 m / 回转半径 0.700 m 的
            /// 实测折算，§14 量到 62.2 px 靶面 ↔ 291 px 轨道）。有了它，"轨道
            /// 半径"可以直接由**本片自己的四个靶面点**反推，不必再去问上一帧的
            /// 圆心估计——跨帧混用几何是 lit_ratio 退化成常数的根因。
            double plate_orbit_ratio = 0.214;
            /// 状态阈值（**归一化点亮面积 = 点亮像素 / 轨道半径²**，待按标注帧标定）
            double unlit_below = 0.06;    // 低于它 = 未点亮
            double active_above = 0.22;   // 高于它 = 已激活（SCUT 的面积比 1.4~1.8 折算）
            /// 灯臂点亮判据：靶心→圆心连线上采样的点亮比例
            int arm_samples = 9;
            double arm_lit_fraction = 0.5;   // 采样点里点亮超过这个比例就算灯臂亮
        };

        struct Result
        {
            State state = State::Unknown;
            double lit_ratio = -1.0;      // 点亮面积 / 轨道半径²
            double arm_lit_ratio = -1.0;  // 灯臂上点亮的采样比例
            // 诊断（标定阈值必须看这三个量）：归一化用的轨道半径、ROI 边长、
            // 以及 ROI 内的点亮像素数。lit_ratio 是"比值"，一旦 orbit/ROI 取错
            // 就会变成一个常数，光看比值查不出来。
            double orbit_px = -1.0;
            double lit_area = -1.0;
            int roi_px = 0;
            bool valid = false;
        };

        RuneBladeState() = default;
        explicit RuneBladeState(const Config& config) : config_(config) {}

        /// @param hub 圆心肌像素 @param blade 该片靶心像素 @param orbit_px 轨道半径像素
        Result classify(const cv::Mat& bgr, cv::Point2f hub, cv::Point2f blade,
                        double orbit_px) const;

        const Config& config() const { return config_; }
        static const char* name(State state);

    private:
        Config config_;
    };
} // namespace auto_aim::energy

#endif // AUTO_BUFF_RUNE_BLADE_STATE_HPP
