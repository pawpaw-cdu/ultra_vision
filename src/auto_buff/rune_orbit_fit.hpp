#ifndef AUTO_AIM_ENERGY_RUNE_ORBIT_FIT_HPP
#define AUTO_AIM_ENERGY_RUNE_ORBIT_FIT_HPP

// 回转轨道（椭圆）拟合：把"观测到的靶心像素"按时间窗口攒起来拟合椭圆。
//
// 为什么需要（实测根因）：测距原来用**瞬时**的"R 标像素 → 靶心像素"偏移，
// 除以 fx·R/offset 得距离。但回转圆在图像里是**椭圆**，而且 R 标的投影
// **不是椭圆中心**（透视下圆心的投影偏向远端）。于是偏移在远侧被压短、近侧被
// 拉长，测距误差只跟靶心的图像方位有关：上半/左侧 +20%，下半 −1%
// （见 docs/energy_rune_issue_audit.md §13 的分箱表）。这是系统性的几何误差，
// 调阈值/换模型都修不掉 —— 就是那个"修了几轮的长尾"。
//
// 椭圆拟合同时给出两样东西：
//   * 半长轴 a ⇒ 测距 d = fx · R / a（消掉foreshortening）；
//   * 椭圆中心 ⇒ 真正的"符心投影"（R 标像素 ≠ 它的投影）。
//
// 所有扇叶都在同一条轨道上，所以不同片、不同轮的样本可以混用；圆盘静止 ⇒
// 窗口内的拟合参数是常数，没有滞后问题（只在刚开始跟踪时退化为瞬时估计）。

#include <deque>

#include <opencv2/opencv.hpp>

namespace auto_aim::energy
{
    class RuneOrbitFit
    {
    public:
        struct Config
        {
            double window_s = 2.5;      // 参与拟合的时间窗口
            int min_samples = 8;        // 少于这么多样本不拟合
            /// 椭圆半长轴的合理范围（像素）：超出说明样本弧段太短/有野值。
            double min_semi_major_px = 15.0;
            double max_semi_major_px = 200.0;
            /// 半长轴/半短轴之比的合理范围（太扁说明样本几乎共线）。
            double min_axis_ratio = 0.35;
            /// 拟合出的椭圆中心离原点多远还能接受（相对向量在原点；偏得多说明
            /// 样本被云台平移污染）。
            double max_center_offset_px = 25.0;
            double outlier_gate = 2.5;  // 残差 > gate×中位残差的样本丢掉后重拟合
        };

        struct Result
        {
            bool valid = false;
            cv::Point2f center{0.0f, 0.0f};  // 椭圆中心（符心的投影）
            double semi_major_px = 0.0;
            double semi_minor_px = 0.0;
            double angle_deg = 0.0;
            int samples = 0;
        };

        RuneOrbitFit() = default;
        explicit RuneOrbitFit(const Config& config) : config_(config) {}

        /// @brief 记录一次"靶心像素"观测（同一轨道，不同片可混用）。
        void add(const cv::Point2f& blade_px, double now);
        /// @brief 按当前窗口拟合（样本不足/参数不合理时返回 valid=false）。
        Result fit(double now) const;
        void reset() { samples_.clear(); }
        int sampleCount() const { return static_cast<int>(samples_.size()); }

    private:
        struct Sample
        {
            cv::Point2f point;
            double time = 0.0;
        };
        std::vector<cv::Point2f> windowPoints(double now) const;
        static bool fitOnce(const std::vector<cv::Point2f>& points, Result& out);

        Config config_;
        std::deque<Sample> samples_;
    };
} // namespace auto_aim::energy

#endif // AUTO_AIM_ENERGY_RUNE_ORBIT_FIT_HPP
