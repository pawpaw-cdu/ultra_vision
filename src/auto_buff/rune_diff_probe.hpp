#ifndef AUTO_AIM_ENERGY_RUNE_DIFF_PROBE_HPP
#define AUTO_AIM_ENERGY_RUNE_DIFF_PROBE_HPP

// 帧差探针（差分法）：在相机运动下用"配准后帧间差"找"又亮又在动"的点亮扇叶。
//
// 为什么需要它：网络只检测"亮起区域"，本平台实测点亮窗口内只有 ~30% 的帧能检出，
// 观测断流是当前最大的误差来源。差分法是很多队伍在实车上用的补充手段：
// 静止的亮区（地面反光等）在帧间差里相互抵消，只有"亮且在动"的扇叶会留下差分量。
//
// 关键前提（实测）：
//   * 相机**静止**时效果极好（仿真静态机位：100% 帧有候选，与网络同时检出的帧
//     位置差中位 1.5 px）；
//   * 相机**运动**时必须先做全局运动补偿，否则整幅背景都被当成变化
//     （实拍素材实测：连通域面积 15k~110k px²、与网络位置差中位 ~190 px）。
//     这里用相位相关估计全局平移并补偿。
//
// 探针只输出"候选方位 + 强度"，不做身份判定（谁亮/是否已打过）——那仍然要靠
// 网络类别 + 相位关联（见 rune_detector 的 phase_assoc）。

#include <opencv2/opencv.hpp>
#include <optional>

namespace auto_aim::energy
{
    struct RuneDiffProbeConfig
    {
        bool enabled = false;
        double threshold = 20.0;        // 帧间差二值化阈值（0~255）
        double min_area = 6.0;          // 连通域最小面积（px²）
        double ring_low = 0.45;         // 候选必须落在 [low, high] × 轨道半径
        double ring_high = 1.8;
        bool compensate_motion = true;  // 云台运动时必备
        double max_shift_px = 60.0;     // 相位相关给出的平移超过它就认为配准失败
    };

    struct RuneDiffCandidate
    {
        cv::Point2f center{0.0f, 0.0f};
        double area = 0.0;
        double radius_px = 0.0;   // 到给定圆心的距离
        double strength = 0.0;    // 连通域面积（当前排序依据）
    };

    class RuneDiffProbe
    {
    public:
        explicit RuneDiffProbe(RuneDiffProbeConfig config = {}) : config_(config) {}

        void setConfig(const RuneDiffProbeConfig& config) { config_ = config; }
        const RuneDiffProbeConfig& config() const { return config_; }

        /// @brief 处理一帧。hub/radius 由调用方给（来自网络观测或状态估计）；
        ///        hub 无效（x<0）时不做环带过滤，只找最强连通域。
        /// @return 最可能的点亮扇叶候选（没有则 nullopt）。
        std::optional<RuneDiffCandidate> update(const cv::Mat& bgr, cv::Point2f hub,
                                               double orbit_radius_px);

        void reset() { previous_gray_.release(); }

    private:
        RuneDiffProbeConfig config_;
        cv::Mat previous_gray_;
    };
} // namespace auto_aim::energy

#endif // AUTO_AIM_ENERGY_RUNE_DIFF_PROBE_HPP
