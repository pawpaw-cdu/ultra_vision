#ifndef AUTO_AIM_PERCEPTION_ARMOR_SOURCE_HPP
#define AUTO_AIM_PERCEPTION_ARMOR_SOURCE_HPP

// 装甲板观测源：**两个入口（真机 node.cpp / 仿真 node_sim.cpp）共用的检测前端**。
//
// 抽出来的原因：识别这块以前只在 node_sim 里写了一遍（异步神经网络 + 动态 ROI +
// 阶梯重捕 + 批量 PnP），真机入口 node.cpp 还在跑老的灯条流程 —— 识别上的所有优化
// 都到不了真机。现在两边都只调这一个组件；换检测器/调 ROI 只改一处。
//
// 它负责：
//   · 传统灯条检测（Detector）与神经网络检测（AsyncArmorDetector）二选一；
//   · 神经网络的**动态 ROI**（跟住估计出的四块板）与未锁定时的**居中阶梯重捕**；
//   · 对每块检测做 PnP（solveArmorPnP），产出直接可进估计器的 Armor；
//   · 记录"这批观测来自哪一帧"（异步推理时可能是几帧之前的图）。
// 它**不负责**：时间对齐/姿态反旋/估计/控制 —— 那些在 AimPipeline 与各入口里。

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

#include <opencv2/opencv.hpp>

#include "perception/detector.hpp"
#include "perception/pnp_solver.hpp"
#include "common/types.hpp"
#include "Kalman/tracker.hpp"

#ifdef ULTRA_VISION_USE_OPENVINO
#include "perception/async_detector.hpp"
#include "perception/nn_detector.hpp"
#endif

namespace auto_aim
{
    struct ArmorSourceConfig
    {
        DetectorConfig classical;
        PnpGeometry pnp;
        bool use_neural = false;
        bool dynamic_roi = true;
#ifdef ULTRA_VISION_USE_OPENVINO
        NnDetectorConfig neural;
#endif
    };

    struct ArmorSourceResult
    {
        std::vector<Armor> armors;      // 已做 PnP 的检测
        std::vector<Light> lights;      // 传统流程的灯条（显示用）
        int pnp_count = 0;
        bool fresh = false;             // 本帧是否拿到了**新**的检测结果
        uint64_t timestamp_us = 0;      // 观测对应的帧时间戳（异步时是旧帧的时间戳）
    };

    class ArmorSource
    {
    public:
        ArmorSource(const ArmorSourceConfig& config,
                    const cv::Mat& camera_matrix, const cv::Mat& dist_coeffs);

        /// @brief 仿真里处理尺寸会变，内参要跟着重建；真机一次设定。
        void setIntrinsics(const cv::Mat& camera_matrix, const cv::Mat& dist_coeffs);

        /// @brief 送入一帧，取回当前可用的观测。
        /// @param predicted_plates 估计器预测的四块板（相机系，供动态 ROI 用；可空）
        /// @param tracker_state    估计器状态（未锁定时改用居中阶梯重捕）
        /// @param image_will_be_modified 调用方是否还要在这张图上画东西
        ///        （是 → 神经网络的异步线程需要独立副本，避免数据竞争）
        ArmorSourceResult update(const cv::Mat& image, uint64_t sequence,
                                 uint64_t frame_timestamp_us,
                                 const std::array<cv::Mat, 4>& predicted_plates,
                                 TrackerState tracker_state, bool image_will_be_modified);

#ifdef ULTRA_VISION_USE_OPENVINO
        bool neuralEnabled() const { return neural_ != nullptr; }
        /// 神经网络检测器的统计（同步流程返回 0）
        double neuralFps() const { return neural_ ? neural_->fps() : 0.0; }
        double neuralLatencyMs() const { return neural_ ? neural_->latencyMs() : 0.0; }
#else
        bool neuralEnabled() const { return false; }
        double neuralFps() const { return 0.0; }
        double neuralLatencyMs() const { return 0.0; }
#endif

    private:
        cv::Rect inferenceRoi(const cv::Mat& image,
                              const std::array<cv::Mat, 4>& predicted_plates,
                              TrackerState tracker_state) const;

        ArmorSourceConfig config_;
        cv::Mat camera_matrix_;
        cv::Mat dist_coeffs_;
        Detector detector_;
        std::vector<Light> lights_;
#ifdef ULTRA_VISION_USE_OPENVINO
        std::unique_ptr<AsyncArmorDetector> neural_;
#endif
    };
} // namespace auto_aim

#endif // AUTO_AIM_PERCEPTION_ARMOR_SOURCE_HPP
