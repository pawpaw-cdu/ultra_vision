#pragma once
//
// 深大 RP-26Rune 五点检测器的 harness 封装。
// 实现见 rp26_detector.cpp（从他们的 NNDetector.cpp 复制，保留全部推理细节）。

#include <array>
#include <memory>
#include <vector>

#include <opencv2/core.hpp>

#include "power_rune_interface.hpp"

namespace rp26_sim
{
enum class Rp26RuneMode
{
    Small,
    Large,
};

/// @brief 一条网络结果（保留模型原始类别，映射留给调用方）。
struct Rp26Detection
{
    int model_class_id = -1; // 0 未击打 / 1 小符已击打 / 2 大符已击打
    float confidence = 0.0F;
    float quality = 0.0F;
    cv::Point2f center{0.0F, 0.0F};
    std::array<cv::Point2f, 5> keypoints{};
    std::array<float, 5> keypoint_confidences{};
};

class Rp26RuneDetector
{
public:
    Rp26RuneDetector();
    ~Rp26RuneDetector();

    Rp26RuneDetector(const Rp26RuneDetector &) = delete;
    Rp26RuneDetector &operator=(const Rp26RuneDetector &) = delete;

    std::vector<Rp26Detection> infer(const cv::Mat &image);

    /// @brief 按深大的模式映射规则转成 power_rune::RuneInput::NNRuneInfo。
    static std::vector<power_rune::RuneInput::NNRuneInfo> toRuneInfos(
        const std::vector<Rp26Detection> &detections,
        Rp26RuneMode mode);

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};
} // namespace rp26_sim
