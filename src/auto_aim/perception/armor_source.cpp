#include "armor_source.hpp"

#include <iostream>

#include "visualization/projection.hpp"
#ifdef ULTRA_VISION_USE_OPENVINO
#include "perception/async_detector.hpp"
#include "perception/nn_detector.hpp"
#endif

namespace
{
    // 未锁定时的**居中阶梯重捕**：7~8 m 处全帧只有 ~11 px，网络看不见，
    // 用逐级放大的中心裁剪把它"找出来"；一旦进入 TRACKING 自动交回估计 ROI。
    cv::Rect ladderRoi(const cv::Mat& image, uint64_t sequence)
    {
        static const double kLadder[] = {1.0, 0.7, 0.5, 0.35};
        const double scale = kLadder[(sequence / 3) % 4];
        if (scale >= 0.999) return {};
        return cv::Rect(static_cast<int>(image.cols * (1.0 - scale) / 2.0),
                        static_cast<int>(image.rows * (1.0 - scale) / 2.0),
                        static_cast<int>(image.cols * scale),
                        static_cast<int>(image.rows * scale));
    }
} // namespace

namespace auto_aim
{
    ArmorSource::ArmorSource(const ArmorSourceConfig& config,
                             const cv::Mat& camera_matrix, const cv::Mat& dist_coeffs)
        : config_(config), camera_matrix_(camera_matrix), dist_coeffs_(dist_coeffs),
          detector_(cv::Mat(), config.classical)
    {
#ifdef ULTRA_VISION_USE_OPENVINO
        if (config_.use_neural) {
            AsyncDetectorConfig async_cfg;
            async_cfg.detector = config_.neural;
            if (!async_cfg.detector.model_path.empty()) {
                neural_ = std::make_unique<AsyncArmorDetector>(async_cfg);
                std::cout << "Neural armor detector: " << async_cfg.detector.model_path
                          << " (device " << async_cfg.detector.device << ", async)"
                          << std::endl;
            } else {
                std::cerr << "神经网络检测器 enabled 但 model_path 为空；回退传统灯条流程"
                          << std::endl;
            }
        }
#else
        if (config_.use_neural) {
            std::cerr << "神经网络检测器 enabled 但本次构建没有 OpenVINO；回退传统灯条流程"
                      << std::endl;
        }
#endif
    }

    void ArmorSource::setIntrinsics(const cv::Mat& camera_matrix, const cv::Mat& dist_coeffs)
    {
        camera_matrix_ = camera_matrix;
        dist_coeffs_ = dist_coeffs;
    }

    cv::Rect ArmorSource::inferenceRoi(const cv::Mat& image,
                                       const std::array<cv::Mat, 4>& predicted_plates,
                                       TrackerState tracker_state) const
    {
        if (!config_.dynamic_roi) return {};
        // 1) 已锁定：只网络只看"四块板可能在的地方"
        std::vector<cv::Point2f> projected;
        for (const auto& position : predicted_plates) {
            if (position.rows != 3 || position.cols != 1) continue;   // 估计器未初始化时是空 Mat
            cv::Point2f point;
            if (projectPoint(position, camera_matrix_, dist_coeffs_, point)) {
                projected.push_back(point);
            }
        }
        if (projected.size() >= 2 && tracker_state != TrackerState::LOST) {
            cv::Rect box = cv::boundingRect(projected);
            const int margin = static_cast<int>(0.35 * std::max(box.width, box.height)) + 24;
            box.x -= margin;
            box.y -= margin;
            box.width += 2 * margin;
            box.height += 2 * margin;
            box &= cv::Rect(0, 0, image.cols, image.rows);
            if (box.width >= 96 && box.height >= 96) return box;
        }
        return {};   // 2) 未锁定 → 交给调用方用阶梯重捕（见 update）
    }

    ArmorSourceResult ArmorSource::update(const cv::Mat& image, uint64_t sequence,
                                          uint64_t frame_timestamp_us,
                                          const std::array<cv::Mat, 4>& predicted_plates,
                                          TrackerState tracker_state,
                                          bool image_will_be_modified)
    {
        ArmorSourceResult result;
        result.timestamp_us = frame_timestamp_us;
#ifdef ULTRA_VISION_USE_OPENVINO
        if (neural_) {
            cv::Rect roi = inferenceRoi(image, predicted_plates, tracker_state);
            if (roi.empty()) roi = ladderRoi(image, sequence);
            // submit **接管**像素（worker 在另一线程读），所以调用方还要画图时先复制一份。
            cv::Mat detector_input = image;
            if (image_will_be_modified) detector_input = image.clone();
            neural_->submit(std::move(detector_input), sequence, frame_timestamp_us, roi);

            AsyncArmorDetector::Result nn_result;
            if (!neural_->takeLatest(nn_result)) return result;   // 本帧没新结果（异步正常）

            result.timestamp_us = nn_result.timestamp_us;
            result.fresh = true;
            result.armors = std::move(nn_result.armors);
            for (auto& armor : result.armors) {
                armor.solve_result = solveArmorPnP(
                    armor, camera_matrix_, dist_coeffs_, config_.pnp.small_width,
                    config_.pnp.large_width, config_.pnp.height);
                if (armor.solve_result) ++result.pnp_count;
            }
            return result;
        }
#endif
        // ---- 传统灯条流程（同步）----
        cv::Mat gray;
        cv::Mat binary;
        cv::Mat dilated;
        detector_.gray_img(image, gray);
        detector_.binary_img(gray, binary, config_.classical.binary_threshold);
        detector_.open_close_img(binary, dilated);
        std::vector<std::vector<cv::Point>> contours;
        detector_.find_contours(dilated, contours);
        lights_.clear();
        detector_.find_lights(contours, lights_, image, config_.classical.enemy_color);
        result.lights = lights_;
        if (!lights_.empty()) detector_.find_armors(lights_, result.armors);
        for (auto& armor : result.armors) {
            armor.solve_result = solveArmorPnP(
                armor, camera_matrix_, dist_coeffs_, config_.pnp.small_width,
                config_.pnp.large_width, config_.pnp.height);
            if (armor.solve_result) ++result.pnp_count;
        }
        result.fresh = true;
        return result;
    }
} // namespace auto_aim
