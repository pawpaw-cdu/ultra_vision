#include "auto_buff/detection/rune_model.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iostream>
#include <stdexcept>

#include <openvino/openvino.hpp>

namespace auto_aim::energy
{
    struct RuneModel::Impl
    {
        RuneModelConfig config;
        ov::Core core;
        ov::CompiledModel compiled;
        ov::InferRequest request;
        bool input_is_uint8 = false;
        cv::Point2f canvas_center{0.0f, 0.0f};
        bool canvas_center_valid = false;
        bool last_roi_active = false;
        cv::Rect last_roi_crop{};
        double last_letterbox_scale = 1.0;
    };

    RuneModel::RuneModel(const RuneModelConfig& config) : impl_(std::make_unique<Impl>())
    {
        if (config.model_path.empty()) {
            throw std::runtime_error("RuneModel: model_path is empty");
        }
        impl_->config = config;

        const std::shared_ptr<ov::Model> model = impl_->core.read_model(config.model_path);
        // A static export keeps the anchor grid baked in, so its input size
        const ov::PartialShape declared_shape = model->input().get_partial_shape();
        if (declared_shape.is_dynamic() && config.input_size > 0) {
            model->reshape({{model->input().get_any_name(),
                             ov::Shape{1, 3, static_cast<std::size_t>(config.input_size),
                                       static_cast<std::size_t>(config.input_size)}}});
        } else if (declared_shape.rank().is_static() && declared_shape[2].is_static() &&
                   declared_shape[3].is_static()) {
            // 注意 rank[2] 是高、rank[3] 是宽：非方形输入（如 480x640）必须分别取，
            const int declared_height = static_cast<int>(declared_shape[2].get_length());
            const int declared_width = static_cast<int>(declared_shape[3].get_length());
            if (declared_width != impl_->config.input_size ||
                declared_height != (impl_->config.input_height > 0 ? impl_->config.input_height
                                                                  : impl_->config.input_size)) {
                std::cerr << "RuneModel: model input is " << declared_width << "x"
                          << declared_height << "; config says " << impl_->config.input_size << "x"
                          << impl_->config.input_height << "; using the model's own size"
                          << std::endl;
                impl_->config.input_size = declared_width;
                impl_->config.input_height = declared_height;
            }
        }
        impl_->input_is_uint8 =
            model->input().get_element_type() == ov::element::u8;

        ov::AnyMap properties;
        properties.emplace(ov::hint::performance_mode(
            config.performance_mode == "throughput"
                ? ov::hint::PerformanceMode::THROUGHPUT
                : ov::hint::PerformanceMode::LATENCY));
        if (config.num_threads > 0) {
            properties.emplace(ov::inference_num_threads(config.num_threads));
        }
        impl_->compiled = impl_->core.compile_model(model, impl_->config.device, properties);
        impl_->request = impl_->compiled.create_infer_request();
    }

    RuneModel::~RuneModel() = default;
    RuneModel::RuneModel(RuneModel&&) noexcept = default;
    RuneModel& RuneModel::operator=(RuneModel&&) noexcept = default;

    void RuneModel::setCanvasScale(double scale)
    {
        if (!(scale > 0.0)) return;
        impl_->config.input_pad_scale = scale;
    }

    void RuneModel::setCanvasCenter(const cv::Point2f& center, bool valid)
    {
        impl_->canvas_center = center;
        impl_->canvas_center_valid = valid;
    }

    bool RuneModel::lastRoiActive() const { return impl_->last_roi_active; }
    cv::Rect RuneModel::lastRoiCrop() const { return impl_->last_roi_crop; }
    double RuneModel::lastLetterboxScale() const { return impl_->last_letterbox_scale; }

    std::vector<RuneModel::Object> RuneModel::detect(const cv::Mat& image)
    {
        const RuneModelConfig& config = impl_->config;
        std::vector<Object> objects;
        if (image.empty()) return objects;
        impl_->last_roi_active = false;
        impl_->last_roi_crop = {};

        const int input_size = config.input_size;
        const int input_height = config.input_height > 0 ? config.input_height : input_size;

        // Optional canvas transform, see RuneModelConfig::input_pad_scale: >1 pads
        cv::Mat canvas;
        double canvas_offset_x = 0.0;
        double canvas_offset_y = 0.0;
        const double canvas_scale = config.input_pad_scale > 0.0 ? config.input_pad_scale : 1.0;
        if (canvas_scale > 1.001) {
            const int canvas_width = static_cast<int>(std::lround(image.cols * canvas_scale));
            const int canvas_height = static_cast<int>(std::lround(image.rows * canvas_scale));
            canvas = cv::Mat(canvas_height, canvas_width, CV_8UC3,
                             cv::Scalar(config.pad_value, config.pad_value, config.pad_value));
            canvas_offset_x = (canvas_width - image.cols) / 2.0;
            canvas_offset_y = (canvas_height - image.rows) / 2.0;
            image.copyTo(canvas(cv::Rect(static_cast<int>(canvas_offset_x),
                                         static_cast<int>(canvas_offset_y),
                                         image.cols, image.rows)));
        } else if (canvas_scale < 0.999) {
            const int crop_width =
                std::max(2, static_cast<int>(std::lround(image.cols * canvas_scale)));
            const int crop_height =
                std::max(2, static_cast<int>(std::lround(image.rows * canvas_scale)));
            const double center_x = impl_->canvas_center_valid
                                        ? impl_->canvas_center.x
                                        : (image.cols - 1) * 0.5;
            const double center_y = impl_->canvas_center_valid
                                        ? impl_->canvas_center.y
                                        : (image.rows - 1) * 0.5;
            const int crop_x = std::clamp(
                static_cast<int>(std::lround(center_x - crop_width * 0.5)),
                0, std::max(0, image.cols - crop_width));
            const int crop_y = std::clamp(
                static_cast<int>(std::lround(center_y - crop_height * 0.5)),
                0, std::max(0, image.rows - crop_height));
            canvas = image(cv::Rect(crop_x, crop_y, crop_width, crop_height)).clone();
            canvas_offset_x = -static_cast<double>(crop_x);
            canvas_offset_y = -static_cast<double>(crop_y);
            impl_->last_roi_active = impl_->canvas_center_valid;
            impl_->last_roi_crop = cv::Rect(crop_x, crop_y, crop_width, crop_height);
        } else {
            canvas = image;
        }

        const double scale = std::min(static_cast<double>(input_size) / canvas.cols,
                                      static_cast<double>(input_height) / canvas.rows);
        impl_->last_letterbox_scale = scale;
        const int scaled_width = std::max(1, static_cast<int>(std::lround(canvas.cols * scale)));
        const int scaled_height = std::max(1, static_cast<int>(std::lround(canvas.rows * scale)));
        // fit_to_window_letterbox: centre the padded frame, which is what the
        const int pad_left = (input_size - scaled_width) / 2;
        const int pad_top = (input_height - scaled_height) / 2;

        // Letterbox: keep the aspect ratio, pad with the export's pad value, and
        cv::Mat padded_input(input_height, input_size, CV_8UC3,
                             cv::Scalar(config.pad_value, config.pad_value, config.pad_value));
        cv::Mat letterbox = padded_input(cv::Rect(pad_left, pad_top, scaled_width, scaled_height));
        cv::resize(canvas, letterbox, letterbox.size(), 0.0, 0.0, cv::INTER_LINEAR);

        cv::Mat blob;
        if (impl_->input_is_uint8) {
            blob = cv::dnn::blobFromImage(padded_input, 1.0, cv::Size(), cv::Scalar(),
                                          config.reverse_input_channels, false, CV_8U);
        } else {
            blob = cv::dnn::blobFromImage(padded_input, config.input_scale, cv::Size(),
                                          cv::Scalar(), config.reverse_input_channels, false,
                                          CV_32F);
        }
        ov::Tensor input_tensor(impl_->compiled.input().get_element_type(),
                                {1, 3, static_cast<std::size_t>(input_height),
                                 static_cast<std::size_t>(input_size)},
                                blob.data);
        impl_->request.set_input_tensor(input_tensor);

        const auto start = std::chrono::steady_clock::now();
        impl_->request.infer();
        const auto finish = std::chrono::steady_clock::now();
        latency_ms_ =
            std::chrono::duration<double, std::milli>(finish - start).count();

        const ov::Tensor output = impl_->request.get_output_tensor();
        const ov::Shape shape = output.get_shape();
        if (shape.size() != 3) {
            std::cerr << "RuneModel: unexpected output rank" << std::endl;
            return objects;
        }

        // Accept either [1, attributes, anchors] or the transposed export.
        const RuneModelConfig& cfg = impl_->config;
        const int keypoint_dim = 3;   // 深大五点模型：x, y, confidence
        const int attributes = cfg.num_classes + cfg.num_keypoints * keypoint_dim;
        const bool attributes_first = shape[1] == static_cast<std::size_t>(attributes);
        const std::size_t anchors = attributes_first ? shape[2] : shape[1];
        const std::size_t stride = attributes_first ? shape[2] : shape[1];
        if (stride < static_cast<std::size_t>(attributes)) {
            std::cerr << "RuneModel: unexpected output shape" << std::endl;
            return objects;
        }
        const float* data = output.data<const float>();
        const std::size_t total = anchors * stride;
        // 越界保护：轴序判断一旦反了，data[...] 会静默踩内存并把进程挂死
        const auto at = [&](std::size_t anchor, int attribute) {
            const std::size_t index =
                attributes_first ? anchor + anchors * static_cast<std::size_t>(attribute)
                                 : anchor * stride + static_cast<std::size_t>(attribute);
            return index < total ? data[index] : 0.0f;
        };
        if (std::getenv("ULTRA_VISION_RUNE_DEBUG")) {
            std::cerr << "[rune output] shape=[" << shape[0] << "," << shape[1] << "," << shape[2]
                      << "] attributes=" << attributes << " anchors=" << anchors
                      << " stride=" << stride
                      << " layout=" << (attributes_first ? "[N,C,A]" : "[N,A,C]")
                      << " letterbox=scale " << scale << " pad(" << pad_left << "," << pad_top
                      << ") canvas_offset(" << canvas_offset_x << "," << canvas_offset_y << ")"
                      << std::endl;
        }

        // 网络输入坐标 -> 原图坐标。
        const auto to_image_x = [&](double value) {
            return (value - pad_left) / scale - canvas_offset_x;
        };
        const auto to_image_y = [&](double value) {
            return (value - pad_top) / scale - canvas_offset_y;
        };

        std::vector<Object> candidates;
        std::vector<float> qualities;
        float anchor_best_score = 0.0f;
        last_best_score_ = 0.0f;
        for (std::size_t anchor = 0; anchor < anchors; ++anchor) {
            float score = 0.0f;
            int class_id = 0;
            // N 类分数：逐 anchor 取最大类（与参考实现一致）。
            for (int c = 0; c < cfg.num_classes; ++c) {
                float value = at(anchor, c);
                if (cfg.class_scores_are_logits) {
                    value = 1.0f / (1.0f + std::exp(-value));
                }
                if (c == 0 || value > score) {
                    score = value;
                    class_id = c;
                }
            }
            anchor_best_score = std::max(anchor_best_score, score);
            if (score < config.confidence_threshold) continue;

            Object object;
            object.prob = score;
            object.class_id = class_id;

            // 关键点：逐点置信度过滤（作者的判据：conf >= 阈值、有效点 >= 3、
            const int keypoint_base = cfg.num_classes;
            bool negative = false;
            int valid_keypoints = 0;
            float keypoint_confidence_sum = 0.0f;
            for (int k = 0; k < cfg.num_keypoints; ++k) {
                const int row = keypoint_base + k * keypoint_dim;
                const float raw_x = at(anchor, row);
                const float raw_y = at(anchor, row + 1);
                if (raw_x < 0.0f || raw_y < 0.0f) {
                    negative = true;
                    break;
                }
                const float x = static_cast<float>(std::clamp(
                    to_image_x(raw_x), 0.0, static_cast<double>(image.cols - 1)));
                const float y = static_cast<float>(std::clamp(
                    to_image_y(raw_y), 0.0, static_cast<double>(image.rows - 1)));
                const float keypoint_confidence = at(anchor, row + 2);
                object.keypoints.emplace_back(x, y);
                object.keypoint_confidence.push_back(keypoint_confidence);
                if (keypoint_confidence >= cfg.keypoint_confidence_threshold) {
                    ++valid_keypoints;
                    keypoint_confidence_sum += keypoint_confidence;
                }
            }
            if (negative || valid_keypoints < cfg.min_valid_keypoints) continue;

            // 靶面中心与框：使用 0/1/3/4；k2 是 R 标/旋转中心，不是靶心。
            std::vector<cv::Point2f> plate_points;
            if (object.keypoints.size() >= 5) {
                plate_points = {object.keypoints[0], object.keypoints[1],
                                object.keypoints[3], object.keypoints[4]};
                for (const cv::Point2f& point : plate_points) object.center += point;
                object.center /= static_cast<float>(plate_points.size());
                float plate_radius = 0.0f;
                for (const cv::Point2f& point : plate_points) {
                    plate_radius += static_cast<float>(cv::norm(point - object.center));
                }
                plate_radius /= static_cast<float>(plate_points.size());
                if (plate_radius < config.min_plate_radius_px) continue;
            } else {
                continue;
            }
            object.rect = cv::boundingRect(plate_points);
            const float mean_keypoint_confidence =
                valid_keypoints > 0 ? keypoint_confidence_sum / valid_keypoints : 0.0f;
            candidates.push_back(std::move(object));
            qualities.push_back(score * mean_keypoint_confidence);
        }

        if (std::getenv("ULTRA_VISION_RUNE_DEBUG")) {
            static int printed = 0;
            // 默认每 30 帧打一行（免得刷屏）；诊断单帧/短视频时给
            const bool every_frame = std::getenv("ULTRA_VISION_RUNE_DEBUG_ALL") != nullptr;
            ++printed;
            if (every_frame || printed % 30 == 0) {
                std::cerr << "[rune decode] best class score = " << anchor_best_score
                          << " (threshold " << config.confidence_threshold << "), candidates = "
                          << candidates.size() << std::endl;
            }
        }
        last_best_score_ = anchor_best_score;
        if (candidates.empty()) return objects;

        std::vector<std::size_t> order(candidates.size());
        for (std::size_t i = 0; i < order.size(); ++i) order[i] = i;
        std::sort(order.begin(), order.end(), [&](std::size_t lhs, std::size_t rhs) {
            return qualities[lhs] > qualities[rhs];
        });

        // 参考实现按中心点距离抑制：同一个扇叶被相邻 anchor 重复检出很常见，
        const float distance_threshold = std::max(1.0f, config.nms_center_dist_px);
        for (const std::size_t index : order) {
            bool suppressed = false;
            for (const Object& kept : objects) {
                if (cv::norm(candidates[index].center - kept.center) < distance_threshold) {
                    suppressed = true;
                    break;
                }
            }
            if (!suppressed) objects.push_back(candidates[index]);
        }
        return objects;
    }

    std::vector<RuneModel::Object> RuneModel::detectBest(const cv::Mat& image)
    {
        std::vector<Object> objects = detect(image);
        if (objects.size() <= 1) return objects;

        const auto best = std::max_element(
            objects.begin(), objects.end(),
            [](const Object& lhs, const Object& rhs) { return lhs.prob < rhs.prob; });
        std::vector<Object> best_only;
        best_only.push_back(std::move(*best));
        return best_only;
    }
} // namespace auto_aim::energy
