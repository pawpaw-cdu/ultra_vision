// 由 RP-26Rune-main/src/app_plugin/detector/src/NNDetector.cpp 复制而来，只做三处改动：
//   1. 去掉插件框架依赖（class_loader / Detector / REGISTER_PLUGIN / app::Context），
//      把推理封装成 harness 可以同步调用的 Rp26RuneDetector；
//   2. 类别映射不再依赖他们的 AimMode，改用 harness 的 Rp26RuneMode；
//   3. 去掉 ImgViz 可视化（harness 无 GUI 线程）。
// 预处理、letterbox、后处理、NMS、关键点解码逻辑**逐行保留**，保证与深大实现一致。
#include "rp26_detector.hpp"

#include "json.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <glog/logging.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <openvino/openvino.hpp>
#include <openvino/core/preprocess/pre_post_process.hpp>
#include <openvino/runtime/properties.hpp>

namespace
{
constexpr int kKeypointCount = 5;
constexpr int kKeypointDimension = 3;
constexpr float kDefaultConfidenceThreshold = 0.8F;
constexpr float kDefaultKeypointConfidenceThreshold = 0.8F;
constexpr float kDefaultNmsDistance = 30.0F;
constexpr int kDefaultMinimumValidKeypoints = kKeypointCount;

struct DetectorConfig
{
    std::filesystem::path config_path;
    std::filesystem::path model_path;
    std::string device = "CPU";
    float confidence_threshold = kDefaultConfidenceThreshold;
    float keypoint_confidence_threshold = kDefaultKeypointConfidenceThreshold;
    float nms_distance = kDefaultNmsDistance;
    int minimum_valid_keypoints = kDefaultMinimumValidKeypoints;
};

struct LetterboxResult
{
    cv::Mat image;
    float scale = 1.0F;
    int padding_x = 0;
    int padding_y = 0;
};

struct Detection
{
    int class_id = -1;
    float confidence = 0.0F;
    float quality = 0.0F;
    cv::Point2f center{0.0F, 0.0F};
    std::array<cv::Point2f, kKeypointCount> keypoints{};
    std::array<float, kKeypointCount> keypoint_confidences{};
};

template <typename T>
void read_optional(const cv::FileNode &node, const char *key, T &value)
{
    const cv::FileNode field = node[key];
    if (!field.empty())
        field >> value;
}

DetectorConfig load_detector_config()
{
    DetectorConfig result;
    result.config_path = DETECTOR_CONFIG_DIR / "detect.json";

    J_DETECT.updateJson();
    if (!J_DETECT.config_.isOpened())
        throw std::runtime_error("failed to open detector config: " + result.config_path.string());

    const cv::FileNode node = J_DETECT.config_["rune_detect"];
    if (node.empty() || !node.isMap())
        throw std::runtime_error("detect.json rune_detect configuration must be an object");
    std::string model;
    read_optional(node, "model", model);
    if (node["model"].empty())
        read_optional(node, "model_path", model);
    read_optional(node, "device", result.device);
    if (node["device"].empty())
        read_optional(node, "inference_device", result.device);
    read_optional(node, "conf", result.confidence_threshold);
    if (node["conf"].empty())
        read_optional(node, "confidence_threshold", result.confidence_threshold);
    read_optional(node, "kconf", result.keypoint_confidence_threshold);
    if (node["kconf"].empty())
        read_optional(node, "keypoint_confidence_threshold", result.keypoint_confidence_threshold);
    read_optional(node, "nms", result.nms_distance);
    if (node["nms"].empty())
        read_optional(node, "nms_distance", result.nms_distance);
    if (node["nms"].empty() && node["nms_distance"].empty())
        read_optional(node, "nms_distance_threshold", result.nms_distance);
    read_optional(node, "min_valid_kpts", result.minimum_valid_keypoints);
    if (node["min_valid_kpts"].empty())
        read_optional(node, "min_valid_keypoints", result.minimum_valid_keypoints);
    if (node["min_valid_kpts"].empty() && node["min_valid_keypoints"].empty())
        read_optional(node, "minimum_valid_keypoints", result.minimum_valid_keypoints);

    if (model.empty())
        throw std::runtime_error("detector config is missing the required string field 'model': " +
                                 result.config_path.string());
    if (result.device.empty())
        throw std::runtime_error("detector config field 'device' cannot be empty");
    if (!std::isfinite(result.confidence_threshold) || result.confidence_threshold < 0.0F ||
        result.confidence_threshold > 1.0F)
        throw std::runtime_error("detector config field 'conf' must be in [0, 1]");
    if (!std::isfinite(result.keypoint_confidence_threshold) ||
        result.keypoint_confidence_threshold < 0.0F || result.keypoint_confidence_threshold > 1.0F)
        throw std::runtime_error("detector config field 'kconf' must be in [0, 1]");
    if (!std::isfinite(result.nms_distance) || result.nms_distance <= 0.0F)
        throw std::runtime_error("detector config field 'nms' must be greater than zero");
    if (result.minimum_valid_keypoints < 1 || result.minimum_valid_keypoints > kKeypointCount)
        throw std::runtime_error("detector config field 'min_valid_kpts' must be in [1, 5]");

    // 模型路径解析（比他们多一条兜底，其余一致）：
    //   1. RP26_MODEL_PATH 环境变量（绝对或相对当前目录）
    //   2. 配置里写相对路径时，相对配置目录（他们的原行为）
    //   3. 兜底：harness 源码目录下的 ../../models/openvino/...（本仓库固定位置）
    // 之所以要兜底，是因为工程内有"源码 config/"和"构建目录 config/"两份副本。
    {
        std::vector<std::filesystem::path> candidates;
        if (const char *env = std::getenv("RP26_MODEL_PATH"))
            candidates.emplace_back(env);

        const std::filesystem::path configured(model);
        candidates.push_back(configured.is_absolute() ? configured
                                                      : DETECTOR_CONFIG_DIR / configured);
        candidates.push_back(std::filesystem::path(RP26_HARNESS_SOURCE_DIR) / ".." / ".." /
                             "models" / "openvino" / "RuneDetectionModel-master" / "model" /
                             configured.filename());

        result.model_path.clear();
        for (std::filesystem::path &candidate : candidates)
        {
            candidate = candidate.lexically_normal();
            if (std::filesystem::is_regular_file(candidate))
            {
                result.model_path = candidate;
                break;
            }
        }

        if (result.model_path.empty())
            throw std::runtime_error("detector model not found; tried config '" + model +
                                     "' (set RP26_MODEL_PATH to override)");
    }

    return result;
}

int map_class_for_mode(int model_class, rp26_sim::Rp26RuneMode mode)
{
    // RuneDetectionModel classes: 0 inactive, 1 small activated, 2 big activated.
    if (model_class == 0)
        return 0;
    if (mode == rp26_sim::Rp26RuneMode::Small && model_class == 1)
        return 1;
    if (mode == rp26_sim::Rp26RuneMode::Large && model_class == 2)
        return 1;
    return -1;
}

cv::Point to_integer_point(const cv::Point2f &point)
{
    return {cvRound(point.x), cvRound(point.y)};
}

std::string class_name(int class_id)
{
    switch (class_id)
    {
    case 0:
        return "inactive";
    case 1:
        return "small activated";
    case 2:
        return "big activated";
    default:
        return "class " + std::to_string(class_id);
    }
}
} // namespace

class rp26_sim::Rp26RuneDetector::Impl
{
public:
    void initialize()
    {
        m_config = load_detector_config();

        ov::Core core;
        std::shared_ptr<ov::Model> model = core.read_model(m_config.model_path.string());
        if (model->inputs().size() != 1 || model->outputs().size() != 1)
            throw std::runtime_error("rune model must have exactly one input and one output");

        const ov::Shape input_shape = model->input().get_shape();
        if (input_shape.size() != 4 || input_shape[0] != 1 || input_shape[1] != 3)
            throw std::runtime_error("rune model input must have static NCHW shape [1,3,H,W]");
        if (input_shape[2] == 0 || input_shape[3] == 0)
            throw std::runtime_error("rune model input height and width must be non-zero");

        m_input_height = static_cast<int>(input_shape[2]);
        m_input_width = static_cast<int>(input_shape[3]);

        // Identical to RuneDetectionModel-master: external letterbox followed by
        // BGR u8 NHWC -> RGB f32 [0,1] NCHW in OpenVINO's PrePostProcessor.
        ov::preprocess::PrePostProcessor preprocessor(model);
        preprocessor.input().tensor()
            .set_element_type(ov::element::u8)
            .set_layout("NHWC")
            .set_color_format(ov::preprocess::ColorFormat::BGR);
        preprocessor.input().preprocess()
            .convert_element_type(ov::element::f32)
            .convert_color(ov::preprocess::ColorFormat::RGB)
            .scale(255.0F);
        preprocessor.input().model().set_layout("NCHW");
        model = preprocessor.build();

        m_compiled_model = core.compile_model(
            model,
            m_config.device,
            ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY));
        m_input_tensor = ov::Tensor(
            ov::element::u8,
            {1, static_cast<std::size_t>(m_input_height), static_cast<std::size_t>(m_input_width), 3});
        m_infer_request = m_compiled_model.create_infer_request();
        m_infer_request.set_input_tensor(m_input_tensor);

        const ov::Output<const ov::Node> output_port = m_compiled_model.output();
        const ov::Shape output_shape = output_port.get_shape();
        if (output_shape.size() != 3 || output_shape[0] != 1)
            throw std::runtime_error("rune model output must be [1,C,A] or [1,A,C]");
        if (output_port.get_element_type() != ov::element::f32)
            throw std::runtime_error("rune model output element type must be f32");

        // The layout test deliberately follows RuneDetectionModel-master.
        if (output_shape[1] <= 256 && output_shape[2] >= 100)
        {
            m_output_is_nca = true;
            m_output_channels = static_cast<int>(output_shape[1]);
            m_anchor_count = static_cast<int>(output_shape[2]);
        }
        else
        {
            m_output_is_nca = false;
            m_anchor_count = static_cast<int>(output_shape[1]);
            m_output_channels = static_cast<int>(output_shape[2]);
        }

        // The published contract is the repository's five-point rune format.
        // RuneDetectionModel-master exports 3 + 5*3 = 18 channels.
        if (m_output_channels != 3 + kKeypointCount * kKeypointDimension)
        {
            std::ostringstream message;
            message << "unsupported rune head: expected 18 channels (3 classes + 5x3 keypoints), got "
                    << m_output_channels;
            throw std::runtime_error(message.str());
        }
        m_class_count = 3;

        LOG(INFO) << "[NNDetector] initialized OpenVINO rune model; model=" << m_config.model_path
                  << ", device=" << m_config.device
                  << ", input=" << m_input_width << 'x' << m_input_height
                  << ", output_layout=" << (m_output_is_nca ? "[N,C,A]" : "[N,A,C]")
                  << ", anchors=" << m_anchor_count
                  << ", conf=" << m_config.confidence_threshold
                  << ", kconf=" << m_config.keypoint_confidence_threshold
                  << ", nms=" << m_config.nms_distance
                  << ", min_valid_kpts=" << m_config.minimum_valid_keypoints;
    }

    std::vector<Detection> infer(const cv::Mat &source)
    {
        const cv::Mat bgr = normalize_input(source);
        LetterboxResult letterbox = preprocess_letterbox(bgr);
        if (!letterbox.image.isContinuous())
            letterbox.image = letterbox.image.clone();

        const std::size_t image_bytes = letterbox.image.total() * letterbox.image.elemSize();
        if (image_bytes != m_input_tensor.get_byte_size())
        {
            std::ostringstream message;
            message << "letterbox byte size does not match OpenVINO input tensor: image="
                    << image_bytes << ", tensor=" << m_input_tensor.get_byte_size();
            throw std::runtime_error(message.str());
        }
        std::memcpy(m_input_tensor.data<std::uint8_t>(), letterbox.image.data, image_bytes);
        m_infer_request.infer();

        return postprocess(letterbox, source.cols, source.rows);
    }

    std::vector<Detection> filter_for_mode(const std::vector<Detection> &detections,
                                           rp26_sim::Rp26RuneMode mode) const
    {
        std::vector<Detection> result;
        result.reserve(detections.size());

        for (const Detection &detection : detections)
        {
            const int published_class = map_class_for_mode(detection.class_id, mode);
            if (published_class < 0)
                continue;

            Detection filtered = detection;
            filtered.class_id = published_class;
            result.emplace_back(std::move(filtered));
        }
        return result;
    }

private:
    cv::Mat normalize_input(const cv::Mat &source) const
    {
        if (source.empty())
            throw std::invalid_argument("cannot infer an empty image");

        if (source.type() == CV_8UC3)
            return source;

        cv::Mat bgr;
        if (source.type() == CV_8UC1)
            cv::cvtColor(source, bgr, cv::COLOR_GRAY2BGR);
        else if (source.type() == CV_8UC4)
            cv::cvtColor(source, bgr, cv::COLOR_BGRA2BGR);
        else
            throw std::invalid_argument("rune detector expects an 8-bit 1-, 3-, or 4-channel image");
        return bgr;
    }

    LetterboxResult preprocess_letterbox(const cv::Mat &source) const
    {
        LetterboxResult result;
        result.scale = std::min(
            static_cast<float>(m_input_width) / static_cast<float>(source.cols),
            static_cast<float>(m_input_height) / static_cast<float>(source.rows));

        const int resized_width = static_cast<int>(std::round(source.cols * result.scale));
        const int resized_height = static_cast<int>(std::round(source.rows * result.scale));
        result.padding_x = (m_input_width - resized_width) / 2;
        result.padding_y = (m_input_height - resized_height) / 2;
        const int right_padding = m_input_width - resized_width - result.padding_x;
        const int bottom_padding = m_input_height - resized_height - result.padding_y;

        cv::Mat resized;
        if (resized_width != source.cols || resized_height != source.rows)
            cv::resize(source, resized, cv::Size(resized_width, resized_height));
        else
            resized = source;

        if (result.padding_x > 0 || result.padding_y > 0 || right_padding > 0 || bottom_padding > 0)
        {
            cv::copyMakeBorder(
                resized,
                result.image,
                result.padding_y,
                bottom_padding,
                result.padding_x,
                right_padding,
                cv::BORDER_CONSTANT,
                cv::Scalar(114, 114, 114));
        }
        else
        {
            result.image = resized;
        }
        return result;
    }

    std::vector<Detection> postprocess(
        const LetterboxResult &letterbox,
        int original_width,
        int original_height)
    {
        const ov::Tensor output_tensor = m_infer_request.get_output_tensor();
        const float *const output = output_tensor.data<float>();

        const auto value_at = [this, output](int channel, int anchor)
        {
            return m_output_is_nca
                       ? output[channel * m_anchor_count + anchor]
                       : output[anchor * m_output_channels + channel];
        };

        std::vector<Detection> detections;
        detections.reserve(64);
        for (int anchor = 0; anchor < m_anchor_count; ++anchor)
        {
            int best_class = -1;
            float best_confidence = 0.0F;
            for (int class_index = 0; class_index < m_class_count; ++class_index)
            {
                const float score = value_at(class_index, anchor);
                if (std::isfinite(score) && score > best_confidence)
                {
                    best_confidence = score;
                    best_class = class_index;
                }
            }
            if (best_class < 0 || best_confidence < m_config.confidence_threshold)
                continue;

            Detection detection;
            detection.class_id = best_class;
            detection.confidence = best_confidence;

            int valid_keypoints = 0;
            float keypoint_confidence_sum = 0.0F;
            cv::Point2f valid_center{0.0F, 0.0F};
            bool invalid_keypoint = false;

            for (int keypoint = 0; keypoint < kKeypointCount; ++keypoint)
            {
                const int base = m_class_count + keypoint * kKeypointDimension;
                float x = (value_at(base, anchor) - static_cast<float>(letterbox.padding_x)) /
                          letterbox.scale;
                float y = (value_at(base + 1, anchor) - static_cast<float>(letterbox.padding_y)) /
                          letterbox.scale;
                const float keypoint_confidence = value_at(base + 2, anchor);

                if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(keypoint_confidence) ||
                    x < 0.0F || y < 0.0F)
                {
                    invalid_keypoint = true;
                    break;
                }

                x = std::clamp(x, 0.0F, static_cast<float>(original_width - 1));
                y = std::clamp(y, 0.0F, static_cast<float>(original_height - 1));
                detection.keypoints[keypoint] = {x, y};
                detection.keypoint_confidences[keypoint] = keypoint_confidence;

                if (keypoint_confidence >= m_config.keypoint_confidence_threshold)
                {
                    ++valid_keypoints;
                    keypoint_confidence_sum += keypoint_confidence;
                    valid_center += detection.keypoints[keypoint];
                }
            }

            if (invalid_keypoint || valid_keypoints < m_config.minimum_valid_keypoints)
                continue;

            const float mean_keypoint_confidence =
                keypoint_confidence_sum / static_cast<float>(valid_keypoints);
            detection.quality = detection.confidence * mean_keypoint_confidence;
            detection.center = valid_center * (1.0F / static_cast<float>(valid_keypoints));
            detections.emplace_back(std::move(detection));
        }

        return center_distance_nms(std::move(detections));
    }

    std::vector<Detection> center_distance_nms(std::vector<Detection> detections) const
    {
        std::sort(
            detections.begin(),
            detections.end(),
            [](const Detection &left, const Detection &right)
            {
                return left.quality > right.quality;
            });

        std::vector<Detection> kept;
        kept.reserve(detections.size());
        std::vector<bool> suppressed(detections.size(), false);
        const float threshold_squared = m_config.nms_distance * m_config.nms_distance;

        for (std::size_t index = 0; index < detections.size(); ++index)
        {
            if (suppressed[index])
                continue;

            kept.emplace_back(detections[index]);
            const cv::Point2f center = detections[index].center;
            for (std::size_t candidate = index + 1; candidate < detections.size(); ++candidate)
            {
                if (suppressed[candidate])
                    continue;
                const cv::Point2f delta = center - detections[candidate].center;
                if (delta.dot(delta) < threshold_squared)
                    suppressed[candidate] = true;
            }
        }
        return kept;
    }

    DetectorConfig m_config;
    ov::CompiledModel m_compiled_model;
    ov::Tensor m_input_tensor;
    ov::InferRequest m_infer_request;
    int m_input_width = 0;
    int m_input_height = 0;
    int m_output_channels = 0;
    int m_anchor_count = 0;
    int m_class_count = 0;
    bool m_output_is_nca = true;
};

namespace rp26_sim
{

Rp26RuneDetector::Rp26RuneDetector() : m_impl(std::make_unique<Impl>())
{
    try
    {
        m_impl->initialize();
    }
    catch (const std::exception &error)
    {
        LOG(ERROR) << "[Rp26RuneDetector] initialization failed: " << error.what();
        throw;
    }
}

Rp26RuneDetector::~Rp26RuneDetector() = default;

std::vector<Rp26Detection> Rp26RuneDetector::infer(const cv::Mat &image)
{
    std::vector<Rp26Detection> result;
    for (const Detection &detection : m_impl->infer(image))
    {
        Rp26Detection output;
        output.model_class_id = detection.class_id;
        output.confidence = detection.confidence;
        output.quality = detection.quality;
        output.center = detection.center;
        output.keypoints = detection.keypoints;
        output.keypoint_confidences = detection.keypoint_confidences;
        result.emplace_back(std::move(output));
    }
    return result;
}

std::vector<power_rune::RuneInput::NNRuneInfo> Rp26RuneDetector::toRuneInfos(
    const std::vector<Rp26Detection> &detections,
    Rp26RuneMode mode)
{
    std::vector<power_rune::RuneInput::NNRuneInfo> result;
    result.reserve(detections.size());

    for (const Rp26Detection &detection : detections)
    {
        // 与深大一致：按模式把 model class 0（未击打）与对应模式的"已击打"类保留为
        // 0/1，另一模式的激活类被丢弃（convert_for_mode 的语义）。
        int published_class = 0;
        if (detection.model_class_id == 1)
        {
            if (mode != Rp26RuneMode::Small)
                continue;
            published_class = 1;
        }
        else if (detection.model_class_id == 2)
        {
            if (mode != Rp26RuneMode::Large)
                continue;
            published_class = 1;
        }
        else if (detection.model_class_id != 0)
        {
            continue;
        }

        power_rune::RuneInput::NNRuneInfo rune;
        rune.top = to_integer_point(detection.keypoints[0]);
        rune.left = to_integer_point(detection.keypoints[1]);
        rune.point_R = to_integer_point(detection.keypoints[2]);
        rune.right = to_integer_point(detection.keypoints[3]);
        rune.bottom = to_integer_point(detection.keypoints[4]);
        rune.class_id = published_class;
        result.emplace_back(std::move(rune));
    }
    return result;
}

} // namespace rp26_sim
