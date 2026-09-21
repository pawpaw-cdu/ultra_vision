#include "nn_detector.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <openvino/openvino.hpp>

namespace auto_aim
{
    namespace
    {
        // The 38-class head of the SZU model, in output order. Mirrors
        // sp_vision_25's armor_properties table.
        struct ClassSpec {
            NnColor color;
            NnName name;
            bool large;
        };

        const std::array<ClassSpec, 38>& classTable()
        {
            static const std::array<ClassSpec, 38> table{{
                {NnColor::blue, NnName::sentry, false},
                {NnColor::red, NnName::sentry, false},
                {NnColor::extinguish, NnName::sentry, false},
                {NnColor::blue, NnName::one, false},
                {NnColor::red, NnName::one, false},
                {NnColor::extinguish, NnName::one, false},
                {NnColor::blue, NnName::two, false},
                {NnColor::red, NnName::two, false},
                {NnColor::extinguish, NnName::two, false},
                {NnColor::blue, NnName::three, false},
                {NnColor::red, NnName::three, false},
                {NnColor::extinguish, NnName::three, false},
                {NnColor::blue, NnName::four, false},
                {NnColor::red, NnName::four, false},
                {NnColor::extinguish, NnName::four, false},
                {NnColor::blue, NnName::five, false},
                {NnColor::red, NnName::five, false},
                {NnColor::extinguish, NnName::five, false},
                {NnColor::blue, NnName::outpost, false},
                {NnColor::red, NnName::outpost, false},
                {NnColor::extinguish, NnName::outpost, false},
                {NnColor::blue, NnName::base, true},
                {NnColor::red, NnName::base, true},
                {NnColor::extinguish, NnName::base, true},
                {NnColor::purple, NnName::base, true},
                {NnColor::blue, NnName::base, false},
                {NnColor::red, NnName::base, false},
                {NnColor::extinguish, NnName::base, false},
                {NnColor::purple, NnName::base, false},
                {NnColor::blue, NnName::three, true},
                {NnColor::red, NnName::three, true},
                {NnColor::extinguish, NnName::three, true},
                {NnColor::blue, NnName::four, true},
                {NnColor::red, NnName::four, true},
                {NnColor::extinguish, NnName::four, true},
                {NnColor::blue, NnName::five, true},
                {NnColor::red, NnName::five, true},
                {NnColor::extinguish, NnName::five, true},
            }};
            return table;
        }

        const char* colorName(NnColor color)
        {
            switch (color) {
            case NnColor::blue: return "blue";
            case NnColor::red: return "red";
            case NnColor::extinguish: return "extinguish";
            case NnColor::purple: return "purple";
            }
            return "unknown";
        }

        const char* nameName(NnName name)
        {
            switch (name) {
            case NnName::one: return "one";
            case NnName::two: return "two";
            case NnName::three: return "three";
            case NnName::four: return "four";
            case NnName::five: return "five";
            case NnName::sentry: return "sentry";
            case NnName::outpost: return "outpost";
            case NnName::base: return "base";
            case NnName::not_armor: return "not_armor";
            }
            return "unknown";
        }

        double sigmoid(double value)
        {
            return 1.0 / (1.0 + std::exp(-value));
        }

        // RP24 vocabulary. The published description lists the colours as
        // (red, blue, gray, purple), but the reference implementation decodes
        // channel 0 as blue and 1 as red, so follow the code: 0=blue, 1=red,
        // 2=extinguish, 3=purple.
        NnColor rp24Color(int color_id)
        {
            switch (color_id) {
            case 0: return NnColor::blue;
            case 1: return NnColor::red;
            case 2: return NnColor::extinguish;
            default: return NnColor::purple;
            }
        }

        NnName rp24Name(int num_id)
        {
            switch (num_id) {
            case 0: return NnName::sentry;
            case 1: return NnName::one;
            case 2: return NnName::two;
            case 3: return NnName::three;
            case 4: return NnName::four;
            case 5: return NnName::five;
            case 6: return NnName::outpost;
            default: return NnName::base;   // 7 = Bs, 8 = Bb
            }
        }

        // Order the four corners as top-left, top-right, bottom-right,
        // bottom-left, matching both the model's training convention and the
        // point order used by solveArmorPnP().
        void sortKeypoints(std::array<cv::Point2f, 4>& keypoints)
        {
            std::sort(keypoints.begin(), keypoints.end(),
                [](const cv::Point2f& lhs, const cv::Point2f& rhs) {
                    return lhs.y < rhs.y;
                });
            if (keypoints[0].x > keypoints[1].x) {
                std::swap(keypoints[0], keypoints[1]);   // top-left, top-right
            }
            if (keypoints[2].x > keypoints[3].x) {
                std::swap(keypoints[2], keypoints[3]);   // bottom-left, bottom-right
            }
            std::swap(keypoints[2], keypoints[3]);       // -> top-right, bottom-right
        }
    } // namespace

        struct NnArmorDetector::Impl {
        NnDetectorConfig config;
        ov::Core core;
        ov::CompiledModel compiled;
        NnDetectorTiming timing;
    };

    NnArmorDetector::NnArmorDetector(const NnDetectorConfig& config)
        : impl_(std::make_unique<Impl>())
    {
        impl_->config = config;
        if (config.model_path.empty()) {
            throw std::runtime_error("NnArmorDetector: model_path is empty");
        }

        ov::AnyMap properties;
        properties.emplace(ov::hint::performance_mode(
            config.performance_mode == "throughput"
                ? ov::hint::PerformanceMode::THROUGHPUT
                : ov::hint::PerformanceMode::LATENCY));
        if (config.num_threads > 0) {
            properties.emplace(ov::inference_num_threads(config.num_threads));
        }
        if (config.num_streams > 0) {
            properties.emplace(ov::num_streams(config.num_streams));
        }
        const std::shared_ptr<ov::Model> model =
            impl_->core.read_model(config.model_path);
        // Note: the input size cannot be changed on this export. The anchor grid
        // is baked in as a constant (8400 anchors for 640x640), so a reshape is
        // rejected by shape inference ("Constant f32[1,2,8400]" against a
        // resized branch). Running the network on a smaller input needs a
        // re-export from the training pipeline, and only that would make the
        // dynamic ROI in node_sim pay off: cropping the ROI alone does not
        // reduce cost, because the crop is letterboxed back up to 640x640.
        (void)config.input_size;
        impl_->compiled = impl_->core.compile_model(model, config.device, properties);
    }

    NnArmorDetector::~NnArmorDetector() = default;
    NnArmorDetector::NnArmorDetector(NnArmorDetector&&) noexcept = default;
    NnArmorDetector& NnArmorDetector::operator=(NnArmorDetector&&) noexcept = default;

    std::vector<NnDetection> NnArmorDetector::postprocess(const ov::Tensor& output,
                                                          const cv::Mat& image,
                                                          double scale,
                                                          const cv::Rect& roi)
    {
        std::vector<NnDetection> detections;
        const NnDetectorConfig& config = impl_->config;
        const ov::Shape shape = output.get_shape();
        if (shape.size() != 3) return detections;

        // Lightweight RP24 head: keypoints + score + colors + numbers, no box.
        if (config.output_format == "rp24") {
            constexpr size_t kAttributes = 8 + 1 + 4 + 9;   // 22
            const bool rp24_anchors_last = shape[1] == kAttributes;
            const size_t rp24_anchors = rp24_anchors_last ? shape[2] : shape[1];
            const size_t rp24_stride = rp24_anchors_last ? shape[1] : shape[2];
            if (rp24_stride < kAttributes) return detections;
            const float* rp24_data = output.data<float>();
            const auto rp24_at = [&](size_t anchor, size_t attribute) {
                return rp24_anchors_last
                    ? rp24_data[anchor + rp24_anchors * attribute]
                    : rp24_data[anchor * rp24_stride + attribute];
            };

            std::vector<cv::Rect> boxes;
            std::vector<float> confidences;
            std::vector<NnDetection> candidates;
            for (size_t anchor = 0; anchor < rp24_anchors; ++anchor) {
                // The head emits a pre-sigmoid score.
                const float score = static_cast<float>(sigmoid(rp24_at(anchor, 8)));
                if (score < config.score_threshold) continue;

                int color_id = 0;
                int num_id = 0;
                float best_color = -std::numeric_limits<float>::max();
                float best_num = -std::numeric_limits<float>::max();
                for (int c = 0; c < 4; ++c) {
                    const float value = rp24_at(anchor, static_cast<size_t>(9 + c));
                    if (value > best_color) { best_color = value; color_id = c; }
                }
                for (int n = 0; n < 9; ++n) {
                    const float value = rp24_at(anchor, static_cast<size_t>(13 + n));
                    if (value > best_num) { best_num = value; num_id = n; }
                }

                NnDetection detection;
                detection.class_id = num_id;
                detection.confidence = score;
                detection.color = rp24Color(color_id);
                detection.name = rp24Name(num_id);
                detection.large = (num_id == 8);   // Bb is the only large plate
                for (int k = 0; k < 4; ++k) {
                    detection.keypoints[static_cast<std::size_t>(k)] = cv::Point2f(
                        rp24_at(anchor, static_cast<size_t>(2 * k)) /
                            static_cast<float>(scale) + static_cast<float>(roi.x),
                        rp24_at(anchor, static_cast<size_t>(2 * k + 1)) /
                            static_cast<float>(scale) + static_cast<float>(roi.y));
                }
                detection.box = cv::boundingRect(std::vector<cv::Point2f>(
                    detection.keypoints.begin(), detection.keypoints.end()));
                boxes.emplace_back(static_cast<int>(detection.box.x),
                                   static_cast<int>(detection.box.y),
                                   std::max(1, static_cast<int>(detection.box.width)),
                                   std::max(1, static_cast<int>(detection.box.height)));
                confidences.push_back(score);
                candidates.push_back(detection);
            }
            return selectDetections(std::move(candidates), std::move(boxes),
                                    std::move(confidences), image);
        }

        // Expected layout is [1, 4 + classes + 8, anchors]; find the anchor axis
        // from the shape so a transposed export still works.
        constexpr int kClasses = 38;
        constexpr int kKeypointValues = 8;
        constexpr int kAttributes = 4 + kClasses + kKeypointValues;
        const bool anchors_last = shape[1] == kAttributes;
        const size_t anchors = anchors_last ? shape[2] : shape[1];
        const size_t attributes = anchors_last ? shape[1] : shape[2];
        if (attributes < kAttributes) {
            std::cerr << "NnArmorDetector: unexpected output shape" << std::endl;
            return detections;
        }

        const float* data = output.data<float>();
        const auto at = [&](size_t anchor, size_t attribute) {
            return anchors_last
                ? data[anchor + anchors * attribute]
                : data[anchor * attributes + attribute];
        };

        std::vector<cv::Rect> boxes;
        std::vector<float> confidences;
        std::vector<NnDetection> candidates;
        for (size_t anchor = 0; anchor < anchors; ++anchor) {
            float best_score = 0.0f;
            int best_class = -1;
            for (int cls = 0; cls < kClasses; ++cls) {
                const float score = at(anchor, static_cast<size_t>(4 + cls));
                if (score > best_score) {
                    best_score = score;
                    best_class = cls;
                }
            }
            if (best_class < 0 || best_score < config.score_threshold) continue;

            const float cx = at(anchor, 0);
            const float cy = at(anchor, 1);
            const float w = at(anchor, 2);
            const float h = at(anchor, 3);

            NnDetection detection;
            detection.class_id = best_class;
            detection.confidence = best_score;
            const ClassSpec& spec = classTable()[static_cast<std::size_t>(best_class)];
            detection.color = spec.color;
            detection.name = spec.name;
            detection.large = spec.large;
            for (int index = 0; index < 4; ++index) {
                detection.keypoints[static_cast<std::size_t>(index)] = cv::Point2f(
                    at(anchor, static_cast<size_t>(4 + kClasses + 2 * index)) / static_cast<float>(scale) +
                        static_cast<float>(roi.x),
                    at(anchor, static_cast<size_t>(4 + kClasses + 2 * index + 1)) / static_cast<float>(scale) +
                        static_cast<float>(roi.y));
            }
            detection.box = cv::Rect2f(
                static_cast<float>(roi.x) + static_cast<float>((cx - 0.5f * w) / scale),
                static_cast<float>(roi.y) + static_cast<float>((cy - 0.5f * h) / scale),
                static_cast<float>(w / scale), static_cast<float>(h / scale));

            boxes.emplace_back(static_cast<int>(detection.box.x),
                               static_cast<int>(detection.box.y),
                               std::max(1, static_cast<int>(detection.box.width)),
                               std::max(1, static_cast<int>(detection.box.height)));
            confidences.push_back(best_score);
            candidates.push_back(detection);
        }

        return selectDetections(std::move(candidates), std::move(boxes),
                                std::move(confidences), image);
    }

    std::vector<NnDetection> NnArmorDetector::selectDetections(
        std::vector<NnDetection> candidates, std::vector<cv::Rect> boxes,
        std::vector<float> confidences, const cv::Mat& image)
    {
        const NnDetectorConfig& config = impl_->config;
        std::vector<NnDetection> detections;

        std::vector<int> kept;
        cv::dnn::NMSBoxes(boxes, confidences,
                          static_cast<float>(config.score_threshold),
                          static_cast<float>(config.nms_threshold), kept);

        const KeypointRefiner refiner(config.refiner);
        for (const int index : kept) {
            NnDetection detection = candidates[static_cast<std::size_t>(index)];
            if (detection.confidence < config.min_confidence) continue;
            if (config.filter_by_color) {
                const int wanted = config.enemy_color == 1 ? 1 : 0;
                if (static_cast<int>(detection.color) != wanted) continue;
            }
            sortKeypoints(detection.keypoints);

            // Replace the regressed corners with locally re-detected light-bar
            // endpoints when that is geometrically consistent, otherwise keep
            // what the network produced.
            const RefineResult refined = refiner.refine(image, detection.keypoints);
            if (refined.refined) {
                detection.keypoints = refined.keypoints;
                detection.refined = true;
                detection.refine_shift_px = refined.center_shift_px;
            }
            detections.push_back(detection);
        }
        return detections;
    }

    // Keeps the preprocessed input alive while the request runs: the input
    // tensor wraps the blob's memory, and an asynchronous request reads it after
    // startAsync() has already returned.
    struct NnTicket {
        ov::InferRequest request;
        cv::Mat blob;
        cv::Mat image;
        double scale = 1.0;
        cv::Rect roi;
        double preprocess_ms = 0.0;
        int64 submitted_tick = 0;
    };

    std::shared_ptr<NnTicket> NnArmorDetector::startAsync(
        const cv::Mat& image, const cv::Rect& requested_roi)
    {
        if (image.empty()) return nullptr;
        const int64 tick = cv::getTickCount();
        const NnDetectorConfig& config = impl_->config;

        cv::Rect roi = config.roi;
        if (requested_roi.area() > 0) {
            roi = requested_roi;
            roi &= cv::Rect(0, 0, image.cols, image.rows);
            if (roi.width <= 1 || roi.height <= 1) {
                roi = cv::Rect(0, 0, image.cols, image.rows);
            }
        } else if (config.use_roi) {
            if (roi.width == -1) roi.width = image.cols;
            if (roi.height == -1) roi.height = image.rows;
            roi &= cv::Rect(0, 0, image.cols, image.rows);
            if (roi.width <= 1 || roi.height <= 1) return nullptr;
        } else {
            roi = cv::Rect(0, 0, image.cols, image.rows);
        }

        // 数字变焦（与能量机关的 input_pad_scale 同一个杠杆）：把 ROI 按比例收缩再
        // letterbox 回输入尺寸，远处的目标就被放大回训练尺度。远距（7.5 m）靶面在
        // 640 输入里只有 ~10 px，网络关键点会塌缩 → PnP 给 0.3 m 的荒唐解。
        // A/B 用 ULTRA_VISION_NN_CANVAS_SCALE（0.35~1.0，1.0 = 关）。
        static const double canvas_scale = [] {
            const char* value = std::getenv("ULTRA_VISION_NN_CANVAS_SCALE");
            const double parsed = value != nullptr ? std::atof(value) : 1.0;
            return (parsed > 0.05 && parsed < 1.0) ? parsed : 1.0;
        }();
        if (canvas_scale < 0.999) {
            const int width = std::max(16, static_cast<int>(std::lround(roi.width * canvas_scale)));
            const int height = std::max(16, static_cast<int>(std::lround(roi.height * canvas_scale)));
            roi = cv::Rect(roi.x + (roi.width - width) / 2, roi.y + (roi.height - height) / 2,
                           width, height);
            roi &= cv::Rect(0, 0, image.cols, image.rows);
        }

        const cv::Mat cropped = image(roi);
        const int size = config.input_size;
        const double scale = std::min(
            static_cast<double>(size) / cropped.rows,
            static_cast<double>(size) / cropped.cols);
        const int scaled_w = static_cast<int>(cropped.cols * scale);
        const int scaled_h = static_cast<int>(cropped.rows * scale);
        cv::Mat canvas(size, size, CV_8UC3, cv::Scalar(0, 0, 0));
        cv::resize(cropped, canvas(cv::Rect(0, 0, scaled_w, scaled_h)),
                   cv::Size(scaled_w, scaled_h));

        auto ticket = std::make_shared<NnTicket>();
        ticket->blob = cv::dnn::blobFromImage(
            canvas, 1.0 / 255.0, cv::Size(size, size), cv::Scalar(), true, false);
        ticket->image = image;
        ticket->scale = scale;
        ticket->roi = roi;
        ticket->submitted_tick = tick;

        ov::Tensor input_tensor(
            ov::element::f32,
            {1, 3, static_cast<size_t>(size), static_cast<size_t>(size)},
            ticket->blob.ptr<float>());
        ticket->request = impl_->compiled.create_infer_request();
        ticket->request.set_input_tensor(input_tensor);
        ticket->request.start_async();
        ticket->preprocess_ms =
            (cv::getTickCount() - tick) * 1000.0 / cv::getTickFrequency();
        return ticket;
    }

    std::vector<NnDetection> NnArmorDetector::finish(
        const std::shared_ptr<NnTicket>& ticket)
    {
        std::vector<NnDetection> detections;
        if (!ticket) return detections;
        ticket->request.wait();
        const double waited_ms = (cv::getTickCount() - ticket->submitted_tick) *
            1000.0 / cv::getTickFrequency();
        impl_->timing.preprocess_ms = ticket->preprocess_ms;
        impl_->timing.infer_ms = std::max(0.0, waited_ms - ticket->preprocess_ms);

        const int64 tick = cv::getTickCount();
        detections = postprocess(ticket->request.get_output_tensor(), ticket->image,
                                 ticket->scale, ticket->roi);
        impl_->timing.postprocess_ms =
            (cv::getTickCount() - tick) * 1000.0 / cv::getTickFrequency();
        return detections;
    }

    std::vector<NnDetection> NnArmorDetector::detect(const cv::Mat& image,
                                                     const cv::Rect& roi)
    {
        return finish(startAsync(image, roi));
    }

    const NnDetectorTiming& NnArmorDetector::timing() const
    {
        return impl_->timing;
    }

    std::string NnArmorDetector::runtimeInfo() const
    {
        std::string info;
        try {
            const std::vector<std::string> devices = impl_->core.get_available_devices();
            info += "devices=";
            for (const auto& device : devices) info += device + " ";
        } catch (const std::exception& error) {
            info += std::string("(devices unavailable: ") + error.what() + ")";
        }
        try {
            info += "| threads=" + std::to_string(
                impl_->compiled.get_property(ov::inference_num_threads));
        } catch (const std::exception&) {
        }
        return info;
    }

    std::vector<Armor> NnArmorDetector::detectArmors(const cv::Mat& image,
                                                     const cv::Rect& roi)
    {
        return toArmors(detect(image, roi));
    }

    std::vector<Armor> NnArmorDetector::toArmors(
        const std::vector<NnDetection>& detections)
    {
        std::vector<Armor> armors;
        for (const NnDetection& detection : detections) {
            Armor armor;
            armor.armor_type = detection.large ? large : small;
            armor.number = detection.class_id;
            armor.solve_result = false;
            // Reuse the classical corner slots so solveArmorPnP() is unchanged:
            // it reads left.bottom, left.top, right.top, right.bottom.
            armor.left.top = detection.keypoints[0];
            armor.right.top = detection.keypoints[1];
            armor.right.bottom = detection.keypoints[2];
            armor.left.bottom = detection.keypoints[3];
            armor.left.center = 0.5f * (armor.left.top + armor.left.bottom);
            armor.right.center = 0.5f * (armor.right.top + armor.right.bottom);
            armor.left.color = detection.color == NnColor::red ? 1 : 0;
            armor.right.color = armor.left.color;
            armor.Points_2D = {
                armor.left.bottom, armor.left.top,
                armor.right.top, armor.right.bottom
            };
            armors.push_back(armor);
        }
        return armors;
    }

    std::string NnArmorDetector::labelName(const NnDetection& detection)
    {
        return std::string(colorName(detection.color)) + "/" +
               nameName(detection.name) + "/" +
               (detection.large ? "big" : "small");
    }

    const char* NnArmorDetector::className(int class_id)
    {
        if (class_id < 0 || class_id >= static_cast<int>(classTable().size())) {
            return "unknown";
        }
        static const std::vector<std::string> names = [] {
            std::vector<std::string> result;
            for (const ClassSpec& spec : classTable()) {
                result.push_back(std::string(colorName(spec.color)) + "/" +
                                 nameName(spec.name) + "/" +
                                 (spec.large ? "big" : "small"));
            }
            return result;
        }();
        return names[static_cast<std::size_t>(class_id)].c_str();
    }
} // namespace auto_aim
