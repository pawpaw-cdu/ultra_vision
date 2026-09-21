#ifndef ULTRA_VISION_NN_DETECTOR_HPP
#define ULTRA_VISION_NN_DETECTOR_HPP

// Neural-network armor detector.
//
// The model is the SZU open-source four-keypoint armor detector (YOLO11, 38
// classes = color x number x size, plus 4 corner keypoints). It is deployed as
// OpenVINO IR and wrapped here so that the rest of the pipeline is unchanged:
// detect() returns the same auto_aim::Armor type the classical detector
// produces, so PnP, the whole-chassis EKF and the control chain keep working.
//
// Compared with the classical detector this removes two weak links that the
// joint simulator testing exposed: the light-bar tilt/size heuristics that made
// detection flicker, and the pixel-based large/small classification that fed
// PnP a wrong plate width. Here the size comes from the model class instead.

#include <array>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>

#include "common/types.hpp"
#include "perception/keypoint_refiner.hpp"

namespace auto_aim
{
    // Model class order of the 38-class armor head.
    enum class NnColor { blue = 0, red = 1, extinguish = 2, purple = 3 };
    enum class NnName {
        one = 0, two, three, four, five, sentry, outpost, base, not_armor
    };

    struct NnDetectorConfig {
        std::string model_path;                 // OpenVINO IR .xml
        std::string device = "CPU";
        // Output head of the model:
        //   "yolo11" : [1, 4 + 38 classes + 8 keypoints, anchors]
        //   "rp24"   : [1, anchors, 8 keypoints + 1 score + 4 colors + 9 numbers]
        // The RP24 head is the lightweight MobileNetV3-based detector, which
        // measures ~4.7x faster per frame than the YOLO11 export at the same
        // 640x640 input size.
        std::string output_format = "yolo11";
        int input_size = 640;
        double score_threshold = 0.7;           // raw class score gate
        double nms_threshold = 0.3;
        double min_confidence = 0.8;            // confidence kept after NMS
        bool use_roi = false;
        cv::Rect roi;
        // Same convention as DetectorConfig: 1 = red, 0 = blue.
        int enemy_color = 0;
        bool filter_by_color = true;
        // Scheduling hints. LATENCY keeps one stream and is what a per-frame
        // blocking call wants; THROUGHPUT uses several streams and helps when
        // several infer requests overlap. 0 keeps the OpenVINO default.
        std::string performance_mode = "latency";
        int num_threads = 0;
        int num_streams = 0;
        // Classical refinement of the regressed corners (see keypoint_refiner).
        KeypointRefinerConfig refiner;
    };

    struct NnDetection {
        int class_id = -1;
        NnColor color = NnColor::blue;
        NnName name = NnName::not_armor;
        bool large = false;
        float confidence = 0.0f;
        cv::Rect2f box;
        // top-left, top-right, bottom-right, bottom-left
        std::array<cv::Point2f, 4> keypoints{};
        // Set when the corners were replaced by the local light-bar refinement.
        bool refined = false;
        double refine_shift_px = 0.0;
    };

    // Stage timing of the most recent detect() call, in milliseconds.
    struct NnDetectorTiming {
        double preprocess_ms = 0.0;
        double infer_ms = 0.0;
        double postprocess_ms = 0.0;
    };

    // One inference in flight. Created by startAsync(), consumed by finish().
    // It owns the preprocessed input buffer, which must stay alive until the
    // asynchronous inference completes.
    struct NnTicket;

    class NnArmorDetector
    {
    public:
        explicit NnArmorDetector(const NnDetectorConfig& config);
        ~NnArmorDetector();

        NnArmorDetector(const NnArmorDetector&) = delete;
        NnArmorDetector& operator=(const NnArmorDetector&) = delete;
        NnArmorDetector(NnArmorDetector&&) noexcept;
        NnArmorDetector& operator=(NnArmorDetector&&) noexcept;

        // Raw detections in full-image pixel coordinates.
        // `roi` restricts inference to that region (results are still reported
        // in full-image coordinates). An empty rect uses the configured ROI.
        std::vector<NnDetection> detect(const cv::Mat& image, const cv::Rect& roi = {});

        // Detections converted into the shared armor type, ready for PnP.
        std::vector<Armor> detectArmors(const cv::Mat& image,
                                        const cv::Rect& roi = {});

        static const char* className(int class_id);
        // Semantic label of a detection, e.g. "blue/three/small". Model-agnostic,
        // so tests and logs stay comparable across detection heads.
        static std::string labelName(const NnDetection& detection);

        const NnDetectorTiming& timing() const;
        std::string runtimeInfo() const;

        // Asynchronous path: preprocess + start the request, return a ticket.
        // finish() waits for that ticket and post-processes it. Splitting the
        // two lets several requests overlap, which is what actually raises
        // throughput: a single blocking infer() on a 640x640 input costs
        // ~65 ms, so a synchronous caller cannot exceed ~15 fps no matter how
        // many threads wait on it. The ticket owns the preprocessed buffer,
        // which must outlive the request.
        std::shared_ptr<NnTicket> startAsync(const cv::Mat& image,
                                             const cv::Rect& roi = {});
        std::vector<NnDetection> finish(const std::shared_ptr<NnTicket>& ticket);

        // Convert finished detections into the shared armor type used by PnP.
        static std::vector<Armor> toArmors(
            const std::vector<NnDetection>& detections);

    private:
        // Decode the raw network output into detections. `scale` and `roi`
        // describe how the image was letterboxed into the network input.
        std::vector<NnDetection> postprocess(const ov::Tensor& output,
                                             const cv::Mat& image,
                                             double scale, const cv::Rect& roi);
        // Shared tail of both heads: NMS, confidence/colour filtering and the
        // optional keypoint refinement.
        std::vector<NnDetection> selectDetections(
            std::vector<NnDetection> candidates,
            std::vector<cv::Rect> boxes,
            std::vector<float> confidences,
            const cv::Mat& image);

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };
} // namespace auto_aim

#endif // ULTRA_VISION_NN_DETECTOR_HPP
