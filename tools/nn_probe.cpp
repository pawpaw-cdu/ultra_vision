// Small driver for the neural-network armor detector.
//
// Usage:
//   nn_probe <model.xml> <image-or-video> [max_frames] [--device CPU|GPU]
//
// It reports the detections of the first frames and writes an annotated image
// (nn_probe.jpg) so the model can be checked against real footage or against
// frames captured from the simulator.

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "perception/nn_detector.hpp"

int main(int argc, char** argv)
{
    if (argc < 3) {
        std::cerr << "usage: nn_probe <model.xml> <image-or-video> [max_frames] "
                     "[--device CPU|GPU]\n";
        return 2;
    }

    const std::string model_path = argv[1];
    const std::string input_path = argv[2];
    int max_frames = 60;
    if (argc > 3) max_frames = std::max(1, std::atoi(argv[3]));
    std::string device = "CPU";
    std::string mode = "latency";
    int threads = 0;
    int streams = 0;
    int config_input = 0;
    std::string config_format = "yolo11";
    for (int i = 3; i + 1 < argc; ++i) {
        if (std::string(argv[i]) == "--device") device = argv[i + 1];
        if (std::string(argv[i]) == "--mode") mode = argv[i + 1];
        if (std::string(argv[i]) == "--threads") threads = std::atoi(argv[i + 1]);
        if (std::string(argv[i]) == "--streams") streams = std::atoi(argv[i + 1]);
        if (std::string(argv[i]) == "--input") config_input = std::atoi(argv[i + 1]);
        if (std::string(argv[i]) == "--format") config_format = argv[i + 1];
    }

    auto_aim::NnDetectorConfig config;
    config.model_path = model_path;
    config.device = device;
    config.score_threshold = 0.25;
    config.min_confidence = 0.25;
    config.nms_threshold = 0.5;
    config.filter_by_color = false;
    config.performance_mode = mode;
    config.num_threads = threads;
    config.num_streams = streams;
    if (config_input > 0) config.input_size = config_input;
    config.output_format = config_format;

    auto_aim::NnArmorDetector detector(config);
    std::cout << "model loaded: " << model_path << " (device " << device << ")"
              << std::endl;
    std::cout << "runtime: " << detector.runtimeInfo() << std::endl;

    cv::VideoCapture capture;
    cv::Mat frame;
    const bool from_video = capture.open(input_path);
    if (from_video) {
        if (!capture.read(frame) || frame.empty()) {
            std::cerr << "failed to read first frame from " << input_path << "\n";
            return 1;
        }
    } else {
        frame = cv::imread(input_path);
        if (frame.empty()) {
            std::cerr << "failed to read image " << input_path << "\n";
            return 1;
        }
    }

    int frames = 0;
    int total = 0;
    double total_ms = 0.0;
    double pre_ms = 0.0;
    double infer_ms = 0.0;
    double post_ms = 0.0;
    cv::Mat annotated = frame.clone();
    while (!frame.empty() && frames < max_frames) {
        const int64 start = cv::getTickCount();
        const std::vector<auto_aim::NnDetection> detections = detector.detect(frame);
        const double frame_ms =
            (cv::getTickCount() - start) * 1000.0 / cv::getTickFrequency();
        total_ms += frame_ms;
        pre_ms += detector.timing().preprocess_ms;
        infer_ms += detector.timing().infer_ms;
        post_ms += detector.timing().postprocess_ms;
        total += static_cast<int>(detections.size());

        if (frames == 0) {
            int refined = 0;
            double shift_sum = 0.0;
            for (const auto& detection : detections) {
                std::cout << "  det conf=" << cv::format("%.3f", detection.confidence)
                          << " class=" << detection.class_id
                          << " (" << auto_aim::NnArmorDetector::labelName(detection) << ")"
                          << (detection.refined ? " refined" : " raw")
                          << " shift=" << cv::format("%.1f", detection.refine_shift_px)
                          << " box=" << detection.box << std::endl;
                if (detection.refined) { ++refined; shift_sum += detection.refine_shift_px; }
                for (const auto& point : detection.keypoints) {
                    cv::circle(annotated, point, 3, cv::Scalar(0, 255, 0), -1);
                }
                cv::rectangle(annotated, detection.box, cv::Scalar(0, 165, 255), 2);
            }
            if (!detections.empty()) {
                std::cout << "  refined " << refined << "/" << detections.size()
                          << " detections, mean center shift "
                          << (refined ? shift_sum / refined : 0.0) << " px" << std::endl;
            }
        }

        ++frames;
        if (!capture.isOpened()) break;
        if (!capture.read(frame)) break;
    }

    std::cout << "frames=" << frames << " detections=" << total
              << " mean_infer_ms=" << (frames > 0 ? total_ms / frames : 0.0)
              << std::endl;
    if (frames > 0) {
        std::cout << "breakdown ms: preprocess=" << pre_ms / frames
                  << " infer=" << infer_ms / frames
                  << " postprocess=" << post_ms / frames << std::endl;
    }
    cv::imwrite("nn_probe.jpg", annotated);
    std::cout << "wrote nn_probe.jpg" << std::endl;
    return 0;
}
