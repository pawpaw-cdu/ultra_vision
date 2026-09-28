// Latency probe for the energy-rune keypoint models.
//
// The rune detector's cost is dominated by a single OpenVINO inference, so
// choosing a model (and an input size) needs measurements, not guesses. This
// tool loads any of the exported models, optionally tries to reshape the input,
// and reports the latency distribution plus how many anchors clear the score
// threshold.
//
// Usage:
//   rune_model_bench <model.xml> [--image f.png] [--size 640] [--iters 50]
//                    [--threads 4] [--mode latency|throughput] [--score 0.7]

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>

namespace
{
    double percentile(std::vector<double> values, double fraction)
    {
        if (values.empty()) return 0.0;
        std::sort(values.begin(), values.end());
        const std::size_t index = static_cast<std::size_t>(
            fraction * static_cast<double>(values.size() - 1));
        return values[index];
    }

    cv::Mat loadFrame(const std::string& path, int width)
    {
        if (path.empty()) {
            // A synthetic frame is enough to measure the network.
            return cv::Mat(480, 640, CV_8UC3, cv::Scalar(16, 16, 24));
        }
        if (path.size() > 4 && path.substr(path.size() - 4) == ".mp4") {
            cv::VideoCapture capture(path);
            cv::Mat frame;
            if (capture.isOpened()) {
                for (int i = 0; i < 30; ++i) capture >> frame;
            }
            if (frame.empty()) frame = cv::Mat(480, 640, CV_8UC3, cv::Scalar(16, 16, 24));
            if (width > 0 && frame.cols > width) {
                const double scale = static_cast<double>(width) / frame.cols;
                cv::resize(frame, frame, cv::Size(), scale, scale, cv::INTER_AREA);
            }
            return frame;
        }
        return cv::imread(path);
    }
} // namespace

int main(int argc, char* argv[])
{
    if (argc < 2) {
        std::printf("usage: %s <model.xml> [--image path] [--size N] [--iters N] "
                    "[--threads N] [--mode latency|throughput] [--score S]\n",
                    argv[0]);
        return -1;
    }

    std::string model_path = argv[1];
    std::string image_path;
    int input_size = 0; // 0 = keep the model's own input
    int iterations = 50;
    int threads = 4;
    std::string mode = "latency";
    float score_threshold = 0.7f;

    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        const auto next = [&]() { return (i + 1 < argc) ? std::string(argv[++i]) : std::string(); };
        if (arg == "--image") image_path = next();
        else if (arg == "--size") input_size = std::atoi(next().c_str());
        else if (arg == "--iters") iterations = std::atoi(next().c_str());
        else if (arg == "--threads") threads = std::atoi(next().c_str());
        else if (arg == "--mode") mode = next();
        else if (arg == "--score") score_threshold = std::stof(next());
    }

    ov::Core core;
    std::shared_ptr<ov::Model> model;
    try {
        model = core.read_model(model_path);
    } catch (const std::exception& error) {
        std::printf("failed to read %s: %s\n", model_path.c_str(), error.what());
        return -1;
    }

    const std::string input_name = model->input().get_any_name();
    std::printf("model: %s\n", model_path.c_str());
    std::printf("  input  %s %s\n", input_name.c_str(),
                model->input().get_partial_shape().to_string().c_str());
    std::printf("  output %s\n", model->output().get_partial_shape().to_string().c_str());

    if (input_size > 0) {
        try {
            model->reshape({{input_name, ov::Shape{1, 3, static_cast<std::size_t>(input_size),
                                                   static_cast<std::size_t>(input_size)}}});
            std::printf("  reshape to %d: OK -> %s\n", input_size,
                        model->input().get_partial_shape().to_string().c_str());
        } catch (const std::exception& error) {
            std::printf("  reshape to %d: rejected (%s)\n", input_size, error.what());
        }
    }

    ov::AnyMap properties;
    properties.emplace(ov::hint::performance_mode(
        mode == "throughput" ? ov::hint::PerformanceMode::THROUGHPUT
                             : ov::hint::PerformanceMode::LATENCY));
    if (threads > 0) properties.emplace(ov::inference_num_threads(threads));

    ov::CompiledModel compiled = core.compile_model(model, "CPU", properties);
    ov::InferRequest request = compiled.create_infer_request();

    const ov::Shape input_shape = compiled.input().get_shape();
    const int height = static_cast<int>(input_shape[2]);
    const int width = static_cast<int>(input_shape[3]);

    const cv::Mat frame = loadFrame(image_path, 0);
    if (frame.empty()) {
        std::printf("failed to load the input frame\n");
        return -1;
    }

    std::vector<float> buffer(static_cast<std::size_t>(3) * height * width);
    std::vector<double> manual_ms;
    std::vector<double> blob_ms;
    manual_ms.reserve(static_cast<std::size_t>(iterations));
    blob_ms.reserve(static_cast<std::size_t>(iterations));
    for (int iteration = 0; iteration < iterations; ++iteration) {
        // Variant A: resize + copy, then a scalar per-pixel CHW conversion.
        const auto manual_start = std::chrono::steady_clock::now();
        {
            const double scale = std::min(static_cast<double>(width) / frame.rows,
                                          static_cast<double>(height) / frame.cols);
            cv::Mat resized;
            cv::resize(frame, resized, cv::Size(), scale, scale, cv::INTER_LINEAR);
            cv::Mat blob(height, width, CV_8UC3, cv::Scalar(114, 114, 114));
            resized.copyTo(blob(cv::Rect(0, 0, resized.cols, resized.rows)));
            cv::cvtColor(blob, blob, cv::COLOR_BGR2RGB);
            for (int c = 0; c < 3; ++c) {
                for (int y = 0; y < height; ++y) {
                    const cv::Vec3b* row = blob.ptr<cv::Vec3b>(y);
                    for (int x = 0; x < width; ++x) {
                        buffer[static_cast<std::size_t>(c) * height * width +
                               static_cast<std::size_t>(y) * width + x] = row[x][c] / 255.0f;
                    }
                }
            }
        }
        manual_ms.push_back(std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() - manual_start)
                                .count());

        // Variant B: resize straight into the padded canvas, then one
        // cv::dnn::blobFromImage call for the RGB/scale/NCHW conversion.
        const auto blob_start = std::chrono::steady_clock::now();
        {
            const double scale = std::min(static_cast<double>(width) / frame.rows,
                                          static_cast<double>(height) / frame.cols);
            cv::Mat canvas(height, width, CV_8UC3, cv::Scalar(114, 114, 114));
            cv::Mat roi = canvas(cv::Rect(
                0, 0, static_cast<int>(frame.cols * scale), static_cast<int>(frame.rows * scale)));
            cv::resize(frame, roi, roi.size(), 0.0, 0.0, cv::INTER_LINEAR);
            cv::Mat blob = cv::dnn::blobFromImage(canvas, 1.0 / 255.0, cv::Size(), cv::Scalar(),
                                                  true, false, CV_32F);
            std::copy(blob.ptr<float>(), blob.ptr<float>() + buffer.size(), buffer.begin());
        }
        blob_ms.push_back(std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - blob_start)
                              .count());
    }

    ov::Tensor input_tensor(compiled.input().get_element_type(),
                            {1, 3, static_cast<std::size_t>(height),
                             static_cast<std::size_t>(width)},
                            buffer.data());

    // One warm-up run, then the measured loop.
    request.set_input_tensor(input_tensor);
    request.infer();

    std::vector<double> latencies;
    latencies.reserve(static_cast<std::size_t>(iterations));
    int detections = 0;
    float best_score = 0.0f;
    for (int i = 0; i < iterations; ++i) {
        const auto start = std::chrono::steady_clock::now();
        request.infer();
        const auto finish = std::chrono::steady_clock::now();
        latencies.push_back(std::chrono::duration<double, std::milli>(finish - start).count());

        const ov::Tensor output = request.get_output_tensor();
        const ov::Shape shape = output.get_shape();
        if (shape.size() == 3) {
            const bool attributes_first = shape[1] <= shape[2];
            const std::size_t anchors = attributes_first ? shape[2] : shape[1];
            const std::size_t stride = attributes_first ? shape[1] : shape[2];
            const float* data = output.data<const float>();
            detections = 0;
            best_score = 0.0f;
            for (std::size_t anchor = 0; anchor < anchors; ++anchor) {
                const float value = attributes_first
                                        ? data[anchor + anchors * 4]
                                        : data[anchor * stride + 4];
                best_score = std::max(best_score, value);
                if (value >= score_threshold) ++detections;
            }
        }
    }

    double sum = 0.0;
    for (const double value : latencies) sum += value;
    std::printf("  input size   : %dx%d\n", width, height);
    std::printf("  mode/threads : %s / %d\n", mode.c_str(), threads);
    std::printf("  latency ms   : min %.2f  p50 %.2f  p90 %.2f  mean %.2f\n",
                percentile(latencies, 0.0), percentile(latencies, 0.5),
                percentile(latencies, 0.9), sum / static_cast<double>(latencies.size()));
    std::printf("  preprocess ms: manual p50 %.2f  blobFromImage p50 %.2f\n",
                percentile(manual_ms, 0.5), percentile(blob_ms, 0.5));
    std::printf("  anchors>=%.2f: %d (best score %.3f)\n", score_threshold, detections,
                best_score);
    return 0;
}
