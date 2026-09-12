#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <cstdlib>
#include <string>
#include "../include/config.hpp"
#include "../include/detector.hpp"
#include "../include/tracker.hpp"
#include "../include/utils.hpp"
#include "../include/pnp_solver.hpp"

#ifndef ULTRA_VISION_BUFF_CONFIG_DIR
#define ULTRA_VISION_BUFF_CONFIG_DIR "../config"
#endif
#ifndef ULTRA_VISION_BUFF_VIDEO_PATH
#define ULTRA_VISION_BUFF_VIDEO_PATH "../result.mp4"
#endif

int main(int argc, char* argv[]) {
    std::string config_dir = ULTRA_VISION_BUFF_CONFIG_DIR;
    std::string video_path = ULTRA_VISION_BUFF_VIDEO_PATH;

    if (argc > 1) config_dir = argv[1];
    if (argc > 2) video_path = argv[2];
    if (const char* env = std::getenv("ULTRA_VISION_CONFIG_DIR")) config_dir = env;
    if (const char* env = std::getenv("ULTRA_VISION_VIDEO_PATH")) video_path = env;

    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::cerr << "Error: Cannot open video file " << video_path << std::endl;
        return -1;
    }

    energy::ConfigManager cfg_mgr;
    if (!cfg_mgr.load(config_dir + "/detector.yaml",
                      config_dir + "/tracker.yaml",
                      config_dir + "/solve.yaml")) {
        std::cerr << "Failed to load config" << std::endl;
        return -1;
    }

    energy::Detector detector(cfg_mgr.getDetector());
    energy::Tracker tracker(cfg_mgr.getTracker());
    energy::solver solve(cfg_mgr.getSolvePnP());

    double last_timestamp = 0;
    cv::Mat frame;
    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        double now = std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
        double dt = (last_timestamp > 0) ? (now - last_timestamp) : 0.033;
        if (dt > 0.1) dt = 0.033;
        last_timestamp = now;

        auto det_result = detector.process(frame);
        auto track_result = tracker.update(det_result, dt);
        std::vector<energy::SolvePnPresult> pnp_results;
        if (track_result.valid) {
            for (const auto& blade : track_result.blades) {
                pnp_results.push_back(solve.getSolveResult(blade));
            }
        }

        cv::Mat display = frame.clone();
        energy::drawModel(display, track_result);
        std::vector<cv::Point2f> blade_centers;
        blade_centers.reserve(det_result.blades.size());
        for (const auto& blade : det_result.blades) {
            blade_centers.push_back(blade.center);
        }
        energy::showBladesROI(frame, blade_centers, 80);

        cv::resize(display, display, cv::Size(960, 540));
        cv::imshow("Energy Detection", display);

        if (cv::waitKey(10) == 'q') break;
    }
    return 0;
}
