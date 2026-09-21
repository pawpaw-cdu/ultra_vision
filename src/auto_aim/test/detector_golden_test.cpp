// Golden-frame regression test for the armor detectors.
//
// Both blocking detector bugs found during the simulator joint testing were
// invisible before this test existed:
//   * the light-bar tilt formula used acos(dir . vertical), which scores a
//     perfectly vertical bar as either 0 or 180 degrees, so the |tilt| filter
//     silently dropped about half of all bars;
//   * Light::color was never assigned while find_armors compared it against
//     enemy_color, i.e. the pairing filter read uninitialised memory.
// Neither showed up in any unit test because nothing ever asserted on detector
// output. This test pins the behaviour of both detectors on recorded frames.
//
// Expectations live in assets/detector_expectations.csv:
//   frame,min_lights,min_armors,nn_labels
// nn_labels is a ';'-separated list of model-agnostic labels ("colour/name/size")
// that must all be detected by the network (empty = not checked). Comparing
// labels instead of raw class ids keeps the expectations valid when the
// detection head changes, e.g. from the YOLO11 export to the lighter RP24 one.
//
// The classical detector is checked with "at least" semantics because contour
// geometry is not bit-exact across OpenCV versions; the network classes are
// checked exactly, because a wrong class silently feeds PnP the wrong plate.

#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "config_loader.hpp"
#include "perception/detector.hpp"

#ifdef ULTRA_VISION_USE_OPENVINO
#include "perception/nn_detector.hpp"
#endif

#ifndef ULTRA_VISION_TEST_ASSETS
#define ULTRA_VISION_TEST_ASSETS "src/auto_aim/test/assets"
#endif

#ifndef ULTRA_VISION_CONFIG_DIR
#define ULTRA_VISION_CONFIG_DIR "configs"
#endif

namespace
{
    struct Expectation {
        std::string frame;
        int min_lights = 0;
        int min_armors = 0;
        std::vector<std::string> nn_labels;
    };

    std::vector<Expectation> loadExpectations(const std::string& path)
    {
        std::vector<Expectation> expectations;
        std::ifstream file(path);
        std::string line;
        bool first = true;
        while (std::getline(file, line)) {
            if (first) { first = false; continue; }   // header
            if (line.empty() || line[0] == '#') continue;
            std::stringstream stream(line);
            std::string frame, lights, armors, classes;
            std::getline(stream, frame, ',');
            std::getline(stream, lights, ',');
            std::getline(stream, armors, ',');
            std::getline(stream, classes, ',');
            if (frame.empty()) continue;

            Expectation expectation;
            expectation.frame = frame;
            expectation.min_lights = std::stoi(lights);
            expectation.min_armors = std::stoi(armors);
            std::stringstream label_stream(classes);
            std::string item;
            while (std::getline(label_stream, item, ';')) {
                if (!item.empty()) expectation.nn_labels.push_back(item);
            }
            expectations.push_back(expectation);
        }
        return expectations;
    }

    // Mirrors the classical detection chain used by node_sim / node.
    void runClassicalDetector(const cv::Mat& image, const auto_aim::DetectorConfig& cfg,
                              int& lights_out, int& armors_out)
    {
        auto_aim::Detector detector(image, cfg);
        cv::Mat gray, binary, dilated;
        detector.gray_img(image, gray);
        cv::GaussianBlur(gray, gray, cv::Size(3, 3), 0);
        detector.binary_img(gray, binary, cfg.binary_threshold);
        detector.open_close_img(binary, dilated);

        std::vector<std::vector<cv::Point>> contours;
        detector.find_contours(dilated, contours);
        std::vector<auto_aim::Light> lights;
        detector.find_lights(contours, lights, image, cfg.enemy_color);
        std::vector<auto_aim::Armor> armors;
        if (!lights.empty()) detector.find_armors(lights, armors);
        lights_out = static_cast<int>(lights.size());
        armors_out = static_cast<int>(armors.size());
    }
}

int main()
{
    const std::string assets = ULTRA_VISION_TEST_ASSETS;
    const std::string config_dir = ULTRA_VISION_CONFIG_DIR;
    const std::vector<Expectation> expectations =
        loadExpectations(assets + "/detector_expectations.csv");

    if (expectations.empty()) {
        std::cerr << "FAILED: no expectations loaded from " << assets << std::endl;
        return 1;
    }

    const YAML::Node detector_file = YAML::LoadFile(config_dir + "/detector.yaml");
    const auto_aim::DetectorConfig det_cfg = auto_aim::loadDetectorConfig(detector_file);

    bool passed = true;
    int checked = 0;

#ifdef ULTRA_VISION_USE_OPENVINO
    std::unique_ptr<auto_aim::NnArmorDetector> nn_detector;
    if (auto_aim::neuralDetectorEnabled(detector_file)) {
        const auto_aim::NnDetectorConfig nn_cfg =
            auto_aim::loadNeuralDetectorConfig(detector_file, config_dir);
        if (!nn_cfg.model_path.empty()) {
            nn_detector = std::make_unique<auto_aim::NnArmorDetector>(nn_cfg);
        } else {
            std::cerr << "note: neural detector enabled but no model path; "
                         "skipping the network checks" << std::endl;
        }
    }
#endif

    std::cout << "frame,lights,armors" << std::endl;
    for (const Expectation& expectation : expectations) {
        const std::string path = assets + "/" + expectation.frame;
        const cv::Mat image = cv::imread(path);
        if (image.empty()) {
            std::cerr << "FAILED: cannot read " << path << std::endl;
            passed = false;
            continue;
        }

        int lights = 0;
        int armors = 0;
        runClassicalDetector(image, det_cfg, lights, armors);
        std::cout << expectation.frame << "," << lights << "," << armors << std::endl;
        ++checked;

        if (lights < expectation.min_lights) {
            std::cerr << "FAILED: " << expectation.frame << " classical lights "
                      << lights << " < expected " << expectation.min_lights
                      << " (light-bar filtering rejected too much)" << std::endl;
            passed = false;
        }
        if (armors < expectation.min_armors) {
            std::cerr << "FAILED: " << expectation.frame << " classical armors "
                      << armors << " < expected " << expectation.min_armors
                      << " (light pairing rejected too much)" << std::endl;
            passed = false;
        }

#ifdef ULTRA_VISION_USE_OPENVINO
        if (nn_detector && !expectation.nn_labels.empty()) {
            const std::vector<auto_aim::NnDetection> detections =
                nn_detector->detect(image);
            for (const std::string& wanted : expectation.nn_labels) {
                const bool found = std::any_of(
                    detections.begin(), detections.end(),
                    [&wanted](const auto_aim::NnDetection& detection) {
                        return auto_aim::NnArmorDetector::labelName(detection) == wanted;
                    });
                if (!found) {
                    std::cerr << "FAILED: " << expectation.frame
                              << " neural detector missed label " << wanted << "; got:";
                    for (const auto& detection : detections) {
                        std::cerr << " " << auto_aim::NnArmorDetector::labelName(detection);
                    }
                    std::cerr << std::endl;
                    passed = false;
                }
            }
        }
#endif
    }

    if (!passed) return 1;
    std::cout << "detector_golden_test passed, frames=" << checked << std::endl;
    return 0;
}
