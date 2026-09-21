#ifndef ULTRA_VISION_ASYNC_DETECTOR_HPP
#define ULTRA_VISION_ASYNC_DETECTOR_HPP

// Runs the neural-network armor detector on its own thread.
//
// The joint simulator testing showed that a synchronous neural detector drops
// the whole loop to the inference rate (~15 fps on a CPU device), which starves
// the gimbal controller and the estimator that were both built to run faster
// than the detector. Here the main loop only submits the newest frame and
// consumes finished results, so:
//   * the camera/control loop keeps its own rate,
//   * a slow or uneven detector never blocks it,
//   * each result carries the timestamp of the frame it came from, so the
//     estimator is updated with the correct time base instead of being fed a
//     stale detection as if it were current.
//
// The worker only runs the network: the camera intrinsics of the simulator are
// not known until the first frame arrives, and PnP over a handful of plates
// costs microseconds, so the main loop solves it when it consumes a result.

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "common/types.hpp"
#include "perception/nn_detector.hpp"

namespace auto_aim
{
    struct AsyncDetectorConfig {
        NnDetectorConfig detector;
    };

    class AsyncArmorDetector
    {
    public:
        struct Result {
            uint64_t sequence = 0;
            uint64_t timestamp_us = 0;
            std::vector<Armor> armors;   // camera frame, PnP not yet solved
        };

        explicit AsyncArmorDetector(const AsyncDetectorConfig& config);
        ~AsyncArmorDetector();

        AsyncArmorDetector(const AsyncArmorDetector&) = delete;
        AsyncArmorDetector& operator=(const AsyncArmorDetector&) = delete;

        // Never blocks. If the worker has not picked up the previous frame yet
        // that frame is replaced: an outdated detection is worthless to a
        // control loop.
        // `roi` restricts inference to that region (empty = full frame / the
        // configured ROI). The caller derives it from the current estimate, so
        // the network only looks where the target can be.
        void submit(const cv::Mat& image, uint64_t sequence, uint64_t timestamp_us,
                    const cv::Rect& roi = {});

        // Non-blocking. Returns true once per newly finished result.
        bool takeLatest(Result& result);

        double fps() const;
        double latencyMs() const;
        uint64_t completed() const;
        uint64_t dropped() const;

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };
} // namespace auto_aim

#endif // ULTRA_VISION_ASYNC_DETECTOR_HPP
