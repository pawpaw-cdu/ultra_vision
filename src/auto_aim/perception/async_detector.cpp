#include "async_detector.hpp"

#include <chrono>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>

#include "common/standard_clock.hpp"

namespace auto_aim
{
    struct AsyncArmorDetector::Impl {
        explicit Impl(const AsyncDetectorConfig& cfg)
            : config(cfg), detector(cfg.detector),
              in_flight_limit(cfg.detector.num_streams > 0
                                  ? static_cast<std::size_t>(cfg.detector.num_streams)
                                  : 2u)
        {
            worker = std::thread([this] { run(); });
        }

        ~Impl()
        {
            {
                std::lock_guard<std::mutex> lock(mutex);
                stopping = true;
            }
            ready.notify_all();
            if (worker.joinable()) worker.join();
        }

        // Everything that touches OpenVINO happens here, never on the caller's
        // Metadata travels with each request so a finished result keeps the
        // timestamp of the frame it actually came from.
        struct Pending {
            std::shared_ptr<NnTicket> ticket;
            uint64_t sequence = 0;
            uint64_t timestamp_us = 0;
        };

        // Everything that touches OpenVINO happens here, never on the caller's
        // thread: start_async() blocks once the device is saturated (measured:
        // the main loop fell from 24 fps to 5 fps when the caller issued the
        // requests itself). This mirrors the reference implementation, where
        // preprocessing + start_async live on a dedicated capture thread while
        // the tracking/control loop only consumes finished results.
        void run()
        {
            std::deque<Pending> in_flight;
            while (true) {
                {
                    std::unique_lock<std::mutex> lock(mutex);
                    // Only sleep when there is nothing to harvest; with requests
                    // in flight the loop must keep draining them.
                    if (in_flight.empty()) {
                        ready.wait(lock, [this] { return stopping || has_frame; });
                    }
                    if (stopping && !has_frame) return;

                    // Start as many requests as the device has streams.
                    while (has_frame && in_flight.size() < in_flight_limit) {
                        in_flight.push_back(startPendingFrame());
                    }
                }

                if (in_flight.empty()) continue;

                Pending item = in_flight.front();
                in_flight.pop_front();
                const auto start = std::chrono::steady_clock::now();
                const std::vector<NnDetection> detections =
                    detector.finish(item.ticket);
                std::vector<Armor> armors = NnArmorDetector::toArmors(detections);
                const double latency =
                    std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - start).count();

                std::lock_guard<std::mutex> lock(mutex);
                latest.sequence = item.sequence;
                latest.timestamp_us = item.timestamp_us;
                latest.armors = std::move(armors);
                ++completed_count;
                last_latency_ms = latency;
                const double now = StandardClock::nowSeconds();
                if (last_publish_time > 0.0 && now > last_publish_time) {
                    const double instant = 1.0 / (now - last_publish_time);
                    fps_estimate = fps_estimate > 0.0
                        ? 0.9 * fps_estimate + 0.1 * instant : instant;
                }
                last_publish_time = now;
                ++result_generation;
            }
        }

        // Called with the lock held: takes the mailbox frame and starts it.
        Pending startPendingFrame()
        {
            Pending item;
            item.ticket = detector.startAsync(mailbox_frame, mailbox_roi);
            item.sequence = mailbox_sequence;
            item.timestamp_us = mailbox_timestamp;
            has_frame = false;
            return item;
        }

        AsyncDetectorConfig config;
        NnArmorDetector detector;
        const std::size_t in_flight_limit;
        std::thread worker;

        mutable std::mutex mutex;
        std::condition_variable ready;
        bool stopping = false;

        // Single-slot mailbox: a control loop only ever wants the newest frame,
        // so an unconsumed one is dropped rather than queued.
        bool has_frame = false;
        cv::Mat mailbox_frame;
        cv::Rect mailbox_roi;
        uint64_t mailbox_sequence = 0;
        uint64_t mailbox_timestamp = 0;

        Result latest;
        uint64_t result_generation = 0;
        uint64_t consumed_generation = 0;
        uint64_t completed_count = 0;
        uint64_t dropped_count = 0;
        double last_latency_ms = 0.0;
        double fps_estimate = 0.0;
        double last_publish_time = 0.0;
    };

    AsyncArmorDetector::AsyncArmorDetector(const AsyncDetectorConfig& config)
        : impl_(std::make_unique<Impl>(config))
    {
    }

    AsyncArmorDetector::~AsyncArmorDetector() = default;

    void AsyncArmorDetector::submit(const cv::Mat& image, uint64_t sequence,
                                    uint64_t timestamp_us, const cv::Rect& roi)
    {
        if (image.empty()) return;
        {
            std::lock_guard<std::mutex> lock(impl_->mutex);
            if (impl_->has_frame) ++impl_->dropped_count;
            impl_->mailbox_frame = image.clone();
            impl_->mailbox_roi = roi;
            impl_->mailbox_sequence = sequence;
            impl_->mailbox_timestamp = timestamp_us;
            impl_->has_frame = true;
        }
        impl_->ready.notify_one();
    }

    bool AsyncArmorDetector::takeLatest(Result& result)
    {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        if (impl_->result_generation == impl_->consumed_generation) return false;
        impl_->consumed_generation = impl_->result_generation;
        result = impl_->latest;
        return true;
    }

    double AsyncArmorDetector::fps() const
    {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        return impl_->fps_estimate;
    }

    double AsyncArmorDetector::latencyMs() const
    {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        return impl_->last_latency_ms;
    }

    uint64_t AsyncArmorDetector::completed() const
    {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        return impl_->completed_count;
    }

    uint64_t AsyncArmorDetector::dropped() const
    {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        return impl_->dropped_count;
    }
} // namespace auto_aim
