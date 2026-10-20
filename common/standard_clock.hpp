#ifndef ULTRA_VISION_STANDARD_CLOCK_HPP
#define ULTRA_VISION_STANDARD_CLOCK_HPP

#include <chrono>
#include <cstdint>

namespace auto_aim
{
    // Process-wide monotonic clock. The epoch is aligned once with the local
    // system clock so timestamps remain readable in logs, while elapsed time
    // is measured exclusively with steady_clock.
    class StandardClock
    {
    public:
        using TimePoint = std::chrono::steady_clock::time_point;

        static TimePoint now()
        {
            return std::chrono::steady_clock::now();
        }

        static std::uint64_t nowUs()
        {
            const Epoch epoch = getEpoch();
            const auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - epoch.steady_start);
            return epoch.system_start_us +
                static_cast<std::uint64_t>(elapsed.count());
        }

        static double nowSeconds()
        {
            return secondsFromUs(nowUs());
        }

        static double secondsFromUs(std::uint64_t timestamp_us)
        {
            return static_cast<double>(timestamp_us) * 1e-6;
        }

        /// Prefer the source timestamp only when both clocks are close enough
        /// to represent the same epoch; otherwise local arrival time is safer.
        static std::uint64_t selectFrameTimestampUs(
            std::uint64_t local_timestamp_us, std::uint64_t source_timestamp_us,
            std::uint64_t max_skew_us = 1'000'000)
        {
            if (source_timestamp_us == 0) return local_timestamp_us;
            const std::uint64_t skew = source_timestamp_us > local_timestamp_us
                ? source_timestamp_us - local_timestamp_us
                : local_timestamp_us - source_timestamp_us;
            return skew < max_skew_us ? source_timestamp_us : local_timestamp_us;
        }

    private:
        struct Epoch {
            TimePoint steady_start;
            std::uint64_t system_start_us = 0;
        };

        static Epoch getEpoch()
        {
            static const Epoch epoch = makeEpoch();
            return epoch;
        }

        static Epoch makeEpoch()
        {
            Epoch epoch;
            epoch.steady_start = std::chrono::steady_clock::now();
            epoch.system_start_us = static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::system_clock::now().time_since_epoch()).count());
            return epoch;
        }
    };
}

#endif
