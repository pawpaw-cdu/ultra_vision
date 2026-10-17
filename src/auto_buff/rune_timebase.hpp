#ifndef AUTO_AIM_ENERGY_RUNE_TIMEBASE_HPP
#define AUTO_AIM_ENERGY_RUNE_TIMEBASE_HPP

#include <cstdint>

namespace auto_aim::energy
{
    /// @brief Pick the timestamp used for pose lookup and motion compensation.
    /// Source timestamps are preferred only when both clocks are close enough
    /// to represent the same epoch; otherwise the local arrival time is safer.
    inline uint64_t selectFrameTimestampUs(uint64_t local_timestamp_us,
                                           uint64_t source_timestamp_us,
                                           uint64_t max_skew_us = 1'000'000)
    {
        if (source_timestamp_us == 0) return local_timestamp_us;
        const uint64_t skew = source_timestamp_us > local_timestamp_us
            ? source_timestamp_us - local_timestamp_us
            : local_timestamp_us - source_timestamp_us;
        return skew < max_skew_us ? source_timestamp_us : local_timestamp_us;
    }
} // namespace auto_aim::energy

#endif // AUTO_AIM_ENERGY_RUNE_TIMEBASE_HPP
