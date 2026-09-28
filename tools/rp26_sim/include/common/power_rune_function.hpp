// 影子头文件：RP-26Rune-main/src/core/algorithm/power_rune/include/common/
// power_rune_function.hpp 的副本，只改一处——
//   timestamp_from_nanoseconds() 里 timetool::Timestamp(std::chrono::nanoseconds(ns))
//   在 libc++（macOS）上不成立：那里 system_clock 的 duration 是微秒，
//   time_point 不能从纳秒 duration 直接构造（Linux/libstdc++ 上恰好是纳秒，所以他们自己
//   的工程能编过）。这里改成 duration_cast 显式转换，语义不变。
#pragma once

#include "Time_generated.h"
#include "time/time.hpp"

#include <cmath>
#include <cstdint>
#include <numbers>

namespace power_rune_function
{

struct rad
{
    static constexpr double period =2.0 * std::numbers::pi_v<double>;
};
struct deg
{
    static constexpr double period = 360.0;
};

template<typename T>
double calculate_delta_phase(double new_phase, double old_phase)
{
    constexpr double period = T::period;
    constexpr double half_period = period * 0.5;

    double delta_phase = new_phase - old_phase;
    while (delta_phase > half_period)
    {
        delta_phase -= period;
    }
    while (delta_phase < -half_period)
    {
        delta_phase += period;
    }
    return delta_phase;
}

template<typename T>
double normalize_phase(double phase)
{
    constexpr double period = T::period;
    constexpr double half_period = period * 0.5;

    phase = std::fmod(phase + half_period, period);

    if (phase < 0)
    {
        phase += period;
    }

    return phase - half_period;
}

inline timetool::Timestamp timestamp_from_nanoseconds(uint64_t timestamp_ns)
{
    return timetool::Timestamp(std::chrono::duration_cast<std::chrono::system_clock::duration>(
        std::chrono::nanoseconds(timestamp_ns)));
}

inline foxglove::Time timestamp_to_foxglove_time(const timetool::Timestamp &timestamp)
{
    constexpr uint64_t nanoseconds_per_second = 1'000'000'000ULL;
    const uint64_t timestamp_ns = timetool::to_epoch_nanoseconds(timestamp);
    return foxglove::Time(
        static_cast<uint32_t>(timestamp_ns / nanoseconds_per_second),
        static_cast<uint32_t>(timestamp_ns % nanoseconds_per_second));
}

} // namespace power_rune_function
namespace PRF = power_rune_function;
