#include "ultra_vision/core/standard_clock.hpp"

#include <chrono>
#include <cmath>
#include <iostream>
#include <thread>

namespace
{
    bool expect(bool condition, const char* message)
    {
        if (!condition) std::cerr << "FAILED: " << message << std::endl;
        return condition;
    }
}

int main()
{
    bool passed = true;
    const auto first = ultra_vision::StandardClock::nowUs();
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    const auto second = ultra_vision::StandardClock::nowUs();

    passed &= expect(second > first,
                     "standard clock must be monotonic");
    passed &= expect(second - first >= 1000,
                     "standard clock must advance in microseconds");
    passed &= expect(
        std::abs(ultra_vision::StandardClock::secondsFromUs(1500000) - 1.5) < 1e-9,
        "microsecond conversion must use seconds");

    const auto time_point_first = ultra_vision::StandardClock::now();
    const auto time_point_second = ultra_vision::StandardClock::now();
    passed &= expect(time_point_second >= time_point_first,
                     "time-point API must be monotonic");

    if (!passed) return 1;
    std::cout << "standard_clock_test passed, delta_us="
              << (second - first) << std::endl;
    return 0;
}
