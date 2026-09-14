#ifndef ULTRA_VISION_ENERGY_RUNE_MODULE_HPP
#define ULTRA_VISION_ENERGY_RUNE_MODULE_HPP

#include <cstdint>

#include <opencv2/core.hpp>

namespace ultra_vision::energy_rune
{
    struct EnergyRuneOutput {
        bool valid = false;
        std::uint64_t timestamp_us = 0;
        double yaw = 0.0;
        double pitch = 0.0;
        bool fire = false;
    };

    // Reserved extension point. The first version intentionally ships no
    // energy-rune implementation, but applications can depend on this
    // interface and inject a module later without changing the auto-aim path.
    class IEnergyRuneModule
    {
    public:
        virtual ~IEnergyRuneModule() = default;
        virtual EnergyRuneOutput process(const cv::Mat& bgr,
                                         std::uint64_t local_timestamp_us) = 0;
        virtual void reset() = 0;
    };
}

#endif
