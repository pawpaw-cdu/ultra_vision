#ifndef ULTRA_VISION_IO_FRAME_SOURCE_HPP
#define ULTRA_VISION_IO_FRAME_SOURCE_HPP

#include <cstdint>

#include <opencv2/core.hpp>

namespace ultra_vision::io
{
    struct Frame {
        cv::Mat bgr;
        std::uint64_t sequence = 0;
        std::uint64_t source_timestamp_us = 0;
        std::uint64_t local_timestamp_us = 0;
    };

    class IFrameSource
    {
    public:
        virtual ~IFrameSource() = default;
        virtual bool connect() = 0;
        virtual bool connected() const = 0;
        virtual cv::Mat getFrame(
            std::uint64_t* sequence = nullptr,
            std::uint64_t* source_timestamp_us = nullptr,
            std::uint64_t* local_timestamp_us = nullptr) = 0;
    };
}

#endif
