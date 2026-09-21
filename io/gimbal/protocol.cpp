#include "protocol.hpp"

#include <cstring>

#include "io/serial/crc16.hpp"

namespace io
{
    const char* gimbalModeName(GimbalMode mode)
    {
        switch (mode) {
        case GimbalMode::IDLE: return "IDLE";
        case GimbalMode::AUTO_AIM: return "AUTO_AIM";
        case GimbalMode::SMALL_BUFF: return "SMALL_BUFF";
        case GimbalMode::BIG_BUFF: return "BIG_BUFF";
        }
        return "UNKNOWN";
    }

    void finalizeFrame(VisionToGimbal& frame)
    {
        frame.crc16 = crc16(reinterpret_cast<const uint8_t*>(&frame),
                            sizeof(frame) - sizeof(frame.crc16));
    }

    bool validFrame(const uint8_t* buffer, std::size_t length)
    {
        if (buffer == nullptr || length != sizeof(GimbalToVision)) return false;
        if (buffer[0] != 'S' || buffer[1] != 'P') return false;
        return checkCrc16(buffer, length);
    }

    bool parseFrame(const uint8_t* buffer, std::size_t length, GimbalToVision& out)
    {
        if (!validFrame(buffer, length)) return false;
        std::memcpy(&out, buffer, sizeof(out));
        return true;
    }
} // namespace io
