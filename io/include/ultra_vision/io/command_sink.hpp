#ifndef ULTRA_VISION_IO_COMMAND_SINK_HPP
#define ULTRA_VISION_IO_COMMAND_SINK_HPP

namespace ultra_vision::io
{
    class ICommandSink
    {
    public:
        virtual ~ICommandSink() = default;
        virtual bool sendFireCommand() = 0;
        virtual bool sendGimbalCommand(double yaw, double pitch) = 0;
    };
}

#endif
