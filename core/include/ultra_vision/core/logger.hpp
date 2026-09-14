#ifndef ULTRA_VISION_CORE_LOGGER_HPP
#define ULTRA_VISION_CORE_LOGGER_HPP

#include <string>

namespace ultra_vision
{
    enum class LogLevel {
        DEBUG,
        INFO,
        WARNING,
        ERROR
    };

    class Logger
    {
    public:
        static void configure(bool debug_enabled,
                              const std::string& file_path = "");
        static void shutdown();
        static void log(LogLevel level,
                        const std::string& module,
                        const std::string& message);
        static bool debugEnabled();
    };
}

#endif
