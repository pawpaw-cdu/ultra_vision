#include "ultra_vision/core/logger.hpp"

#include <chrono>
#include <ctime>
#include <fstream>
#include <iostream>
#include <mutex>

namespace ultra_vision
{
    namespace
    {
        std::mutex log_mutex;
        std::ofstream log_file;
        bool debug_enabled = false;

        const char* levelName(LogLevel level)
        {
            switch (level) {
            case LogLevel::DEBUG: return "DEBUG";
            case LogLevel::INFO: return "INFO";
            case LogLevel::WARNING: return "WARN";
            case LogLevel::ERROR: return "ERROR";
            }
            return "UNKNOWN";
        }
    }

    void Logger::configure(bool enabled, const std::string& file_path)
    {
        std::lock_guard<std::mutex> lock(log_mutex);
        debug_enabled = enabled;
        if (!file_path.empty()) {
            log_file.open(file_path, std::ios::out | std::ios::trunc);
        }
    }

    void Logger::shutdown()
    {
        std::lock_guard<std::mutex> lock(log_mutex);
        if (log_file.is_open()) log_file.close();
        debug_enabled = false;
    }

    bool Logger::debugEnabled()
    {
        std::lock_guard<std::mutex> lock(log_mutex);
        return debug_enabled;
    }

    void Logger::log(LogLevel level,
                     const std::string& module,
                     const std::string& message)
    {
        std::lock_guard<std::mutex> lock(log_mutex);
        if (level == LogLevel::DEBUG && !debug_enabled) return;

        const auto now = std::chrono::system_clock::now();
        const auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()).count() % 1000;
        const std::time_t time = std::chrono::system_clock::to_time_t(now);
        char timestamp[32];
        std::strftime(timestamp, sizeof(timestamp), "%H:%M:%S", std::localtime(&time));

        const std::string line =
            std::string(timestamp) + "." + std::to_string(millis) + " [" +
            levelName(level) + "] [" + module + "] " + message;
        std::cerr << line << '\n';
        if (log_file.is_open()) log_file << line << '\n';
    }
}
