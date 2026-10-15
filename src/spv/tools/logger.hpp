#ifndef SPV_TOOLS_LOGGER_HPP
#define SPV_TOOLS_LOGGER_HPP

// sp_vision_25 的 logger 用 spdlog（NUC 有、开发机没有）。为了"两边都能编"，这里
// 换成一个零依赖的桩：接口保持 spdlog 的调用形态（logger()->debug("... {}", x)），
// 但默认什么都不做。要临时看日志就把 SPV_LOG_VERBOSE 设上（打到 stderr）。
//
// 注意：这是**移植层**的替换件，不是 sp_vision 的原件；滤波器/构造/重投影的逻辑
// 一行没动（见 git 历史里的 docs/spv_port.md 的说明）。

#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>

namespace spv_tools_detail
{
    inline bool verbose()
    {
        static const bool v = (std::getenv("SPV_LOG_VERBOSE") != nullptr);
        return v;
    }

    class StubLogger
    {
    public:
        template <typename... Args>
        void debug(const char* fmt, Args&&...) const
        {
            dump("debug", fmt);
        }
        template <typename... Args>
        void info(const char* fmt, Args&&...) const
        {
            dump("info", fmt);
        }
        template <typename... Args>
        void warn(const char* fmt, Args&&...) const
        {
            dump("warn", fmt);
        }
        template <typename... Args>
        void error(const char* fmt, Args&&...) const
        {
            dump("error", fmt);
        }

    private:
        static void dump(const char* level, const char* fmt)
        {
            // 只打格式串本身（不求值参数），够用且零依赖。
            if (verbose()) std::cerr << "[spv/" << level << "] " << fmt << std::endl;
        }
    };
}

namespace tools
{
    inline std::shared_ptr<spv_tools_detail::StubLogger> logger()
    {
        static std::shared_ptr<spv_tools_detail::StubLogger> inst =
            std::make_shared<spv_tools_detail::StubLogger>();
        return inst;
    }
}  // namespace tools

#endif  // SPV_TOOLS_LOGGER_HPP
