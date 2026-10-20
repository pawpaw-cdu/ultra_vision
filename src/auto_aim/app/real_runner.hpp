#ifndef AUTO_AIM_APP_REAL_RUNNER_HPP
#define AUTO_AIM_APP_REAL_RUNNER_HPP

// 组合层：真机入口的主循环与实时控制实现（相机 + 下位机串口 + 图像闭环）。
//
// 与 app/sim_runner 对称：仿真入口把 TCP 帧源/时间基/曲线搬进 SimRunner，真机入口
// 把相机/串口/图像闭环搬进 RealRunner，node.cpp 只留"解析配置目录 + 跑"。
//
// 分层约定（docs/layer_dependencies.md）：能力层（perception/Kalman/control）负责算法，
// 本文件只做真机专属的 IO 与执行层装配（相机 SDK、串口、下发的世界系换算、图像闭环）。

#include <memory>
#include <string>

namespace auto_aim
{
    namespace app
    {
        class RealRunner
        {
        public:
            explicit RealRunner(std::string config_dir);

            /// @brief 跑完整条真机链路，返回进程退出码（相机打不开时 -1）。
            int run();

        private:
            std::string config_dir_;
        };
    } // namespace app
} // namespace auto_aim

#endif // AUTO_AIM_APP_REAL_RUNNER_HPP
