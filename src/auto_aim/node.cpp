// 真机入口（auto_aim_real）——薄组合根。
//
// 相机（Hik/Galaxy）+ 下位机串口 + 图像闭环整条实现都在 app/real_runner.cpp，
// 与仿真入口 app/sim_runner.cpp 同一套分层（能力层 perception/Kalman/control 共用，
// 入口只负责各自的 IO）。本文件只做"解析配置目录 + 跑"。
//
// 注意：本入口需要大恒 Galaxy / 海康 MVS SDK，本机（macOS）不编译；
// 构建与验证见 tools/build_real.sh（NUC）。

#include "app/config_paths.hpp"
#include "app/real_runner.hpp"

int main(int argc, char** argv)
{
    auto_aim::app::RealRunner runner(auto_aim::app::resolveConfigDir(argc, argv));
    return runner.run();
}
