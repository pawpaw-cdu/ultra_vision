// 下位机链路探针：直接用 io::Gimbal（正式读线程 + 协议）跑几秒，
// 打印收帧率 / CRC 错帧率 / 云台状态 / 模式，并可选下发一条测试指令。
//
// 用途：上真机第一步 —— 判定"线通不通、协议对不对、CRC 口径对不对"。
// 比 python 手撸读法可靠：正式读线程是 poll 定长读，不会因为没及时读而丢字节
// （丢一个字节就会整段错位，CRC 通过率暴跌，容易误判成协议不对）。
//
//   tools/gimbal_probe <config_dir> [seconds] [--send]
//     --send: 2 s 后下发一条 mode=1 的静止指令（角度取当前回传值），
//             用来确认下位机能收到我们的帧（它那边的 fps_x 会涨）。

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <thread>

#include <yaml-cpp/yaml.h>

#include "config_loader.hpp"
#include "io/gimbal/gimbal.hpp"

int main(int argc, char* argv[])
{
    const std::string config_dir = argc > 1 ? argv[1] : "configs";
    const double seconds = argc > 2 ? std::atof(argv[2]) : 5.0;
    bool send_command = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--send") send_command = true;
    }

    const YAML::Node serial_file = YAML::LoadFile(config_dir + "/serial.yaml");
    io::SerialConfig config = auto_aim::loadSerialConfig(serial_file);
    std::cout << "[probe] device=" << config.device << " baud=" << config.baud << std::endl;

    io::Gimbal gimbal(config);
    if (!gimbal.connected()) {
        std::cerr << "[probe] 打不开串口，检查 USB / udev 规则 / 是否被别的进程占用"
                  << std::endl;
        return 1;
    }

    const auto start = std::chrono::steady_clock::now();
    auto last_report = start;
    io::Gimbal::Stats previous;
    bool sent = false;
    while (std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() <
           seconds) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        const auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration<double>(now - last_report).count() < 1.0) continue;
        last_report = now;

        const auto stats = gimbal.stats();
        const auto state = gimbal.state();
        const uint64_t frames = stats.frames - previous.frames;
        const uint64_t crc_errors = stats.crc_errors - previous.crc_errors;
        const uint64_t total = frames + crc_errors;
        previous = stats;
        std::cout << "[probe] 收帧 " << frames << "/s"
                  << " CRC错 " << crc_errors
                  << " (通过率 " << (total > 0 ? 100.0 * frames / total : 0.0) << "%)"
                  << " mode=" << io::gimbalModeName(gimbal.mode())
                  << " yaw=" << state.yaw * 180.0 / M_PI << "deg"
                  << " pitch=" << state.pitch * 180.0 / M_PI << "deg"
                  << " vel=(" << state.yaw_vel << "," << state.pitch_vel << ")"
                  << " bullet=" << state.bullet_speed << "m/s"
                  << " count=" << state.bullet_count
                  << " reconnects=" << stats.reconnects << std::endl;

        if (send_command && !sent && state.valid) {
            sent = true;
            const bool ok = gimbal.send(true, false, state.yaw, 0.0, 0.0, state.pitch, 0.0, 0.0);
            std::cout << "[probe] 下发一条 mode=1（控制不开火，角度=当前回传值）: "
                      << (ok ? "OK" : "失败") << std::endl;
        }
    }
    std::cout << "[probe] 结束" << std::endl;
    return 0;
}
