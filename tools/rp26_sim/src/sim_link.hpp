#pragma once
//
// 与 simulator_system/simulator（daedalus）的对接层。
//
// 协议与 Ultra_Vision 现有的 sim_receiver 完全一致（见 src/auto_aim/sim_receiver/
// src/tcp_input.cpp 与 simulator 的 src/tcp.rs、src/main.rs）：
//   * 7666：帧流。32 字节大端头 "DAED" + version + width + height + payload_len
//           + timestamp_us(u64) + seq(u32)，payload 是 JPEG。
//   * 7667：指令。每次连接发一行，支持 "GIMBAL <yaw> <pitch>" / "FIRE" / "RESET"。
//   * 7668：遥测。每行一条 RUNE 事件（hit / activated / state），用于打分。

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace rp26_sim
{

struct SimFrame
{
    cv::Mat bgr;
    uint32_t seq = 0;
    uint64_t source_timestamp_us = 0;
};

class SimLink
{
public:
    SimLink(std::string host, uint16_t frame_port, uint16_t command_port, uint16_t telemetry_port);
    ~SimLink();

    SimLink(const SimLink &) = delete;
    SimLink &operator=(const SimLink &) = delete;

    /// @brief 阻塞读一帧（失败时返回空图像，调用方可重试）。
    bool readFrame(SimFrame &frame);

    bool sendGimbal(double yaw, double pitch);
    bool sendFire();
    bool sendReset();

    /// @brief 取回上次调用之后收到的遥测行。
    std::vector<std::string> pollTelemetry();

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace rp26_sim
