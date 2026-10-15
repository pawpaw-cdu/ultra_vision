#ifndef SIM_RECEIVER_UDP_INPUT_HPP
#define SIM_RECEIVER_UDP_INPUT_HPP

#include <atomic>
#include <cstdint>
#include <map>
#include <memory>
#include <netinet/in.h>
#include <string>
#include <vector>

#include "sim_packet.hpp"

namespace sim_receiver {

// UDP 图像流接收端（仿真器 `DAEDALUS_UDP=1`，或 camera.yaml 里 `protocol: udp`）。
//
// 仿真器把一帧 JPEG 按 1400 字节切片，每片带 40 字节头（大端）：
//   magic(4)="DAED" | version(2)=1 | header_len(2)=40 | frame_id(4)
//   | timestamp_us(8) | total_len(4) | offset(4) | payload_len(4)
//   | gimbal_yaw(4,f32) | gimbal_pitch(4,f32)
//
// 这里做三件事：
//   1. 周期性向 host:port 发订阅报文（任意小报文，仿真器只认来源地址；
//      订阅 TTL 3s，过期就停推，所以要每秒续期一次）；
//   2. 按 frame_id 重组分片，整帧缓冲从 BufferPool 借——拼完或丢弃时自动
//      归还，稳态零分配；
//   3. 只交付**完整**帧，且时间戳不倒退：缺片的、比已交付帧更旧的直接丢。
//      UDP 不重传，视觉端只要"最新一帧"，按时间戳判新旧就够了。
//
// 指令（FIRE / GIMBAL / RESET）不走这里，仍然走 TCP 7667。
class UdpInput : public PacketSource {
public:
    explicit UdpInput(std::string host, uint16_t port);
    ~UdpInput() override;

    UdpInput(const UdpInput&) = delete;
    UdpInput& operator=(const UdpInput&) = delete;

    bool connect() override;
    bool connected() const override;

    // 等到"下一帧完整图像"才返回 Frame；期间没有数据返回 Idle（流正常，
    // 只是暂时没帧——调用方不要把它当断线去重连，重连会跑掉半帧状态）。
    ReadStatus read(SimPacket& packet) override;

    // 读用 poll 带超时，本身就会周期性返回，不需要从别处打断 recvfrom()。
    void shutdown() override {}
    void close() override;

    void setTimeoutMs(int timeout_ms);
    uint16_t port() const { return port_; }
    const std::string& host() const { return host_; }

private:
    struct PendingFrame {
        BufferPool::Buffer data;      // 整帧缓冲，大小 = total_len
        uint32_t total_len = 0;
        uint32_t received = 0;
        uint64_t timestamp_us = 0;
        float gimbal_yaw = 0.0f;
        float gimbal_pitch = 0.0f;
    };

    void subscribe();
    void subscribeIfDue();
    void prunePending(uint32_t newest_complete);

    std::string host_;
    uint16_t port_;
    std::atomic<int> sock_{-1};
    struct sockaddr_in target_ {};
    int timeout_ms_ = 100;
    uint64_t last_subscribe_ms_ = 0;
    uint64_t last_timestamp_us_ = 0;
    uint32_t last_frame_id_ = 0;
    bool delivered_any_ = false;
    std::shared_ptr<BufferPool> pool_;
    std::map<uint32_t, PendingFrame> pending_;
    std::vector<uint8_t> scratch_;   // 单报文接收缓冲（复用，稳态零分配）
};

} // namespace sim_receiver

#endif // SIM_RECEIVER_UDP_INPUT_HPP
