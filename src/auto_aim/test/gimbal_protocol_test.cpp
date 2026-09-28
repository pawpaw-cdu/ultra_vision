// 下位机协议层的离线自检（不需要硬件、不需要相机）。
//
// 覆盖三件事，任何一件错了真机都会"连上但不动/乱动"：
//   1. CRC16 必须是 RM 标准。RM 那张 crc16_table 的前 8 项
//      (0x0000,0x1189,0x2312,0x329b,...) 逐项等于 **CRC-16/MCRF4XX**
//      （多项式 0x1021 反射=0x8408、init 0xffff、refin/refout=true、无异或输出），
//      其标准校验值 "123456789" -> **0x6F91** 钉死；再用逐位实现交叉核对。
//      注意：它**不是** CRC-16/MODBUS（那个是 poly 0xA001、校验值 0x4B37），
//      两者查表口径不同 —— 这一条搞错，真机就是"能连上但 CRC 全不过"。
//   2. 两个结构体的**字节布局**必须和下位机 visual_task.h 的 TJ_T_t/TJ_R_t 一致
//      （43 B / 29 B，各字段偏移见下面的断言）—— 编译器 padding 变了真机就全错；
//   3. 打包 → 校验 → 解析往返一致；改一个字节、改帧头、截断都必须被判非法。

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "io/gimbal/protocol.hpp"
#include "io/serial/crc16.hpp"

namespace
{
    int failures = 0;

    bool expect(bool condition, const char* message)
    {
        if (!condition) {
            std::printf("FAILED: %s\n", message);
            ++failures;
        }
        return condition;
    }

    /// @brief 逐位实现的 CRC16（poly 0x1021 反射=0x8408、init 0xffff、无异或输出
    ///        = CRC-16/MCRF4XX），用来独立核对查表版，而不是拿同一个实现自己比自己。
    uint16_t crc16Bitwise(const uint8_t* data, std::size_t length)
    {
        uint16_t crc = 0xffff;
        for (std::size_t i = 0; i < length; ++i) {
            crc ^= data[i];
            for (int bit = 0; bit < 8; ++bit) {
                crc = (crc & 1u) ? static_cast<uint16_t>((crc >> 1) ^ 0x8408u)
                                 : static_cast<uint16_t>(crc >> 1);
            }
        }
        return crc;
    }
} // namespace

int main()
{
    // ---- 1. CRC16：标准校验值 + 逐位交叉核对 ----
    const char* check = "123456789";
    const uint16_t standard = io::crc16(reinterpret_cast<const uint8_t*>(check), 9);
    expect(standard == 0x6F91,
           "CRC16 不是 RM 标准（\"123456789\" 应为 0x6F91 / MCRF4XX）");

    std::vector<uint8_t> buffer(41);
    for (std::size_t i = 0; i < buffer.size(); ++i) {
        buffer[i] = static_cast<uint8_t>((i * 37 + 11) & 0xff);
    }
    expect(io::crc16(buffer.data(), buffer.size()) ==
               crc16Bitwise(buffer.data(), buffer.size()),
           "查表 CRC16 与逐位实现不一致");
    expect(io::crc16(nullptr, 0) == 0xffff, "空数据的 CRC16 应为初值 0xffff");

    // ---- 2. 字节布局（对齐下位机）----
    expect(sizeof(io::GimbalToVision) == 43, "GimbalToVision 不是 43 字节");
    expect(sizeof(io::VisionToGimbal) == 29, "VisionToGimbal 不是 29 字节");
    {
        io::VisionToGimbal frame;
        const auto* base = reinterpret_cast<const uint8_t*>(&frame);
        expect(reinterpret_cast<const uint8_t*>(&frame.head) - base == 0, "head 偏移错误");
        expect(reinterpret_cast<const uint8_t*>(&frame.mode) - base == 2, "mode 偏移错误");
        expect(reinterpret_cast<const uint8_t*>(&frame.yaw) - base == 3, "yaw 偏移错误");
        expect(reinterpret_cast<const uint8_t*>(&frame.yaw_vel) - base == 7, "yaw_vel 偏移错误");
        expect(reinterpret_cast<const uint8_t*>(&frame.yaw_acc) - base == 11, "yaw_acc 偏移错误");
        expect(reinterpret_cast<const uint8_t*>(&frame.pitch) - base == 15, "pitch 偏移错误");
        expect(reinterpret_cast<const uint8_t*>(&frame.pitch_vel) - base == 19,
               "pitch_vel 偏移错误");
        expect(reinterpret_cast<const uint8_t*>(&frame.pitch_acc) - base == 23,
               "pitch_acc 偏移错误");
        expect(reinterpret_cast<const uint8_t*>(&frame.crc16) - base == 27, "crc16 偏移错误");
    }
    {
        io::GimbalToVision frame;
        const auto* base = reinterpret_cast<const uint8_t*>(&frame);
        expect(reinterpret_cast<const uint8_t*>(&frame.head) - base == 0, "sensor head 偏移错误");
        expect(reinterpret_cast<const uint8_t*>(&frame.mode) - base == 2, "sensor mode 偏移错误");
        expect(reinterpret_cast<const uint8_t*>(&frame.q) - base == 3, "q 偏移错误");
        expect(reinterpret_cast<const uint8_t*>(&frame.yaw) - base == 19, "sensor yaw 偏移错误");
        expect(reinterpret_cast<const uint8_t*>(&frame.bullet_speed) - base == 35,
               "bullet_speed 偏移错误");
        expect(reinterpret_cast<const uint8_t*>(&frame.bullet_count) - base == 39,
               "bullet_count 偏移错误");
        expect(reinterpret_cast<const uint8_t*>(&frame.crc16) - base == 41, "sensor crc16 偏移错误");
    }

    // ---- 3. 下发帧：打包 → 校验 ----
    io::VisionToGimbal command;
    command.mode = 2;   // 控制云台且开火
    command.yaw = 0.5f;
    command.yaw_vel = 1.5f;
    command.yaw_acc = 8.0f;
    command.pitch = -0.2f;
    command.pitch_vel = 0.5f;
    command.pitch_acc = 4.0f;
    io::finalizeFrame(command);
    const auto* raw = reinterpret_cast<const uint8_t*>(&command);
    expect(raw[0] == 'S' && raw[1] == 'P', "下发帧头不是 'S','P'");
    expect(command.crc16 == crc16Bitwise(raw, sizeof(command) - 2), "下发帧 CRC 未覆盖前 27 字节");
    expect(io::checkCrc16(raw, sizeof(command)), "下发帧自校验失败");

    // ---- 4. 上行帧：模拟下位机打包 → 解析 → 字段一致 ----
    io::GimbalToVision sensor;
    sensor.mode = 1;    // 自瞄
    sensor.q[0] = 0.9999f; sensor.q[1] = 0.001f; sensor.q[2] = 0.002f; sensor.q[3] = 0.003f;
    sensor.yaw = 0.25f;
    sensor.yaw_vel = 2.0f;
    sensor.pitch = -0.15f;
    sensor.pitch_vel = -0.5f;
    sensor.bullet_speed = 22.0f;
    sensor.bullet_count = 137;
    const auto* sensor_raw = reinterpret_cast<const uint8_t*>(&sensor);
    sensor.crc16 = crc16Bitwise(sensor_raw, sizeof(sensor) - 2);   // 下位机口径：前 41 字节

    std::vector<uint8_t> wire(sizeof(io::GimbalToVision));
    std::memcpy(wire.data(), &sensor, wire.size());
    io::GimbalToVision parsed;
    expect(io::validFrame(wire.data(), wire.size()), "合法上行帧被判非法");
    expect(io::parseFrame(wire.data(), wire.size(), parsed), "合法上行帧解析失败");
    expect(parsed.mode == 1 && parsed.bullet_count == 137 &&
               parsed.bullet_speed == 22.0f && parsed.yaw == 0.25f && parsed.pitch == -0.15f,
           "解析出的字段与打包值不一致");

    // ---- 5. 非法帧必须被拒 ----
    auto corrupted = wire;
    corrupted[20] ^= 0x01;                       // 改数据字段
    expect(!io::validFrame(corrupted.data(), corrupted.size()), "数据被篡改的帧未被拒");
    auto bad_head = wire;
    bad_head[0] = 'X';
    expect(!io::validFrame(bad_head.data(), bad_head.size()), "帧头错误的帧未被拒");
    expect(!io::validFrame(wire.data(), wire.size() - 1), "长度不足的帧未被拒");
    expect(!io::validFrame(nullptr, sizeof(io::GimbalToVision)), "空指针未被拒");

    if (failures == 0) {
        std::printf("gimbal_protocol_test: 全部通过（CRC=0x6F91 MCRF4XX 标准校验值 / 43B+29B 布局 / "
                    "往返一致 / 非法帧被拒）\n");
        return 0;
    }
    std::printf("gimbal_protocol_test: %d 项失败\n", failures);
    return 1;
}
