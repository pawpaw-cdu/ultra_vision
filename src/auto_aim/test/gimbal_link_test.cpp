// 整链路测试：用**伪终端**冒充下位机，把 io::Gimbal 真跑一遍。
//
// 为什么要有它：协议帧本身（CRC/布局）由 gimbal_protocol_test 钉死了，但真机上
// 出问题的往往是链路层 —— 帧头对不对齐、半帧怎么处理、CRC 错的帧会不会污染状态、
// mode 映射对不对、我们发出去的 mode/角度/CRC 下位机能不能认。这些东西用伪终端
// 在电脑上就能全部验掉，不用等实车。
//
// 覆盖：
//   1. 下位机视角写一帧合法 43B → state/mode/bullet_speed/IMU 队列都更新；
//   2. 我们的 send() 出来的 29B：头、mode(控制+开火=2)、6 个 float、CRC 全对；
//   3. 写一帧 CRC 错的 → 只增加 crc_errors，state 不变；
//   4. mode=0 的帧 → mode() 回到 IDLE（下位机收回控制权）；
//   5. imuAt() 在两个样本之间能插值出中间的姿态。

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <stdlib.h>
#include <unistd.h>

#include "io/gimbal/gimbal.hpp"
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

    /// @brief 逐位 CRC16（CRC-16/MCRF4XX），与下位机口径一致，用来独立造帧。
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

    std::vector<uint8_t> makeMcFrame(uint8_t mode, double yaw, double pitch, double yaw_vel,
                                     double pitch_vel, double bullet_speed, uint16_t bullet_count,
                                     bool break_crc = false)
    {
        io::GimbalToVision frame;
        frame.mode = mode;
        frame.q[0] = 1.0f; frame.q[1] = 0.0f; frame.q[2] = 0.0f; frame.q[3] = 0.0f;
        frame.yaw = static_cast<float>(yaw);
        frame.yaw_vel = static_cast<float>(yaw_vel);
        frame.pitch = static_cast<float>(pitch);
        frame.pitch_vel = static_cast<float>(pitch_vel);
        frame.bullet_speed = static_cast<float>(bullet_speed);
        frame.bullet_count = bullet_count;
        auto* raw = reinterpret_cast<uint8_t*>(&frame);
        frame.crc16 = crc16Bitwise(raw, sizeof(frame) - 2);
        if (break_crc) frame.crc16 ^= 0x00ff;
        std::vector<uint8_t> bytes(sizeof(frame));
        std::memcpy(bytes.data(), &frame, sizeof(frame));
        return bytes;
    }

    bool writeAll(int fd, const std::vector<uint8_t>& bytes)
    {
        std::size_t sent = 0;
        while (sent < bytes.size()) {
            const ssize_t put = ::write(fd, bytes.data() + sent, bytes.size() - sent);
            if (put <= 0) return false;
            sent += static_cast<std::size_t>(put);
        }
        return true;
    }

    /// @brief 从伪终端读满 size 字节（带超时），用于检查我们发出去的帧。
    bool readExact(int fd, uint8_t* buffer, std::size_t size, int timeout_ms)
    {
        std::size_t got = 0;
        const auto deadline = std::chrono::steady_clock::now() +
            std::chrono::milliseconds(timeout_ms);
        while (got < size && std::chrono::steady_clock::now() < deadline) {
            const ssize_t part = ::read(fd, buffer + got, size - got);
            if (part > 0) {
                got += static_cast<std::size_t>(part);
                continue;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        return got == size;
    }
} // namespace

int main()
{
    int master = ::posix_openpt(O_RDWR | O_NOCTTY);
    if (master < 0 || ::grantpt(master) != 0 || ::unlockpt(master) != 0) {
        std::printf("SKIP: 本机不支持伪终端（posix_openpt），跳过链路测试\n");
        return 0;
    }
    const char* slave_path = ::ptsname(master);
    expect(slave_path != nullptr, "ptsname 失败");
    if (slave_path == nullptr) return 1;

    io::SerialConfig config;
    config.device = slave_path;     // Gimbal 打开从端，测试通过主端充当"下位机"
    config.read_timeout_ms = 5;
    io::Gimbal gimbal(config);
    expect(gimbal.connected(), "Gimbal 未能打开伪终端");
    if (!gimbal.connected()) return 1;

    // ---- 1. 下位机 → 视觉：合法帧 ----
    expect(writeAll(master, makeMcFrame(1, 0.30, -0.12, 2.0, -0.5, 22.0, 7)), "写合法帧失败");
    std::this_thread::sleep_for(std::chrono::milliseconds(80));
    {
        const auto state = gimbal.state();
        expect(gimbal.mode() == io::GimbalMode::AUTO_AIM, "mode=1 应映射成 AUTO_AIM");
        expect(state.valid, "合法帧后 state 应为 valid");
        expect(std::abs(state.yaw - 0.30) < 1e-4 && std::abs(state.pitch + 0.12) < 1e-4,
               "yaw/pitch 解析错误");
        expect(std::abs(state.yaw_vel - 2.0) < 1e-4 && std::abs(state.pitch_vel + 0.5) < 1e-4,
               "角速度解析错误");
        expect(std::abs(state.bullet_speed - 22.0) < 1e-3 && state.bullet_count == 7,
               "弹速/弹数解析错误");
        io::ImuSample imu;
        expect(gimbal.latestImu(imu), "IMU 队列为空");
    }

    // ---- 2. 视觉 → 下位机：29B 帧 ----
    expect(gimbal.send(true, true, 0.5, 1.5, 8.0, -0.2, 0.5, 4.0), "send 失败");
    {
        uint8_t buffer[sizeof(io::VisionToGimbal)] = {0};
        expect(readExact(master, buffer, sizeof(buffer), 200), "下位机视角没收到 29B 帧");
        expect(buffer[0] == 'S' && buffer[1] == 'P', "下发帧头错误");
        expect(buffer[2] == 2, "控制且开火时应发 mode=2");
        expect(io::checkCrc16(buffer, sizeof(buffer)), "下发帧 CRC 错误");
        float yaw = 0.0f, yaw_vel = 0.0f, yaw_acc = 0.0f, pitch = 0.0f, pitch_vel = 0.0f,
              pitch_acc = 0.0f;
        std::memcpy(&yaw, buffer + 3, 4);
        std::memcpy(&yaw_vel, buffer + 7, 4);
        std::memcpy(&yaw_acc, buffer + 11, 4);
        std::memcpy(&pitch, buffer + 15, 4);
        std::memcpy(&pitch_vel, buffer + 19, 4);
        std::memcpy(&pitch_acc, buffer + 23, 4);
        expect(yaw == 0.5f && yaw_vel == 1.5f && yaw_acc == 8.0f && pitch == -0.2f &&
                   pitch_vel == 0.5f && pitch_acc == 4.0f,
               "下发帧的角度/前馈字段错位");
    }

    // ---- 3. CRC 错的帧不能改状态 ----
    {
        const auto before = gimbal.state();
        const auto stats_before = gimbal.stats();
        expect(writeAll(master, makeMcFrame(2, 9.9, 9.9, 0, 0, 99.0, 99, true)),
               "写坏帧失败");
        std::this_thread::sleep_for(std::chrono::milliseconds(80));
        const auto after = gimbal.state();
        expect(gimbal.stats().crc_errors > stats_before.crc_errors, "CRC 错误未计数");
        expect(std::abs(after.yaw - before.yaw) < 1e-9 &&
                   std::abs(after.pitch - before.pitch) < 1e-9,
               "CRC 错误的帧污染了云台状态");
    }

    // ---- 4. mode=0 收回控制权 ----
    expect(writeAll(master, makeMcFrame(0, 0.0, 0.0, 0.0, 0.0, 22.0, 7)), "写 mode=0 帧失败");
    std::this_thread::sleep_for(std::chrono::milliseconds(80));
    expect(gimbal.mode() == io::GimbalMode::IDLE, "mode=0 应映射成 IDLE");

    // ---- 5. IMU 插值：两个样本之间取中点 ----
    {
        io::GimbalToVision first;
        first.q[0] = 1.0f; first.q[1] = 0.0f; first.q[2] = 0.0f; first.q[3] = 0.0f;
        first.mode = 1;
        auto* raw = reinterpret_cast<uint8_t*>(&first);
        first.crc16 = crc16Bitwise(raw, sizeof(first) - 2);
        std::vector<uint8_t> bytes(sizeof(first));
        std::memcpy(bytes.data(), &first, sizeof(first));
        expect(writeAll(master, bytes), "写 IMU 帧失败");
        const auto t_before = std::chrono::steady_clock::now();
        std::this_thread::sleep_for(std::chrono::milliseconds(60));
        const auto t_middle = std::chrono::steady_clock::now();
        std::this_thread::sleep_for(std::chrono::milliseconds(60));
        // 第二个样本：绕 Z 轴 90°（w=cos45, z=sin45）
        io::GimbalToVision second = first;
        second.q[0] = 0.9238795f; second.q[3] = 0.3826834f;
        raw = reinterpret_cast<uint8_t*>(&second);
        second.crc16 = crc16Bitwise(raw, sizeof(second) - 2);
        std::memcpy(bytes.data(), &second, sizeof(second));
        expect(writeAll(master, bytes), "写第二个 IMU 帧失败");
        std::this_thread::sleep_for(std::chrono::milliseconds(80));

        io::ImuSample interpolated;
        expect(gimbal.imuAt(t_middle, interpolated), "imuAt 未能插值");
        const double angle = 2.0 * std::atan2(interpolated.z, interpolated.w) * 180.0 / M_PI;
        expect(angle > 1.0 && angle < 89.0,
               "中间时刻的姿态没有落在两个样本之间（应为 0~90°）");
        expect(std::chrono::steady_clock::now() > t_before, "时间基准错误");
    }

    if (failures == 0) {
        std::printf("gimbal_link_test: 全部通过（合法帧→状态/模式/IMU、下发 29B 帧、CRC 错帧隔离、"
                    "mode=0 收回控制、IMU 插值）\n");
        ::close(master);
        return 0;
    }
    std::printf("gimbal_link_test: %d 项失败\n", failures);
    ::close(master);
    return 1;
}
