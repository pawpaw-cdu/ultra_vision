#ifndef ULTRA_VISION_IO_GIMBAL_PROTOCOL_HPP
#define ULTRA_VISION_IO_GIMBAL_PROTOCOL_HPP

// 视觉端 ↔ 下位机（STM32）串口协议 —— 纯数据 + 纯函数，可离线单测。
//
// 这份帧格式是下位机同事给的 visual_task.h 里的 `TJ_T_t` / `TJ_R_t`，
// 与 sp_vision 的 `io/gimbal/gimbal.hpp` 里的 `GimbalToVision` / `VisionToGimbal`
// **逐字段一致**（同一套协议，两边互通）：
//
//   下位机 → 视觉 (43 B)：'S','P' + mode + q(wxyz) + yaw/yaw_vel/pitch/pitch_vel
//                          + bullet_speed + bullet_count + crc16
//   视觉 → 下位机 (29 B)：'S','P' + mode + yaw/yaw_vel/yaw_acc
//                          + pitch/pitch_vel/pitch_acc + crc16
//
// mode（下位机→视觉）：0 空闲 / 1 自瞄 / 2 小符 / 3 大符
// mode（视觉→下位机）：0 不控制 / 1 控制云台不开火 / 2 控制云台且开火
//      → 下位机的 fire_ctrl() 只看这一位，所以**开火决策就在我们这一侧**：
//        mode=2 才会真开火。
//
// CRC16 = RM 标准（见 io/serial/crc16.hpp），覆盖"除最后两字节以外"的全部内容；
// 下位机是 `get_CRC16_check_sum((uint8_t*)&tj_t, 41, 0xffff)` 后整帧发送，
// 我们这边必须用同样的口径（sizeof - 2）。

#include <cstddef>
#include <cstdint>

namespace io
{
#pragma pack(push, 1)
    struct GimbalToVision
    {
        uint8_t head[2] = {'S', 'P'};
        uint8_t mode = 0;      // 0 空闲 / 1 自瞄 / 2 小符 / 3 大符
        float q[4] = {1.0f, 0.0f, 0.0f, 0.0f};   // wxyz
        float yaw = 0.0f;      // 云台绝对 yaw（rad）
        float yaw_vel = 0.0f;
        float pitch = 0.0f;    // 云台绝对 pitch（rad）
        float pitch_vel = 0.0f;
        float bullet_speed = 0.0f;
        uint16_t bullet_count = 0;
        uint16_t crc16 = 0;
    };

    struct VisionToGimbal
    {
        uint8_t head[2] = {'S', 'P'};
        uint8_t mode = 0;      // 0 不控制 / 1 控制不开火 / 2 控制且开火
        float yaw = 0.0f;
        float yaw_vel = 0.0f;
        float yaw_acc = 0.0f;
        float pitch = 0.0f;
        float pitch_vel = 0.0f;
        float pitch_acc = 0.0f;
        uint16_t crc16 = 0;
    };
#pragma pack(pop)

    static_assert(sizeof(GimbalToVision) == 43, "GimbalToVision 必须是 43 字节");
    static_assert(sizeof(VisionToGimbal) == 29, "VisionToGimbal 必须是 29 字节");

    enum class GimbalMode
    {
        IDLE = 0,       // 空闲
        AUTO_AIM = 1,   // 自瞄
        SMALL_BUFF = 2, // 小符
        BIG_BUFF = 3,   // 大符
    };

    const char* gimbalModeName(GimbalMode mode);

    /// @brief 下位机快照：云台姿态、角速度、弹速、累计弹数。
    struct GimbalState
    {
        double yaw = 0.0;
        double yaw_vel = 0.0;
        double pitch = 0.0;
        double pitch_vel = 0.0;
        double bullet_speed = 0.0;
        uint16_t bullet_count = 0;
        bool valid = false;
    };

    /// @brief 填好 CRC16（覆盖除最后两字节外的全部内容）。
    void finalizeFrame(VisionToGimbal& frame);

    /// @brief 校验一整帧（长度 + 头 + CRC）。长度必须正好是 sizeof(GimbalToVision)。
    bool validFrame(const uint8_t* buffer, std::size_t length);

    /// @brief 解析一整帧；返回 false 表示长度/头/CRC 任一不通过（原样返回，不修改 out）。
    bool parseFrame(const uint8_t* buffer, std::size_t length, GimbalToVision& out);
} // namespace io

#endif // ULTRA_VISION_IO_GIMBAL_PROTOCOL_HPP
