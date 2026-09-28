#ifndef ULTRA_VISION_IO_SERIAL_CRC16_HPP
#define ULTRA_VISION_IO_SERIAL_CRC16_HPP

// RoboMaster 标准 CRC16（多项式 0x8005 反射、初值 0xffff、无异或输出）。
// 下位机用的是 `get_CRC16_check_sum(data, len, 0xffff)`（CRC8_CRC16.c），
// 逐字节与下面这张表一致；sp_vision 的 tools/crc.cpp 也是同一张表。
// 校验口径：**帧尾最后两字节**按小端存放 CRC，校验时用"去掉这两字节"的数据重算。

#include <cstddef>
#include <cstdint>

namespace io
{
    uint16_t crc16(const uint8_t* data, std::size_t length);
    /// @brief 校验 `data[0 .. length-3]` 与末尾两字节（小端）是否一致。
    bool checkCrc16(const uint8_t* data, std::size_t length);
} // namespace io

#endif // ULTRA_VISION_IO_SERIAL_CRC16_HPP
