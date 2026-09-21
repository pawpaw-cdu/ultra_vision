#ifndef ULTRA_VISION_IO_SERIAL_PORT_HPP
#define ULTRA_VISION_IO_SERIAL_PORT_HPP

// 极简 POSIX 串口（termios），只做三件事：打开 8N1、带超时读、写。
//
// 为什么不引第三方 serial 库：我们只需要 USB CDC 虚拟串口（下位机 STM32 的
// CDC_Transmit_FS 那一侧），termios 足够；少一个依赖，交叉编译到 aarch64 也简单。
// 读用 poll() + 超时（而不是 VMIN/VTIME），因为 CDC 的读在没数据时要能"立刻返回 0"、
// 由上层决定要不要继续攒包 —— 协议是按 43 字节定长帧读的（见 io/gimbal/protocol.hpp）。

#include <cstddef>
#include <cstdint>
#include <string>

namespace io
{
    class SerialPort
    {
    public:
        SerialPort() = default;
        ~SerialPort();

        SerialPort(const SerialPort&) = delete;
        SerialPort& operator=(const SerialPort&) = delete;

        /// @brief 打开串口。@param device 如 "/dev/gimbal"、"/dev/ttyACM0"
        /// @param baud 115200 等（USB CDC 下无意义，占位）
        /// @param read_timeout_ms 单次 read 的最长等待
        /// @param quiet true 时不打印单次失败（"auto" 探测设备时用，避免一次刷 7 条）
        bool open(const std::string& device, int baud = 115200, int read_timeout_ms = 20,
                  bool quiet = false);
        void close();
        bool isOpen() const { return fd_ >= 0; }
        const std::string& device() const { return device_; }

        /// @brief 尽力读满 size 字节。返回实际读到的字节数；0 = 超时/无数据，
        ///        负数 = 设备错误（上层据此重连）。
        int read(uint8_t* buffer, std::size_t size);

        /// @brief 把缓冲区清空（重连后丢弃残留的半帧）。
        void flushInput();

        bool write(const uint8_t* buffer, std::size_t size);

    private:
        int fd_ = -1;
        std::string device_;
        int read_timeout_ms_ = 20;
    };
} // namespace io

#endif // ULTRA_VISION_IO_SERIAL_PORT_HPP
