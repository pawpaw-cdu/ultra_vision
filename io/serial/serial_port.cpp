#include "serial_port.hpp"

#include <cerrno>
#include <cstring>
#include <iostream>

#include <fcntl.h>
#include <poll.h>
#include <termios.h>
#include <unistd.h>

namespace
{
    speed_t baudToConstant(int baud)
    {
        switch (baud) {
        case 9600: return B9600;
        case 19200: return B19200;
        case 38400: return B38400;
        case 57600: return B57600;
        case 115200: return B115200;
        case 230400: return B230400;
#ifdef B460800
        case 460800: return B460800;
#endif
#ifdef B921600
        case 921600: return B921600;
#endif
        default: return B115200;
        }
    }
} // namespace

namespace io
{
    SerialPort::~SerialPort()
    {
        close();
    }

    bool SerialPort::open(const std::string& device, int baud, int read_timeout_ms, bool quiet)
    {
        close();
        device_ = device;
        read_timeout_ms_ = read_timeout_ms > 0 ? read_timeout_ms : 20;

        // O_NONBLOCK：没数据时 read 立刻返回，绝不把视觉主循环卡住；
        // O_NOCTTY：不把串口当控制终端，避免 Ctrl+C 之类被它吃掉。
        fd_ = ::open(device.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK);
        if (fd_ < 0) {
            if (!quiet) {
                std::cerr << "[serial] open " << device << " failed: " << std::strerror(errno)
                          << std::endl;
            }
            return false;
        }

        termios tty{};
        if (::tcgetattr(fd_, &tty) != 0) {
            if (!quiet) std::cerr << "[serial] tcgetattr failed: " << std::strerror(errno) << std::endl;
            close();
            return false;
        }

        cfmakeraw(&tty);                       // 8N1、无回显、无流控、无特殊字符
        tty.c_cflag |= (CLOCAL | CREAD);
        tty.c_cflag &= ~CRTSCTS;
        tty.c_cc[VMIN] = 0;
        tty.c_cc[VTIME] = 0;                   // 超时交给 poll()
        const speed_t speed = baudToConstant(baud);
        ::cfsetispeed(&tty, speed);
        ::cfsetospeed(&tty, speed);
        if (::tcsetattr(fd_, TCSANOW, &tty) != 0) {
            if (!quiet) std::cerr << "[serial] tcsetattr failed: " << std::strerror(errno) << std::endl;
            close();
            return false;
        }
        ::tcflush(fd_, TCIOFLUSH);
        std::cout << "[serial] opened " << device << " @" << baud << std::endl;
        return true;
    }

    void SerialPort::close()
    {
        if (fd_ >= 0) {
            ::close(fd_);
            fd_ = -1;
        }
    }

    int SerialPort::read(uint8_t* buffer, std::size_t size)
    {
        if (fd_ < 0 || buffer == nullptr || size == 0) return -1;
        std::size_t filled = 0;
        while (filled < size) {
            pollfd descriptor{};
            descriptor.fd = fd_;
            descriptor.events = POLLIN;
            const int ready = ::poll(&descriptor, 1, read_timeout_ms_);
            if (ready < 0) {
                if (errno == EINTR) continue;
                return -1;
            }
            if (ready == 0) break;             // 超时：返回已收到的部分
            if ((descriptor.revents & (POLLERR | POLLHUP | POLLNVAL)) != 0) return -1;

            const ssize_t got = ::read(fd_, buffer + filled, size - filled);
            if (got > 0) {
                filled += static_cast<std::size_t>(got);
                continue;
            }
            if (got == 0) break;
            if (errno == EAGAIN || errno == EWOULDBLOCK) continue;
            if (errno == EINTR) continue;
            return -1;
        }
        return static_cast<int>(filled);
    }

    void SerialPort::flushInput()
    {
        if (fd_ >= 0) ::tcflush(fd_, TCIFLUSH);
    }

    bool SerialPort::write(const uint8_t* buffer, std::size_t size)
    {
        if (fd_ < 0 || buffer == nullptr) return false;
        std::size_t sent = 0;
        while (sent < size) {
            pollfd descriptor{};
            descriptor.fd = fd_;
            descriptor.events = POLLOUT;
            const int ready = ::poll(&descriptor, 1, read_timeout_ms_);
            if (ready < 0) {
                if (errno == EINTR) continue;
                return false;
            }
            if (ready == 0) return false;      // 发不出去就丢弃这一帧（下位机有超时保护）

            const ssize_t put = ::write(fd_, buffer + sent, size - sent);
            if (put > 0) {
                sent += static_cast<std::size_t>(put);
                continue;
            }
            if (put < 0 && (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR)) continue;
            return false;
        }
        return true;
    }
} // namespace io
