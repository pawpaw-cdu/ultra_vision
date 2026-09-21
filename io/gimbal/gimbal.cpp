#include "gimbal.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>

namespace io
{
namespace
{
    constexpr std::size_t kMaxImuSamples = 400;   // ≈2 s @ 200 Hz，够按帧时间戳插值

    /// @brief "auto"：按 USB CDC → USB 转串口的常见顺序找第一个能打开的设备。
    ///        找不到返回空串（调用方按 require_connection 决定报错还是降级）。
    std::string resolveDevice(const std::string& requested, int baud, int timeout_ms,
                              SerialPort* opened)
    {
        if (requested != "auto") {
            if (opened->open(requested, baud, timeout_ms)) return requested;
            return {};
        }
        // auto：逐个候选试开，**静默**失败（一次找不到设备不该刷一屏错误）。
        static const char* const candidates[] = {
            "/dev/gimbal",     // udev 规则固定名字（推荐）
            "/dev/ttyACM0", "/dev/ttyACM1", "/dev/ttyACM2",   // STM32 USB CDC
            "/dev/ttyUSB0", "/dev/ttyUSB1", "/dev/ttyUSB2",   // CP210x / CH340
        };
        for (const char* candidate : candidates) {
            if (opened->open(candidate, baud, timeout_ms, /*quiet=*/true)) return candidate;
        }
        return {};
    }

    void normalizeQuaternion(ImuSample& sample)
    {
        const double norm = std::sqrt(sample.w * sample.w + sample.x * sample.x +
                                      sample.y * sample.y + sample.z * sample.z);
        if (norm < 1e-9) {
            sample = ImuSample{};
            return;
        }
        sample.w /= norm;
        sample.x /= norm;
        sample.y /= norm;
        sample.z /= norm;
    }

    /// @brief 两姿态之间按比例插值（球面线性插值；夹角很小时退化成线性）。
    ImuSample slerp(const ImuSample& a, const ImuSample& b, double ratio)
    {
        double dot = a.w * b.w + a.x * b.x + a.y * b.y + a.z * b.z;
        double bx = b.x;
        double by = b.y;
        double bz = b.z;
        double bw = b.w;
        if (dot < 0.0) {
            dot = -dot;
            bx = -bx;
            by = -by;
            bz = -bz;
            bw = -bw;
        }
        ImuSample out;
        if (dot > 0.9995) {
            out.w = a.w + ratio * (bw - a.w);
            out.x = a.x + ratio * (bx - a.x);
            out.y = a.y + ratio * (by - a.y);
            out.z = a.z + ratio * (bz - a.z);
        } else {
            const double theta = std::acos(std::min(1.0, std::max(-1.0, dot)));
            const double sin_theta = std::sin(theta);
            const double wa = std::sin((1.0 - ratio) * theta) / sin_theta;
            const double wb = std::sin(ratio * theta) / sin_theta;
            out.w = wa * a.w + wb * bw;
            out.x = wa * a.x + wb * bx;
            out.y = wa * a.y + wb * by;
            out.z = wa * a.z + wb * bz;
        }
        normalizeQuaternion(out);
        // yaw/pitch 线性插值（同一包里带回来的云台绝对角）。
        out.yaw = a.yaw + ratio * (b.yaw - a.yaw);
        out.pitch = a.pitch + ratio * (b.pitch - a.pitch);
        return out;
    }
} // namespace

    Gimbal::Gimbal(const SerialConfig& config) : config_(config)
    {
        if (!config_.device.empty()) {
            const std::string opened = resolveDevice(
                config_.device, config_.baud, config_.read_timeout_ms, &serial_);
            if (opened.empty()) {
                std::cerr << "[gimbal] 打不开串口 " << config_.device
                          << "：下位机链路不可用" << std::endl;
                if (config_.require_connection) {
                    throw std::runtime_error("[gimbal] serial is required but cannot be opened");
                }
            } else {
                config_.device = opened;   // 记下实际用的设备（auto 解析结果）
                connected_.store(true);
            }
        } else {
            std::cout << "[gimbal] serial.device 为空：只跑识别，不下发云台指令" << std::endl;
        }
        stats_window_start_ = std::chrono::steady_clock::now();
        if (serial_.isOpen()) {
            thread_ = std::thread(&Gimbal::readLoop, this);
        }
    }

    Gimbal::~Gimbal()
    {
        quit_.store(true);
        if (thread_.joinable()) thread_.join();
        serial_.close();
    }

    GimbalMode Gimbal::mode() const
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        return mode_;
    }

    GimbalState Gimbal::state() const
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        return state_;
    }

    bool Gimbal::latestImu(ImuSample& out) const
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        if (imu_queue_.empty()) return false;
        out = imu_queue_.back();
        return true;
    }

    bool Gimbal::imuAt(std::chrono::steady_clock::time_point timestamp, ImuSample& out) const
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        if (imu_queue_.empty()) return false;
        if (timestamp <= imu_queue_.front().timestamp) {
            out = imu_queue_.front();
            return true;
        }
        if (timestamp >= imu_queue_.back().timestamp) {
            out = imu_queue_.back();
            return true;
        }
        // 队列很短（几百个），线性扫足够；找第一个不早于 timestamp 的样本。
        std::size_t index = 1;
        while (index < imu_queue_.size() &&
               imu_queue_[index].timestamp < timestamp) {
            ++index;
        }
        const ImuSample& before = imu_queue_[index - 1];
        const ImuSample& after = imu_queue_[index];
        const double span = std::chrono::duration<double>(after.timestamp - before.timestamp).count();
        if (!(span > 1e-9)) {
            out = after;
            return true;
        }
        const double ratio =
            std::chrono::duration<double>(timestamp - before.timestamp).count() / span;
        out = slerp(before, after, std::min(1.0, std::max(0.0, ratio)));
        return true;
    }

    bool Gimbal::send(const VisionToGimbal& frame)
    {
        VisionToGimbal packet = frame;
        finalizeFrame(packet);
        if (!serial_.isOpen()) return false;
        const bool ok = serial_.write(reinterpret_cast<const uint8_t*>(&packet), sizeof(packet));
        if (ok) {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            ++stats_.sent_frames;
        }
        return ok;
    }

    bool Gimbal::send(bool control, bool fire, double yaw, double yaw_vel, double yaw_acc,
                      double pitch, double pitch_vel, double pitch_acc)
    {
        VisionToGimbal packet;
        packet.mode = control ? (fire ? 2 : 1) : 0;
        packet.yaw = static_cast<float>(yaw);
        packet.yaw_vel = static_cast<float>(yaw_vel);
        packet.yaw_acc = static_cast<float>(yaw_acc);
        packet.pitch = static_cast<float>(pitch);
        packet.pitch_vel = static_cast<float>(pitch_vel);
        packet.pitch_acc = static_cast<float>(pitch_acc);
        return send(packet);
    }

    Gimbal::Stats Gimbal::stats() const
    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        return stats_;
    }

    bool Gimbal::reconnect()
    {
        if (config_.device.empty()) return false;
        {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            ++stats_.reconnects;
        }
        connected_.store(false);
        serial_.close();
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        if (!serial_.open(config_.device, config_.baud, config_.read_timeout_ms)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            return false;
        }
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            imu_queue_.clear();
            state_ = GimbalState{};
            mode_ = GimbalMode::IDLE;
        }
        connected_.store(true);
        error_count_ = 0;
        std::cout << "[gimbal] 串口重连成功：" << config_.device << std::endl;
        return true;
    }

    void Gimbal::readLoop()
    {
        std::cout << "[gimbal] read thread started (" << config_.device << ")" << std::endl;
        uint8_t head[2] = {0, 0};
        while (!quit_.load()) {
            if (error_count_ >= config_.reconnect_after_errors) {
                error_count_ = 0;
                std::cerr << "[gimbal] 读错误过多，尝试重连" << std::endl;
                if (!reconnect()) continue;
            }

            // 先对齐帧头，再读定长剩余部分：丢掉噪声字节而不是错位解析。
            const int head_len = serial_.read(head, sizeof(head));
            if (head_len <= 0) {
                ++error_count_;
                std::lock_guard<std::mutex> lock(stats_mutex_);
                ++stats_.timeouts;
                continue;
            }
            if (head_len < static_cast<int>(sizeof(head)) || head[0] != 'S' || head[1] != 'P') {
                if (head_len == 1 && head[0] == 'S') {
                    // 半个头，留着下一次再拼（把 'S' 放回去）
                    uint8_t pair[2] = {'S', 0};
                    const int rest = serial_.read(pair + 1, 1);
                    if (rest == 1 && pair[1] == 'P') {
                        head[0] = 'S';
                        head[1] = 'P';
                    } else {
                        ++error_count_;
                        continue;
                    }
                } else {
                    ++error_count_;
                    continue;
                }
            }

            GimbalToVision frame;
            uint8_t* raw = reinterpret_cast<uint8_t*>(&frame);
            raw[0] = head[0];
            raw[1] = head[1];
            const int rest_len = serial_.read(raw + sizeof(head), sizeof(frame) - sizeof(head));
            if (rest_len < static_cast<int>(sizeof(frame) - sizeof(head))) {
                ++error_count_;
                std::lock_guard<std::mutex> lock(stats_mutex_);
                ++stats_.timeouts;
                continue;
            }

            if (!validFrame(raw, sizeof(frame))) {
                std::lock_guard<std::mutex> lock(stats_mutex_);
                ++stats_.crc_errors;
                continue;
            }
            error_count_ = 0;
            const auto received_at = std::chrono::steady_clock::now();

            ImuSample sample;
            sample.timestamp = received_at;
            sample.w = frame.q[0];
            sample.x = frame.q[1];
            sample.y = frame.q[2];
            sample.z = frame.q[3];
            sample.yaw = frame.yaw;
            sample.pitch = frame.pitch;
            normalizeQuaternion(sample);

            {
                std::lock_guard<std::mutex> lock(state_mutex_);
                state_.yaw = frame.yaw;
                state_.yaw_vel = frame.yaw_vel;
                state_.pitch = frame.pitch;
                state_.pitch_vel = frame.pitch_vel;
                state_.bullet_speed = frame.bullet_speed;
                state_.bullet_count = frame.bullet_count;
                state_.valid = true;
                switch (frame.mode) {
                case 0: mode_ = GimbalMode::IDLE; break;
                case 1: mode_ = GimbalMode::AUTO_AIM; break;
                case 2: mode_ = GimbalMode::SMALL_BUFF; break;
                case 3: mode_ = GimbalMode::BIG_BUFF; break;
                default: mode_ = GimbalMode::IDLE; break;
                }
                imu_queue_.push_back(sample);
                while (imu_queue_.size() > kMaxImuSamples) imu_queue_.pop_front();
            }

            std::lock_guard<std::mutex> lock(stats_mutex_);
            ++stats_.frames;
            ++stats_window_frames_;
            const double window = std::chrono::duration<double>(
                received_at - stats_window_start_).count();
            if (window >= 1.0) {
                stats_.fps = stats_window_frames_ / window;
                stats_window_frames_ = 0;
                stats_window_start_ = received_at;
            }
        }
        std::cout << "[gimbal] read thread stopped" << std::endl;
    }
} // namespace io
