#ifndef ULTRA_VISION_IO_GIMBAL_HPP
#define ULTRA_VISION_IO_GIMBAL_HPP

// 下位机链路（串口 + 协议 + 后台读线程）。
//
// 职责边界（和 sp_vision 的 io::Gimbal 对齐）：
//   · 收：把 43 字节帧解析成 GimbalState（云台绝对角、角速度、弹速、弹数）+ 四元数队列；
//   · 发：把"要打到的绝对角度 + 前馈角速度/角加速度 + mode(控制/开火)"打包成 29 字节帧。
//   · 不做任何估计/决策 —— 那些在 Kalman/control 层，这里只搬字节。
//
// 读线程永不阻塞主循环：串口非阻塞 + poll 超时，收不满一帧就下一轮；
// 连续出错超过阈值会尝试重连（USB CDC 掉线后 /dev/gimbal 可能重新枚举）。

#include <atomic>
#include <chrono>
#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <thread>

#include "io/gimbal/protocol.hpp"
#include "io/serial/serial_port.hpp"

namespace io
{
    struct SerialConfig
    {
        std::string device = "/dev/gimbal";   // 空 = 不启用串口（只跑识别）
        int baud = 115200;                    // USB CDC 下无意义
        int read_timeout_ms = 20;
        int reconnect_after_errors = 3000;    // 累计读错误到该值尝试重连
        bool require_connection = false;      // true = 打不开就退出（比赛模式）
        // ---- 下发命令的坐标系（2026-09-29 实测量定）-------------------------
        // 结论（world 回传两次 + 台阶测试）：
        //   C 板 yaw/pitch 闭环比较的**不是**回传的电机角，而是**IMU 世界系角**：
        //     cmd_yaw   = +az_x   （基轴 x 在世界系的方位角）
        //     cmd_pitch = -el_x   （基轴 x 的俯仰角取反）
        //   证据：把这两个"当前值"发回去 → 云台保持不动（0.2~0.9°）；发电机系当前值
        //   → 一次跑 35.7°。台阶：cmd_yaw+10 → 电机 yaw+9.18（同号 1:1）、
        //   cmd_pitch+10 → 电机 pitch−9.45（反号 1:1）。
        //   这条也正是 sp_vision 的做法：它下发的就是世界系绝对角，不需要任何符号处理。
        //
        // 下发形式（**不需要标定任何零点常数**，因为增量加在实测基准上）：
        //   cmd_yaw   = world_yaw_sign·az_x + gain_yaw_sign  / yaw_sign · Δ(内部 yaw)
        //   cmd_pitch = world_pitch_sign·el_x + gain_pitch_sign      · Δ(内部 pitch)
        // 其中 Δ(内部) = 本次要发的角 − 当前姿态角（同一时刻）。gain_* 是"命令帧相对
        // 回传电机系"的增益符号，除以 yaw_sign 是因为内部 yaw = yaw_sign·回传 yaw。
        enum class CommandFrame
        {
            Motor = 0,   // 直接发内部电机角（旧行为；这台车上会跑飞，仅留作对照）
            World = 1,   // 用四元数算世界系角（实测正确）
        };
        CommandFrame command_frame = CommandFrame::World;
        double command_world_yaw_sign = 1.0;    // yaw 通道符号（实测 +1）
        double command_world_pitch_sign = -1.0; // pitch 通道符号（实测 -1）
        /// @brief `command_frame: motor` 时用的换算符号：把"内部角"换算回下位机的回传角。
        ///        本车内部 yaw = yaw_sign·回传 yaw、内部 pitch = 回传 pitch，所以
        ///        command_yaw_sign = yaw_sign（= -1）、command_pitch_sign = +1。
        double command_yaw_sign = -1.0;
        double command_pitch_sign = 1.0;
        double yaw_sign = 1.0;                  // 回传 yaw 符号（和 serial.yaml 的 yaw_sign 一致）
    };

    /// @brief 四元数样本（wxyz），用于按时间戳插值出"某一帧曝光时刻的 IMU 姿态"。
    struct ImuSample
    {
        std::chrono::steady_clock::time_point timestamp;
        double w = 1.0;
        double x = 0.0;
        double y = 0.0;
        double z = 0.0;
        // 同一包里回传的云台绝对角（rad）。带时间戳一起存，才能按帧时刻拿到"那一刻"的
        // yaw/pitch，而不是"处理完这一帧之后"的最新值。
        double yaw = 0.0;
        double pitch = 0.0;
    };

    class Gimbal
    {
    public:
        explicit Gimbal(const SerialConfig& config);
        ~Gimbal();

        Gimbal(const Gimbal&) = delete;
        Gimbal& operator=(const Gimbal&) = delete;

        bool connected() const { return connected_.load(); }
        GimbalMode mode() const;
        GimbalState state() const;

        /// @brief 最近一帧 IMU 四元数（没有就返回 false）。
        bool latestImu(ImuSample& out) const;
        /// @brief 按**单调时钟**时间戳插值出 IMU 姿态（线性 slerp；超出队列范围时取最近端）。
        bool imuAt(std::chrono::steady_clock::time_point timestamp, ImuSample& out) const;

        /// @brief 下发云台指令。`control=false` 时 mode=0（下位机接管，我们不控制）。注意下位机
        ///        只有收到 mode=2 才会**开火**，所以 fire 决策完全在我们这边。
        bool send(bool control, bool fire, double yaw, double yaw_vel, double yaw_acc,
                  double pitch, double pitch_vel, double pitch_acc);
        bool send(const VisionToGimbal& frame);

        struct Stats
        {
            uint64_t frames = 0;        // 收到的合法帧
            uint64_t sent_frames = 0;   // 我们发出去的帧（控制线程 100 Hz）
            uint64_t crc_errors = 0;
            uint64_t timeouts = 0;
            uint64_t reconnects = 0;
            double fps = 0.0;           // 最近一秒的收帧率
        };
        Stats stats() const;

    private:
        void readLoop();
        bool reconnect();

        SerialConfig config_;
        SerialPort serial_;

        std::thread thread_;
        std::atomic<bool> quit_{false};
        std::atomic<bool> connected_{false};

        mutable std::mutex state_mutex_;
        GimbalState state_;
        GimbalMode mode_ = GimbalMode::IDLE;
        std::deque<ImuSample> imu_queue_;      // 按时间升序，长度有上限

        mutable std::mutex stats_mutex_;
        Stats stats_;
        std::chrono::steady_clock::time_point stats_window_start_;
        uint64_t stats_window_frames_ = 0;
        int error_count_ = 0;
    };
} // namespace io

#endif // ULTRA_VISION_IO_GIMBAL_HPP
