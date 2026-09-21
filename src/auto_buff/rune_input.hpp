#ifndef AUTO_AIM_ENERGY_RUNE_INPUT_HPP
#define AUTO_AIM_ENERGY_RUNE_INPUT_HPP

// 输入层：把"帧从哪来、遥测从哪来、指令往哪发"封在一个类里。
// 两个来源：仿真器（7666 帧 / 7667 指令 / 7668 遥测）与视频回放（只有帧）。
//
// 这一层只负责 I/O，不含任何能量机关语义；上层（rune_runner）拿到 Frame 与
// 遥测行后自己解析。这样：换数据源（回放/实时/离线文件）不需要动算法；
// 算法单测也不需要起仿真器。

#include <cstdint>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "auto_buff/rune_config.hpp"

namespace sim_receiver
{
    class VisionDateRecevier;   // 注意：他们的类名是 Recevier（原文拼写）
    class TelemetryReceiver;
}

namespace auto_aim::energy
{
    class RuneInputSource
    {
    public:
        struct Frame
        {
            cv::Mat bgr;
            uint64_t local_timestamp_us = 0;   // 本地到达时刻
            uint64_t source_timestamp_us = 0;  // 仿真曝光时刻（视频模式为 0）
            /// 仿真帧序号（视频模式为 0）。相邻两帧序号之差 - 1 = 我们**跳过的帧数**：
            /// 这是判断"是仿真给得慢，还是我们处理不过来"的唯一硬指标。
            uint64_t sequence = 0;
        };

        RuneInputSource();
        ~RuneInputSource();
        RuneInputSource(const RuneInputSource&) = delete;
        RuneInputSource& operator=(const RuneInputSource&) = delete;

        /// @brief 打开输入：video_path 非空走视频，否则连仿真器。
        /// @return 失败返回 false（配置/视频/连接问题）。
        bool open(const RuneConfig& config, const std::string& config_dir,
                  const std::string& video_path);

        bool videoMode() const { return video_mode_; }
        bool valid() const { return video_mode_ ? capture_.isOpened() : receiver_ != nullptr; }

        /// @brief 取一帧；返回 false 表示流结束或读失败（仿真器没新帧时返回 false）。
        bool readFrame(Frame& frame);

        /// @brief 取回上次调用之后收到的遥测行（视频模式恒为空）。
        std::vector<std::string> pollTelemetry();

        bool sendGimbal(double yaw, double pitch);
        bool sendFire();
        bool sendReset();

        /// @brief 录制帧用的目录（ULTRA_VISION_RUNE_FRAME_DIR）；为空表示不录。
        const std::string& frameDir() const { return frame_dir_; }
        /// @brief 遥测真值写盘流（ULTRA_VISION_RUNE_GT_CSV）；未打开则为空。
        std::ofstream* groundTruth() { return ground_truth_.is_open() ? &ground_truth_ : nullptr; }
        std::ofstream* labelDump() { return label_dump_.is_open() ? &label_dump_ : nullptr; }

    private:
        bool video_mode_ = false;
        cv::VideoCapture capture_;
        std::unique_ptr<sim_receiver::VisionDateRecevier> receiver_;
        std::unique_ptr<sim_receiver::TelemetryReceiver> telemetry_;
        std::ofstream ground_truth_;
        std::ofstream label_dump_;
        std::string frame_dir_;
    };
} // namespace auto_aim::energy

#endif // AUTO_AIM_ENERGY_RUNE_INPUT_HPP
