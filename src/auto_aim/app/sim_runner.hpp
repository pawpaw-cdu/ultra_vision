#ifndef AUTO_AIM_APP_SIM_RUNNER_HPP
#define AUTO_AIM_APP_SIM_RUNNER_HPP

// 组合层：仿真链路的"装配 + 每帧算法阶段 + 仿真专属 IO"。
//
// 分层对齐 sp_vision 的 `tasks/ 能力模块 + src/*.cpp 薄组合根`：
//   · 能力层（detector / armor_source / aim_pipeline / tracker ...）负责算法；
//   · 这里（app/sim_runner）负责把仿真专属的东西装起来 —— 配置、TCP 帧源、
//     曝光→到手的时间基、三条 CSV、曲线 —— 并把一帧拆成命名清晰的几个阶段：
//         fetchFrame → prepareFrame → runAim
//   · 入口 node_sim.cpp 只留"构造 + 一段可读的主循环 + 显示/上报"，是薄组合根。
//
// 主循环里"检测 → 记录 → 瞄准"的数据流由入口按顺序调用上面四个阶段表达；
// 显示、每秒上报、曲线是 IO，入口直接用下面暴露的组件/结果访问器完成。

#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "common/types.hpp"
#include "control/aim_pipeline.hpp"
#include "control/gimbal_controller.hpp"
#include "control/shooter.hpp"
#include "control/target_selector.hpp"
#include "perception/armor_source.hpp"
#include "sim_receiver/vision_date_receiver.hpp"

namespace auto_aim
{
    namespace app
    {
        class SimRunner
        {
        public:
            explicit SimRunner(std::string config_dir);

            /// @brief 读配置、连仿真器、装配检测前端/瞄准流水线/CSV/曲线。
            /// @return false 表示配置或链路无法建立（当前实现恒为 true）。
            bool initialize();

            // ---- 每帧算法阶段（主循环按顺序调用）------------------------------

            /// @brief 取一帧；含空帧计数、"曝光→到手"延迟采样、进帧速率统计。
            /// @return false 表示这一帧为空，调用方 `continue` 等下一帧。
            bool fetchFrame();

            /// @brief 时间基（指令快照）+ 处理尺寸缩放 + 内参 + 检测（含动态 ROI）。
            void prepareFrame();

            /// @brief 估计 → 选板 → 瞄准 → 开火（共用 AimPipeline）。
            void runAim();

            // ---- 入口需要的组件 / 结果 / 循环控制 ----------------------------
            sim_receiver::VisionDateReceiver& receiver() { return *receiver_; }
            ArmorSource& armorSource() { return *armor_source_; }
            AimPipeline& pipeline() { return *pipeline_; }
            Tracker& tracker() { return pipeline_->tracker(); }
            cv::Mat& image() { return image_; }
            const cv::Mat& cameraMatrix() const { return camera_matrix_; }
            const cv::Mat& distCoeffs() const { return dist_coeffs_; }

            const TargetDecision& lastDecision() const { return last_decision_; }
            GimbalControllerSnapshot lastGimbalCommand() const { return last_gimbal_command_; }
            std::string shooterStatus() const;
            int lightCount() const { return light_count_; }
            int armorCount() const { return armor_count_; }
            int pnpCount() const { return pnp_count_; }
            bool showDisplay() const { return show_display_; }
            bool testSecondsExpired() const;
            bool exitRequested() const;
            int key() const { return key_; }
            void setKey(int key) { key_ = key; }

        private:
            std::string config_dir_;

            std::unique_ptr<sim_receiver::VisionDateReceiver> receiver_;
            std::unique_ptr<ArmorSource> armor_source_;
            std::unique_ptr<AimPipeline> pipeline_;
            std::vector<double> source_delay_samples_us_;
            double observation_source_delay_ = 0.0;

            // 配置派生（initialize 里赋值）
            bool use_simulator_camera_ = false;
            double simulator_fov_degrees_ = 45.0;
            int max_processing_width_ = 960;
            bool fire_enabled_ = true;
            bool show_display_ = true;
            bool source_delay_enabled_ = true;
            int missing_frames_ = 0;
            double test_seconds_ = 0.0;
            std::chrono::steady_clock::time_point loop_start_time_{};

            // 几何 / 逐帧数据
            cv::Mat camera_matrix_;
            cv::Mat dist_coeffs_;
            cv::Mat frame_;
            cv::Mat image_;
            int processing_height_ = 0;
            std::uint64_t frame_sequence_ = 0;
            std::uint64_t frame_local_timestamp_us_ = 0;
            std::uint64_t frame_source_timestamp_us_ = 0;
            float frame_gimbal_yaw_ = 0.0f;
            float frame_gimbal_pitch_ = 0.0f;
            ArmorSourceResult source_result_;
            int light_count_ = 0;
            int armor_count_ = 0;
            int pnp_count_ = 0;

            // 跨帧保持的控制状态
            int previous_armor_id_ = -1;
            bool aim_ready_ = false;
            double aim_yaw_error_ = 0.0;
            double aim_pitch_error_ = 0.0;
            double commanded_yaw_ = 0.0;
            double commanded_pitch_ = 0.0;
            double last_detection_time_ = 0.0;
            int key_ = -1;
            GimbalControllerSnapshot last_gimbal_command_{};
            TargetDecision last_decision_{};
            ShooterState shooter_state_ = ShooterState::IDLE;
            ShooterErrorReason shooter_error_ = ShooterErrorReason::NONE;
        };
    } // namespace app
} // namespace auto_aim

#endif // AUTO_AIM_APP_SIM_RUNNER_HPP
