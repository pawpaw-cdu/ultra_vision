#ifndef AUTO_AIM_APP_SIM_DETAIL_HPP
#define AUTO_AIM_APP_SIM_DETAIL_HPP

// 组合层：仿真入口用到的两个纯函数（仿真相机内参重建、开火状态名）。
// 它们被 node_sim.cpp 与 app/sim_runner.cpp 两个 TU 引用，所以放在 header 里用
// `inline`（等价于原来匿名命名空间避免 duplicate symbol 的作用，但没有"未使用"
// 警告）。调用方用 `using namespace auto_aim::app::sim_detail;` 引入，调用点零改动。

#include <cmath>
#include <string>

#include <opencv2/opencv.hpp>

#include "control/shooter.hpp"

namespace auto_aim
{
    namespace app
    {
        namespace sim_detail
        {
            /// @brief 由"行数 + 垂直 FOV"重建针孔相机内参（仿真分辨率固定时用）。
            inline cv::Mat makeCameraMatrix(int rows, int columns, double vertical_fov_degrees)
            {
                const double focal_length = 0.5 * rows /
                    std::tan(vertical_fov_degrees * CV_PI / 360.0);
                return (cv::Mat_<double>(3, 3) <<
                    focal_length, 0.0, 0.5 * columns,
                    0.0, focal_length, 0.5 * rows,
                    0.0, 0.0, 1.0);
            }

            inline const char* shooterStateName(ShooterState state)
            {
                switch (state) {
                case ShooterState::IDLE: return "IDLE";
                case ShooterState::READY: return "READY";
                case ShooterState::FIRING: return "FIRING";
                case ShooterState::ERROR: return "ERROR";
                case ShooterState::END: return "END";
                }
                return "UNKNOWN";
            }

            inline const char* shooterErrorName(ShooterErrorReason reason)
            {
                switch (reason) {
                case ShooterErrorReason::NONE: return "NONE";
                case ShooterErrorReason::OUT_OF_FIRE_WINDOW: return "OUT_OF_FIRE_WINDOW";
                case ShooterErrorReason::GIMBAL_ERROR: return "GIMBAL_ERROR";
                case ShooterErrorReason::NO_TARGET: return "NO_TARGET";
                case ShooterErrorReason::BALLISTICS_INVALID: return "BALLISTICS_INVALID";
                case ShooterErrorReason::ALREADY_HIT: return "ALREADY_HIT";
                case ShooterErrorReason::COOLDOWN: return "COOLDOWN";
                case ShooterErrorReason::RATE_TOO_HIGH: return "RATE_TOO_HIGH";
                case ShooterErrorReason::OUT_OF_RANGE: return "OUT_OF_RANGE";
                case ShooterErrorReason::NOT_SETTLED: return "NOT_SETTLED";
                }
                return "UNKNOWN";
            }

            inline std::string shooterStatusName(ShooterState state,
                                                 ShooterErrorReason reason)
            {
                std::string status = shooterStateName(state);
                if (state == ShooterState::ERROR) {
                    status += "(" + std::string(shooterErrorName(reason)) + ")";
                }
                return status;
            }
        } // namespace sim_detail
    } // namespace app
} // namespace auto_aim

#endif // AUTO_AIM_APP_SIM_DETAIL_HPP
