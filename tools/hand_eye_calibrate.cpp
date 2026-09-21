// 手眼标定：**相机 → 云台** 外参（R_camera2gimbal / t_camera2gimbal）。
// 对齐 sp_vision 的 calibration/calibrate_handeye.cpp：静态标定物 + 多组云台姿态，
// 用 robot-world/hand-eye 法解出来（OpenCV cv::calibrateRobotWorldHandEye）。
//
// 为什么需要：相机装在云台上，光心相对云台旋转中心有平移和偏转。不补这个外参时
//   · 距离越远、偏角越大，瞄准点系统性偏（偏 1° 在 7 m 处就是 12 cm）；
//   · 底盘/云台转动时，观测反旋到世界系也会带着这个偏差一起转。
// 仿真里相机就在云台光心，所以外参是单位阵、零平移，标定结果应当接近单位阵（可用来验流程）。
//
// 两种用法：
//   ① **现场实时标定（推荐）**：一个进程里直接开相机 + 串口，
//      空格采一组 → 立刻解算并显示残差/外参；s 存 yaml；u 撤销；q 退出。
//        ./hand_eye_calibrate --live [--config-dir configs] [--out hand_eye.yaml]
//   ② 离线复算：读一个 {i}.jpg + {i}.yaml（云台 yaw/pitch）的文件夹，用于复盘/CI。
//        ./hand_eye_calibrate <数据文件夹> [--config-dir configs] [--out hand_eye.yaml]
//   自检（不需要硬件）：  ./hand_eye_calibrate --selftest
//
// 标定物默认用**黑白标定板**（棋盘格/圆点阵，见 configs/calibration.yaml）：
// 几十个角点 + 亚像素定位，位姿精度比拿装甲板当标定物高一个量级（板子位姿的误差会
// 1:1 变成外参误差）。`--target armor` 可以退回用装甲板（手边没标定板时的应急）。
// 内参要先标好（tools/calibrate_camera），否则手眼会把内参误差一起吸进去。

#include <array>
#include <limits>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/core/ocl.hpp>
#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>

#include "perception/armor_source.hpp"
#include "config_loader.hpp"
#include "io/gimbal/gimbal.hpp"
#include "visualization/projection.hpp"
#if defined(ULTRA_VISION_USE_HIK_CAMERA)
#include "io/camera/HikCamera.hpp"
using CameraType = rm_ultra::HikCamera;
#elif defined(ULTRA_VISION_USE_GALAXY_CAMERA)
#include "io/camera/GalaxyCamera.hpp"
using CameraType = rm_ultra::GalaxyCamera;
#endif

namespace
{
    /// 帧到手时刻（单调时钟）。取姿态时用它，而不是"处理完这帧之后"的时间。
    using FrameTime = std::chrono::steady_clock::time_point;

#if defined(ULTRA_VISION_USE_HIK_CAMERA) || defined(ULTRA_VISION_USE_GALAXY_CAMERA)
    /// @brief 带时间戳取图。海康驱动有该重载（时间戳在采集线程取图返回时打）；
    ///        其它驱动没有，就退化成"取图后就地打戳"（仍比处理完再打准）。
    bool getImageTimed(CameraType& camera, cv::Mat& image, FrameTime& timestamp, int timeout_ms)
    {
#if defined(ULTRA_VISION_USE_HIK_CAMERA)
        return camera.getImage(image, timestamp, timeout_ms);
#else
        if (!camera.getImage(image, timeout_ms)) return false;
        timestamp = std::chrono::steady_clock::now();
        return true;
#endif
    }
#endif

    struct Sample
    {
        double yaw = 0.0;      // 云台绝对 yaw（rad，C 板回传）
        double pitch = 0.0;    // 云台绝对 pitch
        cv::Mat rvec;          // 标定物在**相机系**的姿态（PnP）
        cv::Mat tvec;          // 标定物在**相机系**的位置
        // 该帧**时刻**的云台姿态（世界←云台），由帧时间戳 + IMU 四元数插值而来。
        // 有它就用它——直接避开"处理完这帧才读姿态"造成的 10~40 ms 错配；
        // 没有（离线回放/只存了 yaw pitch 的旧数据）就退回 gimbalToBase(yaw,pitch)。
        cv::Matx33d R_wb;
        bool has_R_wb = false;
        double time_s = 0.0;   // 采样时刻（会话起点为 0），用来扣除 yaw 的线性漂移
    };

    /// @brief 云台→底盘（base）的旋转：与 tracker 的 cameraToBase 同约定（Y 后 X）。
    cv::Matx33d gimbalToBase(double yaw, double pitch)
    {
        const double cy = std::cos(yaw), sy = std::sin(yaw);
        const double cp = std::cos(pitch), sp = std::sin(pitch);
        // R = Ry(yaw) * Rx(pitch)
        return cv::Matx33d(cy, sy * sp, sy * cp,
                           0.0, cp, -sp,
                           -sy, cy * sp, cy * cp);
    }

#if defined(ULTRA_VISION_USE_HIK_CAMERA) || defined(ULTRA_VISION_USE_GALAXY_CAMERA)
    /// @brief 量 yaw 的漂移率（°/s）。C 板回传的 yaw 是陀螺积分量，静止时自己会爬
    ///        （这台机器实测 ~0.2°/s），pitch 有重力基准所以不漂。云台静止时量一段时间的
    ///        斜率就是零偏；顺便看 pitch 有没有动，用来判断这段里云台是不是真的没被碰。
    double measureYawDrift(io::Gimbal& gimbal, double seconds, bool* pitch_still)
    {
        std::vector<std::pair<double, double>> points;   // (t, yaw°)
        double pitch_min = 1e9, pitch_max = -1e9;
        const auto start = std::chrono::steady_clock::now();
        while (std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() <
               seconds) {
            io::ImuSample imu;
            if (gimbal.latestImu(imu)) {
                const double t = std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                                               start)
                                     .count();
                points.emplace_back(t, imu.yaw * 180.0 / CV_PI);
                pitch_min = std::min(pitch_min, imu.pitch * 180.0 / CV_PI);
                pitch_max = std::max(pitch_max, imu.pitch * 180.0 / CV_PI);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        if (pitch_still != nullptr) *pitch_still = (pitch_max - pitch_min) < 0.5;
        if (points.size() < 10) return 0.0;
        const double n = static_cast<double>(points.size());
        double mt = 0.0, my = 0.0;
        for (const auto& p : points) { mt += p.first; my += p.second; }
        mt /= n; my /= n;
        double num = 0.0, den = 0.0;
        for (const auto& p : points) { num += (p.first - mt) * (p.second - my); den += (p.first - mt) * (p.first - mt); }
        return den > 1e-9 ? num / den : 0.0;
    }
#endif

    /// @brief 取一组样本的"云台姿态"：优先用帧时刻的 IMU 姿态，否则退回 yaw/pitch 参数化。
    cv::Matx33d attitudeOf(const Sample& sample)
    {
        if (sample.has_R_wb) return sample.R_wb;
        return gimbalToBase(sample.yaw, sample.pitch);
    }

    /// @brief wxyz 四元数 → 旋转矩阵。
    cv::Matx33d quaternionToRotation(double w, double x, double y, double z)
    {
        const double norm = std::sqrt(w * w + x * x + y * y + z * z);
        if (norm < 1e-12) return cv::Matx33d::eye();
        w /= norm;
        x /= norm;
        y /= norm;
        z /= norm;
        return cv::Matx33d(
            1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
            2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
            2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y));
    }

    /// @brief 解 R_camera2gimbal / t_camera2gimbal（单位：米）。
    ///
    /// 做法：6 参数（旋转向量 + 平移）高斯-牛顿，最小化"标定物在**底盘系**里的位置
    /// 在各组姿态之间的离散度"—— 标定物是静止的，理想外参下它们应重合成同一点。
    /// 为什么不用 cv::calibrateRobotWorldHandEye：它的 LI 解法对采样分布很挑（我们这组
    /// 数据直接抛 determinant(R) is null），而且它的输出是 gripper→cam 还要自己转置；
    /// 自己写只有 40 行，残差含义明确，任何姿态组合都能给出"质量有多差"。
    /// @brief 解 R_camera2gimbal / t_camera2gimbal（单位：米）+ yaw 漂移率。
    ///
    /// 做法：**7 参数**（旋转向量 + 平移 + yaw 漂移率 β）高斯-牛顿，最小化"标定物在
    /// **底盘系**里的位置在各组姿态之间的离散度"——标定物静止，理想外参下它们应合成一点。
    ///
    /// 为什么多出 β：C 板回传的 yaw 是陀螺积分量，静止时自己会漂（这台实测 0.02~0.2 °/s，
    /// 而且速率会变），表现为"回传 yaw = 真实 yaw + ∫β dt"。把 β 当未知量一起解，等于让数据
    /// 自己告诉我们这段时间漂了多少；解出来的 β 直接可读、可对账（pitch 有重力基准，不用管）。
    bool solve(const std::vector<Sample>& samples, cv::Matx33d& R_cam2gimbal,
               cv::Vec3d& t_cam2gimbal, double& residual_mm, double initial_drift_rad_s = 0.0,
               double* drift_out = nullptr)
    {
        if (samples.size() < 8) return false;
        constexpr int kParams = 7;
        // 参数：rotvec(0..2) + t(3..5) + yaw 漂移率 β(6, rad/s)
        double params[kParams] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, initial_drift_rad_s};
        auto reconstruct = [&samples](const double* p) {
            cv::Matx33d R;
            cv::Rodrigues(cv::Vec3d(p[0], p[1], p[2]), R);
            const cv::Vec3d t(p[3], p[4], p[5]);
            const double beta = p[6];
            std::vector<cv::Vec3d> centers;
            centers.reserve(samples.size());
            for (const auto& sample : samples) {
                // 扣掉漂移：回传姿态 = Rz(+β t) · 真实姿态 → 左乘 Rz(−β t) 还原
                const double a = -beta * sample.time_s;
                const cv::Matx33d Rz(std::cos(a), -std::sin(a), 0.0, std::sin(a), std::cos(a), 0.0,
                                     0.0, 0.0, 1.0);
                const cv::Vec3d p_cam(sample.tvec);
                const cv::Vec3d p_gimbal = R * p_cam + t;
                centers.push_back(Rz * attitudeOf(sample) * p_gimbal);
            }
            return centers;
        };
        auto residuals = [&reconstruct](const double* p, std::vector<double>& out) {
            const auto centers = reconstruct(p);
            cv::Vec3d mean(0, 0, 0);
            for (const auto& c : centers) mean += c;
            mean *= 1.0 / static_cast<double>(centers.size());
            out.clear();
            out.reserve(centers.size() * 3);
            for (const auto& c : centers) {
                const cv::Vec3d d = c - mean;
                out.push_back(d[0]);
                out.push_back(d[1]);
                out.push_back(d[2]);
            }
        };

        std::vector<double> r;
        residuals(params, r);
        for (int iteration = 0; iteration < 80; ++iteration) {
            const int n = static_cast<int>(r.size());
            std::vector<double> jacobian(static_cast<std::size_t>(n) * kParams, 0.0);
            const double step = 1e-6;
            for (int k = 0; k < kParams; ++k) {
                double probe[kParams];
                for (int i = 0; i < kParams; ++i) probe[i] = params[i];
                probe[k] += step;
                std::vector<double> shifted;
                residuals(probe, shifted);
                for (int i = 0; i < n; ++i) {
                    jacobian[static_cast<std::size_t>(i) * kParams + k] = (shifted[i] - r[i]) / step;
                }
            }
            cv::Matx<double, kParams, kParams> jtj(0.0);
            cv::Vec<double, kParams> jtr(0.0);
            for (int i = 0; i < n; ++i) {
                for (int a = 0; a < kParams; ++a) {
                    const double ja = jacobian[static_cast<std::size_t>(i) * kParams + a];
                    jtr[a] += ja * r[i];
                    for (int b = 0; b < kParams; ++b) {
                        jtj(a, b) += ja * jacobian[static_cast<std::size_t>(i) * kParams + b];
                    }
                }
            }
            // 漂移率这一列量纲小得多，给它单独一点正则，避免被旋转/平移项淹没
            for (int a = 0; a < kParams; ++a) jtj(a, a) *= 1.0 + 1e-6;
            for (int a = 0; a < kParams; ++a) jtj(a, a) += 1e-9;
            cv::Vec<double, kParams> delta;
            cv::solve(jtj, -jtr, delta, cv::DECOMP_CHOLESKY);
            for (int a = 0; a < kParams; ++a) params[a] += delta[a];
            residuals(params, r);
            double norm = 0.0;
            for (double value : r) norm += value * value;
            if (norm < 1e-12) break;
        }

        cv::Rodrigues(cv::Vec3d(params[0], params[1], params[2]), R_cam2gimbal);
        t_cam2gimbal = cv::Vec3d(params[3], params[4], params[5]);
        if (drift_out != nullptr) *drift_out = params[6];
        double sum_sq = 0.0;
        for (double value : r) sum_sq += value * value;
        residual_mm = std::sqrt(sum_sq / static_cast<double>(r.size() / 3)) * 1000.0;
        return true;
    }

    /// @brief 标定板参数（configs/calibration.yaml）
    struct BoardConfig
    {
        std::string pattern = "chessboard";   // chessboard | circles
        int cols = 11;
        int rows = 8;
        double square_mm = 15.0;
        std::string grid = "symmetric";
    };

    BoardConfig loadBoardConfig(const std::string& config_dir)
    {
        BoardConfig board;
        const std::string path = config_dir + "/calibration.yaml";
        std::ifstream probe(path);
        if (!probe) return board;   // 没写就用默认
        const YAML::Node file = YAML::LoadFile(path);
        const YAML::Node node = file["board"] ? file["board"] : file;
        if (!node) return board;
        board.pattern = node["pattern"].as<std::string>(board.pattern);
        board.cols = node["cols"].as<int>(board.cols);
        board.rows = node["rows"].as<int>(board.rows);
        board.square_mm = node["square_mm"].as<double>(board.square_mm);
        board.grid = node["grid"].as<std::string>(board.grid);
        return board;
    }

    /// @brief 标定板 3D 点（板面 z=0，单位：米）
    std::vector<cv::Point3f> boardObjectPoints(const BoardConfig& board)
    {
        std::vector<cv::Point3f> points;
        points.reserve(static_cast<std::size_t>(board.cols * board.rows));
        const float step = static_cast<float>(board.square_mm * 1e-3);
        for (int row = 0; row < board.rows; ++row) {
            for (int col = 0; col < board.cols; ++col) {
                points.emplace_back(col * step, row * step, 0.0f);
            }
        }
        return points;
    }

    /// @brief 在一帧里找标定板；找到就返回角点（亚像素）并用 solvePnP 给出板子在相机系的位姿。
    bool detectBoard(const cv::Mat& image, const BoardConfig& board,
                     const cv::Mat& camera_matrix, const cv::Mat& dist_coeffs,
                     std::vector<cv::Point2f>& corners, cv::Mat& rvec, cv::Mat& tvec,
                     double* reprojection_rms_px = nullptr,
                     bool allow_classic_fallback = true)
    {
        const cv::Size pattern(board.cols, board.rows);
        bool found = false;
        if (board.pattern == "circles") {
            const int flags = board.grid == "asymmetric" ? cv::CALIB_CB_ASYMMETRIC_GRID
                                                         : cv::CALIB_CB_SYMMETRIC_GRID;
            found = cv::findCirclesGrid(image, pattern, corners, flags);
        } else {
            // 先用 SB（sector-based）检测：对光照/畸变/大角度更稳，直接给亚像素角点；
            // 失败再退回经典检测（不带 FAST_CHECK —— 那个启发式会把规整的板子也拒掉）。
            // 注意：这里**不能提前 return**，下面还有"角点顺序转置"的统一处理。
            // 不要 EXHAUSTIVE：那是穷举搜索，1440x1080 上每帧几十~上百 ms（实测卡成个位数帧）
            found = cv::findChessboardCornersSB(
                image, pattern, corners, cv::CALIB_CB_NORMALIZE_IMAGE);
            if (!found && allow_classic_fallback) {
                // 经典检测在杂乱背景（工作台）上可能要几百毫秒 —— 只留给"采样时"那次
                // 全分辨率精算用，实时预览阶段不开（否则帧率直接掉到个位数）。
                found = cv::findChessboardCorners(
                    image, pattern, corners,
                    cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE);
            }
        }
        if (!found) return false;
        // 圆点阵**不能**跑 cornerSubPix：它找的是"角点"（两个方向都有梯度的鞍点），而圆心
        // 是亮度平坦的圆盘中心。实测在 1440x1080 上会把圆心拖走 3.9 px（最大 11.7 px，
        // 从暗圆心挪到白底上）——画出来就是"绿点不压圆点"，更糟的是喂给 PnP 的板位姿被
        // 系统性地带偏（这批图重投影 rms 中位 2.46 px、最差 7.26 px，看着像相机糊了）。
        // findCirclesGrid 自己给的就是亚像素质心，直接用。点越大（板越近）拖得越狠，
        // 所以"板子小的时候看着挺准、整块进画面就偏"就是这个原因。
        if (board.pattern != "circles") {
            cv::Mat gray;
            cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
            cv::cornerSubPix(gray, corners, cv::Size(11, 11), cv::Size(-1, -1),
                             cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 30,
                                              0.01));
        }

        // SB / 经典检测的**角点顺序**可能差一个转置（行列互换）—— 这正是"板子找到了、
        // 位姿却差 90°"的经典原因。两种顺序各解一次 PnP，取**重投影误差更小**的那个；
        // 矩形板（cols≠rows）能把两者区分开。选中的顺序会写回 corners，保证画框/采样一致。
        const auto object_points = boardObjectPoints(board);
        auto reprojectionRms = [&](const std::vector<cv::Point2f>& points, const cv::Mat& r,
                                   const cv::Mat& t) {
            std::vector<cv::Point2f> projected;
            cv::projectPoints(object_points, r, t, camera_matrix, dist_coeffs, projected);
            double sum = 0.0;
            for (std::size_t i = 0; i < points.size(); ++i) {
                const double d = cv::norm(points[i] - projected[i]);
                sum += d * d;
            }
            return std::sqrt(sum / static_cast<double>(points.size()));
        };

        std::vector<cv::Point2f> reordered(corners.size());
        for (int row = 0; row < board.rows; ++row) {
            for (int col = 0; col < board.cols; ++col) {
                reordered[static_cast<std::size_t>(row) * board.cols + col] =
                    corners[static_cast<std::size_t>(col) * board.rows + row];
            }
        }
        cv::Mat rvec_direct, tvec_direct, rvec_transposed, tvec_transposed;
        const bool ok_direct = cv::solvePnP(object_points, corners, camera_matrix, dist_coeffs,
                                           rvec_direct, tvec_direct, false,
                                           cv::SOLVEPNP_ITERATIVE);
        const bool ok_transposed = cv::solvePnP(object_points, reordered, camera_matrix, dist_coeffs,
                                                rvec_transposed, tvec_transposed, false,
                                                cv::SOLVEPNP_ITERATIVE);
        if (!ok_direct && !ok_transposed) return false;
        const double rms_direct = ok_direct ? reprojectionRms(corners, rvec_direct, tvec_direct)
                                            : std::numeric_limits<double>::max();
        const double rms_transposed = ok_transposed
            ? reprojectionRms(reordered, rvec_transposed, tvec_transposed)
            : std::numeric_limits<double>::max();
        if (std::getenv("ULTRA_VISION_HAND_EYE_DEBUG")) {
            std::printf("[debug] 直接序 rms=%.4f t=(%.3f,%.3f,%.3f) | 转置序 rms=%.4f t=(%.3f,%.3f,%.3f)\n",
                        rms_direct, tvec_direct.at<double>(0), tvec_direct.at<double>(1),
                        tvec_direct.at<double>(2), rms_transposed,
                        tvec_transposed.at<double>(0), tvec_transposed.at<double>(1),
                        tvec_transposed.at<double>(2));
        }
        if (rms_transposed < rms_direct) {
            corners = reordered;
            rvec = rvec_transposed;
            tvec = tvec_transposed;
            if (reprojection_rms_px != nullptr) *reprojection_rms_px = rms_transposed;
        } else {
            rvec = rvec_direct;
            tvec = tvec_direct;
            if (reprojection_rms_px != nullptr) *reprojection_rms_px = rms_direct;
        }
        return true;
    }

    /// @brief 结果的物理合理性检查。返回空串表示通过，否则返回原因。
    ///        标定最坑的地方就是"残差看着不大、解却是错的"：病态解靠一个巨大的平移加一个
    ///        补偿性的大旋转就能把残差压下去。相机装在云台上，物理上 |t| 就是几厘米、
    ///        旋转就是几度，超出这个量级一定是内参没标 / 采样退化。
    std::string sanityProblem(const std::vector<Sample>& samples, const cv::Matx33d& R,
                              const cv::Vec3d& t)
    {
        const double translation = cv::norm(t);
        cv::Mat rotation_vector;
        cv::Rodrigues(cv::Mat(R), rotation_vector);
        const double rotation_deg = cv::norm(cv::Vec3d(rotation_vector)) * 180.0 / CV_PI;
        double yaw_min = 1e9, yaw_max = -1e9, pitch_min = 1e9, pitch_max = -1e9;
        for (const auto& sample : samples) {
            yaw_min = std::min(yaw_min, sample.yaw);
            yaw_max = std::max(yaw_max, sample.yaw);
            pitch_min = std::min(pitch_min, sample.pitch);
            pitch_max = std::max(pitch_max, sample.pitch);
        }
        const double yaw_span = (yaw_max - yaw_min) * 180.0 / CV_PI;
        const double pitch_span = (pitch_max - pitch_min) * 180.0 / CV_PI;
        if (translation > 0.5) {
            return "解出的平移 " + std::to_string(translation) +
                   " m 远大于相机到云台的实际距离（应 <0.2 m）";
        }
        if (rotation_deg > 20.0) {
            return "解出的旋转 " + std::to_string(rotation_deg) +
                   "° 太大（相机安装偏差通常只有几度）";
        }
        if (std::min(yaw_span, pitch_span) < 8.0) {
            return "采样姿态单薄：yaw 跨度 " + std::to_string(yaw_span) + "°，pitch 跨度 " +
                   std::to_string(pitch_span) +
                   "°（两个轴都要 ≥8°，否则问题退化、解会沿平坦方向乱滑）";
        }
        return {};
    }

    /// @brief 把标定结果写到 yaml（粘到 configs/camera.yaml 对应相机段即可）。
    void writeYaml(const std::string& output, const std::vector<Sample>& samples,
                   const cv::Matx33d& R, const cv::Vec3d& t, double residual_mm,
                   double drift_rad_s = 0.0)
    {
        std::ofstream out(output);
        out << "# 由 tools/hand_eye_calibrate 生成（" << samples.size() << " 组样本，残差 "
            << residual_mm << " mm，yaw 漂移 " << drift_rad_s * 180.0 / CV_PI
            << " °/s）—— 粘到 configs/camera.yaml 对应相机段\n";
        out << "R_camera2gimbal: [";
        for (int row = 0; row < 3; ++row) {
            for (int col = 0; col < 3; ++col) {
                out << (row || col ? "," : "") << R(row, col);
            }
        }
        out << "]\n";
        out << "t_camera2gimbal: [" << t[0] << "," << t[1] << "," << t[2] << "]\n";
        std::cout << "[hand_eye] 已写 " << output << std::endl;
    }

    /// @brief 体检 + 存盘（不通过就只打印原因，不写文件，避免把垃圾粘进配置）
    void saveIfSane(const std::string& output, const std::vector<Sample>& samples,
                    const cv::Matx33d& R, const cv::Vec3d& t, double residual_mm,
                    double drift_rad_s = 0.0)
    {
        const std::string problem = sanityProblem(samples, R, t);
        if (!problem.empty()) {
            std::cout << "[hand_eye] **结果不合物理常识，不写盘**：" << problem << "\n"
                      << "[hand_eye] 最常见原因：① 相机内参还没标（先跑 calibrate_camera，"
                         "把结果写进 configs/camera.yaml）；② 采样只扫了一个轴。" << std::endl;
            return;
        }
        writeYaml(output, samples, R, t, residual_mm, drift_rad_s);
    }

    /// @brief 合成数据自检：已知外参 + 随机云台姿态 → 解回来（带 1 mm 噪声）。
    int selftest()
    {
        const cv::Matx33d truth_R(0.9998, 0.0175, -0.0087,   // 约 1° 偏转
                                  -0.0174, 0.9997, 0.0175,
                                  0.0090, -0.0173, 0.9998);
        const cv::Vec3d truth_t(0.045, 0.105, 0.035);        // 4.5/10.5/3.5 cm 偏置
        const cv::Vec3d target_base(2.0, 0.0, 5.0);          // 标定物在底盘系里的位置
        std::vector<Sample> samples;
        for (int i = 0; i < 16; ++i) {
            // **必须绕两个轴都有变化**：只在一条 (yaw,pitch) 直线上采，旋转轴几乎
            // 只有一个，手眼问题退化。现场采集同理：yaw 大幅扫，pitch 在其中几组单独变。
            const double yaw = 0.55 * std::sin(0.9 * i);
            const double pitch = 0.30 * std::cos(1.7 * i + 0.4);
            // 标定物在**底盘系**固定（位置 + 姿态），随云台姿态变化的是它在相机系的位姿：
            //   R_base2gimbal = R_gimbal2base^T,  t_base2gimbal = 0
            //   p_gimbal = R_base2gimbal · p_base,   R_target2gimbal = R_base2gimbal · R_target2base
            //   p_cam    = R_c2g^T · (p_gimbal − t_c2g),  R_target2cam = R_c2g^T · R_target2gimbal
            const cv::Matx33d R_base2gimbal = gimbalToBase(yaw, pitch).t();
            const cv::Vec3d p_gimbal = R_base2gimbal * target_base;
            const cv::Vec3d p_cam = truth_R.t() * (p_gimbal - truth_t);
            const cv::Matx33d R_target2gimbal = R_base2gimbal;   // 标定物姿态 = 底盘姿态
            const cv::Matx33d R_target2cam = truth_R.t() * R_target2gimbal;
            Sample sample;
            sample.yaw = yaw;
            sample.pitch = pitch;
            cv::Mat rvec;
            cv::Rodrigues(cv::Mat(R_target2cam), rvec);
            sample.rvec = rvec;
            sample.tvec = cv::Mat(cv::Vec3d(p_cam + cv::Vec3d(0.001, -0.001, 0.001)));  // 1 mm 噪声
            samples.push_back(sample);
        }
        cv::Matx33d R;
        cv::Vec3d t;
        double residual = 0.0;
        if (!solve(samples, R, t, residual)) return 1;
        cv::Mat rotation_delta;
        cv::Rodrigues(cv::Mat(R * truth_R.t()), rotation_delta);
        const double rotation_error = cv::norm(cv::Vec3d(rotation_delta));
        const double translation_error = cv::norm(t - truth_t);
        std::printf("hand_eye_calibrate --selftest: 残差 %.2f mm，旋转误差 %.2f°，平移误差 %.1f mm\n",
                    residual, rotation_error * 180.0 / CV_PI, translation_error * 1000.0);
        if (rotation_error > 0.02 || translation_error > 0.005 || residual > 5.0) {
            std::printf("FAILED: 合成数据没解回来\n");
            return 1;
        }
        // ---- ② yaw 漂移 + 7 参数：回传 yaw = 真姿态 × Rz(+β·t)，看 β 能不能解回来 ----
        {
            const double drift_deg_s = 0.2;                     // 模拟 C 板那种漂移
            const double drift_rad_s = drift_deg_s * CV_PI / 180.0;
            cv::Matx33d R_cg = cv::Matx33d::eye();              // 真外参（转动用几度的小量）
            cv::Rodrigues(cv::Vec3d(0.03, -0.02, 0.15), R_cg);
            const cv::Vec3d t_cg(0.03, -0.01, 0.05);            // 真平移 5 cm
            // 靶标要**离开 yaw 轴**（否则绕 z 的转动根本不影响靶心位置，β 不可观测）
            const cv::Vec3d p_world(1.2, 0.35, -1.8);           // 静止靶标在世界里的位置
            std::vector<Sample> samples;
            for (int i = 0; i < 16; ++i) {
                const double yaw = (-18.0 + 36.0 * i / 15.0) * CV_PI / 180.0;
                const double pitch = (-8.0 + 16.0 * (i % 4) / 3.0) * CV_PI / 180.0;
                const cv::Matx33d R_wb = gimbalToBase(yaw, pitch);
                const cv::Vec3d p_cam = R_cg.t() * (R_wb.t() * p_world - t_cg);
                const double t_s = i * 6.0;                      // 每组间隔 6 秒
                const double a = drift_rad_s * t_s;              // 漂移：左乘 Rz(+βt)
                const cv::Matx33d Rz(std::cos(a), -std::sin(a), 0.0, std::sin(a), std::cos(a), 0.0,
                                     0.0, 0.0, 1.0);
                Sample sample;
                sample.yaw = yaw + drift_rad_s * t_s;
                sample.pitch = pitch;
                sample.time_s = t_s;
                sample.R_wb = Rz * R_wb;                         // 这就是"回传姿态"
                sample.has_R_wb = true;
                sample.tvec = (cv::Mat_<double>(3, 1) << p_cam[0], p_cam[1], p_cam[2]);
                samples.push_back(sample);
            }
            cv::Matx33d R_hat;
            cv::Vec3d t_hat;
            double residual = 0.0;
            double beta = 0.0;
            const bool ok = solve(samples, R_hat, t_hat, residual, 0.0, &beta);
            const double translation_error = cv::norm(t_hat - t_cg) * 1000.0;
            cv::Mat delta_rvec;
            cv::Rodrigues(R_hat * R_cg.t(), delta_rvec);
            const double rotation_error = cv::norm(cv::Vec3d(delta_rvec));
            std::printf("yaw 漂移自检：β 真值 %.3f °/s → 解出 %.3f °/s（误差 %.0f%%），"
                        "残差 %.2f mm，平移误差 %.1f mm，转动误差 %.2f°\n",
                        drift_deg_s, beta * 180.0 / CV_PI,
                        100.0 * std::abs(beta - drift_rad_s) / drift_rad_s, residual,
                        translation_error, rotation_error * 180.0 / CV_PI);
            if (!ok || std::abs(beta - drift_rad_s) / drift_rad_s > 0.25 || residual > 2.0 ||
                translation_error > 10.0) {
                std::printf("FAILED: yaw 漂移补偿没解回来\n");
                return 1;
            }
        }

        std::printf("通过\n");
        return 0;
    }
} // namespace

#if defined(ULTRA_VISION_USE_HIK_CAMERA) || defined(ULTRA_VISION_USE_GALAXY_CAMERA)
// ---- 现场实时标定：相机 + 串口 + 每采一组立刻解算 ----
int collectLive(const std::string& config_dir, const std::string& output, int min_samples,
                bool board_mode, bool use_frame_time_attitude, double yaw_drift_deg_s,
                double drift_seconds)
{
    const BoardConfig board = loadBoardConfig(config_dir);
    const YAML::Node camera_file = YAML::LoadFile(config_dir + "/camera.yaml");
    const YAML::Node detector_file = YAML::LoadFile(config_dir + "/detector.yaml");
    const YAML::Node tracker_file = YAML::LoadFile(config_dir + "/tracker.yaml");
    const YAML::Node serial_file = YAML::LoadFile(config_dir + "/serial.yaml");

    auto_aim::ArmorSourceConfig source_cfg;
    source_cfg.classical = auto_aim::loadDetectorConfig(detector_file);
    source_cfg.pnp = auto_aim::loadPnpGeometry(tracker_file);
    source_cfg.dynamic_roi = false;   // 标定时目标就在画面里，别让 ROI/阶梯重捕干扰采样
#ifdef ULTRA_VISION_USE_OPENVINO
    // 板子模式下用不到神经网络，别加载模型（省启动时间与 CPU）
    source_cfg.use_neural = !board_mode && auto_aim::neuralDetectorEnabled(detector_file);
    source_cfg.neural = auto_aim::loadNeuralDetectorConfig(detector_file, config_dir);
#endif
    const std::string camera_name = camera_file["camera"]["name"].as<std::string>("hikcamera");
    const YAML::Node intrinsics = camera_file[camera_name];
    const cv::Mat camera_matrix = auto_aim::readMatFromYaml(intrinsics["camera_matrix"]);
    const cv::Mat dist_coeffs = auto_aim::readMatFromYaml(intrinsics["dist_coeffs"]);
    auto_aim::ArmorSource source(source_cfg, camera_matrix, dist_coeffs);

    CameraType camera;
    const int device_index = camera_file["camera"]["device_index"].as<int>(1);
    if (!camera.init("", device_index)) {
        std::cerr << "[hand_eye] 打不开相机" << std::endl;
        return 1;
    }
    if (const YAML::Node hik = camera_file["hikcamera"]) {
        camera.setAutoExposure(hik["auto_exposure"].as<bool>(false));
        camera.setExposureTime(hik["exposure_ms"].as<double>(6.0) * 1000.0);
        camera.setGain(hik["gain"].as<double>(12.0));
    }
    io::Gimbal gimbal(auto_aim::loadSerialConfig(serial_file));
    if (!gimbal.connected()) {
        std::cerr << "[hand_eye] 下位机链路没通：标定需要云台绝对角（C 板回传的 yaw/pitch）"
                  << std::endl;
        return 1;
    }

    // yaw 漂移补偿（同 auto 模式）
    const auto session_start = std::chrono::steady_clock::now();
    double drift = yaw_drift_deg_s;
    if (std::isnan(drift)) {
        std::cout << "[hand_eye] 先量 yaw 漂移率：请别碰云台 " << drift_seconds << " 秒…" << std::endl;
        bool pitch_still = false;
        drift = measureYawDrift(gimbal, drift_seconds, &pitch_still);
        std::cout << "[hand_eye] yaw 漂移率 = " << drift << " °/s（pitch "
                  << (pitch_still ? "没动" : "动了，这段可能被碰过") << "）" << std::endl;
    }
    const double drift_rad_s = drift * CV_PI / 180.0;

    std::vector<Sample> samples;
    cv::Mat frame;
    uint64_t sequence = 0;
    double last_yaw = 0.0;
    double last_pitch = 0.0;
    int last_detections = 0;
    double last_distance = 0.0;
    int preview_frames = 0;
    std::cout << "[hand_eye] 实时标定：把**标定板静止**摆在画面里（" << board.pattern << " "
              << board.cols << "x" << board.rows << "），云台在 yaw/pitch 上多转几个角度（至少 "
              << min_samples << " 组，必须绕两个不同轴）；画面里显示板子重投影 rms，"
              << "空格=采一组，s=存盘，q=退出" << std::endl;

    // 预览用降采样图：全分辨率每帧"找板 + imshow"在 NUC 上会掉到个位数帧。
    // 采样那一刻再在**全分辨率**上重解一次 —— 交互流畅、标定精度不打折。
    constexpr int kPreviewWidth = 640;
    double preview_scale = 1.0;
    int preview_errors = 0;
    auto preview_timer = std::chrono::steady_clock::now();
    while (true) {
        FrameTime frame_time;
        if (!getImageTimed(camera, frame, frame_time, 500) || frame.empty()) continue;
        ++sequence;
        const auto state = gimbal.state();
        io::ImuSample imu;
        const bool have_imu = use_frame_time_attitude ? gimbal.imuAt(frame_time, imu)
                                                      : gimbal.latestImu(imu);
        cv::Matx33d R_wb_now = imu.w != 0.0 ? quaternionToRotation(imu.w, imu.x, imu.y, imu.z)
                                            : gimbalToBase(state.yaw, state.pitch);
        if (have_imu) {
            last_yaw = imu.yaw;
            last_pitch = imu.pitch;
        } else if (state.valid) {
            last_yaw = state.yaw;
            last_pitch = state.pitch;
        }

        cv::Mat preview;
        cv::Mat preview_matrix;
        if (board_mode && frame.cols > kPreviewWidth) {
            preview_scale = static_cast<double>(kPreviewWidth) / frame.cols;
            cv::resize(frame, preview, cv::Size(), preview_scale, preview_scale, cv::INTER_AREA);
            preview_matrix = camera_matrix.clone();
            preview_matrix.at<double>(0, 0) *= preview_scale;
            preview_matrix.at<double>(1, 1) *= preview_scale;
            preview_matrix.at<double>(0, 2) *= preview_scale;
            preview_matrix.at<double>(1, 2) *= preview_scale;
        } else {
            preview_scale = 1.0;
            preview = frame;
            preview_matrix = camera_matrix;
        }

        // 找板**隔帧跑**：手动对板子不需要每帧都检测，省下的时间全给显示（帧率翻倍）。
        // 中间帧沿用上一帧结果，画面上的角点不会闪。
        static bool have_target = false;
        static cv::Mat rvec;
        static cv::Mat tvec;
        static std::vector<cv::Point2f> board_corners;
        static double board_rms = -1.0;
        const bool run_detect = !board_mode || (sequence % 2 == 1);
        const auto before_detect = std::chrono::steady_clock::now();
        if (board_mode) {
            if (run_detect) {
                board_corners.clear();
                have_target = detectBoard(preview, board, preview_matrix, dist_coeffs,
                                          board_corners, rvec, tvec, &board_rms,
                                          /*allow_classic_fallback=*/false);
                last_detections = have_target ? board.cols * board.rows : 0;
            }
        } else {
            const auto result = source.update(preview, sequence, 0, {},
                                              auto_aim::TrackerState::LOST,
                                              /*image_will_be_modified=*/true);
            const auto_aim::Armor* best = nullptr;
            double best_area = 0.0;
            for (const auto& armor : result.armors) {
                if (!armor.solve_result) continue;
                const double area = cv::norm(armor.right.top - armor.left.top) *
                    cv::norm(armor.left.top - armor.left.bottom);
                if (area > best_area) { best_area = area; best = &armor; }
            }
            if (best != nullptr) {
                have_target = true;
                rvec = best->rvec.clone();
                tvec = best->tvec.clone();
            }
            last_detections = static_cast<int>(result.armors.size());
        }
        const double detect_ms = run_detect
            ? std::chrono::duration<double, std::milli>(
                  std::chrono::steady_clock::now() - before_detect).count()
            : 0.0;
        last_distance = have_target ? cv::norm(tvec) : 0.0;

        int key = -1;
        for (const auto& corner : board_corners) {
            cv::circle(preview, corner, 3, cv::Scalar(0, 255, 0), -1);
        }
        cv::putText(preview,
                    cv::format("samples=%d  board=%d pts  rms=%.2fpx  dist=%.2fm  detect=%.0fms",
                               static_cast<int>(samples.size()), last_detections, board_rms,
                               last_distance, detect_ms),
                    cv::Point(12, 32), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 255), 2);
        cv::putText(preview,
                    cv::format("yaw=%+.1f pitch=%+.1f deg  [space]=采集 [s]=存盘 [u]=撤销 [q]=退出",
                               last_yaw * 180.0 / CV_PI, last_pitch * 180.0 / CV_PI),
                    cv::Point(12, 68), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
        cv::imshow("hand_eye_calibrate", preview);
        key = cv::waitKey(1);

        {
            const auto now = std::chrono::steady_clock::now();
            if (std::chrono::duration<double>(now - preview_timer).count() >= 1.0) {
                preview_timer = now;
                std::cout << "[hand_eye] 预览帧率 ~" << preview_frames << " fps（detect "
                          << detect_ms << " ms）" << std::endl;
                preview_frames = 0;
            }
            ++preview_frames;
        }
        (void)preview_errors;

        if (key == ' ' && have_target && state.valid) {
            Sample sample;
            const double t_s = std::chrono::duration<double>(frame_time - session_start).count();
            sample.time_s = t_s;
            sample.yaw = have_imu ? imu.yaw : state.yaw;   // 原始回传角（β 在 solve 里估）
            sample.pitch = have_imu ? imu.pitch : state.pitch;
            sample.R_wb = R_wb_now;
            sample.has_R_wb = have_imu;
            sample.rvec = rvec.clone();
            sample.tvec = tvec.clone();
            // 预览是降采样图；采样这一刻在**全分辨率**上重解，拿最高精度的位姿进样本
            if (board_mode && preview_scale < 0.999) {
                std::vector<cv::Point2f> full_corners;
                cv::Mat full_rvec;
                cv::Mat full_tvec;
                double full_rms = -1.0;
                if (detectBoard(frame, board, camera_matrix, dist_coeffs, full_corners, full_rvec,
                                full_tvec, &full_rms)) {
                    sample.rvec = full_rvec.clone();
                    sample.tvec = full_tvec.clone();
                    board_rms = full_rms;
                }
            }
            samples.push_back(sample);
            std::cout << "[hand_eye] 采第 " << samples.size() << " 组：板重投影 rms=" << board_rms
                      << " px  yaw="
                      << sample.yaw * 180.0 / CV_PI << "° pitch=" << sample.pitch * 180.0 / CV_PI
                      << "° dist=" << cv::norm(sample.tvec) << " m" << std::endl;
            if (samples.size() >= static_cast<std::size_t>(min_samples)) {
                cv::Matx33d R; cv::Vec3d t; double residual = 0.0;
                if (solve(samples, R, t, residual)) {
                    std::cout << "[hand_eye] 即时解算（" << samples.size() << " 组）：残差 " << residual
                              << " mm，t=(" << t[0] << "," << t[1] << "," << t[2] << ") m"
                              << std::endl;
                }
            }
        } else if (key == 'u' && !samples.empty()) {
            samples.pop_back();
            std::cout << "[hand_eye] 撤销一组，剩 " << samples.size() << " 组" << std::endl;
        } else if (key == 's' || key == 'q' || key == 27) {
            if (samples.size() >= static_cast<std::size_t>(min_samples)) {
                cv::Matx33d R; cv::Vec3d t; double residual = 0.0;
                if (solve(samples, R, t, residual)) {
                    saveIfSane(output, samples, R, t, residual);
                }
            } else {
                std::cerr << "[hand_eye] 只有 " << samples.size() << " 组（需 ≥" << min_samples
                          << "），不存盘" << std::endl;
            }
            break;
        }
    }
    return 0;
}

// ---- 自动采样：云台在动、板子静止时自己记一组，采够覆盖就自己解算+写盘 ----

// ---- 监视模式：只打印"回传角 vs 画面里看板的实际转角"，用来判断 yaw/pitch 通道可不可信 ----
//
// 判据：板子静止、相机绕自己转时，画面里板心的横移量 = 真实转角（Δx ≈ f·tan(Δθ)）。
// 所以每 0.5 s 打一行 yaw/pitch/板心像素，手转云台时对一下：
//   转 20° 回传 yaw 变 20°、板心横移 f·tan20° ≈ 863 px → yaw 通道可信；
//   两者对不上（或板心不动而 yaw 变）→ 回传角不是相机的真实朝向，手眼标定没意义。
int monitorLive(const std::string& config_dir, bool use_frame_time_attitude)
{
    const BoardConfig board = loadBoardConfig(config_dir);
    const YAML::Node camera_file = YAML::LoadFile(config_dir + "/camera.yaml");
    const YAML::Node serial_file = YAML::LoadFile(config_dir + "/serial.yaml");
    const std::string camera_name = camera_file["camera"]["name"].as<std::string>("hikcamera");
    const cv::Mat camera_matrix =
        auto_aim::readMatFromYaml(camera_file[camera_name]["camera_matrix"]);
    const cv::Mat dist_coeffs =
        auto_aim::readMatFromYaml(camera_file[camera_name]["dist_coeffs"]);
    CameraType camera;
    if (!camera.init("", camera_file["camera"]["device_index"].as<int>(1))) return 1;
    if (const YAML::Node hik = camera_file[camera_name]) {
        camera.setAutoExposure(hik["auto_exposure"].as<bool>(false));
        camera.setExposureTime(hik["exposure_ms"].as<double>(6.0) * 1000.0);
        camera.setGain(hik["gain"].as<double>(12.0));
    }
    io::Gimbal gimbal(auto_aim::loadSerialConfig(serial_file));
    if (!gimbal.connected()) {
        std::cerr << "[monitor] 串口没通" << std::endl;
        return 1;
    }
    std::cout << "#  时刻   yaw°    pitch°   板心x   板心y   距m   板倾斜°   (横移 px 对应的真实转角)"
              << std::endl;
    const auto start = std::chrono::steady_clock::now();
    double first_x = 0.0;
    double first_yaw = 0.0;
    bool have_first = false;
    while (true) {
        cv::Mat frame;
        FrameTime frame_time;
        if (!getImageTimed(camera, frame, frame_time, 500) || frame.empty()) continue;
        const auto state = gimbal.state();
        io::ImuSample imu;
        const bool have_imu = use_frame_time_attitude ? gimbal.imuAt(frame_time, imu)
                                                      : gimbal.latestImu(imu);
        const double yaw_now = have_imu ? imu.yaw : state.yaw;
        const double pitch_now = have_imu ? imu.pitch : state.pitch;
        std::vector<cv::Point2f> corners;
        cv::Mat rvec;
        cv::Mat tvec;
        double rms = -1.0;
        const bool found = detectBoard(frame, board, camera_matrix, dist_coeffs, corners, rvec,
                                      tvec, &rms);
        double cx = 0.0, cy = 0.0, tilt = 0.0, dist = 0.0;
        if (found) {
            cx = corners[0].x;   // 用第 0 个角点比用质心更稳（质心会被板被裁掉影响）
            cy = corners[0].y;
            cv::Mat R;
            cv::Rodrigues(rvec, R);
            const cv::Mat n = R * (cv::Mat_<double>(3, 1) << 0.0, 0.0, 1.0);
            tilt = std::acos(std::min(1.0, std::abs(n.at<double>(2)))) * 180.0 / CV_PI;
            dist = cv::norm(tvec);
        }
        const double t = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        if (found && !have_first) {
            first_x = cx;
            first_yaw = yaw_now * 180.0 / CV_PI;
            have_first = true;
        }
        std::printf("%6.1fs %+8.2f %+8.2f %7.0f %7.0f %6.2f %8.1f    ", t,
                    yaw_now * 180.0 / CV_PI, pitch_now * 180.0 / CV_PI, cx, cy, dist, tilt);
        if (found && have_first) {
            const double dx = cx - first_x;
            const double true_deg = std::atan2(dx, camera_matrix.at<double>(0, 0)) * 180.0 / CV_PI;
            std::printf("横移 %+6.0f px = 真实转 %+6.2f°  回传变 %+6.2f°  比值 %.2f", dx, true_deg,
                        yaw_now * 180.0 / CV_PI - first_yaw,
                        (yaw_now * 180.0 / CV_PI - first_yaw) != 0.0
                            ? true_deg / (yaw_now * 180.0 / CV_PI - first_yaw)
                            : 0.0);
        } else {
            std::printf("(没找到板)");
        }
        std::printf("\n");
        std::cout.flush();
        std::this_thread::sleep_for(std::chrono::milliseconds(400));
    }
    return 0;
}
//
// 为什么要这个模式：手动模式每组都要在窗口上按一次空格，SSH/无人值守时按不了键。
// 采样时机由三个条件决定（缺一个就不采）：
//   ① 本帧板子**完整**在画面里（四边留白 ≥20 px）且重投影 rms < 0.5 px；
//   ② 最近 4 帧的板位姿几乎不动（平移抖动 <8 mm、转角抖动 <0.5°）——保证"现在静止"；
//   ③ 这个姿态和已采的每组都差得够远（默认 ≥4°）——保证每组的相对旋转有信息量。
int collectAuto(const std::string& config_dir, const std::string& output, int min_samples,
                bool board_mode, double min_step_deg, double yaw_span_target,
                double pitch_span_target, const std::string& frame_dir,
                bool use_frame_time_attitude, double yaw_drift_deg_s, double drift_seconds)
{
    const BoardConfig board = loadBoardConfig(config_dir);
    const YAML::Node camera_file = YAML::LoadFile(config_dir + "/camera.yaml");
    const YAML::Node serial_file = YAML::LoadFile(config_dir + "/serial.yaml");

    const std::string camera_name = camera_file["camera"]["name"].as<std::string>("hikcamera");
    const YAML::Node intrinsics = camera_file[camera_name];
    const cv::Mat camera_matrix = auto_aim::readMatFromYaml(intrinsics["camera_matrix"]);
    const cv::Mat dist_coeffs = auto_aim::readMatFromYaml(intrinsics["dist_coeffs"]);

    CameraType camera;
    const int device_index = camera_file["camera"]["device_index"].as<int>(1);
    if (!camera.init("", device_index)) {
        std::cerr << "[hand_eye] 打不开相机" << std::endl;
        return 1;
    }
    if (const YAML::Node hik = camera_file[camera_name]) {
        camera.setAutoExposure(hik["auto_exposure"].as<bool>(false));
        camera.setExposureTime(hik["exposure_ms"].as<double>(6.0) * 1000.0);
        camera.setGain(hik["gain"].as<double>(12.0));
    }
    io::Gimbal gimbal(auto_aim::loadSerialConfig(serial_file));
    if (!gimbal.connected()) {
        std::cerr << "[hand_eye] 下位机链路没通（自动模式需要 C 板回传的 yaw/pitch）" << std::endl;
        return 1;
    }
    if (!frame_dir.empty()) std::system(("mkdir -p " + frame_dir).c_str());

    // ---- yaw 漂移补偿：C 板回传的 yaw 是陀螺积分量，静止时自己也以 ~0.2°/s 爬升 ----
    const auto session_start = std::chrono::steady_clock::now();
    double drift = yaw_drift_deg_s;
    if (std::isnan(drift)) {
        std::cout << "[hand_eye-auto] 先量 yaw 漂移率：请**别碰云台** " << drift_seconds << " 秒…"
                  << std::endl;
        bool pitch_still = false;
        drift = measureYawDrift(gimbal, drift_seconds, &pitch_still);
        std::cout << "[hand_eye-auto] yaw 漂移率 = " << drift << " °/s（pitch "
                  << (pitch_still ? "没动，这段云台确实没被碰" : "动了，这段可能被碰过")
                  << "）；每组的 yaw 会按采样时刻扣掉这一项" << std::endl;
    } else {
        std::cout << "[hand_eye-auto] 用命令行给的 yaw 漂移率 " << drift << " °/s" << std::endl;
    }
    const double drift_rad_s = drift * CV_PI / 180.0;

    std::cout << "[hand_eye-auto] 标定板（" << board.pattern << " " << board.cols << "x"
              << board.rows << "）摆在画面里**别动**，然后慢慢转云台：yaw 扫一大段、pitch 分几档。"
              << "采够 " << min_samples << " 组（yaw 跨度 ≥" << yaw_span_target << "°、pitch ≥"
              << pitch_span_target << "°）自动收工。" << std::endl;

    struct Pose
    {
        cv::Mat rvec;
        cv::Mat tvec;
    };
    std::vector<Sample> samples;
    std::vector<Pose> recent;        // 最近几帧（判"此刻静止"）
    std::vector<cv::Vec3d> centers;  // 各组板心在底盘系的位置（判"板有没有被动过"）
    constexpr int kRecentWindow = 4;
    constexpr double kFullMarginPx = 20.0;
    constexpr double kRmsLimitPx = 0.5;
    constexpr double kJitterMm = 8.0;
    constexpr double kJitterDeg = 0.5;
    constexpr double kBoardMovedMm = 60.0;

    cv::Mat frame;
    cv::Matx33d solution_R;
    cv::Vec3d solution_t;
    bool have_solution = false;
    double solution_residual = 0.0;
    double solution_drift = 0.0;   // 求解器估出的 yaw 漂移率（rad/s）
    auto last_status = std::chrono::steady_clock::now();

    while (true) {
        FrameTime frame_time;
        if (!getImageTimed(camera, frame, frame_time, 500) || frame.empty()) continue;
        const auto state = gimbal.state();
        if (!state.valid) continue;
        // 姿态按**帧到手时刻**取（时间戳插值），而不是"处理完这帧之后"的最新值：
        // 相机队列 + 检测耗时加起来 10~40 ms，手转 30~60°/s 时就是 1~3° 的姿态错配，
        // 而这一项在 2.2 m 处会放大成 40~150 mm 的假位移——比我们要测的平移量还大。
        io::ImuSample imu;
        const bool have_imu = use_frame_time_attitude ? gimbal.imuAt(frame_time, imu)
                                                      : gimbal.latestImu(imu);
        const double yaw_now = have_imu ? imu.yaw : yaw_now;
        const double pitch_now = have_imu ? imu.pitch : pitch_now;
        const cv::Matx33d R_wb_now = have_imu
            ? quaternionToRotation(imu.w, imu.x, imu.y, imu.z)
            : gimbalToBase(yaw_now, pitch_now);
        const double attitude_age_ms = have_imu
            ? std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() -
                                                        frame_time)
                  .count()
            : 0.0;

        std::vector<cv::Point2f> corners;
        cv::Mat rvec;
        cv::Mat tvec;
        double rms = -1.0;
        const auto before_detect = std::chrono::steady_clock::now();
        const bool found = detectBoard(frame, board, camera_matrix, dist_coeffs, corners, rvec,
                                      tvec, &rms);
        const double detect_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - before_detect).count();

        // ① 完整在画面里？
        bool full = found;
        if (found) {
            for (const auto& corner : corners) {
                if (corner.x < kFullMarginPx || corner.y < kFullMarginPx ||
                    corner.x > frame.cols - kFullMarginPx || corner.y > frame.rows - kFullMarginPx) {
                    full = false;
                    break;
                }
            }
        }

        // ② 现在静止？（最近 kRecentWindow 帧板位姿的抖动）
        double jitter_mm = 1e9;
        double jitter_deg = 1e9;
        if (found && full) {
            recent.push_back({rvec.clone(), tvec.clone()});
            if (static_cast<int>(recent.size()) > kRecentWindow) recent.erase(recent.begin());
            if (static_cast<int>(recent.size()) >= 2) {
                jitter_mm = 0.0;
                jitter_deg = 0.0;
                const auto& reference = recent.back();
                for (const auto& pose : recent) {
                    jitter_mm = std::max(jitter_mm, cv::norm(pose.tvec - reference.tvec) * 1000.0);
                    cv::Mat R1, R2, delta;
                    cv::Rodrigues(pose.rvec, R1);
                    cv::Rodrigues(reference.rvec, R2);
                    cv::Rodrigues(R1 * R2.t(), delta);
                    jitter_deg = std::max(jitter_deg, cv::norm(delta) * 180.0 / CV_PI);
                }
            }
        } else {
            recent.clear();
        }
        const bool still = static_cast<int>(recent.size()) >= kRecentWindow && jitter_mm < kJitterMm &&
                           jitter_deg < kJitterDeg;
        const bool slow = std::abs(state.yaw_vel) < 0.15 && std::abs(state.pitch_vel) < 0.15;

        // ③ 姿态够不够新鲜？
        double nearest_deg = 1e9;
        for (const auto& sample : samples) {
            const double dy = (sample.yaw - yaw_now) * 180.0 / CV_PI;
            const double dp = (sample.pitch - pitch_now) * 180.0 / CV_PI;
            nearest_deg = std::min(nearest_deg, std::hypot(dy, dp));
        }
        const bool novel = nearest_deg > min_step_deg;

        // 板有没有被移动过：用当前解把板心投回底盘系，看它离中位值多远。
        // 注意：只有在"当前解已经通过体检"时这个判据才有意义——解本身是垃圾的时候，
        // 投回去的点当然乱飞。所以垃圾解阶段只警告、不拦采样，否则会死锁在一组上解不出来。
        bool board_moved = false;
        double moved_mm = 0.0;
        const bool solution_sane = have_solution && sanityProblem(samples, solution_R, solution_t).empty();
        if (have_solution && solution_sane && found && full) {
            const cv::Vec3d center = R_wb_now *
                                     (solution_R * cv::Vec3d(tvec) + solution_t);
            cv::Vec3d median(0, 0, 0);
            for (const auto& c : centers) median += c;
            if (!centers.empty()) {
                median *= 1.0 / static_cast<double>(centers.size());
                moved_mm = cv::norm(center - median) * 1000.0;
                board_moved = moved_mm > kBoardMovedMm;
            }
        }
        // 板相对相机的倾斜角 + 距离：这两个直接决定标定精度（板越正对相机，PnP 的姿态越不灵）
        double board_tilt_deg = 0.0;
        double board_dist_m = 0.0;
        if (found) {
            cv::Mat R_cb;
            cv::Rodrigues(rvec, R_cb);
            const cv::Mat normal = R_cb * (cv::Mat_<double>(3, 1) << 0.0, 0.0, 1.0);
            board_tilt_deg = std::acos(std::min(1.0, std::abs(normal.at<double>(2)))) * 180.0 / CV_PI;
            board_dist_m = cv::norm(tvec);
        }

        // 预览：画点 + 状态
        cv::Mat display = frame.clone();   // 画在副本上，原图要留给"存采样帧"
        for (const auto& corner : corners) {
            cv::circle(display, corner, 4, full ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 165, 255),
                       -1);
        }
        double yaw_span = 0.0;
        double pitch_span = 0.0;
        if (!samples.empty()) {
            double yw_min = 1e9, yw_max = -1e9, pt_min = 1e9, pt_max = -1e9;
            for (const auto& sample : samples) {
                yw_min = std::min(yw_min, sample.yaw);
                yw_max = std::max(yw_max, sample.yaw);
                pt_min = std::min(pt_min, sample.pitch);
                pt_max = std::max(pt_max, sample.pitch);
            }
            yaw_span = (yw_max - yw_min) * 180.0 / CV_PI;
            pitch_span = (pt_max - pt_min) * 180.0 / CV_PI;
        }
        cv::putText(display,
                    cv::format("AUTO samples=%d  rms=%.2fpx  dist=%.2fm  detect=%.0fms",
                               static_cast<int>(samples.size()), rms, found ? cv::norm(tvec) : 0.0,
                               detect_ms),
                    cv::Point(16, 40), cv::FONT_HERSHEY_SIMPLEX, 0.9,
                    found ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255), 2);
        cv::putText(display,
                    cv::format("yaw=%+.1f pitch=%+.1f (span %.1f/%.1f)  tilt=%.0f° d=%.2fm  %s",
                               yaw_now * 180.0 / CV_PI, pitch_now * 180.0 / CV_PI, yaw_span,
                               pitch_span, board_tilt_deg, board_dist_m,
                               !found ? "找板中" : (!full ? "板要完整入画" : (still && slow ? "静止，可采"
                                                                                          : "转动中"))),
                    cv::Point(16, 80), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 255), 2);
        if (found) {
            const char* hint = board_tilt_deg < 20.0 ? "板太正对相机：斜 30° 摆（姿态精度差 3~5 倍）"
                                                     : (board_dist_m > 1.6 ? "板有点远：挪到 1.0~1.5 m"
                                                                           : "摆位 OK");
            cv::putText(display, hint, cv::Point(16, 160), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                        board_tilt_deg < 20.0 ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0), 2);
        }
        if (have_solution) {
            cv::putText(display,
                        cv::format("residual=%.1fmm  t=(%.3f,%.3f,%.3f)m", solution_residual,
                                   solution_t[0], solution_t[1], solution_t[2]),
                        cv::Point(16, 120), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 200, 0), 2);
        }
        cv::resize(display, display, cv::Size(), 0.5, 0.5);
        cv::imshow("hand_eye_auto", display);
        cv::waitKey(1);

        const bool accept = found && full && rms < kRmsLimitPx && still && slow && novel &&
                            !board_moved;
        if (accept) {
            Sample sample;
            const double t_s =
                std::chrono::duration<double>(frame_time - session_start).count();
            sample.time_s = t_s;
            sample.yaw = yaw_now;      // 原始回传角；yaw 漂移由求解器估的 β 统一扣（见 solve）
            sample.pitch = pitch_now;
            sample.rvec = rvec.clone();
            sample.tvec = tvec.clone();
            sample.R_wb = R_wb_now;
            sample.has_R_wb = have_imu;
            samples.push_back(sample);
            std::cout << "[hand_eye-auto] 采第 " << samples.size() << " 组：yaw="
                      << sample.yaw * 180.0 / CV_PI << "° pitch=" << sample.pitch * 180.0 / CV_PI
                      << "° rms=" << rms << "px dist=" << cv::norm(sample.tvec)
                      << "m 覆盖(yaw " << yaw_span << "° / pitch " << pitch_span << "°)" << std::endl;
            if (!frame_dir.empty()) {
                const int index = static_cast<int>(samples.size());
                cv::imwrite(frame_dir + "/" + std::to_string(index) + ".jpg", frame);
                std::ofstream pose_out(frame_dir + "/" + std::to_string(index) + ".yaml");
                io::ImuSample imu;
                // 一并存 C 板回传的四元数：电机角(yaw/pitch)有齿轮/皮带背隙，四元数是 IMU
                // 直测姿态。两组姿态谁更贴合相机实际转动，事后一比就知道能不能拿来解外参。
                pose_out << sample.yaw << " " << sample.pitch;
                if (gimbal.latestImu(imu)) {
                    pose_out << " " << imu.w << " " << imu.x << " " << imu.y << " " << imu.z;
                }
                pose_out << " " << sample.time_s << "\n";
            }
            have_solution = solve(samples, solution_R, solution_t, solution_residual, drift_rad_s,
                                  &solution_drift);
            if (have_solution) {
                const cv::Vec3d center = attitudeOf(sample) *
                                         (solution_R * cv::Vec3d(sample.tvec) + solution_t);
                centers.push_back(center);
                std::cout << "[hand_eye-auto] 即时解算（" << samples.size() << " 组）：残差 "
                          << solution_residual << " mm，t=(" << solution_t[0] << ","
                          << solution_t[1] << "," << solution_t[2] << ") m，yaw 漂移 "
                          << solution_drift * 180.0 / CV_PI << " °/s" << std::endl;
                saveIfSane(output, samples, solution_R, solution_t, solution_residual,
                           solution_drift);
                const std::string problem = sanityProblem(samples, solution_R, solution_t);
                // 收工条件：样本够多（平移项要靠大量样本把 ~1° 的姿态噪声平均掉，8~10 组
                // 连 6 参数都能过拟合出 0 残差，说明不了任何问题）+ 覆盖够宽 + 体检过。
                if (problem.empty() && samples.size() >= static_cast<std::size_t>(min_samples) &&
                    yaw_span >= yaw_span_target && pitch_span >= pitch_span_target &&
                    solution_residual < 15.0) {
                    std::cout << "[hand_eye-auto] 覆盖够了且体检通过，已写 " << output
                              << "；标定结束。" << std::endl;
                    break;
                }
                if (samples.size() >= static_cast<std::size_t>(min_samples)) {
                    std::cout << "[hand_eye-auto] 样本够了但还没收敛（" << (problem.empty() ? "" : problem)
                              << " 残差 " << solution_residual << " mm）——请继续多转几组、"
                                 "板子斜一点、挪近一点" << std::endl;
                }
            }
        } else {
            const auto now = std::chrono::steady_clock::now();
            if (std::chrono::duration<double>(now - last_status).count() >= 2.0) {
                last_status = now;
                std::cout << "[hand_eye-auto] 板=" << (found ? (full ? "完整" : "被裁") : "没找到")
                          << " rms=" << rms << "px 静止=" << (still ? "是" : "否")
                          << "(抖 " << jitter_mm << "mm/" << jitter_deg << "°) 转速="
                          << std::abs(state.yaw_vel) * 180.0 / CV_PI << "°/s 新鲜="
                          << (novel ? "是" : "否") << " 已采 " << samples.size() << " 组"
                          << (board_moved ? "  ⚠板子动过了(" + std::to_string(moved_mm) + "mm)"
                                          : "")
                          << std::endl;
            }
        }
    }
    return 0;
}
#endif

int main(int argc, char* argv[])
{
    cv::ocl::setUseOpenCL(false);   // 工具不需要 OpenCL；macOS 上它的缓存会在检测时报错刷屏
    std::string folder;
    std::string config_dir = "configs";
    std::string output = "hand_eye.yaml";
    int min_samples = 8;
    bool live = false;
    bool auto_mode = false;
    bool monitor = false;
    bool use_frame_time_attitude = true;   // 默认按帧时间戳取姿态（--attitude latest 可退回旧行为）
    double yaw_drift_deg_s = std::nan("");  // --yaw-drift；默认自动量（NaN = 自动）
    double drift_seconds = 6.0;             // 自动量漂移率的时长
    double min_step_deg = 4.0;
    double yaw_span_target = 15.0;
    double pitch_span_target = 10.0;
    std::string frame_dir;
    std::string target = "board";   // board（默认，精度高）| armor（应急）
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            std::cout << "用法:\n"
                         "  hand_eye_calibrate --live                    现场实时标定（相机+串口+"
                         "标定板，空格采样/s 存盘）\n"
                         "  hand_eye_calibrate --auto                    自动采样：板子静止、云台转动时"
                         "自动记，采够覆盖自动解算+写盘（SSH 无人值守用这个）\n"
                         "  hand_eye_calibrate <文件夹> [--target board|armor]  离线复算"
                         "（{i}.jpg + {i}.yaml）\n"
                         "  hand_eye_calibrate --selftest                求解器自检（无需硬件）\n"
                         "选项：--config-dir configs  --out hand_eye.yaml  --min-samples 8\n"
                         "      --min-step 4       自动模式：两组姿态至少差多少度才记\n"
                         "      --yaw-span 15 --pitch-span 10   自动模式：覆盖到什么程度就收工\n"
                         "      --save-frames <目录>  自动模式：把采到的原图+yaw/pitch 存下来，"
                         "方便事后离线复核\n"
                         "      --attitude ts|latest  姿态取法：ts=按帧时间戳插值（默认，推荐）；"
                         "latest=用最新值（对比用）\n"
                         "      --yaw-drift auto|<°/s>  yaw 漂移率：C 板回传的 yaw 是陀螺积分量，"
                         "静止时自己会以 ~0.2°/s 爬升；auto=开跑前量 6 秒（默认）\n"
                         "      --drift-seconds 6      自动量漂移率的时长（这段时间别碰云台）\n";
            return 0;
        }
        if (arg == "--selftest") return selftest();
        else if (arg == "--live") live = true;
        else if (arg == "--auto") auto_mode = true;
        else if (arg == "--monitor") monitor = true;
        else if (arg == "--yaw-drift" && i + 1 < argc) {
            const std::string value = argv[++i];
            yaw_drift_deg_s = (value == "auto") ? std::nan("") : std::atof(value.c_str());
        } else if (arg == "--drift-seconds" && i + 1 < argc) drift_seconds = std::atof(argv[++i]);
        else if (arg == "--attitude" && i + 1 < argc) {
            const std::string mode = argv[++i];
            use_frame_time_attitude = (mode != "latest");
        }
        else if (arg == "--min-step" && i + 1 < argc) min_step_deg = std::atof(argv[++i]);
        else if (arg == "--yaw-span" && i + 1 < argc) yaw_span_target = std::atof(argv[++i]);
        else if (arg == "--pitch-span" && i + 1 < argc) pitch_span_target = std::atof(argv[++i]);
        else if (arg == "--save-frames" && i + 1 < argc) frame_dir = argv[++i];
        else if (arg == "--target" && i + 1 < argc) target = argv[++i];
        else if (arg == "--min-samples" && i + 1 < argc) min_samples = std::atoi(argv[++i]);
        else if (arg == "--config-dir" && i + 1 < argc) config_dir = argv[++i];
        else if (arg == "--out" && i + 1 < argc) output = argv[++i];
        else if (folder.empty()) folder = arg;
    }
    if (live) {
#if defined(ULTRA_VISION_USE_HIK_CAMERA) || defined(ULTRA_VISION_USE_GALAXY_CAMERA)
        return collectLive(config_dir, output, min_samples, target != "armor",
                           use_frame_time_attitude, yaw_drift_deg_s, drift_seconds);
#else
        std::cerr << "--live 需要带相机 SDK 构建（本构建没有相机驱动）" << std::endl;
        return 2;
#endif
    }
    if (auto_mode) {
#if defined(ULTRA_VISION_USE_HIK_CAMERA) || defined(ULTRA_VISION_USE_GALAXY_CAMERA)
        return collectAuto(config_dir, output, std::max(min_samples, 12), target != "armor",
                           min_step_deg, yaw_span_target, pitch_span_target, frame_dir,
                           use_frame_time_attitude, yaw_drift_deg_s, drift_seconds);
#else
        std::cerr << "--auto 需要带相机 SDK 构建（本构建没有相机驱动）" << std::endl;
        return 2;
#endif
    }
    if (monitor) {
#if defined(ULTRA_VISION_USE_HIK_CAMERA) || defined(ULTRA_VISION_USE_GALAXY_CAMERA)
        return monitorLive(config_dir, use_frame_time_attitude);
#else
        std::cerr << "--monitor 需要带相机 SDK 构建" << std::endl;
        return 2;
#endif
    }
    if (folder.empty()) {
        std::cerr << "用法: hand_eye_calibrate --live | <数据文件夹> | --selftest "
                     "[--config-dir configs] [--out hand_eye.yaml] [--min-samples 8] "
                     "[--target board|armor]\n";
        return 2;
    }

    // 用我们自己的检测前端（和真机同一条链）在每张图上找静止装甲板 → PnP 位姿
    const YAML::Node detector_file = YAML::LoadFile(config_dir + "/detector.yaml");
    const YAML::Node tracker_file = YAML::LoadFile(config_dir + "/tracker.yaml");
    const YAML::Node camera_file = YAML::LoadFile(config_dir + "/camera.yaml");
    auto_aim::ArmorSourceConfig source_cfg;
    source_cfg.classical = auto_aim::loadDetectorConfig(detector_file);
    source_cfg.pnp = auto_aim::loadPnpGeometry(tracker_file);
#ifdef ULTRA_VISION_USE_OPENVINO
    source_cfg.use_neural = auto_aim::neuralDetectorEnabled(detector_file);
    source_cfg.neural = auto_aim::loadNeuralDetectorConfig(detector_file, config_dir);
#endif
    const std::string camera_name =
        camera_file["camera"]["name"].as<std::string>("hikcamera");
    const YAML::Node intrinsics = camera_file[camera_name];
    const cv::Mat camera_matrix = auto_aim::readMatFromYaml(intrinsics["camera_matrix"]);
    const cv::Mat dist_coeffs = auto_aim::readMatFromYaml(intrinsics["dist_coeffs"]);
    auto_aim::ArmorSource source(source_cfg, camera_matrix, dist_coeffs);

    const BoardConfig board = loadBoardConfig(config_dir);
    const bool board_mode = target != "armor";
    if (board_mode) {
        std::cout << "[hand_eye] 标定物：标定板（" << board.pattern << " " << board.cols << "x"
                  << board.rows << "，方格 " << board.square_mm << "mm）" << std::endl;
    } else {
        std::cout << "[hand_eye] 标定物：装甲板（应急模式）" << std::endl;
    }

    std::vector<Sample> samples;
    for (int index = 1;; ++index) {
        const std::string image_path = folder + "/" + std::to_string(index) + ".jpg";
        const std::string pose_path = folder + "/" + std::to_string(index) + ".yaml";
        if (!std::ifstream(image_path).good()) break;   // 先判存在，避免 imread 刷警告
        const cv::Mat image = cv::imread(image_path);
        if (image.empty()) break;
        std::ifstream pose_in(pose_path);
        if (!pose_in) {
            std::cerr << "[hand_eye] 缺 " << pose_path << "（应由 tools/calibrate_capture 生成）"
                      << std::endl;
            break;
        }
        double yaw = 0.0;
        double pitch = 0.0;
        pose_in >> yaw >> pitch;

        Sample sample;
        sample.yaw = yaw;
        sample.pitch = pitch;
        if (board_mode) {
            std::vector<cv::Point2f> corners;
            double rms = -1.0;
            if (!detectBoard(image, board, camera_matrix, dist_coeffs, corners, sample.rvec,
                             sample.tvec, &rms)) {
                std::cout << "[hand_eye] 第 " << index << " 张没找到标定板，跳过" << std::endl;
                continue;
            }
            std::cout << "[hand_eye] 第 " << index << " 张：板 " << corners.size()
                      << " 点，重投影 rms=" << rms << " px，距离 " << cv::norm(sample.tvec)
                      << " m" << std::endl;
        } else {
            const auto result = source.update(image, static_cast<uint64_t>(index), 0, {},
                                              auto_aim::TrackerState::LOST, false);
            const auto_aim::Armor* best = nullptr;
            double best_area = 0.0;
            for (const auto& armor : result.armors) {
                if (!armor.solve_result) continue;
                const double area = cv::norm(armor.right.top - armor.left.top) *
                    cv::norm(armor.left.top - armor.left.bottom);
                if (area > best_area) {
                    best_area = area;
                    best = &armor;
                }
            }
            if (best == nullptr || !best->solve_result) {
                std::cout << "[hand_eye] 第 " << index << " 张没检测到装甲板，跳过" << std::endl;
                continue;
            }
            sample.rvec = best->rvec.clone();
            sample.tvec = best->tvec.clone();
            std::cout << "[hand_eye] 第 " << index << " 张：装甲板距离 " << cv::norm(sample.tvec)
                      << " m" << std::endl;
        }
        samples.push_back(sample);
    }
    if (samples.size() < 8) {
        std::cerr << "[hand_eye] 有效样本 " << samples.size()
                  << " 组（<8 组解不稳）：请让云台在 yaw/pitch 上多转几个角度，"
                     "每组保持静止再采集" << std::endl;
        return 1;
    }

    cv::Matx33d R;
    cv::Vec3d t;
    double residual = 0.0;
    if (!solve(samples, R, t, residual)) return 1;
    cv::Mat rotation_vector;
    cv::Rodrigues(cv::Mat(R), rotation_vector);
    const double rotation_error_deg = cv::norm(cv::Vec3d(rotation_vector)) * 180.0 / CV_PI;
    std::cout << "[hand_eye] 样本 " << samples.size() << " 组，残差 " << residual
              << " mm（<10 mm 可用），外参转动量 " << rotation_error_deg << "°" << std::endl;
    std::cout << "[hand_eye] R_camera2gimbal = [";
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            std::cout << (row || col ? "," : "") << R(row, col);
        }
    }
    std::cout << "]\n[hand_eye] t_camera2gimbal = [" << t[0] << "," << t[1] << "," << t[2] << "] m"
              << std::endl;

    saveIfSane(output, samples, R, t, residual);
    return 0;
}
