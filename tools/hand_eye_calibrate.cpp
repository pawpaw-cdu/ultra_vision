// 相机 → 云台外参标定（hand-eye），针对「C 板回传**编码器绝对角**」的情况重写。
//
// 与上一版的区别（上一版是为了应付"陀螺积分 yaw 一直漂"的临时状态）：
//   1) 不再需要任何漂移补偿（ZUPT / settle / 拟合 β）——编码器角不漂；
//   2) 编码器绝对角会在 ±180° 折返，所以样本先**解缠**，跨度判据按解缠后的值算；
//   3) 新增 yaw/pitch **标度自标定**（--fit-yaw-scale / --fit-pitch-scale）：C 板那边的角
//      如果有减速比/单位/符号问题，解出来的 κ 会直接告诉你差多少倍（κ=0.5 就是差 2 倍，
//      κ<0 就是符号反了）；
//   4) 新增**半分交叉验证**：一半样本拟合、另一半检验，检验残差必须和拟合残差同量级才允许
//      写盘（防"残差很小但换一半样本就崩"的过拟合解）；
//   5) 新增**平移可辨识度 σt**：由残差 + 雅可比算平移三轴 1σ，>0.1 m 说明该方向没有信息量
//      （转动范围不够），直接提示，而不是给一个乱跑的 |t|。
//
// 用法：
//   hand_eye_calibrate --auto                 自动采样（推荐）
//   hand_eye_calibrate --live                 手动：空格采样 / s 存盘 / u 撤销 / q 退出
//   hand_eye_calibrate --monitor              实时对表：画面里的真实转角 vs 回传角（含比值）
//   hand_eye_calibrate <样本目录>              离线复算 + 交叉验证
//   hand_eye_calibrate --selftest             合成数据自检（含 ±180 折返与标度错误）
//
// 选项：--config-dir configs --out hand_eye.yaml --min-samples 20 --min-step 3
//       --yaw-span 25 --pitch-span 12 --save-frames <目录>
//       --fit-yaw-scale --fit-pitch-scale
//
// 模型（与 tracker 的 cameraToBaseRotation 完全一致）：
//   p_base = Ry(yaw)·Rx(pitch) · (R_camera2gimbal · p_cam + t_camera2gimbal)
// 标定物静止时，各组样本的 p_base 应重合成一点 → 最小化这个"离散度"。
// 残差 = 板心投回底盘系后到中位点的平均距离（mm），<10 mm 可用。

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/core/ocl.hpp>
#include <opencv2/opencv.hpp>

#include "config_loader.hpp"
#include "io/gimbal/gimbal.hpp"

#if defined(ULTRA_VISION_USE_HIK_CAMERA)
#include "io/camera/HikCamera.hpp"
using CameraType = rm_ultra::HikCamera;
#elif defined(ULTRA_VISION_USE_GALAXY_CAMERA)
#include "io/camera/GalaxyCamera.hpp"
using CameraType = rm_ultra::GalaxyCamera;
#endif

namespace
{
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
    /// 做法：6 参数（旋转向量 + 平移）高斯-牛顿，最小化"标定物在**底盘系**里的位置\n    /// 在各组姿态之间的离散度"—— 标定物是静止的，理想外参下它们应重合成同一点。
    /// 为什么不用 cv::calibrateRobotWorldHandEye：它的 LI 解法对采样分布很挑（我们这组
    /// 数据直接抛 determinant(R) is null），而且它的输出是 gripper→cam 还要自己转置；
    /// 自己写只有 40 行，残差含义明确，任何姿态组合都能给出"质量有多差"。
    /// @brief 旋转矩阵 → wxyz 四元数（|w| 取正，便于人读）。
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

        // SB / 经典检测的**角点顺序**可能差一个转置（行列互换）—— 这正是"板子找到了、\n        // 位姿却差 90°"的经典原因。两种顺序各解一次 PnP，取**重投影误差更小**的那个；
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


    struct Sample
    {
        double yaw = 0.0;      // 云台绝对 yaw（rad，**已解缠**；编码器角会在 ±180° 折返）
        double pitch = 0.0;    // 云台绝对 pitch（rad）
        cv::Mat rvec;          // 标定物在相机系的姿态（PnP）
        cv::Mat tvec;          // 标定物在相机系的位置
        double time_s = 0.0;   // 采样时刻（会话起点为 0，只用于日志/排序）
    };

    /// @brief 把角度折到 (−π, π]。
    double wrapPi(double value)
    {
        while (value > CV_PI) value -= 2.0 * CV_PI;
        while (value <= -CV_PI) value += 2.0 * CV_PI;
        return value;
    }

    /// @brief 按参考值把 yaw 解缠（编码器绝对角在 ±180° 处会跳变）。
    double unwrapYaw(double yaw, double reference)
    {
        double out = yaw;
        while (out - reference > CV_PI) out -= 2.0 * CV_PI;
        while (out - reference < -CV_PI) out += 2.0 * CV_PI;
        return out;
    }

    /// @brief 整组样本按第一个样本解缠。
    void unwrapSamples(std::vector<Sample>& samples)
    {
        if (samples.empty()) return;
        const double reference = samples.front().yaw;
        for (auto& sample : samples) sample.yaw = unwrapYaw(sample.yaw, reference);
    }

    /// @brief 标定参数：yaw/pitch 是否也当未知量一起解（查标度/符号/减速比）。
    struct FitOptions
    {
        bool fit_yaw_scale = false;
        bool fit_pitch_scale = false;
        /// @brief 平移上限（米）。物理上 |t| 就是几个厘米；不加约束时优化器会滑进
        ///        "|t| 好几米 + 补偿性大旋转"的退化盆（残差看着还行但完全是垃圾解）。
        double t_max = 0.30;
        /// @brief 外参旋转上限（rad）。相机是**朝前**装在云台上的，所以相机光轴系到云台系的
        ///        固定旋转只能是几十度的安装偏差。模型里有 `(R,t)→(S·R,S·t)` 配 `yaw→-yaw`
        ///        的镜像对称（残差完全等价），只有这个物理先验能把它排除掉。
        double rot_max = 0.5;
    };

    /// @brief 标定结果。
    struct Fit
    {
        cv::Matx33d R = cv::Matx33d::eye();   // 相机 → 云台
        cv::Vec3d t{0.0, 0.0, 0.0};           // 米
        double yaw_scale = 1.0;
        double pitch_scale = 1.0;
        double residual_mm = 0.0;             // 拟合残差（板心散布）
        cv::Vec3d t_sigma{0.0, 0.0, 0.0};     // 平移三轴 1σ（米）
        double cross_validate_mm = -1.0;      // 半分交叉验证的检验残差（mm）
        bool ok = false;
    };

    /// @brief 一组样本在各组姿态下的"云台姿态"（世界←云台）。
    std::vector<cv::Matx33d> attitudesOf(const std::vector<Sample>& samples, double yaw_scale,
                                         double pitch_scale)
    {
        std::vector<cv::Matx33d> out;
        out.reserve(samples.size());
        for (const auto& sample : samples) {
            out.push_back(gimbalToBase(yaw_scale * sample.yaw, pitch_scale * sample.pitch));
        }
        return out;
    }

    /// @brief 板心投回底盘系（用给定外参）。
    std::vector<cv::Vec3d> projectToBase(const std::vector<Sample>& samples,
                                         const std::vector<cv::Matx33d>& attitudes,
                                         const cv::Matx33d& R, const cv::Vec3d& t)
    {
        std::vector<cv::Vec3d> out;
        out.reserve(samples.size());
        for (std::size_t i = 0; i < samples.size(); ++i) {
            const cv::Vec3d p_cam(samples[i].tvec);
            out.push_back(attitudes[i] * (R * p_cam + t));
        }
        return out;
    }

    /// @brief 散布（mm）：各点到中位点的平均距离。板静止时该项应为 0。
    double spreadMm(const std::vector<cv::Vec3d>& points, const cv::Vec3d& reference)
    {
        if (points.empty()) return 0.0;
        double sum = 0.0;
        for (const auto& point : points) sum += cv::norm(point - reference);
        return sum / static_cast<double>(points.size()) * 1000.0;
    }

    cv::Vec3d medianPoint(std::vector<cv::Vec3d> points)
    {
        if (points.empty()) return {0.0, 0.0, 0.0};
        auto mid = [](std::vector<double>& values) {
            std::sort(values.begin(), values.end());
            return values[values.size() / 2];
        };
        std::vector<double> xs, ys, zs;
        for (const auto& point : points) {
            xs.push_back(point[0]);
            ys.push_back(point[1]);
            zs.push_back(point[2]);
        }
        return {mid(xs), mid(ys), mid(zs)};
    }

    /// @brief 高斯-牛顿求解。参数：rotvec(3) + t(3) [+ yaw 标度] [+ pitch 标度]。
    bool solve(const std::vector<Sample>& samples, const FitOptions& options, Fit& fit)
    {
        if (samples.size() < 8) return false;
        const int kFree = 6 + (options.fit_yaw_scale ? 1 : 0) + (options.fit_pitch_scale ? 1 : 0);
        constexpr int kMax = 8;

        double params[kMax] = {0, 0, 0, 0, 0, 0, 1.0, 1.0};
        int yaw_index = options.fit_yaw_scale ? 6 : -1;
        int pitch_index = -1;
        if (options.fit_pitch_scale) {
            pitch_index = options.fit_yaw_scale ? 7 : 6;
        }

        auto rebuild = [&](const double* p) {
            cv::Matx33d R;
            cv::Rodrigues(cv::Vec3d(p[0], p[1], p[2]), R);
            const cv::Vec3d t(p[3], p[4], p[5]);
            const double ky = yaw_index >= 0 ? p[yaw_index] : 1.0;
            const double kp = pitch_index >= 0 ? p[pitch_index] : 1.0;
            return projectToBase(samples, attitudesOf(samples, ky, kp), R, t);
        };
        auto residuals = [&](const double* p, std::vector<double>& out) {
            const auto points = rebuild(p);
            const cv::Vec3d centre = medianPoint(points);
            out.clear();
            out.reserve(points.size() * 3);
            for (const auto& point : points) {
                const cv::Vec3d d = point - centre;
                out.push_back(d[0]);
                out.push_back(d[1]);
                out.push_back(d[2]);
            }
        };

        std::vector<double> r;
        residuals(params, r);
        double lambda = 1e-3;
        for (int iteration = 0; iteration < 120; ++iteration) {
            const int n = static_cast<int>(r.size());
            std::vector<double> jacobian(static_cast<std::size_t>(n) * kFree, 0.0);
            const double step = 1e-6;
            for (int k = 0; k < kFree; ++k) {
                double probe[kMax];
                for (int i = 0; i < kMax; ++i) probe[i] = params[i];
                probe[k] += step;
                std::vector<double> shifted;
                residuals(probe, shifted);
                for (int i = 0; i < n; ++i) {
                    jacobian[static_cast<std::size_t>(i) * kFree + k] = (shifted[i] - r[i]) / step;
                }
            }
            cv::Mat jtj = cv::Mat::zeros(kFree, kFree, CV_64F);
            cv::Mat jtr = cv::Mat::zeros(kFree, 1, CV_64F);
            for (int i = 0; i < n; ++i) {
                for (int a = 0; a < kFree; ++a) {
                    const double ja = jacobian[static_cast<std::size_t>(i) * kFree + a];
                    jtr.at<double>(a, 0) += ja * r[i];
                    for (int b = 0; b < kFree; ++b) {
                        jtj.at<double>(a, b) += ja * jacobian[static_cast<std::size_t>(i) * kFree + b];
                    }
                }
            }
            for (int a = 0; a < kFree; ++a) {
                jtj.at<double>(a, a) *= 1.0 + 1e-6;
                jtj.at<double>(a, a) += 1e-9;
            }
            cv::Mat delta;
            if (!cv::solve(jtj, -jtr, delta, cv::DECOMP_CHOLESKY)) break;
            double candidate[kMax];
            for (int i = 0; i < kMax; ++i) candidate[i] = params[i];
            for (int a = 0; a < kFree; ++a) candidate[a] += delta.at<double>(a, 0);
            // 平移硬约束：超过 t_max 就按比例压回边界（投影高斯-牛顿），把解锁在物理区域内
            const double t_norm = std::sqrt(candidate[3] * candidate[3] +
                                            candidate[4] * candidate[4] +
                                            candidate[5] * candidate[5]);
            if (t_norm > options.t_max && t_norm > 1e-12) {
                const double ratio = options.t_max / t_norm;
                candidate[3] *= ratio;
                candidate[4] *= ratio;
                candidate[5] *= ratio;
            }
            // 旋转也约束在物理范围内（见 FitOptions::rot_max 的说明）
            const double rot_norm = std::sqrt(candidate[0] * candidate[0] +
                                              candidate[1] * candidate[1] +
                                              candidate[2] * candidate[2]);
            if (rot_norm > options.rot_max && rot_norm > 1e-12) {
                const double ratio = options.rot_max / rot_norm;
                candidate[0] *= ratio;
                candidate[1] *= ratio;
                candidate[2] *= ratio;
            }

            std::vector<double> candidate_residuals;
            residuals(candidate, candidate_residuals);
            double squared = 0.0;
            for (double value : candidate_residuals) squared += value * value;
            if (squared < [&] { double s = 0.0; for (double value : r) s += value * value; return s; }()) {
                for (int i = 0; i < kMax; ++i) params[i] = candidate[i];
                residuals(params, r);
                lambda = std::max(lambda * 0.5, 1e-10);
            } else {
                lambda *= 5.0;
            }
            if (lambda > 1e6) break;
        }

        cv::Rodrigues(cv::Vec3d(params[0], params[1], params[2]), fit.R);
        fit.t = cv::Vec3d(params[3], params[4], params[5]);
        fit.yaw_scale = yaw_index >= 0 ? params[yaw_index] : 1.0;
        fit.pitch_scale = pitch_index >= 0 ? params[pitch_index] : 1.0;
        if (yaw_index >= 0 && std::abs(fit.yaw_scale) < 0.05) fit.yaw_scale = 0.05;   // 防退化

        const auto attitudes = attitudesOf(samples, fit.yaw_scale, fit.pitch_scale);
        const auto points = projectToBase(samples, attitudes, fit.R, fit.t);
        fit.residual_mm = spreadMm(points, medianPoint(points));

        // 平移可辨识度：σ²(JᵀJ)⁻¹ 的对角线（只取平移三轴）
        {
            const int n = static_cast<int>(r.size());
            std::vector<double> jacobian(static_cast<std::size_t>(n) * kFree, 0.0);
            const double step = 1e-6;
            for (int k = 0; k < kFree; ++k) {
                double probe[kMax];
                for (int i = 0; i < kMax; ++i) probe[i] = params[i];
                probe[k] += step;
                std::vector<double> shifted;
                residuals(probe, shifted);
                for (int i = 0; i < n; ++i) {
                    jacobian[static_cast<std::size_t>(i) * kFree + k] = (shifted[i] - r[i]) / step;
                }
            }
            cv::Mat jtj = cv::Mat::zeros(kFree, kFree, CV_64F);
            for (int i = 0; i < n; ++i) {
                for (int a = 0; a < kFree; ++a) {
                    const double ja = jacobian[static_cast<std::size_t>(i) * kFree + a];
                    for (int b = 0; b < kFree; ++b) {
                        jtj.at<double>(a, b) += ja * jacobian[static_cast<std::size_t>(i) * kFree + b];
                    }
                }
            }
            cv::Mat inverse;
            cv::invert(jtj, inverse, cv::DECOMP_SVD);
            // 秩检查：JᵀJ 接近奇异时伪逆会给出一堆假的很小的 σt（实测出现过 1e-8 mm 这种
            // 数字），此时直接把 σt 报成"不可辨识"。
            cv::SVD svd(jtj);
            const double smax = svd.w.at<double>(0);
            const double smin = svd.w.at<double>(svd.w.rows - 1);
            const bool rank_deficient = !(smax > 0.0) || (smin / smax) < 1e-8;
            const double sigma2 = fit.residual_mm * fit.residual_mm * 1e-6;   // mm² → m²
            for (int axis = 0; axis < 3; ++axis) {
                if (rank_deficient) {
                    fit.t_sigma[axis] = 999.0;
                    continue;
                }
                const double variance = sigma2 * inverse.at<double>(3 + axis, 3 + axis);
                fit.t_sigma[axis] = variance > 0.0 ? std::sqrt(variance) : 999.0;
            }
        }
        fit.ok = true;
        return true;
    }

    /// @brief yaw 标度扫描：固定若干候选 κ 各解一次（6 参数），看残差最小在哪。
    ///        比"把 κ 当自由参数"稳得多——自由参数会和旋转/平移互相补偿（实测会解成 -0.08
    ///        这种垃圾）；扫描里每个 κ 下的 R/t 都是最优的，残差曲线才说明问题。
    ///        返回使得残差最小的 κ。
    double scanYawScale(const std::vector<Sample>& samples, const FitOptions& options,
                        bool verbose)
    {
        static const double candidates[] = {-1.6, -1.3, -1.15, -1.05, -1.0, -0.95, -0.85,
                                            -0.7,  -0.5, -0.25,  0.25,  0.4,  0.5,  0.6,
                                             0.7,   0.8,  0.85,  0.9,  0.95, 1.0,  1.05,
                                             1.1,   1.25, 1.5,   2.0,  4.0};
        double best_scale = 1.0;
        double best_residual = 1e18;
        if (verbose) std::cout << "[hand_eye] yaw 标度扫描（残差 mm）：" << std::endl;
        for (double scale : candidates) {
            std::vector<Sample> scaled = samples;
            for (auto& sample : scaled) sample.yaw *= scale;
            FitOptions fixed = options;
            fixed.fit_yaw_scale = false;
            fixed.fit_pitch_scale = false;
            Fit fit;
            if (!solve(scaled, fixed, fit)) continue;
            if (verbose) {
                std::cout << "     κ=" << std::setw(5) << scale << " → 残差 " << fit.residual_mm
                          << " mm，|t|=" << cv::norm(fit.t) << " m" << std::endl;
            }
            if (fit.residual_mm < best_residual) {
                best_residual = fit.residual_mm;
                best_scale = scale;
            }
        }
        return best_scale;
    }

    /// @brief 半分交叉验证：奇偶各半互相拟合/检验，返回检验残差（mm）。
    ///        检验时用**训练集**的板心位置当参考点，所以它测的是"外参能不能推广到没见过的样本"。
    double crossValidate(const std::vector<Sample>& samples, const FitOptions& options,
                         double* in_sample_mm = nullptr)
    {
        if (samples.size() < 16) return -1.0;
        std::vector<Sample> even, odd;
        for (std::size_t i = 0; i < samples.size(); ++i) {
            (i % 2 == 0 ? even : odd).push_back(samples[i]);
        }
        double total = 0.0;
        double training_total = 0.0;
        int folds = 0;
        for (int fold = 0; fold < 2; ++fold) {
            const auto& train = fold == 0 ? even : odd;
            const auto& test = fold == 0 ? odd : even;
            Fit fit;
            if (!solve(train, options, fit)) continue;
            const auto train_attitudes = attitudesOf(train, fit.yaw_scale, fit.pitch_scale);
            const cv::Vec3d centre = medianPoint(projectToBase(train, train_attitudes, fit.R, fit.t));
            const auto test_attitudes = attitudesOf(test, fit.yaw_scale, fit.pitch_scale);
            total += spreadMm(projectToBase(test, test_attitudes, fit.R, fit.t), centre);
            training_total += fit.residual_mm;
            ++folds;
        }
        if (folds == 0) return -1.0;
        if (in_sample_mm != nullptr) *in_sample_mm = training_total / folds;
        return total / folds;
    }

    /// @brief 结果的物理合理性检查。返回空串表示通过。
    std::string sanityProblem(const std::vector<Sample>& samples, const Fit& fit)
    {
        double yaw_min = 1e9, yaw_max = -1e9, pitch_min = 1e9, pitch_max = -1e9;
        for (const auto& sample : samples) {
            yaw_min = std::min(yaw_min, sample.yaw);
            yaw_max = std::max(yaw_max, sample.yaw);
            pitch_min = std::min(pitch_min, sample.pitch);
            pitch_max = std::max(pitch_max, sample.pitch);
        }
        const double yaw_span = (yaw_max - yaw_min) * 180.0 / CV_PI;
        const double pitch_span = (pitch_max - pitch_min) * 180.0 / CV_PI;
        const double translation = cv::norm(fit.t);
        if (translation > 0.3) {
            std::ostringstream out;
            out << "解出的平移 " << translation << " m 远大于相机光心到云台旋转中心的实际距离（应 <0.3 m）";
            return out.str();
        }
        if (std::min(yaw_span, pitch_span) < 8.0) {
            std::ostringstream out;
            out << "采样姿态单薄：yaw 跨度 " << yaw_span << "°，pitch 跨度 " << pitch_span
                << "°（两个轴都要 ≥8°）";
            return out.str();
        }
        return {};
    }

    /// @brief 打印一次拟合的全部关键指标。
    void printFit(const std::vector<Sample>& samples, const Fit& fit, const std::string& tag)
    {
        const double translation = cv::norm(fit.t);
        std::cout << tag << "样本 " << samples.size() << " 组：残差 " << fit.residual_mm
                  << " mm，|t| = " << translation << " m，σt=(" << fit.t_sigma[0] * 1000.0 << ", "
                  << fit.t_sigma[1] * 1000.0 << ", " << fit.t_sigma[2] * 1000.0 << ") mm";
        if (fit.cross_validate_mm >= 0.0) {
            std::cout << "，交叉验证 " << fit.cross_validate_mm << " mm";
        }
        if (fit.yaw_scale != 1.0) {
            std::cout << "，yaw 修正系数 " << fit.yaw_scale << "（回传侧标度约 "
                      << 1.0 / fit.yaw_scale << "）";
        }
        if (fit.pitch_scale != 1.0) {
            std::cout << "，pitch 修正系数 " << fit.pitch_scale << "（回传侧标度约 "
                      << 1.0 / fit.pitch_scale << "）";
        }
        std::cout << std::endl;
        if (std::max({fit.t_sigma[0], fit.t_sigma[1], fit.t_sigma[2]}) > 0.1) {
            std::cout << "      ↑ 平移还不可辨识（σt>0.1 m）：把 yaw 扫得更宽、pitch 多分几档"
                      << std::endl;
        }
        if (fit.cross_validate_mm >= 0.0 &&
            fit.cross_validate_mm > std::max(20.0, 3.0 * fit.residual_mm)) {
            std::cout << "      ↑ 交叉验证明显比拟合差 → 这一版是过拟合解，别用" << std::endl;
        }
    }

    /// @brief 写结果 yaml（可直接粘到 configs/camera.yaml 的 hikcamera 段）。
    void writeYaml(const std::string& output, const std::vector<Sample>& samples, const Fit& fit,
                   const std::string& comment = {})
    {
        std::ofstream out(output);
        out << std::setprecision(10);
        out << "# 由 tools/hand_eye_calibrate 生成（" << samples.size() << " 组样本，残差 "
            << fit.residual_mm << " mm";
        if (fit.cross_validate_mm >= 0.0) out << "，交叉验证 " << fit.cross_validate_mm << " mm";
        out << "，|t| " << cv::norm(fit.t) << " m";
        if (fit.yaw_scale != 1.0) out << "，yaw 标度 " << fit.yaw_scale;
        if (fit.pitch_scale != 1.0) out << "，pitch 标度 " << fit.pitch_scale;
        out << "）—— 粘到 configs/camera.yaml 的 hikcamera 段\n";
        if (!comment.empty()) out << "# " << comment << "\n";
        out << "R_camera2gimbal: [";
        for (int row = 0; row < 3; ++row) {
            for (int column = 0; column < 3; ++column) {
                out << (row || column ? "," : "") << fit.R(row, column);
            }
        }
        out << "]\n";
        out << "t_camera2gimbal: [" << fit.t[0] << "," << fit.t[1] << "," << fit.t[2] << "]\n";
        std::cout << "[hand_eye] 已写 " << output << std::endl;
    }

    /// @brief 体检 + 交叉验证都通过才写盘。
    bool saveIfGood(const std::string& output, const std::vector<Sample>& samples, const Fit& fit)
    {
        const std::string problem = sanityProblem(samples, fit);
        if (!problem.empty()) {
            std::cout << "[hand_eye] **不写盘**：" << problem << std::endl;
            return false;
        }
        if (fit.residual_mm > 20.0) {
            std::cout << "[hand_eye] **不写盘**：残差 " << fit.residual_mm
                      << " mm 太大（目标 <10 mm；先查标定板平整度、内参、yaw 通道）" << std::endl;
            return false;
        }
        if (fit.cross_validate_mm >= 0.0 && fit.cross_validate_mm > std::max(20.0, 3.0 * fit.residual_mm)) {
            std::cout << "[hand_eye] **不写盘**：交叉验证 " << fit.cross_validate_mm
                      << " mm 远大于拟合残差 " << fit.residual_mm << " mm（过拟合）" << std::endl;
            return false;
        }
        writeYaml(output, samples, fit);
        return true;
    }

    /// @brief 运行参数。
    struct Options
    {
        int min_samples = 20;
        double min_step_deg = 3.0;
        double yaw_span_target = 25.0;
        double pitch_span_target = 12.0;
        std::string frame_dir;
        double yaw_scale_fixed = 1.0;   // --yaw-scale：给回传 yaw 乘一个固定标度（含符号）
        double max_rms = 0.8;           // --max-rms：单帧板重投影 rms 上限（px）
        FitOptions fit;
    };

    /// @brief 从 configs/serial.yaml 读 yaw 符号（+1/-1）。C 板编码器 yaw 的正方向如果和
    ///        tracker 约定（正 yaw = 右转）相反，就在这里填 -1 —— 工具和 tracker 共用这一项，
    ///        免得两边各拍一个符号、谁也说不清。
    double loadYawSign(const std::string& config_dir)
    {
        std::ifstream probe(config_dir + "/serial.yaml");
        if (!probe) return 1.0;
        const YAML::Node file = YAML::LoadFile(config_dir + "/serial.yaml");
        const YAML::Node serial = file["serial"] ? file["serial"] : file;
        return serial["yaw_sign"].as<double>(1.0) < 0.0 ? -1.0 : 1.0;
    }

#if defined(ULTRA_VISION_USE_HIK_CAMERA) || defined(ULTRA_VISION_USE_GALAXY_CAMERA)
    /// @brief 相机 + 串口 + 配置，各模式共用。
    struct Hardware
    {
        CameraType camera;
        std::unique_ptr<io::Gimbal> gimbal;
        cv::Mat camera_matrix;
        cv::Mat dist_coeffs;
        BoardConfig board;
        double yaw_sign = 1.0;
        bool ok = false;
    };

    /// @brief 打开相机 + 串口。相机/串口都不可拷贝，所以用出参就地构造。
    bool openHardware(Hardware& hardware, const std::string& config_dir)
    {
        const YAML::Node camera_file = YAML::LoadFile(config_dir + "/camera.yaml");
        const YAML::Node serial_file = YAML::LoadFile(config_dir + "/serial.yaml");
        const std::string camera_name = camera_file["camera"]["name"].as<std::string>("hikcamera");
        hardware.camera_matrix = auto_aim::readMatFromYaml(camera_file[camera_name]["camera_matrix"]);
        hardware.dist_coeffs = auto_aim::readMatFromYaml(camera_file[camera_name]["dist_coeffs"]);
        hardware.board = loadBoardConfig(config_dir);
        hardware.yaw_sign = loadYawSign(config_dir);
        if (!hardware.camera.init("", camera_file["camera"]["device_index"].as<int>(1))) {
            std::cerr << "[hand_eye] 打不开相机（被 MVS 客户端/别的进程占着？）" << std::endl;
            return false;
        }
        if (const YAML::Node hik = camera_file[camera_name]) {
            hardware.camera.setAutoExposure(hik["auto_exposure"].as<bool>(false));
            hardware.camera.setExposureTime(hik["exposure_ms"].as<double>(6.0) * 1000.0);
            hardware.camera.setGain(hik["gain"].as<double>(12.0));
        }
        hardware.gimbal = std::make_unique<io::Gimbal>(auto_aim::loadSerialConfig(serial_file));
        if (!hardware.gimbal->connected()) {
            std::cerr << "[hand_eye] 下位机链路没通：标定需要 C 板回传的 yaw/pitch" << std::endl;
            return false;
        }
        hardware.ok = true;
        return true;
    }

    /// @brief 一帧的观测量。
    struct Observation
    {
        bool found = false;
        bool full = false;
        double rms = -1.0;
        double tilt_deg = 0.0;
        double dist_m = 0.0;
        std::vector<cv::Point2f> corners;
        cv::Mat rvec;
        cv::Mat tvec;
        double yaw = 0.0;       // 解缠 + 符号修正后的 yaw（rad，用于求解）
        double yaw_raw = 0.0;   // **原始**回传 yaw（解缠后，未做符号修正）——存盘用这个，
                                // 离线复算再按 configs/serial.yaml 的 yaw_sign 应用一次，
                                // 避免"采集时取了反、存盘又取反"的双重取反
        double pitch = 0.0;
        bool have_pose = false;
    };

    /// @brief 检测标定板 + 取该帧时刻的云台角（yaw 已解缠）。
    Observation observeAt(Hardware& hardware, const cv::Mat& frame, FrameTime frame_time,
                          double yaw_reference)
    {
        Observation observation;
        // 姿态先取：没找到板的时候也要能显示"这一帧的云台角"（监控模式靠它看比值）
        io::ImuSample pose;
        if (hardware.gimbal->imuAt(frame_time, pose)) {
            observation.yaw_raw = unwrapYaw(pose.yaw, yaw_reference);
            observation.yaw = hardware.yaw_sign * observation.yaw_raw;
            observation.pitch = pose.pitch;
            observation.have_pose = true;
        }
        if (!detectBoard(frame, hardware.board, hardware.camera_matrix, hardware.dist_coeffs,
                         observation.corners, observation.rvec, observation.tvec,
                         &observation.rms)) {
            return observation;
        }
        observation.found = true;
        observation.full = true;
        for (const auto& corner : observation.corners) {
            if (corner.x < 20.0 || corner.y < 20.0 || corner.x > frame.cols - 20.0 ||
                corner.y > frame.rows - 20.0) {
                observation.full = false;
                break;
            }
        }
        cv::Mat rotation;
        cv::Rodrigues(observation.rvec, rotation);
        const cv::Mat normal = rotation * (cv::Mat_<double>(3, 1) << 0.0, 0.0, 1.0);
        observation.tilt_deg =
            std::acos(std::min(1.0, std::abs(normal.at<double>(2)))) * 180.0 / CV_PI;
        observation.dist_m = cv::norm(observation.tvec);
        return observation;
    }

    /// @brief 画预览 + 状态，返回按下的键（-1 = 没按）。
    int drawPreview(const cv::Mat& frame, const Observation& observation,
                    const std::vector<Sample>& samples, const Fit& fit, const Options& options,
                    double yaw_span, double pitch_span, const char* status)
    {
        cv::Mat display = frame.clone();
        for (const auto& corner : observation.corners) {
            cv::circle(display, corner, 4,
                       observation.full ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 165, 255), -1);
        }
        cv::putText(display,
                    cv::format("samples=%d  rms=%.2fpx  dist=%.2fm  tilt=%.0fdeg", 
                               static_cast<int>(samples.size()), observation.rms,
                               observation.dist_m, observation.tilt_deg),
                    cv::Point(16, 40), cv::FONT_HERSHEY_SIMPLEX, 0.9,
                    observation.found ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255), 2);
        cv::putText(display,
                    cv::format("yaw=%+.1f pitch=%+.1f  span %.1f/%.1f  %s",
                               observation.yaw * 180.0 / CV_PI, observation.pitch * 180.0 / CV_PI,
                               yaw_span, pitch_span, status),
                    cv::Point(16, 80), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 255), 2);
        if (fit.ok) {
            cv::putText(display,
                        cv::format("residual=%.1fmm  t=(%.3f,%.3f,%.3f)m  sigt=%.0fmm",
                                   fit.residual_mm, fit.t[0], fit.t[1], fit.t[2],
                                   std::max({fit.t_sigma[0], fit.t_sigma[1], fit.t_sigma[2]}) * 1000.0),
                        cv::Point(16, 120), cv::FONT_HERSHEY_SIMPLEX, 0.8,
                        cv::Scalar(255, 200, 0), 2);
        }
        cv::putText(display, "[space]=force sample  [s]=save  [u]=undo  [q]=quit",
                    cv::Point(16, 160), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(200, 200, 200),
                    2);
        cv::putText(display, "(click this window first, keys go to the focused window)",
                    cv::Point(16, 192), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(160, 160, 160), 2);
        cv::resize(display, display, cv::Size(), 0.5, 0.5);
        cv::imshow("hand_eye", display);
        (void)options;
        return cv::waitKey(1);
    }

    void spansOf(const std::vector<Sample>& samples, double& yaw_span, double& pitch_span)
    {
        yaw_span = 0.0;
        pitch_span = 0.0;
        if (samples.empty()) return;
        double yaw_min = 1e9, yaw_max = -1e9, pitch_min = 1e9, pitch_max = -1e9;
        for (const auto& sample : samples) {
            yaw_min = std::min(yaw_min, sample.yaw);
            yaw_max = std::max(yaw_max, sample.yaw);
            pitch_min = std::min(pitch_min, sample.pitch);
            pitch_max = std::max(pitch_max, sample.pitch);
        }
        yaw_span = (yaw_max - yaw_min) * 180.0 / CV_PI;
        pitch_span = (pitch_max - pitch_min) * 180.0 / CV_PI;
    }

    /// @brief 收尾：不管是怎么退出的（采够/按 q/按 ESC），都做一次解算 + 写盘。
    ///        上一版按 q 直接 break，什么都不输出（实测踩过：采了 64 组按 q 退出后没结果）。
    void finishRun(const std::string& output, const std::vector<Sample>& samples,
                   const Options& options, double yaw_span, double pitch_span)
    {
        std::cout << "[hand_eye] 收尾：共 " << samples.size() << " 组样本，覆盖 yaw " << yaw_span
                  << "° / pitch " << pitch_span << "°" << std::endl;
        if (samples.size() < 8) {
            std::cout << "[hand_eye] 样本不足 8 组，不出结果（样本帧仍在，可用目录模式离线复算）"
                      << std::endl;
            return;
        }
        Fit fit;
        if (!solve(samples, options.fit, fit)) return;
        double in_sample = 0.0;
        fit.cross_validate_mm = crossValidate(samples, options.fit, &in_sample);
        printFit(samples, fit, "[hand_eye] 最终解算：");
        std::cout << "[hand_eye] R_camera2gimbal = [";
        for (int row = 0; row < 3; ++row) {
            for (int column = 0; column < 3; ++column) {
                std::cout << (row || column ? "," : "") << fit.R(row, column);
            }
        }
        std::cout << "]" << std::endl;
        std::cout << "[hand_eye] t_camera2gimbal = [" << fit.t[0] << "," << fit.t[1] << ","
                  << fit.t[2] << "] m" << std::endl;
        saveIfGood(output, samples, fit);
    }

    /// @brief 自动采样：板子静止、云台转动，满足判据就记一组；采够覆盖自动解算+写盘。
    int collectAuto(const std::string& config_dir, const std::string& output, const Options& options)
    {
        Hardware hardware;
        if (!openHardware(hardware, config_dir)) return 1;
        // 默认也把样本帧存下来：万一中途退出/结果不达标，数据不会丢（可以用目录模式离线复算）。
        const std::string frame_dir = options.frame_dir.empty() ? "hand_eye_samples"
                                                                : options.frame_dir;
        std::system(("mkdir -p " + frame_dir).c_str());
        std::cout << "[hand_eye] 样本帧会存到 " << frame_dir << "/（{i}.jpg + {i}.yaml）"
                  << std::endl;

        std::cout << "[hand_eye] yaw 符号（configs/serial.yaml 的 yaw_sign）= " << hardware.yaw_sign
                  << std::endl;
        std::cout << "[hand_eye] 自动采样：标定板（" << hardware.board.pattern << " "
                  << hardware.board.cols << "x" << hardware.board.rows
                  << "）摆在画面里**别动**，然后慢慢转云台：yaw 扫一大段（跨度 ≥"
                  << options.yaw_span_target << "°）、pitch 分几档（≥" << options.pitch_span_target
                  << "°）。采够 " << options.min_samples << " 组自动收工；窗口提示 OK 时可移动。"
                  << "想手动补一组：点一下图像窗口再按空格（s=存盘，u=撤销，q=退出）。"
                  << std::endl;

        std::vector<Sample> samples;
        std::deque<cv::Vec3d> recent;
        constexpr int kRecent = 4;
        const auto session_start = std::chrono::steady_clock::now();
        double yaw_reference = 0.0;
        bool have_reference = false;
        Fit fit;
        auto last_status = std::chrono::steady_clock::now();

        while (true) {
            cv::Mat frame;
            FrameTime frame_time;
            if (!getImageTimed(hardware.camera, frame, frame_time, 500) || frame.empty()) continue;
            if (!have_reference) {
                io::ImuSample pose;
                if (hardware.gimbal->latestImu(pose)) {
                    yaw_reference = pose.yaw;
                    have_reference = true;
                }
                continue;
            }
            const Observation observation =
                observeAt(hardware, frame, frame_time, yaw_reference);

            // 静止判据：最近 4 帧板心抖动 <8 mm（板的位姿变化直接反映相机在不在动）
            double jitter_mm = 1e9;
            if (observation.found && observation.full) {
                recent.push_back(cv::Vec3d(observation.tvec));
                while (static_cast<int>(recent.size()) > kRecent) recent.pop_front();
                if (static_cast<int>(recent.size()) == kRecent) {
                    jitter_mm = 0.0;
                    for (const auto& point : recent) {
                        jitter_mm = std::max(jitter_mm, cv::norm(point - recent.back()) * 1000.0);
                    }
                }
            } else {
                recent.clear();
            }
            const bool still = jitter_mm < 8.0;
            const bool slow = observation.have_pose;

            // 新鲜度：和已采样本的 yaw/pitch 都差得够远
            double nearest_deg = 1e9;
            for (const auto& sample : samples) {
                const double dy = (sample.yaw - observation.yaw) * 180.0 / CV_PI;
                const double dp = (sample.pitch - observation.pitch) * 180.0 / CV_PI;
                nearest_deg = std::min(nearest_deg, std::hypot(dy, dp));
            }
            const bool novel = nearest_deg > options.min_step_deg;

            double yaw_span = 0.0, pitch_span = 0.0;
            spansOf(samples, yaw_span, pitch_span);
            const char* status = !observation.found    ? "NO BOARD"
                                  : !observation.full  ? "BOARD CUT"
                                  : observation.rms > options.max_rms ? "RMS HIGH"
                                  : !still             ? "MOVING"
                                  : !novel             ? "SAME POSE"
                                                       : "OK";
            const int key = drawPreview(frame, observation, samples, fit, options, yaw_span,
                                        pitch_span, status);
            if (key == 'q' || key == 27) {
                finishRun(output, samples, options, yaw_span, pitch_span);
                break;
            }
            if (key == 'u' && !samples.empty()) {
                samples.pop_back();
                std::cout << "[hand_eye] 撤销一组，剩 " << samples.size() << " 组" << std::endl;
                continue;
            }
            if (key == 's') {   // 手动存盘：立刻解算并写 hand_eye.yaml
                if (samples.size() >= 8 && solve(samples, options.fit, fit)) {
                    double in_sample = 0.0;
                    fit.cross_validate_mm = crossValidate(samples, options.fit, &in_sample);
                    printFit(samples, fit, "[hand_eye] 手动存盘：");
                    saveIfGood(output, samples, fit);
                } else {
                    std::cerr << "[hand_eye] 样本不足（需 ≥8 组）" << std::endl;
                }
                continue;
            }
            // 手动补采：空格强制记一组（跳过"静止/新鲜"判据），但板子必须完整、rms 别太离谱
            const bool force = key == ' ';
            const bool accept = observation.found && observation.full && slow &&
                                observation.rms < options.max_rms && novel && (force || still);
            if (accept) {
                Sample sample;
                sample.yaw = observation.yaw;
                sample.pitch = observation.pitch;
                sample.rvec = observation.rvec.clone();
                sample.tvec = observation.tvec.clone();
                sample.time_s = std::chrono::duration<double>(frame_time - session_start).count();
                samples.push_back(sample);
                std::cout << "[hand_eye] " << (force ? "手动" : "自动") << "采第 " << samples.size()
                          << " 组：yaw="
                          << sample.yaw * 180.0 / CV_PI << "° pitch="
                          << sample.pitch * 180.0 / CV_PI << "° rms=" << observation.rms
                          << "px dist=" << cv::norm(sample.tvec) << "m 覆盖(yaw " << yaw_span
                          << "° / pitch " << pitch_span << "°)" << std::endl;
                {
                    const std::string stem = frame_dir + "/" + std::to_string(samples.size());
                    cv::imwrite(stem + ".jpg", frame);
                    std::ofstream pose_out(stem + ".yaml");
                    pose_out << std::setprecision(10) << observation.yaw_raw << " "
                             << observation.pitch << " " << sample.time_s << "\n";
                }
                if (samples.size() >= 8) {
                    if (solve(samples, options.fit, fit)) {
                        double in_sample = 0.0;
                        fit.cross_validate_mm = crossValidate(samples, options.fit, &in_sample);
                        printFit(samples, fit, "[hand_eye] 即时解算：");
                        if (samples.size() >= 12 && samples.size() % 4 == 0) {
                            scanYawScale(samples, options.fit, false);
                        }
                        if (sanityProblem(samples, fit).empty() &&
                            samples.size() >= static_cast<std::size_t>(options.min_samples) &&
                            yaw_span >= options.yaw_span_target &&
                            pitch_span >= options.pitch_span_target && saveIfGood(output, samples, fit)) {
                            std::cout << "[hand_eye] 覆盖够 + 体检通过，标定结束。" << std::endl;
                            break;
                        }
                        if (samples.size() >= static_cast<std::size_t>(options.min_samples) &&
                            yaw_span >= options.yaw_span_target &&
                            pitch_span >= options.pitch_span_target) {
                            std::cout << "[hand_eye] 覆盖够了但这个解没过质检（见上面一行），"
                                         "继续采样；按 q 退出也会做一次收尾解算。" << std::endl;
                        }
                    }
                }
            } else {
                const auto now = std::chrono::steady_clock::now();
                if (std::chrono::duration<double>(now - last_status).count() >= 3.0) {
                    last_status = now;
                    std::cout << "[hand_eye] " << status << "  yaw="
                              << observation.yaw * 180.0 / CV_PI << "° pitch="
                              << observation.pitch * 180.0 / CV_PI << "° rms=" << observation.rms
                              << "px 抖动=" << jitter_mm << "mm 已采 " << samples.size() << " 组"
                              << std::endl;
                }
            }
        }
        return 0;
    }

    /// @brief 手动采样：空格采一组、s 存盘、u 撤销、q 退出。
    int collectLive(const std::string& config_dir, const std::string& output, const Options& options)
    {
        Hardware hardware;
        if (!openHardware(hardware, config_dir)) return 1;
        const std::string frame_dir =
            options.frame_dir.empty() ? "hand_eye_samples" : options.frame_dir;
        std::system(("mkdir -p " + frame_dir).c_str());
        std::vector<Sample> samples;
        double yaw_reference = 0.0;
        bool have_reference = false;
        Fit fit;
        double yaw_span = 0.0, pitch_span = 0.0;
        std::cout << "[hand_eye] 手动采样：先点一下图像窗口（按键只发给被聚焦的窗口），"
                     "然后 空格=采一组，s=存盘，u=撤销，q=退出" << std::endl;
        while (true) {
            cv::Mat frame;
            FrameTime frame_time;
            if (!getImageTimed(hardware.camera, frame, frame_time, 500) || frame.empty()) continue;
            if (!have_reference) {
                io::ImuSample pose;
                if (hardware.gimbal->latestImu(pose)) {
                    yaw_reference = pose.yaw;
                    have_reference = true;
                }
                continue;
            }
            const Observation observation = observeAt(hardware, frame, frame_time, yaw_reference);
            spansOf(samples, yaw_span, pitch_span);
            const char* status = !observation.found ? "NO BOARD"
                                  : !observation.full ? "BOARD CUT"
                                                      : "READY";
            const int key = drawPreview(frame, observation, samples, fit, options, yaw_span,
                                        pitch_span, status);
            if (key == 'q' || key == 27) {
                finishRun(output, samples, options, yaw_span, pitch_span);
                break;
            }
            if (key == 'u' && !samples.empty()) {
                samples.pop_back();
                std::cout << "[hand_eye] 撤销一组，剩 " << samples.size() << " 组" << std::endl;
                continue;
            }
            if (key == ' ' && observation.found && observation.full && observation.have_pose &&
                observation.rms < options.max_rms) {
                Sample sample;
                sample.yaw = observation.yaw;
                sample.pitch = observation.pitch;
                sample.rvec = observation.rvec.clone();
                sample.tvec = observation.tvec.clone();
                samples.push_back(sample);
                {
                    const std::string stem = frame_dir + "/" + std::to_string(samples.size());
                    cv::imwrite(stem + ".jpg", frame);
                    std::ofstream pose_out(stem + ".yaml");
                    pose_out << std::setprecision(10) << observation.yaw_raw << " "
                             << observation.pitch << " " << sample.time_s << "\n";
                }
                std::cout << "[hand_eye] " << "手动" << "采第 " << samples.size()
                          << " 组：yaw="
                          << sample.yaw * 180.0 / CV_PI << "° pitch="
                          << sample.pitch * 180.0 / CV_PI << "° rms=" << observation.rms << "px"
                          << std::endl;
                if (samples.size() >= 8 && solve(samples, options.fit, fit)) {
                    double in_sample = 0.0;
                    fit.cross_validate_mm = crossValidate(samples, options.fit, &in_sample);
                    printFit(samples, fit, "[hand_eye] 即时解算：");
                }
                continue;
            }
            if (key == 's') {
                if (samples.size() >= 8 && solve(samples, options.fit, fit)) {
                    double in_sample = 0.0;
                    fit.cross_validate_mm = crossValidate(samples, options.fit, &in_sample);
                    printFit(samples, fit, "[hand_eye] 解算：");
                    saveIfGood(output, samples, fit);
                } else {
                    std::cerr << "[hand_eye] 样本不足（需 ≥8 组）" << std::endl;
                }
            }
        }
        return 0;
    }

    /// @brief 对表模式：把"画面里测到的真实转角"和"回传角的变化"逐行对比，比值≈1 才说明回传角可信。
    int monitorLive(const std::string& config_dir)
    {
        Hardware hardware;
        if (!openHardware(hardware, config_dir)) return 1;
        std::cout << "#  时刻   回传yaw°  回传pitch°  四元数yaw°  板心x  板心y  距m   真实转角°  回传变化°  比值"
                  << std::endl;
        const auto start = std::chrono::steady_clock::now();
        double first_x = 0.0, first_yaw = 0.0, reference = 0.0;
        bool have_first = false;
        while (true) {
            cv::Mat frame;
            FrameTime frame_time;
            if (!getImageTimed(hardware.camera, frame, frame_time, 500) || frame.empty()) continue;
            io::ImuSample pose;
            if (!hardware.gimbal->imuAt(frame_time, pose)) continue;
            if (!have_first) {
                reference = pose.yaw;
                have_first = true;
            }
            const Observation observation = observeAt(hardware, frame, frame_time, reference);
            double quad_yaw = 0.0;
            {
                const double w = pose.w, qx = pose.x, qy = pose.y, qz = pose.z;
                const double r10 = 2.0 * (qx * qy + qz * w);
                const double r00 = 1.0 - 2.0 * (qy * qy + qz * qz);
                quad_yaw = std::atan2(r10, r00) * 180.0 / CV_PI;
            }
            const double elapsed =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            std::printf("%6.1fs %+9.2f %+11.2f %+11.2f", elapsed,
                        observation.yaw * 180.0 / CV_PI, observation.pitch * 180.0 / CV_PI,
                        quad_yaw);
            if (observation.found) {
                const double x = observation.corners.front().x;
                if (first_x == 0.0 && first_yaw == 0.0) {
                    first_x = x;
                    first_yaw = observation.yaw * 180.0 / CV_PI;
                }
                const double delta_x = x - first_x;
                const double true_deg =
                    std::atan2(delta_x, hardware.camera_matrix.at<double>(0, 0)) * 180.0 / CV_PI;
                const double reported = observation.yaw * 180.0 / CV_PI - first_yaw;
                std::printf(" %6.0f %6.0f %5.2f %11.2f %10.2f %6.2f", x, observation.corners.front().y,
                            observation.dist_m, true_deg, reported,
                            std::abs(reported) > 0.5 ? true_deg / reported : 0.0);
            } else {
                std::printf("  (no board)");
            }
            std::printf("\n");
            std::cout.flush();
            std::this_thread::sleep_for(std::chrono::milliseconds(400));
        }
        return 0;
    }
#endif

    /// @brief 合成数据自检：(a) 精确恢复；(b) yaw ±180 折返；(c) yaw 标度错误可被解出来。
    int selftest()
    {
        // 真实设置
        const cv::Matx33d R_true = [] {
            cv::Matx33d R;
            cv::Rodrigues(cv::Vec3d(0.03, -0.02, 0.15), R);
            return R;
        }();
        const cv::Vec3d t_true(0.03, -0.01, 0.05);
        const cv::Vec3d board(1.2, 0.35, -1.8);      // 靶标要离开 yaw 轴，否则绕 z 转不可观测

        auto makeSamples = [&](double yaw_scale, double yaw_offset, bool wrap) {
            std::vector<Sample> samples;
            for (int i = 0; i < 16; ++i) {
                const double yaw = (-14.0 + 28.0 * i / 15.0) * CV_PI / 180.0;
                const double pitch = (-8.0 + 16.0 * (i % 4) / 3.0) * CV_PI / 180.0;
                const cv::Matx33d attitude = gimbalToBase(yaw, pitch);
                Sample sample;
                sample.yaw = yaw_scale * yaw + yaw_offset;   // C 板回传的角（可能带标度/零偏）
                sample.pitch = pitch;
                if (wrap) sample.yaw = wrapPi(sample.yaw);   // 编码器在 ±180° 折返
                sample.tvec = (cv::Mat_<double>(3, 1) <<
                    (R_true.t() * (attitude.t() * board - t_true))[0],
                    (R_true.t() * (attitude.t() * board - t_true))[1],
                    (R_true.t() * (attitude.t() * board - t_true))[2]);
                samples.push_back(sample);
            }
            unwrapSamples(samples);
            return samples;
        };

        // (a) 基本恢复
        {
            auto samples = makeSamples(1.0, 0.0, false);
            Fit fit;
            if (!solve(samples, {}, fit)) return 1;
            const double translation_error = cv::norm(fit.t - t_true) * 1000.0;
            cv::Mat delta;
            cv::Rodrigues(fit.R * R_true.t(), delta);
            std::printf("自检(a) 基本恢复：残差 %.2f mm，平移误差 %.1f mm，转动误差 %.2f°，|t|=%.3f m\n",
                        fit.residual_mm, translation_error,
                        cv::norm(cv::Vec3d(delta)) * 180.0 / CV_PI, cv::norm(fit.t));
            if (fit.residual_mm > 1.0 || translation_error > 10.0) {
                std::printf("FAILED: 基本恢复没过\n");
                return 1;
            }
        }
        // (b) yaw ±180 折返
        {
            auto samples = makeSamples(1.0, 176.0 * CV_PI / 180.0, true);   // 让 yaw 跨越 ±180
            Fit fit;
            if (!solve(samples, {}, fit)) return 1;
            std::printf("自检(b) yaw 折返(偏移 176°)：残差 %.2f mm，|t|=%.3f m\n", fit.residual_mm,
                        cv::norm(fit.t));
            if (fit.residual_mm > 1.0) {
                std::printf("FAILED: 折返解缠没处理对\n");
                return 1;
            }
        }
        // (c) yaw 标度错误（0.8 倍）+ 把它解出来
        {
            auto samples = makeSamples(0.8, 0.0, false);
            Fit fit;
            FitOptions options;
            options.fit_yaw_scale = true;
            if (!solve(samples, options, fit)) return 1;
            // 注意语义：解出来的是**修正系数** κ（gimbalToBase(κ·回传角)）。回传侧少乘了 0.8，
            // 所以修正系数应当 ≈ 1/0.8 = 1.25；换成"回传侧实际标度"就是 1/κ ≈ 0.80。
            std::printf("自检(c) 回传 yaw 标度 0.8：修正系数解出 %.3f（=回传标度 %.3f），残差 %.2f mm\n",
                        fit.yaw_scale, 1.0 / fit.yaw_scale, fit.residual_mm);
            if (std::abs(fit.yaw_scale - 1.25) > 0.05 || fit.residual_mm > 1.0) {
                std::printf("FAILED: 标度没解出来\n");
                return 1;
            }
            // 不放开标度时应当解不出来（残差大）→ 说明这条诊断是有意义的
            Fit fixed;
            if (!solve(samples, {}, fixed)) return 1;
            std::printf("        不放开标度时残差 %.1f mm（应当明显变大）\n", fixed.residual_mm);
            if (fixed.residual_mm < 5.0) {
                std::printf("FAILED: 标度错误居然没被发现\n");
                return 1;
            }
        }
        std::printf("自检全部通过\n");
        return 0;
    }

}  // namespace

int main(int argc, char* argv[])
{
    cv::ocl::setUseOpenCL(false);   // macOS 上 OpenCL 缓存会刷屏；标定用不到

    std::string folder;
    std::string config_dir = "configs";
    std::string output = "hand_eye.yaml";
    Options options;
    bool live = false;
    bool auto_mode = false;
    bool monitor = false;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            std::cout << "用法:\n"
                         "  hand_eye_calibrate --auto                 自动采样（板子静止、云台转动，"
                         "采够自动解算+写盘）\n"
                         "  hand_eye_calibrate --live                 手动：空格采样 / s 存盘 /"
                         " u 撤销 / q 退出\n"
                         "  hand_eye_calibrate --monitor              对表：画面里的真实转角 vs 回传角\n"
                         "  hand_eye_calibrate <样本目录>              离线复算 + 交叉验证\n"
                         "  hand_eye_calibrate --selftest             合成数据自检\n"
                         "选项：\n"
                         "  --config-dir configs  --out hand_eye.yaml\n"
                         "  --min-samples 20 --min-step 3 --yaw-span 25 --pitch-span 12\n"
                         "  --save-frames <目录>      自动模式把样本帧存下来\n"
                         "  --yaw-scale <k>           给回传 yaw 乘固定标度 k（含符号；标度扫描会给出建议）\n"
                         "  --fit-yaw-scale           把 yaw 标度当未知量解（**噪声下会退化，"
                         "只建议在无噪合成数据上用**；查标度请用扫描）\n"
                         "  --fit-pitch-scale         同上（pitch）\n";
            return 0;
        }
        if (arg == "--selftest") return selftest();
        else if (arg == "--auto") auto_mode = true;
        else if (arg == "--live") live = true;
        else if (arg == "--monitor") monitor = true;
        else if (arg == "--yaw-scale" && i + 1 < argc) options.yaw_scale_fixed = std::atof(argv[++i]);
        else if (arg == "--max-rms" && i + 1 < argc) options.max_rms = std::atof(argv[++i]);
        else if (arg == "--fit-yaw-scale") options.fit.fit_yaw_scale = true;
        else if (arg == "--fit-pitch-scale") options.fit.fit_pitch_scale = true;
        else if (arg == "--min-samples" && i + 1 < argc) options.min_samples = std::atoi(argv[++i]);
        else if (arg == "--min-step" && i + 1 < argc) options.min_step_deg = std::atof(argv[++i]);
        else if (arg == "--yaw-span" && i + 1 < argc) options.yaw_span_target = std::atof(argv[++i]);
        else if (arg == "--pitch-span" && i + 1 < argc)
            options.pitch_span_target = std::atof(argv[++i]);
        else if (arg == "--save-frames" && i + 1 < argc) options.frame_dir = argv[++i];
        else if (arg == "--config-dir" && i + 1 < argc) config_dir = argv[++i];
        else if (arg == "--out" && i + 1 < argc) output = argv[++i];
        else if (folder.empty()) folder = arg;
    }

    if (monitor) {
#if defined(ULTRA_VISION_USE_HIK_CAMERA) || defined(ULTRA_VISION_USE_GALAXY_CAMERA)
        return monitorLive(config_dir);
#else
        std::cerr << "--monitor 需要带相机 SDK 的构建" << std::endl;
        return 2;
#endif
    }
    if (auto_mode || live) {
#if defined(ULTRA_VISION_USE_HIK_CAMERA) || defined(ULTRA_VISION_USE_GALAXY_CAMERA)
        return auto_mode ? collectAuto(config_dir, output, options)
                         : collectLive(config_dir, output, options);
#else
        std::cerr << "--auto/--live 需要带相机 SDK 的构建" << std::endl;
        return 2;
#endif
    }
    if (folder.empty()) {
        std::cerr << "给个样本目录，或 --auto / --live / --monitor / --selftest（--help 看用法）"
                  << std::endl;
        return 2;
    }

    // ---- 离线复算 ----
    const BoardConfig board = loadBoardConfig(config_dir);
    const YAML::Node camera_file = YAML::LoadFile(config_dir + "/camera.yaml");
    const std::string camera_name = camera_file["camera"]["name"].as<std::string>("hikcamera");
    const cv::Mat camera_matrix = auto_aim::readMatFromYaml(camera_file[camera_name]["camera_matrix"]);
    const cv::Mat dist_coeffs = auto_aim::readMatFromYaml(camera_file[camera_name]["dist_coeffs"]);

    std::vector<Sample> samples;
    for (int index = 1;; ++index) {
        const std::string image_path = folder + "/" + std::to_string(index) + ".jpg";
        const std::string pose_path = folder + "/" + std::to_string(index) + ".yaml";
        if (!std::ifstream(image_path).good()) break;
        const cv::Mat image = cv::imread(image_path);
        if (image.empty()) break;
        std::ifstream pose_in(pose_path);
        if (!pose_in) {
            std::cerr << "[hand_eye] 缺 " << pose_path << "（自动模式会一起存）" << std::endl;
            break;
        }
        Sample sample;
        pose_in >> sample.yaw >> sample.pitch;
        if (!(pose_in >> sample.time_s)) sample.time_s = 0.0;
        std::vector<cv::Point2f> corners;
        double rms = -1.0;
        if (!detectBoard(image, board, camera_matrix, dist_coeffs, corners, sample.rvec, sample.tvec,
                         &rms)) {
            std::cout << "[hand_eye] 第 " << index << " 张没找到标定板，跳过" << std::endl;
            continue;
        }
        std::cout << "[hand_eye] 第 " << index << " 张：板 " << corners.size()
                  << " 点，重投影 rms=" << rms << " px，距离 " << cv::norm(sample.tvec) << " m"
                  << std::endl;
        samples.push_back(sample);
    }
    if (samples.size() < 8) {
        std::cerr << "[hand_eye] 有效样本 " << samples.size() << " 组（<8 组解不稳）" << std::endl;
        return 1;
    }
    unwrapSamples(samples);
    const double yaw_sign = loadYawSign(config_dir);
    if (yaw_sign < 0.0) {
        for (auto& sample : samples) sample.yaw = -sample.yaw;
        std::cout << "[hand_eye] 已按 configs/serial.yaml 的 yaw_sign = -1 取反 yaw" << std::endl;
    }
    if (options.yaw_scale_fixed != 1.0) {
        for (auto& sample : samples) sample.yaw *= options.yaw_scale_fixed;
        std::cout << "[hand_eye] 已按 --yaw-scale " << options.yaw_scale_fixed
                  << " 缩放 yaw（符号/标度修正）" << std::endl;
    }

    const double best_scale = scanYawScale(samples, options.fit, true);
    if (std::abs(best_scale - 1.0) > 0.05) {
        std::cout << "[hand_eye] 注意：yaw 标度最优值 " << best_scale
                  << "（不是 1.0）→ C 板回传的 yaw 有 ≈" << 1.0 / best_scale
                  << " 倍的标度问题；下面先用 κ=1 解，要按最优值解就再跑一次并乘上它" << std::endl;
    }

    Fit fit;
    if (!solve(samples, options.fit, fit)) return 1;
    double in_sample = 0.0;
    fit.cross_validate_mm = crossValidate(samples, options.fit, &in_sample);
    printFit(samples, fit, "[hand_eye] 解算：");
    std::cout << "[hand_eye] R_camera2gimbal = [";
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            std::cout << (row || column ? "," : "") << fit.R(row, column);
        }
    }
    std::cout << "]" << std::endl;
    std::cout << "[hand_eye] t_camera2gimbal = [" << fit.t[0] << "," << fit.t[1] << "," << fit.t[2]
              << "] m" << std::endl;
    saveIfGood(output, samples, fit);
    return 0;
}
