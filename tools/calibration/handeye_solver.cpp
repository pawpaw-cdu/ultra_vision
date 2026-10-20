#include "tools/calibration/handeye_solver.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>

#include <opencv2/calib3d.hpp>

#include "tools/calibration/pose.hpp"

namespace calibration
{
    namespace
    {
        std::vector<cv::Matx33d> attitudesOf(const std::vector<Sample>& samples,
                                             double yaw_scale, double pitch_scale)
        {
            std::vector<cv::Matx33d> out;
            out.reserve(samples.size());
            for (const auto& sample : samples) {
                out.push_back(gimbalToBase(yaw_scale * sample.yaw,
                                           pitch_scale * sample.pitch));
            }
            return out;
        }

        std::vector<cv::Vec3d> projectToBase(const std::vector<Sample>& samples,
                                             const std::vector<cv::Matx33d>& attitudes,
                                             const cv::Matx33d& R, const cv::Vec3d& t)
        {
            std::vector<cv::Vec3d> out;
            out.reserve(samples.size());
            for (std::size_t i = 0; i < samples.size(); ++i) {
                out.push_back(attitudes[i] * (R * cv::Vec3d(samples[i].tvec) + t));
            }
            return out;
        }

        double spreadMm(const std::vector<cv::Vec3d>& points,
                        const cv::Vec3d& reference)
        {
            if (points.empty()) return 0.0;
            double sum = 0.0;
            for (const cv::Vec3d& point : points) sum += cv::norm(point - reference);
            return sum / static_cast<double>(points.size()) * 1000.0;
        }

        cv::Vec3d medianPoint(std::vector<cv::Vec3d> points)
        {
            if (points.empty()) return {0.0, 0.0, 0.0};
            const auto mid = [](std::vector<double>& values) {
                std::sort(values.begin(), values.end());
                return values[values.size() / 2];
            };
            std::vector<double> xs, ys, zs;
            for (const cv::Vec3d& point : points) {
                xs.push_back(point[0]);
                ys.push_back(point[1]);
                zs.push_back(point[2]);
            }
            return {mid(xs), mid(ys), mid(zs)};
        }
    } // namespace

    double wrapPi(double value)
    {
        while (value > CV_PI) value -= 2.0 * CV_PI;
        while (value <= -CV_PI) value += 2.0 * CV_PI;
        return value;
    }

    double unwrapYaw(double yaw, double reference)
    {
        double out = yaw;
        while (out - reference > CV_PI) out -= 2.0 * CV_PI;
        while (out - reference < -CV_PI) out += 2.0 * CV_PI;
        return out;
    }

    void unwrapSamples(std::vector<Sample>& samples)
    {
        if (samples.empty()) return;
        const double reference = samples.front().yaw;
        for (Sample& sample : samples) sample.yaw = unwrapYaw(sample.yaw, reference);
    }

    bool solve(const std::vector<Sample>& samples, const FitOptions& options, Fit& fit)
    {
        if (samples.size() < 8) return false;
        const int free_parameters =
            6 + (options.fit_yaw_scale ? 1 : 0) + (options.fit_pitch_scale ? 1 : 0);
        constexpr int kMaxParameters = 8;
        double params[kMaxParameters] = {0, 0, 0, 0, 0, 0, 1.0, 1.0};
        const int yaw_index = options.fit_yaw_scale ? 6 : -1;
        int pitch_index = -1;
        if (options.fit_pitch_scale) pitch_index = options.fit_yaw_scale ? 7 : 6;

        const auto rebuild = [&](const double* p) {
            cv::Matx33d R;
            cv::Rodrigues(cv::Vec3d(p[0], p[1], p[2]), R);
            const cv::Vec3d t(p[3], p[4], p[5]);
            const double yaw_scale = yaw_index >= 0 ? p[yaw_index] : 1.0;
            const double pitch_scale = pitch_index >= 0 ? p[pitch_index] : 1.0;
            return projectToBase(
                samples, attitudesOf(samples, yaw_scale, pitch_scale), R, t);
        };
        const auto residuals = [&](const double* p, std::vector<double>& out) {
            const auto points = rebuild(p);
            const cv::Vec3d centre = medianPoint(points);
            out.clear();
            out.reserve(points.size() * 3);
            for (const cv::Vec3d& point : points) {
                const cv::Vec3d delta = point - centre;
                out.push_back(delta[0]);
                out.push_back(delta[1]);
                out.push_back(delta[2]);
            }
        };

        std::vector<double> residual;
        residuals(params, residual);
        double lambda = 1e-3;
        for (int iteration = 0; iteration < 120; ++iteration) {
            const int n = static_cast<int>(residual.size());
            std::vector<double> jacobian(
                static_cast<std::size_t>(n) * free_parameters, 0.0);
            constexpr double kStep = 1e-6;
            for (int k = 0; k < free_parameters; ++k) {
                double probe[kMaxParameters];
                for (int i = 0; i < kMaxParameters; ++i) probe[i] = params[i];
                probe[k] += kStep;
                std::vector<double> shifted;
                residuals(probe, shifted);
                for (int i = 0; i < n; ++i) {
                    jacobian[static_cast<std::size_t>(i) * free_parameters + k] =
                        (shifted[i] - residual[i]) / kStep;
                }
            }
            cv::Mat jtj = cv::Mat::zeros(free_parameters, free_parameters, CV_64F);
            cv::Mat jtr = cv::Mat::zeros(free_parameters, 1, CV_64F);
            for (int i = 0; i < n; ++i) {
                for (int a = 0; a < free_parameters; ++a) {
                    const double ja =
                        jacobian[static_cast<std::size_t>(i) * free_parameters + a];
                    jtr.at<double>(a, 0) += ja * residual[i];
                    for (int b = 0; b < free_parameters; ++b) {
                        jtj.at<double>(a, b) +=
                            ja * jacobian[static_cast<std::size_t>(i) *
                                              free_parameters +
                                          b];
                    }
                }
            }
            for (int a = 0; a < free_parameters; ++a) {
                jtj.at<double>(a, a) *= 1.0 + 1e-6;
                jtj.at<double>(a, a) += 1e-9;
            }
            cv::Mat delta;
            if (!cv::solve(jtj, -jtr, delta, cv::DECOMP_CHOLESKY)) break;
            double candidate[kMaxParameters];
            for (int i = 0; i < kMaxParameters; ++i) candidate[i] = params[i];
            for (int a = 0; a < free_parameters; ++a) {
                candidate[a] += delta.at<double>(a, 0);
            }
            const double t_norm = std::sqrt(candidate[3] * candidate[3] +
                                            candidate[4] * candidate[4] +
                                            candidate[5] * candidate[5]);
            if (t_norm > options.t_max && t_norm > 1e-12) {
                const double ratio = options.t_max / t_norm;
                candidate[3] *= ratio;
                candidate[4] *= ratio;
                candidate[5] *= ratio;
            }
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
            double candidate_squared = 0.0;
            double current_squared = 0.0;
            for (double value : candidate_residuals) candidate_squared += value * value;
            for (double value : residual) current_squared += value * value;
            if (candidate_squared < current_squared) {
                for (int i = 0; i < kMaxParameters; ++i) params[i] = candidate[i];
                residuals(params, residual);
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
        if (yaw_index >= 0 && std::abs(fit.yaw_scale) < 0.05) fit.yaw_scale = 0.05;
        const auto attitudes = attitudesOf(samples, fit.yaw_scale, fit.pitch_scale);
        fit.residual_mm =
            spreadMm(projectToBase(samples, attitudes, fit.R, fit.t),
                     medianPoint(projectToBase(samples, attitudes, fit.R, fit.t)));

        const int n = static_cast<int>(residual.size());
        std::vector<double> jacobian(
            static_cast<std::size_t>(n) * free_parameters, 0.0);
        constexpr double kStep = 1e-6;
        for (int k = 0; k < free_parameters; ++k) {
            double probe[kMaxParameters];
            for (int i = 0; i < kMaxParameters; ++i) probe[i] = params[i];
            probe[k] += kStep;
            std::vector<double> shifted;
            residuals(probe, shifted);
            for (int i = 0; i < n; ++i) {
                jacobian[static_cast<std::size_t>(i) * free_parameters + k] =
                    (shifted[i] - residual[i]) / kStep;
            }
        }
        cv::Mat jtj = cv::Mat::zeros(free_parameters, free_parameters, CV_64F);
        for (int i = 0; i < n; ++i) {
            for (int a = 0; a < free_parameters; ++a) {
                const double ja =
                    jacobian[static_cast<std::size_t>(i) * free_parameters + a];
                for (int b = 0; b < free_parameters; ++b) {
                    jtj.at<double>(a, b) +=
                        ja * jacobian[static_cast<std::size_t>(i) * free_parameters + b];
                }
            }
        }
        cv::Mat inverse;
        cv::invert(jtj, inverse, cv::DECOMP_SVD);
        cv::SVD svd(jtj);
        const double largest = svd.w.at<double>(0);
        const double smallest = svd.w.at<double>(svd.w.rows - 1);
        const bool rank_deficient =
            !(largest > 0.0) || (smallest / largest) < 1e-8;
        const double sigma2 = fit.residual_mm * fit.residual_mm * 1e-6;
        for (int axis = 0; axis < 3; ++axis) {
            if (rank_deficient) {
                fit.t_sigma[axis] = 999.0;
                continue;
            }
            const double variance = sigma2 * inverse.at<double>(3 + axis, 3 + axis);
            fit.t_sigma[axis] = variance > 0.0 ? std::sqrt(variance) : 999.0;
        }
        fit.ok = true;
        return true;
    }

    double scanYawScale(const std::vector<Sample>& samples, const FitOptions& options,
                        bool verbose)
    {
        static const double candidates[] = {
            -1.6, -1.3, -1.15, -1.05, -1.0, -0.95, -0.85, -0.7, -0.5,
            -0.25, 0.25, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95,
            1.0, 1.05, 1.1, 1.25, 1.5, 2.0, 4.0};
        double best_scale = 1.0;
        double best_residual = 1e18;
        if (verbose) std::cout << "[hand_eye] yaw 标度扫描（残差 mm）：" << std::endl;
        for (double scale : candidates) {
            std::vector<Sample> scaled = samples;
            for (Sample& sample : scaled) sample.yaw *= scale;
            FitOptions fixed = options;
            fixed.fit_yaw_scale = false;
            fixed.fit_pitch_scale = false;
            Fit fit;
            if (!solve(scaled, fixed, fit)) continue;
            if (verbose) {
                std::cout << "     κ=" << std::setw(5) << scale << " → 残差 "
                          << fit.residual_mm << " mm，|t|=" << cv::norm(fit.t)
                          << " m" << std::endl;
            }
            if (fit.residual_mm < best_residual) {
                best_residual = fit.residual_mm;
                best_scale = scale;
            }
        }
        return best_scale;
    }

    double crossValidate(const std::vector<Sample>& samples,
                         const FitOptions& options, double* in_sample_mm)
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
            const auto train_attitudes =
                attitudesOf(train, fit.yaw_scale, fit.pitch_scale);
            const cv::Vec3d centre = medianPoint(
                projectToBase(train, train_attitudes, fit.R, fit.t));
            const auto test_attitudes =
                attitudesOf(test, fit.yaw_scale, fit.pitch_scale);
            total += spreadMm(
                projectToBase(test, test_attitudes, fit.R, fit.t), centre);
            training_total += fit.residual_mm;
            ++folds;
        }
        if (folds == 0) return -1.0;
        if (in_sample_mm != nullptr) *in_sample_mm = training_total / folds;
        return total / folds;
    }

    std::string sanityProblem(const std::vector<Sample>& samples, const Fit& fit)
    {
        double yaw_min = 1e9, yaw_max = -1e9;
        double pitch_min = 1e9, pitch_max = -1e9;
        for (const Sample& sample : samples) {
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
            out << "解出的平移 " << translation
                << " m 远大于相机光心到云台旋转中心的实际距离（应 <0.3 m）";
            return out.str();
        }
        if (std::min(yaw_span, pitch_span) < 8.0) {
            std::ostringstream out;
            out << "采样姿态单薄：yaw 跨度 " << yaw_span << "°，pitch 跨度 "
                << pitch_span << "°（两个轴都要 ≥8°）";
            return out.str();
        }
        return {};
    }

    void printFit(const std::vector<Sample>& samples, const Fit& fit,
                  const std::string& tag)
    {
        const double translation = cv::norm(fit.t);
        std::cout << tag << "样本 " << samples.size() << " 组：残差 "
                  << fit.residual_mm << " mm，|t| = " << translation << " m，σt=("
                  << fit.t_sigma[0] * 1000.0 << ", " << fit.t_sigma[1] * 1000.0
                  << ", " << fit.t_sigma[2] * 1000.0 << ") mm";
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
            std::cout << "      ↑ 平移还不可辨识（σt>0.1 m）：把 yaw 扫得更宽、"
                         "pitch 多分几档"
                      << std::endl;
        }
        if (fit.cross_validate_mm >= 0.0 &&
            fit.cross_validate_mm > std::max(20.0, 3.0 * fit.residual_mm)) {
            std::cout << "      ↑ 交叉验证明显比拟合差 → 这一版是过拟合解，别用"
                      << std::endl;
        }
    }

    void writeYaml(const std::string& output, const std::vector<Sample>& samples,
                   const Fit& fit, const std::string& comment)
    {
        std::ofstream out(output);
        out << std::setprecision(10);
        out << "# 由 tools/hand_eye_calibrate 生成（" << samples.size()
            << " 组样本，残差 " << fit.residual_mm << " mm";
        if (fit.cross_validate_mm >= 0.0) {
            out << "，交叉验证 " << fit.cross_validate_mm << " mm";
        }
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
        out << "t_camera2gimbal: [" << fit.t[0] << "," << fit.t[1] << ","
            << fit.t[2] << "]\n";
        std::cout << "[hand_eye] 已写 " << output << std::endl;
    }

    bool saveIfGood(const std::string& output, const std::vector<Sample>& samples,
                    const Fit& fit)
    {
        const std::string problem = sanityProblem(samples, fit);
        if (!problem.empty()) {
            std::cout << "[hand_eye] **不写盘**：" << problem << std::endl;
            return false;
        }
        if (fit.residual_mm > 20.0) {
            std::cout << "[hand_eye] **不写盘**：残差 " << fit.residual_mm
                      << " mm 太大（目标 <10 mm；先查标定板平整度、内参、yaw 通道）"
                      << std::endl;
            return false;
        }
        if (fit.cross_validate_mm >= 0.0 &&
            fit.cross_validate_mm > std::max(20.0, 3.0 * fit.residual_mm)) {
            std::cout << "[hand_eye] **不写盘**：交叉验证 "
                      << fit.cross_validate_mm << " mm 远大于拟合残差 "
                      << fit.residual_mm << " mm（过拟合）" << std::endl;
            return false;
        }
        writeYaml(output, samples, fit);
        return true;
    }
} // namespace calibration
