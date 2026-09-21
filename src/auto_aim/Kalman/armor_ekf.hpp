#ifndef AUTO_AIM_ARMOR_EKF_HPP
#define AUTO_AIM_ARMOR_EKF_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <utility>
#include <vector>

namespace auto_aim {

// Lightweight fixed-size matrix used by the armor EKF.
template <int R, int C>
class Matrix {
public:
    std::array<std::array<double, C>, R> data;

    Matrix() {
        for (auto& row : data) row.fill(0.0);
    }

    double& operator()(int i, int j) { return data[i][j]; }
    const double& operator()(int i, int j) const { return data[i][j]; }

    template <int OC>
    Matrix<R, OC> operator*(const Matrix<C, OC>& other) const {
        Matrix<R, OC> result;
        for (int i = 0; i < R; ++i) {
            for (int j = 0; j < OC; ++j) {
                for (int k = 0; k < C; ++k) {
                    result(i, j) += (*this)(i, k) * other(k, j);
                }
            }
        }
        return result;
    }

    Matrix<R, C> operator+(const Matrix<R, C>& other) const {
        Matrix<R, C> result;
        for (int i = 0; i < R; ++i) {
            for (int j = 0; j < C; ++j) result(i, j) = (*this)(i, j) + other(i, j);
        }
        return result;
    }

    Matrix<R, C> operator-(const Matrix<R, C>& other) const {
        Matrix<R, C> result;
        for (int i = 0; i < R; ++i) {
            for (int j = 0; j < C; ++j) result(i, j) = (*this)(i, j) - other(i, j);
        }
        return result;
    }

    Matrix<R, C> operator*(double scalar) const {
        Matrix<R, C> result;
        for (int i = 0; i < R; ++i) {
            for (int j = 0; j < C; ++j) result(i, j) = (*this)(i, j) * scalar;
        }
        return result;
    }

    Matrix<C, R> transpose() const {
        Matrix<C, R> result;
        for (int i = 0; i < R; ++i) {
            for (int j = 0; j < C; ++j) result(j, i) = (*this)(i, j);
        }
        return result;
    }

    static Matrix<3, 3> inv3(const Matrix<3, 3>& matrix) {
        double det = matrix(0, 0) * (matrix(1, 1) * matrix(2, 2) - matrix(1, 2) * matrix(2, 1))
                   - matrix(0, 1) * (matrix(1, 0) * matrix(2, 2) - matrix(1, 2) * matrix(2, 0))
                   + matrix(0, 2) * (matrix(1, 0) * matrix(2, 1) - matrix(1, 1) * matrix(2, 0));
        Matrix<3, 3> inv;
        inv(0, 0) = (matrix(1, 1) * matrix(2, 2) - matrix(1, 2) * matrix(2, 1)) / det;
        inv(0, 1) = (matrix(0, 2) * matrix(2, 1) - matrix(0, 1) * matrix(2, 2)) / det;
        inv(0, 2) = (matrix(0, 1) * matrix(1, 2) - matrix(0, 2) * matrix(1, 1)) / det;
        inv(1, 0) = (matrix(1, 2) * matrix(2, 0) - matrix(1, 0) * matrix(2, 2)) / det;
        inv(1, 1) = (matrix(0, 0) * matrix(2, 2) - matrix(0, 2) * matrix(2, 0)) / det;
        inv(1, 2) = (matrix(0, 2) * matrix(1, 0) - matrix(0, 0) * matrix(1, 2)) / det;
        inv(2, 0) = (matrix(1, 0) * matrix(2, 1) - matrix(1, 1) * matrix(2, 0)) / det;
        inv(2, 1) = (matrix(0, 1) * matrix(2, 0) - matrix(0, 0) * matrix(2, 1)) / det;
        inv(2, 2) = (matrix(0, 0) * matrix(1, 1) - matrix(0, 1) * matrix(1, 0)) / det;
        return inv;
    }

    static bool invert(const Matrix<R, R>& matrix, Matrix<R, R>& inverse) {
        auto a = matrix.data;
        inverse = Matrix<R, R>();
        for (int i = 0; i < R; ++i) inverse(i, i) = 1.0;

        for (int column = 0; column < R; ++column) {
            int pivot = column;
            for (int row = column + 1; row < R; ++row) {
                if (std::abs(a[row][column]) > std::abs(a[pivot][column])) pivot = row;
            }
            if (std::abs(a[pivot][column]) < 1e-12) return false;

            if (pivot != column) {
                std::swap(a[pivot], a[column]);
                std::swap(inverse.data[pivot], inverse.data[column]);
            }

            const double diagonal = a[column][column];
            for (int col = 0; col < R; ++col) {
                a[column][col] /= diagonal;
            }
            for (int col = 0; col < R; ++col) {
                inverse(column, col) /= diagonal;
            }

            for (int row = 0; row < R; ++row) {
                if (row == column) continue;
                const double factor = a[row][column];
                if (std::abs(factor) < 1e-15) continue;
                for (int col = 0; col < R; ++col) {
                    a[row][col] -= factor * a[column][col];
                    inverse(row, col) -= factor * inverse(column, col);
                }
            }
        }
        return true;
    }
};

template <int N>
using Vec = Matrix<N, 1>;

// YPD measurement: bearing yaw, pitch, range and the armor radial angle.
// Noise values are variances in rad^2, rad^2, m^2 and rad^2 respectively.
struct YpdObservation {
    int plate_id = -1;
    double yaw = 0.0;
    double pitch = 0.0;
    double distance = 0.0;
    double armor_yaw = 0.0;
    double yaw_noise = 2.5e-3;
    double pitch_noise = 5.0e-3;
    double distance_noise = 1.0e-2;
    double armor_yaw_noise = 4.0e-2;
};

// UV（像素）观测的相机模型：内参 + 世界→相机旋转。
// 相机在 world 原点（与 Tracker::worldToCameraVector 同源：world_to_camera 就是
// 该旋转），所以重投影只需要一次旋转 + 针孔投影。
struct UvCamera {
    double fx = 1.0;
    double fy = 1.0;
    double cx = 0.0;
    double cy = 0.0;
    Matrix<3, 3> world_to_camera;
};

// UV 观测：直接把装甲板**四个角点像素**放进滤波器，观测函数写重投影。
//
// 与 YPD 观测的区别（这是加它的原因）：
//   * YPD 是"PnP 先解一次最小二乘 → 把 yaw/pitch/distance/armor_yaw 当量测"。
//     PnP 会把像素噪声**放大且相关**（尤其是距离：小靶面正对相机时距离是病态方向），
//     而我们只能给一个近似的对角线噪声去描述它；
//   * UV 只做一次最小二乘（EKF 自己），量测噪声就是像素噪声（~1 px，好标定、无相关），
//     四个角点的几何约束按重投影原样进入滤波器。
// 代价：雅可比不再是解析式（这里用中心差分，9 维状态 × 每点 2 次投影，成本可忽略），
// 而且要求 world→camera 的位姿与观测是同一时刻的（Tracker 已经保证）。
struct UvObservation {
    int plate_id = -1;
    // 顺序与 perception/pnp_solver.cpp 的 getArmor3DPoints 一致：
    // left_bottom, left_top, right_top, right_bottom。
    std::array<Vec<2>, 4> corners{};
    double half_width = 0.067;    // 小装甲板 0.134/2（大板用 0.225/2）
    double half_height = 0.0285;  // 0.057/2
    double sigma_px = 1.5;        // 角点像素噪声（含检测偏差，按实测标定）
};

// 紧凑 UV 观测（一个灯条的像素 → 4 维），思路借 awakening-main 的
// `points_to_observation()`：
//     delta  = top - bottom
//     angle  = atan2(delta.x, delta.y)      （灯条在图像里的方向）
//     center = (top + bottom) / 2
//     length = |delta|
// 比"四角点 8 维"更省也更稳：
//   * **没有"哪个角对应哪个物点"的镜像二义**（四角点版本要靠重投影残差搜 4 种配对）；
//   * 三个量各自对应一个物理方向：角度=朝向、长度=尺度/距离、中心=位置，
//     噪声可以按量分别标定（角度 rad、其余 px），不用假设 8 个像素同分布。
// 端点按图像上下取（top 是 v 更小的那个），所以角度天然落在 ±90° 附近，
// 不需要处理灯条"上下颠倒"。
struct UvCompactObservation {
    int plate_id = -1;
    double angle = 0.0;       // atan2(dx, dy)，与参考实现一致
    double center_x = 0.0;
    double center_y = 0.0;
    double length = 0.0;
    double half_width = 0.067;    // 与 PnP 段一致，用于预测
    double half_height = 0.0285;
    double sigma_px = 3.0;    // 中心/长度的像素噪声
    double sigma_angle = 0.02;
};

// State: [x_c, vx, y_c, vy, sin(theta), cos(theta), omega, z_c, vz].
// The four armor plates are distributed on a circle in the x-z plane.
class ArmorEKF {
public:
    struct Config {
        double R = 0.21;
        double process_acc = 1.0;
        // 平移自适应的过程噪声（2026-09-26 加的）。过程噪声 `process_acc` 物理上
        // 是"底盘加速度的不确定度"：靶车静止时它是 0，横移/急停时才有值。
        // 常数取大了 → 静止/近距时把观测噪声当运动吸进来（实测 1 m 近距工况
        // FIRING 从 81/115/82 掉到 16/15/17）；取小了 → 平移时速度估计跟不上
        // 阶跃（横向误差 p10/p90 从 ±10 cm 涨到 -19/+24 cm）。
        // 所以按下式随**估计速度**自适应放大，静止时回到 process_acc：
        //   qa = clamp(process_acc · (1 + (|v|/speed_ref)²), process_acc, process_acc_max)
        double process_acc_speed_ref = 0.28;   // 米/秒，速度尺度
        double process_acc_max = 40.0;         // 上限，防止速度估计野值时爆掉
        double process_omega = 2.0;
        double meas_noise = 0.01;
        double y_noise_mult = 30.0;
        double P_min = 1e-3;
        // Vertical armor-center offsets in the chassis/home frame. Real robot
        // armor sets normally leave these at zero; simulator models can define
        // plate-specific offsets instead of forcing one shared y coordinate.
        std::array<double, 4> armor_y_offsets{{0.0, 0.0, 0.0, 0.0}};
    };

    ArmorEKF() { initMatrices(); }

    void setConfig(const Config& config) {
        cfg_ = config;
        initMatrices();
        reset();
    }

    void setDt(double dt) {
        // 每个周期都重建：Q 里现在含"当前速度"（平移自适应过程噪声），
        // 只在 dt 变化时重建会让它停留在旧速度上。9x9 赋值，代价可忽略。
        cfg_dt_ = dt;
        buildQ();
    }

    void reset() {
        x_ = Vec<9>();
        P_ = Matrix<9, 9>();
        for (int i = 0; i < 9; ++i) P_(i, i) = 1.0;
        initialized_ = false;
    }

    static double angleForPlate(int plate_id) {
        return (1 - plate_id) * 3.14159265358979323846 / 2.0;
    }

    static double normalizeAngle(double angle) {
        return std::atan2(std::sin(angle), std::cos(angle));
    }

    void initFromQuad(const std::vector<std::pair<int, Vec<3>>>& observations) {
        double sum_x = 0.0;
        double sum_y = 0.0;
        double sum_z = 0.0;
        int count = 0;
        for (const auto& [plate_id, observation] : observations) {
            double angle = angleForPlate(plate_id);
            sum_x += observation(0, 0) - cfg_.R * std::cos(angle);
            sum_y += observation(1, 0) - cfg_.armor_y_offsets[plate_id];
            sum_z += observation(2, 0) - cfg_.R * std::sin(angle);
            ++count;
        }
        if (count == 0) return;

        x_(0, 0) = sum_x / count;
        x_(2, 0) = sum_y / count;
        x_(7, 0) = sum_z / count;
        x_(4, 0) = 0.0;
        x_(5, 0) = 1.0;
        x_(6, 0) = 0.0;
        for (int i = 0; i < 9; ++i) {
            for (int j = 0; j < 9; ++j) P_(i, j) = 0.0;
        }
        P_(0, 0) = 1e-2; P_(2, 2) = 1e-2; P_(7, 7) = 1e-2;
        P_(1, 1) = 1.0;  P_(3, 3) = 1.0;  P_(8, 8) = 1.0;
        P_(4, 4) = 0.4; P_(5, 5) = 0.4;
        P_(6, 6) = 100.0;
        initialized_ = true;
    }

    void initAt(const Vec<3>& center, double yaw) {
        x_ = Vec<9>();
        x_(0, 0) = center(0, 0);
        x_(2, 0) = center(1, 0);
        x_(7, 0) = center(2, 0);
        x_(4, 0) = std::sin(yaw);
        x_(5, 0) = std::cos(yaw);

        for (int i = 0; i < 9; ++i) {
            for (int j = 0; j < 9; ++j) P_(i, j) = 0.0;
        }
        P_(0, 0) = 0.01; P_(2, 2) = 0.02; P_(7, 7) = 0.01;
        P_(1, 1) = 1.0;  P_(3, 3) = 1.0;  P_(8, 8) = 1.0;
        P_(4, 4) = 0.4; P_(5, 5) = 0.4;
        P_(6, 6) = 100.0;
        initialized_ = true;
    }

    void init(const Vec<3>& observation, int plate_id) {
        double angle = angleForPlate(plate_id);
        x_(0, 0) = observation(0, 0) - cfg_.R * std::cos(angle);
        x_(2, 0) = observation(1, 0) - cfg_.armor_y_offsets[plate_id];
        x_(7, 0) = observation(2, 0) - cfg_.R * std::sin(angle);
        x_(4, 0) = 0.0;
        x_(5, 0) = 1.0;
        x_(6, 0) = 0.0;
        for (int i = 0; i < 9; ++i) {
            for (int j = 0; j < 9; ++j) P_(i, j) = 0.0;
        }
        P_(0, 0) = 0.1; P_(2, 2) = 0.1; P_(7, 7) = 0.1;
        P_(1, 1) = 1.0; P_(3, 3) = 1.0; P_(8, 8) = 1.0;
        P_(4, 4) = 0.4; P_(5, 5) = 0.4;
        P_(6, 6) = 100.0;
        initialized_ = true;
    }

    void predict() {
        const double dt = cfg_dt_;
        x_(0, 0) += x_(1, 0) * dt;
        x_(2, 0) += x_(3, 0) * dt;
        x_(7, 0) += x_(8, 0) * dt;

        const double omega = x_(6, 0);
        const double sin_theta = x_(4, 0);
        const double cos_theta = x_(5, 0);
        const double cos_delta = std::cos(omega * dt);
        const double sin_delta = std::sin(omega * dt);
        const double sin_new = sin_theta * cos_delta + cos_theta * sin_delta;
        const double cos_new = cos_theta * cos_delta - sin_theta * sin_delta;
        x_(4, 0) = sin_new;
        x_(5, 0) = cos_new;

        Matrix<9, 9> F;
        for (int i = 0; i < 9; ++i) F(i, i) = 1.0;
        F(0, 1) = dt; F(2, 3) = dt; F(7, 8) = dt;
        F(4, 4) = cos_delta; F(4, 5) = sin_delta; F(4, 6) = cos_new * dt;
        F(5, 4) = -sin_delta; F(5, 5) = cos_delta; F(5, 6) = -sin_new * dt;

        P_ = F * P_ * F.transpose() + Q_;
    }

    YpdObservation predictYpd(int plate_id) const {
        const Vec<3> position = armorPosition(plate_id);
        const double horizontal = std::hypot(position(0, 0), position(2, 0));

        YpdObservation measurement;
        measurement.plate_id = plate_id;
        measurement.yaw = std::atan2(position(0, 0), position(2, 0));
        measurement.pitch = std::atan2(position(1, 0), horizontal);
        measurement.distance = std::sqrt(
            position(0, 0) * position(0, 0) +
            position(1, 0) * position(1, 0) +
            position(2, 0) * position(2, 0));
        measurement.armor_yaw = getPlateAngle(plate_id);
        return measurement;
    }

    double updateYpd(
        const YpdObservation& observation,
        double nis_threshold = std::numeric_limits<double>::infinity()) {
        std::vector<YpdObservation> observations{observation};
        return updateYpdBatch(observations, nis_threshold);
    }

    double updateYpdBatch(
        const std::vector<YpdObservation>& observations,
        double nis_threshold = std::numeric_limits<double>::infinity()) {
        if (observations.empty()) return 0.0;

        if (!initialized_) {
            const YpdObservation& first = observations.front();
            if (first.plate_id < 0 || first.plate_id >= 4) return 0.0;
            const double horizontal = first.distance * std::cos(first.pitch);
            Vec<3> position;
            position(0, 0) = horizontal * std::sin(first.yaw);
            position(1, 0) = first.distance * std::sin(first.pitch);
            position(2, 0) = horizontal * std::cos(first.yaw);
            const double yaw = normalizeAngle(
                first.armor_yaw - angleForPlate(first.plate_id));
            Vec<3> center;
            center(0, 0) = position(0, 0) - cfg_.R * std::cos(
                yaw + angleForPlate(first.plate_id));
            center(1, 0) = position(1, 0) -
                cfg_.armor_y_offsets[first.plate_id];
            center(2, 0) = position(2, 0) - cfg_.R * std::sin(
                yaw + angleForPlate(first.plate_id));
            initAt(center, yaw);
            return 1.0;
        }

        std::vector<YpdObservation> accepted;
        std::vector<Vec<4>> residuals;
        std::vector<Matrix<4, 9>> jacobians;
        accepted.reserve(observations.size());
        residuals.reserve(observations.size());
        jacobians.reserve(observations.size());

        double min_weight = 1.0;
        for (const auto& observation : observations) {
            if (observation.plate_id < 0 || observation.plate_id >= 4) continue;
            if (!std::isfinite(observation.yaw) || !std::isfinite(observation.pitch) ||
                !std::isfinite(observation.distance) || !std::isfinite(observation.armor_yaw) ||
                observation.distance <= 1e-4) {
                continue;
            }

            const YpdObservation predicted = predictYpd(observation.plate_id);
            const Vec<4> residual = ypdResidual(observation, predicted);
            const Matrix<4, 9> jacobian = ypdJacobian(observation.plate_id);
            Matrix<4, 4> noise;
            noise(0, 0) = std::max(observation.yaw_noise, 1e-8);
            noise(1, 1) = std::max(observation.pitch_noise, 1e-8);
            noise(2, 2) = std::max(observation.distance_noise, 1e-8);
            noise(3, 3) = std::max(observation.armor_yaw_noise, 1e-8);

            const Matrix<4, 4> innovation =
                jacobian * P_ * jacobian.transpose() + noise;
            Matrix<4, 4> innovation_inverse;
            if (!Matrix<4, 4>::invert(innovation, innovation_inverse)) continue;

            double nis = 0.0;
            for (int row = 0; row < 4; ++row) {
                for (int column = 0; column < 4; ++column) {
                    nis += residual(row, 0) * innovation_inverse(row, column) *
                        residual(column, 0);
                }
            }
            if (nis > nis_threshold) continue;

            min_weight = std::min(min_weight, std::exp(-0.5 * nis));
            accepted.push_back(observation);
            residuals.push_back(residual);
            jacobians.push_back(jacobian);
        }

        if (accepted.empty()) return 0.0;

        const int measurement_dim = 4 * static_cast<int>(accepted.size());
        std::vector<std::vector<double>> H(
            measurement_dim, std::vector<double>(9, 0.0));
        std::vector<double> residual(measurement_dim, 0.0);
        std::vector<double> noise_diagonal(measurement_dim, 0.0);

        for (std::size_t observation_index = 0;
             observation_index < accepted.size(); ++observation_index) {
            const int row_offset = 4 * static_cast<int>(observation_index);
            for (int row = 0; row < 4; ++row) {
                for (int column = 0; column < 9; ++column) {
                    H[row_offset + row][column] = jacobians[observation_index](row, column);
                }
                residual[row_offset + row] = residuals[observation_index](row, 0);
            }
            const YpdObservation& observation = accepted[observation_index];
            noise_diagonal[row_offset + 0] = std::max(observation.yaw_noise, 1e-8);
            noise_diagonal[row_offset + 1] = std::max(observation.pitch_noise, 1e-8);
            noise_diagonal[row_offset + 2] = std::max(observation.distance_noise, 1e-8);
            noise_diagonal[row_offset + 3] = std::max(observation.armor_yaw_noise, 1e-8);
        }

        std::vector<std::vector<double>> innovation(
            measurement_dim, std::vector<double>(measurement_dim, 0.0));
        for (int row = 0; row < measurement_dim; ++row) {
            for (int column = 0; column < measurement_dim; ++column) {
                double value = 0.0;
                for (int state_row = 0; state_row < 9; ++state_row) {
                    for (int state_column = 0; state_column < 9; ++state_column) {
                        value += H[row][state_row] * P_(state_row, state_column) *
                            H[column][state_column];
                    }
                }
                innovation[row][column] = value;
            }
            innovation[row][row] += noise_diagonal[row];
        }

        std::vector<std::vector<double>> innovation_inverse;
        if (!invertDynamic(innovation, innovation_inverse)) return 0.0;

        std::vector<std::vector<double>> state_measurement_covariance(
            9, std::vector<double>(measurement_dim, 0.0));
        for (int state_row = 0; state_row < 9; ++state_row) {
            for (int measurement = 0; measurement < measurement_dim; ++measurement) {
                double value = 0.0;
                for (int state_column = 0; state_column < 9; ++state_column) {
                    value += P_(state_row, state_column) * H[measurement][state_column];
                }
                state_measurement_covariance[state_row][measurement] = value;
            }
        }

        std::vector<std::vector<double>> gain(9, std::vector<double>(measurement_dim, 0.0));
        for (int state_row = 0; state_row < 9; ++state_row) {
            for (int output_measurement = 0; output_measurement < measurement_dim;
                 ++output_measurement) {
                for (int input_measurement = 0; input_measurement < measurement_dim;
                     ++input_measurement) {
                    gain[state_row][output_measurement] +=
                        state_measurement_covariance[state_row][input_measurement] *
                        innovation_inverse[input_measurement][output_measurement];
                }
            }
        }

        for (int state_row = 0; state_row < 9; ++state_row) {
            for (int measurement = 0; measurement < measurement_dim; ++measurement) {
                x_(state_row, 0) += gain[state_row][measurement] * residual[measurement];
            }
        }

        std::vector<std::vector<double>> transform(
            9, std::vector<double>(9, 0.0));
        for (int row = 0; row < 9; ++row) {
            transform[row][row] = 1.0;
            for (int column = 0; column < 9; ++column) {
                for (int measurement = 0; measurement < measurement_dim; ++measurement) {
                    transform[row][column] -= gain[row][measurement] * H[measurement][column];
                }
            }
        }

        std::vector<std::vector<double>> covariance(9, std::vector<double>(9, 0.0));
        for (int row = 0; row < 9; ++row) {
            for (int column = 0; column < 9; ++column) {
                double value = 0.0;
                for (int i = 0; i < 9; ++i) {
                    for (int j = 0; j < 9; ++j) {
                        value += transform[row][i] * P_(i, j) * transform[column][j];
                    }
                }
                covariance[row][column] = value;
            }
        }

        for (int row = 0; row < 9; ++row) {
            for (int column = 0; column < 9; ++column) {
                double value = 0.0;
                for (int measurement = 0; measurement < measurement_dim; ++measurement) {
                    value += gain[row][measurement] * noise_diagonal[measurement] *
                        gain[column][measurement];
                }
                covariance[row][column] += value;
            }
        }

        for (int row = 0; row < 9; ++row) {
            for (int column = 0; column < 9; ++column) {
                P_(row, column) = covariance[row][column];
            }
            if (P_(row, row) < cfg_.P_min) P_(row, row) = cfg_.P_min;
        }

        const double norm = std::sqrt(x_(4, 0) * x_(4, 0) + x_(5, 0) * x_(5, 0));
        if (norm > 1e-6) {
            x_(4, 0) /= norm;
            x_(5, 0) /= norm;
        }
        if (std::getenv("ULTRA_VISION_UV_DEBUG") != nullptr) {
            // 重投影残差的 RMS：这是 UV 观测"到底喂进去多少像素误差"的唯一直接
            // 指标。正常应当是 1~3 px（角点噪声 + 模型偏差）；到十几 px 说明
            // 板面几何/配对错了，滤波器只会学到一个错的状态。
            double sum = 0.0;
            int count = 0;
            for (const auto& residual : residuals) {
                for (int row = 0; row < 8; ++row) {
                    sum += residual(row, 0) * residual(row, 0);
                    ++count;
                }
            }
            std::cerr << "[uv] plates=" << accepted.size()
                      << " residual_rms=" << std::sqrt(sum / std::max(1, count))
                      << " px weight=" << min_weight << std::endl;
        }
        return min_weight;
    }

    double update(
        const Vec<3>& observation, int plate_id,
        double nis_threshold = std::numeric_limits<double>::infinity()) {
        if (!initialized_) {
            init(observation, plate_id);
            return 1.0;
        }

        const double angle = angleForPlate(plate_id);
        const double cos_plate = std::cos(angle);
        const double sin_plate = std::sin(angle);
        const double sin_theta = x_(4, 0);
        const double cos_theta = x_(5, 0);
        const double sin_total = sin_theta * cos_plate + cos_theta * sin_plate;
        const double cos_total = cos_theta * cos_plate - sin_theta * sin_plate;

        Vec<3> predicted;
        predicted(0, 0) = x_(0, 0) + cfg_.R * cos_total;
        predicted(1, 0) = x_(2, 0) + cfg_.armor_y_offsets[plate_id];
        predicted(2, 0) = x_(7, 0) + cfg_.R * sin_total;

        Matrix<3, 9> H;
        // theta is represented by [sin(theta), cos(theta)] in the state, so
        // the position Jacobian must be taken with respect to those variables.
        H(0, 0) = 1.0; H(0, 4) = -cfg_.R * sin_plate; H(0, 5) = cfg_.R * cos_plate;
        H(1, 2) = 1.0;
        H(2, 7) = 1.0; H(2, 4) = cfg_.R * cos_plate;   H(2, 5) = cfg_.R * sin_plate;

        const Vec<3> residual = observation - predicted;
        Matrix<3, 3> R;
        R(0, 0) = cfg_.meas_noise;
        R(1, 1) = cfg_.meas_noise * cfg_.y_noise_mult;
        R(2, 2) = cfg_.meas_noise;

        Matrix<3, 3> S = H * P_ * H.transpose() + R;
        Matrix<3, 3> S_inv = Matrix<3, 3>::inv3(S);
        double distance_sq = 0.0;
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                distance_sq += residual(i, 0) * S_inv(i, j) * residual(j, 0);
            }
        }
        const double weight = std::exp(-0.5 * distance_sq);

        // A wrong plate assignment can look plausible in image space but be
        // inconsistent with the current chassis covariance. Reject it before
        // it contaminates the estimate.
        if (distance_sq > nis_threshold) return 0.0;

        Matrix<9, 3> K = P_ * H.transpose() * S_inv;
        x_ = x_ + K * residual;

        Matrix<9, 9> I;
        for (int i = 0; i < 9; ++i) I(i, i) = 1.0;
        Matrix<9, 9> IKH = I - K * H;
        P_ = IKH * P_ * IKH.transpose() + K * R * K.transpose();
        for (int i = 0; i < 9; ++i) {
            if (P_(i, i) < cfg_.P_min) P_(i, i) = cfg_.P_min;
        }

        const double norm = std::sqrt(x_(4, 0) * x_(4, 0) + x_(5, 0) * x_(5, 0));
        if (norm > 1e-6) {
            x_(4, 0) /= norm;
            x_(5, 0) /= norm;
        }
        return weight;
    }

    double updateAngle(
        double observed_angle, int plate_id, double noise,
        double nis_threshold = std::numeric_limits<double>::infinity()) {
        if (!initialized_) return 0.0;

        const double plate_angle = angleForPlate(plate_id);
        const double sin_plate = std::sin(plate_angle);
        const double cos_plate = std::cos(plate_angle);
        const double sin_total = x_(4, 0) * cos_plate + x_(5, 0) * sin_plate;
        const double cos_total = x_(5, 0) * cos_plate - x_(4, 0) * sin_plate;
        const double predicted_angle = std::atan2(sin_total, cos_total);
        const double residual_angle = normalizeAngle(observed_angle - predicted_angle);

        // d(atan2(s, c)) / ds = c, d(atan2(s, c)) / dc = -s.
        Matrix<1, 9> H;
        H(0, 4) = cos_total;
        H(0, 5) = -sin_total;

        const Matrix<1, 1> S = H * P_ * H.transpose();
        const double innovation_variance = S(0, 0) + noise;
        if (innovation_variance < 1e-9) return 0.0;

        const double nis = residual_angle * residual_angle / innovation_variance;
        if (nis > nis_threshold) return 0.0;

        const Matrix<9, 1> K = P_ * H.transpose() * (1.0 / innovation_variance);
        for (int i = 0; i < 9; ++i) {
            x_(i, 0) += K(i, 0) * residual_angle;
        }

        Matrix<9, 9> I;
        for (int i = 0; i < 9; ++i) I(i, i) = 1.0;
        const Matrix<9, 9> IKH = I - K * H;
        P_ = IKH * P_ * IKH.transpose() + K * K.transpose() * noise;
        for (int i = 0; i < 9; ++i) {
            if (P_(i, i) < cfg_.P_min) P_(i, i) = cfg_.P_min;
        }

        const double norm = std::sqrt(x_(4, 0) * x_(4, 0) + x_(5, 0) * x_(5, 0));
        if (norm > 1e-6) {
            x_(4, 0) /= norm;
            x_(5, 0) /= norm;
        }
        return std::exp(-0.5 * nis);
    }

    // Scalar update of the chassis yaw rate.
    //
    // The rate is already a filter state and the prediction step rotates
    // [sin(theta), cos(theta)] by omega * dt, so the sequence of armor_yaw
    // measurements carries rate information through the P(omega, sin/cos)
    // cross covariance. An external radial-angle estimator is fused here as a
    // gated measurement instead of overwriting the state: overwriting also
    // zeroed that cross covariance, which stopped the filter from ever learning
    // the rate on its own and made the estimate collapse whenever the external
    // estimator lost validity for a frame.
    double updateOmega(
        double observed_omega, double noise,
        double nis_threshold = std::numeric_limits<double>::infinity()) {
        if (!initialized_ || !std::isfinite(observed_omega) ||
            !std::isfinite(noise) || noise <= 0.0) {
            return 0.0;
        }

        Matrix<1, 9> H;
        H(0, 6) = 1.0;

        const Matrix<1, 1> S = H * P_ * H.transpose();
        const double innovation_variance = S(0, 0) + noise;
        if (innovation_variance < 1e-12) return 0.0;

        const double residual = observed_omega - x_(6, 0);
        const double nis = residual * residual / innovation_variance;
        if (nis > nis_threshold) return 0.0;

        const Matrix<9, 1> K =
            P_ * H.transpose() * (1.0 / innovation_variance);
        for (int i = 0; i < 9; ++i) {
            x_(i, 0) += K(i, 0) * residual;
        }

        Matrix<9, 9> I;
        for (int i = 0; i < 9; ++i) I(i, i) = 1.0;
        const Matrix<9, 9> IKH = I - K * H;
        P_ = IKH * P_ * IKH.transpose() + K * K.transpose() * noise;
        return std::exp(-0.5 * nis);
    }

    double updateCenter(
        const Vec<3>& observation, double noise, double y_noise_mult,
        double nis_threshold = std::numeric_limits<double>::infinity()) {
        if (!initialized_) return 0.0;

        Matrix<3, 9> H;
        H(0, 0) = 1.0;
        H(1, 2) = 1.0;
        H(2, 7) = 1.0;

        const Vec<3> predicted = getCenter();
        const Vec<3> residual = observation - predicted;
        Matrix<3, 3> R_noise;
        R_noise(0, 0) = noise;
        R_noise(1, 1) = noise * y_noise_mult;
        R_noise(2, 2) = noise;

        Matrix<3, 3> S = H * P_ * H.transpose() + R_noise;
        Matrix<3, 3> S_inv = Matrix<3, 3>::inv3(S);
        double distance_sq = 0.0;
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                distance_sq += residual(i, 0) * S_inv(i, j) * residual(j, 0);
            }
        }
        if (distance_sq > nis_threshold) return 0.0;

        Matrix<9, 3> K = P_ * H.transpose() * S_inv;
        x_ = x_ + K * residual;

        Matrix<9, 9> I;
        for (int i = 0; i < 9; ++i) I(i, i) = 1.0;
        Matrix<9, 9> IKH = I - K * H;
        P_ = IKH * P_ * IKH.transpose() + K * R_noise * K.transpose();

        const double norm = std::sqrt(x_(4, 0) * x_(4, 0) + x_(5, 0) * x_(5, 0));
        if (norm > 1e-6) {
            x_(4, 0) /= norm;
            x_(5, 0) /= norm;
        }
        return std::exp(-0.5 * distance_sq);
    }

    Vec<9> getState() const { return x_; }
    bool isInit() const { return initialized_; }
    double getYaw() const { return std::atan2(x_(4, 0), x_(5, 0)); }
    double getOmega() const { return x_(6, 0); }

    // Hard state injection. It discards the P(omega, sin/cos) cross covariance,
    // so it is not suitable for fusing a rate estimate; use updateOmega() for
    // that. It remains only for callers that genuinely re-seed the state.
    void setAngularVelocity(double omega, double covariance = 0.04) {
        if (!initialized_ || !std::isfinite(omega)) return;
        x_(6, 0) = omega;
        P_(6, 6) = std::max(cfg_.P_min, covariance);
        for (int i = 0; i < 9; ++i) {
            if (i == 6) continue;
            P_(i, 6) = 0.0;
            P_(6, i) = 0.0;
        }
    }

    double getPlateAngle(int plate_id) const {
        return normalizeAngle(getYaw() + angleForPlate(plate_id));
    }

    Vec<3> getCenter() const {
        Vec<3> center;
        center(0, 0) = x_(0, 0);
        center(1, 0) = x_(2, 0);
        center(2, 0) = x_(7, 0);
        return center;
    }

    Vec<3> armorPosition(int plate_id) const {
        const double angle = angleForPlate(plate_id);
        const double cos_plate = std::cos(angle);
        const double sin_plate = std::sin(angle);
        const double sin_theta = x_(4, 0);
        const double cos_theta = x_(5, 0);

        Vec<3> position;
        position(0, 0) = x_(0, 0) + cfg_.R * (cos_theta * cos_plate - sin_theta * sin_plate);
        position(1, 0) = x_(2, 0) + cfg_.armor_y_offsets[plate_id];
        position(2, 0) = x_(7, 0) + cfg_.R * (sin_theta * cos_plate + cos_theta * sin_plate);
        return position;
    }

    /// @brief 给定状态，算某块装甲板四个角点**在 world 系里的位置**。
    ///        板面几何与 YPD 模型同一套：圆心在底盘中心 + R·[cos φ, 0, sin φ]（φ = θ + α），
    ///        法向是径向，竖直方向 = world 的 +y，切向 = up × radial。
    ///        物点定义与 getArmor3DPoints 一致（y = 板面横向、z = 竖直），
    ///        所以角点顺序同样是 left_bottom, left_top, right_top, right_bottom。
    static std::array<Vec<3>, 4> armorCornersOf(
        const Vec<9>& state, int plate_id, double radius, double half_width, double half_height,
        double y_offset, double side_sign = 1.0, double vertical_sign = 1.0) {
        const double yaw = std::atan2(state(4, 0), state(5, 0));
        const double phi = yaw + angleForPlate(plate_id);
        const double c = std::cos(phi);
        const double s = std::sin(phi);

        Vec<3> center;
        center(0, 0) = state(0, 0) + radius * c;
        center(1, 0) = state(2, 0) + y_offset;
        center(2, 0) = state(7, 0) + radius * s;

        // 切向（板面横向，指向"图像左侧"）与竖直方向。
        // 注意 world 系的 y 是**向下**的：这个 EKF 的 world 就是"云台归零时的相机系"
        // （见 tracker.cpp 的 cameraToBaseRotation：yaw=pitch=0 时 camera_to_base = I），
        // 所以"图像上方"对应 world 的 -y。物点顺序与 getArmor3DPoints 一致
        // （y = 板面横向、z = 竖直），于是 ŷ = radial × up（radial = 外法向）。
        Vec<3> tangent;
        tangent(0, 0) = s;
        tangent(1, 0) = 0.0;
        tangent(2, 0) = -c;
        Vec<3> up;
        up(0, 0) = 0.0;
        up(1, 0) = -1.0;
        up(2, 0) = 0.0;

        // 物点 (0, ±half_w, ±half_h) 的符号：分别为左/右 与 下/上。
        constexpr double side[4] = {1.0, 1.0, -1.0, -1.0};
        constexpr double vertical[4] = {-1.0, 1.0, 1.0, -1.0};
        std::array<Vec<3>, 4> corners;
        for (int i = 0; i < 4; ++i) {
            corners[static_cast<std::size_t>(i)] =
                center + tangent * (half_width * side_sign * side[i]) +
                up * (half_height * vertical_sign * vertical[i]);
        }
        return corners;
    }

    std::array<Vec<3>, 4> armorCorners(int plate_id, double half_width,
                                       double half_height) const {
        return armorCornersOf(x_, plate_id, cfg_.R, half_width, half_height,
                              cfg_.armor_y_offsets[plate_id]);
    }

    /// @brief world 点 → 像素（针孔，无畸变；相机在 world 原点）。
    ///        @return false = 该点在相机后方（这一帧不可用）。
    static bool projectUv(const Vec<3>& world, const UvCamera& camera, Vec<2>& pixel) {
        Vec<3> point;
        for (int row = 0; row < 3; ++row) {
            for (int column = 0; column < 3; ++column) {
                point(row, 0) += camera.world_to_camera(row, column) * world(column, 0);
            }
        }
        if (point(2, 0) <= 1e-3) return false;
        pixel(0, 0) = camera.fx * point(0, 0) / point(2, 0) + camera.cx;
        pixel(1, 0) = camera.fy * point(1, 0) / point(2, 0) + camera.cy;
        return true;
    }

    /// @brief 预测某块板四个角点的像素（观测函数）。
    static bool predictUvCorners(const Vec<9>& state, int plate_id, double radius,
                                 double half_width, double half_height, double y_offset,
                                 const UvCamera& camera, std::array<Vec<2>, 4>& corners,
                                 double side_sign = 1.0, double vertical_sign = 1.0) {
        const auto world = armorCornersOf(state, plate_id, radius, half_width, half_height,
                                          y_offset, side_sign, vertical_sign);
        for (int i = 0; i < 4; ++i) {
            if (!projectUv(world[static_cast<std::size_t>(i)], camera,
                           corners[static_cast<std::size_t>(i)])) {
                return false;
            }
        }
        return true;
    }

    /// @brief 重投影观测的雅可比（8×9：四角点 × u,v）。
    ///        用中心差分而不是解析式：状态只有 9 维、每点两次投影，
    ///        成本可忽略，而手写 8×9 解析式（含 sin/cos 表示、板面切向）很容易写错。
    static Matrix<8, 9> uvJacobian(const Vec<9>& state, int plate_id, double radius,
                                    double half_width, double half_height, double y_offset,
                                    const UvCamera& camera, double side_sign = 1.0,
                                    double vertical_sign = 1.0) {
        const double step[9] = {1e-4, 1e-4, 1e-4, 1e-4, 1e-5, 1e-5, 1e-5, 1e-4, 1e-4};
        Matrix<8, 9> jacobian;
        for (int column = 0; column < 9; ++column) {
            Vec<9> plus = state;
            Vec<9> minus = state;
            plus(column, 0) += step[column];
            minus(column, 0) -= step[column];
            std::array<Vec<2>, 4> corners_plus{};
            std::array<Vec<2>, 4> corners_minus{};
            if (!predictUvCorners(plus, plate_id, radius, half_width, half_height, y_offset,
                                  camera, corners_plus, side_sign, vertical_sign) ||
                !predictUvCorners(minus, plate_id, radius, half_width, half_height, y_offset,
                                  camera, corners_minus, side_sign, vertical_sign)) {
                continue;
            }
            for (int corner = 0; corner < 4; ++corner) {
                for (int axis = 0; axis < 2; ++axis) {
                    jacobian(2 * corner + axis, column) =
                        (corners_plus[static_cast<std::size_t>(corner)](axis, 0) -
                         corners_minus[static_cast<std::size_t>(corner)](axis, 0)) /
                        (2.0 * step[column]);
                }
            }
        }
        return jacobian;
    }

    /// @brief UV 观测更新（可一批多块板：同一刚体，每多一块就多 8 个约束）。
    ///        @return 权重（0 = 全被门限拒掉）
    double updateUvBatch(const std::vector<UvObservation>& observations, const UvCamera& camera,
                         double nis_threshold = std::numeric_limits<double>::infinity()) {
        if (observations.empty() || !initialized_) return 0.0;
        const bool debug = std::getenv("ULTRA_VISION_UV_DEBUG") != nullptr;
        double debug_sq = 0.0;
        int debug_count = 0;

        std::vector<Vec<8>> residuals;
        std::vector<Matrix<8, 9>> jacobians;
        std::vector<UvObservation> accepted;
        accepted.reserve(observations.size());
        residuals.reserve(observations.size());
        jacobians.reserve(observations.size());
        double min_weight = 1.0;

        for (const auto& observation : observations) {
            if (observation.plate_id < 0 || observation.plate_id >= 4) continue;
            // 矩形装甲板的四个角点存在**镜像二义**（PnP 的 IPPE 同样有两个解）：
            // 左右镜像 / 上下镜像 / 两者 都属于"同一个矩形刚体"的合法位姿。
            // 这里按**重投影残差最小**挑配对（4 种符号组合），与 IPPE 选解的口径
            // 一致；成本是 4 倍的投影（几十次浮点运算，可忽略）。
            // 好处：world 系的"上下/左右"符号不必靠猜，换坐标系定义也不会错。
            std::array<Vec<2>, 4> corners{};
            double best_sq = std::numeric_limits<double>::max();
            double best_side = 1.0;
            double best_vertical = 1.0;
            for (const double side_sign : {1.0, -1.0}) {
                for (const double vertical_sign : {1.0, -1.0}) {
                    std::array<Vec<2>, 4> candidate{};
                    if (!predictUvCorners(x_, observation.plate_id, cfg_.R,
                                          observation.half_width, observation.half_height,
                                          cfg_.armor_y_offsets[observation.plate_id], camera,
                                          candidate, side_sign, vertical_sign)) {
                        continue;
                    }
                    double sum = 0.0;
                    for (int i = 0; i < 4; ++i) {
                        const double du =
                            observation.corners[static_cast<std::size_t>(i)](0, 0) -
                            candidate[static_cast<std::size_t>(i)](0, 0);
                        const double dv =
                            observation.corners[static_cast<std::size_t>(i)](1, 0) -
                            candidate[static_cast<std::size_t>(i)](1, 0);
                        sum += du * du + dv * dv;
                    }
                    if (sum < best_sq) {
                        best_sq = sum;
                        best_side = side_sign;
                        best_vertical = vertical_sign;
                        corners = candidate;
                    }
                }
            }
            if (!(best_sq < std::numeric_limits<double>::max())) continue;
            const Matrix<8, 9> jacobian = uvJacobian(
                x_, observation.plate_id, cfg_.R, observation.half_width,
                observation.half_height, cfg_.armor_y_offsets[observation.plate_id], camera,
                best_side, best_vertical);

            Vec<8> residual;
            for (int corner = 0; corner < 4; ++corner) {
                for (int axis = 0; axis < 2; ++axis) {
                    residual(2 * corner + axis, 0) =
                        observation.corners[static_cast<std::size_t>(corner)](axis, 0) -
                        corners[static_cast<std::size_t>(corner)](axis, 0);
                    if (debug) {
                        debug_sq += residual(2 * corner + axis, 0) *
                            residual(2 * corner + axis, 0);
                        ++debug_count;
                    }
                }
            }

            Matrix<8, 8> noise;
            const double sigma_sq = std::max(observation.sigma_px * observation.sigma_px, 1e-6);
            for (int row = 0; row < 8; ++row) noise(row, row) = sigma_sq;

            const Matrix<8, 8> innovation = jacobian * P_ * jacobian.transpose() + noise;
            Matrix<8, 8> innovation_inverse;
            if (!Matrix<8, 8>::invert(innovation, innovation_inverse)) continue;

            double nis = 0.0;
            for (int row = 0; row < 8; ++row) {
                for (int column = 0; column < 8; ++column) {
                    nis += residual(row, 0) * innovation_inverse(row, column) *
                        residual(column, 0);
                }
            }
            if (nis > nis_threshold) continue;

            min_weight = std::min(min_weight, std::exp(-0.5 * nis));
            accepted.push_back(observation);
            residuals.push_back(residual);
            jacobians.push_back(jacobian);
        }
        if (debug) {
            // 门限之前就打：这样"全被 NIS 拒掉"也能看到到底差了多少像素。
            std::cerr << "[uv] raw=" << observations.size()
                      << " residual_rms=" << std::sqrt(debug_sq / std::max(1, debug_count))
                      << " px accepted=" << accepted.size();
            if (!observations.empty()) {
                std::cerr << " plate=" << observations.front().plate_id;
            }
            std::cerr << std::endl;
        }
        if (accepted.empty()) return 0.0;

        const int measurement_dim = 8 * static_cast<int>(accepted.size());
        std::vector<std::vector<double>> H(measurement_dim, std::vector<double>(9, 0.0));
        std::vector<double> residual(measurement_dim, 0.0);
        std::vector<double> noise_diagonal(measurement_dim, 0.0);
        for (std::size_t index = 0; index < accepted.size(); ++index) {
            const int offset = 8 * static_cast<int>(index);
            for (int row = 0; row < 8; ++row) {
                for (int column = 0; column < 9; ++column) {
                    H[offset + row][column] = jacobians[index](row, column);
                }
                residual[offset + row] = residuals[index](row, 0);
                noise_diagonal[offset + row] =
                    std::max(accepted[index].sigma_px * accepted[index].sigma_px, 1e-6);
            }
        }

        std::vector<std::vector<double>> innovation(measurement_dim,
                                                    std::vector<double>(measurement_dim, 0.0));
        for (int row = 0; row < measurement_dim; ++row) {
            for (int column = 0; column < measurement_dim; ++column) {
                double value = 0.0;
                for (int state_row = 0; state_row < 9; ++state_row) {
                    for (int state_column = 0; state_column < 9; ++state_column) {
                        value += H[row][state_row] * P_(state_row, state_column) *
                            H[column][state_column];
                    }
                }
                innovation[row][column] = value;
            }
            innovation[row][row] += noise_diagonal[row];
        }
        std::vector<std::vector<double>> innovation_inverse;
        if (!invertDynamic(innovation, innovation_inverse)) return 0.0;

        std::vector<std::vector<double>> state_measurement(9,
                                                           std::vector<double>(measurement_dim, 0.0));
        for (int state_row = 0; state_row < 9; ++state_row) {
            for (int measurement = 0; measurement < measurement_dim; ++measurement) {
                double value = 0.0;
                for (int state_column = 0; state_column < 9; ++state_column) {
                    value += P_(state_row, state_column) * H[measurement][state_column];
                }
                state_measurement[state_row][measurement] = value;
            }
        }
        std::vector<std::vector<double>> gain(9, std::vector<double>(measurement_dim, 0.0));
        for (int state_row = 0; state_row < 9; ++state_row) {
            for (int output = 0; output < measurement_dim; ++output) {
                for (int input = 0; input < measurement_dim; ++input) {
                    gain[state_row][output] +=
                        state_measurement[state_row][input] * innovation_inverse[input][output];
                }
            }
        }
        for (int state_row = 0; state_row < 9; ++state_row) {
            for (int measurement = 0; measurement < measurement_dim; ++measurement) {
                x_(state_row, 0) += gain[state_row][measurement] * residual[measurement];
            }
        }

        std::vector<std::vector<double>> transform(9, std::vector<double>(9, 0.0));
        for (int row = 0; row < 9; ++row) {
            transform[row][row] = 1.0;
            for (int column = 0; column < 9; ++column) {
                for (int measurement = 0; measurement < measurement_dim; ++measurement) {
                    transform[row][column] -= gain[row][measurement] * H[measurement][column];
                }
            }
        }
        std::vector<std::vector<double>> covariance(9, std::vector<double>(9, 0.0));
        for (int row = 0; row < 9; ++row) {
            for (int column = 0; column < 9; ++column) {
                double value = 0.0;
                for (int i = 0; i < 9; ++i) {
                    for (int j = 0; j < 9; ++j) {
                        value += transform[row][i] * P_(i, j) * transform[column][j];
                    }
                }
                covariance[row][column] = value;
            }
        }
        for (int row = 0; row < 9; ++row) {
            for (int column = 0; column < 9; ++column) {
                double value = 0.0;
                for (int measurement = 0; measurement < measurement_dim; ++measurement) {
                    value += gain[row][measurement] * noise_diagonal[measurement] *
                        gain[column][measurement];
                }
                covariance[row][column] += value;
            }
            if (covariance[row][row] < cfg_.P_min) covariance[row][row] = cfg_.P_min;
        }
        for (int row = 0; row < 9; ++row) {
            for (int column = 0; column < 9; ++column) {
                P_(row, column) = covariance[row][column];
            }
        }

        const double norm = std::sqrt(x_(4, 0) * x_(4, 0) + x_(5, 0) * x_(5, 0));
        if (norm > 1e-6) {
            x_(4, 0) /= norm;
            x_(5, 0) /= norm;
        }
        return min_weight;
    }

    double updateUv(const UvObservation& observation, const UvCamera& camera,
                    double nis_threshold = std::numeric_limits<double>::infinity()) {
        std::vector<UvObservation> observations{observation};
        return updateUvBatch(observations, camera, nis_threshold);
    }

    /// @brief 从状态预测某块板的**两条灯条**（左/右）的紧凑观测量。
    ///        lightbar 0 = 左（lt, lb），1 = 右（rt, rb），与探测器给的点序一致。
    static bool predictUvCompact(const Vec<9>& state, int plate_id, double radius,
                                 double half_width, double half_height, double y_offset,
                                 const UvCamera& camera, int lightbar,
                                 UvCompactObservation& observation) {
        const auto corners = armorCornersOf(state, plate_id, radius, half_width, half_height,
                                            y_offset);
        // 角点顺序 lb, lt, rt, rb：左灯条 = (lt, lb)，右灯条 = (rt, rb)。
        const std::size_t top_index = lightbar == 0 ? 1u : 2u;
        const std::size_t bottom_index = lightbar == 0 ? 0u : 3u;
        Vec<2> top{};
        Vec<2> bottom{};
        if (!projectUv(corners[top_index], camera, top) ||
            !projectUv(corners[bottom_index], camera, bottom)) {
            return false;
        }
        // 按图像上下取端点（v 小的当 top），与参考实现一致。
        if (top(1, 0) > bottom(1, 0)) std::swap(top, bottom);
        const double dx = top(0, 0) - bottom(0, 0);
        const double dy = top(1, 0) - bottom(1, 0);
        observation.plate_id = plate_id;
        observation.angle = std::atan2(dx, dy);
        observation.center_x = 0.5 * (top(0, 0) + bottom(0, 0));
        observation.center_y = 0.5 * (top(1, 0) + bottom(1, 0));
        observation.length = std::sqrt(dx * dx + dy * dy);
        return true;
    }

    /// @brief 通用 EKF 更新（Joseph 形式 + 对称化），DIM = 该量测的维数。
    ///        参考实现（rmcs_auto_aim_v2）逐条更新并用 Joseph 形式，这里同一口径。
    /// @brief 只算 NIS（不更新状态）：用于"多个关联假设里挑一个"。
    template <int DIM>
    double computeNis(const Matrix<DIM, 9>& jacobian, const Vec<DIM>& residual,
                      const Vec<DIM>& noise_variance) const {
        Matrix<DIM, DIM> noise;
        for (int i = 0; i < DIM; ++i) noise(i, i) = std::max(noise_variance(i, 0), 1e-12);
        const Matrix<DIM, DIM> innovation = jacobian * P_ * jacobian.transpose() + noise;
        Matrix<DIM, DIM> innovation_inverse;
        if (!Matrix<DIM, DIM>::invert(innovation, innovation_inverse)) {
            return std::numeric_limits<double>::infinity();
        }
        double nis = 0.0;
        for (int row = 0; row < DIM; ++row) {
            for (int column = 0; column < DIM; ++column) {
                nis += residual(row, 0) * innovation_inverse(row, column) * residual(column, 0);
            }
        }
        return nis;
    }

    template <int DIM>
    double applyUpdate(const Matrix<DIM, 9>& jacobian, const Vec<DIM>& residual,
                       const Vec<DIM>& noise_variance,
                       double nis_threshold = std::numeric_limits<double>::infinity()) {
        Matrix<DIM, DIM> noise;
        for (int i = 0; i < DIM; ++i) noise(i, i) = std::max(noise_variance(i, 0), 1e-12);
        const Matrix<DIM, DIM> innovation = jacobian * P_ * jacobian.transpose() + noise;
        Matrix<DIM, DIM> innovation_inverse;
        if (!Matrix<DIM, DIM>::invert(innovation, innovation_inverse)) return 0.0;

        double nis = 0.0;
        for (int row = 0; row < DIM; ++row) {
            for (int column = 0; column < DIM; ++column) {
                nis += residual(row, 0) * innovation_inverse(row, column) * residual(column, 0);
            }
        }
        if (nis > nis_threshold) return 0.0;

        const Matrix<9, DIM> gain = P_ * jacobian.transpose() * innovation_inverse;
        x_ = x_ + gain * residual;

        Matrix<9, 9> identity;
        for (int i = 0; i < 9; ++i) identity(i, i) = 1.0;
        const Matrix<9, 9> complement = identity - gain * jacobian;
        P_ = complement * P_ * complement.transpose() + gain * noise * gain.transpose();
        // 对称化 + 对角线下限（数值上避免非对称与负定）。
        P_ = (P_ + P_.transpose()) * 0.5;
        for (int i = 0; i < 9; ++i) {
            if (P_(i, i) < cfg_.P_min) P_(i, i) = cfg_.P_min;
        }
        const double norm = std::sqrt(x_(4, 0) * x_(4, 0) + x_(5, 0) * x_(5, 0));
        if (norm > 1e-6) {
            x_(4, 0) /= norm;
            x_(5, 0) /= norm;
        }
        return std::exp(-0.5 * nis);
    }

    /// @brief 紧凑 UV 观测更新。会同时在左/右灯条两种假设里取残差更小的那个
    ///        （探测器的"左右"标签在极端姿态下可能翻转，取小残差即自动对齐）。
    double updateUvCompact(const UvCompactObservation& observation, const UvCamera& camera,
                           double nis_threshold = std::numeric_limits<double>::infinity()) {
        if (!initialized_ || observation.plate_id < 0 || observation.plate_id >= 4) return 0.0;
        const double radius = cfg_.R;
        const double y_offset = cfg_.armor_y_offsets[observation.plate_id];

        // 观测向量 z = [angle, cx, cy, length]；残差的角分量按 2π 归一化。
        Vec<4> z;
        z(0, 0) = observation.angle;
        z(1, 0) = observation.center_x;
        z(2, 0) = observation.center_y;
        z(3, 0) = observation.length;
        Vec<4> variance;
        variance(0, 0) = std::max(observation.sigma_angle * observation.sigma_angle, 1e-8);
        const double sigma_sq = std::max(observation.sigma_px * observation.sigma_px, 1e-6);
        variance(1, 0) = sigma_sq;
        variance(2, 0) = sigma_sq;
        variance(3, 0) = sigma_sq;

        // 关联：在"观测是左灯条 / 右灯条"两个假设里**只选残差更小的那个**应用。
        // （早期实现两个假设都 apply 了一遍，正确假设刚把状态拉过去，错误假设又把它
        //   拽回来 —— 单测 testUpdateDirection 一步就能看出来：观测偏 +12 px，
        //   预测却朝 −44 px 走。）
        double best_nis = std::numeric_limits<double>::infinity();
        Matrix<4, 9> best_jacobian;
        Vec<4> best_residual;
        bool have_best = false;
        for (int lightbar = 0; lightbar < 2; ++lightbar) {
            UvCompactObservation predicted;
            predicted.plate_id = observation.plate_id;
            if (!predictUvCompact(x_, observation.plate_id, radius, observation.half_width,
                                  observation.half_height, y_offset, camera, lightbar,
                                  predicted)) {
                continue;
            }
            Vec<4> residual;
            residual(0, 0) = normalizeAngle(z(0, 0) - predicted.angle);
            residual(1, 0) = z(1, 0) - predicted.center_x;
            residual(2, 0) = z(2, 0) - predicted.center_y;
            residual(3, 0) = z(3, 0) - predicted.length;

            // 雅可比：中心差分（4×9，与角点版本同量级成本）。
            Matrix<4, 9> jacobian;
            const double step[9] = {1e-4, 1e-4, 1e-4, 1e-4, 1e-5, 1e-5, 1e-5, 1e-4, 1e-4};
            bool valid = true;
            for (int column = 0; column < 9 && valid; ++column) {
                Vec<9> plus = x_;
                Vec<9> minus = x_;
                plus(column, 0) += step[column];
                minus(column, 0) -= step[column];
                UvCompactObservation predicted_plus;
                UvCompactObservation predicted_minus;
                if (!predictUvCompact(plus, observation.plate_id, radius,
                                      observation.half_width, observation.half_height, y_offset,
                                      camera, lightbar, predicted_plus) ||
                    !predictUvCompact(minus, observation.plate_id, radius,
                                      observation.half_width, observation.half_height, y_offset,
                                      camera, lightbar, predicted_minus)) {
                    valid = false;
                    break;
                }
                Vec<4> plus_values;
                plus_values(0, 0) = predicted_plus.angle;
                plus_values(1, 0) = predicted_plus.center_x;
                plus_values(2, 0) = predicted_plus.center_y;
                plus_values(3, 0) = predicted_plus.length;
                Vec<4> minus_values;
                minus_values(0, 0) = predicted_minus.angle;
                minus_values(1, 0) = predicted_minus.center_x;
                minus_values(2, 0) = predicted_minus.center_y;
                minus_values(3, 0) = predicted_minus.length;
                // 角分量在 ±π 附近折返：差分也按归一化差值算，避免 2π 跳变。
                plus_values(0, 0) = predicted.angle + normalizeAngle(plus_values(0, 0) - predicted.angle);
                minus_values(0, 0) =
                    predicted.angle + normalizeAngle(minus_values(0, 0) - predicted.angle);
                for (int row = 0; row < 4; ++row) {
                    jacobian(row, column) =
                        (plus_values(row, 0) - minus_values(row, 0)) / (2.0 * step[column]);
                }
            }
            if (!valid) continue;
            const double nis = computeNis<4>(jacobian, residual, variance);
            if (std::getenv("ULTRA_VISION_UV_DEBUG") != nullptr) {
                std::cerr << "[uvc] bar=" << lightbar << " pred(angle=" << predicted.angle
                          << " cx=" << predicted.center_x << " cy=" << predicted.center_y
                          << " len=" << predicted.length << ") z(angle=" << z(0, 0)
                          << " cx=" << z(1, 0) << " cy=" << z(2, 0) << " len=" << z(3, 0)
                          << ") res(angle=" << residual(0, 0) << " cx=" << residual(1, 0)
                          << " cy=" << residual(2, 0) << " len=" << residual(3, 0)
                          << ") nis=" << nis << std::endl;
                for (int row = 0; row < 4; ++row) {
                    std::cerr << "      H[" << row << "]=";
                    for (int column = 0; column < 9; ++column) {
                        std::cerr << ' ' << jacobian(row, column);
                    }
                    std::cerr << std::endl;
                }
            }
            if (nis < best_nis) {
                best_nis = nis;
                best_jacobian = jacobian;
                best_residual = residual;
                have_best = true;
            }
        }
        if (!have_best || best_nis > nis_threshold) return 0.0;
        return applyUpdate<4>(best_jacobian, best_residual, variance, nis_threshold);
    }

    Vec<3> centerFromPlate(const Vec<3>& observation, int plate_id) const {
        const double angle = getYaw() + angleForPlate(plate_id);
        Vec<3> center;
        center(0, 0) = observation(0, 0) - cfg_.R * std::cos(angle);
        center(1, 0) = observation(1, 0) - cfg_.armor_y_offsets[plate_id];
        center(2, 0) = observation(2, 0) - cfg_.R * std::sin(angle);
        return center;
    }

    void dampMotion(double linear_factor, double angular_factor) {
        if (!initialized_) return;
        x_(1, 0) *= linear_factor;
        x_(3, 0) *= linear_factor;
        x_(8, 0) *= linear_factor;
        x_(6, 0) *= angular_factor;
    }

private:
    Vec<4> ypdResidual(
        const YpdObservation& observation,
        const YpdObservation& predicted) const {
        Vec<4> residual;
        residual(0, 0) = normalizeAngle(observation.yaw - predicted.yaw);
        residual(1, 0) = normalizeAngle(observation.pitch - predicted.pitch);
        residual(2, 0) = observation.distance - predicted.distance;
        residual(3, 0) = normalizeAngle(observation.armor_yaw - predicted.armor_yaw);
        return residual;
    }

    Matrix<4, 9> ypdJacobian(int plate_id) const {
        const double plate_angle = angleForPlate(plate_id);
        const double cos_plate = std::cos(plate_angle);
        const double sin_plate = std::sin(plate_angle);
        const double sin_theta = x_(4, 0);
        const double cos_theta = x_(5, 0);

        const double sin_armor = sin_theta * cos_plate + cos_theta * sin_plate;
        const double cos_armor = cos_theta * cos_plate - sin_theta * sin_plate;
        const double px = x_(0, 0) + cfg_.R * cos_armor;
        const double py = x_(2, 0) + cfg_.armor_y_offsets[plate_id];
        const double pz = x_(7, 0) + cfg_.R * sin_armor;

        const double horizontal_sq = px * px + pz * pz;
        const double horizontal = std::sqrt(horizontal_sq);
        const double distance_sq = horizontal_sq + py * py;
        const double distance = std::max(std::sqrt(distance_sq), 1e-9);

        // Position is a direct function of the [sin(theta), cos(theta)]
        // representation, so differentiate the rotation matrix directly.
        // Adding an atan2 chain here is incorrect.
        const double dpx_ds = -cfg_.R * sin_plate;
        const double dpx_dc = cfg_.R * cos_plate;
        const double dpz_ds = cfg_.R * cos_plate;
        const double dpz_dc = cfg_.R * sin_plate;

        Matrix<4, 9> jacobian;
        if (horizontal_sq > 1e-12) {
            const double dyaw_dpx = pz / horizontal_sq;
            const double dyaw_dpz = -px / horizontal_sq;
            jacobian(0, 0) = dyaw_dpx;
            jacobian(0, 7) = dyaw_dpz;
            jacobian(0, 4) = dyaw_dpx * dpx_ds + dyaw_dpz * dpz_ds;
            jacobian(0, 5) = dyaw_dpx * dpx_dc + dyaw_dpz * dpz_dc;

            const double dpitch_dpx = -py * px / (distance_sq * horizontal);
            const double dpitch_dpz = -py * pz / (distance_sq * horizontal);
            const double dpitch_dpy = horizontal / distance_sq;
            jacobian(1, 0) = dpitch_dpx;
            jacobian(1, 2) = dpitch_dpy;
            jacobian(1, 7) = dpitch_dpz;
            jacobian(1, 4) = dpitch_dpx * dpx_ds + dpitch_dpz * dpz_ds;
            jacobian(1, 5) = dpitch_dpx * dpx_dc + dpitch_dpz * dpz_dc;
        }

        const double ddistance_dpx = px / distance;
        const double ddistance_dpy = py / distance;
        const double ddistance_dpz = pz / distance;
        jacobian(2, 0) = ddistance_dpx;
        jacobian(2, 2) = ddistance_dpy;
        jacobian(2, 7) = ddistance_dpz;
        jacobian(2, 4) = ddistance_dpx * dpx_ds + ddistance_dpz * dpz_ds;
        jacobian(2, 5) = ddistance_dpx * dpx_dc + ddistance_dpz * dpz_dc;

        const double angle_norm_sq = sin_theta * sin_theta + cos_theta * cos_theta;
        if (angle_norm_sq > 1e-12) {
            jacobian(3, 4) = cos_theta / angle_norm_sq;
            jacobian(3, 5) = -sin_theta / angle_norm_sq;
        }
        return jacobian;
    }

    static bool invertDynamic(
        std::vector<std::vector<double>> matrix,
        std::vector<std::vector<double>>& inverse) {
        const int size = static_cast<int>(matrix.size());
        if (size == 0) return false;
        inverse.assign(size, std::vector<double>(size, 0.0));
        for (int i = 0; i < size; ++i) inverse[i][i] = 1.0;

        for (int column = 0; column < size; ++column) {
            int pivot = column;
            for (int row = column + 1; row < size; ++row) {
                if (std::abs(matrix[row][column]) > std::abs(matrix[pivot][column])) {
                    pivot = row;
                }
            }
            if (std::abs(matrix[pivot][column]) < 1e-12) return false;

            if (pivot != column) {
                std::swap(matrix[pivot], matrix[column]);
                std::swap(inverse[pivot], inverse[column]);
            }

            const double diagonal = matrix[column][column];
            for (int col = 0; col < size; ++col) {
                matrix[column][col] /= diagonal;
                inverse[column][col] /= diagonal;
            }

            for (int row = 0; row < size; ++row) {
                if (row == column) continue;
                const double factor = matrix[row][column];
                if (std::abs(factor) < 1e-15) continue;
                for (int col = 0; col < size; ++col) {
                    matrix[row][col] -= factor * matrix[column][col];
                    inverse[row][col] -= factor * inverse[column][col];
                }
            }
        }
        return true;
    }

    void initMatrices() {
        cfg_dt_ = 0.01;
        buildQ();
    }

    void buildQ() {
        const double dt = cfg_dt_;
        // 平移自适应：见 Config::process_acc_speed_ref 的注释。
        double qa = cfg_.process_acc;
        if (initialized_ && cfg_.process_acc_speed_ref > 1e-6) {
            const double speed = std::hypot(std::hypot(x_(1, 0), x_(3, 0)), x_(8, 0));
            const double ratio = speed / cfg_.process_acc_speed_ref;
            qa = cfg_.process_acc * (1.0 + ratio * ratio);
            if (qa > cfg_.process_acc_max) qa = cfg_.process_acc_max;
        }
        const double qw = cfg_.process_omega;
        const double dt2 = dt * dt;
        const double dt3 = dt2 * dt;
        const double dt4 = dt3 * dt;

        Q_ = Matrix<9, 9>();
        const int position_idx[3] = {0, 2, 7};
        const int velocity_idx[3] = {1, 3, 8};
        for (int k = 0; k < 3; ++k) {
            const int p = position_idx[k];
            const int v = velocity_idx[k];
        Q_(p, p) = dt4 / 4.0 * qa;
        Q_(p, v) = dt3 / 2.0 * qa;
        Q_(v, p) = dt3 / 2.0 * qa;
        Q_(v, v) = dt2 * qa;
    }

    // theta is represented by [sin(theta), cos(theta)]. Map the same
    // piecewise angular-acceleration noise used by the reference target EKF
    // into this representation instead of injecting independent noise directly
    // into sin/cos, which made a stationary chassis appear to spin.
    const double q_angle = dt4 / 4.0 * qw;
    const double q_angle_omega = dt3 / 2.0 * qw;
    const double q_omega = dt2 * qw;
    double sin_theta = initialized_ ? x_(4, 0) : 0.0;
    double cos_theta = initialized_ ? x_(5, 0) : 1.0;
    const double norm = std::hypot(sin_theta, cos_theta);
    if (norm > 1e-9) {
        sin_theta /= norm;
        cos_theta /= norm;
    } else {
        sin_theta = 0.0;
        cos_theta = 1.0;
    }

    Q_(4, 4) = cos_theta * cos_theta * q_angle;
    Q_(5, 5) = sin_theta * sin_theta * q_angle;
    Q_(4, 5) = -sin_theta * cos_theta * q_angle;
    Q_(5, 4) = Q_(4, 5);
    Q_(4, 6) = cos_theta * q_angle_omega;
    Q_(6, 4) = Q_(4, 6);
    Q_(5, 6) = -sin_theta * q_angle_omega;
    Q_(6, 5) = Q_(5, 6);
    Q_(6, 6) = q_omega;
}

    Config cfg_;
    double cfg_dt_ = 0.01;
    bool initialized_ = false;
    Vec<9> x_;
    Matrix<9, 9> P_;
    Matrix<9, 9> Q_;
};

} // namespace auto_aim

#endif // AUTO_AIM_ARMOR_EKF_HPP
