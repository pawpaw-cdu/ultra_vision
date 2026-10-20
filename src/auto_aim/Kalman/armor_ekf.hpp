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

#include "Kalman/uv_observation.hpp"

#include "Kalman/linalg.hpp"  
#include "Kalman/ypd_observation.hpp"
#include "common/chassis_model.hpp"   
#include "common/types.hpp"    

namespace auto_aim {


// State: [x_c, vx, y_c, vy, sin(theta), cos(theta), omega, z_c, vz].
// The four armor plates are distributed on a circle in the x-z plane.
class ArmorEKF {
public:
    /// 状态维数：9 个运动量 + l(半径差) + h(高度差)。新量追加在末尾，索引 0..8 不变。
    static constexpr int kStateDim = 11;
    struct Config {
        double R = 0.21;
        double process_acc = 1.0;
        // 过程噪声 `process_acc` 物理上是"底盘加速度的不确定度"：
        // 靶车静止时它是 0，横移/急停时才有值。
        double process_acc_speed_ref = 0.28;   // 米/秒，速度尺度
        double process_acc_max = 40.0;         // 上限，防止速度估计野值时爆掉
        double process_omega = 2.0;
        double process_geometry = 1e-7;
        double geometry_delta_max = 0.05;   // |l|,|h| 硬边界（m）
        double P_geometry = 2.5e-3;    // 初始 σ=5cm，允许从 0 收敛到真实差值。
        // 1 = 关闭（退化为标准 EKF）。
        int iekf_iterations = 3;
        double meas_noise = 0.01;
        double y_noise_mult = 30.0;
        double P_min = 1e-3;
        std::array<double, 4> armor_y_offsets{{0.0, 0.0, 0.0, 0.0}};
        // 四块板各自到**车心**的水平距离（米）
        // 0 = 该块板用 R（默认四块相等 → 圆）
        std::array<double, 4> plate_distance{{0.0, 0.0, 0.0, 0.0}};
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

    /// @brief 只改回转半径（在线慢层用）。**不动状态/协方差** —— 半径是几何量
    void setArmorRadius(double radius) {
        if (radius > 0.05 && radius < 1.0) cfg_.R = radius;
    }
    double armorRadius() const { return cfg_.R; }
    double plateRadius(int plate) const {
        if (plate < 0 || plate >= 4) return cfg_.R;
        const double v = cfg_.plate_distance[static_cast<std::size_t>(plate)];
        return v > 0.0 ? v : cfg_.R;
    }
    static bool usesGroupDelta(int plate) { return plate == 1 || plate == 3; }
    double effectiveRadius(int plate) const {
        const double r = plateRadius(plate);
        return usesGroupDelta(plate) ? r + x_(9, 0) : r;
    }
    double effectiveYOffset(int plate) const {
        const int p = std::max(0, std::min(3, plate));
        const double off = cfg_.armor_y_offsets[static_cast<std::size_t>(p)];
        return usesGroupDelta(plate) ? off - x_(10, 0) : off;
    }
    double groupRadiusDelta() const { return x_(9, 0); }
    double groupHeightDelta() const { return x_(10, 0); }

    /// @brief 逐板回转半径（在线慢层用；不动状态/协方差）。
    void setPlateDistance(const std::array<double, 4>& distances) {
        cfg_.plate_distance = distances;
    }
    /// @brief 逐板竖直偏置（home 系，正 = 更靠下）。同样只改几何、不动状态。
    void setArmorYOffsets(const std::array<double, 4>& offsets) {
        cfg_.armor_y_offsets = offsets;
    }

    void reset() {
        x_ = Vec<kStateDim>();
        P_ = Matrix<kStateDim, kStateDim>();
        for (int i = 0; i < kStateDim; ++i) P_(i, i) = 1.0;
        initialized_ = false;
    }

    static double angleForPlate(int plate_id) {
        // 与底盘模型（common/chassis_model.hpp 的 armorPhaseOffset）**同一个约定**
        return armorPhaseOffset(plate_id);
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
            sum_x += observation(0, 0) - plateRadius(plate_id) * std::cos(angle);
            sum_y += observation(1, 0) - cfg_.armor_y_offsets[plate_id];
            sum_z += observation(2, 0) - plateRadius(plate_id) * std::sin(angle);
            ++count;
        }
        if (count == 0) return;

        x_(0, 0) = sum_x / count;
        x_(2, 0) = sum_y / count;
        x_(7, 0) = sum_z / count;
        x_(4, 0) = 0.0;
        x_(5, 0) = 1.0;
        x_(6, 0) = 0.0;
        for (int i = 0; i < kStateDim; ++i) {
            for (int j = 0; j < kStateDim; ++j) P_(i, j) = 0.0;
        }
        P_(0, 0) = 1e-2; P_(2, 2) = 1e-2; P_(7, 7) = 1e-2;
        P_(1, 1) = 1.0;  P_(3, 3) = 1.0;  P_(8, 8) = 1.0;
        P_(4, 4) = 0.4; P_(5, 5) = 0.4;
        P_(6, 6) = 100.0;
        P_(9, 9) = cfg_.P_geometry; P_(10, 10) = cfg_.P_geometry;
        initialized_ = true;
    }

    void initAt(const Vec<3>& center, double yaw) {
        x_ = Vec<kStateDim>();
        x_(0, 0) = center(0, 0);
        x_(2, 0) = center(1, 0);
        x_(7, 0) = center(2, 0);
        x_(4, 0) = std::sin(yaw);
        x_(5, 0) = std::cos(yaw);

        for (int i = 0; i < kStateDim; ++i) {
            for (int j = 0; j < kStateDim; ++j) P_(i, j) = 0.0;
        }
        P_(0, 0) = 0.01; P_(2, 2) = 0.02; P_(7, 7) = 0.01;
        P_(1, 1) = 1.0;  P_(3, 3) = 1.0;  P_(8, 8) = 1.0;
        P_(4, 4) = 0.4; P_(5, 5) = 0.4;
        // ω 的初始方差：**别给 100**。预测一步会生成 P(ω,sin/cos) 交叉协方差
        // RotationRateEstimator 经 updateOmega() 融合进来，不靠这一帧的 yaw。
        P_(6, 6) = 4.0;
        initialized_ = true;
    }

    void init(const Vec<3>& observation, int plate_id) {
        double angle = angleForPlate(plate_id);
        x_(0, 0) = observation(0, 0) - plateRadius(plate_id) * std::cos(angle);
        x_(2, 0) = observation(1, 0) - cfg_.armor_y_offsets[plate_id];
        x_(7, 0) = observation(2, 0) - plateRadius(plate_id) * std::sin(angle);
        x_(4, 0) = 0.0;
        x_(5, 0) = 1.0;
        x_(6, 0) = 0.0;
        for (int i = 0; i < kStateDim; ++i) {
            for (int j = 0; j < kStateDim; ++j) P_(i, j) = 0.0;
        }
        P_(0, 0) = 0.1; P_(2, 2) = 0.1; P_(7, 7) = 0.1;
        P_(1, 1) = 1.0; P_(3, 3) = 1.0; P_(8, 8) = 1.0;
        P_(4, 4) = 0.4; P_(5, 5) = 0.4;
        P_(6, 6) = 100.0;
        P_(9, 9) = cfg_.P_geometry; P_(10, 10) = cfg_.P_geometry;
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
        // l/h 是几何量，给硬边界（防跑飞；正常应在 ±5cm 内）
        x_(9, 0) = std::clamp(x_(9, 0), -cfg_.geometry_delta_max, cfg_.geometry_delta_max);
        x_(10, 0) = std::clamp(x_(10, 0), -cfg_.geometry_delta_max, cfg_.geometry_delta_max);

        Matrix<kStateDim, kStateDim> F;
        for (int i = 0; i < kStateDim; ++i) F(i, i) = 1.0;
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

    /// @brief YPD 观测的**马氏距离平方** d² = rᵀ·S⁻¹·r，S = H·P·Hᵀ + R。
    /// 返回 +inf 表示未初始化 / plate_id 非法 / S 不可逆（调用方应放弃这一对）。
    double ypdMahalanobisSquared(const YpdObservation& observation) const {
        const double inf = std::numeric_limits<double>::infinity();
        if (!initialized_) return inf;
        if (observation.plate_id < 0 || observation.plate_id >= 4) return inf;
        const YpdObservation predicted = predictYpd(observation.plate_id);
        const Vec<4> residual = ypdResidual(observation, predicted);
        const Matrix<4, kStateDim> jacobian = ypdJacobian(observation.plate_id);

        Matrix<4, 4> noise;
        noise(0, 0) = std::max(observation.yaw_noise, 1e-8);
        noise(1, 1) = std::max(observation.pitch_noise, 1e-8);
        noise(2, 2) = std::max(observation.distance_noise, 1e-8);
        noise(3, 3) = std::max(observation.armor_yaw_noise, 1e-8);

        const Matrix<4, 4> innovation =
            jacobian * P_ * jacobian.transpose() + noise;
        Matrix<4, 4> innovation_inverse;
        if (!Matrix<4, 4>::invert(innovation, innovation_inverse)) return inf;

        double squared = 0.0;
        for (int row = 0; row < 4; ++row) {
            for (int column = 0; column < 4; ++column) {
                squared += residual(row, 0) * innovation_inverse(row, column) *
                    residual(column, 0);
            }
        }
        if (!std::isfinite(squared)) return inf;
        return std::max(0.0, squared);
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
            center(0, 0) = position(0, 0) - plateRadius(first.plate_id) * std::cos(
                yaw + angleForPlate(first.plate_id));
            center(1, 0) = position(1, 0) -
                cfg_.armor_y_offsets[first.plate_id];
            center(2, 0) = position(2, 0) - plateRadius(first.plate_id) * std::sin(
                yaw + angleForPlate(first.plate_id));
            initAt(center, yaw);
            return 1.0;
        }

        std::array<YpdObservation, 4> accepted{};
        std::array<Vec<4>, 4> residuals{};
        std::array<bool, 4> plate_seen{};
        int accepted_count = 0;

        double min_weight = 1.0;
        for (const auto& observation : observations) {
            if (observation.plate_id < 0 || observation.plate_id >= 4) continue;
            if (!std::isfinite(observation.yaw) || !std::isfinite(observation.pitch) ||
                !std::isfinite(observation.distance) || !std::isfinite(observation.armor_yaw) ||
                observation.distance <= 1e-4) {
                continue;
            }
            if (plate_seen[static_cast<std::size_t>(observation.plate_id)]) continue;

            const YpdObservation predicted = predictYpd(observation.plate_id);
            const Vec<4> residual = ypdResidual(observation, predicted);
            const Matrix<4, kStateDim> jacobian = ypdJacobian(observation.plate_id);
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
            accepted[static_cast<std::size_t>(accepted_count)] = observation;
            residuals[static_cast<std::size_t>(accepted_count)] = residual;
            ++accepted_count;
            plate_seen[static_cast<std::size_t>(observation.plate_id)] = true;
        }

        if (accepted_count == 0) return 0.0;

        // plate_id is validated against 0..3 above, so accepted observations can
        // contribute at most 4 * 4 = 16 measurement rows.
        constexpr int kMaxMeasurementDim = 16;
        const int measurement_dim = 4 * accepted_count;
        std::array<double, kMaxMeasurementDim> noise_diagonal{};

        for (std::size_t observation_index = 0;
             observation_index < static_cast<std::size_t>(accepted_count);
             ++observation_index) {
            const int row_offset = 4 * static_cast<int>(observation_index);
            const YpdObservation& observation = accepted[observation_index];
            noise_diagonal[row_offset + 0] = std::max(observation.yaw_noise, 1e-8);
            noise_diagonal[row_offset + 1] = std::max(observation.pitch_noise, 1e-8);
            noise_diagonal[row_offset + 2] = std::max(observation.distance_noise, 1e-8);
            noise_diagonal[row_offset + 3] = std::max(observation.armor_yaw_noise, 1e-8);
        }

        // ---- IESEKF：在更新后的状态处重新线性化（迭代 2-5 次）--------------
        // 的 K/H 做一次 Joseph 更新（避免迭代中重复收缩 P）。
        const int iterations = std::max(1, cfg_.iekf_iterations);
        const Vec<kStateDim> x_prior = x_;
        const Matrix<kStateDim, kStateDim> P_prior = P_;
        Matrix<kMaxMeasurementDim, kStateDim> H;
        Vec<kMaxMeasurementDim> residual;
        Matrix<kStateDim, kMaxMeasurementDim> gain;
        bool update_ok = false;

        for (int iteration = 0; iteration < iterations; ++iteration) {
            // 在当前状态 x_k（成员 x_）处重新算 H(x_k) 与残差。
            for (std::size_t observation_index = 0;
                 observation_index < static_cast<std::size_t>(accepted_count);
                 ++observation_index) {
                const int row_offset = 4 * static_cast<int>(observation_index);
                const YpdObservation& observation = accepted[observation_index];
                const YpdObservation predicted = predictYpd(observation.plate_id);
                Vec<4> innovation_residual = ypdResidual(observation, predicted);
                const Matrix<4, kStateDim> jacobian = ypdJacobian(observation.plate_id);
                if (iteration > 0) {
                    Vec<kStateDim> delta;
                    for (int i = 0; i < kStateDim; ++i) delta(i, 0) = x_(i, 0) - x_prior(i, 0);
                    const Vec<4> linearized = jacobian * delta;
                    for (int row = 0; row < 4; ++row) {
                        innovation_residual(row, 0) += linearized(row, 0);
                    }
                }
                for (int row = 0; row < 4; ++row) {
                    for (int column = 0; column < kStateDim; ++column) {
                        H(row_offset + row, column) = jacobian(row, column);
                    }
                    residual(row_offset + row, 0) = innovation_residual(row, 0);
                }
            }

            Matrix<kMaxMeasurementDim, kMaxMeasurementDim> innovation;
            for (int row = 0; row < measurement_dim; ++row) {
                for (int column = 0; column < measurement_dim; ++column) {
                    double value = 0.0;
                    for (int state_row = 0; state_row < kStateDim; ++state_row) {
                        for (int state_column = 0; state_column < kStateDim; ++state_column) {
                            value += H(row, state_row) * P_prior(state_row, state_column) *
                                H(column, state_column);
                        }
                    }
                    innovation(row, column) = value;
                }
                innovation(row, row) += noise_diagonal[row];
            }

            Matrix<kMaxMeasurementDim, kMaxMeasurementDim> innovation_inverse;
            if (!invertDynamic(innovation, innovation_inverse, measurement_dim)) return 0.0;

            Matrix<kStateDim, kMaxMeasurementDim> state_measurement_covariance;
            for (int state_row = 0; state_row < kStateDim; ++state_row) {
                for (int measurement = 0; measurement < measurement_dim; ++measurement) {
                    double value = 0.0;
                    for (int state_column = 0; state_column < kStateDim; ++state_column) {
                        value += P_prior(state_row, state_column) * H(measurement, state_column);
                    }
                    state_measurement_covariance(state_row, measurement) = value;
                }
            }

            for (int state_row = 0; state_row < kStateDim; ++state_row) {
                for (int output_measurement = 0; output_measurement < measurement_dim;
                     ++output_measurement) {
                    double value = 0.0;
                    for (int input_measurement = 0; input_measurement < measurement_dim;
                         ++input_measurement) {
                        value +=
                            state_measurement_covariance(state_row, input_measurement) *
                            innovation_inverse(input_measurement, output_measurement);
                    }
                    gain(state_row, output_measurement) = value;
                }
            }

            // x_{k+1} = x_prior + K·r_k（从先验出发，而不是在上一次结果上累加）。
            for (int state_row = 0; state_row < kStateDim; ++state_row) {
                double value = x_prior(state_row, 0);
                for (int measurement = 0; measurement < measurement_dim; ++measurement) {
                    value += gain(state_row, measurement) * residual(measurement, 0);
                }
                x_(state_row, 0) = value;
            }
            update_ok = true;
        }
        if (!update_ok) return 0.0;

        Matrix<kStateDim, kStateDim> transform;
        for (int row = 0; row < kStateDim; ++row) {
            transform(row, row) = 1.0;
            for (int column = 0; column < kStateDim; ++column) {
                for (int measurement = 0; measurement < measurement_dim; ++measurement) {
                    transform(row, column) -=
                        gain(row, measurement) * H(measurement, column);
                }
            }
        }

        Matrix<kStateDim, kStateDim> covariance;
        for (int row = 0; row < kStateDim; ++row) {
            for (int column = 0; column < kStateDim; ++column) {
                double value = 0.0;
                for (int i = 0; i < kStateDim; ++i) {
                    for (int j = 0; j < kStateDim; ++j) {
                        value += transform(row, i) * P_(i, j) * transform(column, j);
                    }
                }
                covariance(row, column) = value;
            }
        }

        for (int row = 0; row < kStateDim; ++row) {
            for (int column = 0; column < kStateDim; ++column) {
                double value = 0.0;
                for (int measurement = 0; measurement < measurement_dim; ++measurement) {
                    value += gain(row, measurement) * noise_diagonal[measurement] *
                        gain(column, measurement);
                }
                covariance(row, column) += value;
            }
        }

        for (int row = 0; row < kStateDim; ++row) {
            for (int column = 0; column < kStateDim; ++column) {
                P_(row, column) = covariance(row, column);
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
            for (int observation_index = 0;
                 observation_index < accepted_count; ++observation_index) {
                const auto& residual =
                    residuals[static_cast<std::size_t>(observation_index)];
                for (int row = 0; row < 8; ++row) {
                    sum += residual(row, 0) * residual(row, 0);
                    ++count;
                }
            }
            std::cerr << "[uv] plates=" << accepted_count
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
        predicted(0, 0) = x_(0, 0) + plateRadius(plate_id) * cos_total;
        predicted(1, 0) = x_(2, 0) + cfg_.armor_y_offsets[plate_id];
        predicted(2, 0) = x_(7, 0) + plateRadius(plate_id) * sin_total;

        Matrix<3, kStateDim> H;
        // theta is represented by [sin(theta), cos(theta)] in the state, so
        // the position Jacobian must be taken with respect to those variables.
        H(0, 0) = 1.0; H(0, 4) = -plateRadius(plate_id) * sin_plate;
        H(0, 5) = plateRadius(plate_id) * cos_plate;
        H(1, 2) = 1.0;
        H(2, 7) = 1.0; H(2, 4) = plateRadius(plate_id) * cos_plate;
        H(2, 5) = plateRadius(plate_id) * sin_plate;

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

        if (distance_sq > nis_threshold) return 0.0;

        Matrix<kStateDim, 3> K = P_ * H.transpose() * S_inv;
        x_ = x_ + K * residual;

        Matrix<kStateDim, kStateDim> I;
        for (int i = 0; i < kStateDim; ++i) I(i, i) = 1.0;
        Matrix<kStateDim, kStateDim> IKH = I - K * H;
        P_ = IKH * P_ * IKH.transpose() + K * R * K.transpose();
        for (int i = 0; i < kStateDim; ++i) {
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

        // 观测 = θ + plate_angle，状态是 (x4, x5) = (sinθ, cosθ)：
        //     ∂观测/∂x4 = ∂θ/∂s =  cosθ = x5
        //     ∂观测/∂x5 = ∂θ/∂c = −sinθ = −x4
        // **必须用状态自己的 (sinθ, cosθ)**，不能用"旋转之后"的
        Matrix<1, kStateDim> H;
        H(0, 4) = x_(5, 0);
        H(0, 5) = -x_(4, 0);

        const Matrix<1, 1> S = H * P_ * H.transpose();
        const double innovation_variance = S(0, 0) + noise;
        if (innovation_variance < 1e-9) return 0.0;

        const double nis = residual_angle * residual_angle / innovation_variance;
        if (nis > nis_threshold) return 0.0;

        const Matrix<kStateDim, 1> K = P_ * H.transpose() * (1.0 / innovation_variance);
        for (int i = 0; i < kStateDim; ++i) {
            x_(i, 0) += K(i, 0) * residual_angle;
        }

        Matrix<kStateDim, kStateDim> I;
        for (int i = 0; i < kStateDim; ++i) I(i, i) = 1.0;
        const Matrix<kStateDim, kStateDim> IKH = I - K * H;
        P_ = IKH * P_ * IKH.transpose() + K * K.transpose() * noise;
        for (int i = 0; i < kStateDim; ++i) {
            if (P_(i, i) < cfg_.P_min) P_(i, i) = cfg_.P_min;
        }

        const double norm = std::sqrt(x_(4, 0) * x_(4, 0) + x_(5, 0) * x_(5, 0));
        if (norm > 1e-6) {
            x_(4, 0) /= norm;
            x_(5, 0) /= norm;
        }
        return std::exp(-0.5 * nis);
    }

    double updateOmega(
        double observed_omega, double noise,
        double nis_threshold = std::numeric_limits<double>::infinity()) {
        if (!initialized_ || !std::isfinite(observed_omega) ||
            !std::isfinite(noise) || noise <= 0.0) {
            return 0.0;
        }

        Matrix<1, kStateDim> H;
        H(0, 6) = 1.0;

        const Matrix<1, 1> S = H * P_ * H.transpose();
        const double innovation_variance = S(0, 0) + noise;
        if (innovation_variance < 1e-12) return 0.0;

        const double residual = observed_omega - x_(6, 0);
        const double nis = residual * residual / innovation_variance;
        if (nis > nis_threshold) return 0.0;

        const Matrix<kStateDim, 1> K =
            P_ * H.transpose() * (1.0 / innovation_variance);
        for (int i = 0; i < kStateDim; ++i) {
            x_(i, 0) += K(i, 0) * residual;
        }

        Matrix<kStateDim, kStateDim> I;
        for (int i = 0; i < kStateDim; ++i) I(i, i) = 1.0;
        const Matrix<kStateDim, kStateDim> IKH = I - K * H;
        P_ = IKH * P_ * IKH.transpose() + K * K.transpose() * noise;
        return std::exp(-0.5 * nis);
    }

    double updateCenter(
        const Vec<3>& observation, double noise, double y_noise_mult,
        double nis_threshold = std::numeric_limits<double>::infinity()) {
        if (!initialized_) return 0.0;

        Matrix<3, kStateDim> H;
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

        Matrix<kStateDim, 3> K = P_ * H.transpose() * S_inv;
        x_ = x_ + K * residual;

        Matrix<kStateDim, kStateDim> I;
        for (int i = 0; i < kStateDim; ++i) I(i, i) = 1.0;
        Matrix<kStateDim, kStateDim> IKH = I - K * H;
        P_ = IKH * P_ * IKH.transpose() + K * R_noise * K.transpose();

        const double norm = std::sqrt(x_(4, 0) * x_(4, 0) + x_(5, 0) * x_(5, 0));
        if (norm > 1e-6) {
            x_(4, 0) /= norm;
            x_(5, 0) /= norm;
        }
        return std::exp(-0.5 * distance_sq);
    }

    Vec<kStateDim> getState() const { return x_; }

    // ======================================================================
    // UV-EKF（刚体分解）：直接用四个灯条端点像素更新整车状态。
    // 见 Kalman/uv_observation.hpp 的推导与两处设计说明（差分/平均的分块噪声、
    // 以及"一次 8 维更新"取代方案里的"两次顺序更新"）。
    // ======================================================================

    /// 四个灯条端点在**世界系**，顺序 [左灯条上, 左灯条下, 右灯条上, 右灯条下]。
    std::array<Vec<3>, 4> uvEndpoints(int plate_id, const UvGeometry& g) const
    {
        const double phi = std::atan2(x_(4, 0), x_(5, 0)) + angleForPlate(plate_id);
        const double ct = std::cos(phi), st = std::sin(phi);
        const double sa = g.alpha_sin(), ca = g.alpha_cos();
        // 径向 n̂=(ct,0,st)、切向 t̂=(−st,0,ct)、灯条方向 b̂=−(sa·n̂ + ca·ŷ)
        const double nx = ct, nz = st, tx = -st, tz = ct;
        const double bx = -sa * nx, by = -ca, bz = -sa * nz;
        const Vec<3> center = armorPosition(plate_id);
        const double a = 0.5 * g.plate_width, b = 0.5 * g.lightbar_len;
        std::array<Vec<3>, 4> out;
        const double sign_a[4] = {+1.0, +1.0, -1.0, -1.0};   // 左、左、右、右
        const double sign_b[4] = {+1.0, -1.0, +1.0, -1.0};   // 上、下、上、下
        for (int i = 0; i < 4; ++i)
        {
            const double A = sign_a[i] * a, B = sign_b[i] * b;
            out[static_cast<std::size_t>(i)](0, 0) = center(0, 0) + A * tx + B * bx;
            out[static_cast<std::size_t>(i)](1, 0) = center(1, 0) + B * by;
            out[static_cast<std::size_t>(i)](2, 0) = center(2, 0) + A * tz + B * bz;
        }
        return out;
    }

    /// 把世界系点投到像素（针孔 + 给定的 world→camera）。
    Vec<2> uvProject(const Vec<3>& p_world, const UvObservation& o) const
    {
        const double dx = p_world(0, 0) - o.camera_origin[0];
        const double dy = p_world(1, 0) - o.camera_origin[1];
        const double dz = p_world(2, 0) - o.camera_origin[2];
        const double *R = o.R_world2camera.data();
        const double X = R[0] * dx + R[1] * dy + R[2] * dz;
        const double Y = R[3] * dx + R[4] * dy + R[5] * dz;
        const double Z = R[6] * dx + R[7] * dy + R[8] * dz;
        const double z = std::max(1e-3, Z);
        Vec<2> uv;
        uv(0, 0) = o.fx * X / z + o.cx;
        uv(1, 0) = o.fy * Y / z + o.cy;
        return uv;
    }

    /// 8 维残差（预测 − 观测），顺序 [u_lt,v_lt, u_lb,v_lb, u_rt,v_rt, u_rb,v_rb]。
    Vec<8> uvResidual(const UvObservation& o, const UvGeometry & g) const
    {
        const auto pts = uvEndpoints(o.plate_id, g);
        Vec<8> r;
        for (int i = 0; i < 4; ++i)
        {
            const Vec<2> uv = uvProject(pts[static_cast<std::size_t>(i)], o);
            r(2 * i, 0) = uv(0, 0) - o.corners[static_cast<std::size_t>(i)].x;
            r(2 * i + 1, 0) = uv(1, 0) - o.corners[static_cast<std::size_t>(i)].y;
        }
        return r;
    }

    Matrix<8, kStateDim> uvJacobian(const UvObservation & o, const UvGeometry & g,
                                    double base) const
    {
        Matrix<8, kStateDim> H;
        const Vec<8> r0 = uvResidual(o, g);
        for (int j = 0; j < kStateDim; ++j)
        {
            const double delta = 1e-6 * std::max(1.0, std::abs(x_(j, 0)));
            ArmorEKF probe = *this;
            probe.x_(j, 0) += delta;
            const Vec<8> r1 = probe.uvResidual(o, g);
            for (int i = 0; i < 8; ++i) H(i, j) = (r1(i, 0) - r0(i, 0)) / delta;
        }
        (void)base;
        return H;
    }

    /// 差分残差（4 维）：每根灯条 (Δu, Δv) = (观测 − 预测)，与 updateAngle 同约定

    Vec<4> uvDiffResidual(const UvObservation & o, const UvGeometry & g) const
    {
        const auto pts = uvEndpoints(o.plate_id, g);
        Vec<4> r;
        for (int bar = 0; bar < 2; ++bar)
        {
            const Vec<2> top = uvProject(pts[static_cast<std::size_t>(2 * bar)], o);
            const Vec<2> bot = uvProject(pts[static_cast<std::size_t>(2 * bar + 1)], o);
            const auto & ct = o.corners[static_cast<std::size_t>(2 * bar)];
            const auto & cb = o.corners[static_cast<std::size_t>(2 * bar + 1)];
            r(2 * bar, 0) = (ct.x - cb.x) - (top(0, 0) - bot(0, 0));
            r(2 * bar + 1, 0) = (ct.y - cb.y) - (top(1, 0) - bot(1, 0));
        }
        return r;
    }

    /// 平均残差（4 维）：每根灯条端点中点 (ū, v̄)。含 C 与 θ。
    Vec<4> uvAvgResidual(const UvObservation & o, const UvGeometry & g) const
    {
        const auto pts = uvEndpoints(o.plate_id, g);
        Vec<4> r;
        for (int bar = 0; bar < 2; ++bar)
        {
            const Vec<2> top = uvProject(pts[static_cast<std::size_t>(2 * bar)], o);
            const Vec<2> bot = uvProject(pts[static_cast<std::size_t>(2 * bar + 1)], o);
            const auto & ct = o.corners[static_cast<std::size_t>(2 * bar)];
            const auto & cb = o.corners[static_cast<std::size_t>(2 * bar + 1)];
            r(2 * bar, 0) = 0.5 * (ct.x + cb.x) - 0.5 * (top(0, 0) + bot(0, 0));
            r(2 * bar + 1, 0) = 0.5 * (ct.y + cb.y) - 0.5 * (top(1, 0) + bot(1, 0));
        }
        return r;
    }

    /// 通用 4 维更新（数值雅可比）。residual_fn 决定是差分还是平均那一组。
    template <typename ResidualFn>
    double uvUpdate4(const UvObservation & o, double sigma_px, ResidualFn residual_fn,
                     double nis_threshold, double gain = 1.0) {
        const Vec<4> residual = residual_fn(*this, o);
        Matrix<4, kStateDim> H;
        for (int j = 0; j < kStateDim; ++j)
        {
            const double delta = 1e-6 * std::max(1.0, std::abs(x_(j, 0)));
            ArmorEKF probe = *this;
            probe.x_(j, 0) += delta;
            const Vec<4> r1 = residual_fn(probe, o);
            for (int i = 0; i < 4; ++i) H(i, j) = (r1(i, 0) - residual(i, 0)) / delta;
        }
        Matrix<4, 4> R;
        for (int i = 0; i < 4; ++i) R(i, i) = sigma_px * sigma_px;
        const Matrix<kStateDim, 4> PHt = P_ * H.transpose();
        Matrix<4, 4> S = H * PHt + R;
        Matrix<4, 4> Sinv;
        if (!Matrix<4, 4>::invert(S, Sinv)) return 0.0;
        double nis = 0.0;
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                nis += residual(i, 0) * Sinv(i, j) * residual(j, 0);
        if (!std::isfinite(nis) || nis > nis_threshold) return 0.0;
        const Matrix<kStateDim, 4> K = PHt * Sinv;
        for (int i = 0; i < kStateDim; ++i)
        {
            double dx = 0.0;
            for (int j = 0; j < 4; ++j) dx += K(i, j) * residual(j, 0);
            x_(i, 0) += gain * dx;   // 阻尼：见 updateUV 的注释（单板 C↔θ 近退化，全增益会冲过头）
        }
        Matrix<kStateDim, kStateDim> I;
        for (int i = 0; i < kStateDim; ++i) I(i, i) = 1.0;
        const Matrix<kStateDim, kStateDim> IKH = I - K * H;
        P_ = IKH * P_ * IKH.transpose() + K * R * K.transpose();
        for (int i = 0; i < kStateDim; ++i)
            if (P_(i, i) < cfg_.P_min) P_(i, i) = cfg_.P_min;
        const double norm = std::sqrt(x_(4, 0) * x_(4, 0) + x_(5, 0) * x_(5, 0));
        if (norm > 1e-6) { x_(4, 0) /= norm; x_(5, 0) /= norm; }
        return std::exp(-0.5 * nis);
    }

    /// 一次 8 维 UV 更新。返回 NIS 的软权重（0 = 被门限拒绝）。
    double updateUV(
        const UvObservation & o, const UvGeometry & g,
        double nis_threshold = 9.49 /* χ²(4,95%)，每步都是 4 维 */, double gain = 1.0) {
        if (!initialized_ || o.plate_id < 0 || o.plate_id >= 4) return 0.0;
        if (!(o.fx > 1.0) || !(o.fy > 1.0)) return 0.0;

        // **两步**：先差分（与平移无关 → 单独定 θ/R），再平均（在 θ 已知下定 C）。
        const double s_corner = std::sqrt(0.5 * (o.sigma_diff_px * o.sigma_diff_px +
                                                 o.sigma_avg_px * o.sigma_avg_px));
        const double s_diff = (o.sigma_diff_px > 0 ? o.sigma_diff_px : std::sqrt(2.0) * s_corner);
        const double s_avg = (o.sigma_avg_px > 0 ? o.sigma_avg_px : s_corner / std::sqrt(2.0));
        const double w_diff = uvUpdate4(
            o, s_diff,
            [&](const ArmorEKF & e, const UvObservation & oo) { return e.uvDiffResidual(oo, g); },
            nis_threshold, gain);
        const double w_avg = uvUpdate4(
            o, s_avg,
            [&](const ArmorEKF & e, const UvObservation & oo) { return e.uvAvgResidual(oo, g); },
            nis_threshold, gain);
        return std::max(w_diff, w_avg);
    }


    bool isInit() const { return initialized_; }
    double getYaw() const { return std::atan2(x_(4, 0), x_(5, 0)); }
    double getOmega() const { return x_(6, 0); }

    void setAngularVelocity(double omega, double covariance = 0.04) {
        if (!initialized_ || !std::isfinite(omega)) return;
        x_(6, 0) = omega;
        P_(6, 6) = std::max(cfg_.P_min, covariance);
        for (int i = 0; i < kStateDim; ++i) {
            if (i == 6) continue;
            P_(i, 6) = 0.0;
            P_(6, i) = 0.0;
        }
    }

    double getPlateAngle(int plate_id) const {
        return normalizeAngle(getYaw() + angleForPlate(plate_id));
    }

    /// @brief 当前底盘 yaw 估计的标准差（rad）。
    ///        θ = atan2(sinθ, cosθ)，δθ = cosθ·δsinθ − sinθ·δcosθ，所以
    ///        Var(θ) = (c²·P_ss + s²·P_cc − 2·s·c·P_sc) / (s²+c²)。
    ///        用途：把关联门限写成"跟着滤波器自身不确定度放大"的形式 ——
    ///        初始化/丢帧/刚重捕时 yaw 不确定 → 门限宽；收敛后 → 门限紧。
    ///        这是让固定角门限不再随 ω 变大而失配的关键（见 tracker 的
    ///        adaptive_association_gate / max_angle_error 注释）。
    double yawStd() const {
        const double s = x_(4, 0);
        const double c = x_(5, 0);
        const double norm2 = s * s + c * c;
        if (!initialized_ || norm2 < 1e-12) return 0.0;
        const double variance =
            (c * c * P_(4, 4) + s * s * P_(5, 5) - 2.0 * s * c * P_(4, 5)) / norm2;
        return std::sqrt(std::max(0.0, variance));
    }

    /// @brief 某个位置分量(0=x,1=y,2=z)的估计方差（看门狗判断发散用）。
    double positionVariance(int axis) const {
        const int idx = axis == 0 ? 0 : (axis == 1 ? 2 : 7);
        return P_(idx, idx);
    }

    /// @brief 状态协方差对角元（诊断"P 是否真实"用；只读）。
    ///        状态顺序：0=x_c 1=vx 2=y_c 3=vy 4=sinθ 5=cosθ 6=ω 7=z_c 8=vz 9=l 10=h。
    double covarianceDiag(int i) const {
        if (i < 0 || i >= kStateDim) return 0.0;
        return P_(i, i);
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
        position(0, 0) = x_(0, 0) + effectiveRadius(plate_id) * (cos_theta * cos_plate - sin_theta * sin_plate);
        position(1, 0) = x_(2, 0) + effectiveYOffset(plate_id);
        position(2, 0) = x_(7, 0) + effectiveRadius(plate_id) * (sin_theta * cos_plate + cos_theta * sin_plate);
        return position;
    }


    Vec<3> centerFromPlate(const Vec<3>& observation, int plate_id) const {
        const double angle = getYaw() + angleForPlate(plate_id);
        Vec<3> center;
        center(0, 0) = observation(0, 0) - effectiveRadius(plate_id) * std::cos(angle);
        center(1, 0) = observation(1, 0) - effectiveYOffset(plate_id);
        center(2, 0) = observation(2, 0) - effectiveRadius(plate_id) * std::sin(angle);
        return center;
    }

    /// @brief 只平移底盘中心（x/y/z 三个自由度），yaw/速度不动。
    ///        用于"把识别到的装甲板当硬约束"：center += (观测板位置 − 模型板位置)，
    ///        平移后模型算出的那块板就**精确等于观测**，另外三块随之平移（以观测板为锚）。
    void shiftCenter(const Vec<3>& delta) {
        if (!initialized_) return;
        x_(0, 0) += delta(0, 0);
        x_(2, 0) += delta(1, 0);
        x_(7, 0) += delta(2, 0);
    }

    /// @brief 底盘 yaw 翻 180°（sin/cos 同时取反），协方差不动。
    ///        用途：**硬约束**"构造用的那块板一定是离相机最近的那块"被违反时，
    ///        说明模型的径向翻了 180°（PnP 法向翻号 / 关联到对面板），把相位翻回来。
    /// @brief 硬置底盘 yaw（换板时"相位重锚"用）。
    ///        换板那一帧，滤波器的 yaw 还停在旧板的状态上，而车心 = 观测板 − R·radial(yaw+φ)
    ///        → yaw 一滞后，车心就被新板方向拖走（实车 CSV：换板跳 0.43m、同板只抖 0.026m）。
    void setChassisYaw(double yaw) {
        if (!initialized_ || !std::isfinite(yaw)) return;
        x_(4, 0) = std::sin(yaw);
        x_(5, 0) = std::cos(yaw);
        P_(4, 4) = std::max(P_(4, 4), 0.02);
        P_(5, 5) = std::max(P_(5, 5), 0.02);
    }

    void flipChassisYaw() {
        if (!initialized_) return;
        x_(4, 0) = -x_(4, 0);
        x_(5, 0) = -x_(5, 0);
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

    Matrix<4, kStateDim> ypdJacobian(int plate_id) const {
        const double plate_angle = angleForPlate(plate_id);
        const double cos_plate = std::cos(plate_angle);
        const double sin_plate = std::sin(plate_angle);
        const double sin_theta = x_(4, 0);
        const double cos_theta = x_(5, 0);

        const double sin_armor = sin_theta * cos_plate + cos_theta * sin_plate;
        const double cos_armor = cos_theta * cos_plate - sin_theta * sin_plate;
        const double radius = effectiveRadius(plate_id);
        const double px = x_(0, 0) + radius * cos_armor;
        const double py = x_(2, 0) + effectiveYOffset(plate_id);
        const double pz = x_(7, 0) + radius * sin_armor;
        // l/h 对位置的偏导（只作用于 {1,3} 组；h 向上为正、y 向下 ⇒ ∂py/∂h = −1）
        const double group = usesGroupDelta(plate_id) ? 1.0 : 0.0;
        const double dpx_dl = group * cos_armor;
        const double dpz_dl = group * sin_armor;
        const double dpy_dh = -group;

        const double horizontal_sq = px * px + pz * pz;
        const double horizontal = std::sqrt(horizontal_sq);
        const double distance_sq = horizontal_sq + py * py;
        const double distance = std::max(std::sqrt(distance_sq), 1e-9);

        const double dpx_ds = -radius * sin_plate;
        const double dpx_dc = radius * cos_plate;
        const double dpz_ds = radius * cos_plate;
        const double dpz_dc = radius * sin_plate;

        Matrix<4, kStateDim> jacobian;
        if (horizontal_sq > 1e-12) {
            const double dyaw_dpx = pz / horizontal_sq;
            const double dyaw_dpz = -px / horizontal_sq;
            jacobian(0, 0) = dyaw_dpx;
            jacobian(0, 7) = dyaw_dpz;
            jacobian(0, 4) = dyaw_dpx * dpx_ds + dyaw_dpz * dpz_ds;
            jacobian(0, 5) = dyaw_dpx * dpx_dc + dyaw_dpz * dpz_dc;
            jacobian(0, 9) = dyaw_dpx * dpx_dl + dyaw_dpz * dpz_dl;

            const double dpitch_dpx = -py * px / (distance_sq * horizontal);
            const double dpitch_dpz = -py * pz / (distance_sq * horizontal);
            const double dpitch_dpy = horizontal / distance_sq;
            jacobian(1, 0) = dpitch_dpx;
            jacobian(1, 2) = dpitch_dpy;
            jacobian(1, 7) = dpitch_dpz;
            jacobian(1, 4) = dpitch_dpx * dpx_ds + dpitch_dpz * dpz_ds;
            jacobian(1, 5) = dpitch_dpx * dpx_dc + dpitch_dpz * dpz_dc;
            jacobian(1, 9) = dpitch_dpx * dpx_dl + dpitch_dpz * dpz_dl;
            jacobian(1, 10) = dpitch_dpy * dpy_dh;
        }

        const double ddistance_dpx = px / distance;
        const double ddistance_dpy = py / distance;
        const double ddistance_dpz = pz / distance;
        jacobian(2, 0) = ddistance_dpx;
        jacobian(2, 2) = ddistance_dpy;
        jacobian(2, 7) = ddistance_dpz;
        jacobian(2, 4) = ddistance_dpx * dpx_ds + ddistance_dpz * dpz_ds;
        jacobian(2, 5) = ddistance_dpx * dpx_dc + ddistance_dpz * dpz_dc;
        jacobian(2, 9) = ddistance_dpx * dpx_dl + ddistance_dpz * dpz_dl;
        jacobian(2, 10) = ddistance_dpy * dpy_dh;

        const double angle_norm_sq = sin_theta * sin_theta + cos_theta * cos_theta;
        if (angle_norm_sq > 1e-12) {
            jacobian(3, 4) = cos_theta / angle_norm_sq;
            jacobian(3, 5) = -sin_theta / angle_norm_sq;
        }
        return jacobian;
    }

    template <int N>
    static bool invertDynamic(
        Matrix<N, N> matrix,
        Matrix<N, N>& inverse,
        int size) {
        if (size <= 0 || size > N) return false;
        inverse = Matrix<N, N>();
        for (int i = 0; i < size; ++i) inverse(i, i) = 1.0;

        for (int column = 0; column < size; ++column) {
            int pivot = column;
            for (int row = column + 1; row < size; ++row) {
                if (std::abs(matrix(row, column)) > std::abs(matrix(pivot, column))) {
                    pivot = row;
                }
            }
            if (std::abs(matrix(pivot, column)) < 1e-12) return false;

            if (pivot != column) {
                std::swap(matrix.data[pivot], matrix.data[column]);
                std::swap(inverse.data[pivot], inverse.data[column]);
            }

            const double diagonal = matrix(column, column);
            for (int col = 0; col < size; ++col) {
                matrix(column, col) /= diagonal;
                inverse(column, col) /= diagonal;
            }

            for (int row = 0; row < size; ++row) {
                if (row == column) continue;
                const double factor = matrix(row, column);
                if (std::abs(factor) < 1e-15) continue;
                for (int col = 0; col < size; ++col) {
                    matrix(row, col) -= factor * matrix(column, col);
                    inverse(row, col) -= factor * inverse(column, col);
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

        Q_ = Matrix<kStateDim, kStateDim>();
        const int position_idx[3] = {0, 2, 7};
        const int velocity_idx[3] = {1, 3, 8};
        for (int k = 0; k < 3; ++k) {
            const int p = position_idx[k];
            const int v = velocity_idx[k];
        Q_(p, p) = dt4 / 4.0 * qa;
        Q_(p, v) = dt3 / 2.0 * qa;
        Q_(v, p) = dt3 / 2.0 * qa;
        Q_(v, v) = dt2 * qa;
        // l/h：机械常数，但允许缓慢收敛（每步一个很小的方差）。取 4e-7 → 1s(70步)
        // 后 P 增长 ~3e-5，σ≈5mm，足够跟上机械偏差又不会被运动滞后带跑。
        Q_(9, 9) = cfg_.process_geometry;
        Q_(10, 10) = cfg_.process_geometry;
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
    Vec<kStateDim> x_;
    Matrix<kStateDim, kStateDim> P_;
    Matrix<kStateDim, kStateDim> Q_;
};

} // namespace auto_aim

#endif // AUTO_AIM_ARMOR_EKF_HPP
