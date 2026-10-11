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

#include "Kalman/linalg.hpp"   // Matrix / Vec（从本文件抽出）
#include "Kalman/ypd_observation.hpp"
#include "common/chassis_model.hpp"   // armorPhaseOffset（板相位约定，滤波器/模型同源）
#include "common/types.hpp"    // LocalFitResult（局部拟合器输出）

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
        /// l/h（分组长短轴）的过程噪声方差与初始方差（见 kStateDim 注释）。
        // l/h 的过程噪声与硬边界。实车实测（model9.csv）：真实两组深度差只有 ~2.3cm，
        // 但状态 l 在"l ↔ 车心/深度"的耦合里被过拟合推到 −8.6cm（每帧 PnP 位置噪声
        // ±12cm，滤波器能拿 l 去凑），把重投影带坏。所以：
        //   · 过程噪声放慢到 1e-7（1s≈1700步 → σ≈1.3cm，只跟真正的机械偏差）；
        //   · 硬边界收到 ±5cm（机器人前后/左右半径差不可能更大）。
        // 这样 l/h 只做"缓慢的几何修正"，不会去追观测噪声。
        double process_geometry = 1e-7;
        double geometry_delta_max = 0.05;   // |l|,|h| 硬边界（m）
        double P_geometry = 2.5e-3;    // 初始 σ=5cm，允许从 0 收敛到真实差值
        // IESEKF（上科大 2026 §5B.5）：更新内迭代次数。每次用**更新后的状态**
        // 重新线性化观测函数，再更新一次；收敛后相当于在真值附近展开，比只在
        // 先验处展开一次更准。代价是 2-5 倍的更新计算（相对整条链可忽略）。
        // 1 = 关闭（退化为标准 EKF）。
        int iekf_iterations = 3;
        double meas_noise = 0.01;
        double y_noise_mult = 30.0;
        double P_min = 1e-3;
        // Vertical armor-center offsets in the chassis/home frame. Real robot
        // armor sets normally leave these at zero; simulator models can define
        // plate-specific offsets instead of forcing one shared y coordinate.
        std::array<double, 4> armor_y_offsets{{0.0, 0.0, 0.0, 0.0}};
        // 四块板各自到**车心**的水平距离（米）：真车底盘是矩形，前后 ≠ 左右。
        // 0 = 该块板用 R（默认四块相等 → 圆，与仿真靶/旧行为一致）。
        // 由 Tracker 从 ChassisGeometry 同步过来（见 common/chassis_model.hpp）。
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

    /// @brief 只改回转半径（在线慢层用）。**不动状态/协方差** —— 半径是几何量，
    ///        换一个值只是让 armorPosition() 把板摆在别的半径上，
    ///        不需要（也不能）像 setConfig 那样 reset()，否则每次更新半径都丢跟踪。
    void setArmorRadius(double radius) {
        if (radius > 0.05 && radius < 1.0) cfg_.R = radius;
    }
    double armorRadius() const { return cfg_.R; }
    /// @brief 第 plate 块板的回转半径（矩形底盘 → 逐板不同；0 表示回落到 R）。
    double plateRadius(int plate) const {
        if (plate < 0 || plate >= 4) return cfg_.R;
        const double v = cfg_.plate_distance[static_cast<std::size_t>(plate)];
        return v > 0.0 ? v : cfg_.R;
    }
    /// @brief 该块板（1,3 = 前后组）是否受 l/h 影响（sp_vision_25 的 use_l_h）。
    static bool usesGroupDelta(int plate) { return plate == 1 || plate == 3; }
    /// @brief **l/h 生效后**该板的回转半径：R_plate + (前后组 ? l : 0)。
    double effectiveRadius(int plate) const {
        const double r = plateRadius(plate);
        return usesGroupDelta(plate) ? r + x_(9, 0) : r;
    }
    /// @brief **l/h 生效后**该板的竖直偏置（世界系 y 向下；h 向上为正 ⇒ 前后组 − h）。
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
    ///        底盘模型的 plate_height（向上为正）取负号同步到这里，二者必须一致，
    ///        否则"滤波器摆的板"和"画出来的板"会差一个高度。
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
        // 与底盘模型（common/chassis_model.hpp 的 armorPhaseOffset）**同一个约定**：
        // k=1 在 0°（正面）、k=0 在 +90°、k=2 在 −90°、k=3 在 180°。两处必须一致，
        // 否则"滤波器摆的板"和"画出来的板"会差 90°/180°。
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
        // （量级 ≈ P_ω·dt），随后第一帧的 yaw 量测就把它当"转速信息"吃进去 ——
        // 实车实测（tools/replay_dataset --dump，静止靶重捕那一帧）：
        // P_ω=100 时 ω 被一帧拉到 -199°/s，乘 386 ms 提前量就是瞄点甩 5~8°，
        // 下一帧再弹回来（云台上就是"抽一下"）。改成 4（σ_ω = 2 rad/s = 115°/s，
        // 仍是小陀螺量级）后，同一帧只到 -6°/s。真正的转速由多帧
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
    ///
    /// 这是上科大 2026 §5B.4 推荐的关联判据，也是本项目 `tracker.matchCost` 里
    /// 那套"手写代价 + 固定/自适应角度门限"的正统替代：
    ///   · S 已经按每个维度的**真实噪声/不确定度**归一化（角度是 rad、距离是 m，
    ///     而且距离噪声随距离平方增长），所以深度方向的正常抖动不会被误判成
    ///     "换了块板"，角度方向真的对不上又会被敏锐拒绝；
    ///   · 门限就是卡方门限 χ²(4, 95%) ≈ 9.488，不再需要按 ω 手调角度；
    ///   · 初始化/丢帧/刚重捕时 P 大 → S 大 → 门限自动变宽；收敛后 P 小 → 收紧。
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

        std::vector<YpdObservation> accepted;
        std::vector<Vec<4>> residuals;
        std::vector<Matrix<4, kStateDim>> jacobians;
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
            accepted.push_back(observation);
            residuals.push_back(residual);
            jacobians.push_back(jacobian);
        }

        if (accepted.empty()) return 0.0;

        const int measurement_dim = 4 * static_cast<int>(accepted.size());
        std::vector<double> noise_diagonal(measurement_dim, 0.0);

        for (std::size_t observation_index = 0;
             observation_index < accepted.size(); ++observation_index) {
            const int row_offset = 4 * static_cast<int>(observation_index);
            const YpdObservation& observation = accepted[observation_index];
            noise_diagonal[row_offset + 0] = std::max(observation.yaw_noise, 1e-8);
            noise_diagonal[row_offset + 1] = std::max(observation.pitch_noise, 1e-8);
            noise_diagonal[row_offset + 2] = std::max(observation.distance_noise, 1e-8);
            noise_diagonal[row_offset + 3] = std::max(observation.armor_yaw_noise, 1e-8);
        }

        // ---- IESEKF：在更新后的状态处重新线性化（迭代 2-5 次）--------------
        // 迭代式（上科大 2026 §5B.5）：
        //   r_k = z − h(x_k) + H(x_k)·(x_k − x_prior)
        //   x_{k+1} = x_prior + K(x_k)·r_k,   K 用先验协方差 P_prior 算
        // 迭代 1 次（k=0，x_k=x_prior）退化为标准 EKF。协方差只在最后用最后一次
        // 的 K/H 做一次 Joseph 更新（避免迭代中重复收缩 P）。
        const int iterations = std::max(1, cfg_.iekf_iterations);
        const Vec<kStateDim> x_prior = x_;
        const Matrix<kStateDim, kStateDim> P_prior = P_;
        std::vector<std::vector<double>> H(
            measurement_dim, std::vector<double>(kStateDim, 0.0));
        std::vector<double> residual(measurement_dim, 0.0);
        std::vector<std::vector<double>> gain(kStateDim, std::vector<double>(measurement_dim, 0.0));
        bool update_ok = false;

        for (int iteration = 0; iteration < iterations; ++iteration) {
            // 在当前状态 x_k（成员 x_）处重新算 H(x_k) 与残差。
            for (std::size_t observation_index = 0;
                 observation_index < accepted.size(); ++observation_index) {
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
                    H[row_offset + row] = std::vector<double>(kStateDim, 0.0);
                    for (int column = 0; column < kStateDim; ++column) {
                        H[row_offset + row][column] = jacobian(row, column);
                    }
                    residual[row_offset + row] = innovation_residual(row, 0);
                }
            }

            std::vector<std::vector<double>> innovation(
                measurement_dim, std::vector<double>(measurement_dim, 0.0));
            for (int row = 0; row < measurement_dim; ++row) {
                for (int column = 0; column < measurement_dim; ++column) {
                    double value = 0.0;
                    for (int state_row = 0; state_row < kStateDim; ++state_row) {
                        for (int state_column = 0; state_column < kStateDim; ++state_column) {
                            value += H[row][state_row] * P_prior(state_row, state_column) *
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
                kStateDim, std::vector<double>(measurement_dim, 0.0));
            for (int state_row = 0; state_row < kStateDim; ++state_row) {
                for (int measurement = 0; measurement < measurement_dim; ++measurement) {
                    double value = 0.0;
                    for (int state_column = 0; state_column < kStateDim; ++state_column) {
                        value += P_prior(state_row, state_column) * H[measurement][state_column];
                    }
                    state_measurement_covariance[state_row][measurement] = value;
                }
            }

            for (int state_row = 0; state_row < kStateDim; ++state_row) {
                for (int output_measurement = 0; output_measurement < measurement_dim;
                     ++output_measurement) {
                    double value = 0.0;
                    for (int input_measurement = 0; input_measurement < measurement_dim;
                         ++input_measurement) {
                        value +=
                            state_measurement_covariance[state_row][input_measurement] *
                            innovation_inverse[input_measurement][output_measurement];
                    }
                    gain[state_row][output_measurement] = value;
                }
            }

            // x_{k+1} = x_prior + K·r_k（从先验出发，而不是在上一次结果上累加）。
            for (int state_row = 0; state_row < kStateDim; ++state_row) {
                double value = x_prior(state_row, 0);
                for (int measurement = 0; measurement < measurement_dim; ++measurement) {
                    value += gain[state_row][measurement] * residual[measurement];
                }
                x_(state_row, 0) = value;
            }
            update_ok = true;
        }
        if (!update_ok) return 0.0;

        std::vector<std::vector<double>> transform(
            kStateDim, std::vector<double>(kStateDim, 0.0));
        for (int row = 0; row < kStateDim; ++row) {
            transform[row][row] = 1.0;
            for (int column = 0; column < kStateDim; ++column) {
                for (int measurement = 0; measurement < measurement_dim; ++measurement) {
                    transform[row][column] -= gain[row][measurement] * H[measurement][column];
                }
            }
        }

        std::vector<std::vector<double>> covariance(kStateDim, std::vector<double>(kStateDim, 0.0));
        for (int row = 0; row < kStateDim; ++row) {
            for (int column = 0; column < kStateDim; ++column) {
                double value = 0.0;
                for (int i = 0; i < kStateDim; ++i) {
                    for (int j = 0; j < kStateDim; ++j) {
                        value += transform[row][i] * P_(i, j) * transform[column][j];
                    }
                }
                covariance[row][column] = value;
            }
        }

        for (int row = 0; row < kStateDim; ++row) {
            for (int column = 0; column < kStateDim; ++column) {
                double value = 0.0;
                for (int measurement = 0; measurement < measurement_dim; ++measurement) {
                    value += gain[row][measurement] * noise_diagonal[measurement] *
                        gain[column][measurement];
                }
                covariance[row][column] += value;
            }
        }

        for (int row = 0; row < kStateDim; ++row) {
            for (int column = 0; column < kStateDim; ++column) {
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

        // A wrong plate assignment can look plausible in image space but be
        // inconsistent with the current chassis covariance. Reject it before
        // it contaminates the estimate.
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
        // (sin_total, cos_total) —— 后者只在 angleForPlate(plate_id) == 0
        // （本相位表里就是 plate 1）时才相等。
        // 2026-10-11 修：原来写的是 (cos_total, -sin_total)，导致 plate 0/2 的
        // 更新增益恰好为 0（yaw 完全不动）、plate 3 反号；而 updateWithLocalFit
        // 固定传 plate_id=0 → **局部拟合器的 yaw 观测整条链路是空操作**。
        // 有限差分与直接注入双重验证见 docs/local_fit_tracker_responsibility_audit.md
        // 审计项 D（修复前：plate 0/2 → +0.000°，plate 3 → −4.751°，plate 1 → +4.751°）。
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

    // 局部拟合器（小陀螺 θ/ω 拟合）的强观测注入。
    //
    // 正面观测时单帧 PnP 的 yaw 病态（左右边像素差趋零），此时不应喂 updateYpd 的
    // yaw 分量；改用拟合器给出的**绝对 yaw + 角速度**做时序强观测。这里按质量缩放
    // 观测噪声并走 NIS 门控（沿用 updateAngle / updateOmega，保留 ω↔sin/cos 交叉协方差，
    // 不覆盖状态）。
    //
    // 约定：`fit.yaw` 已是 **EKF 坐标系**下的 chassis yaw（plate 0 相位）。相机系→
    // EKF 系的换算由调用方（Tracker::applyLocalFit）完成。返回综合有效权重（0 表示未采用）。
    /// @brief 相位表里"零偏移"的那块板：angleForPlate(1) == 0。
    ///        想吃进"**底盘 yaw 本身**"这个观测量时，就用它当 plate_id ——
    ///        updateAngle() 的预测值 = θ + angleForPlate(plate_id) = θ。
    ///        为什么需要它：updateAngle 的语义是"某块板的相位"，直接传 0 表示
    ///        "plate 0 的相位 = θ+90°"，和"底盘 yaw"差 90°（历史上就是把这两个
    ///        混在一起，才让 yaw 通道失效的）。
    static constexpr int kChassisYawPlateId = 1;

    double updateWithLocalFit(
        const LocalFitResult& fit, double yaw_noise = 0.02, double omega_noise = 0.05,
        double nis_threshold = std::numeric_limits<double>::infinity()) {
        if (!initialized_ || !fit.valid) return 0.0;
        const double q = std::clamp(fit.quality, 0.05, 1.0);
        // LocalFitResult::yaw 是**底盘 yaw**（与 angleForPlate 同一张相位表，见
        // estimation/local_trajectory_fitter.hpp 的 plate_yaw_offsets 注释），
        // 所以以"零偏移板"注入。fit.ref_armor_id 只用于调用方的门控/诊断 ——
        // 表统一之后这里不再需要相位换算（见审计报告 P1-1/P1-2 的说明）。
        const double yaw_w =
            updateAngle(fit.yaw, kChassisYawPlateId, yaw_noise / q, nis_threshold);
        const double omega_w =
            updateOmega(fit.omega, omega_noise / q, nis_threshold);
        return std::max(yaw_w, omega_w);
    }

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
    ///        新板的实测径向是**直接的**底盘相位观测（误差=PnP 那几度），用它重锚比让
    ///        滤波器慢慢追要稳得多。只动 sin/cos，ω 与协方差保留（yaw 方差略放宽）。
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

        // Position is a direct function of the [sin(theta), cos(theta)]
        // representation, so differentiate the rotation matrix directly.
        // Adding an atan2 chain here is incorrect.
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
