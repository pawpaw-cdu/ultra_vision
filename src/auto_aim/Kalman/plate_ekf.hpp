#ifndef AUTO_AIM_PLATE_EKF_HPP
#define AUTO_AIM_PLATE_EKF_HPP

#include <algorithm>
#include <cmath>
#include <limits>

#include "Kalman/armor_ekf.hpp"   // 复用 Matrix / Vec / normalizeAngle

namespace auto_aim
{
    /// @brief 装甲板模型
    ///
    /// 直接跟踪**当前可见那块板**的中心与朝向，而不是整车：
    ///   状态 x = [x, vx, y, vy, z, vz, θ, ω]（8 维，笛卡尔 + 板朝向）
    ///   观测 z = [x, y, z, θ]（板心位置 + 板朝向，来自 PnP / YPD）
    /// θ 用普通角度表示，残差做 [-π,π] 归一化，所以 h 是线性的。
    ///
    /// 为什么要它：低速时整车模型的"半径 + 切向"是弱可观测方向，单板工况下会
    /// 稳定偏出板心十几厘米；直接跟踪板心没有这个问题。高速自旋时板心走圆周、
    /// 匀速假设失效，应切回整车模型（`ArmorEKF`）——两模型按 |ω| 自适应切换。
    class PlateEKF
    {
    public:
        struct Config
        {
            double process_acc = 1.0;        // 平移加速度方差 (m/s²)²
            double process_yaw_acc = 10.0;   // 角加速度方差 (rad/s²)²
            double meas_noise = 1e-4;        // 位置测量方差 m²
            double yaw_meas_noise = 4e-4;    // 朝向测量方差 rad²
            double y_noise_mult = 10.0;      // 竖直方向额外放大
            double P_min = 1e-6;
        };

        PlateEKF() { reset(); }
        explicit PlateEKF(const Config& config) : cfg_(config) { reset(); }

        void setConfig(const Config& config)
        {
            cfg_ = config;
            reset();
        }

        void reset()
        {
            x_ = Vec<8>();
            P_ = Matrix<8, 8>();
            for (int i = 0; i < 8; ++i) P_(i, i) = 1.0;
            Q_ = Matrix<8, 8>();
            initialized_ = false;
        }

        bool isInit() const { return initialized_; }
        double getYaw() const { return x_(6, 0); }
        double getOmega() const { return x_(7, 0); }
        Vec<3> getCenter() const
        {
            Vec<3> center;
            center(0, 0) = x_(0, 0);
            center(1, 0) = x_(2, 0);
            center(2, 0) = x_(4, 0);
            return center;
        }
        Vec<3> getVelocity() const
        {
            Vec<3> velocity;
            velocity(0, 0) = x_(1, 0);
            velocity(1, 0) = x_(3, 0);
            velocity(2, 0) = x_(5, 0);
            return velocity;
        }

        /// @brief 用一块板的中心 + 朝向初始化；速度/角速度置零、给较大方差。
        void init(const Vec<3>& center, double yaw)
        {
            x_ = Vec<8>();
            x_(0, 0) = center(0, 0);
            x_(2, 0) = center(1, 0);
            x_(4, 0) = center(2, 0);
            x_(6, 0) = yaw;
            for (int i = 0; i < 8; ++i) {
                for (int j = 0; j < 8; ++j) P_(i, j) = 0.0;
            }
            P_(0, 0) = 1e-2; P_(2, 2) = 1e-2; P_(4, 4) = 1e-2;
            P_(1, 1) = 1.0;  P_(3, 3) = 1.0;  P_(5, 5) = 1.0;
            P_(6, 6) = 0.4;
            P_(7, 7) = 4.0;
            initialized_ = true;
        }

        void setDt(double dt) { cfg_dt_ = dt; buildQ(); }

        void predict()
        {
            if (!initialized_) return;
            const double dt = cfg_dt_;
            Matrix<8, 8> F;
            for (int i = 0; i < 8; ++i) F(i, i) = 1.0;
            F(0, 1) = dt; F(2, 3) = dt; F(4, 5) = dt; F(6, 7) = dt;
            x_(0, 0) += x_(1, 0) * dt;
            x_(2, 0) += x_(3, 0) * dt;
            x_(4, 0) += x_(5, 0) * dt;
            x_(6, 0) += x_(7, 0) * dt;
            x_(6, 0) = ArmorEKF::normalizeAngle(x_(6, 0));
            P_ = F * P_ * F.transpose() + Q_;
        }

        /// @brief 4 维观测 [x, y, z, θ] 的 EKF 更新（h 线性，θ 残差做角度归一化）。
        ///        返回 NIS 权重（exp(-d²/2)）；被门限拒绝时返回 0。
        double update(const Vec<3>& position, double yaw, double nis_threshold,
                      double position_noise = -1.0, double yaw_noise = -1.0)
        {
            if (!initialized_) {
                init(position, yaw);
                return 1.0;
            }
            const double pos_noise = position_noise > 0.0 ? position_noise : cfg_.meas_noise;
            const double angle_noise = yaw_noise > 0.0 ? yaw_noise : cfg_.yaw_meas_noise;

            Matrix<4, 8> H;
            H(0, 0) = 1.0;
            H(1, 2) = 1.0;
            H(2, 4) = 1.0;
            H(3, 6) = 1.0;

            Vec<4> residual;
            residual(0, 0) = position(0, 0) - x_(0, 0);
            residual(1, 0) = position(1, 0) - x_(2, 0);
            residual(2, 0) = position(2, 0) - x_(4, 0);
            residual(3, 0) = ArmorEKF::normalizeAngle(yaw - x_(6, 0));

            Matrix<4, 4> R;
            R(0, 0) = pos_noise;
            R(1, 1) = pos_noise * cfg_.y_noise_mult;
            R(2, 2) = pos_noise;
            R(3, 3) = angle_noise;

            const Matrix<4, 4> S = H * P_ * H.transpose() + R;
            Matrix<4, 4> S_inverse;
            if (!Matrix<4, 4>::invert(S, S_inverse)) return 0.0;

            double nis = 0.0;
            for (int row = 0; row < 4; ++row) {
                for (int column = 0; column < 4; ++column) {
                    nis += residual(row, 0) * S_inverse(row, column) * residual(column, 0);
                }
            }
            if (nis > nis_threshold) return 0.0;

            const Matrix<8, 4> K = P_ * H.transpose() * S_inverse;
            x_ = x_ + K * residual;
            x_(6, 0) = ArmorEKF::normalizeAngle(x_(6, 0));

            Matrix<8, 8> identity;
            for (int i = 0; i < 8; ++i) identity(i, i) = 1.0;
            const Matrix<8, 8> IKH = identity - K * H;
            P_ = IKH * P_ * IKH.transpose() + K * R * K.transpose();
            for (int i = 0; i < 8; ++i) {
                if (P_(i, i) < cfg_.P_min) P_(i, i) = cfg_.P_min;
            }
            return std::exp(-0.5 * nis);
        }

        /// @brief 用 8 维状态直接覆盖（模型切换时从另一个模型无缝接管）。
        void seed(const Vec<3>& center, const Vec<3>& velocity, double yaw, double omega,
                  double position_covariance, double yaw_covariance)
        {
            x_ = Vec<8>();
            x_(0, 0) = center(0, 0);
            x_(2, 0) = center(1, 0);
            x_(4, 0) = center(2, 0);
            x_(1, 0) = velocity(0, 0);
            x_(3, 0) = velocity(1, 0);
            x_(5, 0) = velocity(2, 0);
            x_(6, 0) = ArmorEKF::normalizeAngle(yaw);
            x_(7, 0) = omega;
            for (int i = 0; i < 8; ++i) {
                for (int j = 0; j < 8; ++j) P_(i, j) = 0.0;
            }
            P_(0, 0) = position_covariance; P_(2, 2) = position_covariance;
            P_(4, 4) = position_covariance;
            P_(1, 1) = 1.0; P_(3, 3) = 1.0; P_(5, 5) = 1.0;
            P_(6, 6) = yaw_covariance;
            P_(7, 7) = 4.0;
            initialized_ = true;
        }

    private:
        void buildQ()
        {
            const double dt = cfg_dt_;
            const double dt2 = dt * dt;
            const double dt3 = dt2 * dt;
            const double dt4 = dt3 * dt;
            Q_ = Matrix<8, 8>();
            const int position_index[3] = {0, 2, 4};
            const int velocity_index[3] = {1, 3, 5};
            for (int axis = 0; axis < 3; ++axis) {
                const int p = position_index[axis];
                const int v = velocity_index[axis];
                Q_(p, p) = dt4 / 4.0 * cfg_.process_acc;
                Q_(p, v) = dt3 / 2.0 * cfg_.process_acc;
                Q_(v, p) = dt3 / 2.0 * cfg_.process_acc;
                Q_(v, v) = dt2 * cfg_.process_acc;
            }
            Q_(6, 6) = dt4 / 4.0 * cfg_.process_yaw_acc;
            Q_(6, 7) = dt3 / 2.0 * cfg_.process_yaw_acc;
            Q_(7, 6) = dt3 / 2.0 * cfg_.process_yaw_acc;
            Q_(7, 7) = dt2 * cfg_.process_yaw_acc;
        }

        Config cfg_;
        double cfg_dt_ = 0.01;
        bool initialized_ = false;
        Vec<8> x_;
        Matrix<8, 8> P_;
        Matrix<8, 8> Q_;
    };
}

#endif // AUTO_AIM_PLATE_EKF_HPP
