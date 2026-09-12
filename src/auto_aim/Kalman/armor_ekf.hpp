#ifndef AUTO_AIM_ARMOR_EKF_HPP
#define AUTO_AIM_ARMOR_EKF_HPP

#include <algorithm>
#include <array>
#include <cmath>
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
};

template <int N>
using Vec = Matrix<N, 1>;

// State: [x_c, vx, y_c, vy, sin(theta), cos(theta), omega, z_c, vz].
// The four armor plates are distributed on a circle in the x-z plane.
class ArmorEKF {
public:
    struct Config {
        double R = 0.21;
        double process_acc = 1.0;
        double process_omega = 2.0;
        double meas_noise = 0.01;
        double y_noise_mult = 30.0;
        double P_min = 1e-3;
    };

    ArmorEKF() { initMatrices(); }

    void setConfig(const Config& config) {
        cfg_ = config;
        initMatrices();
        reset();
    }

    void setDt(double dt) {
        if (std::abs(dt - cfg_dt_) > 1e-9) {
            cfg_dt_ = dt;
            buildQ();
        }
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
            sum_y += observation(1, 0);
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
        P_(4, 4) = 1e-2; P_(5, 5) = 1e-2;
        P_(6, 6) = 4.0;
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
        P_(4, 4) = 0.02; P_(5, 5) = 0.02;
        P_(6, 6) = 4.0;
        initialized_ = true;
    }

    void init(const Vec<3>& observation, int plate_id) {
        double angle = angleForPlate(plate_id);
        x_(0, 0) = observation(0, 0) - cfg_.R * std::cos(angle);
        x_(2, 0) = observation(1, 0);
        x_(7, 0) = observation(2, 0) - cfg_.R * std::sin(angle);
        x_(4, 0) = 0.0;
        x_(5, 0) = 1.0;
        x_(6, 0) = 0.0;
        for (int i = 0; i < 9; ++i) {
            for (int j = 0; j < 9; ++j) P_(i, j) = 0.0;
        }
        P_(0, 0) = 0.1; P_(2, 2) = 0.1; P_(7, 7) = 0.1;
        P_(1, 1) = 1.0; P_(3, 3) = 1.0; P_(8, 8) = 1.0;
        P_(4, 4) = 1.0; P_(5, 5) = 1.0;
        P_(6, 6) = 4.0;
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
        predicted(1, 0) = x_(2, 0);
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
        position(1, 0) = x_(2, 0);
        position(2, 0) = x_(7, 0) + cfg_.R * (sin_theta * cos_plate + cos_theta * sin_plate);
        return position;
    }

    Vec<3> centerFromPlate(const Vec<3>& observation, int plate_id) const {
        const double angle = getYaw() + angleForPlate(plate_id);
        Vec<3> center;
        center(0, 0) = observation(0, 0) - cfg_.R * std::cos(angle);
        center(1, 0) = observation(1, 0);
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
    void initMatrices() {
        cfg_dt_ = 0.01;
        buildQ();
    }

    void buildQ() {
        const double dt = cfg_dt_;
        const double qa = cfg_.process_acc;
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
        Q_(6, 6) = dt2 * qw;
        Q_(4, 4) = dt2 * qw * 0.5;
        Q_(5, 5) = dt2 * qw * 0.5;
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
