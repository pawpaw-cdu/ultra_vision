#ifndef AUTO_AIM_ARMOR_EKF_HPP
#define AUTO_AIM_ARMOR_EKF_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

namespace ultra_vision {

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

} // namespace ultra_vision

#endif // AUTO_AIM_ARMOR_EKF_HPP
