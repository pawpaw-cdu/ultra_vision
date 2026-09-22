#ifndef AUTO_AIM_LINALG_HPP
#define AUTO_AIM_LINALG_HPP

// 从 Kalman/armor_ekf.hpp 抽出的轻量定长线性代数（Matrix/Vec）。
// 纯搬家，数值与接口不变 —— 目的只是让 armor_ekf.hpp 回到"只讲一个 EKF"的职责。

#include <algorithm>
#include <array>
#include <cmath>

namespace auto_aim
{
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
}

#endif // AUTO_AIM_LINALG_HPP
