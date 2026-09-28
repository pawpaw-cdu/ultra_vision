#ifndef AUTO_AIM_ENERGY_SUPPORT_SINE_FITTER_HPP
#define AUTO_AIM_ENERGY_SUPPORT_SINE_FITTER_HPP

// RANSAC sine fitter for the large-rune angular velocity.
//
// The simulator drives the large rune with omega(t) = a*sin(w*t + phi) + b,
// a = 0.78..1.045, w = 1.884..2.0, b = 2.09 - a. This fitter recovers
// (A, omega, phi, C) from the EKF's spd channel so the aimer can integrate the
// true profile instead of extrapolating a constant rate.
//
// Ported from sp_vision_25 `tools/ransac_sine_fitter.{hpp,cpp}`.

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <deque>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

namespace auto_aim::energy
{
    class RansacSineFitter
    {
    public:
        struct Result
        {
            double A = 0.0;
            double omega = 0.0;
            double phi = 0.0;
            double C = 0.0;
            int inliers = 0;
        };

        Result best_result_;

        RansacSineFitter(int max_iterations, double threshold, double min_omega, double max_omega)
            : max_iterations_(max_iterations),
              threshold_(threshold),
              min_omega_(min_omega),
              max_omega_(max_omega),
              gen_(std::random_device{}())
        {
        }

        void addData(double t, double v)
        {
            if (!fit_data_.empty() && (t - fit_data_.back().first > 5.0)) fit_data_.clear();
            fit_data_.emplace_back(t, v);
        }

        void fit()
        {
            if (fit_data_.size() < 3) return;

            std::uniform_real_distribution<double> omega_dist(min_omega_, max_omega_);
            std::vector<std::size_t> indices(fit_data_.size());
            std::iota(indices.begin(), indices.end(), 0);

            for (int iter = 0; iter < max_iterations_; ++iter) {
                std::shuffle(indices.begin(), indices.end(), gen_);

                std::vector<std::pair<double, double>> sample;
                sample.reserve(3);
                for (int i = 0; i < 3; ++i) sample.push_back(fit_data_[indices[i]]);

                const double omega = omega_dist(gen_);
                Eigen::Vector3d params;
                if (!fitPartialModel(sample, omega, params)) continue;

                const double a1 = params(0);
                const double a2 = params(1);
                const double c = params(2);
                const double amplitude = std::sqrt(a1 * a1 + a2 * a2);
                const double phi = std::atan2(a2, a1);

                const int inliers = evaluateInliers(amplitude, omega, phi, c);
                if (inliers > best_result_.inliers) {
                    best_result_.A = amplitude;
                    best_result_.omega = omega;
                    best_result_.phi = phi;
                    best_result_.C = c;
                    best_result_.inliers = inliers;
                }
            }

            if (fit_data_.size() > 150) fit_data_.pop_front();
        }

        double sineFunction(double t, double a, double omega, double phi, double c) const
        {
            return a * std::sin(omega * t + phi) + c;
        }

    private:
        bool fitPartialModel(const std::vector<std::pair<double, double>>& sample, double omega,
                             Eigen::Vector3d& params) const
        {
            Eigen::MatrixXd design(sample.size(), 3);
            Eigen::VectorXd values(sample.size());
            for (std::size_t i = 0; i < sample.size(); ++i) {
                const double t = sample[i].first;
                design(static_cast<Eigen::Index>(i), 0) = std::sin(omega * t);
                design(static_cast<Eigen::Index>(i), 1) = std::cos(omega * t);
                design(static_cast<Eigen::Index>(i), 2) = 1.0;
                values(static_cast<Eigen::Index>(i)) = sample[i].second;
            }
            // Eigen 5.x 把 `bdcSvd` 做成了**模板**（参数是模板形参），而 Eigen ≤4（例如
            // Ubuntu 的 3.4.0）是**普通成员函数**（参数是运行时 unsigned int）。
            // 本工程要在两边都能编：macOS/Eigen 5.0 与 Ubuntu/Eigen 3.4。
#if EIGEN_VERSION_AT_LEAST(5, 0, 0)
            params = design.bdcSvd<Eigen::ComputeThinU | Eigen::ComputeThinV>().solve(values);
#else
            params = design.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(values);
#endif
            return params.allFinite();
        }

        int evaluateInliers(double a, double omega, double phi, double c) const
        {
            int count = 0;
            for (const auto& point : fit_data_) {
                const double predicted = a * std::sin(omega * point.first + phi) + c;
                if (std::abs(point.second - predicted) < threshold_) ++count;
            }
            return count;
        }

        int max_iterations_;
        double threshold_;
        double min_omega_;
        double max_omega_;
        std::mt19937 gen_;
        std::deque<std::pair<double, double>> fit_data_;
    };
} // namespace auto_aim::energy

#endif // AUTO_AIM_ENERGY_SUPPORT_SINE_FITTER_HPP
