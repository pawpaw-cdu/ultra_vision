#ifndef AUTO_AIM_ENERGY_SUPPORT_EKF_HPP
#define AUTO_AIM_ENERGY_SUPPORT_EKF_HPP

// Minimal extended Kalman filter used by the energy-rune estimator.
//
// Ported from sp_vision_25 `tools/extended_kalman_filter.{hpp,cpp}` so the
// energy module is self-contained (no sp_vision build dependency). The only
// behavioural difference is that the chi-square bookkeeping of the original
// is dropped: nothing in the rune pipeline consumed it.

#include <Eigen/Dense>
#include <functional>

namespace auto_aim::energy
{
    class ExtendedKalmanFilter
    {
    public:
        Eigen::VectorXd x;
        Eigen::MatrixXd P;

        ExtendedKalmanFilter() = default;

        ExtendedKalmanFilter(
            const Eigen::VectorXd& x0, const Eigen::MatrixXd& P0,
            std::function<Eigen::VectorXd(const Eigen::VectorXd&, const Eigen::VectorXd&)>
                x_add = [](const Eigen::VectorXd& a, const Eigen::VectorXd& b) { return a + b; });

        Eigen::VectorXd predict(const Eigen::MatrixXd& F, const Eigen::MatrixXd& Q);

        Eigen::VectorXd predict(
            const Eigen::MatrixXd& F, const Eigen::MatrixXd& Q,
            std::function<Eigen::VectorXd(const Eigen::VectorXd&)> f);

        Eigen::VectorXd update(
            const Eigen::VectorXd& z, const Eigen::MatrixXd& H, const Eigen::MatrixXd& R,
            std::function<Eigen::VectorXd(const Eigen::VectorXd&, const Eigen::VectorXd&)>
                z_subtract = [](const Eigen::VectorXd& a, const Eigen::VectorXd& b) {
                    return a - b;
                });

        Eigen::VectorXd update(
            const Eigen::VectorXd& z, const Eigen::MatrixXd& H, const Eigen::MatrixXd& R,
            std::function<Eigen::VectorXd(const Eigen::VectorXd&)> h,
            std::function<Eigen::VectorXd(const Eigen::VectorXd&, const Eigen::VectorXd&)>
                z_subtract = [](const Eigen::VectorXd& a, const Eigen::VectorXd& b) {
                    return a - b;
                });

    private:
        Eigen::MatrixXd I_;
        std::function<Eigen::VectorXd(const Eigen::VectorXd&, const Eigen::VectorXd&)> x_add_;
    };

    inline ExtendedKalmanFilter::ExtendedKalmanFilter(
        const Eigen::VectorXd& x0, const Eigen::MatrixXd& P0,
        std::function<Eigen::VectorXd(const Eigen::VectorXd&, const Eigen::VectorXd&)> x_add)
        : x(x0),
          P(P0),
          I_(Eigen::MatrixXd::Identity(x0.rows(), x0.rows())),
          x_add_(std::move(x_add))
    {
    }

    inline Eigen::VectorXd ExtendedKalmanFilter::predict(
        const Eigen::MatrixXd& F, const Eigen::MatrixXd& Q)
    {
        return predict(F, Q, [&](const Eigen::VectorXd& state) { return F * state; });
    }

    inline Eigen::VectorXd ExtendedKalmanFilter::predict(
        const Eigen::MatrixXd& F, const Eigen::MatrixXd& Q,
        std::function<Eigen::VectorXd(const Eigen::VectorXd&)> f)
    {
        P = F * P * F.transpose() + Q;
        x = f(x);
        return x;
    }

    inline Eigen::VectorXd ExtendedKalmanFilter::update(
        const Eigen::VectorXd& z, const Eigen::MatrixXd& H, const Eigen::MatrixXd& R,
        std::function<Eigen::VectorXd(const Eigen::VectorXd&, const Eigen::VectorXd&)> z_subtract)
    {
        return update(z, H, R, [&](const Eigen::VectorXd& state) { return H * state; },
                      std::move(z_subtract));
    }

    inline Eigen::VectorXd ExtendedKalmanFilter::update(
        const Eigen::VectorXd& z, const Eigen::MatrixXd& H, const Eigen::MatrixXd& R,
        std::function<Eigen::VectorXd(const Eigen::VectorXd&)> h,
        std::function<Eigen::VectorXd(const Eigen::VectorXd&, const Eigen::VectorXd&)> z_subtract)
    {
        const Eigen::MatrixXd K = P * H.transpose() * (H * P * H.transpose() + R).inverse();
        // Joseph form keeps the covariance symmetric and positive definite.
        P = (I_ - K * H) * P * (I_ - K * H).transpose() + K * R * K.transpose();
        x = x_add_(x, K * z_subtract(z, h(x)));
        return x;
    }
} // namespace auto_aim::energy

#endif // AUTO_AIM_ENERGY_SUPPORT_EKF_HPP
