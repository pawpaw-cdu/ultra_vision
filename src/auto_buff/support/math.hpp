#ifndef AUTO_AIM_ENERGY_SUPPORT_MATH_HPP
#define AUTO_AIM_ENERGY_SUPPORT_MATH_HPP

// Spherical/rectangular conversion helpers for the energy-rune estimator.
//
// The energy pipeline works in the same convention as sp_vision_25: a right
// handed, z-up frame where a point is described by ypd = (yaw, pitch,
// distance) and the rune pose by ypr = (yaw, pitch, roll).
//
// Ported from sp_vision_25 `tools/math_tools.{hpp,cpp}`, reduced to what the
// rune pipeline uses.

#include <Eigen/Dense>
#include <cmath>

namespace auto_aim::energy
{
    constexpr double kPi = 3.14159265358979323846;

    // O(1) wrap to (-pi, pi]. The loop version that the reference used spins
    // for hundreds of millions of iterations if a caller ever passes a huge
    // angle (a timestamp mix-up once did exactly that and stalled the loop).
    inline double limitRad(double angle)
    {
        if (!std::isfinite(angle)) return 0.0;
        double wrapped = std::fmod(angle + kPi, 2.0 * kPi);
        if (wrapped < 0.0) wrapped += 2.0 * kPi;
        return wrapped - kPi;
    }

    inline double limitMinMax(double input, double low, double high)
    {
        if (input > high) return high;
        if (input < low) return low;
        return input;
    }

    // ypr = (yaw, pitch, roll), intrinsic Z-Y-X: R = Rz(yaw) * Ry(pitch) * Rx(roll).
    inline Eigen::Matrix3d rotationMatrix(const Eigen::Vector3d& ypr)
    {
        const double roll = ypr[2];
        const double pitch = ypr[1];
        const double yaw = ypr[0];
        const double cy = std::cos(yaw), sy = std::sin(yaw);
        const double cp = std::cos(pitch), sp = std::sin(pitch);
        const double cr = std::cos(roll), sr = std::sin(roll);
        Eigen::Matrix3d R;
        R << cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr,
             sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr,
             -sp,     cp * sr,                cp * cr;
        return R;
    }

    // Inverse of rotationMatrix(): yaw about z, then pitch about y, then roll about x.
    inline Eigen::Vector3d matrixToYpr(const Eigen::Matrix3d& R)
    {
        const double yaw = std::atan2(R(1, 0), R(0, 0));
        const double pitch = std::atan2(-R(2, 0), std::hypot(R(2, 1), R(2, 2)));
        const double roll = std::atan2(R(2, 1), R(2, 2));
        return {limitRad(yaw), limitRad(pitch), limitRad(roll)};
    }

    inline Eigen::Vector3d xyz2ypd(const Eigen::Vector3d& xyz)
    {
        const double x = xyz[0], y = xyz[1], z = xyz[2];
        const double yaw = std::atan2(y, x);
        const double pitch = std::atan2(z, std::sqrt(x * x + y * y));
        const double distance = std::sqrt(x * x + y * y + z * z);
        return {yaw, pitch, distance};
    }

    inline Eigen::Matrix3d xyz2ypdJacobian(const Eigen::Vector3d& xyz)
    {
        const double x = xyz[0], y = xyz[1], z = xyz[2];
        const double r2 = x * x + y * y;
        const double r = std::sqrt(r2);
        const double d2 = r2 + z * z;
        const double d = std::sqrt(d2);

        Eigen::Matrix3d J;
        J << -y / r2, x / r2, 0.0,
             -(x * z) / (d2 * r), -(y * z) / (d2 * r), r / d2,
             x / d, y / d, z / d;
        return J;
    }

    inline Eigen::Vector3d ypd2xyz(const Eigen::Vector3d& ypd)
    {
        const double yaw = ypd[0], pitch = ypd[1], distance = ypd[2];
        return {distance * std::cos(pitch) * std::cos(yaw),
                distance * std::cos(pitch) * std::sin(yaw),
                distance * std::sin(pitch)};
    }

    inline Eigen::Matrix3d ypd2xyzJacobian(const Eigen::Vector3d& ypd)
    {
        const double yaw = ypd[0], pitch = ypd[1], distance = ypd[2];
        const double cy = std::cos(yaw), sy = std::sin(yaw);
        const double cp = std::cos(pitch), sp = std::sin(pitch);
        Eigen::Matrix3d J;
        J << -distance * cp * sy, -distance * sp * cy, cp * cy,
             distance * cp * cy, -distance * sp * sy, cp * sy,
             0.0, distance * cp, sp;
        return J;
    }
} // namespace auto_aim::energy

#endif // AUTO_AIM_ENERGY_SUPPORT_MATH_HPP
