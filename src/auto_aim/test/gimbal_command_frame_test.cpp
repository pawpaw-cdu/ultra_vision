#include "control/gimbal_command_frame.hpp"

#include <cmath>
#include <iostream>

namespace
{
    bool expect(bool condition, const char* message)
    {
        if (!condition) std::cerr << "FAILED: " << message << std::endl;
        return condition;
    }

    bool near(double lhs, double rhs, double tolerance = 1e-9)
    {
        return std::abs(lhs - rhs) <= tolerance;
    }

    io::ImuSample sampleFrom(double w, double x, double y, double z)
    {
        io::ImuSample imu;
        const double norm = std::sqrt(w * w + x * x + y * y + z * z);
        imu.w = w / norm;
        imu.x = x / norm;
        imu.y = y / norm;
        imu.z = z / norm;
        imu.yaw = 0.0;
        imu.pitch = 0.0;
        return imu;
    }

    double wrapPi(double value)
    {
        while (value > M_PI) value -= 2.0 * M_PI;
        while (value <= -M_PI) value += 2.0 * M_PI;
        return value;
    }
}

int main()
{
    bool passed = true;

    // Δ=0（期望姿态 == 当前姿态）时，命令必须**正好等于当前反馈**：
    const double yaws[] = {0.0, 0.35, 1.5, -2.2};
    const double pitches[] = {-0.25, 0.0, 0.30, 0.62};
    for (double yaw : yaws) {
        for (double pitch : pitches) {
            // 构造一个"IMU 相对云台零位转了 90°"的安装：q = R(姿态) · R_mounting
            const auto base = auto_aim::gimbalToBase(yaw, pitch);
            const auto mounting = auto_aim::gimbalToBase(0.0, 1.5707963267948966);
            const auto world = auto_aim::mul(base, mounting);
            // 旋转矩阵 → 四元数（w,x,y,z）
            const double trace = world[0] + world[4] + world[8];
            double w = 0.5 * std::sqrt(std::max(0.0, 1.0 + trace));
            double x = 0.5 * std::sqrt(std::max(0.0, 1.0 + world[0] - world[4] - world[8]));
            double y = 0.5 * std::sqrt(std::max(0.0, 1.0 - world[0] + world[4] - world[8]));
            double z = 0.5 * std::sqrt(std::max(0.0, 1.0 - world[0] - world[4] + world[8]));
            if (world[7] - world[5] < 0.0) x = -x;   // (2,1)-(1,2)
            if (world[2] - world[6] < 0.0) y = -y;   // (0,2)-(2,0)
            if (world[3] - world[1] < 0.0) z = -z;   // (1,0)-(0,1)
            const io::ImuSample imu = sampleFrom(w, x, y, z);

            double az_feedback = 0.0;
            double el_feedback = 0.0;
            double az_cmd = 0.0;
            double el_cmd = 0.0;
            passed &= expect(auto_aim::imuWorldAngles(imu, az_feedback, el_feedback),
                             "imuWorldAngles must accept a unit quaternion");
            passed &= expect(auto_aim::desiredWorldAngles(yaw, pitch, yaw, pitch, imu, az_cmd,
                                                          el_cmd),
                             "desiredWorldAngles must accept a unit quaternion");
            passed &= expect(near(wrapPi(az_cmd - az_feedback), 0.0, 1e-9) &&
                                 near(el_cmd - el_feedback, 0.0, 1e-9),
                             "zero pose change must reproduce the measured feedback exactly");
        }
    }

    // 非单位四元数必须被拒绝，而不是拿垃圾值去下发。
    io::ImuSample bad;
    bad.w = 0.0;
    bad.x = 0.0;
    bad.y = 0.0;
    bad.z = 0.0;
    double az = 0.0;
    double el = 0.0;
    passed &= expect(!auto_aim::desiredWorldAngles(0.0, 0.0, 0.0, 0.0, bad, az, el),
                     "a degenerate quaternion must be rejected");

    if (!passed) return 1;
    std::cout << "gimbal_command_frame_test passed" << std::endl;
    return 0;
}
