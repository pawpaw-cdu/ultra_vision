// ArmorEKF 相位观测 updateAngle 的回归自检。

#include "Kalman/armor_ekf.hpp"

#include <cmath>
#include <cstdio>
#include <iostream>
#include <limits>

namespace
{
    constexpr double kPi = 3.14159265358979323846;

    int failures = 0;

    bool expect(bool condition, const char* message)
    {
        if (!condition)
        {
            std::cerr << "FAILED: " << message << std::endl;
            ++failures;
        }
        return condition;
    }

    auto_aim::ArmorEKF makeInitEkf()
    {
        auto_aim::ArmorEKF ekf;
        ekf.setConfig(auto_aim::ArmorEKF::Config{});
        auto_aim::Vec<3> obs;
        obs(0, 0) = 0.0;
        obs(1, 0) = -0.25;   // 板心比车心低 0.25 m（相机系 y 向下）
        obs(2, 0) = 3.0;
        ekf.init(obs, 1);
        return ekf;
    }

    double yawDeltaDeg(double before, double after)
    {
        return std::remainder(after - before, kPi) * 180.0 / kPi;
    }

    /// @brief 四条板位：注入"该板相位 +5°"，yaw 必须朝正方向动，且不能是 0。
    void testPlatePhaseGain()
    {
        const double desired = 5.0 * kPi / 180.0;
        for (int plate = 0; plate < 4; ++plate)
        {
            auto_aim::ArmorEKF ekf = makeInitEkf();
            const double before = ekf.getYaw();
            // 观测的语义：这块板的相位 = 预测相位 + 5°
            ekf.updateAngle(before + auto_aim::ArmorEKF::angleForPlate(plate) + desired, plate,
                            0.02, std::numeric_limits<double>::infinity());
            const double delta = yawDeltaDeg(before, ekf.getYaw());

            char msg[160];
            std::snprintf(msg, sizeof(msg),
                          "plate %d: 观测偏 +5deg 时底盘 yaw 必须朝正方向动（实测 %+.3f deg）",
                          plate, delta);
            expect(delta > 1.0, msg);
            std::snprintf(msg, sizeof(msg),
                          "plate %d: yaw 变化不该超过观测本身（实测 %+.3f deg）", plate, delta);
            expect(delta <= 5.0 + 1e-6, msg);
        }
    }

    /// @brief 四条板位：增益应一致（同一 P、同一噪声）—— 修复前 plate 0/2 是 0、plate 3 是负的。
    void testPlatePhaseGainUniform()
    {
        const double desired = 5.0 * kPi / 180.0;
        double gain[4] = {0.0, 0.0, 0.0, 0.0};
        for (int plate = 0; plate < 4; ++plate)
        {
            auto_aim::ArmorEKF ekf = makeInitEkf();
            const double before = ekf.getYaw();
            ekf.updateAngle(before + auto_aim::ArmorEKF::angleForPlate(plate) + desired, plate,
                            0.02, std::numeric_limits<double>::infinity());
            gain[plate] = yawDeltaDeg(before, ekf.getYaw()) / (desired * 180.0 / kPi);
        }
        for (int plate = 1; plate < 4; ++plate)
        {
            char msg[160];
            std::snprintf(msg, sizeof(msg),
                          "plate %d 的增益必须与 plate 0 一致（%+.3f vs %+.3f）",
                          plate, gain[plate], gain[0]);
            expect(std::abs(gain[plate] - gain[0]) < 0.05, msg);
        }
    }

}

int main()
{
    testPlatePhaseGain();
    testPlatePhaseGainUniform();

    if (failures == 0)
    {
        std::cout << "armor_ekf_angle_test: all checks passed" << std::endl;
        return 0;
    }
    std::cerr << "armor_ekf_angle_test: " << failures << " check(s) failed" << std::endl;
    return 1;
}
