// ArmorEKF 相位观测（updateAngle / updateWithLocalFit）的回归自检。
//
// 为什么单独钉这一组：`updateAngle` 的雅可比一度写成"旋转之后"的 (cos(θ+p),
// −sin(θ+p))，而不是状态自己的 (cosθ, −sinθ)。结果是——
//   · plate 0 / 2 → 更新增益恰好为 0（yaw 完全不动）
//   · plate 3     → 增益 −1（把 yaw 往反方向推，正反馈）
//   · plate 1     → 正确（唯一一个 angleForPlate == 0 的板）
// 而 `updateWithLocalFit` 固定传 plate_id=0 → **局部拟合器的 yaw 观测整条链路
// 是空操作**，依赖它的三套机制（模 90° 吸附 / yaw 常值偏置学习 / 按 |ω| 混合）
// 全部空转。这个 bug 在真机上表现为"开了 use_yaw 却毫无变化"。
// 见 docs/local_fit_tracker_responsibility_audit.md 审计项 D。
//
// 判据（对每个 plate_id）：告诉滤波器"这块板的相位比预测大 5°"，
// 底盘 yaw 必须**朝同方向**移动，且增益接近 1（K = P/(P+R) 不会正好 1）。

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

    /// @brief updateWithLocalFit 的 yaw 通道必须真的能推动状态（修复前完全不动）。
    void testLocalFitYawChannel()
    {
        auto_aim::ArmorEKF ekf = makeInitEkf();
        const double before = ekf.getYaw();

        auto_aim::LocalFitResult fit;
        fit.valid = true;
        fit.quality = 1.0;
        fit.ref_armor_id = 1;
        fit.yaw = before + 0.05;   // 拟合说"底盘 yaw 就是当前值 +2.86°"
        fit.omega = ekf.getOmega();   // ω 不动，隔离 yaw 通道
        ekf.updateWithLocalFit(fit, 0.02, 0.05, std::numeric_limits<double>::infinity());

        const double delta = yawDeltaDeg(before, ekf.getYaw());
        expect(delta > 0.5, "updateWithLocalFit 的 yaw 通道必须推动底盘 yaw（修复前恒为 0）");
        expect(delta <= 0.05 * 180.0 / kPi + 1e-6,
               "updateWithLocalFit 的 yaw 变化不该超过观测残差");
    }

    /// @brief ω 通道独立可用（与 yaw 无关），避免以后把两条通道又绑在一起。
    void testLocalFitOmegaChannel()
    {
        auto_aim::ArmorEKF ekf = makeInitEkf();
        const double before = ekf.getOmega();

        auto_aim::LocalFitResult fit;
        fit.valid = true;
        fit.quality = 1.0;
        fit.ref_armor_id = 1;
        fit.yaw = ekf.getYaw();       // yaw 不动
        fit.omega = before + 2.0;
        ekf.updateWithLocalFit(fit, 0.02, 0.05, std::numeric_limits<double>::infinity());

        expect(ekf.getOmega() > before + 0.5, "updateWithLocalFit 的 omega 通道必须能推动角速度");
    }
}

int main()
{
    testPlatePhaseGain();
    testPlatePhaseGainUniform();
    testLocalFitYawChannel();
    testLocalFitOmegaChannel();

    if (failures == 0)
    {
        std::cout << "armor_ekf_angle_test: all checks passed" << std::endl;
        return 0;
    }
    std::cerr << "armor_ekf_angle_test: " << failures << " check(s) failed" << std::endl;
    return 1;
}
