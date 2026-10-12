// 局部拟合 yaw 质量门的回归自检（estimation/local_fit_gate.hpp）。
//
// 钉住的是本轮（2026-10-12）用实车检测流量出来的分层结论：
// 拟合 rmse < 3px → yaw σ≈0.6°（可用）；3~8px → 9.7°；8~16px → 22.6°（不可用）。
// 而 ω 在所有档都稳 —— 所以门控必须**对 yaw 严、对 ω 松**，这一条要写死在测试里，
// 否则以后有人"顺手统一门限"就会把好不容易修好的 ω 通道一起关掉。
//
// 纯头文件，不依赖 Tracker / EKF / Ceres。

#include "estimation/local_fit_gate.hpp"

#include <cmath>
#include <cstdio>
#include <iostream>

namespace
{
    constexpr double kPi = 3.14159265358979323846;
    constexpr double kDeg = kPi / 180.0;
    int failures = 0;

    bool expect(bool ok, const char* what)
    {
        if (!ok)
        {
            std::cerr << "FAILED: " << what << std::endl;
            ++failures;
        }
        return ok;
    }

    auto_aim::LocalFitResult makeFit(double rmse_px, double omega, double span_deg)
    {
        auto_aim::LocalFitResult f;
        f.valid = true;
        f.reprojection_rmse_px = rmse_px;
        f.omega = omega;
        f.phase_span_rad = span_deg * kDeg;
        return f;
    }

    void testYawGateLayers()
    {
        auto_aim::LocalFitGateConfig cfg;   // 默认 3px / 20°
        struct Case { double rmse; double span; auto_aim::LocalFitYawVerdict want; const char* tag; };
        const Case cases[] = {
            {0.5, 60.0, auto_aim::LocalFitYawVerdict::ok, "好拟合+足跨度 → 放行"},
            {2.9, 25.0, auto_aim::LocalFitYawVerdict::ok, "边界内 → 放行"},
            {3.5, 60.0, auto_aim::LocalFitYawVerdict::bad_rmse, "rmse 超限 → 拦（实测 σ_ψ≈10°）"},
            {12.0, 90.0, auto_aim::LocalFitYawVerdict::bad_rmse, "实测 22°那一档 → 拦"},
            {0.5, 10.0, auto_aim::LocalFitYawVerdict::bad_span, "跨度不足 → 拦（平谷）"},
            {2.0, 15.0, auto_aim::LocalFitYawVerdict::bad_span, "跨度差一点 → 拦"},
        };
        for (const Case& c : cases)
        {
            const auto v = auto_aim::evaluateLocalFitYaw(makeFit(c.rmse, 3.0, c.span), cfg);
            if (!expect(v == c.want, c.tag))
                std::printf("  rmse=%.1f span=%.0f° → %s（期望 %s）\n", c.rmse, c.span,
                            auto_aim::localFitYawVerdictName(v),
                            auto_aim::localFitYawVerdictName(c.want));
        }
    }

    void testOmegaGateIsIndependentOfSpan()
    {
        // ω 与 yaw 的门控**不完全对称**：ω 不看相位跨度（静止时 ω≈0 是有效观测，
        // 正好压掉单帧 PnP 造出来的假转速），但同样要看 rmse。
        auto_aim::LocalFitGateConfig cfg;
        const auto static_good = makeFit(2.0, 0.01, 3.0);   // 静止但拟合好
        expect(auto_aim::localFitOmegaUsable(static_good, cfg), "静止+好拟合：ω 可用（跨度为 0 也不拦）");
        expect(auto_aim::evaluateLocalFitYaw(static_good, cfg) ==
                   auto_aim::LocalFitYawVerdict::bad_span,
               "同一份结果 yaw 因跨度不足被拦");
        const auto bad_rmse = makeFit(15.0, 4.5, 60.0);
        expect(!auto_aim::localFitOmegaUsable(bad_rmse, cfg), "rmse 差：ω 也被拦");
        auto_aim::LocalFitResult invalid;
        expect(!auto_aim::localFitOmegaUsable(invalid, cfg), "无效结果 ω 不可用");
        auto_aim::LocalFitGateConfig off;
        off.omega_max_rmse_px = 0.0;   // A/B：不门
        expect(auto_aim::localFitOmegaUsable(bad_rmse, off), "omega_max_rmse_px=0 时退回旧行为");
    }

    void testRealMeasuredDistribution()
    {
        // 用实测的三档分布跑一遍，验证"放行的那一档"确实只有 σ≈0.6° 那一档：
        // 8~16px 那一档（占实测 68%）必须**全部**被拦。
        auto_aim::LocalFitGateConfig cfg;
        int passed = 0, blocked = 0;
        for (double rmse : {2.0, 2.5, 3.0, 5.0, 12.0, 15.9})
        {
            const auto v = auto_aim::evaluateLocalFitYaw(makeFit(rmse, 3.0, 60.0), cfg);
            (v == auto_aim::LocalFitYawVerdict::ok ? passed : blocked)++;
        }
        std::printf("  6 个 rmse 样本：放行 %d、拦下 %d（期望放行 3 = 2.0/2.5/3.0）\n", passed,
                    blocked);
        expect(passed == 3 && blocked == 3, "rmse 门槛恰好卡在 3px");
    }
}  // namespace

int main()
{
    testYawGateLayers();
    testOmegaGateIsIndependentOfSpan();
    testRealMeasuredDistribution();
    if (failures == 0)
    {
        std::cout << "local_fit_gate_test: all checks passed" << std::endl;
        return 0;
    }
    std::cerr << "local_fit_gate_test: " << failures << " check(s) failed" << std::endl;
    return 1;
}
