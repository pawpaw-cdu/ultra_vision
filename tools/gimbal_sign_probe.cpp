// 下发角度的**语义探针**（真机，跑在 auto_aim **之前**，别同时跑）。
//
// 为什么需要它：本项目内部（估计器/瞄准/手眼外参）用"正 yaw = 右转"，而 C 板回传的
// yaw 可能与它反号（configs/serial.yaml 的 yaw_sign = -1）。下位机的命令帧与回传帧
// **必须同坐标系**（它拿 error = cmd − current 做闭环，`ext_gimbal_CV_ctrl_t` 里
// `yaw // yaw电机角度`），所以只反读、不反写 = 闭环符号反 → 有目标就朝一个方向一直转。
//
// 但 2026-09-29 第一次量的时候发现"命令 +3° → 回传变了 41°"，说明**不能只看一个稳态值**
// （稳态值会把"符号"和"零点差/尺度"混在一起）。所以这个探针分四段，逐段拿时间曲线：
//   A 静置不发      ：下位机自己在动吗（它有自己的自瞄/扫描逻辑就会动）
//   B 发"当前值"    ：命令有权威吗（同系同零点 → 应该基本不动）
//   C 发"当前值+3°" ：符号 + 尺度（看前 200 ms 的方向，和稳态增量）
//   D 发"当前值"    ：能不能拉回来
// 每段都按 100 Hz 下发、按 20 ms 采样回传，最后强制 mode=0。
//
// 用法：
//   ./build/gimbal_sign_probe [--config-dir configs] [--step 3] [--axis yaw|pitch|both]
//                             [--hold 2.0] [--quiet]
// 安全：台阶默认 3°、每段 ≤2 s，结束强制 mode=0（不控制）。
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include <yaml-cpp/yaml.h>

#include "config_loader.hpp"
#include "control/gimbal_command_frame.hpp"
#include "io/gimbal/gimbal.hpp"

namespace
{
    struct Sample
    {
        double t = 0.0;
        double yaw = 0.0;
        double pitch = 0.0;
        double imu_yaw = 0.0;
        double imu_pitch = 0.0;
        int mode = -1;
    };

    /// @brief 四元数 (w,x,y,z) → yaw/pitch（度）。只用来判断"命令比较的是哪个量"。
    void quatToYawPitch(const io::ImuSample& imu, double& yaw_deg, double& pitch_deg)
    {
        const double w = imu.w, x = imu.x, y = imu.y, z = imu.z;
        const double sin_pitch = 2.0 * (w * y - z * x);
        pitch_deg = std::asin(std::max(-1.0, std::min(1.0, sin_pitch))) * 180.0 / M_PI;
        yaw_deg = std::atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)) *
            180.0 / M_PI;
    }

    /// @brief 在 hold 秒内以 100 Hz 下发 (yaw_cmd, pitch_cmd)，同时按 20 ms 采样回传角。
    ///        control=false 时不下发任何帧（只看下位机自己的行为）。
    std::vector<Sample> runPhase(io::Gimbal& gimbal, bool control, double yaw_cmd,
                                 double pitch_cmd, double hold)
    {
        std::vector<Sample> out;
        const auto start = std::chrono::steady_clock::now();
        const auto deadline = start + std::chrono::duration<double>(hold);
        auto next_send = start;
        auto next_sample = start;
        while (std::chrono::steady_clock::now() < deadline) {
            const auto now = std::chrono::steady_clock::now();
            if (control && now >= next_send) {
                gimbal.send(true, false, yaw_cmd, 0.0, 0.0, pitch_cmd, 0.0, 0.0);
                next_send = now + std::chrono::milliseconds(10);   // 100 Hz
            }
            if (now >= next_sample) {
                const auto state = gimbal.state();
                Sample sample;
                sample.t = std::chrono::duration<double>(now - start).count();
                sample.yaw = state.yaw;
                sample.pitch = state.pitch;
                io::ImuSample imu;
                if (gimbal.imuAt(now, imu)) {
                    quatToYawPitch(imu, sample.imu_yaw, sample.imu_pitch);
                }
                sample.mode = static_cast<int>(gimbal.mode());
                out.push_back(sample);
                next_sample = now + std::chrono::milliseconds(20);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        return out;
    }

    void printProfile(const char* tag, const std::vector<Sample>& samples, bool on_pitch)
    {
        if (samples.empty()) return;
        const double base = on_pitch ? samples.front().pitch : samples.front().yaw;
        std::cout << "  " << tag << "（相对第一帧，度）：";
        for (std::size_t i = 0; i < samples.size(); i += 5) {   // 每 100 ms 一个点
            const double v = (on_pitch ? samples[i].pitch : samples[i].yaw) - base;
            std::cout << std::fixed << std::setprecision(1) << v * 180.0 / M_PI << " ";
        }
        std::cout << std::endl;
    }

    double changeAt(const std::vector<Sample>& samples, bool on_pitch, double after_seconds)
    {
        if (samples.empty()) return 0.0;
        const double base = on_pitch ? samples.front().pitch : samples.front().yaw;
        double best = 0.0;
        for (const auto& sample : samples) {
            if (sample.t < after_seconds) continue;
            const double v = (on_pitch ? sample.pitch : sample.yaw) - base;
            if (std::abs(v) > std::abs(best)) best = v;
        }
        return best * 180.0 / M_PI;
    }

    double finalAt(const std::vector<Sample>& samples, bool on_pitch)
    {
        if (samples.empty()) return 0.0;
        const double base = on_pitch ? samples.front().pitch : samples.front().yaw;
        const auto& last = samples.back();
        return ((on_pitch ? last.pitch : last.yaw) - base) * 180.0 / M_PI;
    }

    /// @brief 某个量的最终绝对值（度）。
    double finalValue(const std::vector<Sample>& samples, bool on_pitch, bool use_imu)
    {
        if (samples.empty()) return 0.0;
        const auto& last = samples.back();
        const double v = use_imu ? (on_pitch ? last.imu_pitch : last.imu_yaw)
                                 : (on_pitch ? last.pitch : last.yaw);
        return v * 180.0 / M_PI;
    }

    /// @brief `--q` 模式：只读不发，把四元数和"每个基轴在世界系里的方位/俯仰"打出来。
    ///        用来看：(1) 四元数是否可用（模长≈1、随云台平滑变化）；
    ///        (2) 哪个轴是瞄准方向、它和回传电机角是什么关系（符号/零点）。
    ///        零风险：全程不发任何控制帧。
    int runQuaternionDump(io::Gimbal& gimbal, double seconds)
    {
        std::cout << "只读模式（不发任何控制帧）。请**用手慢慢转 yaw（左右各 ~30°），"
                     "再慢慢抬落 pitch**；每 100 ms 一行。" << std::endl;
        std::cout << "t | 电机yaw° 电机pitch° | q(w,x,y,z) |q| | "
                     "各基轴在世界系的方位角/俯仰：x(a/E) y(a/E) z(a/E)" << std::endl;
        const auto start = std::chrono::steady_clock::now();
        auto next = start;
        while (std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() <
               seconds) {
            const auto now = std::chrono::steady_clock::now();
            if (now >= next) {
                next = now + std::chrono::milliseconds(100);
                io::ImuSample imu;
                const auto state = gimbal.state();
                const bool have = gimbal.latestImu(imu);
                std::cout << std::fixed << std::setprecision(3)
                          << std::chrono::duration<double>(now - start).count() << " | "
                          << state.yaw * 180.0 / M_PI << " " << state.pitch * 180.0 / M_PI
                          << " | ";
                if (!have) {
                    std::cout << "(没有 IMU 样本)" << std::endl;
                    continue;
                }
                const double w = imu.w, x = imu.x, y = imu.y, z = imu.z;
                const double norm = std::sqrt(w * w + x * x + y * y + z * z);
                std::cout << w << "," << x << "," << y << "," << z << " | " << norm << " | ";
                const double R[3][3] = {
                    {1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)},
                    {2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)},
                    {2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)}};
                for (int c = 0; c < 3; ++c) {
                    const double az = std::atan2(R[1][c], R[0][c]) * 180.0 / M_PI;
                    const double el = std::atan2(R[2][c], std::hypot(R[0][c], R[1][c])) *
                        180.0 / M_PI;
                    std::cout << (c == 0 ? "x" : (c == 1 ? "y" : "z")) << "(" << az << "/" << el
                              << ") ";
                }
                std::cout << std::endl;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        return 0;
    }

    // ---- 世界系（IMU）命令候选 -------------------------------------------------
    // 从 `--q` 的实测关系（基轴 x 的方位 ≈ 电机 yaw − 90°，俯仰 ≈ 电机 pitch − 2.3°）
    // 推出：下位机闭环比的是 IMU 系角，不是回传的电机角。下面用四元数算出这个角的
    // 四种约定（yaw/pitch 各两个符号），逐个"保持当前值"试，哪个不动就是它。
    struct WorldAngles
    {
        double az_x = 0.0;   // 基轴 x 在世界系的方位角
        double el_x = 0.0;   // 基轴 x 在世界系的俯仰角
    };

    // 和 node 下发时用的是**同一个**函数（src/auto_aim/control/gimbal_command_frame.hpp），
    // 免得探针量出来的约定和实际下发代码的口径漂移。
    bool worldAngles(const io::ImuSample& imu, WorldAngles& out)
    {
        return auto_aim::imuWorldAngles(imu, out.az_x, out.el_x);
    }

    /// @brief `--world` 模式：自动找出"命令发哪个系"，并打印该怎么写进代码。
    int runWorldFrameProbe(io::Gimbal& gimbal, double step_deg, double hold)
    {
        const double step = step_deg * M_PI / 180.0;
        // 候选：(yaw 符号, pitch 符号) × (世界系 x 轴 / 直接发电机角)
        struct Candidate {
            const char* name;
            int yaw_sign;
            int pitch_sign;
            bool use_world;
        };
        const Candidate candidates[] = {
            {"世界系 +az_x / +el_x", +1, +1, true},
            {"世界系 -az_x / +el_x", -1, +1, true},
            {"世界系 +az_x / -el_x", +1, -1, true},
            {"世界系 -az_x / -el_x", -1, -1, true},
            {"电机系 原样（对照）", +1, +1, false},
        };
        struct Result {
            double dyaw = 0.0;
            double dpitch = 0.0;
            double yaw_cmd = 0.0;
            double pitch_cmd = 0.0;
        };
        std::vector<Result> results;

        std::cout << "≈ 每个候选下发 1.5 s 的\"当前值\"，哪个保持不动就是它（step=" << step_deg
                  << "°）。" << std::endl;
        for (const auto& candidate : candidates) {
            io::ImuSample imu;
            const auto ref = gimbal.state();
            WorldAngles world;
            const bool have_q = gimbal.latestImu(imu) && worldAngles(imu, world);
            double yaw_cmd = 0.0, pitch_cmd = 0.0;
            if (candidate.use_world) {
                if (!have_q) {
                    std::cout << "  " << candidate.name << "：没有可用四元数，跳过" << std::endl;
                    results.push_back({0.0, 0.0, 0.0, 0.0});
                    continue;
                }
                yaw_cmd = candidate.yaw_sign * world.az_x;
                pitch_cmd = candidate.pitch_sign * world.el_x;
            } else {
                yaw_cmd = candidate.yaw_sign * ref.yaw;
                pitch_cmd = candidate.pitch_sign * ref.pitch;
            }
            const auto phase = runPhase(gimbal, true, yaw_cmd, pitch_cmd, hold);
            const double dyaw = changeAt(phase, false, hold * 0.5);
            const double dpitch = changeAt(phase, true, hold * 0.5);
            results.push_back({dyaw, dpitch, yaw_cmd, pitch_cmd});
            std::cout << "  " << candidate.name << "：yaw 偏 " << dyaw << "°，pitch 偏 " << dpitch
                      << "°；末帧 mode=" << (phase.empty() ? -1 : phase.back().mode) << std::endl;
            // 每个候选之后**不发任何帧**（mode=0 心跳都不发）：下位机自己接管就停住，
            // 不会因为"用错坐标系保持当前位置"而继续跑。
            runPhase(gimbal, false, 0.0, 0.0, 0.8);
        }

        // 选 |yaw|+|pitch| 偏差最小的当赢家
        std::size_t best = 0;
        double best_score = 1e9;
        for (std::size_t i = 0; i < results.size(); ++i) {
            const double score = std::abs(results[i].dyaw) + std::abs(results[i].dpitch);
            if (score < best_score) {
                best_score = score;
                best = i;
            }
        }
        std::cout << "==> 最像的约定：" << candidates[best].name << "（偏差 " << results[best].dyaw
                  << "°/" << results[best].dpitch << "°）" << std::endl;

        // 用赢家做方向台阶：命令 +step，看电机角怎么动 → 定"我们的内部增量该加还是该减"
        if (best_score < 30.0) {
            io::ImuSample imu;
            WorldAngles world;
            if (gimbal.latestImu(imu) && worldAngles(imu, world)) {
                const auto ref = gimbal.state();
                const double yaw_base =
                    candidates[best].use_world ? candidates[best].yaw_sign * world.az_x
                                              : candidates[best].yaw_sign * ref.yaw;
                const double pitch_base =
                    candidates[best].use_world ? candidates[best].pitch_sign * world.el_x
                                               : candidates[best].pitch_sign * ref.pitch;
                const auto step_phase =
                    runPhase(gimbal, true, yaw_base + step, pitch_base, hold);
                const double dyaw = changeAt(step_phase, false, 0.0);
                const double dpitch = changeAt(step_phase, true, 0.0);
                std::cout << "==> 赢家约定下命令 yaw+" << step_deg << "° → 电机 yaw 动 " << dyaw
                          << "°，pitch 动 " << dpitch << "°" << std::endl;
                const auto step2 = runPhase(gimbal, true, yaw_base, pitch_base + step, hold);
                std::cout << "==> 赢家约定下命令 pitch+" << step_deg << "° → 电机 yaw 动 "
                          << changeAt(step2, false, 0.0) << "°，pitch 动 "
                          << changeAt(step2, true, 0.0) << "°" << std::endl;
            }
        }
        std::cout << "把上面整段贴给我：我会把下发改成这个约定（"
                     "大概率是\"命令 = 用四元数算出的世界系角\"），"
                     "并用 --q 的数据把偏置一起标出来。" << std::endl;
        return 0;
    }
}

int main(int argc, char** argv)
{
    std::string config_dir = "configs";
    // 默认 10°：pitch 轴有摩擦/回差，3° 的命令可能根本推不动（实测只动 0.3°）。
    double step_deg = 10.0;
    double hold = 1.5;
    std::string axis = "both";
    bool quiet = false;
    bool q_mode = false;
    double q_seconds = 12.0;
    bool world_mode = false;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--config-dir" && i + 1 < argc) config_dir = argv[++i];
        else if (arg == "--step" && i + 1 < argc) step_deg = std::atof(argv[++i]);
        else if (arg == "--hold" && i + 1 < argc) hold = std::atof(argv[++i]);
        else if (arg == "--axis" && i + 1 < argc) axis = argv[++i];
        else if (arg == "--pitch") axis = "pitch";
        else if (arg == "--q") {
            q_mode = true;
            if (i + 1 < argc && argv[i + 1][0] != '-') q_seconds = std::atof(argv[++i]);
        }
        else if (arg == "--world") world_mode = true;
        else if (arg == "--quiet") quiet = true;
        else if (arg == "--help" || arg == "-h") {
            std::cout << "用法: gimbal_sign_probe [--config-dir configs] [--step 度] "
                         "[--hold 秒] [--axis yaw|pitch|both] [--quiet] [--q 秒] [--world]\n"
                         "  A 静置不发 / B 发当前回传值 / C 当前值+step / D 当前值-step\n"
                         "  每段打印：电机角、IMU 角、mode；B 段直接判定命令在哪个坐标系\n"
                         "  --q [秒]：只读不发，打印四元数 + 各基轴的世界方位（手动转云台）\n";
            std::cout << "  --world：自动试\"命令发世界系还是电机系\"的几种约定，"
                         "并给出方向台阶结果（会动云台，先清空范围）\n";
            return 0;
        }
    }

    const YAML::Node serial_file = YAML::LoadFile(config_dir + "/serial.yaml");
    const io::SerialConfig serial_cfg = auto_aim::loadSerialConfig(serial_file);
    const YAML::Node serial = serial_file["serial"] ? serial_file["serial"] : serial_file;
    const double report_sign = serial["yaw_sign"].as<double>(1.0) < 0.0 ? -1.0 : 1.0;

    io::Gimbal gimbal(serial_cfg);
    if (!gimbal.connected()) {
        std::cerr << "打不开下位机串口（device=" << serial_cfg.device << "）" << std::endl;
        return 1;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(300));

    if (q_mode) {
        return runQuaternionDump(gimbal, q_seconds);
    }
    if (world_mode) {
        return runWorldFrameProbe(gimbal, step_deg, hold);
    }

    const auto state0 = gimbal.state();
    std::cout << std::fixed << std::setprecision(3)
              << "[probe] 起始：回传原始 yaw = " << state0.yaw << " rad，pitch = "
              << state0.pitch << " rad，mode = " << io::gimbalModeName(gimbal.mode())
              << "（视觉端 configs/serial.yaml: yaw_sign/回传 = " << report_sign
              << "，command_frame/下发 = "
              << (serial_cfg.command_frame == io::SerialConfig::CommandFrame::World ? "world"
                                                                                    : "motor")
              << "）"
              << std::endl;
    std::cout << "[probe] 探针**按原始坐标系原样下发**（不做任何符号转换），"
                 "step = " << step_deg << "°，每段 " << hold << " s，100 Hz 下发/20 ms 采样。"
              << std::endl;
    std::cout << "[probe] 注意：实测「命令 3° → 实际动 41°」，所以下发期间云台可能大幅"
                 "转动，请清空云台活动范围、手离开。" << std::endl;

    const double step = step_deg * M_PI / 180.0;
    bool on_pitch = axis == "pitch";
    const int trials = (axis == "both") ? 2 : 1;
    for (int trial = 0; trial < trials; ++trial) {
        if (axis == "both") on_pitch = (trial == 1);
        const char* name = on_pitch ? "pitch" : "yaw";
        const auto ref = gimbal.state();
        const double base_yaw = ref.yaw;
        const double base_pitch = ref.pitch;
        // C 用"当前（B 之后）真实回传值 + step"当命令基准：这样测的就是
        // "命令加了 step，电机角跟着加了多少"，不会被 B 的零点差混进来。
        const auto ref_c = gimbal.state();
        const double cmd_up_yaw = on_pitch ? ref_c.yaw : ref_c.yaw + step;
        const double cmd_up_pitch = on_pitch ? ref_c.pitch + step : ref_c.pitch;

        std::cout << "\n===== " << name << " =====" << std::endl;

        // A 静置不发：下位机自己在动吗
        const auto quiet_phase = runPhase(gimbal, false, 0.0, 0.0, 0.8);
        if (!quiet) printProfile("[A 静置不发]", quiet_phase, on_pitch);
        std::cout << "  A 静置 0.8s 内自身变化 " << changeAt(quiet_phase, on_pitch, 0.0)
                  << "°（不是 0 说明 C 板自己在动）" << std::endl;

        // B 发"当前值"
        const auto hold_phase = runPhase(gimbal, true, base_yaw, base_pitch, hold);
        if (!quiet) printProfile("[B 发当前值]", hold_phase, on_pitch);
        const double hold_drift = changeAt(hold_phase, on_pitch, 0.3);
        std::cout << "  B 发当前值 → 最大偏移 " << hold_drift
                  << "°（≈0 才说明命令有权威且零点一致）" << std::endl;

        // B 的判据：命令值 = 当前回传值。收敛后**哪个量等于命令值**，命令就在哪个系里。
        const double cmd_b = on_pitch ? base_pitch : base_yaw;
        const double motor_after_b = finalValue(hold_phase, on_pitch, false);
        const double imu_after_b = finalValue(hold_phase, on_pitch, true);
        const double cmd_b_deg = cmd_b * 180.0 / M_PI;
        const double err_motor = std::abs(motor_after_b - cmd_b_deg);
        const double err_imu = std::abs(imu_after_b - cmd_b_deg);
        std::cout << "  B 判据：命令 " << cmd_b * 180.0 / M_PI << "°；收敛后 电机角 "
                  << motor_after_b << "° / IMU 角 " << imu_after_b << "°；"
                  << "末帧 mode=" << (hold_phase.empty() ? -1 : hold_phase.back().mode)
                  << std::endl;
        std::cout << "     → 命令跟哪个量对得上：电机角差 " << err_motor << "°，IMU 角差 "
                  << err_imu << "° → "
                  << (err_motor < 3.0 && err_motor <= err_imu
                          ? "命令在**电机系**（与回传同系）"
                          : (err_imu < 3.0 ? "命令在**IMU/世界系**（不是回传的那个量！）"
                                           : "两个都对不上 → 命令系是第三种（或含尺度）"))
                  << std::endl;

        // C 发"当前值 + step"（绝对值，原始系原样发）
        const auto step_phase = runPhase(gimbal, true, cmd_up_yaw, cmd_up_pitch, hold);
        if (!quiet) printProfile("[C 当前值+step]", step_phase, on_pitch);
        const double early = changeAt(step_phase, on_pitch, 0.0);
        const double steady = changeAt(step_phase, on_pitch, hold * 0.6);
        std::cout << "  C 命令 +" << step_deg << "° → 前段(峰值) " << early << "°，后段稳态 "
                  << steady << "°；电机角到 " << finalValue(step_phase, on_pitch, false)
                  << "° / IMU 角到 " << finalValue(step_phase, on_pitch, true)
                  << "°；末帧 mode=" << (step_phase.empty() ? -1 : step_phase.back().mode)
                  << std::endl;

        // D 反方向台阶（从当前真实位置再发一次，看是不是两个方向都能动）
        const auto ref2 = gimbal.state();
        const double neg_cmd_yaw = on_pitch ? ref2.yaw : ref2.yaw - step;
        const double neg_cmd_pitch = on_pitch ? ref2.pitch - step : ref2.pitch;
        const auto neg_phase = runPhase(gimbal, true, neg_cmd_yaw, neg_cmd_pitch, hold);
        if (!quiet) printProfile("[D 当前值-step]", neg_phase, on_pitch);
        std::cout << "  D 命令 -" << step_deg << "° → 变化 " << changeAt(neg_phase, on_pitch, 0.0)
                  << "°；末帧 mode=" << (neg_phase.empty() ? -1 : neg_phase.back().mode)
                  << std::endl;
        if (!neg_phase.empty() && neg_phase.back().mode != 1) {
            std::cout << "     ⚠️ 末帧 mode≠1（AUTO_AIM）：下位机已经不受我们控制了 —— "
                         "命令再发也不动（本次 C/D 全 0 就是这个原因）。" << std::endl;
        }

        // 回到起始值，别把云台留在偏的位置
        runPhase(gimbal, true, base_yaw, base_pitch, 1.0);
        (void)hold_drift;
    }

    // 结束：强制不控制，云台交回下位机
    for (int i = 0; i < 30; ++i) {
        gimbal.send(false, false, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    std::cout << "\n[probe] 已发 mode=0（不控制）。"
                 "把上面 A~D 的四行结果发给我，就能定下命令的语义（绝对/增量 + 符号 + 零点）。"
              << std::endl;
    return 0;
}
