#ifndef AUTO_AIM_GIMBAL_COMMAND_FRAME_HPP
#define AUTO_AIM_GIMBAL_COMMAND_FRAME_HPP

#include <cmath>
#include <array>

#include "io/gimbal/gimbal.hpp"

namespace auto_aim
{
    /// @brief 由云台 IMU 四元数 (w,x,y,z) 算出"世界系云台角"：基轴 x 的方位角 az_x
    ///        与俯仰角 el_x（弧度，z 轴向上为正）。
    ///
    /// 为什么要它：2026-09-29 在实车量出下位机的 yaw/pitch 闭环比较的是**这两个量**
    /// （world 回传实测）：把 `(+az_x, -el_x)` 的当前值发回去，云台
    /// 保持不动（0.2~0.9°）；发回传的电机角当前值，一次跑 35.7°。台阶测试 1:1。
    /// 也就是说"IMU 相对电机系被转了 90° 且 yaw 反向"，命令必须发世界系角 —— 这正是
    /// sp_vision 的做法（它下发的就是世界系绝对角，因此不需要任何符号/零点处理）。
    ///
    /// 返回 false 表示四元数不可用（模长离 1 太远），调用方应放弃这一拍而不是退化发别的口径。
    inline bool imuWorldAngles(const io::ImuSample& imu, double& az_x, double& el_x)
    {
        const double w = imu.w, x = imu.x, y = imu.y, z = imu.z;
        const double norm = std::sqrt(w * w + x * x + y * y + z * z);
        if (!std::isfinite(norm) || norm < 0.5 || norm > 1.5) return false;
        // world <- gimbal 旋转矩阵（与 node.cpp 里填 base_to_world 的公式一致）
        const double r00 = 1.0 - 2.0 * (y * y + z * z);
        const double r10 = 2.0 * (x * y + z * w);
        const double r20 = 2.0 * (x * z - y * w);
        az_x = std::atan2(r10, r00);
        el_x = std::atan2(r20, std::hypot(r00, r10));
        return true;
    }

    // ---- 期望姿态 → C 板命令通道（2026-09-29）---------------------------------
    // 为什么不能"世界项 + 对角增量"：台阶测试实测两个通道在云台轴系里是**耦合**的——
    //   cmd_yaw +10°  → 电机 yaw +9.18°、pitch +0.75°
    //   cmd_pitch +10°→ 电机 yaw −9.01°、pitch −9.45°
    // 也就是 pitch 通道会把 yaw 带走同样多。用对角系数发增量必然在某些方向上推错，
    // 闭环里还会形成"pitch 推 → yaw 转 → 瞄点变"的耦合正反馈（实车现象：pitch 一路
    // 爬到限位、yaw 同时来回摆）。
    // 正确做法是**用几何算**：每拍都能从四元数解出"IMU 相对云台零位的安装关系"，
    // 于是把期望姿态直接换算成 C 板比较的那两个量，耦合自动处理，也不需要标系数。
    using Mat3 = std::array<double, 9>;   // 行主序

    inline Mat3 mul(const Mat3& a, const Mat3& b)
    {
        Mat3 out{};
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                double sum = 0.0;
                for (int k = 0; k < 3; ++k) sum += a[3 * i + k] * b[3 * k + j];
                out[3 * i + j] = sum;
            }
        }
        return out;
    }

    inline Mat3 transpose(const Mat3& a)
    {
        Mat3 out{};
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) out[3 * i + j] = a[3 * j + i];
        }
        return out;
    }

    /// @brief 我们的姿态约定：世界(底盘/零位系) ← 云台零位系，与手眼标定工具的
    ///        `gimbalToBase(yaw, pitch) = Ry(yaw)·Rx(pitch)` 完全一致。
    inline Mat3 gimbalToBase(double yaw, double pitch)
    {
        const double cy = std::cos(yaw), sy = std::sin(yaw);
        const double cp = std::cos(pitch), sp = std::sin(pitch);
        return Mat3{
            cy, sy * sp, sy * cp,
            0.0, cp, -sp,
            -sy, cy * sp, cy * cp,
        };
    }

    inline Mat3 imuToWorld(const io::ImuSample& imu)
    {
        const double w = imu.w, x = imu.x, y = imu.y, z = imu.z;
        return Mat3{
            1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
            2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
            2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
        };
    }

    /// @brief 从旋转矩阵取"基轴 x 的方位角/俯仰角"（与 imuWorldAngles 同口径）。
    inline void azElOfRotation(const Mat3& r, double& az_x, double& el_x)
    {
        az_x = std::atan2(r[3], r[0]);                                   // (1,0),(0,0)
        el_x = std::atan2(r[6], std::hypot(r[0], r[3]));                  // (2,0)
    }

    /// @brief 期望云台姿态 → C 板命令通道值。
    /// @param yaw_now/pitch_now 当前姿态（我们内部约定，rad；pitch 就是回传 pitch）
    /// @param yaw_des/pitch_des 期望姿态（同约定，rad）
    /// @param imu_now           当前四元数（用于解出 IMU↔云台零位的安装关系）
    /// @return az_x/el_x：与 C 板闭环反馈同口径的两个量（rad）
    inline bool desiredWorldAngles(double yaw_now, double pitch_now, double yaw_des,
                                   double pitch_des, const io::ImuSample& imu_now,
                                   double& az_x, double& el_x)
    {
        const double norm = std::sqrt(imu_now.w * imu_now.w + imu_now.x * imu_now.x +
                                      imu_now.y * imu_now.y + imu_now.z * imu_now.z);
        if (!std::isfinite(norm) || norm < 0.5 || norm > 1.5) return false;
        az_x = 0.0;
        el_x = 0.0;
        // ---- 姿态约定：tracker 与下发**差一个两轴镜像**（2026-09-29 量定）----
        // tracker/外参那侧用 (+yaw_sign·yaw, +pitch)：手眼样本验过（pitch 标度 +1 →
        //   残差 7.2 mm，−1 → 302 mm），所以那边不能动。
        // 但 IMU 相对相机的安装是**两轴都镜像**的：用 --q 的真实数据（姿态+四元数）
        //   把"当前姿态反解出安装关系、再去预测其它姿态"做跨姿态自洽检验：
        //     (yaw+1,pitch+1) → az 最大偏差 21°、el 89°   ✗
        //     (yaw+1,pitch−1) → az 21°、el 1.4°            ✗（yaw 反）
        //     (yaw−1,pitch+1) → az 19°、el 89°             ✗
        //     (yaw−1,pitch−1) → az 5.1°、el 1.4°           ✓
        // 再用同一帧做方向检验（期望 Y+5° 应让命令 yaw 通道减小；期望 P+5° 应让命令
        // pitch 通道减小，两者都是由实测台阶关系定的）：
        //     (yaw−1,pitch−1) → Δcmd_yaw −4.22 ✓、Δcmd_pitch −4.99 ✓（1:1，交叉 ≤0.35°）
        //     其余三种组合总有一个轴反（这正是实车"改了 pitch 之后 yaw 又反了"的原因）。
        // Δ=0 时两式相消、命令精确等于实测反馈 —— 所以 hold 测试区分不出这些约定，
        // 必须靠上面这两条跨姿态/方向的检验。
        constexpr double kPoseYawSign = -1.0;
        constexpr double kPosePitchSign = -1.0;
        const Mat3 pose_now =
            gimbalToBase(kPoseYawSign * yaw_now, kPosePitchSign * pitch_now);
        const Mat3 pose_des =
            gimbalToBase(kPoseYawSign * yaw_des, kPosePitchSign * pitch_des);
        // IMU 相对"云台零位系"的安装关系：由当前姿态反解出来，不用标定。
        const Mat3 mounting = mul(transpose(pose_now), imuToWorld(imu_now));
        const Mat3 desired = mul(pose_des, mounting);
        azElOfRotation(desired, az_x, el_x);
        return true;
    }

    /// @brief 期望姿态相对当前姿态的"命令通道增量"，外加**运行时自检的符号**。
    ///        为什么要自检：这台车的通道约定（坐标系/镜像/耦合）已经被证明不能靠
    ///        推理定死（hold 测试对约定不敏感、台阶测试只覆盖单通道）。所以下发时
    ///        分成两部分：
    ///           cmd = 实测基准（当前 az/el，保证接管瞬间不跳）
    ///                 + sign ⊙（期望姿态对应的 az/el − 基准）        ← sign 由图像自检
    ///        node 里用"推动命令时靶板在画面里的误差是变小还是变大"来定 sign，
    ///        推反了就地翻转（见 node.cpp 的 auto-sign 一段）。
    inline bool desiredWorldAnglesDelta(double yaw_now, double pitch_now, double yaw_des,
                                        double pitch_des, const io::ImuSample& imu_now,
                                        double& d_az, double& d_el)
    {
        double az_now = 0.0;
        double el_now = 0.0;
        double az_des = 0.0;
        double el_des = 0.0;
        if (!desiredWorldAngles(yaw_now, pitch_now, yaw_now, pitch_now, imu_now, az_now, el_now)) {
            return false;
        }
        if (!desiredWorldAngles(yaw_now, pitch_now, yaw_des, pitch_des, imu_now, az_des, el_des)) {
            return false;
        }
        d_az = az_des - az_now;
        d_el = el_des - el_now;
        return true;
    }
}

#endif   // AUTO_AIM_GIMBAL_COMMAND_FRAME_HPP
