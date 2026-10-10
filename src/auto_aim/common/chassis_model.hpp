#ifndef ULTRA_VISION_AUTO_AIM_COMMON_CHASSIS_MODEL_HPP
#define ULTRA_VISION_AUTO_AIM_COMMON_CHASSIS_MODEL_HPP

// ============================ 底盘模型（重做版） ============================
//
// 这一层原来是"东一块西一块"拼出来的（overlay 一套、real_runner 一套、replay 一套），
// 而且几处把**世界系**的向量和**相机系**的向量混在一起用，真机上就表现成
// "四块板高度不统一、底盘完全不对"（仿真里相机姿态≈0，混坐标系看不出来，所以
// "仿真没问题、实车有问题"）。现在收敛成一个模块、一把尺子：
//
//   底盘慢层几何 ChassisGeometry
//        · 四块板到**车心**的水平距离 plate_distance[4]（真车是矩形底盘：
//          前后一个值、左右另一个值；默认全相等=圆，仿真靶就是圆）
//        · 四块板板心**相对车心**的高度 plate_height[4]（真机四块板同一高度；
//          默认全 0，可由观测/拟合器的慢层收敛）
//        · 装甲板安装倾角 armor_pitch（真机 15° 支架；仿真 0）
//        · 每块板的板面尺寸（大/小板）
//   快层状态（EKF）：车心 (x,y,z)、底速、底盘 yaw、ω
//
//   唯一构造入口 chassisArmorPose(center, yaw, geometry, plate)
//        → 板心（世界系）+ 刚体三轴（法向/切向/灯条方向，世界系）
//   所有消费者（重投影画框、重投影残差、ROI、选板、瞄点）都走它，不许再自己拼。
//
// **坐标系铁律**：本模块只在**世界系（home/EKF 系）**里算模型；
// 要画到图上时，由调用方用**被显示那一帧的姿态**做一次统一变换
// （Tracker::modelAt(dt, render_pose)），不允许任何函数混用两系向量。
//
// 世界系约定（与 ArmorEKF 一致）：x-z 为水平面，**y 向下**；"向上" = (0,-1,0)。
// 板的相位：φ_k = yaw + armorPhaseOffset(k)，armorPhaseOffset(k) = (1-k)·90°，
// 即 k=0 在 +90°（车左侧）、k=1 在 0°（径向 +x，正面）、k=2 在 −90°、k=3 在 180°。

#include <array>
#include <cmath>

#include <opencv2/core.hpp>

#include "common/armor_geometry.hpp"

namespace auto_aim
{
    /// @brief 装甲板相位偏移（rad）。与 ArmorEKF::angleForPlate 同源，
    ///        放在这里是为了让"模型构造"不依赖滤波器实现。
    inline double armorPhaseOffset(int plate)
    {
        return (1 - plate) * 3.14159265358979323846 / 2.0;
    }

    /// @brief 底盘慢层几何（一次比赛/一场测试里基本不变，允许慢收敛）。
    struct ChassisGeometry
    {
        /// 车心 → 板心的水平距离（米），按板号。真车矩形底盘：前后 ≠ 左右。
        std::array<double, 4> plate_distance{{0.25, 0.25, 0.25, 0.25}};
        /// 板心相对车心的高度（米，**向上为正**）。默认四块同高 = 0。
        std::array<double, 4> plate_height{{0.0, 0.0, 0.0, 0.0}};
        /// 装甲板安装倾角 α（rad）。真机 15° 支架；仿真靶 0。
        double armor_pitch = 15.0 * 3.14159265358979323846 / 180.0;
        /// 哪几块是**大**装甲板（真机通常前后大、左右小；仿真靶四块一样）。
        std::array<bool, 4> plate_large{{false, false, false, false}};
        /// 板面等效半宽/半高（米）。**必须与 PnP 的点语义一致**（configs 的 pnp 段）。
        double small_half_width = 0.06575;
        double large_half_width = 0.1105;
        double half_height = 0.030;

        double distance(int plate) const
        {
            return plate >= 0 && plate < 4 ? plate_distance[static_cast<std::size_t>(plate)]
                                           : plate_distance[1];
        }
        double height(int plate) const
        {
            return plate >= 0 && plate < 4 ? plate_height[static_cast<std::size_t>(plate)]
                                           : plate_height[1];
        }
        double halfWidth(int plate) const
        {
            const bool large = plate >= 0 && plate < 4 &&
                plate_large[static_cast<std::size_t>(plate)];
            return large ? large_half_width : small_half_width;
        }
        bool isLarge(int plate) const
        {
            return plate >= 0 && plate < 4 && plate_large[static_cast<std::size_t>(plate)];
        }
        /// 有效半径（诊断/兼容用）：四块板距离的均值。
        double meanDistance() const
        {
            double sum = 0.0;
            for (double d : plate_distance) sum += d;
            return sum / 4.0;
        }
    };

    /// @brief 一块装甲板的刚体姿态（板心 + 三轴，**同一坐标系**）。
    struct ArmorPose
    {
        cv::Vec3d center{0.0, 0.0, 0.0};
        ArmorAxes axes;
    };

    /// @brief **唯一**的底盘模型构造入口（世界系）。
    /// @param chassis_center 车心（世界系，y 向下）
    /// @param chassis_yaw    底盘 yaw（世界系）
    /// @param geometry       慢层几何
    /// @param plate          板号 0..3
    inline ArmorPose chassisArmorPose(const cv::Vec3d& chassis_center, double chassis_yaw,
                                      const ChassisGeometry& geometry, int plate)
    {
        const double phi = chassis_yaw + armorPhaseOffset(plate);
        const double d = geometry.distance(plate);
        const double h = geometry.height(plate);   // 向上为正

        ArmorPose pose;
        // 车心 + 水平径向 d + 竖直抬升 h（世界系 y 向下 → 抬升是 −y）。
        pose.center = cv::Vec3d(chassis_center[0] + d * std::cos(phi),
                                chassis_center[1] - h,
                                chassis_center[2] + d * std::sin(phi));
        // 径向（水平）与车体竖直（世界向上）→ 三轴（含 15° 安装倾角）。
        const cv::Vec3d radial(std::cos(phi), 0.0, std::sin(phi));
        const cv::Vec3d up(0.0, -1.0, 0.0);
        pose.axes = armorAxesFromRadial(radial, up, geometry.armor_pitch);
        return pose;
    }

    /// @brief 四块板的模型（世界系），一次算完。
    inline std::array<ArmorPose, 4> chassisArmorPoses(const cv::Vec3d& chassis_center,
                                                      double chassis_yaw,
                                                      const ChassisGeometry& geometry)
    {
        std::array<ArmorPose, 4> poses;
        for (int plate = 0; plate < 4; ++plate) {
            poses[static_cast<std::size_t>(plate)] =
                chassisArmorPose(chassis_center, chassis_yaw, geometry, plate);
        }
        return poses;
    }

    /// @brief 底盘的运动状态（由局部拟合器一次弧段拟合出来的 ω 与旋转中心平移速度判定）。
    ///        现场思路：**纯旋转时旋转中心不该动** —— 这给"旋转中心/模型构造"一个先验，
    ///        避免车心被单块检测板的噪声拖着跑（"旋转中心随检测板跑飞"）。
    enum class ChassisMotion
    {
        Still,                    // 既不怎么转也不怎么平移
        Rotating,                 // 纯旋转（小陀螺）→ 中心可用强先验
        Translating,              // 纯平移
        RotatingAndTranslating,   // 陀螺 + 跑动
    };

    /// @param omega     拟合角速度（rad/s）
    /// @param speed     旋转中心平移速度（m/s，|drift|）
    /// @param spin_min  判定"在转"的最小 |ω|（rad/s）
    /// @param move_max  判定"没怎么平移"的最大速度（m/s）
    inline ChassisMotion classifyChassisMotion(double omega, double speed, double spin_min,
                                               double move_max)
    {
        const bool spinning = std::abs(omega) >= spin_min;
        const bool moving = speed >= move_max;
        if (spinning && moving) return ChassisMotion::RotatingAndTranslating;
        if (spinning) return ChassisMotion::Rotating;
        if (moving) return ChassisMotion::Translating;
        return ChassisMotion::Still;
    }

    /// @brief 由运动状态给出"把滤波器车心拉向拟合中心"的系数（0=不拉，1=完全替换）。
    ///        纯旋转：中心物理上不变 → 强拉（但不硬替，留一点柔性给慢漂移）；
    ///        有平移：中心本来就在动 → 只轻拉，主要靠滤波器自己跟。
    inline double chassisCenterAnchorGain(ChassisMotion motion, double gain_rotating,
                                          double gain_other)
    {
        return motion == ChassisMotion::Rotating ? gain_rotating : gain_other;
    }

    /// @brief 把一个世界系姿态转到相机系（板心用点变换，三轴用方向变换）。
    ///        调用方必须传**同一时刻**的相机姿态 —— 这就是"模型重投影到哪一帧"
    ///        的唯一开关，禁止再用 tracker 内部存的那个（观测时刻的）姿态去画。
    inline ArmorPose armorPoseToCamera(const ArmorPose& world_pose,
                                       const cv::Vec3d& camera_pos,
                                       const cv::Matx33d& world_to_camera)
    {
        ArmorPose out;
        out.center = world_to_camera * (world_pose.center - camera_pos);
        out.axes.normal = world_to_camera * world_pose.axes.normal;
        out.axes.tangent = world_to_camera * world_pose.axes.tangent;
        out.axes.bar = world_to_camera * world_pose.axes.bar;
        return out;
    }
}  // namespace auto_aim

#endif  // ULTRA_VISION_AUTO_AIM_COMMON_CHASSIS_MODEL_HPP
