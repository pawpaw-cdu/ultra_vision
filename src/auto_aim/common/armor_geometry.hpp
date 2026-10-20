#ifndef ULTRA_VISION_AUTO_AIM_COMMON_ARMOR_GEOMETRY_HPP
#define ULTRA_VISION_AUTO_AIM_COMMON_ARMOR_GEOMETRY_HPP

// 装甲板的**刚体姿态**（一个板 = 三个正交轴 + 一个板心），全项目**唯一**的模型构造入口。
//
// 为什么单独立一个头文件：历史上"整车轮廓"在两个地方各写了一套，而且各错一点 ——
//   · app/real_runner.cpp 用相机系切向 + 世界竖直（对，但漏了板自身的安装倾角）
//   · app/overlay.hpp     用世界系 yaw 造切向（相机一有 pitch 就把板画歪）+
//                         拿**观测板 PnP 的 rvec** 当所有板的竖直（把观测板的 roll 传染给整车）
// 结果同一个模型在两个窗口里长得不一样，也没有任何一个能对上真机 15° 支架。
//
// 现场事实（用户确认）：真机装甲板装在 15° 支架上 —— 灯条**不是**世界竖直，
// 而是"板顶向车内倾 α"。局部拟合器的车辆系模型正是这么建的（见
// estimation/local_trajectory_fitter.cpp 的 barEndpointV）：
//     p_v = (R − b·sinα)·n̂ + a·t̂ + (h + b·cosα)·ẑ,   b ∈ [−L/2, +L/2], a ∈ ±W/2
// 即：板心 (R·n̂ + h·ẑ)，切向 t̂，**灯条方向 = cosα·ẑ − sinα·n̂**（顶部向车内倾）。
// 于是"灯条倾角 α"从描述性参数变成模型构造的**约束条件**：
// 只要 α 标定对，板心+切向+灯条方向就能把四个角点定死，不用再借任何观测板的东西。

#include <cmath>

#include <opencv2/core.hpp>

namespace auto_aim
{
    /// @brief 一块装甲板的刚体姿态（三个轴必须与板心同坐标系：相机系或 home 系）。
    struct ArmorAxes
    {
        cv::Vec3d normal;    // 板面法向（指向车外）
        cv::Vec3d tangent;   // 板面切向 = 宽度方向（水平，垂直于法向）
        cv::Vec3d bar;       // 灯条方向（板面内的"竖直"，带安装倾角 α）
    };

    /// @brief 由"径向 + 车体竖直 + 安装倾角"构造刚体姿态（与局部拟合器同构）。
    /// @param radial     车心→板心方向（不必归一化；就是那根半径）
    /// @param chassis_up 车体竖直（不必与 radial 正交，会被正交化）
    /// @param tilt_rad   装甲板安装倾角 α（真机 ≈15°，仿真 0°）
    ///
    /// 约定（与 barEndpointV 逐位一致）：灯条顶端向**车内**倾 α，
    /// 所以 bar = cosα·up − sinα·normal、normal' = cosα·normal + sinα·up。
    inline ArmorAxes armorAxesFromRadial(const cv::Vec3d& radial, const cv::Vec3d& chassis_up,
                                         double tilt_rad)
    {
        ArmorAxes axes;
        const double rn = cv::norm(radial);
        if (!(rn > 1e-9)) return axes;
        const cv::Vec3d n = radial / rn;

        // 车体竖直去掉径向分量后正交化（径向是水平的，正常只需一次）。
        cv::Vec3d up = chassis_up - n * chassis_up.dot(n);
        const double un = cv::norm(up);
        if (!(un > 1e-9)) return axes;
        up /= un;

        const double c = std::cos(tilt_rad);
        const double s = std::sin(tilt_rad);
        axes.tangent = up.cross(n);
        const double tn = cv::norm(axes.tangent);
        if (!(tn > 1e-9)) return axes;
        axes.tangent /= tn;
        axes.bar = c * up - s * n;
        axes.normal = c * n + s * up;
        return axes;
    }

    /// @brief **PnP 关键点**用的基向量：由"板法向方位角 ψ + 机械俯仰"给出
    ///        (normal, tangent, vertical) 三轴（相机系）。
    ///
    /// 与上面 `armorAxesFromRadial` 的关系（**别混用，这是两个用途**）：
    ///   · `armorAxesFromRadial` 用"径向 + 竖直"描述**实物板**（画框/整车模型用），
    ///     角点顺序 lb/lt/rt/rb 与图像上下**同向**；
    ///   · 本函数描述的是**神经网络四个关键点**那块等效矩形（`getArmor3DPoints`），
    ///     它的 y/z 标注是 180° 旋转过的（lb 落在图像上沿）。两者给出的角点**集合**
    ///     相同（矩形对 180° 旋转对称），所以位置与法向完全一致，只是标注不同；
    ///     但**同一次计算里必须只用一套**，混用会让带倾角 α 的深度不对称（顶/底
    ///     端点深度不同）被错误配对，产生系统性重投影偏差。
    ///
    /// 数学与 `perception/pnp_solver.cpp` 原先的局部 `armorBasis` **逐位一致**，
    /// 提取到这里是为了让刚体几何与 `refineArmorYaw` 共用同一份约定
    /// （ψ 的含义必须完全相同，否则"拟合器给的 yaw"接不进 2D 优化）。
    ///
    /// @param psi   板法向在相机 x-z 平面内的方位角（rad），`atan2(n_z, n_x)`
    /// @param pitch 机械俯仰（RM 规范 ≈15°）
    inline void armorKeypointBasis(double psi, double pitch, cv::Vec3d& normal,
                                   cv::Vec3d& tangent, cv::Vec3d& vertical)
    {
        const cv::Vec3d n0(std::cos(psi), 0.0, std::sin(psi));
        const cv::Vec3d v0(0.0, 1.0, 0.0);   // 相机 y 向下
        const cv::Vec3d t0 = v0.cross(n0);
        const double cp = std::cos(pitch);
        const double sp = std::sin(pitch);
        normal = cp * n0 + sp * v0;
        vertical = -sp * n0 + cp * v0;
        tangent = t0;   // 绕切向轴旋转，切向不变
    }

    /// @brief 四个角点（左下、左上、右上、右下）——纯刚体构造，三轴都按三维向量参与，
    ///        不再有"切向只有 x/z、竖直只补 y"这种只在相机 pitch≈0 时才成立的写法。
    inline std::vector<cv::Vec3d> armorCorners(const cv::Vec3d& center, const ArmorAxes& axes,
                                               double half_width, double half_height)
    {
        const cv::Vec3d t = axes.tangent * half_width;
        const cv::Vec3d v = axes.bar * half_height;
        return {center - t - v, center - t + v, center + t + v, center + t - v};
    }
}  // namespace auto_aim

#endif  // ULTRA_VISION_AUTO_AIM_COMMON_ARMOR_GEOMETRY_HPP
