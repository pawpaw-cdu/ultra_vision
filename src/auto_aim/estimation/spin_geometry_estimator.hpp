#ifndef AUTO_AIM_ESTIMATION_SPIN_GEOMETRY_ESTIMATOR_HPP
#define AUTO_AIM_ESTIMATION_SPIN_GEOMETRY_ESTIMATOR_HPP

// 小陀螺灯条轨迹几何估计器（离线/低频全局先验，不接管主链路）。
//
// 思路：跟踪一根灯条上下端点在像素系的轨迹（pitch=0 → 矩形；pitch≠0 → 扇带），
// 用非线性最小二乘反解旋转中心 C、回转半径 R、灯条 pitch、旋转轴方向与 yaw(t)，
// 再据刚体几何外推同一辆车的其余三块板。详见 docs/spin_geometry_estimator.md。
//
// 依赖：Eigen3 + Ceres（可选组件，只有找到 Ceres 时才编译；不链接进 auto_aim_sim/real）。
// 关键实测结论（见 docs）：单灯条需要 ~200° 以上转角覆盖；重投影误差不能当有效性判据；
// yaw(t) 需用三次 B 样条并配多起点初值，否则剧烈变速下会落进"近/远"错误分支。

#include <array>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace auto_aim
{
    struct SpinIntrinsics
    {
        double fx = 1447.0, fy = 1447.0, cx = 720.0, cy = 540.0;
    };

    struct SpinBarObservation
    {
        double t = 0.0;               // 时间戳（秒，单调）
        int plate = 0;                // 0..3（同一帧内可给多根：联合约束，强烈建议）
        int side = 1;                 // +1 / -1（同一块板的两根灯条）
        double u_top = 0.0, v_top = 0.0;
        double u_bot = 0.0, v_bot = 0.0;
    };

    struct SpinGeometryConfig
    {
        SpinIntrinsics K;
        double bar_offset_c = 0.060;  // 灯条相对板中心的切向偏移（已知）
        double bar_length_L = 0.055;  // 灯条长度（已知，提供尺度）
        double distance_hint = 5.0;   // 距离先验（仅用于初值）
        bool estimate_axis = true;    // 是否把旋转轴方向（2 DOF）一起估
        int spline_knots = 10;        // yaw(t) 三次 B 样条控制点数
    };

    // 说明（实测）：真实检测噪声下，**只给一根灯条会频繁落进"近/远"错误分支**，
    // yaw 误差可达 50°+；把同一帧的多根灯条（4 板 × 2）一起喂进来，yaw 回到 ~1~2°。
    // 因此在线使用时务必给**多个** SpinBarObservation（同一 t、不同 plate/side）。

    struct SpinGeometryResult
    {
        bool valid = false;
        Eigen::Vector3d center{0.0, 0.0, 0.0};  // 旋转中心 = 整车中心（相机系）
        double radius = 0.0;                    // 回转半径 R
        double pitch = 0.0;                     // 灯条 pitch（rad，径向外倾）
        double axis_tx = 0.0, axis_tz = 0.0;    // 旋转轴 â = normalize((tx,1,tz))
        double t_begin = 0.0, t_end = 0.0;
        std::vector<double> spline;             // yaw(t) 控制点
        double arc_coverage_deg = 0.0;          // 观测覆盖的转角（有效性门控用）
        double reproj_rmse_px = 0.0;            // 仅供诊断，不可当有效性判据
        double message_cost = 0.0;

        // yaw(t)：三次 B 样条求值。
        double yawAt(double t) const;
    };

    class SpinGeometryEstimator
    {
    public:
        explicit SpinGeometryEstimator(SpinGeometryConfig config = {});

        /// @brief 用一窗灯条端点观测拟合全局几何。
        /// @return 是否得到有效解（弧覆盖不足 / 观测太少 / 优化失败 → false）。
        bool fit(const std::vector<SpinBarObservation>& observations);

        const SpinGeometryResult& result() const { return result_; }

        /// @brief 外推 4 块板 × 2 根灯条（相位 theta）的 8 段 3D 端点。
        std::vector<std::array<Eigen::Vector3d, 2>> allBars(double theta) const;

    private:
        SpinGeometryConfig config_;
        SpinGeometryResult result_;
    };
}  // namespace auto_aim

#endif  // AUTO_AIM_ESTIMATION_SPIN_GEOMETRY_ESTIMATOR_HPP
