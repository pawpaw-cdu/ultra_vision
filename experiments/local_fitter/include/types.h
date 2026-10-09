#pragma once
// 坐标系：相机系 C：Z 前、X 右、Y 下；车辆系 V：Z 上、X 前、Y 右。
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cmath>
#include <vector>

namespace rm
{
    using Vec2 = Eigen::Matrix<double, 2, 1>;
    using Vec3 = Eigen::Matrix<double, 3, 1>;
    using Mat3 = Eigen::Matrix<double, 3, 3>;

    struct CameraIntrinsics
    {
        double fx = 800, fy = 800, cx = 432, cy = 960, k1 = 0, k2 = 0, p1 = 0, p2 = 0, k3 = 0;
    };

    struct LightBarObservation
    {
        double timestamp = 0.0;
        Vec2 top_px{0, 0}, bottom_px{0, 0};
        double confidence = 1.0;
        int armor_id = 0;
        // 灯条在板面切向的偏移(m)：一块装甲板有两根灯条（左 a=+W/2、右 a=-W/2）。
        // 单灯条模型（提示词原始设定）取 0。
        double bar_offset = 0.0;
    };

    struct VehicleState
    {
        double timestamp = 0.0;
        Vec3 center_cam{0, 0, 0};
        double radius = 0.2;
        double yaw = 0.0;
        double pitch_install = 15.0 * M_PI / 180.0;
        double angular_velocity = 0.0;
        Vec3 drift_velocity{0, 0, 0};
        bool valid = false;
        double quality = 0.0;
    };

    // V→C 的基旋转（云台俯仰为 0 时）。
    inline Mat3 baseRotation()
    {
        Mat3 m;
        m << 0.0, 1.0, 0.0,
             0.0, 0.0, -1.0,
             1.0, 0.0, 0.0;
        return m;
    }

    inline Mat3 rotationVehicleToCamera(double gimbal_pitch)
    {
        return Eigen::AngleAxisd(gimbal_pitch, Vec3::UnitX()).toRotationMatrix() * baseRotation();
    }

    inline Vec2 project(const Vec3& p_c, const CameraIntrinsics& K)
    {
        return { K.fx * p_c.x() / p_c.z() + K.cx, K.fy * p_c.y() / p_c.z() + K.cy };
    }

    // 车辆系灯条两端点 → 相机系。
    inline void barEndpointsCamera(const Vec3& C, double h, double theta, double alpha, double R,
                                   double L, double gimbal_pitch, Vec3& top, Vec3& bottom)
    {
        const double r_top = R - 0.5 * L * std::sin(alpha);
        const double r_bot = R + 0.5 * L * std::sin(alpha);
        const double h_top = h + 0.5 * L * std::cos(alpha);
        const double h_bot = h - 0.5 * L * std::cos(alpha);
        const Vec3 pv_top(r_top * std::cos(theta), r_top * std::sin(theta), h_top);
        const Vec3 pv_bot(r_bot * std::cos(theta), r_bot * std::sin(theta), h_bot);
        const Mat3 rcv = rotationVehicleToCamera(gimbal_pitch);
        top = rcv * pv_top + C;
        bottom = rcv * pv_bot + C;
    }

    // 通用灯条端点：灯条在板面切向偏移 a（米）处，绕竖直轴 yaw=theta，
    // 安装倾角 alpha。a=0 时退化为 barEndpointsCamera（保留原单灯条模型）。
    //
    // 板面局部：n=(cosθ,sinθ,0) 径向、t=(-sinθ,cosθ,0) 切向、ẑ 竖直。
    // 角点(a,b) = ρ·n + a·t + z·ẑ，其中 ρ = R - b·sinα，z = h + b·cosα。
    // b=+L/2 为顶端（半径更小、更高），b=-L/2 为底端——与提示词 4.2 一致。
    inline void barEndpointsCameraOffset(const Vec3& C, double h, double theta, double alpha,
                                         double R, double L, double a, double gimbal_pitch,
                                         Vec3& top, Vec3& bottom)
    {
        auto corner = [&](double b) {
            const double rho = R - b * std::sin(alpha);
            const double z = h + b * std::cos(alpha);
            return Vec3(rho * std::cos(theta) - a * std::sin(theta),
                        rho * std::sin(theta) + a * std::cos(theta), z);
        };
        const Vec3 pv_top = corner(+0.5 * L);
        const Vec3 pv_bot = corner(-0.5 * L);
        const Mat3 rcv = rotationVehicleToCamera(gimbal_pitch);
        top = rcv * pv_top + C;
        bottom = rcv * pv_bot + C;
    }

    // 一块装甲板的 4 个角点（相机系）：[LT, LB, RT, RB]
    // 左灯条 a=+W/2，右灯条 a=-W/2；L 为灯条长度（板高），W 为板宽。
    inline void plateCornersCamera(const Vec3& C, double h, double theta, double alpha,
                                   double R, double L, double W, double gimbal_pitch,
                                   Vec3 corners[4])
    {
        barEndpointsCameraOffset(C, h, theta, alpha, R, L, +0.5 * W, gimbal_pitch,
                                 corners[0], corners[1]);
        barEndpointsCameraOffset(C, h, theta, alpha, R, L, -0.5 * W, gimbal_pitch,
                                 corners[2], corners[3]);
    }
}  // namespace rm
