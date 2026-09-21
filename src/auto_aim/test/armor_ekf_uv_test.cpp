// UV（像素重投影）观测的自检：不依赖仿真器/图像，纯数学。
//
// 为什么要有它：UV 观测的雅可比是**中心差分**（ArmorEKF::uvJacobian），
// 手写的重投影/板面几何一旦符号错（板面切向、竖直方向、sin/cos 表示），
// 现象是"滤波器慢慢被带偏"而不是报错 —— 只有"喂真值 + 看收敛"才抓得住。
// 它同时钉住两件事：
//   * 从状态算出的四个角点与"物点定义 + 针孔投影"必须自洽（残差为 0 时状态不动）；
//   * 有 1 px 角点噪声时，迭代更新必须把角速度/位置/朝向收敛到真值
//     （雅可比符号错时这一条必然失败）。

#include "Kalman/armor_ekf.hpp"

#include <array>
#include <cmath>
#include <cstdio>
#include <random>

namespace
{
    using auto_aim::ArmorEKF;
    using auto_aim::UvCamera;
    using auto_aim::UvObservation;
    using auto_aim::Vec;

    int failures = 0;

    void check(bool condition, const char* what)
    {
        if (!condition) {
            std::printf("FAIL: %s\n", what);
            ++failures;
        }
    }

    UvCamera makeCamera()
    {
        UvCamera camera;
        camera.fx = 1286.0;
        camera.fy = 1288.0;
        camera.cx = 645.0;
        camera.cy = 483.0;
        // 相机在 world 原点、yaw=pitch=0：这个 EKF 的 world 就是"云台归零时的相机系"
        // （tracker.cpp 的 cameraToBaseRotation 在 yaw=pitch=0 时给出单位阵），
        // 所以这里 world→camera = I（y 向下、z 向前）。
        for (int i = 0; i < 3; ++i) camera.world_to_camera(i, i) = 1.0;
        return camera;
    }

    /// 板面法向（= 径向）是否朝向相机：相机在原点 ⇒ 判据 radial · center < 0。
    /// 只有这一类板会被检测器看到（背面板的"左/右"标签会镜像，实际不会出现在观测里）。
    bool plateFacesCamera(int plate_id, double yaw)
    {
        const double phi = yaw + ArmorEKF::angleForPlate(plate_id);
        return std::cos(phi) * 2.5 + std::sin(phi) * 6.0 < 0.0;
    }

    /// 真值状态：底盘中心固定 (2.5, 0, 6.0)，绕竖直轴以 omega 自转（四块板绕圈）。
    Vec<9> truthState(double yaw, double omega)
    {
        Vec<9> state;
        state(0, 0) = 2.5;              // x_c
        state(2, 0) = 0.0;              // y_c
        state(4, 0) = std::sin(yaw);    // sin(theta)
        state(5, 0) = std::cos(yaw);    // cos(theta)
        state(6, 0) = omega;
        state(7, 0) = 6.0;              // z_c
        return state;
    }

    Vec<3> plateCenter(int plate_id)
    {
        const double angle = ArmorEKF::angleForPlate(plate_id);
        Vec<3> center;
        center(0, 0) = 2.5 + 0.21 * std::cos(angle);
        center(1, 0) = 0.0;
        center(2, 0) = 6.0 + 0.21 * std::sin(angle);
        return center;
    }

    void testProjectionIsSelfConsistent()
    {
        const UvCamera camera = makeCamera();
        // init(center, plate_id) 会把朝向设成 0，所以自洽性检查也用 yaw=0 的真值。
        const Vec<9> state = truthState(0.0, 0.0);
        // 用**朝向相机**的那块板（yaw=0 时是 2 号：径向 = -z，指向相机原点）。
        constexpr int plate = 2;
        constexpr double half_width = 0.067;
        constexpr double half_height = 0.0285;

        std::array<Vec<2>, 4> corners{};
        const bool ok = ArmorEKF::predictUvCorners(state, plate, 0.21, half_width, half_height, 0.0,
                                                   camera, corners);
        check(ok, "uv: a plate in front of the camera projects");
        if (!ok) return;

        // 物点顺序 left_bottom, left_top, right_top, right_bottom：
        // 左角的 u 更小、上角的 v 更小（图像 v 向下增长）。
        check(corners[0](0, 0) < corners[2](0, 0),
              "uv: left corners project left of right corners");
        check(corners[0](1, 0) > corners[1](1, 0),
              "uv: bottom corners project below top corners");
        // 板宽 0.134 m、距离 ~6 m、fx≈1286 → 像素宽度 ≈ 1286*0.134/6 ≈ 29 px。
        const double pixel_width = corners[2](0, 0) - corners[1](0, 0);
        check(std::abs(pixel_width - 29.0) < 5.0,
              "uv: projected plate width matches pinhole geometry (~29 px at 6 m)");

        // 把"预测"当观测喂回去：状态不应被推动（自洽性）。
        ArmorEKF::Config config;
        config.R = 0.21;
        ArmorEKF ekf;
        ekf.setConfig(config);
        ekf.setDt(1.0 / 30.0);
        const Vec<3> center_of_plate = plateCenter(plate);
        ekf.init(center_of_plate, plate);

        // 此时滤波器自己算出的角点应当与真值预测的角点一致（同一套几何）。
        std::array<Vec<2>, 4> predicted{};
        check(ArmorEKF::predictUvCorners(ekf.getState(), plate, config.R, half_width,
                                         half_height,
                                         0.0, camera, predicted),
              "uv: the filter's own state projects");
        double self_error = 0.0;
        for (int i = 0; i < 4; ++i) {
            self_error = std::max(self_error, std::abs(predicted[static_cast<std::size_t>(i)](0, 0) -
                                                       corners[static_cast<std::size_t>(i)](0, 0)));
            self_error = std::max(self_error, std::abs(predicted[static_cast<std::size_t>(i)](1, 0) -
                                                       corners[static_cast<std::size_t>(i)](1, 0)));
        }
        check(self_error < 1e-6,
              "uv: init(center, plate_id) lands on the same geometry as the truth state");
    }

    /// 紧凑 UV（角度/中心/长度）也必须能独立把状态收敛到真值。
    /// 这条最容易抓到的错误：角度残差的归一化、长度对距离的敏感性、
    /// 以及"左右灯条取残差小者"会选错分支。
    /// 一步更新的**方向**检验：观测比预测偏 +Δ，更新后预测必须朝 +Δ 动。
    /// 雅可比符号/坐标系写错时，滤波器会往反方向跑（"用了 UV 反而更差"的典型成因），
    /// 而这条检查与收敛速度、噪声大小都无关，是最锐利的判据。
    void testUpdateDirection()
    {
        const UvCamera camera = makeCamera();
        constexpr double half_width = 0.067;
        constexpr double half_height = 0.0285;
        constexpr double radius = 0.21;
        constexpr int plate = 2;   // yaw=0 时朝向相机的那块

        ArmorEKF::Config config;
        config.R = radius;

        // --- 四角点版本 ---
        {
            ArmorEKF ekf;
            ekf.setConfig(config);
            ekf.setDt(1.0 / 30.0);
            ekf.init(plateCenter(plate), plate);
            std::array<Vec<2>, 4> before{};
            check(ArmorEKF::predictUvCorners(ekf.getState(), plate, radius, half_width,
                                             half_height, 0.0, camera, before),
                  "direction: corners project");
            UvObservation observation;
            observation.plate_id = plate;
            observation.half_width = half_width;
            observation.half_height = half_height;
            observation.sigma_px = 1.0;
            const double shift = 12.0;
            for (int i = 0; i < 4; ++i) {
                observation.corners[static_cast<std::size_t>(i)] = before[static_cast<std::size_t>(i)];
                observation.corners[static_cast<std::size_t>(i)](0, 0) += shift;
            }
            ekf.updateUv(observation, camera);
            std::array<Vec<2>, 4> after{};
            check(ArmorEKF::predictUvCorners(ekf.getState(), plate, radius, half_width,
                                             half_height, 0.0, camera, after),
                  "direction: corners project after update");
            const double moved = after[0](0, 0) - before[0](0, 0);
            check(moved > 0.5 * shift,
                  "direction: corners update moves the prediction toward the observation");
        }

        // --- 紧凑（角度/中心/长度）版本 ---
        {
            ArmorEKF ekf;
            ekf.setConfig(config);
            ekf.setDt(1.0 / 30.0);
            ekf.init(plateCenter(plate), plate);
            auto_aim::UvCompactObservation before;
            check(ArmorEKF::predictUvCompact(ekf.getState(), plate, radius, half_width,
                                             half_height, 0.0, camera, 0, before),
                  "direction: compact projects");
            auto_aim::UvCompactObservation observation = before;
            observation.sigma_px = 1.0;
            observation.sigma_angle = 0.01;
            const double shift = 12.0;
            observation.center_x = before.center_x + shift;
            ekf.updateUvCompact(observation, camera);
            auto_aim::UvCompactObservation after;
            check(ArmorEKF::predictUvCompact(ekf.getState(), plate, radius, half_width,
                                             half_height, 0.0, camera, 0, after),
                  "direction: compact projects after update");
            const double moved = after.center_x - before.center_x;
            check(moved > 0.5 * shift,
                  "direction: compact update moves the prediction toward the observation");
            std::printf("direction: corner moved %.2f px, compact moved %.2f px (shift %.0f)\n",
                        after.center_x - before.center_x, moved, shift);
        }
    }

    void testCompactObservationConverges()
    {
        const UvCamera camera = makeCamera();
        constexpr double half_width = 0.067;
        constexpr double half_height = 0.0285;
        constexpr double radius = 0.21;
        constexpr double omega = 1.0;
        constexpr double dt = 1.0 / 30.0;

        ArmorEKF::Config config;
        config.R = radius;
        ArmorEKF ekf;
        ekf.setConfig(config);
        ekf.setDt(dt);
        ekf.init(plateCenter(2), 2);

        std::mt19937 generator(20260926);
        std::normal_distribution<double> pixel_noise(0.0, 1.0);
        std::normal_distribution<double> angle_noise(0.0, 0.01);

        double center_error = 0.0;
        double yaw_error = 0.0;
        double ohserve_count = 0.0;
        for (int frame = 0; frame < 120; ++frame) {
            const double time = frame * dt;
            const int plate = frame % 4;
            if (!plateFacesCamera(plate, omega * time)) {
                ekf.setDt(dt);
                ekf.predict();
                continue;
            }
            const Vec<9> truth = truthState(omega * time, omega);
            // 用真值状态生成紧凑观测量（左灯条）。
            auto_aim::UvCompactObservation truth_observation;
            if (!ArmorEKF::predictUvCompact(truth, plate, radius, half_width, half_height, 0.0,
                                            camera, 0, truth_observation)) {
                check(false, "uv compact: truth plate must be projectable");
                return;
            }
            truth_observation.center_x += pixel_noise(generator);
            truth_observation.center_y += pixel_noise(generator);
            truth_observation.length += pixel_noise(generator);
            truth_observation.angle += angle_noise(generator);
            truth_observation.sigma_px = 1.5;
            truth_observation.sigma_angle = 0.02;

            ekf.setDt(dt);
            ekf.predict();
            ohserve_count += ekf.updateUvCompact(truth_observation, camera);

            if (frame >= 90) {
                const Vec<9> estimate = ekf.getState();
                const double dx = estimate(0, 0) - truth(0, 0);
                const double dz = estimate(7, 0) - truth(7, 0);
                center_error = std::sqrt(dx * dx + dz * dz);
                const double estimated_yaw = std::atan2(estimate(4, 0), estimate(5, 0));
                yaw_error = std::abs(std::atan2(std::sin(estimated_yaw - omega * time),
                                                std::cos(estimated_yaw - omega * time)));
            }
        }
        check(ohserve_count > 10.0, "uv compact: observations are accepted");
        check(center_error < 0.12, "uv compact: chassis centre error converges below 0.12 m");
        check(yaw_error < 0.12, "uv compact: yaw error converges below 0.12 rad");
        std::printf("uv compact: centre %.3f m, yaw %.3f rad\n", center_error, yaw_error);
    }

    void testConvergesToTruth()
    {
        const UvCamera camera = makeCamera();
        constexpr double half_width = 0.067;
        constexpr double half_height = 0.0285;
        constexpr double radius = 0.21;
        constexpr double omega = 1.2;
        constexpr double dt = 1.0 / 30.0;

        ArmorEKF::Config config;
        config.R = radius;
        ArmorEKF ekf;
        ekf.setConfig(config);
        ekf.setDt(dt);
        // 位置用真值初始化，但角速度未知（omega=0）——滤波器必须自己把它估出来。
        ekf.init(plateCenter(2), 2);

        std::mt19937 generator(20260926);
        std::normal_distribution<double> pixel_noise(0.0, 1.0);

        double center_error = 0.0;
        double yaw_error = 0.0;
        double omega_error = 0.0;
        double residual_rms = 0.0;
        int residual_frames = 0;
        for (int frame = 0; frame < 120; ++frame) {
            const double time = frame * dt;
            const int plate = frame % 4;
            const Vec<9> truth = truthState(omega * time, omega);
            // 只喂**朝向相机**的板：背面板在真实检测里不会出现（左右标签会镜像）。
            if (!plateFacesCamera(plate, omega * time)) {
                ekf.setDt(dt);
                ekf.predict();
                continue;
            }
            std::array<Vec<2>, 4> corners{};
            if (!ArmorEKF::predictUvCorners(truth, plate, radius, half_width, half_height, 0.0,
                                            camera, corners)) {
                check(false, "uv: truth plate must be projectable");
                return;
            }
            UvObservation observation;
            observation.plate_id = plate;
            observation.half_width = half_width;
            observation.half_height = half_height;
            for (auto& corner : corners) {
                corner(0, 0) += pixel_noise(generator);
                corner(1, 0) += pixel_noise(generator);
            }
            observation.corners = corners;

            ekf.setDt(dt);
            ekf.predict();
            ekf.updateUv(observation, camera);

            if (frame >= 90) {
                const Vec<9> estimate = ekf.getState();
                const double dx = estimate(0, 0) - truth(0, 0);
                const double dz = estimate(7, 0) - truth(7, 0);
                center_error = std::sqrt(dx * dx + dz * dz);
                const double estimated_yaw = std::atan2(estimate(4, 0), estimate(5, 0));
                yaw_error = std::abs(std::atan2(std::sin(estimated_yaw - omega * time),
                                                std::cos(estimated_yaw - omega * time)));
                omega_error = std::abs(estimate(6, 0) - omega);
                std::array<Vec<2>, 4> predicted{};
                if (ArmorEKF::predictUvCorners(estimate, plate, radius, half_width,
                                               half_height, 0.0, camera, predicted)) {
                    double sum = 0.0;
                    for (int i = 0; i < 4; ++i) {
                        const double du = predicted[static_cast<std::size_t>(i)](0, 0) -
                                          observation.corners[static_cast<std::size_t>(i)](0, 0);
                        const double dv = predicted[static_cast<std::size_t>(i)](1, 0) -
                                          observation.corners[static_cast<std::size_t>(i)](1, 0);
                        sum += du * du + dv * dv;
                    }
                    residual_rms += std::sqrt(sum / 8.0);
                    ++residual_frames;
                }
            }
        }

        check(center_error < 0.10, "uv: chassis centre error converges below 0.10 m");
        check(yaw_error < 0.10, "uv: yaw error converges below 0.10 rad");
        check(omega_error < 0.25, "uv: yaw rate converges below 0.25 rad/s");
        if (residual_frames > 0) {
            residual_rms /= static_cast<double>(residual_frames);
        }
        check(residual_rms < 3.0, "uv: steady-state corner residual stays near 1 px");
        std::printf("uv ekf: centre %.3f m, yaw %.3f rad, omega %.3f rad/s, residual %.2f px\n",
                    center_error, yaw_error, omega_error, residual_rms);
    }
} // namespace

int main()
{
    testProjectionIsSelfConsistent();
    testUpdateDirection();
    testCompactObservationConverges();
    testConvergesToTruth();
    if (failures == 0) {
        std::printf("armor_ekf_uv_test: all checks passed\n");
        return 0;
    }
    std::printf("armor_ekf_uv_test: %d check(s) failed\n", failures);
    return 1;
}
