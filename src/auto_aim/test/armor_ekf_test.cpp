#include "Kalman/armor_ekf.hpp"
#include "Kalman/plate_ekf.hpp"
#include "Kalman/tracker.hpp"

#include <cmath>
#include <iostream>
#include <random>

namespace
{
    constexpr double kPi = 3.14159265358979323846;

    bool expect(bool condition, const char* message)
    {
        if (!condition) std::cerr << "FAILED: " << message << std::endl;
        return condition;
    }

    auto_aim::YpdObservation makeObservation(
        double center_x,
        double center_z,
        double yaw,
        double omega,
        double time,
        int plate_id,
        double radius,
        double center_y = 0.0,
        double armor_y_offset = 0.0)
    {
        const double phase = yaw + omega * time +
            auto_aim::ArmorEKF::angleForPlate(plate_id);
        const double x = center_x + radius * std::cos(phase);
        const double y = center_y + armor_y_offset;
        const double z = center_z + radius * std::sin(phase);
        const double horizontal = std::hypot(x, z);
        const double distance = std::sqrt(x * x + y * y + z * z);

        auto_aim::YpdObservation observation;
        observation.plate_id = plate_id;
        observation.yaw = std::atan2(x, z);
        observation.pitch = std::atan2(y, horizontal);
        observation.distance = distance;
        observation.armor_yaw = auto_aim::ArmorEKF::normalizeAngle(phase);
        observation.yaw_noise = 2.5e-3;
        observation.pitch_noise = 5.0e-3;
        observation.distance_noise = 1.0e-2;
        observation.armor_yaw_noise = 4.0e-2;
        return observation;
    }

    auto_aim::Armor makeTrackerArmor(
        double center_z,
        double yaw,
        double omega,
        double time,
        int plate_id,
        double radius)
    {
        const double phase = yaw + omega * time +
            auto_aim::ArmorEKF::angleForPlate(plate_id);
        const cv::Vec3d radial(std::cos(phase), 0.0, std::sin(phase));
        const cv::Vec3d position(
            radius * radial[0], 0.0, center_z + radius * radial[2]);
        const cv::Vec3d vertical(0.0, 1.0, 0.0);
        const cv::Vec3d third = radial.cross(vertical);

        cv::Mat rotation = (cv::Mat_<double>(3, 3) <<
            radial[0], vertical[0], third[0],
            radial[1], vertical[1], third[1],
            radial[2], vertical[2], third[2]);

        auto_aim::Armor armor;
        armor.solve_result = true;
        armor.armor_type = auto_aim::small;
        armor.tvec = (cv::Mat_<double>(3, 1) <<
            position[0], position[1], position[2]);
        cv::Rodrigues(rotation, armor.rvec);
        return armor;
    }

    /// @brief 与 makeTrackerArmor 同源，但给板心位置和径向角都加高斯噪声 ——
    ///        用来复现"5 rad/s 小陀螺 + PnP/角点噪声"的观测流（确定性种子）。
    auto_aim::Armor makeNoisyTrackerArmor(
        double center_z,
        double yaw,
        double omega,
        double time,
        int plate_id,
        double radius,
        std::mt19937& rng,
        double sigma_pos,
        double sigma_angle)
    {
        std::normal_distribution<double> position_noise(0.0, sigma_pos);
        std::normal_distribution<double> angle_noise(0.0, sigma_angle);
        const double phase = yaw + omega * time +
            auto_aim::ArmorEKF::angleForPlate(plate_id) + angle_noise(rng);
        const cv::Vec3d radial(std::cos(phase), 0.0, std::sin(phase));
        const cv::Vec3d position(
            radius * radial[0] + position_noise(rng),
            position_noise(rng),
            center_z + radius * radial[2] + position_noise(rng));
        const cv::Vec3d vertical(0.0, 1.0, 0.0);
        const cv::Vec3d third = radial.cross(vertical);

        cv::Mat rotation = (cv::Mat_<double>(3, 3) <<
            radial[0], vertical[0], third[0],
            radial[1], vertical[1], third[1],
            radial[2], vertical[2], third[2]);

        auto_aim::Armor armor;
        armor.solve_result = true;
        armor.armor_type = auto_aim::small;
        armor.tvec = (cv::Mat_<double>(3, 1) <<
            position[0], position[1], position[2]);
        cv::Rodrigues(rotation, armor.rvec);
        return armor;
    }

    cv::Mat cameraToHomeRotation(double yaw, double pitch)
    {
        const double cy = std::cos(yaw);
        const double sy = std::sin(yaw);
        const double cp = std::cos(pitch);
        const double sp = std::sin(pitch);

        const cv::Mat bevy = (cv::Mat_<double>(3, 3) <<
            cy, -sy * sp, -sy * cp,
            0.0, cp, -sp,
            sy, cy * sp, cy * cp);
        const cv::Mat optical_sign = (cv::Mat_<double>(3, 3) <<
            1.0, 0.0, 0.0,
            0.0, -1.0, 0.0,
            0.0, 0.0, -1.0);
        return optical_sign * bevy * optical_sign;
    }

    cv::Vec3d transformPoint(const cv::Mat& rotation, const cv::Vec3d& point)
    {
        const cv::Mat input = (cv::Mat_<double>(3, 1) <<
            point[0], point[1], point[2]);
        const cv::Mat output = rotation * input;
        return {output.at<double>(0), output.at<double>(1), output.at<double>(2)};
    }

    auto_aim::Armor toCameraArmor(
        const auto_aim::Armor& home_armor, double yaw, double pitch)
    {
        const cv::Mat home_to_camera = cameraToHomeRotation(yaw, pitch).t();
        auto_aim::Armor camera_armor = home_armor;
        const cv::Vec3d position(
            home_armor.tvec.at<double>(0),
            home_armor.tvec.at<double>(1),
            home_armor.tvec.at<double>(2));
        const cv::Vec3d camera_position = transformPoint(home_to_camera, position);
        camera_armor.tvec = (cv::Mat_<double>(3, 1) <<
            camera_position[0], camera_position[1], camera_position[2]);

        cv::Mat home_rotation;
        cv::Rodrigues(home_armor.rvec, home_rotation);
        const cv::Mat camera_rotation = home_to_camera * home_rotation;
        camera_armor.rvec = home_armor.rvec.clone();
        cv::Rodrigues(camera_rotation, camera_armor.rvec);
        return camera_armor;
    }
}

int main()
{
    bool passed = true;
    auto_aim::ArmorEKF::Config config;
    config.R = 0.21;
    config.process_acc = 1.0;
    config.process_omega = 400.0;

    const auto_aim::TrackerConfig tracker_config;
    passed &= expect(tracker_config.process_noise_vel >= 400.0,
                     "tracker default must use small-gyro angular process noise");

    auto_aim::ArmorEKF ekf;
    ekf.setConfig(config);

    constexpr double center_x = 0.0;
    constexpr double center_z = 5.0;
    constexpr double true_omega = 3.0;
    constexpr double radius = 0.21;
    constexpr int plate_id = 2;

    auto_aim::ArmorEKF::Config offset_config = config;
    offset_config.armor_y_offsets = {{-0.0284, 0.0071, 0.0151, 0.0062}};
    auto_aim::ArmorEKF offset_ekf;
    offset_ekf.setConfig(offset_config);
    constexpr double offset_center_y = 0.8;
    offset_ekf.updateYpd(makeObservation(
        center_x, center_z, 0.0, 0.0, 0.0, 0, radius,
        offset_center_y, offset_config.armor_y_offsets[0]));
    passed &= expect(std::abs(offset_ekf.armorPosition(0)(1, 0) -
                              (offset_center_y + offset_config.armor_y_offsets[0])) < 1e-9,
                     "EKF must preserve the calibrated y offset of the observed plate");
    passed &= expect(std::abs(offset_ekf.armorPosition(2)(1, 0) -
                              (offset_center_y + offset_config.armor_y_offsets[2])) < 1e-9,
                     "EKF must project calibrated y offsets onto other armor plates");

    const auto initial = makeObservation(
        center_x, center_z, 0.0, true_omega, 0.0, plate_id, radius);
    ekf.updateYpd(initial);

    constexpr double dt = 1.0 / 30.0;
    for (int frame = 1; frame <= 90; ++frame) {
        const double time = frame * dt;
        ekf.setDt(dt);
        ekf.predict();
        const auto observation = makeObservation(
            center_x, center_z, 0.0, true_omega, time, plate_id, radius);
        ekf.updateYpd(observation);
    }

    passed &= expect(std::abs(ekf.getOmega() - true_omega) < 0.35,
                     "EKF must recover small-gyro angular velocity");
    passed &= expect(std::abs(ekf.getCenter()(0, 0) - center_x) < 0.08,
                     "EKF must keep the rotating chassis centered");
    passed &= expect(std::abs(ekf.getCenter()(2, 0) - center_z) < 0.08,
                     "EKF must preserve the rotating chassis range");

    auto_aim::Tracker tracker(tracker_config);
    constexpr int tracker_plate_id = 2;
    constexpr double tracker_center_z = 5.42;
    constexpr double tracker_dt = 1.0 / 30.0;
    for (int frame = 0; frame <= 14; ++frame) {
        const double time = frame * tracker_dt;
        std::vector<auto_aim::Armor> armors{
            makeTrackerArmor(tracker_center_z, 0.0, true_omega, time,
                             tracker_plate_id, radius)
        };
        tracker.update(armors, time);
    }
    passed &= expect(std::abs(tracker.getOmega() - true_omega) < 0.6,
                     "Tracker update path must not damp small-gyro omega");

    const double omega_before_loss = tracker.getOmega();
    for (int frame = 1; frame <= 5; ++frame) {
        const double time = (14 + frame) * tracker_dt;
        tracker.update(std::vector<auto_aim::Armor>{}, time);
    }
    passed &= expect(std::abs(tracker.getOmega() - omega_before_loss) < 1e-6,
                     "short detection losses must preserve constant angular velocity");

    auto_aim::Tracker intermittent_tracker(tracker_config);
    for (int frame = 0; frame <= 90; ++frame) {
        const double time = frame * tracker_dt;
        if (frame < 3 || frame % 10 < 4) {
            std::vector<auto_aim::Armor> armors{
                makeTrackerArmor(tracker_center_z, 0.0, true_omega, time,
                                 tracker_plate_id, radius)
            };
            intermittent_tracker.update(armors, time);
        } else {
            intermittent_tracker.update(std::vector<auto_aim::Armor>{}, time);
        }
    }
    passed &= expect(std::abs(intermittent_tracker.getOmega() - true_omega) < 0.8,
                     "intermittent detections must still converge small-gyro omega");

    auto_aim::TrackerConfig reacquire_config = tracker_config;
    reacquire_config.max_lost_frames = 2;
    reacquire_config.reacquire_frames = 3;
    reacquire_config.reacquire_center_error = 0.20;
    reacquire_config.reacquire_max_dt = 0.20;
    auto_aim::Tracker reacquire_tracker(reacquire_config);
    constexpr double reacquire_omega = 0.0;
    for (int frame = 0; frame <= 14; ++frame) {
        const double time = frame * tracker_dt;
        reacquire_tracker.update(std::vector<auto_aim::Armor>{
            makeTrackerArmor(tracker_center_z, 0.0, reacquire_omega, time,
                             tracker_plate_id, radius)}, time);
    }
    for (int frame = 15; frame <= 17; ++frame) {
        reacquire_tracker.update(
            std::vector<auto_aim::Armor>{}, frame * tracker_dt);
    }
    const double reacquire_time = 18.0 * tracker_dt;
    const bool reacquired = reacquire_tracker.update(std::vector<auto_aim::Armor>{
        makeTrackerArmor(tracker_center_z, 0.0, reacquire_omega, reacquire_time,
                         tracker_plate_id, radius)}, reacquire_time);
    passed &= expect(reacquired &&
                         reacquire_tracker.getState() == auto_aim::TrackerState::TRACKING,
                     "a nearby observation inside the reacquire window must restore tracking");

    auto_aim::TrackerConfig timestamp_config = tracker_config;
    timestamp_config.max_camera_pose_dt = 0.02;
    auto_aim::Tracker timestamp_tracker(timestamp_config);
    auto_aim::CameraPose camera_pose;
    camera_pose.valid = true;
    camera_pose.timestamp_valid = true;
    camera_pose.timestamp = 1.0;
    timestamp_tracker.setCameraPose(camera_pose);
    passed &= expect(timestamp_tracker.update(std::vector<auto_aim::Armor>{
                         makeTrackerArmor(5.0, 0.0, 0.0, 1.0,
                                          tracker_plate_id, radius)}, 1.0),
                     "a measurement with a synchronized camera pose must be accepted");
    passed &= expect(!timestamp_tracker.update(std::vector<auto_aim::Armor>{
                         makeTrackerArmor(5.0, 0.0, 0.0, 1.1,
                                          tracker_plate_id, radius)}, 1.1),
                     "a measurement with a stale camera pose must be rejected");

    auto_aim::Tracker world_tracker(tracker_config);
    const auto_aim::Armor home_armor = makeTrackerArmor(
        5.0, 0.0, 0.0, 0.0, tracker_plate_id, radius);
    cv::Vec3d expected_center(0.0, 0.0, 5.0);
    for (int frame = 0; frame <= 45; ++frame) {
        const double time = frame * tracker_dt;
        const double camera_yaw = 0.35 * std::sin(0.17 * frame);
        const double camera_pitch = 0.08 * std::sin(0.11 * frame);
        world_tracker.setCameraAngles(camera_yaw, camera_pitch);
        const auto camera_armor = toCameraArmor(
            home_armor, camera_yaw, camera_pitch);
        world_tracker.update(
            std::vector<auto_aim::Armor>{camera_armor}, time);

        expected_center = transformPoint(
            cameraToHomeRotation(camera_yaw, camera_pitch).t(),
            cv::Vec3d(0.0, 0.0, 5.0));
        const auto center = world_tracker.getTargetCenterArray();
        if (frame >= 8) {
            const double center_error = std::sqrt(
                std::pow(center[0] - expected_center[0], 2) +
                std::pow(center[1] - expected_center[1], 2) +
                std::pow(center[2] - expected_center[2], 2));
            passed &= expect(center_error < 0.03,
                             "world-frame EKF must not drift when the gimbal moves");
            passed &= expect(std::abs(world_tracker.getOmega()) < 0.25,
                             "stationary chassis must keep near-zero omega");
        }
    }

    // ---- PnP yaw 翻解保护（pnp_yaw_trust_deg）-------------------------------
    // 平面靶 IPPE 有两个解：同一块板的 yaw 会跳（实测 ±12°，极端镜像）。位置（bearing）
    // 是良态的，所以位置对、yaw 被翻的观测流必须：① 仍然选对板；② 不把整车模型带歪；
    // ③ 不把假的 yaw 喂给差分测速器（那正是"模型突然转/崩"的来源）。
    {
        auto makeFlippedArmor = [](double center_z, double yaw, double omega, double time,
                                   int plate_id, double radius, double yaw_error) {
            auto_aim::Armor armor = makeTrackerArmor(center_z, yaw, omega, time, plate_id,
                                                     radius);
            if (yaw_error == 0.0) return armor;
            cv::Mat rotation;
            cv::Rodrigues(armor.rvec, rotation);
            cv::Mat spin;
            cv::Rodrigues(cv::Vec3d(0.0, 1.0, 0.0) * yaw_error, spin);   // 绕竖直轴转 yaw_error
            cv::Mat flipped = spin * rotation;                            // 位置不动，只翻朝向
            cv::Rodrigues(flipped, armor.rvec);
            return armor;
        };

        struct YawFlipResult {
            std::vector<double> plate_errors;
            double median_plate_error = 0.0;
            double max_plate_error = 0.0;
            double omega_error = 0.0;
            int wrong_plate_frames = 0;
        };
        auto run = [&](double trust_deg) {
            auto_aim::TrackerConfig cfg = tracker_config;
            cfg.pnp_yaw_trust_deg = trust_deg;
            auto_aim::Tracker tracker(cfg);
            constexpr double kCenterZ = 4.0;
            constexpr double kOmega = 2.0;
            constexpr double kRadius = 0.25;
            YawFlipResult out;
            for (int frame = 0; frame <= 80; ++frame) {
                const double time = frame * tracker_dt;
                // 交替翻解（±40°）—— 差分测速器最怕这种：常值偏置它还能抵消，交替就是假 ω。
                // 现实场景：**偶尔**翻解（5 帧里 1 帧），幅度 ±12°（本工程实测的
                // IPPE 双解抖动）。交替每帧都翻是极端情况，单板 yaw 本来就不可观测，
                // 门控只能保证"不炸"，这里测的是它该有的收益：坏帧别污染 ω/模型。
                const double flip = (frame % 5 == 0) ? ((frame % 10 == 0) ? 0.21 : -0.21)
                                                     : 0.0;
                tracker.update(std::vector<auto_aim::Armor>{
                    makeFlippedArmor(kCenterZ, 0.0, kOmega, time, tracker_plate_id, kRadius,
                                     flip)}, time);
                if (frame < 30) continue;   // 等它收敛
                if (tracker.getLastObservedPlateId() != tracker_plate_id)
                    ++out.wrong_plate_frames;
                const auto model = tracker.modelAt(0.0);
                if (!model.valid) continue;
                double worst = 0.0;
                for (int plate = 0; plate < 4; ++plate) {
                    const double phase = kOmega * time +
                        auto_aim::ArmorEKF::angleForPlate(plate);
                    const cv::Vec3d truth(kRadius * std::cos(phase), 0.0,
                                          kCenterZ + kRadius * std::sin(phase));
                    worst = std::max(worst, cv::norm(model.poses[plate].center - truth));
                }
                out.plate_errors.push_back(worst);
                out.max_plate_error = std::max(out.max_plate_error, worst);
            }
            out.omega_error = std::abs(tracker.getOmega() - kOmega);
            if (!out.plate_errors.empty()) {
                std::sort(out.plate_errors.begin(), out.plate_errors.end());
                out.median_plate_error = out.plate_errors[out.plate_errors.size() / 2];
            }
            return out;
        };

        const YawFlipResult guarded = run(25.0);
        const YawFlipResult naive = run(0.0);     // 老行为：yaw 无门控
        passed &= expect(guarded.wrong_plate_frames == 0,
                         "flipped PnP yaw must not make the association pick a neighbor plate");
        // 单板 + yaw 被翻时 yaw 本来就不可观测：门控下模型误差应在"半块板"量级以内，
        // 关键是**别让它变成几十厘米/几米**（naive 那一路）。
        passed &= expect(guarded.median_plate_error < 0.10,
                         "flipped PnP yaw must not swing the chassis model (guarded, median)");
        passed &= expect(guarded.omega_error < 0.3,
                         "flipped PnP yaw must not corrupt the differenced omega estimate");
        passed &= expect(guarded.median_plate_error + 0.05 < naive.median_plate_error,
                         "yaw trust gate must be clearly better than feeding the flipped yaw");
        std::cout << "yaw-flip: guarded med/max_plate_err=" << guarded.median_plate_error << "/"
                  << guarded.max_plate_error
                  << " omega_err=" << guarded.omega_error
                  << " | naive med/max_plate_err=" << naive.median_plate_error << "/"
                  << naive.max_plate_error
                  << " omega_err=" << naive.omega_error << std::endl;
    }

    // ---- 装甲板模型（双模型低速档，上科大 2026 §9.4.4）----------------------
    {
        // ---- 底盘模型：慢层几何 + 输出口径 + 外推 + 渲染姿态 ----------------
        // 这组断言盯的正是真机上"四块板高度不统一/底盘不对"的三个来源：
        //   · 半径（配置 0.25 / 实测 0.20~0.21）——在线改半径只动几何，不能 reset 滤波器；
        //   · 矩形底盘（前后距离 ≠ 左右距离）+ 四块板**同一高度** + 15° 安装倾角；
        //   · 重投影必须外推到"被显示那一帧"（观测延迟在 3 rad/s 上是 10~30°）。
        auto_aim::Tracker online_tracker(tracker_config);
        for (int frame = 0; frame <= 30; ++frame) {
            const double time = frame * tracker_dt;
            online_tracker.update(std::vector<auto_aim::Armor>{
                makeTrackerArmor(tracker_center_z, 0.0, true_omega, time,
                                 tracker_plate_id, radius)}, time);
        }
        auto_aim::CameraPose identity_pose;   // invalid → 用 tracker 内部姿态（无姿态=单位阵）
        const auto before_radius = online_tracker.modelAt(0.0, identity_pose);
        passed &= expect(before_radius.valid &&
                             std::abs(before_radius.geometry.meanDistance() -
                                      tracker_config.armor_radius) < 1e-9,
                         "chassis model must use the configured radius by default");
        passed &= expect(std::abs(before_radius.geometry.armor_pitch -
                                  tracker_config.armor_tilt_deg * kPi / 180.0) < 1e-9,
                         "chassis model must use the configured armor pitch (15 deg default)");

        online_tracker.setArmorRadius(0.185);
        passed &= expect(std::abs(online_tracker.getArmorRadius() - 0.185) < 1e-9,
                         "setArmorRadius must update the tracker geometry");
        passed &= expect(online_tracker.getState() == auto_aim::TrackerState::TRACKING,
                         "changing the radius must not reset the filter (state must stay TRACKING)");
        const auto after_radius = online_tracker.modelAt(0.0, identity_pose);
        passed &= expect(after_radius.valid, "chassis model must stay valid after setArmorRadius");
        for (int plate = 0; plate < 4; ++plate) {
            const double dx = after_radius.poses[plate].center[0] -
                              after_radius.center_camera[0];
            const double dy = after_radius.poses[plate].center[1] -
                              after_radius.center_camera[1];
            const double dz = after_radius.poses[plate].center[2] -
                              after_radius.center_camera[2];
            const double plate_radius = std::sqrt(dx * dx + dy * dy + dz * dz);
            passed &= expect(std::abs(plate_radius - 0.185) < 1e-6,
                             "modeled plate must sit on the new armor radius");
        }

        // 外推：yaw 前进 ω·dt，四块板绕模型中心刚性旋转同一个角
        constexpr double extrapolate_dt = 0.08;
        const auto m0 = online_tracker.modelAt(0.0, identity_pose);
        const auto m1 = online_tracker.modelAt(extrapolate_dt, identity_pose);
        const double yaw_step = auto_aim::ArmorEKF::normalizeAngle(
            m1.world_yaw - m0.world_yaw);
        passed &= expect(std::abs(yaw_step - m0.omega * extrapolate_dt) < 0.02,
                         "continuity: chassis model yaw must advance by omega*dt");
        const double phi = m0.omega * extrapolate_dt;
        for (int plate = 0; plate < 4; ++plate) {
            const double dx = m0.poses[plate].center[0] - m0.center_camera[0];
            const double dy = m0.poses[plate].center[1] - m0.center_camera[1];
            const double dz = m0.poses[plate].center[2] - m0.center_camera[2];
            const double ex = m0.center_camera[0] + dx * std::cos(phi) - dz * std::sin(phi);
            const double ey = m0.center_camera[1] + dy;
            const double ez = m0.center_camera[2] + dx * std::sin(phi) + dz * std::cos(phi);
            const double err = std::sqrt(
                std::pow(m1.poses[plate].center[0] - ex, 2) +
                std::pow(m1.poses[plate].center[1] - ey, 2) +
                std::pow(m1.poses[plate].center[2] - ez, 2));
            passed &= expect(err < 0.01,
                             "continuity: plates must rotate rigidly with the extrapolated chassis");
        }

        // ---- 矩形底盘 + 同一高度 + 板面带 15° 倾角 --------------------------
        // 真车不是"四块板在一个圆上"：前后距离 ≠ 左右距离，板心同一高度。
        // 这条是最小回归：每块板必须落在自己那条半径/高度上，姿态绕自身切线倾 α。
        auto_aim::ChassisGeometry rect;
        rect.plate_distance = {{0.28, 0.22, 0.28, 0.22}};
        rect.plate_height = {{0.01, 0.01, 0.01, 0.01}};
        rect.armor_pitch = 15.0 * kPi / 180.0;
        rect.plate_large = {{false, true, false, true}};
        online_tracker.setChassisGeometry(rect);
        const auto rect_model = online_tracker.modelAt(0.0, identity_pose);
        passed &= expect(rect_model.valid, "rectangular chassis model must be valid");
        for (int plate = 0; plate < 4; ++plate) {
            const double px = rect_model.poses[plate].center[0] - rect_model.center_camera[0];
            const double py = rect_model.poses[plate].center[1] - rect_model.center_camera[1];
            const double pz = rect_model.poses[plate].center[2] - rect_model.center_camera[2];
            const double horizontal = std::sqrt(px * px + pz * pz);
            passed &= expect(std::abs(horizontal - rect.distance(plate)) < 1e-6,
                             "rectangular chassis: each plate sits on its own distance");
            passed &= expect(std::abs(py + rect.height(plate)) < 1e-6,
                             "chassis plates share the configured height (world y points down)");
            const double phase = rect_model.world_yaw + auto_aim::armorPhaseOffset(plate);
            const cv::Vec3d radial(std::cos(phase), 0.0, std::sin(phase));
            passed &= expect(std::abs(rect_model.poses[plate].axes.bar.dot(radial) +
                                      std::sin(rect.armor_pitch)) < 1e-9,
                             "bar axis leans inward by exactly the armor pitch");
        }
        passed &= expect(std::abs(online_tracker.armorHalfWidth(1) - rect.large_half_width) < 1e-12 &&
                             std::abs(online_tracker.armorHalfWidth(0) - rect.small_half_width) < 1e-12,
                         "plate size follows the per-plate large/small flag");
    }

    // ---- 装甲板模型（双模型低速档，上科大 2026 §9.4.4）----------------------
    {
        auto_aim::PlateEKF::Config plate_config;
        plate_config.process_acc = 50.0;
        plate_config.process_yaw_acc = 100.0;
        auto_aim::PlateEKF plate(plate_config);
        auto_aim::Vec<3> center;
        center(0, 0) = 0.0;
        center(1, 0) = 0.0;
        center(2, 0) = 5.0;
        plate.init(center, 0.0);
        constexpr double plate_dt = 1.0 / 30.0;
        constexpr double true_vx = 0.4;
        constexpr double true_plate_omega = 0.5;
        std::mt19937 plate_rng(7);
        std::normal_distribution<double> plate_noise(0.0, 0.001);
        for (int frame = 1; frame <= 90; ++frame) {
            plate.setDt(plate_dt);
            plate.predict();
            auto_aim::Vec<3> observation;
            observation(0, 0) = true_vx * frame * plate_dt + plate_noise(plate_rng);
            observation(1, 0) = plate_noise(plate_rng);
            observation(2, 0) = 5.0 + plate_noise(plate_rng);
            plate.update(observation, true_plate_omega * frame * plate_dt +
                             plate_noise(plate_rng),
                         9.488, 1e-4, 4e-4);
        }
        std::cout << "plate ekf: vx=" << plate.getVelocity()(0, 0)
                  << " omega=" << plate.getOmega() << std::endl;
        passed &= expect(std::abs(plate.getVelocity()(0, 0) - true_vx) < 0.1,
                         "PlateEKF must estimate a translating plate's velocity");
        passed &= expect(std::abs(plate.getOmega() - true_plate_omega) < 0.15,
                         "PlateEKF must estimate a plate's yaw rate");
    }

    // ---- 双模型（§9.4.4）：Tracker 并行跑装甲板模型 -------------------------
    {
        auto_aim::TrackerConfig dual_config = tracker_config;
        dual_config.plate_model_enabled = true;
        auto_aim::Tracker dual_tracker(dual_config);
        std::mt19937 dual_rng(11u);
        constexpr double dual_omega = 1.0;
        for (int frame = 0; frame <= 150; ++frame) {
            const double time = frame * tracker_dt;
            dual_tracker.update(std::vector<auto_aim::Armor>{
                makeNoisyTrackerArmor(tracker_center_z, 0.0, dual_omega, time,
                                      tracker_plate_id, radius, dual_rng, 0.01, 0.01)},
                time);
        }
        std::cout << "dual model: ekf_omega=" << dual_tracker.getOmega()
                  << " plate_omega=" << dual_tracker.getPlateModelOmega()
                  << " has_plate=" << dual_tracker.hasPlateModel() << std::endl;
        passed &= expect(dual_tracker.hasPlateModel(),
                         "dual model must initialize the armor-plate model");
        passed &= expect(std::abs(dual_tracker.getPlateModelOmega() - dual_omega) < 0.5,
                         "armor-plate model must track a slow plate's yaw rate");
    }

    // ---- IESEKF：先验偏离真值较远时，迭代重新线性化不应比单次 EKF 更差 --------
    {
        auto_aim::ArmorEKF::Config single_config = config;
        single_config.iekf_iterations = 1;   // 标准 EKF
        auto_aim::ArmorEKF::Config iterative_config = config;
        iterative_config.iekf_iterations = 4;
        auto_aim::ArmorEKF single_ekf;
        auto_aim::ArmorEKF iterative_ekf;
        single_ekf.setConfig(single_config);
        iterative_ekf.setConfig(iterative_config);

        auto_aim::Vec<3> center;
        center(0, 0) = center_x;
        center(2, 0) = center_z;
        constexpr double yaw_offset = 0.79;   // 先验偏 45°
        single_ekf.initAt(center, yaw_offset);
        iterative_ekf.initAt(center, yaw_offset);

        const auto observation = makeObservation(
            center_x, center_z, 0.0, 0.0, 0.0, 0, radius);
        single_ekf.updateYpd(observation);
        iterative_ekf.updateYpd(observation);

        const double single_error = std::abs(auto_aim::ArmorEKF::normalizeAngle(
            single_ekf.getPlateAngle(0) - observation.armor_yaw));
        const double iterative_error = std::abs(auto_aim::ArmorEKF::normalizeAngle(
            iterative_ekf.getPlateAngle(0) - observation.armor_yaw));
        std::cout << "iesekf yaw error (deg): single=" << single_error * 180.0 / kPi
                  << " iterative=" << iterative_error * 180.0 / kPi << std::endl;
        passed &= expect(iterative_error <= single_error * 1.05 + 1e-9,
                         "IESEKF must not be worse than a single EKF update");
    }

    // ---- 5 rad/s（小陀螺）通用性：关联门限必须随 yaw 不确定度自适应 ----------
    // 旧行为用固定的 20° 角门限：5 rad/s 下 ω 估计还没建立，预测每帧落后
    // ω·dt≈10°，两三帧就顶穿门限 → 关联返回 inf → 观测被拒 → ω 永远学不到。
    // 这里喂确定性观测流（单板可见、可见板号随旋转切换、带 PnP 噪声）做 A/B：
    // adaptive_association_gate + soft_nis_gate 打开能收敛，关闭（旧行为）不能。
    {
        const auto spin_omega = [&](bool adaptive, bool soft, bool mahalanobis,
                                    double yaw_trust_deg = 25.0) {
            auto_aim::TrackerConfig spin_config = tracker_config;
            spin_config.adaptive_association_gate = adaptive;
            spin_config.soft_nis_gate = soft;
            spin_config.mahalanobis_matching = mahalanobis;
            // 0 = 老的 yaw 硬门限关联（A/B 用；见 pnp_yaw_trust_deg）
            spin_config.pnp_yaw_trust_deg = yaw_trust_deg;
            auto_aim::Tracker spin_tracker(spin_config);
            std::mt19937 rng(20261005u);
            constexpr double spin_rate = 5.0;
            constexpr double spin_center_z = 5.0;
            constexpr double spin_yaw0 = 0.3;
            constexpr double spin_sigma_pos = 0.012;
            constexpr double spin_sigma_angle = 0.012;
            constexpr double spin_dt = 1.0 / 12.0;   // 低帧率：每帧转 24° > 旧 20° 门限
            for (int frame = 0; frame <= 60; ++frame) {
                const double time = frame * spin_dt;
                if (frame > 0 && frame % 6 == 0) {
                    spin_tracker.update(std::vector<auto_aim::Armor>{}, time);
                    continue;   // 模拟一次检测丢失
                }
                // 只有最正对相机的板可见：center·radial = center_z·sinφ 最负者。
                int visible = 0;
                double most_front = 1e9;
                for (int plate = 0; plate < 4; ++plate) {
                    const double phase = spin_yaw0 + spin_rate * time +
                        auto_aim::ArmorEKF::angleForPlate(plate);
                    const double facing = std::sin(phase);
                    if (facing < most_front) {
                        most_front = facing;
                        visible = plate;
                    }
                }
                spin_tracker.update(std::vector<auto_aim::Armor>{
                    makeNoisyTrackerArmor(
                        spin_center_z, spin_yaw0, spin_rate, time, visible, radius,
                        rng, spin_sigma_pos, spin_sigma_angle)}, time);
            }
            return spin_tracker.getOmega();
        };
        const double spin_maha = spin_omega(true, true, true);
        const double spin_new = spin_omega(true, true, false);
        // legacy = 老的 yaw 硬门限关联（把 pnp_yaw_trust_deg 设 0 才等价于旧行为）
        const double spin_old = spin_omega(false, false, false, /*yaw_trust_deg=*/0.0);
        std::cout << "5rad/s deterministic: mahalanobis=" << spin_maha
                  << " adaptive_gate=" << spin_new
                  << " legacy=" << spin_old << std::endl;
        passed &= expect(std::abs(spin_maha - 5.0) < 1.0,
                         "5 rad/s must converge with Mahalanobis association");
        passed &= expect(std::abs(spin_new - 5.0) < 1.0,
                         "5 rad/s must converge with the adaptive association gate");
        passed &= expect(std::abs(spin_old - 5.0) > 1.0,
                         "legacy fixed gate must reproduce the documented 5 rad/s failure");
    }

    // ---- 收敛看门狗 ----------------------------------------------------------
    // 判据一（量测长期被拒）、判据二（状态发散）、判据三（帧间隔过长）各测一次，
    // 外加"正常数据不许误触发"和"命中后必须能重新初始化并再次收敛"。
    auto_aim::Tracker clean_tracker(tracker_config);
    for (int frame = 0; frame <= 120; ++frame) {
        const double time = frame * tracker_dt;
        clean_tracker.update(std::vector<auto_aim::Armor>{
            makeTrackerArmor(tracker_center_z, 0.0, true_omega, time,
                             tracker_plate_id, radius)}, time);
    }
    {
        const auto wd = clean_tracker.watchdogStats();
        passed &= expect(wd.resets == 0,
                         "watchdog must not trip on consistent observations");
        passed &= expect(wd.converged,
                         "watchdog must report converged on consistent observations");
        passed &= expect(wd.reject_ratio < 0.05,
                         "watchdog reject ratio must stay near zero on good data");
    }

    // 判据一：把 NIS 门限压到 0（任何量测都被拒）→ 必须在 min_samples 帧后触发。
    {
        auto_aim::TrackerConfig reject_config = tracker_config;
        // 这条专门测"硬 NIS 门限 + reject 比例"的旧路径，所以显式关掉软门限。
        reject_config.soft_nis_gate = false;
        reject_config.ypd_nis_threshold = 0.0;
        reject_config.watchdog_min_samples = 8;
        reject_config.watchdog_window = 20;
        reject_config.watchdog_reject_ratio = 0.4;
        auto_aim::Tracker reject_tracker(reject_config);
        bool saw_lost = false;
        for (int frame = 0; frame <= 40; ++frame) {
            const double time = frame * tracker_dt;
            reject_tracker.update(std::vector<auto_aim::Armor>{
                makeTrackerArmor(tracker_center_z, 0.0, true_omega, time,
                                 tracker_plate_id, radius)}, time);
            if (reject_tracker.getState() == auto_aim::TrackerState::LOST) saw_lost = true;
        }
        const auto wd = reject_tracker.watchdogStats();
        passed &= expect(wd.resets >= 1 && saw_lost,
                         "watchdog must trip when measurements are persistently rejected");
        passed &= expect(!wd.converged,
                         "watchdog must not report converged while rejecting measurements");
    }

    // 判据二：把允许的 |center| 压到 1 mm（任何真实状态都越界）→ 必须触发。
    {
        auto_aim::TrackerConfig diverge_config = tracker_config;
        diverge_config.watchdog_max_range = 0.001;
        auto_aim::Tracker diverge_tracker(diverge_config);
        for (int frame = 0; frame <= 20; ++frame) {
            const double time = frame * tracker_dt;
            diverge_tracker.update(std::vector<auto_aim::Armor>{
                makeTrackerArmor(tracker_center_z, 0.0, true_omega, time,
                                 tracker_plate_id, radius)}, time);
        }
        passed &= expect(diverge_tracker.watchdogStats().resets >= 1,
                         "watchdog must trip when the state leaves its physical bounds");
    }

    // 判据三 + 恢复：间隔 0.3 s > max_frame_gap=0.05 → 触发；下一帧必须能用观测
    // 重新初始化（跳过 tryReacquire），并在 3 帧后重新报"已收敛"，且不再重复触发。
    {
        auto_aim::TrackerConfig gap_config = tracker_config;
        gap_config.watchdog_max_frame_gap = 0.05;
        gap_config.watchdog_min_converge_updates = 3;
        auto_aim::Tracker gap_tracker(gap_config);
        for (int frame = 0; frame <= 30; ++frame) {
            const double time = frame * tracker_dt;
            gap_tracker.update(std::vector<auto_aim::Armor>{
                makeTrackerArmor(tracker_center_z, 0.0, 0.0, time,
                                 tracker_plate_id, radius)}, time);
        }
        passed &= expect(gap_tracker.watchdogStats().resets == 0,
                         "watchdog must not trip while frames arrive on time");
        const double gap_time = 30.0 * tracker_dt + 0.30;
        gap_tracker.update(std::vector<auto_aim::Armor>{
            makeTrackerArmor(tracker_center_z, 0.0, 0.0, gap_time,
                             tracker_plate_id, radius)}, gap_time);
        const int resets_after_gap = gap_tracker.watchdogStats().resets;
        passed &= expect(resets_after_gap == 1,
                         "watchdog must trip exactly once on a long frame gap");
        for (int frame = 1; frame <= 3; ++frame) {
            const double time = gap_time + frame * tracker_dt;
            gap_tracker.update(std::vector<auto_aim::Armor>{
                makeTrackerArmor(tracker_center_z, 0.0, 0.0, time,
                                 tracker_plate_id, radius)}, time);
        }
        const auto wd = gap_tracker.watchdogStats();
        passed &= expect(wd.resets == resets_after_gap,
                         "watchdog must not re-trip after re-initialization");
        passed &= expect(wd.converged,
                         "watchdog must report converged again after re-initialization");
    }

    if (!passed) return 1;
    std::cout << "armor_ekf_test passed, ekf_omega=" << ekf.getOmega()
              << ", tracker_omega=" << tracker.getOmega()
              << ", intermittent_omega=" << intermittent_tracker.getOmega()
              << std::endl;
    return 0;
}
