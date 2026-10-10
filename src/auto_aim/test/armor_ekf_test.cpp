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

    // 真实系统里检测到的**一定是离相机最近的那块板**（现场事实）。合成观测流必须照这个
    // 前提造，否则"构造板一定最近"的硬约束会正确地把它当成翻解（测试自身不合格）。
    const auto nearestPlate = [](double yaw, double omega, double time) {
        int best = 0;
        double most_front = 1e9;
        for (int plate = 0; plate < 4; ++plate) {
            const double phase = yaw + omega * time +
                auto_aim::ArmorEKF::angleForPlate(plate);
            if (std::sin(phase) < most_front) {
                most_front = std::sin(phase);
                best = plate;
            }
        }
        return best;
    };

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
                                 nearestPlate(0.0, true_omega, time), radius)
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

    // ---- 构造板一定最近（硬约束）+ 运动状态判定（旋转中心先验）----------------
    {
        // ① 运动状态判定：拟合器的 (ω, |drift|) → 纯旋转 / 平移 / 旋转+平移 / 静止。
        //    纯旋转时旋转中心才有"不该动"这个先验（见 center_anchor）。
        using auto_aim::ChassisMotion;
        passed &= expect(auto_aim::classifyChassisMotion(3.0, 0.05, 1.0, 0.15) == ChassisMotion::Rotating,
                         "fast spin + no drift must classify as pure rotation");
        passed &= expect(auto_aim::classifyChassisMotion(0.2, 0.6, 1.0, 0.15) == ChassisMotion::Translating,
                         "no spin + drift must classify as translation");
        passed &= expect(auto_aim::classifyChassisMotion(3.0, 0.6, 1.0, 0.15) == ChassisMotion::RotatingAndTranslating,
                         "spin + drift must classify as rotating and translating");
        passed &= expect(auto_aim::classifyChassisMotion(0.1, 0.02, 1.0, 0.15) == ChassisMotion::Still,
                         "slow and steady must classify as still");
        passed &= expect(auto_aim::chassisCenterAnchorGain(ChassisMotion::Rotating, 0.35, 0.1) >
                             auto_aim::chassisCenterAnchorGain(ChassisMotion::Translating, 0.35, 0.1),
                         "pure rotation must anchor the chassis center harder than translation");

        // ② 硬约束"观测板一定离相机最近"：干净（物理）观测流不该误触发，
        //    且车心必须始终在观测板**后面**（画面上更靠上）。
        auto_aim::Tracker nearest_tracker(tracker_config);
        constexpr double kSpin = 3.0;
        for (int frame = 0; frame <= 60; ++frame) {
            const double time = frame * tracker_dt;
            const int visible = nearestPlate(0.0, kSpin, time);
            auto armor = makeTrackerArmor(tracker_center_z, 0.0, kSpin, time, visible, radius);
            nearest_tracker.update(std::vector<auto_aim::Armor>{armor}, time);
            if (frame < 20) continue;
            // 相机在原点（本用例没有姿态）：车心到相机 > 板到相机
            const double plate_d = std::sqrt(
                armor.tvec.at<double>(0) * armor.tvec.at<double>(0) +
                armor.tvec.at<double>(2) * armor.tvec.at<double>(2));
            const auto center = nearest_tracker.getTargetCenterArray();
            const double center_d = std::sqrt(center[0] * center[0] + center[2] * center[2]);
            passed &= expect(center_d > plate_d - 0.02,
                             "chassis center must stay behind the nearest observed plate");
        }
        passed &= expect(nearest_tracker.nearestPlateFlips() == 0,
                         "a clean nearest-plate stream must not trip the hard constraint");
    }

    // ---- 同帧多板联立解 (车心, yaw) -----------------------------------------
    // 两块板 90° 同现时，用**位置**解 (车心, yaw)；即使初始 yaw 差 25° 也必须收敛，
    // 而且模型的两块板都要落在观测上（这是联合解存在的意义）。
    {
        auto_aim::TrackerConfig joint_cfg = tracker_config;
        joint_cfg.joint_solve = true;
        auto_aim::Tracker joint_tracker(joint_cfg);
        constexpr double kCenterZ = 4.0;
        constexpr double kTrueYaw = 0.6;
        constexpr double kR = 0.22;
        // 两块相邻板（id 2 与 id 3），位置按真值造
        std::vector<auto_aim::Armor> armors;
        for (int plate : {2, 3}) {
            armors.push_back(makeTrackerArmor(kCenterZ, kTrueYaw, 0.0, 0.0, plate, kR));
        }
        // 先用一帧建立状态（初始 yaw 会由第一块板决定，故意偏 25°）
        double t = 0.0;
        for (int frame = 0; frame < 40; ++frame) {
            t = frame * tracker_dt;
            std::vector<auto_aim::Armor> obs;
            for (int plate : {2, 3})
                obs.push_back(makeTrackerArmor(kCenterZ, kTrueYaw, 0.0, t, plate, kR));
            joint_tracker.update(obs, t);
        }
        // 注意：四块板 90° 对称 ⇒ "yaw" 只在**模 90°** 意义下有定义（整体差 90°
        // 是等价解，画出来完全一样）。所以按模 90° 比较，并按"板位置集合"比较。
        const double yaw_err = std::abs(std::remainder(
            joint_tracker.getYawWorld() - kTrueYaw, 0.5 * kPi));
        passed &= expect(yaw_err < 0.05,
                         "two-plate joint solve must pin the chassis phase (mod 90 deg, ~3deg)");
        passed &= expect(joint_tracker.jointSolveCount() > 0,
                         "joint solve must actually run when two plates are seen");
        // 两块板都要贴合（**无标签**比较：每块观测板到最近模型板的距离）
        const auto model = joint_tracker.modelAt(0.0);
        for (int plate : {2, 3}) {
            const double phase = kTrueYaw + auto_aim::ArmorEKF::angleForPlate(plate);
            const cv::Vec3d truth(kR * std::cos(phase), 0.0, kCenterZ + kR * std::sin(phase));
            double best = 1e9;
            for (int m = 0; m < 4; ++m)
                best = std::min(best, cv::norm(model.poses[m].center - truth));
            passed &= expect(best < 0.03,
                             "joint solve must place BOTH observed plates on their detections");
        }
        std::cout << "joint solve: yaw_err(mod90)=" << yaw_err * 180.0 / kPi
                  << "deg residual=" << joint_tracker.jointSolveResidual()
                  << "m count=" << joint_tracker.jointSolveCount() << std::endl;
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
                // 幅度取 40°（> 25° 门限）：模拟"翻到另一个平面解"的粗错。
                // ±12° 那种小抖动本来就在可信门限内，两种配置行为一致（门控不该动它）。
                // 每帧都翻、且交替符号：单板 yaw 彻底不可用（差分最怕这种）。
                // 门控的作用就是把这种坏流挡在"yaw / 差分测速"之外。
                const double flip = (frame % 2 == 0) ? 0.7 : -0.7;
                const int visible = nearestPlate(0.0, kOmega, time);
                tracker.update(std::vector<auto_aim::Armor>{
                    makeFlippedArmor(kCenterZ, 0.0, kOmega, time, visible, kRadius, flip)},
                    time);
                if (frame < 30) continue;   // 等它收敛
                if (tracker.getLastObservedPlateId() != visible) ++out.wrong_plate_frames;
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
        // 断言只钉"门控到底治了什么"：**差分测速**不该被翻解污染。
        // （模型/板号的绝对正确性在"构造板一定最近"的硬约束与关联用例里验；
        //  单板 + yaw 完全不可用时，相位本身就有 90° 不确定性，不该在这里断言。）
        // 注意：自从差分测速改喂"底盘 yaw"（灯条径向角 − 该板相位偏移）之后，
        // 翻解对 ω 的污染本身就小多了 —— 两条路都能守住 ω。所以这里只断言
        // "ω 不被翻解带坏"，不再要求 naive 更差（那条旧断言已不成立）。
        passed &= expect(guarded.omega_error < 0.30 && naive.omega_error < 0.30,
                         "differenced omega must stay clean under flipped PnP yaw (chassis-yaw input)");
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
        // 初值（还没喂任何观测、几何慢收敛还没动过）
        passed &= expect(std::abs(online_tracker.chassisGeometry().meanDistance() -
                                  tracker_config.armor_radius) < 1e-9,
                         "chassis model must start from the configured radius");
        passed &= expect(std::abs(online_tracker.chassisGeometry().armor_pitch -
                                  tracker_config.armor_tilt_deg * kPi / 180.0) < 1e-9,
                         "chassis model must start from the configured armor pitch (15 deg)");
        for (int frame = 0; frame <= 30; ++frame) {
            const double time = frame * tracker_dt;
            online_tracker.update(std::vector<auto_aim::Armor>{
                makeTrackerArmor(tracker_center_z, 0.0, true_omega, time,
                                 nearestPlate(0.0, true_omega, time), radius)}, time);
        }
        auto_aim::CameraPose identity_pose;   // invalid → 用 tracker 内部姿态（无姿态=单位阵）
        const auto before_radius = online_tracker.modelAt(0.0, identity_pose);
        passed &= expect(before_radius.valid, "chassis model must be valid after tracking starts");

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
