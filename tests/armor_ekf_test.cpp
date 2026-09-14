#include "ultra_vision/estimation/armor_ekf.hpp"
#include "ultra_vision/estimation/tracker.hpp"

#include <cmath>
#include <iostream>

namespace
{
    constexpr double kPi = 3.14159265358979323846;

    bool expect(bool condition, const char* message)
    {
        if (!condition) std::cerr << "FAILED: " << message << std::endl;
        return condition;
    }

    ultra_vision::YpdObservation makeObservation(
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
            ultra_vision::ArmorEKF::angleForPlate(plate_id);
        const double x = center_x + radius * std::cos(phase);
        const double y = center_y + armor_y_offset;
        const double z = center_z + radius * std::sin(phase);
        const double horizontal = std::hypot(x, z);
        const double distance = std::sqrt(x * x + y * y + z * z);

        ultra_vision::YpdObservation observation;
        observation.plate_id = plate_id;
        observation.yaw = std::atan2(x, z);
        observation.pitch = std::atan2(y, horizontal);
        observation.distance = distance;
        observation.armor_yaw = ultra_vision::ArmorEKF::normalizeAngle(phase);
        observation.yaw_noise = 2.5e-3;
        observation.pitch_noise = 5.0e-3;
        observation.distance_noise = 1.0e-2;
        observation.armor_yaw_noise = 4.0e-2;
        return observation;
    }

    ultra_vision::Armor makeTrackerArmor(
        double center_z,
        double yaw,
        double omega,
        double time,
        int plate_id,
        double radius)
    {
        const double phase = yaw + omega * time +
            ultra_vision::ArmorEKF::angleForPlate(plate_id);
        const cv::Vec3d radial(std::cos(phase), 0.0, std::sin(phase));
        const cv::Vec3d position(
            radius * radial[0], 0.0, center_z + radius * radial[2]);
        const cv::Vec3d vertical(0.0, 1.0, 0.0);
        const cv::Vec3d third = radial.cross(vertical);

        cv::Mat rotation = (cv::Mat_<double>(3, 3) <<
            radial[0], vertical[0], third[0],
            radial[1], vertical[1], third[1],
            radial[2], vertical[2], third[2]);

        ultra_vision::Armor armor;
        armor.solve_result = true;
        armor.armor_type = ultra_vision::small;
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

    ultra_vision::Armor toCameraArmor(
        const ultra_vision::Armor& home_armor, double yaw, double pitch)
    {
        const cv::Mat home_to_camera = cameraToHomeRotation(yaw, pitch).t();
        ultra_vision::Armor camera_armor = home_armor;
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
    ultra_vision::ArmorEKF::Config config;
    config.R = 0.21;
    config.process_acc = 1.0;
    config.process_omega = 400.0;

    const ultra_vision::TrackerConfig tracker_config;
    passed &= expect(tracker_config.process_noise_vel >= 400.0,
                     "tracker default must use small-gyro angular process noise");

    ultra_vision::ArmorEKF ekf;
    ekf.setConfig(config);

    constexpr double center_x = 0.0;
    constexpr double center_z = 5.0;
    constexpr double true_omega = 3.0;
    constexpr double radius = 0.21;
    constexpr int plate_id = 2;

    ultra_vision::ArmorEKF::Config offset_config = config;
    offset_config.armor_y_offsets = {{-0.0284, 0.0071, 0.0151, 0.0062}};
    ultra_vision::ArmorEKF offset_ekf;
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

    ultra_vision::Tracker tracker(tracker_config);
    constexpr int tracker_plate_id = 2;
    constexpr double tracker_center_z = 5.42;
    constexpr double tracker_dt = 1.0 / 30.0;
    for (int frame = 0; frame <= 14; ++frame) {
        const double time = frame * tracker_dt;
        std::vector<ultra_vision::Armor> armors{
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
        tracker.update(std::vector<ultra_vision::Armor>{}, time);
    }
    passed &= expect(std::abs(tracker.getOmega() - omega_before_loss) < 1e-6,
                     "short detection losses must preserve constant angular velocity");

    ultra_vision::Tracker intermittent_tracker(tracker_config);
    for (int frame = 0; frame <= 90; ++frame) {
        const double time = frame * tracker_dt;
        if (frame < 3 || frame % 10 < 4) {
            std::vector<ultra_vision::Armor> armors{
                makeTrackerArmor(tracker_center_z, 0.0, true_omega, time,
                                 tracker_plate_id, radius)
            };
            intermittent_tracker.update(armors, time);
        } else {
            intermittent_tracker.update(std::vector<ultra_vision::Armor>{}, time);
        }
    }
    passed &= expect(std::abs(intermittent_tracker.getOmega() - true_omega) < 0.8,
                     "intermittent detections must still converge small-gyro omega");

    ultra_vision::TrackerConfig reacquire_config = tracker_config;
    reacquire_config.max_lost_frames = 2;
    reacquire_config.reacquire_frames = 3;
    reacquire_config.reacquire_center_error = 0.20;
    reacquire_config.reacquire_max_dt = 0.20;
    ultra_vision::Tracker reacquire_tracker(reacquire_config);
    constexpr double reacquire_omega = 0.0;
    for (int frame = 0; frame <= 14; ++frame) {
        const double time = frame * tracker_dt;
        reacquire_tracker.update(std::vector<ultra_vision::Armor>{
            makeTrackerArmor(tracker_center_z, 0.0, reacquire_omega, time,
                             tracker_plate_id, radius)}, time);
    }
    for (int frame = 15; frame <= 17; ++frame) {
        reacquire_tracker.update(
            std::vector<ultra_vision::Armor>{}, frame * tracker_dt);
    }
    const double reacquire_time = 18.0 * tracker_dt;
    const bool reacquired = reacquire_tracker.update(std::vector<ultra_vision::Armor>{
        makeTrackerArmor(tracker_center_z, 0.0, reacquire_omega, reacquire_time,
                         tracker_plate_id, radius)}, reacquire_time);
    passed &= expect(reacquired &&
                         reacquire_tracker.getState() == ultra_vision::TrackerState::TRACKING,
                     "a nearby observation inside the reacquire window must restore tracking");

    ultra_vision::TrackerConfig timestamp_config = tracker_config;
    timestamp_config.max_camera_pose_dt = 0.02;
    ultra_vision::Tracker timestamp_tracker(timestamp_config);
    ultra_vision::CameraPose camera_pose;
    camera_pose.valid = true;
    camera_pose.timestamp_valid = true;
    camera_pose.timestamp = 1.0;
    timestamp_tracker.setCameraPose(camera_pose);
    passed &= expect(timestamp_tracker.update(std::vector<ultra_vision::Armor>{
                         makeTrackerArmor(5.0, 0.0, 0.0, 1.0,
                                          tracker_plate_id, radius)}, 1.0),
                     "a measurement with a synchronized camera pose must be accepted");
    passed &= expect(!timestamp_tracker.update(std::vector<ultra_vision::Armor>{
                         makeTrackerArmor(5.0, 0.0, 0.0, 1.1,
                                          tracker_plate_id, radius)}, 1.1),
                     "a measurement with a stale camera pose must be rejected");

    ultra_vision::Tracker world_tracker(tracker_config);
    const ultra_vision::Armor home_armor = makeTrackerArmor(
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
            std::vector<ultra_vision::Armor>{camera_armor}, time);

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

    if (!passed) return 1;
    std::cout << "armor_ekf_test passed, ekf_omega=" << ekf.getOmega()
              << ", tracker_omega=" << tracker.getOmega()
              << ", intermittent_omega=" << intermittent_tracker.getOmega()
              << std::endl;
    return 0;
}
