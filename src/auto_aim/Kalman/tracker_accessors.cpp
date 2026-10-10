#include "tracker.hpp"
#include "Kalman/tracker_utils.hpp"

#include <cmath>

using namespace auto_aim::tracker_detail;

namespace auto_aim
{
    cv::Mat Tracker::getTargetCenter() const
    {
        if (plate_model_weight_ > 0.0) return worldToCamera(blendedCenter());
        return worldToCamera(ekf_.getCenter());
    }

    std::array<double, 3> Tracker::getTargetCenterArray() const
    {
        const Vec<3> center = worldToCameraVector(
            plate_model_weight_ > 0.0 ? blendedCenter() : ekf_.getCenter());
        return {center(0, 0), center(1, 0), center(2, 0)};
    }

    std::array<double, 3> Tracker::getTargetVelocity() const
    {
        if (plate_model_weight_ > 0.0) {
            const Vec<3> velocity = worldToCameraVector(blendedVelocity());
            return {velocity(0, 0), velocity(1, 0), velocity(2, 0)};
        }
        const Vec<9> state = ekf_.getState();
        Vec<3> velocity;
        velocity(0, 0) = state(1, 0);
        velocity(1, 0) = state(3, 0);
        velocity(2, 0) = state(8, 0);
        velocity = worldToCameraVector(velocity);
        return {velocity(0, 0), velocity(1, 0), velocity(2, 0)};
    }

    std::array<double, 3> Tracker::getTargetCenterWorldArray() const
    {
        const Vec<3> center = ekf_.getCenter();
        return {center(0, 0), center(1, 0), center(2, 0)};
    }

    std::array<double, 3> Tracker::getTargetVelocityWorld() const
    {
        const Vec<9> state = ekf_.getState();
        return {state(1, 0), state(3, 0), state(8, 0)};
    }

    std::array<double, 3> Tracker::worldToCameraPoint(
        const std::array<double, 3>& point) const
    {
        Vec<3> home;
        home(0, 0) = point[0];
        home(1, 0) = point[1];
        home(2, 0) = point[2];
        const Vec<3> camera = pointToCamera(camera_pose_, home);
        return {camera(0, 0), camera(1, 0), camera(2, 0)};
    }

    double Tracker::worldToCameraYaw(double yaw) const
    {
        Vec<3> radial;
        radial(0, 0) = std::cos(yaw);
        radial(1, 0) = 0.0;
        radial(2, 0) = std::sin(yaw);
        const Vec<3> camera = worldToCameraVector(radial);
        return std::atan2(camera(2, 0), camera(0, 0));
    }

    std::array<cv::Mat, 4> Tracker::getEstimatedArmorPositions() const
    {
        std::array<cv::Mat, 4> positions;
        for (int plate = 0; plate < 4; ++plate) {
            positions[plate] = worldToCamera(ekf_.armorPosition(plate));
        }
        return positions;
    }

    double Tracker::getYaw() const
    {
        if (plate_model_weight_ > 0.0) {
            // 与整车模型同口径：world 系底盘 yaw（板朝向减去该板号的相位）。
            const double yaw = blendedYawWorld();
            Vec<3> radial;
            radial(0, 0) = std::cos(yaw);
            radial(1, 0) = 0.0;
            radial(2, 0) = std::sin(yaw);
            const Vec<3> camera = worldToCameraVector(radial);
            return std::atan2(camera(2, 0), camera(0, 0));
        }
        Vec<3> radial;
        radial(0, 0) = std::cos(ekf_.getYaw());
        radial(1, 0) = 0.0;
        radial(2, 0) = std::sin(ekf_.getYaw());
        radial = worldToCameraVector(radial);
        return std::atan2(radial(2, 0), radial(0, 0));
    }

    void Tracker::updateRotationRate(const Armor& armor, double timestamp)
    {
        const auto radial_angle = armorRadialAngle(armor);
        if (!radial_angle ||
            !rotation_rate_estimator_.update(*radial_angle, timestamp)) {
            // One unusable sample must not erase a rate the filter already
            // holds. Halving omega here used to make a single dropped frame
            // (or a plate switch) look like the chassis had stopped spinning.
            return;
        }
        ekf_.updateOmega(rotation_rate_estimator_.omega(),
                         cfg_.omega_measure_noise,
                         cfg_.omega_nis_threshold);
    }

    void Tracker::setCameraPose(const CameraPose& pose)
    {
        if (!pose.valid || !std::isfinite(pose.yaw) || !std::isfinite(pose.pitch)) {
            return;
        }
        camera_pose_ = pose;
    }

    void Tracker::setCameraAngles(double yaw, double pitch)
    {
        if (std::isfinite(yaw) && std::isfinite(pitch)) {
            camera_pose_.valid = true;
            camera_pose_.timestamp_valid = false;
            camera_pose_.yaw = yaw;
            camera_pose_.pitch = pitch;
        }
    }

    void Tracker::setCameraIntrinsics(double fx, double fy, double cx, double cy)
    {
        if (!(fx > 1.0) || !(fy > 1.0)) return;
        camera_fx_ = fx;
        camera_fy_ = fy;
        camera_cx_ = cx;
        camera_cy_ = cy;
        camera_intrinsics_valid_ = true;
    }

    void Tracker::setCameraDistortion(const cv::Mat& dist_coeffs)
    {
        camera_dist_ = dist_coeffs.clone();
        camera_dist_valid_ = !camera_dist_.empty() &&
                             (camera_dist_.rows * camera_dist_.cols) >= 4;
    }

    Armor Tracker::toWorld(const Armor& armor) const
    {
        if (!armor.solve_result || armor.tvec.empty()) return armor;

        if (!camera_pose_.valid) return armor;
        const Matrix<3, 3> camera_to_world = cameraToWorldRotation(camera_pose_);
        Armor result = armor;
        const Vec<3> position = pointToWorld(camera_pose_, toVec(armor.tvec));
        result.tvec = toMat(position);
        const auto world_rotation = transformRotation(camera_to_world, armor.rvec);
        if (world_rotation) result.rvec = *world_rotation;
        return result;
    }

    Vec<3> Tracker::worldToCameraVector(const Vec<3>& value) const
    {
        const Matrix<3, 3> world_to_camera = cameraToWorldRotation(
            camera_pose_).transpose();
        return transformVector(world_to_camera, value);
    }

    // ---- 双模型：由板模型反推整车状态 --------------------------------------
    // 板模型只跟踪可见板；用"板心 − R·径向"反推底盘中心，板号沿用整车模型的关联
    // 结果。这样 selector/ROI 依赖的"中心 + yaw + 半径"几何不需要改动。
    double Tracker::plateDerivedYaw() const
    {
        const int plate = std::max(0, std::min(3, plate_id_));
        return ArmorEKF::normalizeAngle(
            plate_ekf_.getYaw() - ArmorEKF::angleForPlate(plate));
    }

    Vec<3> Tracker::plateDerivedCenter() const
    {
        const int plate = std::max(0, std::min(3, plate_id_));
        const Vec<3> observed = plate_ekf_.getCenter();
        const double psi = plate_ekf_.getYaw();
        Vec<3> center;
        center(0, 0) = observed(0, 0) - cfg_.armor_radius * std::cos(psi);
        center(1, 0) = observed(1, 0) - cfg_.armor_y_offsets[plate];
        center(2, 0) = observed(2, 0) - cfg_.armor_radius * std::sin(psi);
        return center;
    }

    Vec<3> Tracker::plateDerivedVelocity() const
    {
        // 板心速度 − 板绕底盘中心的切向速度(R·ω·(−sinψ,0,cosψ)) = 底盘平移速度。
        const Vec<3> plate_velocity = plate_ekf_.getVelocity();
        const double psi = plate_ekf_.getYaw();
        const double tangential = cfg_.armor_radius * plate_ekf_.getOmega();
        Vec<3> velocity;
        velocity(0, 0) = plate_velocity(0, 0) + tangential * std::sin(psi);
        velocity(1, 0) = plate_velocity(1, 0);
        velocity(2, 0) = plate_velocity(2, 0) - tangential * std::cos(psi);
        return velocity;
    }

    // BOTH 档：两模型线性融合（world 系），w=1 全板模型、w=0 全整车模型。
    Vec<3> Tracker::blendedCenter() const
    {
        const Vec<3> a = ekf_.getCenter();
        const Vec<3> b = plateDerivedCenter();
        const double w = plate_model_weight_;
        Vec<3> out;
        for (int i = 0; i < 3; ++i) {
            out(i, 0) = (1.0 - w) * a(i, 0) + w * b(i, 0);
        }
        return out;
    }

    Vec<3> Tracker::blendedVelocity() const
    {
        const Vec<9> state = ekf_.getState();
        Vec<3> a;
        a(0, 0) = state(1, 0);
        a(1, 0) = state(3, 0);
        a(2, 0) = state(8, 0);
        const Vec<3> b = plateDerivedVelocity();
        const double w = plate_model_weight_;
        Vec<3> out;
        for (int i = 0; i < 3; ++i) {
            out(i, 0) = (1.0 - w) * a(i, 0) + w * b(i, 0);
        }
        return out;
    }

    double Tracker::blendedYawWorld() const
    {
        const double a = ekf_.getYaw();
        const double b = plateDerivedYaw();
        return ArmorEKF::normalizeAngle(a + plate_model_weight_ *
                                               ArmorEKF::normalizeAngle(b - a));
    }



}
