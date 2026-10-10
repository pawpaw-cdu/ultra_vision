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

    // 第 plate 块板的刚体姿态（相机系）。径向直接由滤波器的 yaw + 板相位给出
    // （和 armorPosition() 用的完全是同一个相位，所以板心与姿态必然自洽）；
    // 车体竖直 = 世界系竖直转到相机系（EKF 的 world y 向下 → up = −y）；
    // 灯条方向再由安装倾角 α 定死。这样"板心 + 切向 + 灯条方向"三件套把板定死，
    // 不需要借任何观测板的东西（老 overlay 拿观测板 PnP 的 rvec 当整车竖直，
    // 会把观测板的 roll 传染给另外三块板）。
    ArmorAxes Tracker::armorAxesCamera(int plate) const
    {
        return armorAxesForYaw(ekf_.getYaw(), plate);
    }

    ArmorAxes Tracker::armorAxesForYaw(double world_yaw, int plate) const
    {
        const int clamped = std::max(0, std::min(3, plate));
        const double plate_angle = ArmorEKF::normalizeAngle(
            world_yaw + ArmorEKF::angleForPlate(clamped));
        const cv::Vec3d radial_home(std::cos(plate_angle), 0.0, std::sin(plate_angle));
        const cv::Vec3d up_home(0.0, -1.0, 0.0);   // world y 向下 → 竖直向上是 −y
        const cv::Vec3d radial_camera = worldDirectionToCamera(radial_home);
        const cv::Vec3d up_camera = worldDirectionToCamera(up_home);
        return armorAxesFromRadial(radial_camera, up_camera,
                                   cfg_.armor_tilt_deg * CV_PI / 180.0);
    }

    // 输出模型：双模型权重生效后的中心/底速/yaw/ω，按 (v, ω) 外推 dt 秒后摆四块板。
    // 与 getEstimatedArmorPositions()（纯整车 EKF、不外推）的区别正是这两点：
    //   ① 用**输出口径**的状态（plate_model_weight_>0 时中心/yaw 来自板模型反推），
    //      这样画出来的模型和瞄点/选板用的是同一套几何，不会"框在我这儿、瞄在那儿"；
    //   ② 外推到"现在"，不再把观测延迟显示成模型相位滞后。
    Tracker::OutputModel Tracker::predictOutputModel(double dt) const
    {
        OutputModel out;
        if (state_ == TrackerState::LOST || !ekf_.isInit()) return out;
        const double clamped_dt = std::max(0.0, std::min(dt, 0.30));

        const Vec<3> center_now = blendedCenter();
        const Vec<3> velocity = blendedVelocity();
        const double omega = getOmega();
        const double yaw0 = blendedYawWorld();
        const double radius = cfg_.armor_radius;

        out.omega = omega;
        out.radius = radius;
        out.world_yaw = ArmorEKF::normalizeAngle(yaw0 + omega * clamped_dt);
        Vec<3> center = center_now;
        if (cfg_.model_lead_center) {
            for (int i = 0; i < 3; ++i) center(i, 0) += velocity(i, 0) * clamped_dt;
        }
        out.center = worldToCamera(center);

        for (int plate = 0; plate < 4; ++plate) {
            const double phase = out.world_yaw + ArmorEKF::angleForPlate(plate);
            Vec<3> position = center;
            position(0, 0) += radius * std::cos(phase);
            position(1, 0) += cfg_.armor_y_offsets[plate];
            position(2, 0) += radius * std::sin(phase);
            out.positions[plate] = worldToCamera(position);
        }
        out.valid = true;
        return out;
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

    // 拟合器的"车辆系→相机系"完整旋转：R_cv = G^T · R_cb(yaw,pitch)^T · B
    //   G   = camera→gimbal 手眼外参（CameraPose.camera_to_gimbal）
    //   R_cb= camera→home 旋转（相机光轴与云台 yaw/pitch 的关系，见 tracker_utils）
    //   B   = 拟合器车辆系(X前/Y右/Z上) → home 系(X右/Y下/Z前)：X_v→(0,0,1),
    //         Y_v→(1,0,0), Z_v→(0,-1,0)
    // 外参单位阵且 pitch=0 时 R_cb=I、G=I → R_cv=B，与旧行为逐位一致（仿真不受影响）。
    std::array<double, 9> Tracker::fitterRcv() const
    {
        const auto cb = tracker_detail::cameraToBaseRotation(camera_pose_.yaw, camera_pose_.pitch);
        const auto g = tracker_detail::cameraToGimbal(camera_pose_);
        const double B[3][3] = {{0.0, 1.0, 0.0}, {0.0, 0.0, -1.0}, {1.0, 0.0, 0.0}};
        // M = G^T · R_cb^T
        double M[3][3];
        for (int i = 0; i < 3; ++i)
            for (int l = 0; l < 3; ++l)
            {
                double v = 0.0;
                for (int k = 0; k < 3; ++k) v += g(k, i) * cb(l, k);
                M[i][l] = v;
            }
        std::array<double, 9> out{};
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
            {
                double v = 0.0;
                for (int l = 0; l < 3; ++l) v += M[i][l] * B[l][j];
                out[i * 3 + j] = v;
            }
        return out;
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
