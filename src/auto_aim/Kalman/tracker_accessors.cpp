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
        const auto state = ekf_.getState();
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
        const auto state = ekf_.getState();
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

    void Tracker::syncGeometryToEkf()
    {
        ekf_.setPlateDistance(geometry_.plate_distance);
        std::array<double, 4> offsets{{0.0, 0.0, 0.0, 0.0}};
        for (std::size_t i = 0; i < offsets.size(); ++i)
            offsets[i] = -geometry_.plate_height[i];
        ekf_.setArmorYOffsets(offsets);
    }

    void Tracker::syncGeometryFromConfig()
    {
        // 慢层几何的初值全部来自配置（此后只由拟合器慢层/在线慢收敛修改）：
        for (std::size_t i = 0; i < geometry_.plate_distance.size(); ++i) {
            // 逐板距离配了就用它（真车矩形底盘）
            const double configured = cfg_.plate_distance[i];
            geometry_.plate_distance[i] =
                configured > 0.05 ? configured : cfg_.armor_radius;
        }
        geometry_.plate_large = cfg_.plate_large;
        for (auto& h : geometry_.plate_height) h = 0.0;
        geometry_.armor_pitch = cfg_.armor_tilt_deg * CV_PI / 180.0;
        geometry_.small_half_width = 0.5 * cfg_.armor_small_width;
        geometry_.large_half_width = 0.5 * cfg_.armor_large_width;
        geometry_.half_height = 0.5 * cfg_.armor_height;
        syncGeometryToEkf();
    }

    void Tracker::setChassisGeometry(const ChassisGeometry& geometry)
    {
        geometry_ = geometry;
        if (geometry_.meanDistance() > 0.05 && geometry_.meanDistance() < 1.0) {
            cfg_.armor_radius = geometry_.meanDistance();
            ekf_.setArmorRadius(cfg_.armor_radius);
        }
        syncGeometryToEkf();
        cfg_.armor_tilt_deg = geometry_.armor_pitch * 180.0 / CV_PI;
    }

    void Tracker::updateChassisFromFit(double radius, double armor_pitch_rad, double height_up)
    {
        const double alpha = std::clamp(cfg_.local_fit.radius_alpha, 0.01, 1.0);
        const double max_step = std::max(1e-4, cfg_.local_fit.radius_max_step);
        bool changed = false;
        if (std::isfinite(radius) && radius > 0.10 && radius < 0.50) {
            const double mean = geometry_.meanDistance();
            const double step = std::clamp(radius - mean, -max_step, max_step);
            const double delta = alpha * step;
            for (auto& d : geometry_.plate_distance) d += delta;
            cfg_.armor_radius = mean + delta;
            ekf_.setArmorRadius(cfg_.armor_radius);
            armor_radius_online_ = cfg_.armor_radius;
            changed = true;
        }
        if (std::isfinite(armor_pitch_rad) && std::abs(armor_pitch_rad) < 1.0) {
            const double step = std::clamp(armor_pitch_rad - geometry_.armor_pitch,
                                           -0.02, 0.02);
            geometry_.armor_pitch += alpha * step;
            cfg_.armor_tilt_deg = geometry_.armor_pitch * 180.0 / CV_PI;
            changed = true;
        }
        // 板心高度：拟合器的 h 是"板心在车辆系里相对旋转中心的高度"，向上为正，
        if (std::isfinite(height_up) && std::abs(height_up) < 0.5) {
            const double mean = 0.25 * (geometry_.plate_height[0] + geometry_.plate_height[1] +
                                        geometry_.plate_height[2] + geometry_.plate_height[3]);
            const double step = std::clamp(height_up - mean, -0.02, 0.02);
            const double target = mean + alpha * step;
            for (auto& h : geometry_.plate_height) h = target;
            changed = true;
        }
        if (changed) syncGeometryToEkf();
    }

    // 把滤波器状态里的 l/h（前后组半径差/高度差）叠加到慢层几何上
    ChassisGeometry Tracker::effectiveChassisGeometry() const
    {
        ChassisGeometry geometry = geometry_;
        if (ekf_.isInit()) {
            const double dl = ekf_.groupRadiusDelta();
            const double dh = ekf_.groupHeightDelta();
            for (int plate : {1, 3}) {
                geometry.plate_distance[plate] += dl;
                geometry.plate_height[plate] += dh;
            }
        }
        return geometry;
    }

    ArmorPose Tracker::armorPose(int plate) const
    {
        Vec<3> center = ekf_.getCenter();
        if (plate_model_weight_ > 0.0) center = blendedCenter();
        const double yaw = blendedYawWorld();
        return chassisArmorPose(cv::Vec3d(center(0, 0), center(1, 0), center(2, 0)), yaw,
                                effectiveChassisGeometry(), plate);
    }

    bool Tracker::solveChassisFromPlates(const std::vector<std::pair<int, Vec<3>>>& plates,
                                         Vec<3>& center_out, double& yaw_out,
                                         double& residual_out) const
    {
        if (plates.size() < 2) return false;
        // 水平偏移（x,z）与它对 yaw 的导数
        const auto offset = [&](int plate, double psi, double* dpsi) {
            const double phase = psi + ArmorEKF::angleForPlate(plate);
            const double r = geometry_.distance(plate);
            if (dpsi != nullptr) {
                dpsi[0] = -r * std::sin(phase);
                dpsi[1] = r * std::cos(phase);
            }
            return std::array<double, 2>{{ r * std::cos(phase), r * std::sin(phase) }};
        };

        double yaw = ekf_.isInit() ? ekf_.getYaw() : 0.0;
        const int ref = plates.front().first;
        const double px_ref = plates.front().second(0, 0);
        const double pz_ref = plates.front().second(2, 0);
        for (int iter = 0; iter < 8; ++iter) {
            double jtj = 0.0;      // Σ JᵀJ（信息量：退化时为 0）
            double jtr = 0.0;      // Σ Jᵀr
            double d_ref[2];
            const auto o_ref = offset(ref, yaw, d_ref);
            for (std::size_t i = 1; i < plates.size(); ++i) {
                const int plate = plates[i].first;
                // 与参考板相反（Δφ≈180°）时两块板的"方向"共线 → yaw 不可观，跳过这一对
                const double dphi = ArmorEKF::normalizeAngle(
                    ArmorEKF::angleForPlate(plate) - ArmorEKF::angleForPlate(ref));
                if (std::abs(std::abs(dphi) - M_PI) < 0.35) continue;
                double d_i[2];
                const auto o_i = offset(plate, yaw, d_i);
                const double rx = (plates[i].second(0, 0) - px_ref) - (o_i[0] - o_ref[0]);
                const double rz = (plates[i].second(2, 0) - pz_ref) - (o_i[1] - o_ref[1]);
                const double jx = d_i[0] - d_ref[0];
                const double jz = d_i[1] - d_ref[1];
                jtj += jx * jx + jz * jz;
                jtr += jx * rx + jz * rz;
            }
            if (jtj < 1e-4) return false;         // 信息不足（板对退化）
            const double step = std::clamp(jtr / jtj, -0.5, 0.5);
            yaw += step;
            if (std::abs(step) < 1e-5) break;
        }

        // 车心 = 各板反推的平均（y 用观测板高 + 板高偏置反推）
        Vec<3> sum;
        for (const auto& entry : plates) {
            const auto o = offset(entry.first, yaw, nullptr);
            sum(0, 0) += entry.second(0, 0) - o[0];
            sum(2, 0) += entry.second(2, 0) - o[1];
            sum(1, 0) += entry.second(1, 0) + geometry_.height(entry.first);
        }
        const double inv = 1.0 / static_cast<double>(plates.size());
        for (int i = 0; i < 3; ++i) sum(i, 0) *= inv;

        // 残差（各板模型位置与观测位置的水平距离，均方根）
        double sq = 0.0;
        for (const auto& entry : plates) {
            const auto o = offset(entry.first, yaw, nullptr);
            const double dx = entry.second(0, 0) - (sum(0, 0) + o[0]);
            const double dz = entry.second(2, 0) - (sum(2, 0) + o[1]);
            sq += dx * dx + dz * dz;
        }
        residual_out = std::sqrt(sq * inv);
        center_out = sum;
        yaw_out = ArmorEKF::normalizeAngle(yaw);
        return true;
    }

    Tracker::ModelSnapshot Tracker::modelAt(double dt, const CameraPose& render_pose) const
    {
        ModelSnapshot out;
        if (state_ == TrackerState::LOST || !ekf_.isInit()) return out;
        double clamped_dt = std::max(0.0, std::min(dt, 0.30));
        // 未收敛（刚入靶/刚重捕：ω 还不可信）或 ω 明显是野值时**不外推** 
        if (!watchdogStats().converged || !std::isfinite(getOmega()) ||
            std::abs(getOmega()) > 20.0) {
            clamped_dt = 0.0;
        }

        Vec<3> center = blendedCenter();
        const Vec<3> velocity = blendedVelocity();
        if (cfg_.model_lead_center) {
            for (int i = 0; i < 3; ++i) center(i, 0) += velocity(i, 0) * clamped_dt;
        }
        out.world_yaw = ArmorEKF::normalizeAngle(blendedYawWorld() + getOmega() * clamped_dt);
        out.omega = getOmega();
        // 模型几何 = 慢层几何 + 滤波器状态里的 l/h（见 effectiveChassisGeometry）
        const ChassisGeometry geometry = effectiveChassisGeometry();
        out.geometry = geometry;

        const cv::Vec3d world_center(center(0, 0), center(1, 0), center(2, 0));
        const std::array<ArmorPose, 4> world_poses =
            chassisArmorPoses(world_center, out.world_yaw, geometry);

        // 世界系 → render 帧相机系：位置用 pointToCamera，方向用旋转矩阵。
        CameraPose pose = render_pose;
        if (!pose.valid) pose = camera_pose_; 
        const Matrix<3, 3> world_to_camera = tracker_detail::cameraToWorldRotation(pose).transpose();
        cv::Matx33d rotation;
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) rotation(r, c) = world_to_camera(r, c);
        }
        const Vec<3> camera_center = tracker_detail::pointToCamera(pose, center);
        const cv::Vec3d camera_pos(camera_center(0, 0), camera_center(1, 0), camera_center(2, 0));
        out.center_camera = camera_pos;

        // 板心：世界系板心逐点变换（注意不是"车心变换 + 再加偏移"，两者等价但前者不易错）。
        for (int plate = 0; plate < 4; ++plate) {
            ArmorPose world_pose = world_poses[static_cast<std::size_t>(plate)];
            Vec<3> world;
            world(0, 0) = world_pose.center[0];
            world(1, 0) = world_pose.center[1];
            world(2, 0) = world_pose.center[2];
            const Vec<3> camera = tracker_detail::pointToCamera(pose, world);
            ArmorPose camera_pose_out;
            camera_pose_out.center = cv::Vec3d(camera(0, 0), camera(1, 0), camera(2, 0));
            camera_pose_out.axes.normal = rotation * world_pose.axes.normal;
            camera_pose_out.axes.tangent = rotation * world_pose.axes.tangent;
            camera_pose_out.axes.bar = rotation * world_pose.axes.bar;
            out.poses[static_cast<std::size_t>(plate)] = camera_pose_out;
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

    void Tracker::updateRotationRate(const Armor& armor, double timestamp, bool yaw_valid,
                                     int plate_id)
    {
        if (!yaw_valid) return;
        const auto radial_angle = armorRadialAngle(armor);
        if (!radial_angle) return;
        // **底盘 yaw** = 灯条径向角 − 该板相位偏移。换板时径向角跳 ±90°，
        const double chassis_yaw = ArmorEKF::normalizeAngle(
            *radial_angle - ArmorEKF::angleForPlate(
                (plate_id >= 0 && plate_id <= 3) ? plate_id : 0));
        if (!rotation_rate_estimator_.update(chassis_yaw, timestamp)) {
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
        const auto state = ekf_.getState();
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
