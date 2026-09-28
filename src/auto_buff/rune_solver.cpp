#include "auto_buff/rune_solver.hpp"

#include <cmath>
#include <iostream>

#include <opencv2/core/eigen.hpp>

#include "auto_buff/support/math.hpp"

namespace auto_aim::energy
{
    namespace
    {
        // Maps a camera-frame (optical, x right / y down / z forward) point to
        // the z-up world frame used by the estimator: X = forward, Y = left,
        // Z = up. The world frame coincides with the camera frame the rune was
        // calibrated against, which keeps yaw = 0 for a centred rune.
        const Eigen::Matrix3d kOpticalToWorld =
            (Eigen::Matrix3d() << 0.0, 0.0, 1.0,
             -1.0, 0.0, 0.0,
             0.0, -1.0, 0.0)
                .finished();

        // Camera optical frame -> optical frame of the calibration pose, given
        // the absolute gimbal yaw/pitch. Matches the simulator's
        // camera.rotation = home * Ry(-yaw) * Rx(pitch).
        Eigen::Matrix3d opticalRotation(double yaw, double pitch)
        {
            const double cy = std::cos(yaw), sy = std::sin(yaw);
            const double cp = std::cos(pitch), sp = std::sin(pitch);
            Eigen::Matrix3d R;
            R << cy, sy * sp, sy * cp,
                 0.0, cp, -sp,
                 -sy, cy * sp, cy * cp;
            return R;
        }
    } // namespace

    RuneSolver::RuneSolver(const RuneSolverConfig& config) : config_(config)
    {
        RuneOrbitFit::Config orbit_config;
        orbit_config.window_s = config.orbit_fit_window_s;
        orbit_config.min_samples = config.orbit_fit_min_samples;
        orbit_fit_ = RuneOrbitFit(orbit_config);
        camera_matrix_scaled_ = config_.camera_matrix;
        const double radius = config_.target_radius_m;
        const double half = config_.target_half_width_m;
        // Keypoint order used by the network: radially outer corner, right,
        // inner, left, then the plate centre and a point on the arm.
        object_points_ = {
            cv::Point3f(0.0f, 0.0f, static_cast<float>(radius + half)),
            cv::Point3f(0.0f, static_cast<float>(half), static_cast<float>(radius)),
            cv::Point3f(0.0f, 0.0f, static_cast<float>(radius - half)),
            cv::Point3f(0.0f, static_cast<float>(-half), static_cast<float>(radius)),
            cv::Point3f(0.0f, 0.0f, static_cast<float>(radius)),
            cv::Point3f(0.0f, 0.0f, static_cast<float>(config_.arm_point_radius_m)),
            cv::Point3f(0.0f, 0.0f, 0.0f),
        };
    }

    void RuneSolver::setCameraPose(double yaw, double pitch)
    {
        R_gimbal2world_ =
            kOpticalToWorld * opticalRotation(yaw, pitch) * config_.R_camera2gimbal.transpose();
    }

    void RuneSolver::updateImageSize(int width, int height)
    {
        if (!config_.auto_scale_intrinsics || width <= 0 || height <= 0) return;
        if (width == config_.calibration_width && height == config_.calibration_height) {
            camera_matrix_scaled_ = config_.camera_matrix;
            return;
        }
        const double scale_x =
            static_cast<double>(width) / std::max(1, config_.calibration_width);
        const double scale_y =
            static_cast<double>(height) / std::max(1, config_.calibration_height);
        camera_matrix_scaled_ = config_.camera_matrix.clone();
        camera_matrix_scaled_.at<double>(0, 0) *= scale_x;
        camera_matrix_scaled_.at<double>(1, 1) *= scale_y;
        camera_matrix_scaled_.at<double>(0, 2) *= scale_x;
        camera_matrix_scaled_.at<double>(1, 2) *= scale_y;
    }

    const cv::Mat& RuneSolver::cameraMatrix() const
    {
        return camera_matrix_scaled_.empty() ? config_.camera_matrix : camera_matrix_scaled_;
    }

    void RuneSolver::solve(std::optional<PowerRune>& rune)
    {
        if (!rune.has_value()) return;
        PowerRune& observation = rune.value();
        if (observation.target().points.size() < 4) return;

        std::vector<cv::Point2f> image_points(observation.target().points.begin(),
                                              observation.target().points.begin() + 4);
        const std::vector<cv::Point3f> object_points(object_points_.begin(),
                                                     object_points_.begin() + 4);
        // 平面四点的 IPPE 一定有两个解，它们的重投影误差几乎一样（实测都是
        // 0.0x px），但第二个解的靶面法线朝向反了、圆心能偏到隔壁扇叶上。仿真
        // 里噪声会让两个解逐帧翻转，云台因此左右大幅摆动。用"靶面法线朝向相机"
        // 这个物理约束来选解（能量机关的靶面就是朝着观察者的）。
        std::vector<cv::Mat> rvecs;
        std::vector<cv::Mat> tvecs;
        const int solutions = cv::solvePnPGeneric(
            object_points, image_points, cameraMatrix(), config_.distort_coeffs, rvecs, tvecs,
            false, cv::SOLVEPNP_IPPE);
        if (solutions <= 0) {
            std::cerr << "[RuneSolver] solvePnP failed" << std::endl;
            return;
        }
        // 两个解的重投影误差几乎一样，区分它们要用"圆心投影"：正确解的圆心
        // 反投影会落在网络给出的 R 标上（实测偏差 7~8 px = 关键点噪声量级），
        // 错误解能偏出 90 px（整整一片扇叶）。这也是选解唯一可靠的物理依据——
        // R 标是模型直接训出来的、就在画面里。
        double best_score = std::numeric_limits<double>::max();
        const cv::Point3f origin(0.0f, 0.0f, 0.0f);
        for (int index = 0; index < solutions; ++index) {
            std::vector<cv::Point2f> projected_origin;
            cv::projectPoints(std::vector<cv::Point3f>{origin},
                              rvecs[static_cast<std::size_t>(index)],
                              tvecs[static_cast<std::size_t>(index)], cameraMatrix(),
                              config_.distort_coeffs, projected_origin);
            const double center_error =
                cv::norm(projected_origin.front() - observation.r_center);
            std::vector<cv::Point2f> reprojected;
            cv::projectPoints(object_points, rvecs[static_cast<std::size_t>(index)],
                              tvecs[static_cast<std::size_t>(index)], cameraMatrix(),
                              config_.distort_coeffs, reprojected);
            double error = 0.0;
            for (std::size_t i = 0; i < reprojected.size(); ++i) {
                error += cv::norm(reprojected[i] - image_points[i]);
            }
            error /= static_cast<double>(reprojected.size());
            const double score = center_error + 0.5 * error;
            if (std::getenv("ULTRA_VISION_RUNE_DEBUG")) {
                std::cerr << "[RuneSolver] solution " << index << " center=" << center_error
                          << " px reproj=" << error << " px" << std::endl;
            }
            if (score < best_score) {
                best_score = score;
                rvecs[static_cast<std::size_t>(index)].copyTo(rvec_);
                tvecs[static_cast<std::size_t>(index)].copyTo(tvec_);
            }
        }

        // Sanity gate: the plate must be in front of the camera and the four
        // corners must reproject onto what the network detected. IPPE returns
        // the second-best solution silently otherwise, and that solution puts
        // the rune centre metres away from the truth.
        if (tvec_[2] <= 0.1) {
            observation.solved = false;
            return;
        }
        {
            std::vector<cv::Point2f> reprojected;
            cv::projectPoints(object_points, rvec_, tvec_, cameraMatrix(), config_.distort_coeffs,
                              reprojected);
            double error = 0.0;
            for (std::size_t i = 0; i < reprojected.size(); ++i) {
                error += cv::norm(reprojected[i] - image_points[i]);
            }
            error /= static_cast<double>(reprojected.size());
            if (error > max_reprojection_error_px) {
                observation.solved = false;
                return;
            }
        }

        cv::Mat rotation;
        cv::Rodrigues(rvec_, rotation);
        Eigen::Matrix3d R_buff2camera;
        cv::cv2eigen(rotation, R_buff2camera);
        Eigen::Vector3d t_buff2camera;
        cv::cv2eigen(tvec_, t_buff2camera);

        // buff -> camera -> that camera's optical frame.
        const Eigen::Vector3d xyz_in_camera = t_buff2camera;
        const Eigen::Vector3d blade_xyz_in_camera =
            R_buff2camera * Eigen::Vector3d(0.0, 0.0, config_.target_radius_m) + t_buff2camera;

        // ... -> world.
        const Eigen::Matrix3d R_buff2gimbal = config_.R_camera2gimbal * R_buff2camera;
        const Eigen::Vector3d xyz_in_gimbal =
            config_.R_camera2gimbal * xyz_in_camera + config_.t_camera2gimbal;
        const Eigen::Vector3d blade_xyz_in_gimbal =
            config_.R_camera2gimbal * blade_xyz_in_camera + config_.t_camera2gimbal;
        const Eigen::Matrix3d R_buff2world = R_gimbal2world_ * R_buff2gimbal;

        observation.xyz_in_world = R_gimbal2world_ * xyz_in_gimbal;
        observation.ypd_in_world = xyz2ypd(observation.xyz_in_world);
        observation.blade_xyz_in_world = R_gimbal2world_ * blade_xyz_in_gimbal;
        observation.blade_ypd_in_world = xyz2ypd(observation.blade_xyz_in_world);
        observation.ypr_in_world = matrixToYpr(R_buff2world);

        // ---- 相位观测（深大做法：平面内的"圆心 -> 靶心"向量角）----
        // EKF 的刀片预测是 Rz(yaw)·Rx(roll)·(0,0,R)：roll=0 时靶心在 +Z（画面
        // 12 点方向），roll 增大向 -Y（画面里顺时针）。所以把观测到的
        // "圆心 -> 靶心"单位向量投影到该基上取 atan2 即可，和 EKF 的状态定义
        // 完全一致，不需要另一套标定。
        {
            const Eigen::Vector3d radial =
                observation.blade_xyz_in_world - observation.xyz_in_world;
            if (radial.norm() > 1e-6) {
                const double yaw = observation.ypr_in_world[0];
                const Eigen::Vector3d u(0.0, 0.0, 1.0);                       // 相位 0
                const Eigen::Vector3d v(std::sin(yaw), -std::cos(yaw), 0.0);  // 相位 +90°
                const Eigen::Vector3d direction = radial.normalized();
                observation.phase_rad = std::atan2(direction.dot(v), direction.dot(u));
                observation.phase_valid = true;
            }
        }

        // Depth correction: keep the well-conditioned direction from PnP, but
        // take the range from the plate's pixel offset and its known orbital
        // radius. Without this the reported range is ~25-30% too long, the
        // predicted flight time inherits that error, and the aim leads by a
        // whole blade slot instead of hitting the lit one.
        if (config_.correct_depth_with_radius) {
            const double plate_offset_px =
                cv::norm(observation.target().center - observation.r_center);
            const double focal = cameraMatrix().at<double>(0, 0);
            // 用回转椭圆的**半长轴**代替瞬时偏移：瞬时偏移在轨道远侧被透视压短
            // （实测测距偏差只跟靶心方位有关，远侧 +20%、近侧 -1%），半长轴则
            // 是整圈的量，不带这个 foreshortening。样本不够时退回瞬时偏移。
            double radius_px = plate_offset_px;
            RuneOrbitFit::Result orbit;
            if (config_.correct_depth_with_orbit_fit) {
                const double sample_time =
                    std::chrono::duration<double>(
                        std::chrono::steady_clock::now().time_since_epoch())
                        .count();
                // 喂**相对向量**（靶心 − 符心）：云台在动，绝对像素的轨迹是
                // "轨道 + 平移"，拟合会得到一个偏大的椭圆（实测距离塌到 3.9 m）。
                // 相对量把云台运动消掉，剩下的就是轨道本身的投影形状。
                orbit_fit_.add(observation.target().center - observation.r_center,
                               sample_time);
                orbit = orbit_fit_.fit(sample_time);
                if (orbit.valid) radius_px = orbit.semi_major_px;
            }
            last_orbit_ = orbit;
            if (radius_px > config_.min_plate_offset_px && focal > 0.0) {
                const double corrected = focal * config_.target_radius_m / radius_px;
                const double scale = corrected / observation.ypd_in_world[2];
                if (scale > 0.3 && scale < 3.0) {
                    const Eigen::Vector3d blade_offset =
                        observation.blade_xyz_in_world - observation.xyz_in_world;
                    observation.xyz_in_world *= scale;
                    observation.ypd_in_world[2] = corrected;
                    // The plate keeps its offset from the centre; only the
                    // centre's range is corrected.
                    observation.blade_xyz_in_world =
                        observation.xyz_in_world + blade_offset;
                    observation.blade_ypd_in_world = xyz2ypd(observation.blade_xyz_in_world);
                }
            }
        }
        // 距离闸门：见 RuneSolverConfig::min_distance_m。误检（关键点塌缩成
        // 几像素）经深度校正会算出 20 m 以上的距离，这种观测必须丢掉。
        const double observed_distance = observation.ypd_in_world[2];
        if (!(observed_distance >= config_.min_distance_m &&
              observed_distance <= config_.max_distance_m)) {
            if (std::getenv("ULTRA_VISION_RUNE_DEBUG")) {
                std::cerr << "[RuneSolver] observation distance " << observed_distance
                          << " m out of range, dropped" << std::endl;
            }
            observation.solved = false;
            return;
        }
        observation.solved = true;
    }

    cv::Point2f RuneSolver::projectToImage(const Eigen::Vector3d& xyz_in_world) const
    {
        const Eigen::Vector3d xyz_in_camera =
            config_.R_camera2gimbal.transpose() *
            (R_gimbal2world_.transpose() * xyz_in_world - config_.t_camera2gimbal);
        // 点已在相机系：rvec/tvec 取零。写法与 reproject() 保持一致
        // （用 vector<Point3f> + Vec3d，避免 projectPoints 的 Mat 类型断言）。
        const std::vector<cv::Point3f> points{
            cv::Point3f(static_cast<float>(xyz_in_camera[0]),
                        static_cast<float>(xyz_in_camera[1]),
                        static_cast<float>(xyz_in_camera[2]))};
        const cv::Vec3d zero(0.0, 0.0, 0.0);
        std::vector<cv::Point2f> projected;
        cv::projectPoints(points, zero, zero, cameraMatrix(), config_.distort_coeffs, projected);
        return projected.empty() ? cv::Point2f(0.0f, 0.0f) : projected.front();
    }

    std::vector<cv::Point2f> RuneSolver::reproject(const Eigen::Vector3d& xyz_in_world, double yaw,
                                                   double roll) const
    {
        const Eigen::Matrix3d R_buff2world = rotationMatrix(Eigen::Vector3d(yaw, 0.0, roll));
        const Eigen::Matrix3d R_buff2camera =
            config_.R_camera2gimbal.transpose() * R_gimbal2world_.transpose() * R_buff2world;
        const Eigen::Vector3d t_buff2camera =
            config_.R_camera2gimbal.transpose() *
            (R_gimbal2world_.transpose() * xyz_in_world - config_.t_camera2gimbal);

        cv::Mat R_cv;
        cv::eigen2cv(R_buff2camera, R_cv);
        cv::Vec3d rvec;
        cv::Rodrigues(R_cv, rvec);
        const cv::Vec3d tvec(t_buff2camera[0], t_buff2camera[1], t_buff2camera[2]);

        std::vector<cv::Point2f> image_points;
        cv::projectPoints(object_points_, rvec, tvec, cameraMatrix(),
                          config_.distort_coeffs, image_points);
        return image_points;
    }
} // namespace auto_aim::energy
