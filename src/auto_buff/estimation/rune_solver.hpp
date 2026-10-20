#ifndef AUTO_AIM_ENERGY_RUNE_SOLVER_HPP
#define AUTO_AIM_ENERGY_RUNE_SOLVER_HPP

// PnP for the energy-rune target plate.
//
// Ported from sp_vision_25 `tasks/auto_buff/buff_solver.{hpp,cpp}`. The object
// geometry (plate at a 0.7 m radius, 254 mm across) comes from that project and
// matches the simulator model; it stays configurable because a real rune has to
// be measured on site.
//
// Frame convention: the estimate lives in a right-handed, z-up world frame
// (yaw around z, pitch up) so the estimator and aimer can be shared with the
// reference implementation. `setCameraPose()` converts the gimbal's absolute
// yaw/pitch into that frame.

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <optional>
#include <vector>

#include "auto_buff/common/rune_types.hpp"

namespace auto_aim::energy
{
    struct RuneSolverConfig
    {
        cv::Mat camera_matrix;
        cv::Mat distort_coeffs;

        // Resolution the intrinsics were calibrated at. When frames arrive at
        // another size (a recorded video, a different capture binning), the
        // intrinsics are scaled with it so the geometry stays self-consistent.
        int calibration_width = 640;
        int calibration_height = 480;
        bool auto_scale_intrinsics = true;

        double target_radius_m = 0.700;     // rune centre -> plate centre
        double target_half_width_m = 0.127; // plate half extent
        double arm_point_radius_m = 0.220;  // second keypoint radius
        // 观测距离的物理上下限：能量机关在 1.5~15 m 之外要么不可能、要么早已
        // 打不到。误检的关键点塌缩会算出荒谬距离；这条闸门直接判该帧不可解，
        // 交给状态机惯性滑行，避免云台被一路带偏。
        double min_distance_m = 1.5;
        double max_distance_m = 15.0;

        // Camera -> gimbal extrinsic. Identity when the camera is the gimbal.
        Eigen::Matrix3d R_camera2gimbal = Eigen::Matrix3d::Identity();
        Eigen::Vector3d t_camera2gimbal = Eigen::Vector3d::Zero();
    };

    class RuneSolver
    {
    public:
        explicit RuneSolver(const RuneSolverConfig& config);

        // Absolute gimbal angles of the current frame (simulator convention:
        // positive yaw turns right, positive pitch turns up).
        void setCameraPose(double yaw, double pitch);

        // Keeps the intrinsics consistent with the incoming frame size.
        void updateImageSize(int width, int height);

        // Fills xyz/ypd/ypr and blade_xyz/blade_ypd of the observation.
        void solve(std::optional<PowerRune>& rune);

        const Eigen::Matrix3d& R_gimbal2world() const { return R_gimbal2world_; }
        const std::vector<cv::Point3f>& objectPoints() const { return object_points_; }

        // 把一个世界系点投影回图像（用最近一次 setCameraPose 的相机位姿）。
        // 用于记录/可视化"云台瞄在画面哪个像素"——只有角度的话很难判断瞄偏没偏。
        cv::Point2f projectToImage(const Eigen::Vector3d& xyz_in_world) const;

        // Reprojects the rune model, for visualization only.
        std::vector<cv::Point2f> reproject(const Eigen::Vector3d& xyz_in_world, double yaw,
                                           double roll) const;

        // Camera pose of the most recent solve(), for overlays.
        const cv::Vec3d& lastRvec() const { return rvec_; }
        const cv::Vec3d& lastTvec() const { return tvec_; }
        double max_reprojection_error_px = 4.0;

    private:
        const cv::Mat& cameraMatrix() const;

        RuneSolverConfig config_;
        cv::Mat camera_matrix_scaled_;
        std::vector<cv::Point3f> object_points_;
        Eigen::Matrix3d R_gimbal2world_ = Eigen::Matrix3d::Identity();
        cv::Vec3d rvec_{0.0, 0.0, 0.0};
        cv::Vec3d tvec_{0.0, 0.0, 0.0};
    };
} // namespace auto_aim::energy

#endif // AUTO_AIM_ENERGY_RUNE_SOLVER_HPP
