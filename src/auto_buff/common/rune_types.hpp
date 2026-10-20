#ifndef AUTO_AIM_ENERGY_RUNE_TYPES_HPP
#define AUTO_AIM_ENERGY_RUNE_TYPES_HPP

// Energy-rune (大小能量机关) data types.
//
// Ported from sp_vision_25 `tasks/auto_buff/buff_type.{hpp,cpp}`. The model
// sees only the currently lit fan blade(s); PowerRune slots them onto the
// five 72-degree positions, decides which blade is the current target, and
// keeps the geometry in the z-up world frame produced by RuneSolver.

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <optional>
#include <stdexcept>
#include <vector>

namespace auto_aim::energy
{
    enum class RuneMode { Small, Large };

    enum FanBladeType { Target, Unlit, Lit };

    struct FanBlade
    {
        cv::Point2f center;
        std::vector<cv::Point2f> points; // four plate corners, then plate center
        // 深大五点模型直接给出 R 标（机关旋转中心）的图像坐标，比早先
        // "靶心 + 臂点外推" 稳得多；只有在该点可信时才置位。
        cv::Point2f rune_center{0.0f, 0.0f};
        bool has_rune_center = false;
        double angle = 0.0;
        double width = 0.0;
        double height = 0.0;
        FanBladeType type = Unlit;

        FanBlade() = default;

        FanBlade(const std::vector<cv::Point2f>& keypoints, cv::Point2f keypoints_center,
                 FanBladeType blade_type);

        explicit FanBlade(FanBladeType blade_type);
    };

    struct PowerRune
    {
        cv::Point2f r_center;           // rune center, image frame
        std::vector<FanBlade> fanblades; // five slots, starting at the target
        int light_num = 0;              // number of lit blades this frame
        // Set by RuneSolver once the plate pose passed its sanity checks. An
        // unsolved observation must never reach the estimator: at long range a
        // flipped PnP solution reports the rune centre behind the camera and
        // the aimer would swing the gimbal there.
        bool solved = false;
        // 相位观测（EKF 的 roll 状态）：平面内"圆心 -> 靶心"方向相对 12 点方向的
        // 夹角，沿用深大 PhaseMotionEstimator 的定义。比 PnP 的 roll 稳得多
        // （靶心方向测量精度 ~0.5°，而 6 m 处 28 px 的靶面让 roll 有 ±5~8° 噪声）。
        double phase_rad = 0.0;
        bool phase_valid = false;

        Eigen::Vector3d xyz_in_world{0.0, 0.0, 0.0};  // rune center, m
        Eigen::Vector3d ypr_in_world{0.0, 0.0, 0.0};  // rune pose, rad
        Eigen::Vector3d ypd_in_world{0.0, 0.0, 0.0};  // rune center, spherical

        Eigen::Vector3d blade_xyz_in_world{0.0, 0.0, 0.0}; // target blade centre, m
        Eigen::Vector3d blade_ypd_in_world{0.0, 0.0, 0.0}; // target blade centre, spherical

        // ---- "观测到的这片不是我们要打的那片"时的修正（现场提示的用法）--------
        // 网络只看得到图案：已激活片是 class 1/2，未激活片是 class 0。当我们只
        // 能观测到**已激活**的那片时，它相对"跟踪槽位"在机关平面里偏了
        // slot_offset × 72°。估计器要的是"跟踪槽位那片"的观测，所以：
        //   * 相位测量按 -slot_offset×72° 折算回跟踪槽位；
        //   * 靶心位置的预测也按 +slot_offset×72° 去正对这片观测。
        // 不做这个折算的话，估计器会把相位拉到观测到的那片（瞄点整跳一片，
        // 实测 92 帧 >2° 跳变里有 10 帧落在这类观测帧上）。
        int slot_offset = 0;            // 观测片比跟踪槽位超前几个槽位（0~4）
        bool slot_offset_valid = false; // 该帧的观测是否来自"非跟踪槽位"的片

        PowerRune() = default;
        PowerRune(std::vector<FanBlade>& blades, cv::Point2f center,
                  std::optional<PowerRune> last_powerrune);

        FanBlade& target() { return fanblades.front(); }
        const FanBlade& target() const { return fanblades.front(); }

        bool is_unsolve() const { return unsolvable_; }

    private:
        bool unsolvable_ = false;

        double atan_angle(cv::Point2f point) const; // [0, 2pi)
    };
} // namespace auto_aim::energy

#endif // AUTO_AIM_ENERGY_RUNE_TYPES_HPP
