#ifndef AUTO_AIM_ENERGY_RUNE_TARGET_HPP
#define AUTO_AIM_ENERGY_RUNE_TARGET_HPP

// Energy-rune state estimator.
//
// Ported from sp_vision_25 `tasks/auto_buff/buff_target.{hpp,cpp}`.
//
// State (z-up world frame from RuneSolver):
//   small (7): [R_yaw, v_R_yaw, R_pitch, R_dis, yaw, roll, spd]
//   large (10): the same plus [a, w, phi] for omega(t) = a*sin(w*t+phi) + 2.09 - a
// The rune centre and the lit plate centre are both used as measurements, so
// the estimate keeps the whole mechanism (centre, attitude, phase and rate)
// even while individual blades light up and go dark.

#include <Eigen/Dense>
#include <array>
#include <cmath>
#include <functional>
#include <optional>
#include <string>

#include "auto_buff/rune_types.hpp"
#include "auto_buff/support/math.hpp"
#include "auto_buff/support/ekf.hpp"
#include "auto_buff/support/sine_fitter.hpp"

namespace auto_aim::energy
{
    /// @brief 像素残差观测：把网络给的**像素**直接喂进滤波器（观测函数=重投影），
    ///        省掉"PnP 先解 3D、再当 3D 量测"的两级最小二乘。
    ///
    /// 为什么：PnP 只对 4 个点解一次最小二乘、丢掉了像素噪声的各向异性，还要靠
    /// "R 标投影"在 IPPE 两个解之间做离散选择（错了圆心能甩出 90 px）；像素残差
    /// 只有一个连续代价，权重就是实测像素方差（R 标 0.4~0.7 px、四点 ~1 px）。
    ///
    /// 物点顺序与 RuneSolver::object_points_ 一致（靶面四点为**边中点**，到靶心
    /// 距离 = half_width；R 标在原点）。
    struct RunePixelModel
    {
        /// 世界系点 → 像素（由调用方注入求解器的投影，保持解算器/估计器解耦）。
        std::function<cv::Point2f(const Eigen::Vector3d&)> project;
        double radius = 0.700;        // 符心 → 靶心
        double half_width = 0.145;    // 靶面"边中点"到靶心（标定值，见 docs §14）
        /// 一片观测：靶面四点 + R 标 + 它相对**跟踪槽位**的偏移。
        /// 一帧可以有多片（含 class 1/2 的已激活片）：刚体是同一个，每多一片
        /// 就多一组几何约束，正是"已激活片的数据用来修正精度"。
        struct Blade
        {
            std::array<cv::Point2f, 4> plate_uv{};
            cv::Point2f rmark_uv{0.0f, 0.0f};
            bool rmark_valid = false;
            int slot_offset = 0;
            bool slot_offset_valid = false;
            int class_id = 0;
        };
        std::vector<Blade> blades;
        double sigma_plate_px = 1.0;  // 四点像素噪声（实测 ~1 px）
        /// R 标的像素噪声。注意：k2 的**随机**噪声只有 0.4~0.7 px，但它有**系统
        /// 偏移**（实测把它按 0.6 px 信时，稳态距离带符号偏差 −0.169 m = −2.7%；
        /// 按 3.0 px 信时偏差 −0.032 m、|误差| 中位 0.369→0.189 m）。
        /// σ 应当覆盖"系统+随机"的总误差，否则滤波器会跟着系统偏移走。
        double sigma_rmark_px = 3.0;
        /// σ 放大系数（对角近似相关项）。实测把它放到 4.0 几乎不改变 P(3,3)
        /// （0.057 → 0.064 m）：P 被**过程噪声 Q** 压着，不是被 R 压着
        /// （见 docs §19 的一致性实验）。所以保持 1.0，不做无依据的放大。
        double sigma_scale = 1.0;
        /// 单点残差门限：超过就先不更新（防止槽位误关联把状态带偏）。
        double max_residual_px = 8.0;
        /// 距离锚（可选）：标定后的 PnP 距离比"纯像素"更准（实测状态中位 +1.6% vs
        /// +2.4%），所以像素管方向/相位、距离再补一条标量观测。<=0 表示不锚。
        double pnp_distance_m = -1.0;
        double sigma_distance_m = 0.25;

        bool valid() const { return static_cast<bool>(project) && !blades.empty(); }
    };

    // Votes on the rotation direction so the angle channel never wraps the
    // wrong way (the rune's direction is fixed per team).
    class DirectionVoter
    {
    public:
        void vote(double angle_last, double angle_now);
        int clockwise() const { return clockwise_ > 0 ? 1 : -1; }

    private:
        int clockwise_ = 0;
    };

    class RuneTarget
    {
    public:
        // `max_coast_frames` keeps the estimate (and therefore the aim) alive
        // through short detection dropouts: the filter is propagated with its
        // own motion model instead of being invalidated by a single missed
        // frame. The detector misses frames regularly at long range, and
        // resetting on each of them restarted the phase estimate constantly.
        explicit RuneTarget(RuneMode mode, int max_coast_frames = 12,
                            double max_distance_jump_ratio = 0.35,
                            double max_center_jump_deg = 20.0);
        ~RuneTarget() = default;

        void getTarget(const std::optional<PowerRune>& rune, double timestamp);
        /// @brief 注入本帧的像素观测（不设或 project 为空 ⇒ 退回原来的 3D 量测）。
        void setPixelModel(const RunePixelModel& model) { pixel_model_ = model; }
        void clearPixelModel() { pixel_model_ = RunePixelModel{}; }
        /// @brief 按当前观测**重建**估计器（参考实现确认换片后就是这么做的：
        ///        把目标整个切到新片，而不是让滤波器慢慢漂过去）。
        void rebuildFrom(const PowerRune& rune, double timestamp) { init(timestamp, rune); }
        // Advances the estimate without a measurement. Only meaningful on a
        // copy: like the reference implementation, it mutates the state.
        void predict(double dt);

        bool isUnsolvable() const { return unsolvable_; }
        // 本帧是否发生了"换叶/换靶"（观测被二次确认后接管、估计器重建）。
        // 火控用它判断"是否正在换叶"，比"指令角帧间差超过 N 度"可靠得多：
        // 大符点亮的靶心 33 ms 内的**真实**运动只有约 0.4°，而抖动可以到 4~16°，
        // 用角度阈值会把抖动误判成换叶（实测 22% 的帧对被误判，导致云台控制被
        // 关掉一半时间、开火被抑制）。
        bool bladeSwitched() const { return blade_switched_; }
        const Eigen::VectorXd& ekfX() const { return ekf_.x; }
        double spd() const { return spd_; }
        /// @brief 滤波器自己声称的距离不确定度 σ_d = sqrt(P(3,3))（一致性检查用：
        ///        若实际误差远大于它，说明 R 写得过于乐观，观测维度再多也只是"假准"）。
        double distanceSigma() const
        {
            return ekf_.P.rows() > 3 ? std::sqrt(std::max(0.0, ekf_.P(3, 3))) : 0.0;
        }
        double phaseSigma() const
        {
            return ekf_.P.rows() > 5 ? std::sqrt(std::max(0.0, ekf_.P(5, 5))) : 0.0;
        }

        // Point in the rune's own frame (x = 0 plane, +z radially outward).
        /// @param phase_offset 相对跟踪片的相位偏移（弧度）：瞄/算"往前 k 个槽位"那片用。
        Eigen::Vector3d pointBuffToWorld(const Eigen::Vector3d& point_in_buff,
                                         double phase_offset = 0.0) const;

        RuneMode mode() const { return mode_; }
        void reset();

    private:
        void init(double nowtime, const PowerRune& rune);
        void update(double nowtime, const PowerRune& rune);
        void coastOrInvalidate(double timestamp);
        void predictSmall(double dt);
        void predictLarge(double dt);
        /// @brief 靶心位置测量（measurement 2）的雅可比。
        ///        @param phase_offset 观测片相对跟踪槽位的相位偏移（弧度，默认 0）；
        ///               观测到已激活片时要按 +offset 求导，否则线性化点错一整片。
        Eigen::MatrixXd jacobian(double phase_offset = 0.0) const;
        /// @brief 状态 x 下，物体系点 p_obj（可带相位偏移）在世界系的位置。
        Eigen::Vector3d pointBuffToWorldAt(const Eigen::VectorXd& x,
                                           const Eigen::Vector3d& point_in_buff,
                                           double phase_offset) const;
        /// @brief 用像素残差更新一次（观测函数=重投影，雅可比有限差分）。
        void updateWithPixels();
        // 连续滑行（没有观测）了多少帧；超过它就把下一个观测当作"重新捕获"，
        // 直接重置估计器（观测优先，见 getTarget 里的说明）。
        int reacquire_lost_frames() const { return std::max(2, max_coast_frames_ / 2); }

        RuneMode mode_;
        int max_coast_frames_ = 12;
        double max_distance_jump_ratio_ = 0.35;
        double max_center_jump_rad_ = 20.0 * kPi / 180.0;
        Eigen::VectorXd x0_;
        Eigen::MatrixXd P0_;
        Eigen::MatrixXd A_;
        Eigen::MatrixXd Q_;
        ExtendedKalmanFilter ekf_;
        RunePixelModel pixel_model_;   // 本帧像素观测（空 ⇒ 走原来的 3D 量测）
        DirectionVoter voter_;
        RansacSineFitter spd_fitter_;
        double fit_spd_ = 0.0;
        double lasttime_ = 0.0;
        double start_time_ = -1.0;
        int lost_count_ = 0;
        int coaster_frames_ = 0;
        // 可疑观测（换片/大跳变）的二次确认缓存。
        bool pending_valid_ = false;
        double pending_phase_ = 0.0;
        double pending_yaw_ = 0.0;
        double pending_pitch_ = 0.0;
        double spd_ = 0.0;
        bool first_in_ = true;
        bool unsolvable_ = true;
        bool blade_switched_ = false;
    };
} // namespace auto_aim::energy

#endif // AUTO_AIM_ENERGY_RUNE_TARGET_HPP
