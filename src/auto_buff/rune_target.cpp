#include "auto_buff/rune_target.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>

#include "auto_buff/support/math.hpp"

namespace auto_aim::energy
{
    namespace
    {
        constexpr double kSmallOmega = kPi / 3.0;      // fixed small-rune rate
        constexpr double kLargePeakOmega = 2.09;       // a + b, independent of a
        constexpr double kMaxLostUpdates = 6.0;

        int stateSize(RuneMode mode) { return mode == RuneMode::Small ? 7 : 10; }
    } // namespace

    void DirectionVoter::vote(double angle_last, double angle_now)
    {
        if (std::abs(clockwise_) > 50) return;
        if (angle_last > angle_now) {
            --clockwise_;
        } else {
            ++clockwise_;
        }
    }

    RuneTarget::RuneTarget(RuneMode mode, int max_coast_frames, double max_distance_jump_ratio,
                           double max_center_jump_deg)
        : mode_(mode),
          max_coast_frames_(max_coast_frames),
          max_distance_jump_ratio_(max_distance_jump_ratio),
          max_center_jump_rad_(max_center_jump_deg * kPi / 180.0),
          spd_fitter_(100, 0.5, 1.884, 2.000)
    {
    }

    void RuneTarget::reset()
    {
        first_in_ = true;
        unsolvable_ = true;
        lost_count_ = 0;
        coaster_frames_ = 0;
        pending_valid_ = false;
        start_time_ = -1.0;
        lasttime_ = 0.0;
        spd_ = 0.0;
    }

    Eigen::Vector3d RuneTarget::pointBuffToWorld(const Eigen::Vector3d& point_in_buff,
                                                 double phase_offset) const
    {
        if (unsolvable_) return Eigen::Vector3d::Zero();

        const Eigen::Matrix3d R_buff2world =
            rotationMatrix(Eigen::Vector3d(ekf_.x[4], 0.0, ekf_.x[5] + phase_offset));
        const double radius = ekf_.x[3];
        const double pitch = ekf_.x[2];
        const double yaw = ekf_.x[0];
        const Eigen::Vector3d center_in_world(
            radius * std::cos(pitch) * std::cos(yaw),
            radius * std::cos(pitch) * std::sin(yaw),
            radius * std::sin(pitch));
        return R_buff2world * point_in_buff + center_in_world;
    }

    void RuneTarget::getTarget(const std::optional<PowerRune>& rune, double timestamp)
    {
        blade_switched_ = false;
        if (!rune.has_value() || !rune->solved) {
            coastOrInvalidate(timestamp);
            return;
        }

        if (first_in_) {
            unsolvable_ = true;
            init(timestamp, rune.value());
            first_in_ = false;
        }

        if (lost_count_ > kMaxLostUpdates) {
            unsolvable_ = true;
            lost_count_ = 0;
            first_in_ = true;
            return;
        }
        lost_count_ = 0;

        // ---- 观测优先 + 二次确认（现场反馈："模型和估计器在抢云台"）----
        const double slot = 2.0 * kPi / 5.0;
        const double roll_raw = rune->phase_valid ? rune->phase_rad : rune->ypr_in_world[2];
        // 观测到的是"已激活的那片"时（slot_offset_valid），把它折算回**跟踪槽位**：
        const double roll_measurement =
            rune->slot_offset_valid ? roll_raw - rune->slot_offset * slot : roll_raw;
        if (!first_in_) {
            const double measured_distance = rune->ypd_in_world[2];
            const double distance_limit =
                std::max(1.5, max_distance_jump_ratio_ * std::abs(ekf_.x[3]));
            const bool distance_jump =
                std::abs(measured_distance - ekf_.x[3]) > distance_limit;
            const bool center_jump =
                std::abs(limitRad(rune->ypd_in_world[0] - ekf_.x[0])) > max_center_jump_rad_ ||
                std::abs(limitRad(rune->ypd_in_world[1] - ekf_.x[2])) > max_center_jump_rad_;
            const bool slot_changed =
                std::abs(limitRad(roll_measurement - ekf_.x[5])) > 0.35 * slot;
            // 非跟踪槽位的观测不能用来"接管/重置"估计器：它的相位是折算过的，
            if (!rune->slot_offset_valid && (distance_jump || center_jump || slot_changed)) {
                constexpr double kConfirmCenterRad = 6.0 * kPi / 180.0;
                const bool confirmed =
                    pending_valid_ &&
                    std::abs(limitRad(roll_measurement - pending_phase_)) < 0.35 * slot &&
                    std::abs(limitRad(rune->ypd_in_world[0] - pending_yaw_)) < kConfirmCenterRad &&
                    std::abs(limitRad(rune->ypd_in_world[1] - pending_pitch_)) < kConfirmCenterRad;
                if (confirmed) {
                    if (std::getenv("ULTRA_VISION_RUNE_DEBUG")) {
                        std::cerr << "[rune target] observation takes over (confirmed)"
                                  << std::endl;
                    }
                    init(timestamp, rune.value());
                    first_in_ = false;
                    unsolvable_ = false;
                    pending_valid_ = false;
                    coaster_frames_ = 0;
                    // 观测确认接管：这就是"换叶/换靶"的真实信号（相位/圆心跳变），
                    blade_switched_ = true;
                    return;
                }
                pending_phase_ = roll_measurement;
                pending_yaw_ = rune->ypd_in_world[0];
                pending_pitch_ = rune->ypd_in_world[1];
                pending_valid_ = true;
                coastOrInvalidate(timestamp);
                return;
            }
        }
        pending_valid_ = false;

        coaster_frames_ = 0;
        unsolvable_ = false;
        update(timestamp, rune.value());

        // Divergence guards: a small rune must keep |omega| near pi/3, and a
        if (mode_ == RuneMode::Small) {
            const double spd = ekf_.x[6];
            if (std::abs(spd) > kSmallOmega + kPi / 18.0 ||
                std::abs(spd) < kSmallOmega - kPi / 18.0) {
                unsolvable_ = true;
                first_in_ = true;
            }
        } else {
            if (ekf_.x[7] > 1.045 * 1.5 || ekf_.x[7] < 0.78 / 1.5 ||
                ekf_.x[8] > 2.0 * 1.5 || ekf_.x[8] < 1.884 / 1.5) {
                first_in_ = true;
            }
        }
    }

    void RuneTarget::coastOrInvalidate(double timestamp)
    {
        ++lost_count_;
        ++coaster_frames_;
        if (first_in_ || lost_count_ > max_coast_frames_) {
            unsolvable_ = true;
            return;
        }
        // Coast on the motion model: advance the state to `timestamp` and keep
        const double dt = timestamp - lasttime_;
        if (dt > 0.0) {
            // Never trust an unbounded time step: a stalled stream would
            const double step = std::min(dt, 0.5);
            // 滑行时圆心在世界上是静止的：只有相位该继续走。这里先把"圆心偏航
            ekf_.x[1] = 0.0;
            if (mode_ == RuneMode::Small) {
                predictSmall(step);
            } else {
                predictLarge(step);
            }
            lasttime_ = timestamp;
        }
    }

    void RuneTarget::init(double nowtime, const PowerRune& rune)
    {
        lasttime_ = nowtime;
        const int n = stateSize(mode_);
        x0_.resize(n);
        P0_.resize(n, n);
        A_.resize(n, n);
        Q_.resize(n, n);

        // 相位"解缠"：换片/观测接管时，观测给的是包到 (-π, π] 的相位，直接当
        double phase_init = rune.phase_valid ? rune.phase_rad : rune.ypr_in_world[2];
        // 观测到的是"已激活的那片"（不在跟踪槽位）时，先把相位折算回跟踪槽位：
        if (rune.slot_offset_valid) {
            phase_init -= rune.slot_offset * 2.0 * kPi / 5.0;
        }
        if (ekf_.x.size() > 5 && std::isfinite(ekf_.x[5])) {
            phase_init = ekf_.x[5] + limitRad(phase_init - ekf_.x[5]);
        }

        if (mode_ == RuneMode::Small) {
            x0_ << rune.ypd_in_world[0], 0.0, rune.ypd_in_world[1], rune.ypd_in_world[2],
                rune.ypr_in_world[0], phase_init, kSmallOmega * voter_.clockwise();
        } else {
            // a = 0.9125, w = 1.942 and b = 2.09 - a = 1.1775 are the middle of
            x0_ << rune.ypd_in_world[0], 0.0, rune.ypd_in_world[1], rune.ypd_in_world[2],
                rune.ypr_in_world[0], phase_init, 1.1775, 0.9125, 1.942, 0.0;
        }

        P0_.setZero();
        P0_.diagonal().head(6).setConstant(10.0);
        P0_(6, 6) = (mode_ == RuneMode::Small) ? 1e-2 : 100.0;
        if (mode_ == RuneMode::Large) {
            P0_(7, 7) = 10.0;
            P0_(8, 8) = 10.0;
            P0_(9, 9) = 400.0;
        }
        A_.setIdentity();
        Q_.setZero();

        auto add_angles = [n](const Eigen::VectorXd& a, const Eigen::VectorXd& b) {
            Eigen::VectorXd c = a + b;
            c[0] = limitRad(c[0]);
            c[2] = limitRad(c[2]);
            c[4] = limitRad(c[4]);
            c[5] = limitRad(c[5]);
            if (n >= 10) c[9] = limitRad(c[9]);
            return c;
        };
        ekf_ = ExtendedKalmanFilter(x0_, P0_, add_angles);
    }


    Eigen::Vector3d RuneTarget::pointBuffToWorldAt(const Eigen::VectorXd& x,
                                                   const Eigen::Vector3d& point_in_buff,
                                                   double phase_offset) const
    {
        const Eigen::Vector3d center(
            x[3] * std::cos(x[2]) * std::cos(x[0]),
            x[3] * std::cos(x[2]) * std::sin(x[0]),
            x[3] * std::sin(x[2]));
        const Eigen::Matrix3d R_buff2world =
            rotationMatrix(Eigen::Vector3d(x[4], 0.0, x[5] + phase_offset));
        return R_buff2world * point_in_buff + center;
    }

    void RuneTarget::updateWithPixels()
    {
        const RunePixelModel& model = pixel_model_;
        if (!model.valid()) return;

        // 物点：靶面四点（边中点，顺序与 RuneSolver::object_points_ 一致）+ R 标。
        const std::array<Eigen::Vector3d, 4> plate_points{
            Eigen::Vector3d(0.0, 0.0, model.radius + model.half_width),
            Eigen::Vector3d(0.0, model.half_width, model.radius),
            Eigen::Vector3d(0.0, 0.0, model.radius - model.half_width),
            Eigen::Vector3d(0.0, -model.half_width, model.radius)};

        const std::array<int, 5> params{0, 2, 3, 4, 5};   // R_yaw/R_pitch/R_dis/yaw/roll
        const auto subtract_pixels = [](const Eigen::VectorXd& a, const Eigen::VectorXd& b) {
            return Eigen::VectorXd(a - b);
        };
        const int state_size = static_cast<int>(ekf_.x.size());

        // 一帧内多片：每片都是同一个刚体的独立观测（含已激活片）。
        for (const RunePixelModel::Blade& blade : model.blades) {
            const double phase_offset =
                blade.slot_offset_valid ? blade.slot_offset * 2.0 * kPi / 5.0 : 0.0;
            for (int index = 0; index < 5; ++index) {
                const bool is_rmark = index == 4;
                if (is_rmark && !blade.rmark_valid) continue;
                const Eigen::Vector3d object_point =
                    is_rmark ? Eigen::Vector3d::Zero()
                             : plate_points[static_cast<std::size_t>(index)];
                const double sigma =
                    (is_rmark ? model.sigma_rmark_px : model.sigma_plate_px) * model.sigma_scale;
                const cv::Point2f observed =
                    is_rmark ? blade.rmark_uv : blade.plate_uv[static_cast<std::size_t>(index)];

                auto predict_pixels = [&](const Eigen::VectorXd& x) -> Eigen::VectorXd {
                    const Eigen::Vector3d world =
                        pointBuffToWorldAt(x, object_point, phase_offset);
                    const cv::Point2f pixel = model.project(world);
                    return (Eigen::Vector2d() << pixel.x, pixel.y).finished();
                };

                // 残差门限：槽位/关联错了会让残差很大（一整片 ≈ 60 px），
                const Eigen::VectorXd predicted = predict_pixels(ekf_.x);
                const Eigen::Vector2d observation(observed.x, observed.y);
                if ((observation - predicted).norm() > model.max_residual_px) continue;

                // 有限差分雅可比：只有 5 个相关状态分量，每帧多算 25 次投影，
                Eigen::MatrixXd H = Eigen::MatrixXd::Zero(2, state_size);
                const Eigen::VectorXd x0 = ekf_.x;
                for (const int param : params) {
                    const double step = (param == 3) ? 0.01 : 1e-3;
                    Eigen::VectorXd plus = x0;
                    plus[param] += step;
                    H.col(param) = (predict_pixels(plus) - predicted) / step;
                }
                Eigen::Matrix2d R = Eigen::Matrix2d::Zero();
                R(0, 0) = sigma * sigma;
                R(1, 1) = sigma * sigma;
                ekf_.update(observation, H, R, predict_pixels, subtract_pixels);
            }
        }

        // 距离锚：标量观测 z = PnP 距离，H 只取 R_dis 那一列（线性）。
        if (model.pnp_distance_m > 1.0) {
            Eigen::VectorXd z(1);
            z << model.pnp_distance_m;
            Eigen::MatrixXd H = Eigen::MatrixXd::Zero(1, state_size);
            H(0, 3) = 1.0;
            Eigen::MatrixXd R = Eigen::MatrixXd::Zero(1, 1);
            R(0, 0) = model.sigma_distance_m * model.sigma_distance_m;
            auto predict_distance = [](const Eigen::VectorXd& x) -> Eigen::VectorXd {
                Eigen::VectorXd out(1);
                out << x[3];
                return out;
            };
            auto subtract_scalar = [](const Eigen::VectorXd& a, const Eigen::VectorXd& b) {
                return Eigen::VectorXd(a - b);
            };
            ekf_.update(z, H, R, predict_distance, subtract_scalar);
        }
    }

    void RuneTarget::update(double nowtime, const PowerRune& rune)
    {
        const Eigen::Vector3d& rune_ypd = rune.ypd_in_world;
        const Eigen::Vector3d& rune_ypr = rune.ypr_in_world;
        const Eigen::Vector3d& blade_ypd = rune.blade_ypd_in_world;
        // 更新前的相位角速度（大符帧间限幅要用）。
        const double spd_before = ekf_.x.size() > 6 ? ekf_.x[6] : 0.0;

        // Blade changes light up a different 72-degree slot; fold the
        const double roll_raw = rune.phase_valid ? rune.phase_rad : rune_ypr[2];
        // 观测到的是**已激活的那片**时，先折算回跟踪槽位（-offset×72°），
        const double roll_measurement =
            rune.slot_offset_valid ? roll_raw - rune.slot_offset * 2.0 * kPi / 5.0 : roll_raw;
        if (std::abs(roll_measurement - ekf_.x[5]) > kPi / 12.0) {
            for (int i = -5; i <= 5; ++i) {
                const double candidate = ekf_.x[5] + i * 2.0 * kPi / 5.0;
                if (std::abs(candidate - roll_measurement) < kPi / 5.0) {
                    ekf_.x[5] += i * 2.0 * kPi / 5.0;
                    break;
                }
            }
        }

        voter_.vote(ekf_.x[5], roll_measurement);
        if (mode_ == RuneMode::Small && voter_.clockwise() * ekf_.x[6] < 0.0) {
            ekf_.x[6] *= -1.0;
        }

        const double angle_last = ekf_.x[5];
        // 同 coastOrInvalidate：圆心不随符自转移动，清零它的角速度状态。
        ekf_.x[1] = 0.0;
        if (mode_ == RuneMode::Small) {
            predictSmall(nowtime - lasttime_);
        } else {
            predictLarge(nowtime - lasttime_);
        }

        // 像素残差观测（可选）：直接吃网络给的像素，替代下面两条 3D 量测。
        const bool pixel_mode = pixel_model_.valid();
        if (pixel_mode) {
            updateWithPixels();
        } else {
        // Measurement 1: rune centre and attitude.
        Eigen::MatrixXd H1(4, stateSize(mode_));
        H1.setZero();
        H1(0, 0) = 1.0; // R_yaw
        H1(1, 2) = 1.0; // R_pitch
        H1(2, 3) = 1.0; // R_dis
        H1(3, 5) = 1.0; // roll
        Eigen::Matrix4d R1 = Eigen::Matrix4d::Zero();
        R1(0, 0) = 0.01;
        R1(1, 1) = 0.01;
        R1(2, 2) = 0.5;
        // roll（相位）这一项要放得很松：PnP 的 roll 只由靶面四点的"姿态"决定，
        R1(3, 3) = 0.1;

        auto subtract_angles = [](const Eigen::VectorXd& a, const Eigen::VectorXd& b) {
            Eigen::VectorXd c = a - b;
            c[0] = limitRad(c[0]);
            c[1] = limitRad(c[1]);
            c[3] = limitRad(c[3]);
            return c;
        };
        const Eigen::VectorXd z1 =
            (Eigen::Vector4d() << rune_ypd[0], rune_ypd[1], rune_ypd[2], roll_measurement)
                .finished();
        ekf_.update(z1, H1, R1, subtract_angles);

        // Measurement 2: the lit plate centre, predicted through the rune
        const double blade_phase_offset =
            rune.slot_offset_valid ? rune.slot_offset * 2.0 * kPi / 5.0 : 0.0;
        const Eigen::MatrixXd H2 = jacobian(blade_phase_offset);
        Eigen::Matrix3d R2 = Eigen::Matrix3d::Zero();
        // 靶心方向的观测精度被严重低估了：上面 PnP 的重投影误差实测 0.01~0.07 px，
        R2(0, 0) = 0.01;
        R2(1, 1) = 0.01;
        R2(2, 2) = 0.5;

        const double blade_radius = 0.7;
        auto predict_blade = [&](const Eigen::VectorXd& x) -> Eigen::VectorXd {
            const Eigen::Vector3d rune_point(
                x[3] * std::cos(x[2]) * std::cos(x[0]),
                x[3] * std::cos(x[2]) * std::sin(x[0]),
                x[3] * std::sin(x[2]));
            const Eigen::Matrix3d R_buff2world =
                rotationMatrix(Eigen::Vector3d(x[4], 0.0, x[5] + blade_phase_offset));
            return xyz2ypd(R_buff2world * Eigen::Vector3d(0.0, 0.0, blade_radius) + rune_point);
        };
        auto subtract_blade = [](const Eigen::VectorXd& a, const Eigen::VectorXd& b) {
            Eigen::VectorXd c = a - b;
            c[0] = limitRad(c[0]);
            c[1] = limitRad(c[1]);
            return c;
        };
        const Eigen::VectorXd z2 =
            (Eigen::Vector3d() << blade_ypd[0], blade_ypd[1], blade_ypd[2]).finished();
        ekf_.update(z2, H2, R2, predict_blade, subtract_blade);
        }   // end of 3D-measurement path

        if (mode_ == RuneMode::Large) {
            // Recover the sinusoid behind omega so the aimer can integrate it.
            if (ekf_.x[6] < kLargePeakOmega && ekf_.x[6] >= 0.0) {
                spd_fitter_.addData(nowtime, ekf_.x[6]);
            }
            spd_fitter_.fit();
            fit_spd_ = spd_fitter_.sineFunction(
                nowtime, spd_fitter_.best_result_.A, spd_fitter_.best_result_.omega,
                spd_fitter_.best_result_.phi, spd_fitter_.best_result_.C);

            // 物理限幅 + 帧间限幅：大符的相位角速度由模型 speed(t)=a·sin(wt+φ)+b
            {
                constexpr double kLargeRateSlewRadSS = 3.0; // 帧间变化率上界
                const double dt = std::max(1e-3, nowtime - lasttime_);
                const double limited = std::clamp(
                    ekf_.x[6], spd_before - kLargeRateSlewRadSS * dt,
                    spd_before + kLargeRateSlewRadSS * dt);
                ekf_.x[6] = std::clamp(limited, 0.0, kLargePeakOmega);
            }
        }

        // 小符的转速是规则给定的常量（±π/3 rad/s，深大 SmallRuneKalmanFilter
        if (mode_ == RuneMode::Small) {
            ekf_.x[6] = kSmallOmega * (voter_.clockwise() >= 0 ? 1.0 : -1.0);
        }

        spd_ = voter_.clockwise() * (ekf_.x[5] - angle_last) / (nowtime - lasttime_);
        if (std::abs(spd_) > 4.0) spd_ = 0.0;
        if (mode_ == RuneMode::Large) spd_ = fit_spd_;

        lasttime_ = nowtime;
    }

    void RuneTarget::predict(double dt)
    {
        if (mode_ == RuneMode::Small) {
            predictSmall(dt);
        } else {
            predictLarge(dt);
        }
    }

    void RuneTarget::predictSmall(double dt)
    {
        A_.setIdentity();
        A_(0, 1) = dt; // R_yaw += v * dt
        A_(5, 6) = dt; // roll  += omega * dt

        Q_.setZero();
        const double variance = 0.001;
        Q_(0, 0) = dt * dt * dt * dt / 4.0 * variance;
        Q_(0, 1) = dt * dt * dt / 2.0 * variance;
        Q_(1, 0) = Q_(0, 1);
        Q_(1, 1) = dt * dt * variance;

        auto propagate = [&](const Eigen::VectorXd& x) -> Eigen::VectorXd {
            Eigen::VectorXd prior = A_ * x;
            prior[0] = limitRad(prior[0]);
            prior[2] = limitRad(prior[2]);
            prior[4] = limitRad(prior[4]);
            prior[5] = limitRad(prior[5]);
            return prior;
        };
        ekf_.predict(A_, Q_, propagate);
    }

    void RuneTarget::predictLarge(double dt)
    {
        const double a = ekf_.x[7];
        const double w = ekf_.x[8];
        const double phi = ekf_.x[9];
        const double t = lasttime_ + dt;
        const double sign = voter_.clockwise();

        A_.setIdentity();
        A_(0, 1) = dt;
        A_(5, 6) = sign * dt;
        A_(6, 7) = std::sin(w * t + phi) - 1.0;
        A_(6, 8) = t * a * std::cos(w * t + phi);
        A_(6, 9) = a * std::cos(w * t + phi);

        Q_.setZero();
        const double variance = 0.9;
        Q_(0, 0) = dt * dt * dt * dt / 4.0 * variance;
        Q_(0, 1) = dt * dt * dt / 2.0 * variance;
        Q_(1, 0) = Q_(0, 1);
        Q_(1, 1) = dt * dt * variance;
        Q_(5, 5) = 0.09;
        Q_(6, 6) = 0.5;
        Q_(9, 9) = 1.0;

        auto propagate = [&](const Eigen::VectorXd& x) -> Eigen::VectorXd {
            Eigen::VectorXd prior = x;
            prior[0] = limitRad(prior[0] + dt * prior[1]);
            prior[2] = limitRad(prior[2]);
            prior[4] = limitRad(prior[4]);
            prior[5] = limitRad(
                prior[5] +
                sign * (-a / w * std::cos(w * t + phi) + a / w * std::cos(w * lasttime_ + phi) +
                        (kLargePeakOmega - a) * dt));
            prior[6] = a * std::sin(w * t + phi) + kLargePeakOmega - a;
            return prior;
        };
        ekf_.predict(A_, Q_, propagate);
    }

    Eigen::MatrixXd RuneTarget::jacobian(double phase_offset) const
    {
        const int n = stateSize(mode_);
        Eigen::MatrixXd H0 = Eigen::MatrixXd::Zero(5, n);
        H0(0, 0) = 1.0; // R_yaw
        H0(1, 2) = 1.0; // R_pitch
        H0(2, 3) = 1.0; // R_dis
        H0(3, 4) = 1.0; // yaw
        H0(4, 5) = 1.0; // roll

        const Eigen::Matrix3d H_ypd2xyz =
            ypd2xyzJacobian(Eigen::Vector3d(ekf_.x[0], ekf_.x[2], ekf_.x[3]));
        Eigen::MatrixXd H1 = Eigen::MatrixXd::Zero(5, 5);
        H1.block<3, 3>(0, 0) = H_ypd2xyz;
        H1(3, 3) = 1.0;
        H1(4, 4) = 1.0;

        const double yaw = ekf_.x[4];
        const double roll = ekf_.x[5] + phase_offset;
        const double cy = std::cos(yaw), sy = std::sin(yaw);
        const double cr = std::cos(roll), sr = std::sin(roll);
        const double radius = 0.7;
        Eigen::MatrixXd H2 = Eigen::MatrixXd::Zero(3, 5);
        H2 << 1.0, 0.0, 0.0, radius * cy * sr, radius * sy * cr,
            0.0, 1.0, 0.0, radius * sy * sr, -radius * cy * cr,
            0.0, 0.0, 1.0, 0.0, -radius * sr;

        // 靶心位置同样要在"观测片"的相位上求导（半径向量绕 z 转 phase_offset）。
        const Eigen::Vector3d blade_in_buff =
            rotationMatrix(Eigen::Vector3d(0.0, 0.0, phase_offset)) *
            Eigen::Vector3d(0.0, 0.0, radius);
        const Eigen::Vector3d blade_xyz = pointBuffToWorld(blade_in_buff);
        const Eigen::Matrix3d H3 = xyz2ypdJacobian(blade_xyz);
        return H3 * H2 * H1 * H0;
    }
} // namespace auto_aim::energy
