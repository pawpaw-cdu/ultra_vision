// Tracker 内的局部拟合器接入（小陀螺 θ/ω 强观测）。
// 缓冲 + 低频异步拟合 + EKF 注入；拟合算法由外部注入（见 tracker.hpp）。

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>

#include "Kalman/tracker.hpp"

namespace auto_aim
{
    namespace
    {
        // 灯条在板面切向的偏移（±W/2）；与 LocalTrajectoryFitter 的 plate_width 一致。
        constexpr double k_local_fit_plate_width = 0.135;

        double wrapPi(double a)
        {
            return std::remainder(a, 2.0 * M_PI);
        }
    }  // namespace

    void Tracker::setLocalFitter(LocalFitterCallback fn)
    {
        std::lock_guard<std::mutex> lock(local_fit_mutex_);
        local_fitter_ = std::move(fn);
    }

    Tracker::~Tracker()
    {
        stopLocalFitWorker();
    }

    void Tracker::pushArmorLightBars(const Armor& armor, double timestamp, int armor_id)
    {
        if (!cfg_.local_fit.enabled) return;
        if (armor_id < 0 || armor_id > 3) return;

        std::lock_guard<std::mutex> lock(local_fit_mutex_);
        const double t = timestamp;

        // 分段弧：换板 / 断帧都开新段（一段 = 一块板连续可见的弧）。
        const double gap = local_fit_buffer_.empty() ? 0.0
                                                     : t - local_fit_buffer_.back().timestamp;
        if (armor_id != local_fit_segment_plate_ || gap > cfg_.local_fit.max_result_age) {
            local_fit_buffer_.clear();
            local_fit_segment_plate_ = armor_id;
        }
        auto push_bar = [&](const Light& bar, double offset)
        {
            LightBarObservation o;
            o.timestamp = t;
            o.u_top = bar.top.x;    o.v_top = bar.top.y;
            o.u_bot = bar.bottom.x; o.v_bot = bar.bottom.y;
            o.armor_id = armor_id;
            o.bar_offset = offset;
            o.confidence = 1.0;
            local_fit_buffer_.push_back(o);
        };
        push_bar(armor.left, +0.5 * k_local_fit_plate_width);
        push_bar(armor.right, -0.5 * k_local_fit_plate_width);

        // 段内只保留最近 window_frames 帧（每帧 2 根灯条）
        const size_t cap = static_cast<size_t>(2 * std::max(1, cfg_.local_fit.window_frames));
        if (local_fit_buffer_.size() > cap)
            local_fit_buffer_.erase(local_fit_buffer_.begin(),
                                    local_fit_buffer_.end() - static_cast<long>(cap));
    }

    bool Tracker::runLocalFitNow()
    {
        if (!cfg_.local_fit.enabled) return false;

        std::vector<LightBarObservation> window;
        LocalFitterCallback fitter;
        {
            std::lock_guard<std::mutex> lock(local_fit_mutex_);
            if (!local_fitter_ || static_cast<int>(local_fit_buffer_.size()) < cfg_.local_fit.min_bars)
                return false;
            window = local_fit_buffer_;
            fitter = local_fitter_;
        }

        LocalFitResult result;
        if (!fitter(window, result) || !result.valid) return false;

        std::lock_guard<std::mutex> lock(local_fit_mutex_);
        latest_local_fit_ = result;
        return true;
    }

    void Tracker::startLocalFitWorker()
    {
        if (!cfg_.local_fit.enabled || local_fit_running_.exchange(true)) return;
        local_fit_thread_ = std::thread([this]()
        {
            while (local_fit_running_.load())
            {
                runLocalFitNow();
                std::this_thread::sleep_for(std::chrono::milliseconds(30));
            }
        });
    }

    void Tracker::stopLocalFitWorker()
    {
        local_fit_running_.store(false);
        if (local_fit_thread_.joinable()) local_fit_thread_.join();
    }

    std::optional<LocalFitResult> Tracker::latestLocalFit() const
    {
        std::lock_guard<std::mutex> lock(local_fit_mutex_);
        return latest_local_fit_;
    }

    double Tracker::applyLatestLocalFit()
    {
        if (!cfg_.local_fit.enabled || !ekf_.isInit()) return 0.0;
        std::optional<LocalFitResult> fit;
        {
            std::lock_guard<std::mutex> lock(local_fit_mutex_);
            fit = latest_local_fit_;
        }
        if (!fit || !fit->valid) return 0.0;
        if (last_timestamp_ - fit->timestamp > cfg_.local_fit.max_result_age) return 0.0;

        // 相机系 → EKF 系：yaw 相差相机自身 yaw（相机系底盘 yaw → 世界系底盘 yaw）。
        LocalFitResult conv = *fit;
        const double cam_yaw = camera_pose_.valid ? camera_pose_.yaw : 0.0;
        conv.yaw = wrapPi(conv.yaw + cam_yaw * cfg_.local_fit.yaw_rate_sign);
        conv.omega *= cfg_.local_fit.yaw_rate_sign;

        return ekf_.updateWithLocalFit(conv, cfg_.local_fit.yaw_noise, cfg_.local_fit.omega_noise,
                                       std::numeric_limits<double>::infinity());
    }
}  // namespace auto_aim
