// Tracker 内的局部拟合器接入（小陀螺 θ/ω 强观测）。
// 缓冲 + 低频异步拟合 + EKF 注入；拟合算法由外部注入（见 tracker.hpp）。

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
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

        std::lock_guard<std::mutex> lock(local_fit_mutex_);
        const double t = timestamp;
        const double cu = 0.5 * (armor.left.center.x + armor.right.center.x);
        const double cv = 0.5 * (armor.left.center.y + armor.right.center.y);

        // 只接受"能定深度、在合理距离"的板：滤掉近处道具/杂检（无 PnP、或 z 离谱）。
        if (!camera_intrinsics_valid_ || !armor.solve_result || armor.tvec.rows != 3) return;
        const double depth = armor.tvec.at<double>(2);
        if (!(depth > cfg_.local_fit.min_depth && depth < cfg_.local_fit.max_depth)) return;

        // A3 正对门控：sep/(W·fx/depth)=cos(板相对相机偏角)，要求 > front_gate。
        // 只有正对时图像左右才对应板面切向的 ±W/2；离开正对区后左右翻号，喂进去会追镜像解。
        const double sep = armor.right.center.x - armor.left.center.x;
        const double sep_max = k_local_fit_plate_width * camera_fx_ / std::max(0.5, depth);
        const bool front = sep > cfg_.local_fit.front_gate * sep_max;

        // A1 身份：按图像板心"最近邻 + 常速预测"关联到某条已开弧（不用 plate_id 分身份）
        LocalFitArc* arc = nullptr;
        double best = 1e9;
        for (auto& a : local_fit_arcs_)
        {
            const double dt = t - a.t;
            if (dt <= 0 || dt > cfg_.local_fit.max_result_age) continue;
            // 板号提示：同一弧内板号应一致（armor_id 是几何选的板号，做身份提示足够）。
            if (armor_id >= 0 && armor_id <= 3 && a.plate_hint >= 0 && a.plate_hint != armor_id) continue;
            const double d = std::hypot(cu - (a.u + a.vu * dt), cv - (a.v + a.vv * dt));
            if (d < cfg_.local_fit.assoc_gate_px && d < best) { best = d; arc = &a; }
        }
        if (!front)
        {
            // 板转离正对区 → 关掉命中的那条弧（清空），等 worker 拟合完/丢弃，
            // 避免把两次正面弧首尾拼起来（相位跨 >90° 会出镜像解）。
            if (arc) { arc->obs.clear(); arc->plate_hint = -1; }
            return;
        }
        if (arc == nullptr)
        {
            local_fit_arcs_.push_back(LocalFitArc{});
            arc = &local_fit_arcs_.back();
            arc->plate_hint = (armor_id >= 0 && armor_id <= 3) ? armor_id : -1;
            // 两段式相对身份：记种子帧 PnP 板心 (x,z) 与时间（用位置定相位，不用病态 yaw）。
            arc->seed_cx = armor.tvec.at<double>(0);
            arc->seed_cz = armor.tvec.at<double>(2);
            arc->seed_t = t;
        }
        else
        {
            const double dt = t - arc->t;
            if (dt > 0)
            {
                const double a = 0.5;
                arc->vu = a * (cu - arc->u) / dt + (1 - a) * arc->vu;
                arc->vv = a * (cv - arc->v) / dt + (1 - a) * arc->vv;
            }
        }
        arc->u = cu; arc->v = cv; arc->t = t;
        // 用种子帧的板号（plateIdFor）作 yaw 偏移：给对了，拟合输出的 yaw 就是"底盘相位"
        // （板偏移被去掉），EKF 才能按 plate_id=0 去比。未知则按 0。
        const int pid = (arc->plate_hint >= 0 && arc->plate_hint <= 3) ? arc->plate_hint : 0;
        auto push_bar = [&](const Light& bar, double offset)
        {
            LightBarObservation o;
            o.timestamp = t;
            o.u_top = bar.top.x;    o.v_top = bar.top.y;
            o.u_bot = bar.bottom.x; o.v_bot = bar.bottom.y;
            o.armor_id = pid;
            o.bar_offset = offset;
            o.confidence = 1.0;
            arc->obs.push_back(o);
        };
        push_bar(armor.left, +0.5 * k_local_fit_plate_width);
        push_bar(armor.right, -0.5 * k_local_fit_plate_width);

        // 弧内只保留最近 window_frames 帧（每帧 2 根灯条）
        const size_t cap = static_cast<size_t>(2 * std::max(1, cfg_.local_fit.window_frames));
        if (arc->obs.size() > cap)
            arc->obs.erase(arc->obs.begin(), arc->obs.end() - static_cast<long>(cap));

        // 清理长时间不活跃的空弧
        local_fit_arcs_.erase(
            std::remove_if(local_fit_arcs_.begin(), local_fit_arcs_.end(),
                           [&](const LocalFitArc& a) {
                               return a.obs.empty() && t - a.t > cfg_.local_fit.max_result_age;
                           }),
            local_fit_arcs_.end());
    }

    bool Tracker::runLocalFitNow()
    {
        if (!cfg_.local_fit.enabled) return false;

        struct AuxArc { std::vector<LightBarObservation> obs; double cx, cz, t; };
        std::vector<LightBarObservation> window;
        std::vector<AuxArc> aux;
        int ref_hint = 0;
        LocalFitterCallback fitter;
        std::string dbg_k;
        {
            std::lock_guard<std::mutex> lock(local_fit_mutex_);
            if (!local_fitter_) return false;
            // 收集 >= min_bars 的弧，按长度降序
            std::vector<const LocalFitArc*> cand;
            for (const auto& a : local_fit_arcs_)
                if (static_cast<int>(a.obs.size()) >= cfg_.local_fit.min_bars) cand.push_back(&a);
            if (cand.empty()) return false;
            std::sort(cand.begin(), cand.end(), [](const LocalFitArc* x, const LocalFitArc* y) {
                return x->obs.size() > y->obs.size();
            });
            fitter = local_fitter_;
            window = cand.front()->obs;   // 参考弧（最长）
            ref_hint = (cand.front()->plate_hint >= 0 && cand.front()->plate_hint <= 3)
                           ? cand.front()->plate_hint : 0;
            if (cfg_.local_fit.joint_plates)
            {
                // 其余弧先暂存（相位偏移要等参考弧拟合出中心 C 后才能定）。
                for (std::size_t i = 1; i < cand.size() &&
                     static_cast<int>(aux.size()) + 1 < cfg_.local_fit.joint_max_arcs; ++i)
                    aux.push_back(AuxArc{ cand[i]->obs, cand[i]->seed_cx, cand[i]->seed_cz, cand[i]->seed_t });
            }
        }
        LocalFitResult result;
        bool ok = false;
        if (aux.empty())
        {
            ok = fitter(window, result) && result.valid;
        }
        else
        {
            // 两段式多板联合：① 先拟合参考弧得旋转中心 C 与 θ(t)=yaw+ω(t−t_end)；
            // ② 各弧用"种子帧 PnP 板心相对 C 的方位角"定相位偏移 k（用位置量，避开病态 yaw）；
            // ③ 堆叠后联合拟合。
            const std::vector<LightBarObservation> ref_window = window;
            LocalFitResult ref;
            if (fitter(ref_window, ref) && ref.valid)
            {
                const double Cx = ref.center[0], Cz = ref.center[2];
                std::vector<int> used{ref_hint};
                for (const auto& ax : aux)
                {
                    const double th_obs = std::atan2(ax.cx - Cx, ax.cz - Cz);
                    const double th_ref = ref.yaw + ref.omega * (ax.t - ref.timestamp);
                    int k = static_cast<int>(std::lround(
                        std::remainder(th_obs - th_ref, 2.0 * M_PI) / (0.5 * M_PI)));
                    k = ((k % 4) + 4) % 4;
                    const int k_abs = (ref_hint + k) % 4;   // 参考弧的绝对板号 + 相对偏移
                    if (std::find(used.begin(), used.end(), k_abs) != used.end()) continue;
                    used.push_back(k_abs);
                    dbg_k += std::to_string(k_abs);
                    for (const auto& o : ax.obs) { LightBarObservation c = o; c.armor_id = k_abs; window.push_back(c); }
                }
                ok = fitter(window, result) && result.valid;
            }
            if (!ok) { window = ref_window; ok = fitter(window, result) && result.valid; }  // 退化为单板
        }
        // 诊断/实验：每次拟合结果落 CSV（env ULTRA_VISION_LOCAL_FIT_LOG 指定路径）。
            if (const char* p = std::getenv("ULTRA_VISION_LOCAL_FIT_LOG")) {
                static std::ofstream log(p, std::ios::out);
                log << result.timestamp << ',' << window.size() << ',' << result.omega << ','
                    << result.reprojection_rmse_px << ',' << (ok ? 1 : 0) << ',' << dbg_k << '\n';
            }
        if (!ok) return false;

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

        // 板号绝对偏移不可靠（plateIdFor 在多板同现时会错）→ 拟合 yaw 可能整体差 k·90°。
        // EKF 自身 yaw 平滑、不会差 90°：把 fit.yaw 吸附到"与 EKF yaw 最近的、模 90° 同余"
        // 的那个值，消掉未知的 k·90°（只修正板偏移，不改 ω/中心）。
        const double ekf_yaw = ekf_.getYaw();
        conv.yaw = ekf_yaw + std::remainder(conv.yaw - ekf_yaw, 0.5 * M_PI);

        return ekf_.updateWithLocalFit(conv, cfg_.local_fit.yaw_noise, cfg_.local_fit.omega_noise,
                                       std::numeric_limits<double>::infinity());
    }
}  // namespace auto_aim
