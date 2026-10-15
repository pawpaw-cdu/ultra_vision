// Tracker 内的局部拟合器接入（小陀螺 θ/ω 强观测）。
// 缓冲 + 低频异步拟合 + EKF 注入；拟合算法由外部注入（见 tracker.hpp）。

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <limits>

#include <opencv2/calib3d.hpp>

#include "Kalman/tracker.hpp"
#include "Kalman/tracker_utils.hpp"
#include "estimation/local_fit_gate.hpp"

namespace auto_aim
{
    namespace
    {
        // 灯条在板面切向的偏移（±W/2）；与 LocalTrajectoryFitter 的 plate_width 一致。
        // 板宽改由 yaml 给（local_fit.plate_width）——原来这里硬编码 0.135，
        // 把拟合器 yaml 里的 plate_width 完全架空（见 tracker.hpp 的同名注释）。

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
        const double plate_width = cfg_.local_fit.plate_width;
        const double sep_max = plate_width * camera_fx_ / std::max(0.5, depth);
        const bool front = sep > cfg_.local_fit.front_gate * sep_max;

        // A1 身份：按图像板心"最近邻 + 常速预测"关联到某条已开弧。
        // **板号（plateIdFor）只当偏好，不当硬条件**：
        //   · 板号来自模型 yaw，会被 PnP 平面解抖动带着跳；当硬条件时一跳就每帧新建弧，
        //     弧永远攒不到 min_bars（实车 log: push=4280 却 fit=4/5）。
        //   · 但完全不用板号也不行：同帧两块板在图像里接近时会把两块板串进一条弧，
        //     弧里混了两块板的数据 → 拟合残差变大、有效解比例掉（仿真实测 100%→24%）。
        // 所以：**优先接板号一致的弧；没有才退到最近的弧**。
        LocalFitArc* arc = nullptr;
        LocalFitArc* fallback = nullptr;
        double best = 1e9;
        double best_fallback = 1e9;
        for (auto& a : local_fit_arcs_)
        {
            const double dt = t - a.t;
            if (dt <= 0 || dt > cfg_.local_fit.max_result_age) continue;
            const double d = std::hypot(cu - (a.u + a.vu * dt), cv - (a.v + a.vv * dt));
            if (d >= cfg_.local_fit.assoc_gate_px) continue;
            const bool hint_match = !(armor_id >= 0 && armor_id <= 3 && a.plate_hint >= 0 &&
                                      a.plate_hint != armor_id);
            if (hint_match) {
                if (d < best) { best = d; arc = &a; }
            } else if (d < best_fallback) {
                best_fallback = d;
                fallback = &a;
            }
        }
        if (arc == nullptr) arc = fallback;
        if (!front)
        {
            // 板离开正对区：**只暂停，不清空**。
            // 真机上板在门控阈值附近会来回抖，一帧不满足就把整条弧清掉 ⇒ 弧永远攒不够
            // min_bars ⇒ 拟合器几乎不触发（实车 log: fit=2/5 就是这个原因）。
            // "两次正面弧首尾拼起来"由下面的 gap 检查挡住（距上次累计太久才清）。
            return;
        }
        // 距上次累计太久（板转过一圈又回来 / 中间长时间被挡）→ 清空重来：
        // θ(t) 线性模型跨过 >90° 相位会出镜像解。
        const double arc_gap_timeout = std::max(0.25, 1.5 * cfg_.local_fit.max_result_age);
        if (arc != nullptr && (t - arc->t) > arc_gap_timeout)
        {
            arc->obs.clear();
            arc->plate_hint = -1;
        }
        if (arc == nullptr)
        {
            ++local_fit_new_arcs_;
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
            ++local_fit_links_;
            // 只在弧还很新时刷新板号提示（避免把两块板串起来后身份被改来改去）。
            if (arc->obs.size() <= 2 && armor_id >= 0 && armor_id <= 3)
                arc->plate_hint = armor_id;
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
            // 拟合器按**针孔**重投影，喂进来的端点必须先去畸变（实车强畸变不补偿会让重投影
            // 系统性偏差、离光心越远越大；仿真 dist=0 时这步等价于不变）。
            if (camera_dist_valid_ && camera_intrinsics_valid_)
            {
                const cv::Mat K = (cv::Mat_<double>(3, 3) << camera_fx_, 0.0, camera_cx_,
                                   0.0, camera_fy_, camera_cy_, 0.0, 0.0, 1.0);
                const std::vector<cv::Point2f> in{ bar.top, bar.bottom };
                std::vector<cv::Point2f> out;
                cv::undistortPoints(in, out, K, camera_dist_, cv::noArray(), K);
                if (out.size() == 2)
                {
                    o.u_top = out[0].x; o.v_top = out[0].y;
                    o.u_bot = out[1].x; o.v_bot = out[1].y;
                }
            }
            o.armor_id = pid;
            o.bar_offset = offset;
            o.confidence = 1.0;
            // 诊断：落"推给拟合器的原始观测"（拟合器真正看到的东西，和观测 CSV 里
            // tracker.getLastObservedArmor() 会重复/滞后不同）。
            if (const char* p = std::getenv("ULTRA_VISION_LOCAL_FIT_PUSH")) {
                static std::ofstream dump(p, std::ios::out);
                dump << static_cast<long long>(o.timestamp * 1e6) << ',' << o.armor_id << ','
                     << o.bar_offset << ',' << o.u_top << ',' << o.v_top << ',' << o.u_bot << ','
                     << o.v_bot << '\n';
            }
            arc->obs.push_back(o);
            ++local_fit_pushed_;
        };
        push_bar(armor.left, +0.5 * plate_width);
        push_bar(armor.right, -0.5 * plate_width);

        // 弧内只保留最近 window_frames 帧（每帧 2 根灯条）
        last_push_time_ = t;
        const size_t cap = static_cast<size_t>(2 * std::max(1, cfg_.local_fit.window_frames));
        if (arc->obs.size() > cap)
            arc->obs.erase(arc->obs.begin(), arc->obs.end() - static_cast<long>(cap));

        // 清理长时间不活跃的弧：**不管里面还有没有 obs** 都要删。
        // 暂停语义下（一帧丢门控不清空）旧弧会一直留着，不清理就会堆成几百条，
        // 而且 worker 会挑中它们去拟合几十秒前的数据。
        const double idle_timeout = std::max(1.0, 4.0 * cfg_.local_fit.max_result_age);
        local_fit_arcs_.erase(
            std::remove_if(local_fit_arcs_.begin(), local_fit_arcs_.end(),
                           [&](const LocalFitArc& a) { return t - a.t > idle_timeout; }),
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
            // 收集 >= min_bars 的弧，并且**必须是当前还活着的弧**：
            // 距最后一次累计不超过 max_result_age —— 否则拟合的是几十秒前的旧弧，
            // 结果会被 applyLatestLocalFit 的老化门限丢掉（实车 log: arcs=176、apply=2
            // 就是挑中旧弧、结果全废）。并列时先按"最新"，再按长度。
            std::vector<const LocalFitArc*> cand;
            for (const auto& a : local_fit_arcs_) {
                if (static_cast<int>(a.obs.size()) < cfg_.local_fit.min_bars) continue;
                if (last_push_time_ - a.t > cfg_.local_fit.max_result_age) continue;
                cand.push_back(&a);
            }
            if (cand.empty()) { ++local_fit_cand_empty_; return false; }
            std::sort(cand.begin(), cand.end(), [](const LocalFitArc* x, const LocalFitArc* y) {
                if (x->obs.size() != y->obs.size()) return x->obs.size() > y->obs.size();
                return x->t > y->t;
            });
            for (const auto& a : local_fit_arcs_)
                local_fit_obs_max_ = std::max(local_fit_obs_max_,
                                              static_cast<int>(a.obs.size()));
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
        ++local_fit_attempts_;
        const auto fit_begin = std::chrono::steady_clock::now();
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
                    << result.reprojection_rmse_px << ',' << (ok ? 1 : 0) << ',' << dbg_k << ','
                    << result.R << '\n';
            }
        local_fit_last_ms_ =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() -
                                                     fit_begin).count();
        local_fit_last_rmse_ = result.reprojection_rmse_px;
        local_fit_last_R_ = result.R;
        local_fit_last_bars_ = static_cast<int>(window.size());
        if (!ok) return false;
        ++local_fit_valids_;

        std::lock_guard<std::mutex> lock(local_fit_mutex_);
        latest_local_fit_ = result;
        ++local_fit_applied_;
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
        ++local_fit_apply_calls_;
        if (!fit || !fit->valid) { ++local_fit_apply_no_fit_; return 0.0; }
        const double age = last_timestamp_ - fit->timestamp;
        local_fit_apply_age_ms_ = age * 1000.0;
        // 快层（ω/yaw）必须新鲜；慢层（R/α/h/中心）允许"几秒前"的结果 ——
        // 拟合在低功耗机器上要几百 ms，一刀切会把慢层全丢掉（实车：geometry=0）。
        const bool fresh = age <= cfg_.local_fit.max_result_age;
        const bool slow_ok = age <= cfg_.local_fit.max_slow_result_age;
        if (!fresh && !slow_ok) { ++local_fit_apply_stale_; return 0.0; }

        // 相机系 → EKF 系：yaw 相差相机自身 yaw（相机系底盘 yaw → 世界系底盘 yaw）。
        LocalFitResult conv = *fit;
        const double cam_yaw = camera_pose_.valid ? camera_pose_.yaw : 0.0;
        conv.yaw = wrapPi(conv.yaw + cam_yaw * cfg_.local_fit.yaw_rate_sign);
        conv.omega *= cfg_.local_fit.yaw_rate_sign;

        // ---- 延迟量测：把拟合 yaw 用**它自己的 ω** 外推到当前时刻 -----------------
        // NUC 上一次弧段拟合要 ~0.8~1s，发布时 age≈1s > max_result_age(0.2s)，
        // 于是原来那句"fresh 才注入"直接把它跳过 —— 这正是"开了 use_yaw 却毫无变化"
        // 的原因（实车 model3 vs model4 的 trusted_fit 一模一样：38.9 vs 40.4px）。
        // yaw/ω 是同一个弧段解，用 ω 外推到 now 再用即可（标准延迟量测处理），
        // 外推误差随 age 增长 → yaw 噪声按 age 放大。
        double yaw_noise_eff = cfg_.local_fit.yaw_noise;
        if (!fresh && slow_ok) {
            conv.yaw = wrapPi(conv.yaw + conv.omega * age);
            yaw_noise_eff *= 1.0 + age / std::max(0.05, cfg_.local_fit.max_result_age);
        }

        // ---- 相位代表元：先试"精确注入"，不可信才退回"模 90° 吸附" ----------------
        // 2026-10-11 起 LocalFitResult::yaw 是**底盘 yaw**，且拟合器与 EKF 用同一张
        // 相位表（见 local_trajectory_fitter.hpp 的 plate_yaw_offsets 注释），所以
        // 正常情况它和 EKF yaw 直接可比 —— 精确注入保留了绝对相位，逐板几何下也成立。
        // 但参考板号（plateIdFor → 几何选板，读 EKF 状态）本身可能错：一旦错 k 号，
        // 拟合 yaw 就整体差 k·90°。所以残差超门限时不信它，退回"与 EKF yaw 最近的、
        // 模 90° 同余"的那个值（只在四块板对称的假设下成立的安全退路）。
        // 注意这一步要在**延迟外推之后**做，否则比较的是两个不同时刻的 yaw。
        const double ekf_yaw = ekf_.getYaw();
        {
            const double gate = cfg_.local_fit.ref_yaw_gate_deg * M_PI / 180.0;
            const bool ref_ok = fit->ref_armor_id >= 0 && fit->ref_armor_id < 4;
            const double residual = ArmorEKF::normalizeAngle(conv.yaw - ekf_yaw);
            local_fit_yaw_residual_ = residual;   // 诊断：见 localFitYawResidual()
            if (!(gate > 0.0) || !ref_ok || std::abs(residual) > gate) {
                conv.yaw = ekf_yaw + std::remainder(conv.yaw - ekf_yaw, 0.5 * M_PI);
                ++local_fit_yaw_snaps_;
            } else {
                ++local_fit_yaw_exact_;
            }
        }

        // ---- 拟合 yaw 的常值相位偏置在线学习 ------------------------------------
        // 实测（model4 vs model5）：直接注入拟合 yaw 后，慢转工况从 0.3px 劣化到 30px
        // —— 拟合器自带 ≈8~12° 的常值偏差（它自己的几何/约定），而 EKF 的滞后 ∝ω、
        // 整圈平均≈0。把"拟合 yaw − EKF yaw"（模 90° 内）慢低通当地标，注入前减掉，
        // 就只取拟合器"无滞后"的那部分。
        {
            const double diff = std::remainder(conv.yaw - ekf_yaw, 0.5 * M_PI);
            const double limit = cfg_.local_fit.yaw_bias_limit_deg * M_PI / 180.0;
            if (limit > 0.0 && std::abs(diff) < limit) {
                if (!fit_yaw_bias_valid_) {
                    fit_yaw_bias_ = diff;      // 首次直接建立
                    fit_yaw_bias_valid_ = true;
                } else {
                    const double a = std::clamp(cfg_.local_fit.yaw_bias_alpha, 0.001, 0.5);
                    fit_yaw_bias_ = (1.0 - a) * fit_yaw_bias_ + a * diff;
                }
            }
            conv.yaw = wrapPi(conv.yaw - fit_yaw_bias_);
        }

        // 默认只喂 ω（+中心），不喂 yaw（见 config.use_yaw；2026-10-10 起默认开）。
        // 并**按 |ω| 混合**（见 LocalFitConfig::yaw_use_omega_* 注释）：
        // 静态/慢转时 EKF 自己的 yaw 更准（0.3px），高速时拟合 yaw 无滞后更准。
        double yaw_weight = 1.0;
        if (cfg_.local_fit.use_yaw && cfg_.local_fit.yaw_use_omega_full >
                                          cfg_.local_fit.yaw_use_omega_min) {
            const double omega_abs = std::abs(ekf_.getOmega());
            yaw_weight = std::clamp(
                (omega_abs - cfg_.local_fit.yaw_use_omega_min) /
                    (cfg_.local_fit.yaw_use_omega_full - cfg_.local_fit.yaw_use_omega_min),
                0.0, 1.0);
        }
        // ---- yaw 通道的质量门（2026-10-12）--------------------------------------
        // 实测（同一份实车检测流，对照整车 EKF 的 yaw，模 90°）：
        //   rmse <3px → σ_ψ=0.57° ｜ 3~8px → 9.7° ｜ 8~16px → 22.6°；ω 各档都稳。
        // 而在加这道门之前，自旋时 |ω|≥2.5 就给满权重 → 那一坨 68% 的坏 yaw 直接进 EKF。
        // ω 通道保持宽松（localFitOmegaUsable）：它是"无滞后"的那一路，也是我们在
        // 自旋段真正要用的量。
        LocalFitGateConfig gate_cfg;
        gate_cfg.yaw_max_rmse_px = cfg_.local_fit.yaw_max_rmse_px;
        gate_cfg.yaw_min_phase_span_deg = cfg_.local_fit.yaw_min_phase_span_deg;
        gate_cfg.omega_max_rmse_px = cfg_.local_fit.omega_max_rmse_px;
        const LocalFitYawVerdict yaw_verdict = evaluateLocalFitYaw(*fit, gate_cfg);
        local_fit_yaw_verdict_ = static_cast<int>(yaw_verdict);
        local_fit_phase_span_deg_ = fit->phase_span_rad * 180.0 / M_PI;
        if (yaw_verdict != LocalFitYawVerdict::ok) {
            yaw_weight = 0.0;
            ++local_fit_yaw_gated_;
        }

        if (!cfg_.local_fit.use_yaw || std::getenv("ULTRA_VISION_LOCAL_FIT_NO_YAW") != nullptr) {
            conv.yaw = ekf_yaw;
            yaw_weight = 0.0;
        } else if (yaw_weight < 1.0) {
            conv.yaw = wrapPi(ekf_yaw + yaw_weight * std::remainder(conv.yaw - ekf_yaw,
                                                                   0.5 * M_PI));
            // 权重越小 = 这条 yaw 观测越不可信 → 噪声按 1/w² 放大；w→0 时等于不喂。
            yaw_noise_eff /= std::max(0.05, yaw_weight * yaw_weight);
        }
        if (yaw_weight <= 0.0) conv.yaw = ekf_yaw;

        // ω 通道同样过质量门：拟合没贴住观测时，连角速度一起丢掉（实测见 gate 头文件）。
        const bool omega_ok = localFitOmegaUsable(*fit, gate_cfg);
        if (!omega_ok) ++local_fit_omega_gated_;
        const double fit_weight = (fresh || slow_ok) && omega_ok
            ? ekf_.updateWithLocalFit(conv, yaw_noise_eff, cfg_.local_fit.omega_noise,
                                      std::numeric_limits<double>::infinity())
            : 0.0;

        // ---- 慢层几何写回（R / 安装倾角 α / 板心高度 h）------------------------
        // 拟合器在一个弧段上把这三个量一起解出来了，它们正是底盘模型的慢层参数
        // （见 common/chassis_model.hpp）。**2026-10-11 之前这段调用被删掉了**：
        // updateChassisFromFit 零调用点、local_fit_apply_geometry_ 无自增 →
        // use_fit_radius 变成死配置（而 yaml 里它一直是 true）、状态行的 geometry= 恒为 0。
        // 见 git 历史里的 docs/local_fit_tracker_responsibility_audit.md 审计项 F。
        if (cfg_.local_fit.use_fit_radius && slow_ok)
        {
            updateChassisFromFit(fit->R, fit->alpha, fit->h);
            ++local_fit_apply_geometry_;
        }

        // ---- 旋转中心先验：拟合器判出的运动状态决定"中心该不该动" ----------
        // 纯旋转 → 旋转中心物理上不变 → 把车心往拟合中心拉（强）；有平移 → 只轻拉。
        // 这一步是"防止旋转中心随检测板跑飞"的关键，也是模型构造的先验来源。
        if (cfg_.local_fit.center_anchor && slow_ok)
        {
            const double speed = std::sqrt(fit->drift[0] * fit->drift[0] +
                                           fit->drift[1] * fit->drift[1] +
                                           fit->drift[2] * fit->drift[2]);
            const ChassisMotion motion = classifyChassisMotion(
                fit->omega, speed, cfg_.local_fit.center_anchor_spin_min,
                cfg_.local_fit.center_anchor_move_max);
            last_motion_ = motion;   // 诊断：状态行会显示
            Vec<3> camera_center;
            camera_center(0, 0) = fit->center[0];
            camera_center(1, 0) = fit->center[1];
            camera_center(2, 0) = fit->center[2];
            const Vec<3> fit_center_home = camera_pose_.valid
                ? tracker_detail::pointToWorld(camera_pose_, camera_center)
                : camera_center;
            const Vec<3> current = ekf_.getCenter();
            const double error = std::sqrt(std::pow(fit_center_home(0, 0) - current(0, 0), 2) +
                                           std::pow(fit_center_home(1, 0) - current(1, 0), 2) +
                                           std::pow(fit_center_home(2, 0) - current(2, 0), 2));
            const double gain = chassisCenterAnchorGain(
                motion, cfg_.local_fit.center_anchor_gain_rotating,
                cfg_.local_fit.center_anchor_gain_other);
            if (error < cfg_.local_fit.center_anchor_max_error && gain > 0.0)
            {
                Vec<3> delta;
                for (int i = 0; i < 3; ++i)
                    delta(i, 0) = gain * (fit_center_home(i, 0) - current(i, 0));
                ekf_.shiftCenter(delta);
            }
        }

        // 可选：把拟合出的**旋转中心**也当观测（also_update_center，默认关）。
        // 单板工况下模型中心只能靠"钉在观测板上"得到，擦边/正面观测的 PnP 位置误差
        // （实测 ±0.15 m）会 1:1 进模型中心 —— 自旋时整台车的重投影框被单块板拖着跑，
        // 就是现场看到的"绿色框对得上、模型跑飞"。拟合中心是整段弧的最小二乘解，
        // 抖动小一个量级，正好补这一路（拟合器输出的是**相机系**中心）。
        double center_weight = 0.0;
        if (cfg_.local_fit.also_update_center)
        {
            Vec<3> camera_point;
            camera_point(0, 0) = fit->center[0];
            camera_point(1, 0) = fit->center[1];
            camera_point(2, 0) = fit->center[2];
            // 云台姿态已知 → 转到 home 系；未知（仿真里 tracker 不喂姿态）时
            // EKF 本来就工作在相机系（toWorld 原样返回），直接用即可。
            const Vec<3> observation = camera_pose_.valid
                ? tracker_detail::pointToWorld(camera_pose_, camera_point)
                : camera_point;
            center_weight = ekf_.updateCenter(observation, cfg_.local_fit.center_noise,
                                              cfg_.y_noise_mult,
                                              std::numeric_limits<double>::infinity());
        }
        return std::max(fit_weight, center_weight);
    }
}  // namespace auto_aim
