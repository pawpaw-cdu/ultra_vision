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
        push_bar(armor.left, +0.5 * k_local_fit_plate_width);
        push_bar(armor.right, -0.5 * k_local_fit_plate_width);

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
            if (cand.empty()) return false;
            std::sort(cand.begin(), cand.end(), [](const LocalFitArc* x, const LocalFitArc* y) {
                if (x->obs.size() != y->obs.size()) return x->obs.size() > y->obs.size();
                return x->t > y->t;
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
        ++local_fit_attempts_;
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

        // 默认只喂 ω（+中心），不喂 yaw——yaw 是最脆弱的量，容易把 EKF 拉坏（见 config.use_yaw）。
        if (!cfg_.local_fit.use_yaw || std::getenv("ULTRA_VISION_LOCAL_FIT_NO_YAW") != nullptr)
            conv.yaw = ekf_yaw;

        const double fit_weight = ekf_.updateWithLocalFit(
            conv, cfg_.local_fit.yaw_noise, cfg_.local_fit.omega_noise,
            std::numeric_limits<double>::infinity());

        // 慢层几何：拟合器在一个弧段上把 (R, 安装倾角 α, 板心高度 h) 一起解出来了 ——
        // 这三个量正好是"底盘模型"的慢层参数（见 common/chassis_model.hpp）。
        // 用它们在线收敛几何，比在 yaml 里写死一个 0.25（实测 0.20~0.21）靠谱得多，
        // 也是"四块板同高、矩形底盘"能被观测纠正的入口。半径/高度是几何量，
        // 改它们不动状态/协方差。
        if (cfg_.local_fit.use_fit_radius)
            updateChassisFromFit(fit->R, fit->alpha, fit->h);

        // ---- 旋转中心先验：拟合器判出的运动状态决定"中心该不该动" ----------
        // 纯旋转 → 旋转中心物理上不变 → 把车心往拟合中心拉（强）；有平移 → 只轻拉。
        // 这一步是"防止旋转中心随检测板跑飞"的关键，也是模型构造的先验来源。
        if (cfg_.local_fit.center_anchor)
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
