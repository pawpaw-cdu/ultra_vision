// Energy-rune (大小能量机关) 跟踪与瞄准入口。

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/opencv.hpp>

#include "common/standard_clock.hpp"
#include "control/gimbal_controller.hpp"
#include "control/aim_signal_filter.hpp"
#include "auto_buff/control/rune_aimer.hpp"
#include "auto_buff/control/rune_blade_selector.hpp"
#include "auto_buff/control/rune_aim_bridge.hpp"
#include "auto_buff/control/rune_fire_gate.hpp"
#include "auto_buff/app/rune_input.hpp"
#include "auto_buff/app/rune_runner.hpp"
#include "auto_buff/control/rune_aim_fallback.hpp"
#include "auto_buff/control/rune_round_guard.hpp"
#include "auto_buff/estimation/rune_slot_lattice.hpp"
#include "auto_buff/config/rune_config.hpp"
#include "auto_buff/detection/rune_detector.hpp"
#include "auto_buff/estimation/rune_solver.hpp"
#include "auto_buff/estimation/rune_target.hpp"
#include "auto_buff/common/rune_types.hpp"
#include "auto_buff/support/math.hpp"

#ifndef ULTRA_VISION_CONFIG_DIR
#define ULTRA_VISION_CONFIG_DIR "configs"
#endif

namespace
{
    struct Telemetry
    {
        int hits = 0;
        int scored = 0;
        int activations = 0;
        double first_activation_s = -1.0;
    };

    // Pulls one `key=value` field out of a simulator telemetry line.
    std::string telemetryField(const std::string& line, const std::string& key)
    {
        const std::string needle = key + "=";
        const auto position = line.find(needle);
        if (position == std::string::npos) return {};
        const auto start = position + needle.size();
        const auto end = line.find(' ', start);
        return line.substr(start, end == std::string::npos ? std::string::npos : end - start);
    }

    void consumeTelemetry(const std::vector<std::string>& lines, const std::string& rune_mode,
                          Telemetry& telemetry, bool& scored_hit_this_frame)
    {
        scored_hit_this_frame = false;
        for (const std::string& line : lines) {
            if (telemetryField(line, "mode") != rune_mode) continue;
            const std::string event = telemetryField(line, "event");
            if (event == "hit") {
                if (telemetryField(line, "accurate") == "1") ++telemetry.hits;
                if (telemetryField(line, "scored") == "1") {
                    ++telemetry.scored;
                    scored_hit_this_frame = true;
                }
            }
            if (event == "activated" || telemetryField(line, "change") == "1") {
                ++telemetry.activations;
                if (telemetry.first_activation_s < 0.0) {
                    telemetry.first_activation_s = std::atof(telemetryField(line, "t").c_str());
                }
            }
        }
    }

    const char* statusName(auto_aim::energy::RuneTrackStatus status)
    {
        switch (status) {
        case auto_aim::energy::RuneTrackStatus::Track: return "TRACK";
        case auto_aim::energy::RuneTrackStatus::TemporaryLost: return "TEMP_LOST";
        case auto_aim::energy::RuneTrackStatus::Lost: return "LOST";
        }
        return "UNKNOWN";
    }

} // namespace

int auto_aim::energy::runRune(const auto_aim::energy::RuneConfig& config,
                              const std::string& config_dir, const std::string& video_path)
{
    const bool video_mode = !video_path.empty();
    const bool fire_enabled = std::getenv("ULTRA_VISION_DISABLE_FIRE") == nullptr;

    std::unique_ptr<auto_aim::energy::RuneDetector> detector;
    try {
        detector = std::make_unique<auto_aim::energy::RuneDetector>(config.detector);
    } catch (const std::exception& error) {
        std::cerr << "Failed to create the rune detector: " << error.what() << std::endl;
        return -1;
    }
    auto_aim::energy::RuneSolver solver(config.solver);
    auto_aim::energy::RuneAimer aimer(config.aimer);
    auto_aim::energy::RuneTarget target(config.mode, config.max_coast_frames,
                                        config.max_distance_jump_ratio,
                                        config.max_center_jump_deg);
    // ---- 瞄准解算放在控制线程（参照 sp_vision 的 multithread/commandgener）----
    const bool aim_in_control =
        (std::getenv("ULTRA_VISION_RUNE_AIM_IN_CONTROL") == nullptr ||
         std::string(std::getenv("ULTRA_VISION_RUNE_AIM_IN_CONTROL")) != "0");
    // 输入层：仿真器/视频统一成一个接口，下面只管"取帧、取遥测、发指令"。
    auto_aim::energy::RuneInputSource input;
    if (!input.open(config, config_dir, video_path)) return -1;

    // 观测隔离实验：保持物理相机不动，避免云台闭环把目标在画面里来回拖动。
    enum class GimbalFreezeMode { None, Now, AfterActivation };
    GimbalFreezeMode freeze_mode = GimbalFreezeMode::None;
    if (const char* freeze = std::getenv("ULTRA_VISION_RUNE_GIMBAL_FREEZE")) {
        const std::string value(freeze);
        if (value == "activation") {
            freeze_mode = GimbalFreezeMode::AfterActivation;
        } else if (value != "0") {
            freeze_mode = GimbalFreezeMode::Now;
        }
    }
    bool gimbal_frozen = freeze_mode == GimbalFreezeMode::Now;
    double frozen_yaw = 0.0;
    double frozen_pitch = 0.0;

    // 云台由 100 Hz 轨迹生成器驱动：视觉只发布目标角与目标角速度。
    auto_aim::GimbalController gimbal_controller(
        config.gimbal,
        [&input, &gimbal_frozen, &frozen_yaw, &frozen_pitch](double yaw, double pitch) {
            const double sent_yaw = gimbal_frozen ? frozen_yaw : yaw;
            const double sent_pitch = gimbal_frozen ? frozen_pitch : pitch;
            return input.sendGimbal(sent_yaw, sent_pitch);
        },
        config.gimbal_command_rate_hz);
    auto_aim::AimSignalFilter aim_filter(config.aim_filter);
    // 瞄准桥（帧循环 <-> 控制线程）：帧循环只发布状态，控制线程按 100 Hz 重算瞄准。
    auto_aim::energy::RuneAimBridge aim_bridge(config, aimer);
    if (aim_in_control) {
        std::cout << "Energy rune: 瞄准解算在控制线程（100 Hz，to_now 预测）" << std::endl;
        aim_bridge.install(gimbal_controller, aim_filter, fire_enabled, {});
    }

    double commanded_yaw = 0.0;
    double commanded_pitch = 0.0;
    double smoothed_lead_latency_s = -1.0;   // 帧到瞄准延迟的 EMA（提前量用）
    if (const char* pose = std::getenv("ULTRA_VISION_RUNE_POSE")) {
        double yaw_deg = 0.0;
        double pitch_deg = 0.0;
        if (std::sscanf(pose, "%lf,%lf", &yaw_deg, &pitch_deg) == 2) {
            commanded_yaw = yaw_deg * CV_PI / 180.0;
            commanded_pitch = pitch_deg * CV_PI / 180.0;
        }
    }
    if (gimbal_frozen) {
        frozen_yaw = commanded_yaw;
        frozen_pitch = commanded_pitch;
    }

    cv::Mat frame;
    auto_aim::energy::RuneInputSource::Frame rune_input_frame;
    int frames = 0;
    int missing_frames = 0;
    int last_status = -1;
    // Scoring counters, filled from the simulator's ground-truth channel.
    int shots_fired = 0;
    // 目标角阶跃检测（清零前馈用）：上一帧发布的目标角。
    double previous_target_yaw = 0.0;
    bool have_previous_target = false;
    int shots_held_for_window = 0;
    bool scored_hit_this_frame = false;
    // ---- 一轮（点亮窗口）的记账 + 命中保持 + 禁火窗口 ----------------------
    auto_aim::energy::RuneRoundGuard round_guard(
            auto_aim::energy::RuneRoundGuard::Config{
            config.round_end_frames, config.blade_slot_jump_px, config.fire_start_delay_s,
            config.round_window_s, config.post_hit_switch_phase_rad, 2.0 * CV_PI / 5.0,
            config.post_hit_hold_max_s, false});
    // 上一次成功解算出观测的本地时间（用于"观测寿命"判定，见 RuneAimer）。
    double last_solved_time = -1.0;
    // 观测层只提供几何/相位；开火资格由独立状态机按计划槽位判定。
    auto_aim::energy::RuneFireGate fire_gate(
        auto_aim::energy::RuneFireGate::Config{config.detector.engageable_latch_s});
    auto_aim::energy::RuneBladeSelector blade_selector;
    // 丢目标时的云台策略（停在最后已知符心 / 兜底回标定位姿），见其头文件。
    auto_aim::energy::RuneAimFallback fallback(
        auto_aim::energy::RuneAimFallback::Config{config.park_on_center_max_age_s,
                                                  config.recenter_after_frames});
    // ---- 扇叶槽位格点（1..5 号位的记账）------------------------------------
    auto_aim::energy::RuneSlotLattice::Config lattice_config;
    lattice_config.activate_votes = config.slot_activate_votes;
    lattice_config.reset_silence_s = config.slot_lattice_reset_silence_s;
    auto_aim::energy::RuneSlotLattice slot_lattice(
        lattice_config, std::getenv("ULTRA_VISION_RUNE_DEBUG") != nullptr);
    int fired_slot_id = -1;                  // 命中瞬间开火的那个槽位（喂回格点）
    cv::Point2f last_fired_center{0.0f, 0.0f};
    Telemetry telemetry_stats;
    const bool rune_is_small =
        config.mode == auto_aim::energy::RuneMode::Small;
    const char* rune_mode_name = rune_is_small ? "small" : "large";
    const double test_seconds = std::getenv("ULTRA_VISION_RUNE_TEST_SECONDS")
                                    ? std::atof(std::getenv("ULTRA_VISION_RUNE_TEST_SECONDS"))
                                    : 0.0;
    const auto run_start = std::chrono::steady_clock::now();
    auto fps_start = std::chrono::steady_clock::now();
    // 仿真帧序号：用来量"我们跳过了多少帧"（是仿真给得慢，还是我们处理不过来）。
    uint64_t last_source_sequence = 0;
    long long dropped_since_print = 0;

    while (true) {
        const double wall_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - run_start).count();
        if (test_seconds > 0.0 && wall_seconds >= test_seconds) {
            std::cout << "Reached ULTRA_VISION_RUNE_TEST_SECONDS=" << test_seconds
                      << " s, exiting" << std::endl;
            break;
        }

        if (!video_mode) {
            consumeTelemetry(input.pollTelemetry(), rune_mode_name, telemetry_stats,
                             scored_hit_this_frame);
            if (freeze_mode == GimbalFreezeMode::AfterActivation && !gimbal_frozen &&
                telemetry_stats.scored > 0) {
                gimbal_frozen = true;
                frozen_yaw = commanded_yaw;
                frozen_pitch = commanded_pitch;
                std::cout << "Energy rune: first scored hit observed; gimbal frozen at yaw="
                          << frozen_yaw * 180.0 / CV_PI
                          << " deg pitch=" << frozen_pitch * 180.0 / CV_PI
                          << " deg for observation audit" << std::endl;
            }
            if (scored_hit_this_frame) {
                round_guard.noteScoredHit(auto_aim::StandardClock::nowSeconds());
            }
        }

        rune_input_frame = {};
        if (!input.readFrame(rune_input_frame)) {
            if (video_mode) break;
            ++missing_frames;
            if (missing_frames == 1 || missing_frames % 60 == 0) {
                std::cerr << "Waiting for simulator frames..." << std::endl;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }
        if (rune_input_frame.sequence > 0 && last_source_sequence > 0 &&
            rune_input_frame.sequence > last_source_sequence + 1) {
            dropped_since_print += static_cast<long long>(rune_input_frame.sequence -
                                                          last_source_sequence - 1);
        }
        if (rune_input_frame.sequence > 0) last_source_sequence = rune_input_frame.sequence;
        frame = rune_input_frame.bgr;
        const uint64_t frame_local_timestamp_us = rune_input_frame.local_timestamp_us;
        const uint64_t frame_source_timestamp_us = rune_input_frame.source_timestamp_us;
        missing_frames = 0;

        // 时间基准：优先用仿真给的曝光时刻（与本地同一 epoch，见 StandardClock），
        const uint64_t frame_time_us = auto_aim::StandardClock::selectFrameTimestampUs(
            frame_local_timestamp_us, frame_source_timestamp_us);
        const double timestamp = auto_aim::StandardClock::secondsFromUs(frame_time_us);

        solver.updateImageSize(frame.cols, frame.rows);

        // 用轨迹生成器的历史快照取"这一帧曝光时刻"的云台姿态，而不是我们自己
        if (!video_mode && !gimbal_frozen) {
            const auto pose = gimbal_controller.snapshotAt(frame_time_us);
            if (pose.valid) {
                commanded_yaw = pose.yaw;
                commanded_pitch = pose.pitch;
            }
        }

        // 用轨迹生成器的历史快照取"这一帧曝光时刻"的云台姿态，而不是我们
        solver.setCameraPose(commanded_yaw, commanded_pitch);

        std::optional<auto_aim::energy::PowerRune> rune =
            config.detector.multi_candidate ? detector->detect(frame, timestamp)
                                            : detector->detectBest(frame, timestamp);
        // 采样时刻放在**推理之后**：`now` 要参与瞄准的提前量计算
        const double now = auto_aim::StandardClock::nowSeconds();
        solver.solve(rune);
        // ---- 扇叶 ID：把当帧观测喂给槽位格点，取出 1..5 号位的记账 ----------
        {
            auto_aim::energy::RuneSlotLattice::Frame lattice_frame;
            lattice_frame.solved =
                rune.has_value() && rune->solved && target.ekfX().size() > 5;
            if (lattice_frame.solved) {
                lattice_frame.roll = target.ekfX()[5];
                lattice_frame.target_class = detector->targetClass();
                lattice_frame.hub = rune->r_center;
                lattice_frame.target_center = rune->target().center;
                // 过滤前的**全部**候选（含已激活片）：格点用它们的相位差把
                const auto& candidates = detector->lastCandidates();
                lattice_frame.candidates.reserve(candidates.size());
                for (const auto& candidate : candidates) {
                    lattice_frame.candidates.emplace_back(candidate.class_id, candidate.center);
                }
                // 5 个槽位在图像里的预测位置：k=0 是当前瞄准的那片，k=1..4 依次
                const double slot_step = 2.0 * CV_PI / 5.0;
                const Eigen::Vector3d rune_center_in_world =
                    target.pointBuffToWorld(Eigen::Vector3d::Zero());
                const double yaw = target.ekfX()[4];
                for (int slot = 0; slot < 5; ++slot) {
                    const auto points =
                        solver.reproject(rune_center_in_world, yaw,
                                         lattice_frame.roll + slot * slot_step);
                    if (points.size() < 5) break;
                    lattice_frame.slot_centers[static_cast<std::size_t>(slot)] = points[4];
                    lattice_frame.slot_centers_valid = slot == 4;
                }
            }
            // 命中反馈只作补充：命中瞬间开火的那一槽位也标记（万一该帧没检到它）。
            lattice_frame.scored_hit = scored_hit_this_frame;
            lattice_frame.fired_slot = fired_slot_id;
            // 本轮是否还有点亮的扇叶 —— **不能**直接用 round_guard 的 blade_lit：
            lattice_frame.round_active =
                (rune.has_value() && rune->solved) ||
                (last_solved_time > 0.0 &&
                 now - last_solved_time < config.slot_lattice_round_end_s);
            slot_lattice.update(lattice_frame, now);
        }
        const auto slot_snapshot = slot_lattice.snapshot();
        const auto& slot_activation = slot_snapshot.activated;
        int planned_slot = slot_snapshot.slot_id;
        // 开火资格在观测/估计器更新之后由 RuneFireGate 单独计算。
        bool aimed_slot_activated =
            planned_slot >= 0 &&
            slot_activation[static_cast<std::size_t>(planned_slot)];
        // 观测到的是**已激活**的那片（class 1/2）时告诉估计器：它相对跟踪槽位偏了
        const int observed_class = detector->targetClass();
        if (rune.has_value() && rune->solved &&
            slot_snapshot.observed_offset_valid && observed_class > 0) {
            rune->slot_offset = slot_snapshot.observed_offset;
            rune->slot_offset_valid = true;
        }
        // 把网络给的像素直接注入估计器：同一刚体的每一片都提供一组几何约束。
        if (rune.has_value() && rune->solved) {
            auto_aim::energy::RunePixelModel pixel_model;
            pixel_model.project = [&solver](const Eigen::Vector3d& world) {
                return solver.projectToImage(world);
            };
            if (const char* rmark_sigma = std::getenv("ULTRA_VISION_RUNE_RMARK_SIGMA_PX")) {
                pixel_model.sigma_rmark_px = std::atof(rmark_sigma);
            }
            pixel_model.radius = config.solver.target_radius_m;
            pixel_model.half_width = config.solver.target_half_width_m;
            if (rune->ypd_in_world[2] > 1.0) {
                pixel_model.pnp_distance_m = rune->ypd_in_world[2];
            }
            auto_aim::energy::RunePixelModel::Blade locked_blade;
            if (rune->target().points.size() >= 4) {
                for (std::size_t i = 0; i < 4; ++i) {
                    locked_blade.plate_uv[i] = rune->target().points[i];
                }
                locked_blade.slot_offset = slot_snapshot.observed_offset;
                locked_blade.slot_offset_valid = slot_snapshot.observed_offset_valid;
                locked_blade.class_id = detector->targetClass();
                if (rune->target().has_rune_center) {
                    locked_blade.rmark_uv = rune->target().rune_center;
                    locked_blade.rmark_valid = true;
                }
                pixel_model.blades.push_back(locked_blade);
            }
            for (const auto& candidate : detector->lastCandidates()) {
                if (!candidate.has_plate) continue;
                // 锁定的那片已经加过（同位置），跳过。
                if (cv::norm(candidate.center - rune->target().center) < 1.0f) continue;
                auto_aim::energy::RunePixelModel::Blade blade;
                blade.plate_uv = candidate.plate_points;
                blade.rmark_uv = candidate.rmark;
                blade.rmark_valid = candidate.has_rmark;
                blade.class_id = candidate.class_id;
                blade.slot_offset_valid =
                    slot_lattice.slotOffsetForCandidate(candidate.center, blade.slot_offset);
                pixel_model.blades.push_back(blade);
            }
            target.setPixelModel(pixel_model);
        } else {
            target.clearPixelModel();
        }
        target.getTarget(rune, timestamp);
        // 换叶信号：估计器确认"观测接管"时才认为换叶（见 RuneTarget::bladeSwitched）。
        aimer.setObservedBladeSwitch(target.bladeSwitched());
        if (target.bladeSwitched()) detector->resetPhaseAssociation();

        auto_aim::energy::RuneTarget predicted = target;
        // 观测寿命：超过 observation_life_s 没有新的解算结果，就不再外推扇叶
        if (rune.has_value() && rune->solved) last_solved_time = now;
        if (rune.has_value() && rune->solved) {
            // 记录"符心"对应的云台角，丢目标时用来把相机停住（park）。
            fallback.noteCenter(-rune->ypd_in_world[0], rune->ypd_in_world[1], now);
        }
        // 观测寿命：小符角速度是已知常量（±π/3），滑行期相位预测可靠 ⇒ 允许更长的
        const double observation_life =
            config.aimer.observation_life_s * (rune_is_small ? 2.25 : 1.0);
        const bool observation_fresh =
            last_solved_time > 0.0 && (now - last_solved_time) <= observation_life;
        // ---- 扇叶选择：命中后搜索“下一片”，与观测片分离 -------------------
        std::vector<auto_aim::energy::RuneBladeSelector::Candidate> mapped_candidates;
        const float orbit_px =
            (rune.has_value() && rune->solved)
                ? static_cast<float>(cv::norm(rune->target().center - rune->r_center))
                : 0.0f;
        if (!predicted.isUnsolvable() && slot_snapshot.slot_id >= 0) {
            const float lit_tolerance = std::max(3.0f, 0.05f * orbit_px);
            for (const auto& candidate : detector->lastCandidates()) {
                int offset = 0;
                if (!slot_lattice.slotOffsetForCandidate(candidate.center, offset)) continue;
                auto_aim::energy::RuneBladeSelector::Candidate mapped;
                mapped.class_id = candidate.class_id;
                mapped.center = candidate.center;
                mapped.slot = (slot_snapshot.slot_id + offset) % 5;
                for (const cv::Point2f& lit : detector->litBladeCenters()) {
                    if (cv::norm(lit - candidate.center) <= lit_tolerance) {
                        mapped.lit = true;
                        break;
                    }
                }
                mapped_candidates.push_back(mapped);
            }
        }
        auto_aim::energy::RuneBladeSelector::Frame selector_frame;
        selector_frame.solved = !predicted.isUnsolvable();
        selector_frame.observation_fresh = observation_fresh;
        selector_frame.hold = round_guard.snapshot().hold;
        selector_frame.scored_hit = scored_hit_this_frame;
        selector_frame.observed_slot = slot_snapshot.slot_id;
        selector_frame.hit_slot = fired_slot_id;
        selector_frame.hit_center = last_fired_center;
        selector_frame.orbit_px = orbit_px;
        selector_frame.activated = slot_activation;
        selector_frame.candidates = std::move(mapped_candidates);
        blade_selector.update(selector_frame, now);
        const auto blade_selection = blade_selector.snapshot();
        if (blade_selection.retargeting && blade_selection.valid) {
            planned_slot = blade_selection.planned_slot;
            aimed_slot_activated =
                planned_slot >= 0 &&
                slot_activation[static_cast<std::size_t>(planned_slot)];
        }

        // ---- 火控边界：只判断“计划槽位能不能开火” --------------------------
        int planned_slot_class = blade_selection.target_class;
        if (planned_slot_class < 0 && !predicted.isUnsolvable() &&
            slot_snapshot.slot_id >= 0 && planned_slot >= 0) {
            for (const auto& candidate : detector->lastCandidates()) {
                int offset = 0;
                if (!slot_lattice.slotOffsetForCandidate(candidate.center, offset)) continue;
                const int candidate_slot = (slot_snapshot.slot_id + offset) % 5;
                if (candidate_slot != planned_slot) continue;
                if (candidate.class_id == 0) {
                    planned_slot_class = 0;
                    break;
                }
                if (planned_slot_class < 0) {
                    planned_slot_class = candidate.class_id;
                }
            }
        }
        auto_aim::energy::RuneFireGate::Frame fire_frame;
        fire_frame.solved = !predicted.isUnsolvable();
        fire_frame.observation_fresh = observation_fresh;
        fire_frame.planned_slot = planned_slot;
        fire_frame.planned_slot_class = planned_slot_class;
        fire_frame.activated = slot_activation;
        fire_gate.update(fire_frame, now);
        const bool fire_engageable = fire_gate.engageable();
        // 提前量用的"帧到瞄准延迟"：优先用曝光时刻做主时钟；瞬时值抖动大，做 EMA。
        if (config.aimer.frame_time_from_exposure) {
            const double instant = std::max(0.0, now - timestamp);
            if (smoothed_lead_latency_s < 0.0 || !config.aimer.smooth_lead_latency) {
                smoothed_lead_latency_s = instant;
            } else {
                smoothed_lead_latency_s += 0.2 * (instant - smoothed_lead_latency_s);
            }
        } else {
            smoothed_lead_latency_s = -1.0;
        }

        // ---- 图像相位（对 72° 取模）：类别命中/命中保持都用它，比 EKF 的 roll
        constexpr double kSlot = 2.0 * CV_PI / 5.0;
        double aim_phase = std::numeric_limits<double>::quiet_NaN();
        if (rune.has_value() && rune->solved) {
            const cv::Point2f center = rune->r_center;
            const cv::Point2f aim = rune->target().center;
            const double aim_angle = std::atan2(aim.y - center.y, aim.x - center.x);
            aim_phase = std::fmod(aim_angle, kSlot);
            if (aim_phase < 0.0) aim_phase += kSlot;
        }
        {
            auto_aim::energy::RuneRoundGuard::Frame round_frame;
            round_frame.solved = rune.has_value() && rune->solved;
            round_frame.aim_phase = aim_phase;
            round_frame.blade_class = detector->targetClass();
            round_frame.scored_hit = scored_hit_this_frame;
            round_frame.blade_switched =
                target.bladeSwitched() || slot_snapshot.switch_confirmed;
            if (round_frame.solved) {
                const cv::Point2f picked = rune->target().center;
                const double orbit_px = std::max(1.0, cv::norm(picked - rune->r_center));
                for (const auto& candidate : detector->lastCandidates()) {
                    if (candidate.class_id == 0) continue;
                    if (cv::norm(candidate.center - picked) > 0.35 * orbit_px) {
                        ++round_frame.active_witness;
                    }
                }
                round_frame.blade_center = picked;
            }
            // Update before publishing the aim state: the control thread must use
            round_guard.update(round_frame, now);
        }
        const auto round_snapshot = round_guard.snapshot();
        scored_hit_this_frame = false;

        // 瞄准解算：aim_in_control=true 时由控制线程按 100 Hz 用"当前时刻"重算
        auto_aim::energy::RuneCommand command;
        int commanded_slot_offset = 0;
        if (aim_in_control) {
            auto_aim::energy::RuneAimBridge::Input aim_input(
                config.mode, config.max_coast_frames, config.max_distance_jump_ratio,
                config.max_center_jump_deg);
            aim_input.target = predicted;
            aim_input.frame_time = timestamp;
            aim_input.latency = smoothed_lead_latency_s;
            aim_input.valid = !predicted.isUnsolvable();
            // 扇叶选择已经在观测/模型更新后完成。这里只把“计划物理槽”
            int retarget_offset = 0;
            if (slot_snapshot.slot_id >= 0 && planned_slot >= 0 &&
                planned_slot != slot_snapshot.slot_id) {
                retarget_offset =
                    ((planned_slot - slot_snapshot.slot_id) % 5 + 5) % 5;
            }
            const bool holding = round_snapshot.hold;
            commanded_slot_offset = retarget_offset;
            aim_input.center_only =
                (!observation_fresh || holding || aimed_slot_activated) &&
                retarget_offset == 0;
            aim_input.slot_offset = retarget_offset;
            // 与帧级闸门同源（见上面的 blade_engageable）：类别说"可打"**并且**
            aim_input.engageable = fire_engageable;
            aim_input.hold = holding;
            aim_bridge.publish(aim_input);
            command = aim_bridge.lastCommand();
        } else {
            command = (observation_fresh && !round_snapshot.hold)
                          ? aimer.aim(predicted, timestamp, now, smoothed_lead_latency_s)
                          : aimer.aimCenter(predicted, timestamp, now, smoothed_lead_latency_s);
        }

        const bool window_open = round_guard.windowOpen(command.fly_time);

        if (!video_mode && command.control) {
            fallback.noteControlAvailable();
            // 角度包线仍然保留：野值不能把目标角发布出去。单帧限幅交给轨迹
            const double yaw_limit = config.aim_yaw_limit_deg * CV_PI / 180.0;
            const double pitch_limit = config.aim_pitch_limit_deg * CV_PI / 180.0;
            if (std::abs(command.yaw) <= yaw_limit && std::abs(command.pitch) <= pitch_limit) {
                auto_aim::GimbalTargetAngles target_angles;
                target_angles.valid = true;
                // 先过瞄准角递归滤波：它把单帧抖动和换靶尖点抹平，并给出
                target_angles.yaw = command.yaw;
                target_angles.pitch = command.pitch;
                target_angles.yaw_velocity = command.yaw_velocity;
                target_angles.pitch_velocity = command.pitch_velocity;
                if (config.aim_filter.enabled) {
                    const auto filtered =
                        aim_filter.update(command.yaw, command.pitch, 0, now);
                    if (filtered.valid) {
                        target_angles.yaw = filtered.yaw;
                        target_angles.pitch = filtered.pitch;
                        target_angles.yaw_velocity = filtered.yaw_velocity;
                        target_angles.pitch_velocity = filtered.pitch_velocity;
                    }
                }
                // 目标角阶跃（换片/观测接管）会让"速度前馈"出现尖峰，轨迹生成器
                {
                    const double step = std::abs(auto_aim::energy::limitRad(target_angles.yaw -
                                                          previous_target_yaw));
                    if (have_previous_target &&
                        step > config.aimer.switch_angle_deg * CV_PI / 180.0) {
                        target_angles.yaw_velocity = 0.0;
                        target_angles.pitch_velocity = 0.0;
                    }
                    const double velocity_limit =
                        config.max_target_velocity_deg_s * CV_PI / 180.0;
                    target_angles.yaw_velocity =
                        std::max(-velocity_limit, std::min(velocity_limit,
                                                           target_angles.yaw_velocity));
                    target_angles.pitch_velocity =
                        std::max(-velocity_limit, std::min(velocity_limit,
                                                           target_angles.pitch_velocity));
                    previous_target_yaw = target_angles.yaw;
                    have_previous_target = true;
                }
                if (!aim_in_control) gimbal_controller.setTargetAngles(target_angles);
                // 与 sp_vision 的 buff_aimer 对齐：唯一的禁火条件是"正在换叶"，
                const auto aim_snapshot = gimbal_controller.snapshot();
                const double fire_thresh = config.fire_thresh_deg * CV_PI / 180.0;
                const bool aim_ready =
                    aim_snapshot.valid && std::abs(aim_snapshot.yaw_error) <= fire_thresh &&
                    std::abs(aim_snapshot.pitch_error) <= fire_thresh;
                // 火控闸门只来自 RuneFireGate；“帧级”和“控制线程”共用同一结果。
                const bool blade_engageable = fire_engageable;
                const bool fire_request = aim_in_control && aim_bridge.consumeFireRequest();
                // aim_in_control 模式下开火判据已经在控制线程里用"同一时刻"的三个
                const bool fire_ok =
                    aim_in_control
                        ? (fire_request && fire_enabled && !gimbal_frozen &&
                           blade_engageable)
                        : (command.shoot && fire_enabled && !gimbal_frozen &&
                           !command.blade_switched && aim_ready &&
                           (!round_snapshot.hold) && blade_engageable);
                if (fire_ok) {
                    input.sendFire();
                    ++shots_fired;
                    fired_slot_id =
                        slot_snapshot.slot_id >= 0
                            ? (slot_snapshot.slot_id + commanded_slot_offset) % 5
                            : -1;
                    if (slot_snapshot.slot_centers_valid &&
                        commanded_slot_offset >= 0 && commanded_slot_offset < 5) {
                        last_fired_center =
                            slot_snapshot.slot_centers[
                                static_cast<std::size_t>(commanded_slot_offset)];
                    } else if (rune.has_value()) {
                        last_fired_center = rune->target().center;
                    }
                } else if (command.shoot && fire_enabled && !window_open) {
                    ++shots_held_for_window;
                }
            }
        } else if (!video_mode) {
            // 暂时没有可用目标。策略在 RuneAimFallback 里（停靠最后已知符心优先，
            const auto fallback_decision = fallback.decide(now, commanded_yaw, commanded_pitch);
            // 原实现里回中与 `target.reset()` 是同一步（与是否真的发出回中指令无关）。
            if (fallback_decision.reset_estimator) target.reset();
            if (fallback_decision.kind == auto_aim::energy::RuneAimFallback::Decision::Kind::
                                               AimLastCenter) {
                auto_aim::GimbalTargetAngles parked;
                parked.valid = true;
                parked.yaw = fallback_decision.yaw;
                parked.pitch = fallback_decision.pitch;
                if (aim_in_control) {
                    auto_aim::energy::RuneAimBridge::Input park_input(
                        config.mode, config.max_coast_frames, config.max_distance_jump_ratio,
                        config.max_center_jump_deg);
                    if (target.isUnsolvable()) {
                        // 估计器彻底失效 → 停在最后已知符心。
                        park_input.valid = false;
                        park_input.park = parked.valid;
                        park_input.park_yaw = parked.yaw;
                        park_input.park_pitch = parked.pitch;
                    } else {
                        // 只是本帧没有观测、估计器还在滑行 → **保持可瞄准**：
                        park_input.target = target;
                        park_input.valid = true;
                        park_input.park = false;
                        park_input.frame_time = timestamp;
                        park_input.latency = smoothed_lead_latency_s;
                        park_input.center_only = (now - last_solved_time) > observation_life;
                        park_input.engageable = fire_engageable;
                    }
                    aim_bridge.publish(park_input);
                } else {
                    gimbal_controller.setTargetAngles(parked);
                }
                commanded_yaw = fallback_decision.yaw;
                commanded_pitch = fallback_decision.pitch;
            } else if (fallback_decision.kind ==
                       auto_aim::energy::RuneAimFallback::Decision::Kind::Recenter) {
                // 兜底：回标定位姿 + 复位估计器（相位/几何全部丢弃）。
                commanded_yaw = fallback_decision.yaw;
                commanded_pitch = fallback_decision.pitch;
                auto_aim::GimbalTargetAngles home;
                home.valid = true;
                home.yaw = fallback_decision.yaw;
                home.pitch = fallback_decision.pitch;
                if (aim_in_control) {
                    auto_aim::energy::RuneAimBridge::Input home_input(
                        config.mode, config.max_coast_frames, config.max_distance_jump_ratio,
                        config.max_center_jump_deg);
                    home_input.valid = false;
                    home_input.park = home.valid;
                    home_input.park_yaw = home.yaw;
                    home_input.park_pitch = home.pitch;
                    aim_bridge.publish(home_input);
                } else {
                    gimbal_controller.setTargetAngles(home);
                }
                std::cout << "recentering: no usable target for too long, back to the "
                             "calibration pose"
                          << std::endl;
            }
        }
        ++frames;

        const int status = static_cast<int>(detector->status());
        if (status != last_status) {
            std::cout << "rune status -> " << statusName(detector->status())
                      << std::endl;
            last_status = status;
        }

        const auto now_steady = std::chrono::steady_clock::now();
        const double elapsed = std::chrono::duration<double>(now_steady - fps_start).count();
        if (elapsed >= 1.0) {
            std::cout << "Energy rune FPS: " << frames / elapsed
                      << ", nn_ms: " << detector->latencyMs()
                      << ", lost: " << detector->lostCount()
                      << ", solvable: " << !target.isUnsolvable()
                      << ", control: " << command.control
                      << ", fire: " << command.shoot;
            if (dropped_since_print > 0) std::cout << ", dropped: " << dropped_since_print;
            std::cout << std::endl;
            dropped_since_print = 0;
            fps_start = now_steady;
            frames = 0;
        }
    }

    // 退出顺序：必须先停控制线程，再让它引用的对象析构。aim_bridge 是后构造的
    gimbal_controller.stop();

    std::cout << "=== rune run summary (simulator ground truth) ===" << std::endl
              << "  shots fired     : " << shots_fired << std::endl
              << "  hits on target  : " << telemetry_stats.hits << std::endl
              << "  scored hits     : " << telemetry_stats.scored << std::endl
              << "  activations     : " << telemetry_stats.activations << std::endl
              << "  shots held back : " << shots_held_for_window << std::endl
              << "  parked frames   : " << fallback.parkedFrames() << std::endl
              << "  recenter count  : " << fallback.recenterCount() << std::endl;
    if (telemetry_stats.first_activation_s >= 0.0) {
        std::cout << "  activation at   : " << telemetry_stats.first_activation_s
                  << " s (sim clock)" << std::endl;
    }

    return 0;
}
