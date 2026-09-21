// Energy-rune (大小能量机关) 跟踪与瞄准入口。
//
// Two inputs, mirroring the auto-aim entry points:
//   * the TCP simulator (default): frames, timestamps and the GIMBAL/FIRE
//     command channel are shared with node_sim.cpp;
//   * a video file, for offline review of recorded footage.
//
// In video mode there is no gimbal feedback, so the camera pose stays at the
// configured angles and the computed commands are only printed. Use
// ULTRA_VISION_RUNE_POSE="yaw_deg,pitch_deg" when the recording was made with
// the gimbal away from the calibration pose.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>

#include <opencv2/opencv.hpp>

#include "common/standard_clock.hpp"
#include "control/gimbal_controller.hpp"
#include "control/aim_signal_filter.hpp"
#include "auto_buff/rune_aimer.hpp"
#include "auto_buff/rune_aim_bridge.hpp"
#include "auto_buff/rune_diag.hpp"
#include "auto_buff/rune_input.hpp"
#include "auto_buff/rune_runner.hpp"
#include "auto_buff/rune_aim_fallback.hpp"
#include "auto_buff/rune_round_guard.hpp"
#include "auto_buff/rune_slot_lattice.hpp"
#include "auto_buff/rune_target_selector.hpp"
#include "auto_buff/rune_blade_finder.hpp"
#include "auto_buff/rune_config.hpp"
#include "auto_buff/rune_detector.hpp"
#include "auto_buff/rune_solver.hpp"
#include "auto_buff/rune_target.hpp"
#include "auto_buff/rune_types.hpp"
#include "auto_buff/support/math.hpp"

#ifndef ULTRA_VISION_CONFIG_DIR
#define ULTRA_VISION_CONFIG_DIR "configs"
#endif

namespace rune_diag = auto_aim::energy::rune_diag;

namespace
{
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

} // namespace

int auto_aim::energy::runRune(const auto_aim::energy::RuneConfig& config,
                              const std::string& config_dir, const std::string& video_path)
{
    const bool video_mode = !video_path.empty();
    const bool fire_enabled = std::getenv("ULTRA_VISION_DISABLE_FIRE") == nullptr;

    auto bore_sight = auto_aim::energy::rune_diag::BoreSight::fromEnvironment();
    const bool show_display = std::getenv("ULTRA_VISION_NO_DISPLAY") == nullptr;

    std::unique_ptr<auto_aim::energy::RuneDetector> detector;
    // 帧差探针（诊断用，默认关闭；ULTRA_VISION_RUNE_DIFF=1 打开）。
    auto_aim::energy::RuneDiffProbe diff_probe(config.diff_probe);
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
    // 帧循环只发布"最新目标状态 + 该帧时刻 + 平滑延迟"，控制线程按 100 Hz 用
    // **当前时刻**重算瞄准/弹道（to_now），避免目标角在两帧之间变旧；
    // 没有目标时按帧循环给的"停靠角"继续下发。ULTRA_VISION_RUNE_AIM_IN_CONTROL=0 可关。
    // 试射模式下必须关掉"控制线程算瞄准"，否则 provider 会用目标角覆盖试射角。
    const bool aim_in_control =
        (std::getenv("ULTRA_VISION_RUNE_AIM_IN_CONTROL") == nullptr ||
         std::string(std::getenv("ULTRA_VISION_RUNE_AIM_IN_CONTROL")) != "0") &&
        !bore_sight.enabled;
    // 输入层：仿真器/视频统一成一个接口，下面只管"取帧、取遥测、发指令"。
    auto_aim::energy::RuneInputSource input;
    if (!input.open(config, config_dir, video_path)) return -1;
    const std::string& frame_dir = input.frameDir();

    // 云台由 100 Hz 轨迹生成器驱动：视觉只发布目标角与目标角速度，轨迹生成器
    // 负责加加速度受限的平滑运动和高频下发。
    // 记录真正下发出去的高频指令，用于量化平滑度（每帧 CSV 看不到 100 Hz 细节）。
    auto_aim::energy::rune_diag::GimbalTrace trace;
    auto_aim::GimbalController gimbal_controller(
        config.gimbal,
        [&input, &trace](double yaw, double pitch) {
            trace.record(yaw, pitch, auto_aim::StandardClock::nowSeconds());
            return input.sendGimbal(yaw, pitch);
        },
        config.gimbal_command_rate_hz);
    auto_aim::AimSignalFilter aim_filter(config.aim_filter);
    // 瞄准桥（帧循环 <-> 控制线程）：帧循环只发布状态，控制线程按 100 Hz 重算瞄准。
    auto_aim::energy::RuneAimBridge aim_bridge(config, aimer);
    if (aim_in_control) {
        std::cout << "Energy rune: 瞄准解算在控制线程（100 Hz，to_now 预测）" << std::endl;
        aim_bridge.install(gimbal_controller, aim_filter, fire_enabled, {});
    }

    // 逐帧诊断 CSV（列定义在诊断层 rune_diag.cpp；未设置路径则完全不写）。
    std::ofstream recorder;
    if (const char* path = std::getenv("ULTRA_VISION_RUNE_CSV")) {
        if (!auto_aim::energy::rune_diag::openRecorder(recorder, path)) {
            std::cerr << "Failed to open rune CSV: " << path << std::endl;
        }
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

    cv::Mat frame;
    auto_aim::energy::RuneInputSource::Frame rune_input_frame;
    int frames = 0;
    int saved_frames = 0;
    int missing_frames = 0;
    int last_status = -1;
    // Scoring counters, filled from the simulator's ground-truth channel.
    int shots_fired = 0;
    bool fired_this_frame = false;
    // 发火/瞄准时轨迹生成器的到位误差（度），-1 表示无快照
    double aim_err_yaw_deg = -1.0;
    double aim_err_pitch_deg = -1.0;
    // 目标角阶跃检测（清零前馈用）：上一帧发布的目标角。
    double previous_target_yaw = 0.0;
    bool have_previous_target = false;
    int shots_held_for_window = 0;
    bool scored_hit_this_frame = false;
    // ---- 一轮（点亮窗口）的记账 + 命中保持 + 禁火窗口 ----------------------
    // 规则与状态都在 rune_round_guard.{hpp,cpp}：本轮何时开始/结束、类别命中
    // （class0 → class1/2）、命中后的 hold、以及窗口是否还开着。
    // 注：hold 在**本帧**发布之后才更新（与拆分前的内联顺序一致，因此发布出去的
    // hold 比测量晚一帧），这样 A/B 数据可以直接对照。
    auto_aim::energy::RuneRoundGuard round_guard(
            auto_aim::energy::RuneRoundGuard::Config{
            config.round_end_frames, config.blade_slot_jump_px, config.fire_start_delay_s,
            config.round_window_s, config.post_hit_switch_phase_rad, 2.0 * CV_PI / 5.0,
            config.post_hit_hold_max_s, config.post_hit_require_active_witness});
    // 上一次成功解算出观测的本地时间（用于"观测寿命"判定，见 RuneAimer）。
    double last_solved_time = -1.0;
    // "这片可以打"的锁存截止时刻（见 RuneDetectorConfig::engageable_latch_s）。
    double target_engageable_until = 0.0;
    // 丢目标时的云台策略（停在最后已知符心 / 兜底回标定位姿），见其头文件。
    auto_aim::energy::RuneAimFallback fallback(
        auto_aim::energy::RuneAimFallback::Config{config.park_on_center_max_age_s,
                                                  config.recenter_after_frames});
    static const bool dark_channel_enabled =
        std::getenv("ULTRA_VISION_RUNE_DARK_CHANNEL") != nullptr &&
        std::string(std::getenv("ULTRA_VISION_RUNE_DARK_CHANNEL")) != "0";
    // 选 target（sp_vision 规则）：亮片数 + 像素距离，不依赖网络类别。
    auto_aim::energy::RuneTargetSelector target_selector;
    auto_aim::energy::RuneBladeFinder blade_finder;
    // ---- 扇叶槽位格点（1..5 号位的记账）------------------------------------
    // 状态与规则都在 rune_slot_lattice.{hpp,cpp}：单扇叶构建 5 槽、每帧重新
    // 对齐、丢帧不重锚、静默才复位。这里只负责喂输入、取快照。
    auto_aim::energy::RuneSlotLattice::Config lattice_config;
    lattice_config.activate_votes = config.slot_activate_votes;
    lattice_config.reset_silence_s = config.slot_lattice_reset_silence_s;
    // 换片判据：默认"状态相位跳变"；=1 用深大式"观测偏移 vs 运动模型预测"（需配合
    // ULTRA_VISION_RUNE_SLOT_OFFSET=0 让估计器跟着观测走，否则会反复推进）。
    lattice_config.switch_by_geometry =
        std::getenv("ULTRA_VISION_RUNE_SWITCH_BY_GEOMETRY") != nullptr &&
        std::string(std::getenv("ULTRA_VISION_RUNE_SWITCH_BY_GEOMETRY")) != "0";
    // 激活记账的两个新开关（见 rune_slot_lattice.hpp 的说明）：
    //   * book_retired：换片 ⇒ 被换下的那片已激活（几何证据，新模型下唯一的来源）；
    //   * rotate_booking：换片时是否旋转整张掩码（按槽位号=物理片号的定义，不该旋转）。
    lattice_config.book_retired_on_switch = config.slot_book_retired_on_switch;
    lattice_config.rotate_booking_on_switch = config.slot_rotate_booking_on_switch;
    auto_aim::energy::RuneSlotLattice slot_lattice(
        lattice_config, std::getenv("ULTRA_VISION_RUNE_DEBUG") != nullptr);
    int fired_slot_id = -1;                  // 命中瞬间开火的那个槽位（喂回格点）
    auto_aim::energy::rune_diag::RuneTelemetry telemetry_stats;
    // 本地时钟 - 仿真时钟（见遥测解析处），用于把 CSV 的时间轴对齐到仿真。

    // 像素残差观测开关（默认开，`=0` 回到 3D 量测）与"多片同帧"开关。
    static const bool pixel_residual_enabled =
        std::getenv("ULTRA_VISION_RUNE_PIXEL_RESIDUAL") == nullptr ||
        std::string(std::getenv("ULTRA_VISION_RUNE_PIXEL_RESIDUAL")) != "0";
    // R 标那条量测要不要用（ULTRA_VISION_RUNE_PIXEL_RMARK=0 关掉）
    // 命中后转到"新亮片"（sp_vision 的判据）：ULTRA_VISION_RUNE_RETARGET_NEW_BLADE=0 可关
    // 确认换片后重建估计器（三家参考都这么做）：ULTRA_VISION_RUNE_REBUILD_ON_SWITCH=0 可关
    // 实测：开它（每次确认换片都重建）记账漂移更大（前后半移位差 3~4 格 vs 1 格），
    // 端到端相当。默认关，开关保留（ULTRA_VISION_RUNE_REBUILD_ON_SWITCH=1 打开）。
    static const bool rebuild_on_switch =
        std::getenv("ULTRA_VISION_RUNE_REBUILD_ON_SWITCH") != nullptr &&
        std::string(std::getenv("ULTRA_VISION_RUNE_REBUILD_ON_SWITCH")) != "0";
    static const bool retarget_new_blade =
        std::getenv("ULTRA_VISION_RUNE_RETARGET_NEW_BLADE") == nullptr ||
        std::string(std::getenv("ULTRA_VISION_RUNE_RETARGET_NEW_BLADE")) != "0";
    static const bool pixel_use_rmark =
        std::getenv("ULTRA_VISION_RUNE_PIXEL_RMARK") == nullptr ||
        std::string(std::getenv("ULTRA_VISION_RUNE_PIXEL_RMARK")) != "0";
    static const bool pixel_distance_anchor =
        std::getenv("ULTRA_VISION_RUNE_PIXEL_DISTANCE_ANCHOR") == nullptr ||
        std::string(std::getenv("ULTRA_VISION_RUNE_PIXEL_DISTANCE_ANCHOR")) != "0";
    static const bool pixel_multi_blade =
        std::getenv("ULTRA_VISION_RUNE_PIXEL_MULTI_BLADE") == nullptr ||
        std::string(std::getenv("ULTRA_VISION_RUNE_PIXEL_MULTI_BLADE")) != "0";
    // 阶段计时（ULTRA_VISION_RUNE_PROFILE=1）：分出"推理之外的时间花在哪"。
    rune_diag::StageProfiler profiler(std::getenv("ULTRA_VISION_RUNE_PROFILE") != nullptr);
    const auto stage_ms = [](const auto& start) {
        return std::chrono::duration<double, std::milli>(
                   std::chrono::steady_clock::now() - start)
            .count();
    };
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
    long long dropped_frames = 0;
    long long dropped_since_print = 0;
    int key = -1;

    while (key != 27 && key != 'e' && key != 'E') {
        const double wall_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - run_start).count();
        if (test_seconds > 0.0 && wall_seconds >= test_seconds) {
            std::cout << "Reached ULTRA_VISION_RUNE_TEST_SECONDS=" << test_seconds
                      << " s, exiting" << std::endl;
            break;
        }

        if (!video_mode) {
            // 遥测（仿真真值）：命中/计分/激活 + 仿真时钟偏移，全部交给诊断层。
            auto_aim::energy::rune_diag::consumeTelemetry(
                input.pollTelemetry(), input.groundTruth(), rune_mode_name,
                telemetry_stats, &scored_hit_this_frame);
            if (scored_hit_this_frame) {
                round_guard.noteScoredHit(auto_aim::StandardClock::nowSeconds());
            }
        }

        rune_input_frame = {};
        const auto stage_read_start = std::chrono::steady_clock::now();
        if (!input.readFrame(rune_input_frame)) {
            if (video_mode) break;
            ++missing_frames;
            if (missing_frames == 1 || missing_frames % 60 == 0) {
                std::cerr << "Waiting for simulator frames..." << std::endl;
            }
            key = cv::waitKey(1);
            continue;
        }
        profiler.add("read", stage_ms(stage_read_start));
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
        // 两者相差过大说明时钟不可比，退回本地到达时刻。
        uint64_t frame_time_us = frame_local_timestamp_us;
        if (frame_source_timestamp_us > 0) {
            const long long skew = static_cast<long long>(frame_source_timestamp_us) -
                                   static_cast<long long>(frame_local_timestamp_us);
            if (std::llabs(skew) < 1000000LL) frame_time_us = frame_source_timestamp_us;
        }
        const double timestamp = auto_aim::StandardClock::secondsFromUs(frame_time_us);

        solver.updateImageSize(frame.cols, frame.rows);

        // 用轨迹生成器的历史快照取"这一帧曝光时刻"的云台姿态，而不是我们自己
        // 记的下发值：异步下发 + 帧延迟下两者能差好几度。
        if (!video_mode) {
            const auto pose = gimbal_controller.snapshotAt(frame_time_us);
            if (pose.valid) {
                commanded_yaw = pose.yaw;
                commanded_pitch = pose.pitch;
            }
        }

        // 用轨迹生成器的历史快照取"这一帧曝光时刻"的云台姿态，而不是我们
        // 自己记的下发值：异步下发 + 帧延迟下，两者能差好几度。
        if (!video_mode) {
            const auto pose = gimbal_controller.snapshotAt(frame_local_timestamp_us);
            if (pose.valid) {
                commanded_yaw = pose.yaw;
                commanded_pitch = pose.pitch;
            }
        }

        // The estimate is expressed relative to the pose the rune was
        // calibrated in; feed the current absolute gimbal angles.
        solver.setCameraPose(commanded_yaw, commanded_pitch);

        const auto stage_detect_start = std::chrono::steady_clock::now();
        // 判定探针（只用于量吞吐上限）：隔帧跑一次推理，看帧率会不会涨。
        // 涨 ⇒ 我们是限速方（仿真被我们拖慢），不涨 ⇒ 仿真自己就这个速度。
        static const bool probe_skip_detect =
            std::getenv("ULTRA_VISION_RUNE_SKIP_DETECT") != nullptr;
        static std::optional<auto_aim::energy::PowerRune> probe_last_rune;
        static int probe_frame_index = 0;
        std::optional<auto_aim::energy::PowerRune> rune;
        if (probe_skip_detect && (++probe_frame_index % 2 == 1)) {
            rune = probe_last_rune;
        } else {
            rune = config.detector.multi_candidate ? detector->detect(frame)
                                                   : detector->detectBest(frame);
            if (probe_skip_detect) probe_last_rune = rune;
        }
        profiler.add("detect", stage_ms(stage_detect_start));
        // 采样时刻放在**推理之后**：`now` 要参与瞄准的提前量计算
        //   detection_age = now - timestamp(曝光时刻)
        // 现在它包含"传输 + 排队 + 推理(实测 p50 41 ms)"的全部延迟，云台锁上新片
        // 的那几帧才不会系统性滞后（现场反馈：归中后重新锁片时总差一点、抖动大）。
        const double now = auto_aim::StandardClock::nowSeconds();
        const auto stage_solve_start = std::chrono::steady_clock::now();
        solver.solve(rune);
        profiler.add("solve", stage_ms(stage_solve_start));
        // ---- 扇叶 ID：把当帧观测喂给槽位格点，取出 1..5 号位的记账 ----------
        // **必须在喂估计器之前**：格点同时给出"本帧锁定的那片相对跟踪槽位偏了几个
        // 槽位"，观测到已激活片时估计器要靠它把观测折算回跟踪槽位（不然瞄点会整跳
        // 一片）。规则见 rune_slot_lattice.hpp 的注释。
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
                // class 1/2 折算成槽位偏移，直接记账"该槽已激活"。
                const auto& candidates = detector->lastCandidates();
                lattice_frame.candidates.reserve(candidates.size());
                for (const auto& candidate : candidates) {
                    lattice_frame.candidates.emplace_back(candidate.class_id, candidate.center);
                }
                // 5 个槽位在图像里的预测位置：k=0 是当前瞄准的那片，k=1..4 依次
                // 往后 72°。用解算层投影（含透视压缩），比"候选与靶心的图像夹角
                // ÷72°"准得多（实测后者把掩码一致率封在 ~45%）。
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
            // 它只要连续 5 帧没有解算结果（≈165 ms）就置假，而我们的检出占空比
            // 只有 ~80%，于是"回合结束"的事件每 0.3~0.5 s 触发一次，
            // clearActivation 把记账整张抹掉（实测 607 个已锚定帧里 255 帧掩码是
            // 00000 = 42%）。真实回合结束是"机关整片熄灭/复位"，量级是秒：
            // 这里改成"这么久没有任何解算结果才算本轮结束"，与格点自己的
            // reset_silence_s 同一量级。
            lattice_frame.round_active =
                (rune.has_value() && rune->solved) ||
                (last_solved_time > 0.0 &&
                 now - last_solved_time < config.slot_lattice_round_end_s);
            // 换片里程计的驱动信号（估计器确认观测接管）。
            lattice_frame.blade_switched = target.bladeSwitched();
            // 绝对相位 + 运动模型角速度（深大式换片判据的输入）
            lattice_frame.observed_phase_valid = rune->phase_valid;
            lattice_frame.observed_phase = rune->phase_rad;
            lattice_frame.phase_rate =
                target.ekfX().size() > 6 ? target.ekfX()[6] : 0.0;
            slot_lattice.update(lattice_frame, now);
        }
        const auto slot_snapshot = slot_lattice.snapshot();
        // ---- 开火闸门的两个"可打"条件（帧级，本帧唯一口径）-------------------
        // (1) 类别：五点模型里只有 class 0（未激活）才是本轮目标；
        // (2) 槽位记账：我们**自己记的**"这一槽已经打过"同样不许开火。
        // 第 (2) 条以前一直关着（记账靠网络 class 1/2，新模型下这条路是死的，§48）；
        // 现在记账改成几何来源（换片 ⇒ 被换下的那片已激活，见 rune_slot_lattice），
        // 才重新有资格进闸门。A/B：ULTRA_VISION_RUNE_BLOCK_HIT_SLOT=0/1。
        const bool class_aware = config.detector.model.output_layout == "v8_pose_5kpt";
        // 按几何连续性选出来的目标不受类别闸门约束：只剩一片未激活时网络经常把
        // 已激活片标成 class0、把目标标成 class1，用类别卡会把火挡死（实测
        // "最后一片怎么都打不出去"）。这种情况下的保护交给"命中后保持"与槽位记账。
        if (rune.has_value() && rune->solved &&
            (detector->targetClass() == 0 || detector->targetPickedByGeometry())) {
            target_engageable_until = now + config.detector.engageable_latch_s;
        }
        const bool class_engageable =
            !class_aware || !config.detector.require_inactive_class ||
            target_engageable_until > now;
        static const bool block_hit_slot = [] {
            const char* env = std::getenv("ULTRA_VISION_RUNE_BLOCK_HIT_SLOT");
            return env == nullptr ? true : std::string(env) != "0";
        }();
        const bool aimed_slot_activated =
            config.fire_block_on_hit_slot && block_hit_slot && slot_snapshot.slot_id >= 0 &&
            slot_snapshot.activated[static_cast<std::size_t>(slot_snapshot.slot_id)];
        // 观测到的是**已激活**的那片（class 1/2）时告诉估计器：它相对跟踪槽位偏了
        // 几个槽位，请按这个偏移折算（相位 -offset×72°、靶心预测 +offset×72°）。
        const int observed_class = detector->targetClass();
        // A/B 开关：ULTRA_VISION_RUNE_SLOT_OFFSET=0 关掉折算（回到"瞄点跟着观测片跑"
        // 的旧行为），用来分离"折算"这一项对云台瞄准数据的影响。
        // 默认**折算**（观测折算回跟踪槽位：瞄点稳、记账不漂，实测 §28）。
    // ULTRA_VISION_RUNE_SLOT_OFFSET=0 ⇒ 观测按原样进估计器（估计器跟着观测片走，
    // 与三家参考一致，但实测记账漂移更大、端到端相当）。
    static const bool slot_offset_enabled =
        std::getenv("ULTRA_VISION_RUNE_SLOT_OFFSET") == nullptr ||
        std::string(std::getenv("ULTRA_VISION_RUNE_SLOT_OFFSET")) != "0";
        if (slot_offset_enabled && rune.has_value() && rune->solved &&
            slot_snapshot.observed_offset_valid && observed_class > 0) {
            rune->slot_offset = slot_snapshot.observed_offset;
            rune->slot_offset_valid = true;
        }
        const auto stage_target_start = std::chrono::steady_clock::now();
        // 像素残差观测（默认开；ULTRA_VISION_RUNE_PIXEL_RESIDUAL=0 可关回 3D 量测）：
        // 把网络给的像素直接注入估计器，观测函数=重投影。
        // 多片同帧（ULTRA_VISION_RUNE_PIXEL_MULTI_BLADE=1，默认开）：连 class 1/2 的
        // 已激活片一起用 —— 同一个刚体，每多一片就多一组几何约束。
        if (pixel_residual_enabled && rune.has_value() && rune->solved) {
            auto_aim::energy::RunePixelModel pixel_model;
            pixel_model.project = [&solver](const Eigen::Vector3d& world) {
                return solver.projectToImage(world);
            };
            if (const char* rmark_sigma = std::getenv("ULTRA_VISION_RUNE_RMARK_SIGMA_PX")) {
                pixel_model.sigma_rmark_px = std::atof(rmark_sigma);
            }
            pixel_model.radius = config.solver.target_radius_m;
            pixel_model.half_width = config.solver.target_half_width_m;
            // 距离锚（ULTRA_VISION_RUNE_PIXEL_DISTANCE_ANCHOR=0 可关）
            if (pixel_distance_anchor && rune->ypd_in_world[2] > 1.0) {
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
                if (pixel_use_rmark && rune->target().has_rune_center) {
                    locked_blade.rmark_uv = rune->target().rune_center;
                    locked_blade.rmark_valid = true;
                }
                pixel_model.blades.push_back(locked_blade);
            }
            if (pixel_multi_blade) {
                for (const auto& candidate : detector->lastCandidates()) {
                    if (!candidate.has_plate) continue;
                    // 锁定的那片已经加过（同位置），跳过。
                    if (cv::norm(candidate.center - rune->target().center) < 1.0f) continue;
                    auto_aim::energy::RunePixelModel::Blade blade;
                    blade.plate_uv = candidate.plate_points;
                    blade.rmark_uv = candidate.rmark;
                    blade.rmark_valid = pixel_use_rmark && candidate.has_rmark;
                    blade.class_id = candidate.class_id;
                    blade.slot_offset_valid =
                        slot_lattice.slotOffsetForCandidate(candidate.center, blade.slot_offset);
                    pixel_model.blades.push_back(blade);
                }
            }
            target.setPixelModel(pixel_model);
        } else {
            target.clearPixelModel();
        }
        // 参考实现的做法：**确认换片后直接重建估计器到新片**（而不是让滤波器慢慢漂）。
        if (rebuild_on_switch && rune.has_value() && rune->solved &&
            slot_snapshot.switch_confirmed) {
            target.rebuildFrom(rune.value(), timestamp);
        }
        target.getTarget(rune, timestamp);
        profiler.add("target+lattice", stage_ms(stage_target_start));
        // 换叶信号：估计器确认"观测接管"时才认为换叶（见 RuneTarget::bladeSwitched）。
        // 检测层的相位关联在换叶后必须重新捕获（否则方向/相位残留会挡掉新靶）。
        aimer.setObservedBladeSwitch(target.bladeSwitched());
        if (target.bladeSwitched()) detector->resetPhaseAssociation();

        auto_aim::energy::RuneTarget predicted = target;
        // 观测寿命：超过 observation_life_s 没有新的解算结果，就不再外推扇叶
        // （外推会把云台按 60°/s 甩出去、40°+ 后触发归位），改为瞄圆心稳住相机。
        if (rune.has_value() && rune->solved) last_solved_time = now;
        if (rune.has_value() && rune->solved) {
            // 记录"符心"对应的云台角，丢目标时用来把相机停住（park）。
            fallback.noteCenter(-rune->ypd_in_world[0], rune->ypd_in_world[1], now);
        }
        // 观测寿命：小符角速度是已知常量（±π/3），滑行期相位预测可靠 ⇒ 允许更长的
        // "继续瞄扇叶"窗口；大符角速度会变，滑行误差随时间涨，窗口保持较短。
        const double observation_life =
            config.aimer.observation_life_s * (rune_is_small ? 2.25 : 1.0);
        const bool observation_fresh =
            last_solved_time > 0.0 && (now - last_solved_time) <= observation_life;
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

        // 瞄准解算：aim_in_control=true 时由控制线程按 100 Hz 用"当前时刻"重算
        // （sp_vision 的 commandgener 结构），这里只发布状态、并取回它算出的指令。
        auto_aim::energy::RuneCommand command;
        if (aim_in_control) {
            auto_aim::energy::RuneAimBridge::Input aim_input(
                config.mode, config.max_coast_frames, config.max_distance_jump_ratio,
                config.max_center_jump_deg);
            aim_input.target = predicted;
            aim_input.frame_time = timestamp;
            aim_input.latency = smoothed_lead_latency_s;
            aim_input.valid = !predicted.isUnsolvable();
            // 命中后保持期间**瞄符心**而不是继续盯着刚打过的片：
            //   * 那片已经激活，打它不计分（实测 66% 的命中如此）；
            //   * 符心是静止点，等新靶亮起时"符心 → 新靶"的摆动最多半个回
            //     转半径，而"旧靶 → 对面新靶"可能差一整片（小符 6 m 处约 13°）。
            // 命中保持期内按 sp_vision 的判据转到**新亮起的那片**：在候选里找
            // "按我们的记账还没打过"的那片（优先 class 0），用它的槽位偏移瞄准。
            int retarget_offset = 0;
            if (retarget_new_blade && round_guard.snapshot().hold) {
                // sp_vision：命中后要打的是"新亮起的那片"，由选择器给出（几何判据）。
                const auto selection =
                    target_selector.select(detector->litBladeCenters());
                if (selection.valid) {
                    int offset = 0;
                    if (slot_lattice.slotOffsetForCandidate(selection.center, offset)) {
                        retarget_offset = offset;
                    }
                }
            }
            const bool holding = round_guard.snapshot().hold;
            aim_input.center_only = (!observation_fresh || holding) && retarget_offset == 0;
            aim_input.slot_offset = retarget_offset;
            // 与帧级闸门同源（见上面的 blade_engageable）：类别说"可打"**并且**
            // 我们自己的槽位记账没有把这一槽标成已激活，才允许这帧开火。
            // 需要重新瞄一片（retarget_offset != 0）时，瞄的是**另一片**：被记账
            // 挡掉的是"跟踪槽位"而不是真正要打的那片，这时只按类别判。
            aim_input.engageable =
                class_engageable &&
                (retarget_offset != 0 || !aimed_slot_activated);
            // 注意：与本帧的 round_guard.update() 相比，这里是**上一帧**的结论
            // （与拆分前的内联顺序一致，发布出去的 hold 晚一帧）。
            aim_input.hold = holding;
            aim_bridge.publish(aim_input);
            command = aim_bridge.lastCommand();
        } else {
            command = (observation_fresh && !round_guard.snapshot().hold)
                          ? aimer.aim(predicted, timestamp, now, smoothed_lead_latency_s)
                          : aimer.aimCenter(predicted, timestamp, now, smoothed_lead_latency_s);
        }

        // 静态瞄点试射（判决实验）：固定角度 + 固定间隔开火，绕过目标相关闸门。
        // 用来分离"弹道/发射模型对不对"和"动态瞄准（提前量/云台到位）好不好"：
        // 瞄在轨道上（半径 0.7 m ≈ ±6.6°）应能稳定打中；0°（符心）打不到。
        bore_sight.apply(command, now);

        // 瞄准点（命中时刻的预测靶心）反投影到图像：用于曲线/误差归因。
        cv::Point2f aim_pixel(0.0f, 0.0f);
        Eigen::Vector3d aim_world = Eigen::Vector3d::Zero();
        double aim_roll_deg = 0.0;
        if (command.control && predicted.ekfX().size() > 5) {
            const double radius = observation_fresh ? config.aimer.target_radius_m : 0.0;
            aim_world = predicted.pointBuffToWorld(Eigen::Vector3d(0.0, 0.0, radius));
            aim_pixel = solver.projectToImage(aim_world);
            aim_roll_deg = predicted.ekfX()[5] * 180.0 / CV_PI;
        }

        // ---- 图像相位（对 72° 取模）：类别命中/命中保持都用它，比 EKF 的 roll
        //      稳得多，也不会被解缠影响。
        constexpr double kSlot = 2.0 * CV_PI / 5.0;
        std::vector<double> lit_phases;
        double aim_phase = std::numeric_limits<double>::quiet_NaN();
        // 未取模的图像相位（度，0~360）：用于和 EKF 的相位画在同一条曲线上对比。
        double aim_phase_full = std::numeric_limits<double>::quiet_NaN();
        if (rune.has_value() && rune->solved) {
            const cv::Point2f center = rune->r_center;
            for (const cv::Point2f& blade : detector->litBladeCenters()) {
                const double angle = std::atan2(blade.y - center.y, blade.x - center.x);
                double phase = std::fmod(angle, kSlot);
                if (phase < 0.0) phase += kSlot;
                lit_phases.push_back(phase);
            }
            const cv::Point2f aim = rune->target().center;
            const double aim_angle = std::atan2(aim.y - center.y, aim.x - center.x);
            aim_phase = std::fmod(aim_angle, kSlot);
            if (aim_phase < 0.0) aim_phase += kSlot;
            aim_phase_full = std::fmod(aim_angle * 180.0 / CV_PI, 360.0);
            if (aim_phase_full < 0.0) aim_phase_full += 360.0;
        }
        // 一轮记账 / 类别命中（class0 → class1/2）/ 命中保持：全部在
        // RuneRoundGuard 里（自带头文件说明；可喂序列帧单测）。
        // 槽位/轮次状态机（rune_activation）已删除：实测与仿真 activated 掩码
        // 44~86% 不一致，且它的开火闸门一直是关的——sp_vision 也没有这一层。
        {
            auto_aim::energy::RuneRoundGuard::Frame round_frame;
            round_frame.solved = rune.has_value() && rune->solved;
            round_frame.aim_phase = aim_phase;
            round_frame.blade_class = detector->targetClass();
            round_frame.scored_hit = scored_hit_this_frame;
            // 确认换片（估计器的"观测接管"）：命中后保持的正确解禁信号。
            // 加上槽位格点的几何换片判据（"点亮的那片换到别的槽位了"）：
            // 命中后放行的速度不再完全取决于估计器什么时候接管观测（实测 0.4~0.8 s）。
            round_frame.blade_switched = target.bladeSwitched() || slot_snapshot.switch_confirmed;
            // "已激活片作证"：命中后机关里至少有一片已激活，它应当被读成 class 1/2、
            // 且位置与锁定片明显不同（同一片只会有一次检测/一个类别）。看不到它，
            // 说明"锁定片读成 class 0"很可能是把已激活片误读了 → 不许解禁开火。
            if (round_frame.solved) {
                const cv::Point2f picked = rune->target().center;
                const double orbit_px = std::max(1.0, cv::norm(picked - rune->r_center));
                for (const auto& candidate : detector->lastCandidates()) {
                    if (candidate.class_id == 0) continue;
                    if (cv::norm(candidate.center - picked) > 0.35 * orbit_px) {
                        ++round_frame.active_witness;
                    }
                }
            }
            if (round_frame.solved) round_frame.blade_center = rune->target().center;
            round_guard.update(round_frame, now);
        }
        const auto round_snapshot = round_guard.snapshot();
        scored_hit_this_frame = false;
        fired_this_frame = false;
        aim_err_yaw_deg = -1.0;
        aim_err_pitch_deg = -1.0;
        const bool window_open = round_guard.windowOpen(command.fly_time);

        if (!video_mode && command.control) {
            fallback.noteControlAvailable();
            // 角度包线仍然保留：野值不能把目标角发布出去。单帧限幅交给轨迹
            // 生成器的速度/加速度/加加速度约束，不再做 8°/帧的硬截断，
            // 否则目标角每帧都被削平，反而制造阶跃。
            const double yaw_limit = config.aim_yaw_limit_deg * CV_PI / 180.0;
            const double pitch_limit = config.aim_pitch_limit_deg * CV_PI / 180.0;
            if (std::abs(command.yaw) <= yaw_limit && std::abs(command.pitch) <= pitch_limit) {
                auto_aim::GimbalTargetAngles target_angles;
                target_angles.valid = true;
                // 先过瞄准角递归滤波：它把单帧抖动和换靶尖点抹平，并给出
                // 干净的角度与角速度，再交给轨迹生成器做加加速度受限的运动。
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
                // 被推过它的线性区间就变成超调振荡 —— 实测下发角在 6↔38° 之间来回摆，
                // 而目标角本身是稳的。这里做两道限制：
                //   1) 阶跃超过 switch_angle_deg 的帧，前馈清零，交给生成器自己爬；
                //   2) 前馈速度包络限幅（默认 200°/s，约 1.7 倍靶心最大转速）。
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
                // 换叶期间不开火并重置节流计时；其余按 fire_gap_time 节流。
                // 之前额外加的"本轮 2.5 s 窗口"闸门会让命中第一片之后窗口永久
                // 关闭、再也不发弹，这里不再作为开火条件（仅保留统计）。
                // sp_vision planner 的判火思路：不看"有没有到窗口"，而看
                // **击发时刻云台是否真的到了瞄点**。这里用轨迹生成器的
                // 状态快照（含期望角与误差）作判据，fire_thresh 是容差。
                const auto aim_snapshot = gimbal_controller.snapshot();
                if (aim_snapshot.valid) {
                    aim_err_yaw_deg = aim_snapshot.yaw_error * 180.0 / CV_PI;
                    aim_err_pitch_deg = aim_snapshot.pitch_error * 180.0 / CV_PI;
                }
                const double fire_thresh = config.fire_thresh_deg * CV_PI / 180.0;
                const bool aim_ready =
                    aim_snapshot.valid && std::abs(aim_snapshot.yaw_error) <= fire_thresh &&
                    std::abs(aim_snapshot.pitch_error) <= fire_thresh;
                // 可打性在发布状态之前就算好了（见上面的 class_engageable /
                // aimed_slot_activated）——"帧级闸门"和"控制线程闸门"必须同一口径。
                const bool blade_engageable = class_engageable && !aimed_slot_activated;
                if (rune_diag::debugEnabled()) {
                    // 开火被哪个闸门挡住，是调"每轮命中节奏"的关键信息。
                    rune_diag::printFireGateDebug({command.shoot, command.blade_switched,
                                                   aim_ready, round_snapshot.hold,
                                                   blade_engageable, window_open});
                }
                const bool fire_request = aim_in_control && aim_bridge.consumeFireRequest();
                // aim_in_control 模式下开火判据已经在控制线程里用"同一时刻"的三个
                // 条件判过（见 provider），这里只负责真正发火，不再用 30 Hz 的采样值
                // 二次裁决（否则会把已经判好的火又挡掉）。
                const bool fire_ok =
                    bore_sight.enabled
                        ? (command.shoot && fire_enabled)
                        : (aim_in_control
                               ? (fire_request && fire_enabled)
                               : (command.shoot && fire_enabled && !command.blade_switched &&
                                  aim_ready && (!round_snapshot.hold) && blade_engageable));
                if (fire_ok) {
                    input.sendFire();
                    ++shots_fired;
                    fired_slot_id = slot_snapshot.slot_id;
                    // 记录"真的发了这一发"，供离线归因（CSV 列 fired）。
                    fired_this_frame = true;
                } else if (command.shoot && fire_enabled && !window_open) {
                    ++shots_held_for_window;
                }
            }
        } else if (!video_mode) {
            // 暂时没有可用目标。策略在 RuneAimFallback 里（停靠最后已知符心优先，
            // 兜底才回标定位姿），这里只负责执行判决。
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
                        // 让控制线程继续用"预测到当前时刻"的解算结果（sp_vision 的
                        // commandgener 也是拿 0.2 s 内的旧目标继续算指令），
                        // 观测过期则退化为瞄符心。
                        park_input.target = target;
                        park_input.valid = true;
                        park_input.park = false;
                        park_input.frame_time = timestamp;
                        park_input.latency = smoothed_lead_latency_s;
                        park_input.center_only = (now - last_solved_time) > observation_life;
                        park_input.engageable =
                            !config.detector.require_inactive_class || detector->targetClass() == 0;
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
        if (command.shoot && !fire_enabled) {
            rune_diag::printFireSuppressed(command);
        }

        ++frames;
        if (rune_diag::debugEnabled()) {
            // 逐帧状态轨迹（排查"跟丢/云台跑偏"只能靠它），格式在诊断层。
            rune_diag::printFrameDebug(
                {frames, *detector, rune.has_value() ? &rune.value() : nullptr,
                 commanded_yaw, commanded_pitch, command});
        }
        rune_diag::dumpFrame(frame_dir, frame, frames, saved_frames);

        const int status = static_cast<int>(detector->status());
        if (status != last_status) {
            std::cout << "rune status -> " << rune_diag::runeStatusName(detector->status())
                      << std::endl;
            last_status = status;
        }

        // 帧差探针（差分法，诊断）：圆心/轨道半径用状态估计投影到本帧图像得到，
        // 具体差分/配准/连通域在诊断层（rune_diag::probeDiff）。
        auto_aim::energy::rune_diag::DiffSample diff_sample;
        if (config.diff_probe.enabled) {
            cv::Point2f hub_px(-1.0f, -1.0f);
            double orbit_px = 0.0;
            if (!target.isUnsolvable()) {
                const Eigen::Vector3d hub_world =
                    target.pointBuffToWorld(Eigen::Vector3d::Zero());
                hub_px = solver.projectToImage(hub_world);
                const Eigen::Vector3d blade_world = target.pointBuffToWorld(
                    Eigen::Vector3d(0.0, 0.0, config.aimer.target_radius_m));
                orbit_px = cv::norm(solver.projectToImage(blade_world) - hub_px);
            }
            diff_sample = rune_diag::probeDiff(diff_probe, frame, hub_px, orbit_px,
                                               detector->lastLitBladeCenter());
        }
        if (recorder) {
            // 逐帧诊断行：本循环只把"来源"凑齐，列的搬运/格式化在诊断层
            // （rune_diag::makeRow，见 rune_diag.cpp）。
            auto_aim::energy::rune_diag::RuneDiagSources sources{
                *detector, target, rune.has_value() ? &rune.value() : nullptr, command};
            sources.local_time_us = frame_local_timestamp_us;
            sources.sent_yaw = commanded_yaw;
            sources.sent_pitch = commanded_pitch;
            sources.aim_phase_deg = aim_phase_full;
            sources.aim_err_yaw_deg = aim_err_yaw_deg;
            sources.aim_err_pitch_deg = aim_err_pitch_deg;
            sources.fired = fired_this_frame;
            sources.gate_hold = round_snapshot.hold;
            sources.gate_window_open = window_open;
            // 与帧级闸门同口径（blade_engageable 在开火分支里才定义）
            sources.gate_engageable = class_engageable && !aimed_slot_activated;
            sources.aim_pixel = aim_pixel;
            sources.aim_world = aim_world;
            sources.aim_roll_deg = aim_roll_deg;
            sources.slot_id = slot_snapshot.slot_id;
            sources.slot_activated = slot_snapshot.activated;
            sources.slot_votes = slot_snapshot.votes;
            sources.slot_centers = slot_snapshot.slot_centers;
            const auto orbit = solver.lastOrbitFit();
            sources.orbit_valid = orbit.valid;
            sources.orbit_semi_major_px = orbit.semi_major_px;
            sources.orbit_center = orbit.center;
            sources.orbit_samples = orbit.samples;
            // 暗片通道（默认关，ULTRA_VISION_RUNE_DARK_CHANNEL=1 打开）：
            // §36 的崩溃排查 —— 先加输入守卫（槽位中心必须有效、轨道半径合理）。
            if (dark_channel_enabled && slot_snapshot.slot_centers_valid &&
                rune->r_center.x > 1.0f) {
            // 有它之后，槽位归属不再依赖状态投影（信息更完备；rm_vision_core 那套
            // 5 片配准的前提）。
                std::array<double, 5> angles{};
                const double hub_x = rune->r_center.x;
                const double hub_y = rune->r_center.y;
                for (int slot = 0; slot < 5; ++slot) {
                    const cv::Point2f center = slot_snapshot.slot_centers[
                        static_cast<std::size_t>(slot)];
                    angles[static_cast<std::size_t>(slot)] =
                        std::atan2(center.y - hub_y, center.x - hub_x);
                }
                const double orbit_px = cv::norm(slot_snapshot.slot_centers[0] - rune->r_center);
                // 期望面积用**网络自己检出的靶面面积**（深大用相对误差而非绝对阈值）
                double expected_area = 0.0;
                if (rune->target().points.size() >= 4) {
                    std::vector<cv::Point2f> quad(rune->target().points.begin(),
                                                  rune->target().points.begin() + 4);
                    expected_area = cv::contourArea(quad);
                }
                // 尺度基准：用**实测靶面半尺寸**（网络四点构成四边形 → 半边长）
                const double plate_half_px = expected_area > 1.0 ? 0.5 * std::sqrt(expected_area)
                                                                : 9.0;
                const int search_px = std::max(3, static_cast<int>(std::lround(0.5 * plate_half_px)));
                const auto blades = blade_finder.find(frame, rune->r_center, orbit_px, angles,
                                                      expected_area, plate_half_px, search_px);
                sources.slot_found_count = blades.count();
                for (int slot = 0; slot < 5; ++slot) {
                    sources.slot_found_centers[static_cast<std::size_t>(slot)] =
                        blades.centers[static_cast<std::size_t>(slot)];
                }
            }
            // 暗片通道（最小版）：在 5 个槽位方向量"靶面存在度"——靶面有内部纹路，
            // 边缘密度显著高于背景；亮片由网络给出，暗片靠这一条补。
            {
                std::string scores;
                int mask = 0;
                for (int slot = 0; slot < 5; ++slot) {
                    const cv::Point2f center = slot_snapshot.slot_centers[
                        static_cast<std::size_t>(slot)];
                    if (!slot_snapshot.slot_centers_valid || center.x <= 1.0f) {
                        scores += "0|";
                        continue;
                    }
                    constexpr int kHalf = 9;   // 6 m 处靶面约 14 px 半径
                    cv::Rect roi(static_cast<int>(center.x) - kHalf,
                                 static_cast<int>(center.y) - kHalf, 2 * kHalf, 2 * kHalf);
                    roi &= cv::Rect(0, 0, frame.cols, frame.rows);
                    double score = 0.0;
                    if (roi.width > 4 && roi.height > 4) {
                        cv::Mat gray;
                        cv::cvtColor(frame(roi), gray, cv::COLOR_BGR2GRAY);
                        cv::Mat lap;
                        cv::Laplacian(gray, lap, CV_32F);
                        cv::Scalar mean, stddev;
                        cv::meanStdDev(lap, mean, stddev);
                        score = stddev[0] * stddev[0];   // 拉普拉斯方差：纹理多则大
                        if (score > 300.0) mask |= (1 << slot);
                    }
                    char buffer[16];
                    std::snprintf(buffer, sizeof(buffer), "%.0f|", score);
                    scores += buffer;
                }
                sources.slot_scores = scores;
                sources.slot_present_mask = mask;
            }
            sources.sigma_dis_m = target.distanceSigma();
            sources.sigma_roll_deg = target.phaseSigma() * 180.0 / CV_PI;
            sources.orbit_ratio = orbit.semi_major_px > 1e-6
                                      ? orbit.semi_minor_px / orbit.semi_major_px
                                      : 0.0;
            sources.sim_offset_valid = telemetry_stats.sim_offset_valid;
            sources.sim_offset_s = telemetry_stats.sim_offset_s;
            sources.diff = diff_sample;
            const auto stage_csv_start = std::chrono::steady_clock::now();
            auto_aim::energy::rune_diag::writeRow(
                recorder, auto_aim::energy::rune_diag::makeRow(sources));
            profiler.add("csv", stage_ms(stage_csv_start));
        }

        const char* dump_setting = std::getenv("ULTRA_VISION_RUNE_DUMP");
        const bool dump_enabled = dump_setting != nullptr;
        // 叠加显示 / 落盘 PNG / imshow 全在诊断层（rune_diag::presentFrame）；
        // 两者都关掉时它只让出 1 ms，保持原来的帧节奏。
        auto_aim::energy::rune_diag::RuneOverlayInput overlay{
            *detector, target, solver, command, predicted,
            rune.has_value() ? &rune.value() : nullptr, config.mode, fire_enabled};
        key = rune_diag::presentFrame(frame, overlay, show_display, dump_setting, dump_enabled);

        const auto now_steady = std::chrono::steady_clock::now();
        const double elapsed = std::chrono::duration<double>(now_steady - fps_start).count();
        if (elapsed >= 1.0) {
            rune_diag::printFpsLine(frames / elapsed, *detector, target, command,
                                    dropped_since_print);
            dropped_frames += dropped_since_print;
            dropped_since_print = 0;
            profiler.flushAndReset();
            fps_start = now_steady;
            frames = 0;
        }
    }

    cv::destroyAllWindows();

    // 退出顺序：必须先停控制线程，再让它引用的对象析构。aim_bridge 是后构造的
    // （先析构），而控制线程的 provider 会回调 aim_bridge 的互斥量；不显式 join
    // 的话线程会去锁一个已经析构的 mutex（实测退出时 abort）。
    gimbal_controller.stop();

    rune_diag::printSummary(shots_fired, telemetry_stats, shots_held_for_window,
                            fallback.parkedFrames(), fallback.recenterCount(),
                            trace.stepsDeg(), trace.ratesDegPerSec());

    return 0;
}
