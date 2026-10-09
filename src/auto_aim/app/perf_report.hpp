#ifndef AUTO_AIM_APP_PERF_REPORT_HPP
#define AUTO_AIM_APP_PERF_REPORT_HPP

// 组合层：每秒性能/状态打印。从 node_sim 主循环抽出（纯输出，无副作用）。

#include <iostream>
#include <chrono>
#include <string>

#include "Kalman/tracker.hpp"
#include "app/perf_stats.hpp"
#include "control/aim_pipeline.hpp"
#include "control/target_selector.hpp"
#include "perception/armor_source.hpp"
#include "sim_receiver/vision_date_receiver.hpp"

namespace auto_aim
{
namespace app
{
        inline void reportPerSecond(PerfStats& stats, double elapsed,
                            const Tracker& tracker, ArmorSource& armor_source,
                            const TargetDecision& last_decision,
                            const std::string& shooter_status,
                            const AimPipeline& pipeline, bool prof_stages,
                            bool source_delay_enabled, double aim_jump_limit,
                            int& aim_jump_rejections,
                            const sim_receiver::VisionDateReceiver& receiver,
                            std::chrono::steady_clock::time_point now,
                            std::chrono::steady_clock::time_point& fps_start,
                            int light_count, int armor_count, int pnp_count)
        {
            const char* const kStageNames[PerfStats::kStageCount] = {
                "roi", "submit", "pnp+ekf", "csv", "control", "overlay", "resize"};
        if (elapsed >= 1.0) {
            const char* tracker_state = "LOST";
            if (tracker.getState() == auto_aim::TrackerState::DETECTING) {
                tracker_state = "DETECTING";
            } else if (tracker.getState() == auto_aim::TrackerState::TRACKING) {
                tracker_state = "TRACKING";
            } else if (tracker.getState() == auto_aim::TrackerState::TEMP_LOST) {
                tracker_state = "TEMP_LOST";
            }
            std::cout << "Processed FPS: " << stats.fps_frames / elapsed
                      << ", lights: " << light_count
                      << ", armors: " << armor_count
                      << ", PnP: " << pnp_count
                      << ", det_fps: " << (armor_source.neuralEnabled()
                             ? armor_source.neuralFps() : stats.fps_frames / elapsed)
                      << ", det_ms: " << armor_source.neuralLatencyMs()
                      << ", tracker: " << tracker_state
                      << ", fresh: " << tracker.hasObservation()
                      << ", lost: " << tracker.getLostCount()
                      << ", omega: " << tracker.getOmega()
                      << ", rate: " << tracker.getRotationRateOmega()
                      << ", dec: " << last_decision.valid
                      << ", spin: " << tracker.hasRotationRate()
                      << ", aim_settled: " << stats.aim_settled_updates << "/"
                      << stats.aim_valid_updates
                      << ", shooter: " << shooter_status
                      << ", stats.plate_switches: " << stats.plate_switches;
            {
                const auto wd = tracker.watchdogStats();
                std::cout << ", wd(reset=" << wd.resets << ", reject=" << wd.reject_ratio
                          << ", conv=" << (wd.converged ? 1 : 0) << ")"
                          << ", still=" << (pipeline.targetStill() ? 1 : 0);
            }
            std::cout << std::endl;
            if (stats.fps_frames > 0) {
                const double n = static_cast<double>(stats.fps_frames);
                std::cout << "  loop ms: wait=" << stats.prof_wait_ms / n
                          << " work=" << stats.prof_work_ms / n
                          << " display=" << stats.prof_display_ms / n << std::endl;
                if (prof_stages) {
                    std::cout << "  stage ms:";
                    double accounted = 0.0;
                    for (int stage = 0; stage < auto_aim::app::PerfStats::kStageCount; ++stage) {
                        const double value = stats.prof_stage_sum[stage] / n;
                        accounted += value;
                        std::cout << ' ' << kStageNames[stage] << '=' << value;
                    }
                    std::cout << " | 合计 " << accounted << " / work "
                              << stats.prof_work_ms / n
                              << " | pnp+ekf CPU " << stats.prof_ekf_cpu_sum / n << std::endl;
                }
            }
            // 进帧速率：自瞄需要 ≥15~20 fps；掉到个位数先看仿真器日志的
            // [tcp] encoded_fps（源头在那边），再决定是换机器跑仿真还是查网络。
            if (stats.net_frames > 0) {
                const double span = stats.net_prev_local_us > stats.net_first_local_us
                    ? (stats.net_prev_local_us - stats.net_first_local_us) / 1e6 : 0.0;
                const double net_fps = span > 1e-3
                    ? (stats.net_frames - 1) / span : static_cast<double>(stats.net_frames);
                if (!stats.source_delay_samples_us.empty()) {
                    std::vector<double> sorted = stats.source_delay_samples_us;
                    std::sort(sorted.begin(), sorted.end());
                    stats.observation_source_delay =
                        sorted[sorted.size() / 2] / 1e6;   // 秒，供下一秒的观测时间用
                    double worst = sorted.back() / 1000.0;
                    std::cout << "  source_delay: median=" << stats.observation_source_delay * 1000.0
                              << "ms max=" << worst << "ms (曝光→到手，已计入观测年龄"
                              << (source_delay_enabled ? "" : "，当前被 ULTRA_VISION_SOURCE_DELAY=0 关掉")
                              << ")" << std::endl;
                    if (source_delay_enabled &&
                        (stats.observation_source_delay <= 0.0005 || stats.observation_source_delay >= 0.120)) {
                        std::cerr << "[net] source_delay=" << stats.observation_source_delay * 1000.0
                                  << " ms 超出合理区间(0~120)，**不用于补偿** —— "
                                     "两台机器的时钟大概率没对齐（NTP 没同步/差时区）；"
                                     "先对齐时钟，或 ULTRA_VISION_SOURCE_DELAY=0 关掉这一项。"
                                  << std::endl;
                    }
                }
                stats.source_delay_samples_us.clear();
                if (aim_jump_rejections > 0) {
                    std::cout << "  aim_guard: 按住 " << aim_jump_rejections
                              << " 次（瞄点跳变超过 " << aim_jump_limit * 180.0 / CV_PI
                              << "°）" << std::endl;
                }
                aim_jump_rejections = 0;
                std::cout << "  assoc: det=" << stats.assoc_detections
                          << " nomatch=" << stats.assoc_nomatch
                          << " ambiguous=" << stats.assoc_ambiguous
                          << " scale=" << stats.assoc_scale
                          << " accepted=" << stats.assoc_accepted << "\n";
                const uint64_t dropped_now = receiver.droppedPackets();
                const uint64_t dropped_delta = dropped_now - stats.net_dropped_prev;
                stats.net_dropped_prev = dropped_now;
                std::cout << "  net: fps=" << net_fps
                          << " wait+recv+decode=" << stats.net_recv_ms / stats.net_frames << "ms"
                          << " empty=" << stats.net_empty
                          << " decode_dropped=" << dropped_delta << std::endl;
                if (net_fps < 12.0) {
                    std::cerr << "[net] 进帧只有 " << net_fps
                              << " fps（自瞄要 ≥15~20）。先看**仿真器**那边的日志："
                              << "[tcp] encoded_fps= 若是低值，瓶颈在渲染/编码（软件渲染最常见，"
                              << "把仿真器放到有 GPU 的机器上用 ULTRA_VISION_SIM_HOST 连过去）；"
                              << "若 encoded_fps 正常而这里低，就是网络/本机解码。"
                              << std::endl;
                }
            }
            stats.assoc_detections = stats.assoc_nomatch = stats.assoc_ambiguous = stats.assoc_scale = stats.assoc_accepted = 0;
            stats.net_frames = 0;
            stats.net_empty = 0;
            stats.net_recv_ms = 0.0;
            stats.net_first_local_us = 0;
            stats.net_prev_local_us = 0;
            stats.prof_wait_ms = stats.prof_work_ms = stats.prof_display_ms = 0.0;
            for (int stage = 0; stage < auto_aim::app::PerfStats::kStageCount; ++stage) stats.prof_stage_sum[stage] = 0.0;
            stats.prof_ekf_cpu_sum = 0.0;
            fps_start = now;
            stats.fps_frames = 0;
            stats.aim_valid_updates = 0;
            stats.aim_settled_updates = 0;
            stats.plate_switches = 0;
        }
}
}
}

#endif // AUTO_AIM_APP_PERF_REPORT_HPP
