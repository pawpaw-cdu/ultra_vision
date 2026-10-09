#ifndef AUTO_AIM_APP_PERF_STATS_HPP
#define AUTO_AIM_APP_PERF_STATS_HPP

// 组合层：主循环的统计累加器。原本是 node_sim 主函数里散落的 20 多个局部变量，
// 现在收进一个结构体，便于把主循环整体搬进 app/sim_runner（入口只留 IO）。

#include <cstdint>
#include <vector>

namespace auto_aim
{
    namespace app
    {
        struct PerfStats
        {
            // 帧率 / 分段耗时
            int fps_frames = 0;
            double prof_wait_ms = 0.0;
            double prof_work_ms = 0.0;
            double prof_display_ms = 0.0;
            static constexpr int kStageCount = 7;
            double prof_stage_sum[kStageCount] = {0.0};
            double prof_ekf_cpu_sum = 0.0;
            // 进帧速率
            int net_frames = 0;
            int net_empty = 0;
            std::uint64_t net_dropped_prev = 0;
            double net_recv_ms = 0.0;
            std::uint64_t net_first_local_us = 0;
            std::uint64_t net_prev_local_us = 0;
            // 关联层
            int assoc_detections = 0;
            int assoc_ambiguous = 0;
            int assoc_nomatch = 0;
            int assoc_scale = 0;
            int assoc_accepted = 0;
            // 曝光→到手延迟（每秒中位）
            std::vector<double> source_delay_samples_us;
            double observation_source_delay = 0.0;
            // 状态机 / 决策
            int aim_valid_updates = 0;
            int aim_settled_updates = 0;
            int plate_switches = 0;
            int stable_spin_updates = 0;
            int missing_frames = 0;
        };
    }
}

#endif // AUTO_AIM_APP_PERF_STATS_HPP
