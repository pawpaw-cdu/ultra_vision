#ifndef AUTO_AIM_ENERGY_RUNE_ROUND_GUARD_HPP
#define AUTO_AIM_ENERGY_RUNE_ROUND_GUARD_HPP

// 一轮（一次点亮窗口）的记账 + 命中后保持 + 禁火窗口。
//
// 为什么要独立成类：主循环里"什么时候算新的一轮、什么时候算这片已经打过、
// 窗口还开着没有"这三件事互相耦合，而且都要在**发火前**给出唯一答案。
// 抽出来之后：
//   * 帧循环只负责构造 Frame（本帧观测）与读 Snapshot（本帧结论）；
//   * 规则集中在一处，可以单独喂序列帧做单测（不需要图像/仿真器）。
//
// 语义（与改动前的内联实现逐条等价，便于 A/B）：
//   * 看到解算结果 → 本轮开始；靶心图像位移 > blade_slot_jump_px 视为换片，
//     也重开一轮（仿真里 build_new_round 会立刻换靶，不等点亮消失）；
//   * 连续 round_end_frames 帧没有解算结果 → 本轮结束；
//   * window_open = 本轮存活 且 年龄 ≥ fire_start_delay_s 且
//     年龄 + 弹丸飞行时间 ≤ round_window_s（超窗口的弹丸一定落在已熄灭的靶上）；
//   * class_hit：同一片靶"先连续看到 class 0（未激活）、再看到 class 1/2
//     （已激活）"才算命中。仿真里未激活图案会被网络读成 class 1，缺这层条件
//     就会在每片新靶上误判一次命中；
//   * hold：收到计分命中后保持禁火，直到图像相位跳变 > post_hit_switch_phase_rad
//     才解禁（用相位而不是像素位移：符一直在转，像素位移会自己走到阈值）。

#include <cmath>
#include <limits>

#include <opencv2/core.hpp>

namespace auto_aim::energy
{
    class RuneRoundGuard
    {
    public:
        struct Config
        {
            int round_end_frames = 5;              // 连续多少帧无解算 → 本轮结束
            double blade_slot_jump_px = 25.0;      // 靶心位移超过它 → 视为换片
            double fire_start_delay_s = 0.15;      // 新靶点亮后先稳定一会儿再打
            double round_window_s = 2.5;           // 一轮点亮窗口（规则）
            double post_hit_switch_phase_rad = 0.30;  // hold 解禁的相位跳变阈值
            double phase_slot_rad = 2.0 * 3.14159265358979323846 / 5.0;  // 72° 槽位
            /// 命中后保持的**最长**时间：这段时间内即使没确认换片也放行，防止
            /// "闸门一直关着"（现场反馈过的失效模式）。<=0 表示不设上限。
            double post_hit_hold_max_s = 1.2;
            /// hold 解禁时是否要求"看到已激活片作证"：命中后机关里至少有一片是
            /// 已激活的，它应当被网络读成 class 1/2 且位置与锁定片明显不同。
            /// 看不到它，说明"锁定片读成 class 0"很可能是误判（已激活片被读成
            /// 未激活），这时不放行 —— 否则会再打一遍刚打过的片。
            bool require_active_witness = false;
        };

        /// @brief 本帧输入。
        struct Frame
        {
            bool solved = false;             // 本帧有解算结果
            cv::Point2f blade_center{0.0f, 0.0f};   // 锁定扇叶靶心（图像坐标）
            // 靶心的**图像相位**对 72° 取模（弧度）；比 EKF 的 roll 稳，不受解缠影响。
            double aim_phase = std::numeric_limits<double>::quiet_NaN();
            int blade_class = -1;            // 锁定扇叶的类别（0=未激活）
            bool scored_hit = false;         // 本帧收到计分命中反馈（遥测）
            /// 估计器确认"换到另一片了"（观测接管 / 相位整跳 > 0.35 槽位）。
            /// 这才是 hold 的正确解禁信号：符一直在转，只看相位漂移会自己走到阈值
            /// （0.30 rad ≈ 0.28 s 就放行，然后又把刚打过的片再打一遍）。
            bool blade_switched = false;
            /// 本帧"与锁定片位置明显不同、且被判为 class 1/2"的候选个数（见
            /// RuneRoundGuard::Config::require_active_witness）。
            int active_witness = 0;
        };

        /// @brief 本帧结论。
        struct Snapshot
        {
            bool blade_lit = false;          // 本轮是否还有点亮的扇叶
            double round_age_s = 1e9;        // 本轮已经过去多久（无轮次时 1e9）
            bool hold = false;               // 命中后保持：换靶确认前不打
            bool class_hit = false;          // 本帧类别命中（class0 → class1/2）
        };

        RuneRoundGuard() = default;
        explicit RuneRoundGuard(const Config& config) : config_(config) {}

        void update(const Frame& frame, double now);

        /// @brief 遥测反馈：命中即开新一轮（仿真的 build_new_round 立刻换靶）。
        void noteScoredHit(double now);

        Snapshot snapshot() const { return snapshot_; }
        const Config& config() const { return config_; }

        /// @brief 本轮窗口是否还开着（要用当帧弹丸飞行时间判定）。
        bool windowOpen(double fly_time_s) const
        {
            return snapshot_.blade_lit && snapshot_.round_age_s >= config_.fire_start_delay_s &&
                   snapshot_.round_age_s + fly_time_s <= config_.round_window_s;
        }

        void reset();

    private:
        Config config_;
        Snapshot snapshot_;
        int missing_blade_frames_ = 0;
        double round_start_ = 0.0;
        bool have_blade_position_ = false;
        cv::Point2f last_blade_position_{0.0f, 0.0f};
        bool hold_ = false;
        double hold_phase_ = std::numeric_limits<double>::quiet_NaN();
        double hold_start_time_ = 0.0;
        int class_zero_frames_ = 0;
        double class_phase_ = std::numeric_limits<double>::quiet_NaN();
    };
} // namespace auto_aim::energy

#endif // AUTO_AIM_ENERGY_RUNE_ROUND_GUARD_HPP
