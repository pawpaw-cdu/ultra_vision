#ifndef AUTO_AIM_ENERGY_RUNE_DIAG_HPP
#define AUTO_AIM_ENERGY_RUNE_DIAG_HPP

// 能量机关的**诊断层**：CSV 逐帧记录、遥测（仿真真值）解析、录帧、静态试射、汇总打印。
//
// 为什么要单独一层（边界）：这些内容都是"开发期观测"，不属于能量机关算法本身
// （感知/几何/估计/瞄准/开火判据）。对照 sp_vision：他们的算法库
// (tasks/auto_buff) 里没有 CSV/录帧/试射，那些都放在独立的小入口里。
// 拆出来后：
//   * rune_runner.cpp 只做"收帧 → 感知 → 估计 → 瞄准发布 → 开火闸门"；
//   * 本文件负责所有落盘/打印，且整层可以用 ULTRA_VISION_RUNE_NO_DIAG=1 关掉；
//   * 关掉后不留任何 I/O，便于实车/性能测试。

#include <array>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "auto_buff/rune_aimer.hpp"
#include "auto_buff/rune_diff_probe.hpp"
#include "auto_buff/rune_detector.hpp"
#include "auto_buff/rune_solver.hpp"
#include "auto_buff/rune_target.hpp"
#include "auto_buff/rune_types.hpp"

namespace auto_aim::energy::rune_diag
{
    /// @brief 一帧要落盘的全部诊断量（列顺序即 CSV 列顺序，见 writeRow）。
    struct RuneDiagRow
    {
        uint64_t local_time_us = 0;
        const char* status = "UNKNOWN";
        double latency_ms = 0.0;

        bool has_rune = false;
        cv::Point2f r_center{0.0f, 0.0f};
        cv::Point2f blade_center{0.0f, 0.0f};
        // 开火闸门逐帧状态（见 RuneDiagSources 的说明）
        bool gate_hold = false;
        bool gate_window_open = false;
        bool gate_engageable = false;

        bool state_valid = false;
        double state_yaw = 0.0;      // state[0]
        double state_pitch = 0.0;    // state[2]
        double state_dis = 0.0;      // state[3]
        double measured_dis = 0.0;
        double state_center_yaw = 0.0;  // state[4]
        double state_roll = 0.0;        // state[5]
        double state_spd = 0.0;         // state[6]

        double cmd_yaw = 0.0;
        double cmd_pitch = 0.0;
        double fly_time = 0.0;
        int control = 0;
        int shoot = 0;
        int switched = 0;
        double target_yaw = 0.0;
        double target_pitch = 0.0;
        double sent_yaw = 0.0;
        double sent_pitch = 0.0;
        int class0 = 0;
        int class1 = 0;
        int class2 = 0;
        int blade_class = -1;
        /// 观测片相对跟踪槽位的偏移（-1 = 未折算，观测片即跟踪槽位）。
        int obs_slot_offset = -1;

        double aim_phase_deg = 0.0;
        double ekf_roll_deg = 0.0;
        double plate_r_px = 0.0;
        int cand_n = 0;
        double cand_bright = 0.0;
        double obs_phase_deg = 0.0;
        double pnp_roll_deg = 0.0;
        double sim_t = 0.0;
        float best_cls_score = 0.0f;
        double buff_yaw_deg = 0.0;
        int slot_id = -1;
        std::array<bool, 5> slot_activated{};
        std::array<int, 5> slot_votes{};   // 各槽位当前证据票数（投票阈值记账）
        // 5 个槽位在图像里的预测位置（k=0 = 当前瞄准槽位），用于离线核对归属。
        std::array<cv::Point2f, 5> slot_centers{};
        // 回转椭圆拟合（测距用）与靶面压扁程度：长尾误差的归因量（见 §13.3）。
        int orbit_ok = 0;
        double orbit_semi_major_px = 0.0;
        cv::Point2f orbit_center{0.0f, 0.0f};
        int orbit_samples = 0;
        double orbit_ratio = 0.0;
        double sigma_dis_m = 0.0;
        double sigma_roll_deg = 0.0;
        double plate_aspect = 1.0;
        std::string slot_scores;   // 5 个槽位方向的"靶面存在度"（暗片通道诊断）
        int slot_present_mask = 0;
        int slot_found_count = 0;  // 暗片通道实测到的靶面数
        std::array<cv::Point2f, 5> slot_found_centers{};

        cv::Point2f aim_pixel{0.0f, 0.0f};
        double aim_world_yaw_deg = 0.0;
        double aim_world_pitch_deg = 0.0;
        double aim_roll_deg = 0.0;

        int refine_n = 0;
        int refine_usable = 0;
        int refine_dropped = 0;
        int refine_fused = 0;
        double state_lit_ratio = -1.0;    // 经典状态分类：归一化点亮面积
        double state_arm_ratio = -1.0;    // 灯臂点亮比例
        int state_code = -1;              // 1=未点亮 2=未激活 3=已激活
        double state_orbit_px = -1.0;     // 归一化用的轨道半径（像素）
        double state_lit_area = -1.0;     // ROI 内点亮像素数
        int state_roi_px = 0;             // ROI 边长（像素）
        double classic_lit_area = -1.0;   // SCUT 式经典特征：点亮像素面积
        double classic_lit_ratio = -1.0;  // 点亮像素占比
        int plate_refined = 0;             // 靶面精修是否采信（网络 ROI → 经典提取）
        double plate_size_ratio = -1.0;    // 精修后面积 / 网络面积
        double armor_solidity = -1.0;      // 装甲板轮廓实心度（精修）
        double armor_area_ratio = -1.0;    // 装甲板面积 / 网络四点面积
        double armor_circularity = -1.0;   // 4πA/P²
        double refine_arm_solidity = -1.0;
        double refine_arm_aspect = -1.0;
        double refine_gap_px = -1.0;

        cv::Point2f k2_px{0.0f, 0.0f};
        int hub_src = 0;
        double hub_plate_dist_px = 0.0;
        int target_valid = 0;
        int blade_switch_obs = 0;
        double picked_bright = 0.0;
        int assoc_reject = 0;
        double assoc_residual_deg = -1.0;
        cv::Point2f lit_px{0.0f, 0.0f};
        cv::Point2f picked_px{0.0f, 0.0f};
        int fired = 0;
        double aim_err_yaw_deg = -1.0;
        double aim_err_pitch_deg = -1.0;

        double diff_ok = 0.0;
        double diff_px = 0.0;
        double diff_py = 0.0;
        double diff_area = 0.0;
        double diff_to_lit_px = -1.0;

        // 过滤前的全部候选，编码成 "class:x:y|class:x:y|…"（整数像素）。
        // 离线分析用：判断"同一片被几个框重复检出"、"已激活片落在哪个槽位"，
        // 只看 class0/1/2 的计数是不足以定位这类问题的。
        std::string candidates;
    };

    /// @brief 仿真遥测（真值）统计。
    struct RuneTelemetry
    {
        int hits = 0;            // 命中事件（含非计分）
        int scored = 0;          // 计分命中（点亮且未打过）
        int activations = 0;     // 激活事件
        double first_activation_s = -1.0;
        bool sim_offset_valid = false;
        double sim_offset_s = 0.0;   // 本地时刻 - 仿真时刻
    };

    /// @brief 静态瞄点试射（判决实验）：固定角度 + 固定间隔开火。
    struct BoreSight
    {
        bool enabled = false;
        double yaw = 0.0;
        double pitch = 0.0;
        double interval_s = 0.5;
        double last_shot_time = -1e9;

        static BoreSight fromEnvironment();
        /// @brief 覆盖指令：固定角 + 按间隔开火（绕过一切目标相关闸门），
        ///        并在开火时更新内部节流时间。
        void apply(RuneCommand& command, double now);
    };

    /// @brief 帧差探针（差分法）的诊断结果，对应 CSV 的 diff_* 列。
    struct DiffSample
    {
        double ok = 0.0;
        double px = 0.0;
        double py = 0.0;
        double area = 0.0;
        double to_lit_px = -1.0;
    };

    /// @brief 跟踪状态的可读名（CSV status 列与终端打印共用）。
    const char* runeStatusName(RuneTrackStatus status);

    /// @brief 组装一帧 CSV 行所需的**来源**（只读引用；rune 可能为空）。
    ///        行的搬运逻辑集中在诊断层，帧循环只负责把来源填进来。
    struct RuneDiagSources
    {
        const RuneDetector& detector;
        const RuneTarget& target;
        const PowerRune* rune = nullptr;   // 本帧观测（没有就是 nullptr）
        const RuneCommand& command;

        uint64_t local_time_us = 0;
        double sent_yaw = 0.0;             // 轨迹生成器实际下发的角
        double sent_pitch = 0.0;
        double aim_phase_deg = 0.0;        // 锁定扇叶的图像相位（0~360）
        double aim_err_yaw_deg = -1.0;     // 到位误差（-1 = 无快照）
        double aim_err_pitch_deg = -1.0;
        bool fired = false;
        // 开火闸门的三个"可打性"条件（逐帧写 CSV）：配合本轮的弹道窗口，
        // 才能看出"这一片为什么没打出去"（§51）。
        bool gate_hold = false;            // 命中后保持中
        bool gate_window_open = false;     // 弹道窗口还开着
        bool gate_engageable = false;      // 类别/槽位判定"这片可以打"
        std::string slot_scores;           // 5 个槽位方向的"靶面存在度"（暗片通道）
        int slot_present_mask = 0;
        int slot_found_count = 0;          // 实测到的靶面数（暗片通道）
        std::array<cv::Point2f, 5> slot_found_centers{};
        cv::Point2f aim_pixel{0.0f, 0.0f};
        Eigen::Vector3d aim_world = Eigen::Vector3d::Zero();
        double aim_roll_deg = 0.0;
        int slot_id = -1;
        std::array<bool, 5> slot_activated{};
        std::array<int, 5> slot_votes{};
        std::array<cv::Point2f, 5> slot_centers{};
        bool orbit_valid = false;
        double orbit_semi_major_px = 0.0;
        cv::Point2f orbit_center{0.0f, 0.0f};
        int orbit_samples = 0;
        double orbit_ratio = 0.0;
        double sigma_dis_m = 0.0;
        double sigma_roll_deg = 0.0;
        bool sim_offset_valid = false;
        double sim_offset_s = 0.0;
        DiffSample diff;
    };

    RuneDiagRow makeRow(const RuneDiagSources& sources);

    /// @brief 打开逐帧 CSV 并写表头。
    bool openRecorder(std::ofstream& recorder, const std::string& path);
    void writeRow(std::ofstream& recorder, const RuneDiagRow& row);

    /// @brief 跑一次帧差探针：圆心/轨道半径由调用方按状态投影给出，
    ///        lit_px 是当前"最亮候选"（点亮片）位置，用于算 diff_to_lit。
    DiffSample probeDiff(RuneDiffProbe& probe, const cv::Mat& frame, cv::Point2f hub_px,
                         double orbit_px, cv::Point2f lit_px);

    /// @brief 解析遥测行：更新统计、可选写 ground truth 原始行。
    void consumeTelemetry(const std::vector<std::string>& lines, std::ofstream* ground_truth,
                          const std::string& rune_mode_name, RuneTelemetry& telemetry,
                          bool* scored_hit_this_frame = nullptr);

    /// @brief 按 ULTRA_VISION_RUNE_FRAME_DIR/STRIDE 落盘帧（stride<=0 不落盘）。
    void dumpFrame(const std::string& frame_dir, const cv::Mat& frame, int frame_index, int& saved);

    // ---- 打印 / 叠加显示（开发期观测，全部集中在这一层）---------------------

    /// @brief 是否打开逐帧调试打印（ULTRA_VISION_RUNE_DEBUG=1）。
    bool debugEnabled();

    /// @brief [frame] 逐帧状态轨迹：排查"跟丢/云台跑偏"只能靠它，角度一律输出度。
    struct RuneFrameDebug
    {
        int frame_index = 0;
        const RuneDetector& detector;
        const PowerRune* rune = nullptr;      // 本帧观测（可空）
        double commanded_yaw = 0.0;           // 轨迹生成器当前姿态
        double commanded_pitch = 0.0;
        const RuneCommand& command;
    };
    void printFrameDebug(const RuneFrameDebug& debug);

    /// @brief [fire] 开火闸门逐帧状态：被哪个条件挡住是调"每轮命中节奏"的关键。
    struct RuneFireGateDebug
    {
        bool shoot = false;
        bool switched = false;
        bool aim_ready = false;
        bool hold = false;
        bool engageable = false;
        bool window_open = false;
    };
    void printFireGateDebug(const RuneFireGateDebug& gate);

    /// @brief [fire suppressed]：干跑（ULTRA_VISION_DISABLE_FIRE）时的发火意图。
    void printFireSuppressed(const RuneCommand& command);

    /// @brief 每秒一行的 FPS / 推理延迟 / 跟踪摘要（脚本靠它抓吞吐）。
    /// @param skipped_frames 本统计周期内被跳过的仿真帧数（帧序号差 - 1 累计）。
    void printFpsLine(double fps, const RuneDetector& detector, const RuneTarget& target,
                      const RuneCommand& command, long long skipped_frames = 0);

    /// @brief 逐帧阶段耗时统计（`ULTRA_VISION_RUNE_PROFILE=1` 打开后随 FPS 行打印）。
    ///        用途：帧循环里 `nn_ms` 只算了推理，剩下的时间花在哪一层要能量出来，
    ///        否则"提帧率"只能是猜（实测推理 16.5 ms 而帧周期 ~45 ms）。
    class StageProfiler
    {
    public:
        explicit StageProfiler(bool enabled) : enabled_(enabled) {}

        bool enabled() const { return enabled_; }
        /// @brief 累加一个阶段的耗时（毫秒）。
        void add(const char* stage, double milliseconds);
        /// @brief 打印并清空（每个统计周期调用一次）。
        void flushAndReset() const;

    private:
        bool enabled_ = false;
        mutable std::vector<std::pair<std::string, std::pair<double, int>>> stages_;
    };

    /// @brief 真正下发出去的云台指令轨迹：帧级 CSV 看不到 100 Hz 的细节，
    ///        平滑度（步长/角速度分布）靠它统计，喂给 printSummary。
    class GimbalTrace
    {
    public:
        /// @brief 记录一条下发指令（由轨迹生成器的下发回调调用）。
        void record(double yaw, double pitch, double now);

        const std::vector<double>& stepsDeg() const { return steps_deg_; }
        const std::vector<double>& ratesDegPerSec() const { return rates_deg_s_; }
        bool empty() const { return steps_deg_.empty(); }

    private:
        std::vector<double> steps_deg_;
        std::vector<double> rates_deg_s_;
        double last_yaw_ = 0.0;
        double last_pitch_ = 0.0;
        double last_time_ = 0.0;
        bool valid_ = false;
    };

    /// @brief 叠加显示所需的来源。
    struct RuneOverlayInput
    {
        const RuneDetector& detector;
        const RuneTarget& target;
        const RuneSolver& solver;
        const RuneCommand& command;
        const RuneTarget& predicted;        // 帧循环推进过的副本（含命中时刻预测）
        const PowerRune* observation = nullptr;
        RuneMode mode = RuneMode::Small;
        bool fire_enabled = true;
    };

    /// @brief 叠加显示一帧（观测黄框 + 预测绿/紫框 + 状态文本），按
    ///        ULTRA_VISION_RUNE_DUMP=N 落盘 PNG，按 show_display 决定 imshow。
    ///        @return waitKey 结果（没有窗口时返回 1）。
    int presentFrame(const cv::Mat& frame, const RuneOverlayInput& overlay, bool show_display,
                     const char* dump_setting, bool dump_enabled);

    /// @brief 汇总打印（与旧版逐字保持一致，方便脚本抓取）。
    void printSummary(int shots_fired, const RuneTelemetry& telemetry, int shots_held_for_window,
                      int parked_frames, int recenter_count,
                      const std::vector<double>& gimbal_steps_deg,
                      const std::vector<double>& gimbal_rates_deg_s);
} // namespace auto_aim::energy::rune_diag

#endif // AUTO_AIM_ENERGY_RUNE_DIAG_HPP
