#ifndef AUTO_AIM_ENERGY_RUNE_CONFIG_HPP
#define AUTO_AIM_ENERGY_RUNE_CONFIG_HPP

// 能量机关流水线的配置**数据结构**（怎么读在 rune_config.cpp）。
//
// One file (configs/buff.yaml) describes the whole pipeline: camera
// intrinsics, network, rune geometry and fire control. Geometry defaults come
// from the simulator model (0.7 m radius, 254 mm plate); a real rune has to be
// measured on site, which is why nothing is hard-coded in the code.
//
// 注意：这里不再包含 yaml-cpp —— 只有实现（rune_config.cpp）需要它，
// 其余模块只依赖 RuneConfig 这个类型。

#include <string>

#include "auto_buff/rune_aimer.hpp"
#include "auto_buff/rune_diff_probe.hpp"
#include "auto_buff/rune_detector.hpp"
#include "auto_buff/rune_solver.hpp"
#include "auto_buff/support/math.hpp"
#include "control/aim_signal_filter.hpp"
#include "control/gimbal_aimer.hpp"

namespace auto_aim::energy
{
    struct RuneConfig
    {
        bool enabled = false;
        RuneMode mode = RuneMode::Small;
        RuneDetectorConfig detector;
        RuneDiffProbeConfig diff_probe;
        RuneSolverConfig solver;
        RuneAimerConfig aimer;
        std::string simulator_config = "simulator.yaml";
        std::string model_path;
        bool visualize = true;
        int max_coast_frames = 30;
        int recenter_after_frames = 15;
        // 喂给云台轨迹生成器的目标角速度包络（度/秒）。
        double max_target_velocity_deg_s = 200.0;
        // 换靶阶跃帧是否清零速度前馈（之前为了压住超调加的；A/B 用）。
        bool zero_feedforward_on_step = true;
        // 扇叶 ID 闸门：锁定的槽位已在"已激活"集合里就不开火（见 rune_node 的说明）。
        bool fire_block_on_hit_slot = true;
        // 丢目标时把相机停在"最后已知符心"的最长时间（秒）；超过才回标定位姿兜底。
        double park_on_center_max_age_s = 1.5;
        double max_distance_jump_ratio = 0.35;
        double max_center_jump_deg = 20.0;
        double max_slew_deg_per_frame = 8.0;
        double aim_yaw_limit_deg = 50.0;
        double aim_pitch_limit_deg = 35.0;
        double fire_thresh_deg = 1.5;
        // 一轮点亮窗口：小符/大符都是 2.5 s（大符二次窗口另算 1 s）。
        // 弹丸飞行时间加上去超过窗口末端就不再开火——那发一定落在已熄灭的靶上。
        double round_window_s = 2.5;
        // 新靶刚点亮的这段时间里滤波还在换叶，先不打，避免开局那几发乱飞。
        double fire_start_delay_s = 0.15;
        auto_aim::GimbalAimConfig gimbal;
        double gimbal_command_rate_hz = 100.0;
        auto_aim::AimSignalFilterConfig aim_filter;
        // 连续多少帧看不到点亮的扇叶就认为本轮结束。
        int round_end_frames = 5;
        // 换靶判定：命中靶心在图像里整跳一个槽位（72°）以上，说明已经进入
        // 新一轮——它和"命中反馈"一起用来重置本轮窗口。
        double blade_slot_jump_px = 25.0;
        // 槽位格点静默复位时间：这么久完全没有亮片就清空"已激活"记账
        // （见 rune_slot_lattice.hpp —— 只清记账，保留格点编号）。
        double slot_lattice_reset_silence_s = 5.0;
        // "本轮结束"的判据：连续这么久没有任何解算结果才算机关复位（清空已激活
        // 记账）。不能沿用 round_guard 的 5 帧（≈165 ms）—— 检出占空比 ~80% 时
        // 它会把记账每 0.3~0.5 s 抹一次（§49 实测 42% 的帧掩码被清成 00000）。
        double slot_lattice_round_end_s = 0.8;
        // 槽位"已激活"的投票阈值：每帧证据 +2、每 4 帧衰减 1，攒到这么多票才置位。
        // 单帧的 ±1 槽折算噪声靠它压下去（实测阈值 1 ⇒ 掩码一致率 15.6%、过记到 3 片）。
        int slot_activate_votes = 4;
        // 换片时把"被换下的那片"直接记成已激活（几何证据；不依赖网络类别，也不
        // 依赖命中反馈 —— 深大模型在实拍上对已激活片根本不出框，见 §48）。
        bool slot_book_retired_on_switch = true;
        // 换片时是否把整张已激活掩码跟着旋转。槽位号 = 物理片号，换片本来就不改变
        // 它，再旋转一次属于重复补偿（单测 testLatticeRetiredSlotBooksActivation 钉住）。
        bool slot_rotate_booking_on_switch = false;
        // 命中后保持（hold）的解禁阈值：锁定靶心的图像相位跳变超过它才认为
        // 换到了另一片（参考实现的 target_switch_threshold = 0.30 rad）。
        double post_hit_switch_phase_rad = 0.30;
        // 命中后保持（hold）的最长时间：即使没有确认换片也放行，防止闸门卡死。
        double post_hit_hold_max_s = 1.2;
        // hold 解禁时是否要求"已激活片作证"（见 RuneRoundGuard::Config）。A/B 开关。
        bool post_hit_require_active_witness = false;
    };

    /// @brief 读 `configs/buff.yaml`（或任何同构文件）→ RuneConfig。
    ///        实现在 rune_config.cpp：头文件只放数据结构，方便别处只依赖类型。
    RuneConfig loadRuneConfig(const std::string& path);

} // namespace auto_aim::energy

#endif // AUTO_AIM_ENERGY_RUNE_CONFIG_HPP
