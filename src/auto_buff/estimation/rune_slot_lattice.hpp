#ifndef AUTO_AIM_ENERGY_RUNE_SLOT_LATTICE_HPP
#define AUTO_AIM_ENERGY_RUNE_SLOT_LATTICE_HPP

// 扇叶槽位格点（1..5 号位的记账）：把"现在瞄的是第几片、哪几片已经打过"做成
// 一个自带状态的小类，与主循环解耦、可单独测试。
//
// 为什么不是"相位累加"：网络只看得到**亮起的**扇叶（最后一片亮起前机关是残缺的），
// 实测按相位累加推 ID 有 78% 不一致。这里用"单扇叶锚定 + 每次重新对齐"：
//   1) 第一次拿到 class 0（未激活=该打的那片）时，把当前相位记作 0 号槽参考；
//   2) 之后每帧用 `roll` 相对参考相位的差取整到 72° 的整数倍得到当前槽位；
//   3) **丢帧不重锚**（检出占空比只有 ~40%，重锚会让同一物理片换号）；只有
//      长时间（默认 5 s）完全没有亮片才清空"已激活"记账，编号保留。
// 已激活片用"网络类别语义"直接记账：过滤前的候选里 class 1/2 按与当前靶心的
// 相位差折算槽位偏移，标成已激活（比靠命中反馈可靠：后者有 ~0.25 s 飞行延迟）。
//
// 2026-09-24 修正（有实测依据）：一开始是"看到一次就置位、5 s 静默才清空"的
// 累计掩码。实测与仿真真值对比：掩码完全一致只有 15.6%，已激活片数 p50 我方 3
// / 真值 1（**过记**），而且闸门会挡掉的帧里 91.8% 是误挡。原因不是重复框
// （原始 class1 框数 vs 真值已激活片数就有 79% 一致），而是：
//   1) 单帧的槽位折算有 ±1 槽噪声（图像角度 vs 机关平面角度在倾斜投影下不等距），
//      一旦置位就永不撤销 ⇒ 邻居槽位被陆续污染；
//   2) 掩码在"本轮结束"时不复位（真值每一轮都从 00000 开始）。
// 现在改成：**投票阈值 + 衰减**（要连续多帧指向同一槽位才算激活）+ **回合结束复位**。

#include <array>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>

namespace auto_aim::energy
{
    class RuneSlotLattice
    {
    public:
        /// @brief 记账参数。
        struct Config
        {
            /// 连续证据累积到多少票才算"该槽位已激活"（每帧 +gain，未命中每
            /// decay_every 帧 -1）。阈值高 ⇒ 更保守、更少过记。
            int activate_votes = 4;
            int vote_gain = 2;
            int decay_every_frames = 4;
            /// 命中锚重锚的确认帧数。
            int reanchor_confirm_frames = 3;
            /// 完全没有亮片这么久 ⇒ 机关复位（清空"已激活"记账，保留编号）。
            double reset_silence_s = 5.0;
            /// 几何证据的保鲜期：某个槽位"最近这么久内没有新证据"就取消已激活。
            /// 与真值的差异：真值里一片被激活后整轮保留，但**我们的**几何证据
            /// （class 1/2 候选的偏移）会被邻居污染，只靠投票+慢衰减回收不了
            /// （实测掩码一致率因此卡在 30~50%）。自己打中的那片另算，见 strong_。
            double evidence_max_age_s = 1.5;
        };

        /// @brief 一帧的输入。
        struct Frame
        {
            bool solved = false;             // 本帧有解算结果（rune->solved）
            double roll = 0.0;               // EKF 相位（state[5]）
            int target_class = -1;           // 锁定扇叶的类别（0=未激活）
            cv::Point2f hub{0.0f, 0.0f};      // 圆心（图像坐标）
            cv::Point2f target_center{0.0f, 0.0f};  // 锁定扇叶靶心（图像坐标）
            // 过滤前的全部候选：(class_id, 靶心位置)。用于把已激活片折算到槽位。
            std::vector<std::pair<int, cv::Point2f>> candidates;
            // 5 个槽位**在图像里的预测位置**（k=0 对应当前瞄准的槽位）。有它就不必
            // 再用"候选与靶心的图像夹角 ÷72°"折算槽位：倾斜投影下 5 片在图像里的
            // 夹角并不等距（实测按角度分桶的掩码一致率封顶 ~45%），最近邻才稳。
            std::array<cv::Point2f, 5> slot_centers{};
            bool slot_centers_valid = false;
            bool scored_hit = false;         // 本帧收到计分命中反馈
            int fired_slot = -1;             // 命中瞬间开火的那个槽位
            // 本轮是否还有点亮的扇叶（来自 RuneRoundGuard）。真值里每一轮开始时
            // 掩码都归零，所以这里下降沿就是"机关复位"的信号。
            bool round_active = true;
        };

        /// @brief 记账快照（喂给火控/CSV）。
        struct Snapshot
        {
            int slot_id = -1;                        // 当前瞄准的槽位；-1 = 未锚定
            std::array<bool, 5> activated{};         // 已激活槽位
            std::array<int, 5> votes{};              // 各槽位的证据票数（诊断用）
            std::array<cv::Point2f, 5> slot_centers{};  // 本帧投影出的槽位中心
            bool slot_centers_valid = false;
            /// 本帧**锁定观测**的那片相对跟踪槽位的偏移（0~4）：观测到已激活片时
            /// 用它把观测折算回跟踪槽位（见 rune_types.hpp 的 slot_offset 说明）。
            int observed_offset = 0;              // 0~4（相对当前片）
            int observed_offset_signed = 0;       // -2~2（换片里程计用）
            bool observed_offset_valid = false;
            /// 本帧是否**确认了一次换片**（片号刚被推进）——上层据此重建估计器
            /// （参考实现：确认换片后把目标整个换到新片，而不是让滤波器慢慢漂过去）。
            bool switch_confirmed = false;
            // 供"多片同帧的像素残差更新"查询任意候选的槽位偏移用。
            cv::Point2f hub{0.0f, 0.0f};
            cv::Point2f target_center{0.0f, 0.0f};
            bool anchored() const { return slot_id >= 0; }
        };

        RuneSlotLattice() = default;
        explicit RuneSlotLattice(const Config& config, bool debug = false)
            : config_(config), debug_(debug)
        {
        }

        /// @brief 每帧更新一次。
        void update(const Frame& frame, double now);

        Snapshot snapshot() const { return snapshot_; }

        /// @brief 给任意候选像素，返回它相对**跟踪槽位**的偏移（0~4）。
        ///        @return false 表示离投影槽位都太远、无法归属（调用方应丢弃该观测）。
        bool slotOffsetForCandidate(const cv::Point2f& candidate_px, int& offset) const;
        int slotId() const { return snapshot_.slot_id; }
        const std::array<bool, 5>& activated() const { return snapshot_.activated; }

        /// @brief 清空全部记账（换模式/重开时用）。
        void reset();

    private:
        double slotStep() const;
        void clearActivation();
        void vote(std::array<int, 5>& votes, int slot, int amount) const;
        void refreshActivated() const;
        /// @brief 候选落在哪个槽位（相对当前瞄准槽位的偏移 0~4）：
        ///        优先用"投影出的 5 个槽位中心"做最近邻，没有就退回图像夹角 ÷72°。
        ///        @return false 表示离所有槽位都太远、不能作为证据（注意：偏移本身
        ///        可以是 -1，那是"后一片"= 槽位 4 —— 不能用 -1 当哨兵）。
        bool candidateSlotOffset(const Frame& frame, const cv::Point2f& candidate,
                                 double target_phase, double slot_step, int& offset) const;

        Config config_;
        bool debug_ = false;

        bool anchored_ = false;
        double reference_phase_ = 0.0;   // 锚定时的相位（诊断用）
        int blade_index_ = -1;           // 物理扇叶编号（换片事件驱动，唯一 ID 来源）
        // 命中锚重锚：用"本帧观测到的已激活片图案"去对齐 strong_（我们打中的片），
        // 纠正参考系漂移。连续 reanchor_confirm_frames 帧同一个更正才应用。
        double last_roll_ = 0.0;         // 上一帧的状态相位
        bool have_last_roll_ = false;
        int reanchor_shift_ = 0;
        int reanchor_frames_ = 0;
        void rotateBooking(int shift);   // 片号推进时，已记账的位要一起旋转
        double last_seen_time_ = 0.0;    // 最近一次"有解算结果"的时刻
        // 本帧由"换片"推出来的、刚被激活的那几片的图像位置（在**本帧**几何下）。
        std::vector<cv::Point2f> retired_centers_;
        bool have_seen_ = false;
        int decay_tick_ = 0;
        bool round_active_ = true;
        std::array<int, 5> votes_{};     // 可变的票数（snapshot 里是拷贝）
        std::array<double, 5> last_evidence_time_{};   // 每个槽位最近一次几何证据时刻
        std::array<bool, 5> strong_{};   // 自己打中的片（计分命中）：整轮保留
        double now_ = 0.0;
        mutable Snapshot snapshot_;
    };
} // namespace auto_aim::energy

#endif // AUTO_AIM_ENERGY_RUNE_SLOT_LATTICE_HPP
