#ifndef AUTO_AIM_ENERGY_RUNE_DETECTOR_HPP
#define AUTO_AIM_ENERGY_RUNE_DETECTOR_HPP

// Energy-rune observation layer: keypoints -> fan blades -> rune centre.
//
// Ported from sp_vision_25 `tasks/auto_buff/buff_detector.{hpp,cpp}`. Only the
// currently lit blades are visible to the network; this layer decides where
// the rune centre is and which blade is the current target. It performs no
// state estimation (that is RuneTarget's job).

#include <opencv2/opencv.hpp>
#include <array>
#include <optional>
#include <string>
#include <vector>

#include "auto_buff/rune_model.hpp"
#include "auto_buff/rune_refiner.hpp"
#include "auto_buff/rune_plate_refiner.hpp"
#include "auto_buff/rune_blade_state.hpp"
#include "auto_buff/rune_types.hpp"

namespace auto_aim::energy
{
    enum class RuneTrackStatus { Track, TemporaryLost, Lost };

    struct RuneDetectorConfig
    {
        RuneModelConfig model;
        // 观测精修（轮廓几何 + 可用性闸门，来自深大 RP-26Rune 的思路，
        // 实现见 rune_refiner.hpp）。默认关闭，A/B 用。
        RuneRefinerConfig refiner;
        /// 靶面几何精修（网络 ROI → 经典角点提取；见 rune_plate_refiner.hpp）
        RunePlateRefiner::Config plate_refiner;
        /// 扇叶状态分类（经典特征；见 rune_blade_state.hpp）
        RuneBladeState::Config blade_state;
        // 画布缩放的**自适应**（= 逐帧改 model.input_pad_scale）。
        //
        // 实测（2026-09-24，§47）：五点模型能不能出框，几乎只取决于"机关在
        // 640 宽的输入里有多大"。同一个缓冲段里改缩放系数，召回率能差 4~9 倍：
        //   buff_two（7 m，轨道 ~125 px）  1.0 → 2%   0.7 → 18%
        //   buff_three（3.4 m，轨道 ~291 px）1.0 → 61%  1.6 → 81%  2.4 → 3%
        //   simulator low（6 m，轨道 ~65 px）1.0 → 73%  1.3 → 90%  2.0 → 4%
        // 三组数据的峰值都落在"网络输入里轨道半径 ≈ 50~60 px"，所以这里按
        // 上一帧量到的轨道半径反解缩放系数，把机关钉在训练尺度上 —— 远近通吃，
        // 也不用再按距离手调 pad。
        bool canvas_adaptive = false;
        double canvas_target_orbit_px = 58.0;  // 网络输入里想要的轨道半径（像素）
        double canvas_scale_min = 0.4;
        double canvas_scale_max = 2.6;
        // 还没拿到尺度（或连续丢帧超过它）时，按这个阶梯轮流试，直到重新捕获。
        int canvas_sweep_after_lost = 6;
        std::vector<double> canvas_sweep_scales{1.0, 1.5, 0.7, 2.1, 0.5, 1.8};
        // 我们自己的颜色：0=红（掩膜看 R−B）1=蓝（看 B−R）。
        int our_color = 0;
        // 相位关联选片（见下面 PhaseAssocConfig 的说明）。
        struct PhaseAssocConfig
        {
            bool enabled = true;
            double tolerance_rad = 0.6;      // 关联残差容差（约 34°）
            double rate_min_rad_s = 0.0;     // 自估角速度下限（大符模型允许掉到 0）
            double rate_max_rad_s = 2.09;    // 大符峰值角速度（模型 a+b 的上界）
            double nominal_rate_rad_s = 1.1775;
            double max_gap_s = 0.6;          // 超过它就重新捕获
            double rate_smooth = 0.35;       // 自估角速度的 EMA 权重
            // 关联判据：候选的"隐含角速度"= wrap(相位差)/dt 必须落在这个区间里
            // （对 dt 自适应，比固定相位残差更符合物理：机关转速是有界的）。
            double rate_accept_margin = 0.25; // 区间外再放宽的比例
            /// 观测整体离开预测位置时，要"连续这么多帧看到同一个新位置"才认换片。
            /// 已激活片变多以后，网络会在相邻的亮片之间**逐帧翻烧饼**（尤其是待
            /// 激活片夹在两片已激活片之间时），不确认就跟着走 = 估计器被 72° 来回
            /// 拽、瞄点在几片之间甩。真实换片是持续事件，两帧足够区分。
            int switch_confirm_frames = 2;
        };
        PhaseAssocConfig phase_assoc;
        int max_lost_frames = 20;   // sp_vision LOSE_MAX
        int gray_threshold = 100;   // used to localise the rune centre
        int dilate_size = 5;
        double center_mask_ratio = 0.8;    // mask radius / blade half-size
        double center_extrapolation = 1.4; // (arm point - plate centre) scale
        // Use every lit blade instead of only the best one. The reference
        // implementation tracks the single best blade, which is enough to keep
        // the five-slot lattice; the multi-candidate path additionally reports
        // which slots are lit, which the activation state machine needs.
        bool multi_candidate = false;
        // When the slot lattice cannot be resolved from several blades (for
        // example recorded footage where all five glow), fall back to the
        // highest-confidence blade instead of dropping the frame.
        bool fallback_single_blade = true;
        // The network fires on the dark (unlit) blade artwork too, so ranking
        // candidates by confidence alone occasionally picks a blade that is not
        // glowing - the aim then lands on the neighbouring target. Rank by how
        // bright the plate region actually is, using the network score only as
        // a tie-breaker.
        bool prefer_bright_blade = true;
        double blade_roi_ratio = 0.6;
        // 锁定滞回（借鉴 sp_vision 装甲板选板的 lock_switch_margin 与
        // Ultra_Vision 选板器的 lock_switch_margin 思路）：默认保持当前锁定的
        // 扇叶，只有另一个候选"明显更亮"且连续若干帧都更亮时才换。
        // 否则两个亮度接近的候选会来回易主，云台就表现为左右反复拉扯。
        double lock_switch_margin = 1.35;   // 亮度比阈值
        double lock_switch_delta = 12.0;    // 或亮度差阈值（0~255）
        int lock_switch_frames = 3;         // 连续多少帧确认才允许换
        int lock_max_miss = 3;              // 锁定目标丢失多少帧后重新选
        // 已锁定时：不再比亮度，只跟"离当前瞄准位置最近"的那一片；只有最近
        // 候选离锁定位置超过该像素距离（说明确实换靶了）才重新选。
        // 大符每轮有 2 片同时点亮，靠亮度比会在这两片之间反复跳。
        double lock_break_px = 60.0;
        // 五点模型给了 R 标（旋转中心）时不需要这个；旧模型/仿真域差异导致
        // R 标点不可用时，按 "靶心 + 轨道半径" 的环带找最亮方位来估圆心。
        // 比例 = 圆心到靶心距离 / 靶面点半径 = 0.70 m / 0.16 m ≈ 4.4。
        double center_orbit_ratio = 4.4;
        // 环带亮点搜索的置信门槛："小窗峰值 − 大窗均值" 超过它才算找到了 R 标。
        double hub_brightness_margin = 60.0;
        // 五点模型里靶面四点的索引顺序，对应 PnP 物点的 外→右→内→左
        // （RuneSolver::object_points_ 的顺序）。
        // 深大的语义是 0=top 1=left 2=point_R 3=right 4=bottom，绕靶心一圈
        // 就是 0→1→4→3，与物点顺序一一对应。标定方法（换模型/换相机后重测）：
        // 用 PnP 解出的圆心反投影，应当落在网络给出的 R 标点上；本机实测
        // 0,1,4,3 偏差 7~8 px（关键点噪声量级），旋转一位会偏 90 px。
        std::string plate_point_order = "0,1,4,3";
        // plate_point_order 解析结果（见 RuneDetectorConfig::plate_point_order）。
        std::vector<int> plate_point_indices{0, 1, 4, 3};
        // 只把"未激活(class 0)"的扇叶当目标（深大参考实现的语义）。关掉后
        // 任何类别的候选都能被选中/开火——仿真里网络对自绘图案的分类不可靠时
        // 用它可以兜底，但实车务必打开（否则会一直打已经打过的靶）。
        bool require_inactive_class = true;
        // 类别优先 + 亮点裕度：非 class0 候选的亮点分要超过最佳 class0 候选这么多倍，
        // 才认为"网络把点亮片判错了类"并采信它（离线实测 23% 帧存在误判）。
        double inactive_class_margin = 1.3;
        // 没有 class0（未激活）观测时，是否允许"救回"一帧：改去追一个 class!=0 的
        // 候选（旧行为，靠亮点裕度判）。
        //
        // **默认关**，因为三家参考实现都不这么做：
        //   * sp_vision（tasks/auto_buff/buff_type.cpp 的 PowerRune）：亮片数不变时
        //     目标 = 离上一帧目标最近的那片，完全不看类别；新亮一片时 = 离旧亮片
        //     最远的那片；
        //   * rm_vision_core（rune_detector_calc.cpp）：`filterInactiveFan` 在
        //     inactive 为空时直接返回 false（本帧不给目标）；active 集合只当
        //     **几何参考**给槽位打分，从不当作瞄准目标；
        //   * RP-26Rune（PowerRunePlane.cpp）：`construct_rune_correspondence` 要求
        //     "至少有一个可用的未激活扇叶"，否则整帧不做配准。
        // 实测（2026-09-24）：多片已激活时，网络会在相邻亮片之间逐帧换框，
        // 这一路"救回"就是把瞄准点拖到已激活片上的直接原因（30 s 里 >40° 的
        // 观测跳变 26 次）；而实拍上新模型**从不输出 class 1/2**（buff_three:
        // 336 帧 class0 / 0 帧 class1），所以这一路在实拍上永远无用。
        // 打开后行为回到旧版，供 A/B。
        bool rescue_missing_inactive = false;
        // "这一片可以打"的**锁存时间**（秒）：类别的/几何的可打证据只要出现过，
        // 就在这段时间内保持可打，不必**本帧**再看到一次。
        //
        // 为什么需要：4 片已激活时，网络对剩下那片的检出率掉到 ~19%（模型对
        // "已激活外观"本来就弱，§48），而开火请求是 0.7 s 节流的脉冲 ——
        // 两者几乎对不上。实测最后一片的窗口里（hold=0、2.5 s 弹道窗口内），
        // "本帧有 class0 或几何观测"只占 6~10% 的帧，2.5 s 里只打出 1~2 发，
        // 本轮随后超时失败（现场"最后一片怎么都打不出去"）。
        // 估计器在这段时间是**外推**的（小符转速恒定 60°/s），瞄点仍然在预测的
        // 扇叶上（实测瞄准点落在扇叶上而非符心的帧占 96~99%），所以放行是安全的。
        // 保护仍然在：命中后保持（hold）、弹道窗口、以及"没有解算结果就瞄符心"。
        double engageable_latch_s = 0.8;
    };

    class RuneDetector
    {
    public:
        explicit RuneDetector(const RuneDetectorConfig& config);

        // All lit blades above the threshold (small rune lights one, large two).
        // timestamp：这一帧的曝光时刻（秒）；相位关联选片用它算 dt。
        std::optional<PowerRune> detect(const cv::Mat& bgr_image, double timestamp = -1.0);
        // Only the highest-confidence blade, which is what sp_vision's default
        // path uses: it is enough to rebuild the five-slot lattice because the
        // lattice spacing is known.
        std::optional<PowerRune> detectBest(const cv::Mat& bgr_image, double timestamp = -1.0);

        RuneTrackStatus status() const { return status_; }
        int lostCount() const { return lost_; }
        // 本帧**过滤前**的全部候选（类别 + 靶心位置）。类别是语义：
        //   0 = 未激活（该打的那片，也是构建格点模型的锚点）
        //   1/2 = 已激活（已经打过的片）——它们不该被丢弃，而是用来更新
        //        "哪些槽位已激活"以及校验格点模型（现场提示）。
        struct CandidateInfo
        {
            int class_id = 0;
            cv::Point2f center{0.0f, 0.0f};
            // 五点模型的靶面四点（边中点，顺序与 RuneSolver::object_points_ 一致）
            // 与 R 标像素。多片同帧观测（含已激活片）要用它们做像素残差更新。
            std::array<cv::Point2f, 4> plate_points{};
            cv::Point2f rmark{0.0f, 0.0f};
            bool has_rmark = false;
            bool has_plate = false;
        };
        const std::vector<CandidateInfo>& lastCandidates() const { return last_candidates_; }

        // 精修诊断（CSV 用）：本帧候选数、精修判定可用的数量、被精修丢弃的数量，
        // 以及最后一次精修的灯臂实心度/长宽比与圆心差。
        struct RefineStats
        {
            int candidates = 0;
            int usable = 0;
            int dropped = 0;
            int center_fused = 0;
            double last_armor_solidity = -1.0;
            double last_armor_area_ratio = -1.0;
            double last_armor_circularity = -1.0;
            double last_arm_solidity = -1.0;
            double last_arm_aspect = -1.0;
            double last_gap_px = -1.0;
            // 靶面精修（网络 ROI → 经典提取）的统计
            int plate_refined = 0;
            int plate_refiner_tried = 0;
            double last_plate_area_ratio = -1.0;
            double last_plate_size_ratio = -1.0;
            // SCUT 式经典状态特征（在锁定片的网络 ROI 内量）：点亮像素的**面积**
            // 与占比 —— 他们就是用轮廓面积把"未激活/已激活"分开的（§46）。
            double classic_lit_area = -1.0;
            double classic_lit_ratio = -1.0;
            // 状态分类（经典）：锁定片的归一化点亮面积、灯臂点亮比例、三态
            double state_lit_ratio = -1.0;
            double state_arm_ratio = -1.0;
            int state_code = -1;
            // 归一化用的轨道半径、ROI 边长、ROI 内点亮像素数（标定阈值要用）
            double state_orbit_px = -1.0;
            double state_lit_area = -1.0;
            int state_roi_px = 0;
        };
        const RefineStats& refineStats() const { return refine_stats_; }

        // 圆心（hub）来源：0 = 网络 k2（R 标点），1 = 多片射线求交，
        // 2 = 环带亮点搜索。诊断"圆心为什么逐帧跳"用。
        enum class HubSource
        {
            RMarkKeypoint = 0,
            MultiBladeRay = 1,
            RingBrightness = 2,
        };
        HubSource hubSource() const { return hub_source_; }
        // 本帧选中扇叶的原始 k2（R 标点）与靶心，便于和 hub 对比。
        cv::Point2f lastBladeRMark() const { return last_blade_rmark_; }
        cv::Point2f lastBladeCenter() const { return last_blade_center_; }

        // 相位选靶提示：上层（状态估计器）把"这一帧预测的扇叶相位"喂进来。
        // 本帧各类别（0=未激活，1=小符已激活，2=大符已激活）的候选数量，以及
        // 当前锁定（瞄准）的那片扇叶的类别；-1 表示没有锁定。
        const std::array<int, 3>& classCounts() const { return class_counts_; }
        /// @brief 换叶后重新捕获相位关联（清掉相位/方向残留），由节点在
        ///        RuneTarget::bladeSwitched() 为真时调用。
        void resetPhaseAssociation()
        {
            assoc_valid_ = false;
            assoc_sign_ = 0;
            assoc_sign_mismatch_ = 0;
        }
        int targetClass() const { return target_class_; }
        /// @brief 本帧的目标是**按几何连续性**选出来的（不是按类别挑的）。
        ///        这时类别只当参考：开火闸门不能因为一个错标的 class 把火挡掉
        ///        （见 rune_detector.cpp 的"几何连续优先"一节）。-1 = 本帧没选。
        bool targetPickedByGeometry() const { return picked_by_geometry_; }
        // 诊断用：本帧候选数量与最亮点判据值（看"是不是把亮靶选丢了"）。
        int candidateCount() const { return candidate_count_; }
        double bestBladeBrightness() const { return best_brightness_; }
        // 本帧**选中**那片扇叶的亮点分（与 bestBladeBrightness 比较即可判断
        // "是不是锁到了暗片"：仿真里打中非点亮扇叶会直接判本轮失败）。
        double pickedBladeBrightness() const { return picked_brightness_; }
        // 本帧**最亮**候选（按亮点判据，即"点亮的那片"）的靶心位置。
        // 与 lastBladeCenter()（实际锁定的那片）对比就能判断"是不是锁到暗片"。
        cv::Point2f lastLitBladeCenter() const { return lit_center_; }
        // 相位关联诊断：本帧是否丢弃了观测、以及关联残差（度）。
        bool phaseAssocRejected() const { return assoc_rejected_; }
        double phaseAssocResidualRad() const { return assoc_residual_last_; }
        // 网络本帧最高类别分（不管过没过阈值），见 RuneModel::lastBestScore()。
        float modelBestScore() const { return model_.lastBestScore(); }
        // 本帧所有"点亮"扇叶的靶心（含未被锁定/瞄准的那几片）。大符每轮点亮
        // 两片，激活状态机需要看到全部才谈得上二次窗口。
        const std::vector<cv::Point2f>& litBladeCenters() const { return lit_centers_; }
        double latencyMs() const { return model_.latencyMs(); }
        const RuneModel& model() const { return model_; }

    private:
        // multi_blade_hub：由多片扇叶的"圆心→靶心"射线求交得到的圆心（可选，
        // 见 buildRune）。比单片的 R 标点稳，优先使用。
        cv::Point2f estimateRuneCenter(std::vector<FanBlade>& blades, const cv::Mat& bgr_image,
                                       const std::optional<cv::Point2f>& multi_blade_hub);
        std::optional<PowerRune> buildRune(std::vector<RuneModel::Object>&& results,
                                          const cv::Mat& bgr_image, double timestamp);
        /// @brief 逐帧算画布缩放（config.canvas_adaptive），让机关保持在训练尺度。
        void updateCanvasScale(const cv::Mat& bgr_image);
        void handleLost();

        RuneDetectorConfig config_;
        RuneModel model_;
        RuneRefiner refiner_;
        RunePlateRefiner plate_refiner_;
        RuneBladeState blade_state_;
        RefineStats refine_stats_;
        HubSource hub_source_ = HubSource::RMarkKeypoint;
        // 自适应画布缩放的状态：上一帧的轨道半径（像素，EMA 平滑）、
        // 重捕扫描的阶梯下标、连续丢帧数。
        double last_orbit_px_ = -1.0;
        double frame_orbit_px_ = -1.0;   // 本帧由靶面尺寸反推出的轨道半径（像素）
        double canvas_scale_ = 1.0;
        int canvas_sweep_index_ = 0;
        int lost_frames_ = 0;
        cv::Point2f last_blade_rmark_{0.0f, 0.0f};
        cv::Point2f last_blade_center_{0.0f, 0.0f};
        // 相位关联状态（不依赖 EKF）：上一次关联到的相位/时刻 + 自估角速度。
        bool assoc_valid_ = false;
        double assoc_phase_ = 0.0;
        double assoc_time_ = -1.0;
        double assoc_rate_ = 1.1775;
        // 待确认的"换片观测"：新位置 + 连续帧数（见 PhaseAssocConfig::switch_confirm_frames）
        double pending_switch_phase_ = 0.0;
        int pending_switch_frames_ = 0;
        double assoc_residual_last_ = -1.0;
        int assoc_sign_ = 0; // 旋转方向（首次关联时确定，之后必须一致）
        // 连续因"方向不符"被拒的帧数：连续多帧没有任何候选通过方向检查，说明
        // 保存的方向可能过期（换轮/换队/数据异常），清掉重新捕获，避免
        // "方向错 → 全部拒绝 → 永远锁不上"的死锁。
        int assoc_sign_mismatch_ = 0;
        // 本帧是否因为"离预测相位太远"而丢弃了观测（诊断用）。
        bool assoc_rejected_ = false;
        RuneTrackStatus status_ = RuneTrackStatus::Lost;
        int lost_ = 0;
        std::optional<PowerRune> last_powerrune_;
        std::vector<cv::Point2f> lit_centers_;
        std::array<int, 3> class_counts_{0, 0, 0};
        int target_class_ = -1;
        bool picked_by_geometry_ = false;
        int candidate_count_ = 0;
        std::vector<CandidateInfo> last_candidates_;
        double best_brightness_ = 0.0;
        double picked_brightness_ = 0.0;
        cv::Point2f lit_center_{0.0f, 0.0f};
        // 锁定状态：上一帧选中的靶心位置 + 换靶投票
        bool has_lock_ = false;
        cv::Point2f locked_center_{0.0f, 0.0f};
        int lock_miss_ = 0;
        int switch_votes_ = 0;
    };
} // namespace auto_aim::energy

#endif // AUTO_AIM_ENERGY_RUNE_DETECTOR_HPP
