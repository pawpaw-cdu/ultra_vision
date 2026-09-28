#ifndef AUTO_AIM_ENERGY_RUNE_TARGET_SELECTOR_HPP
#define AUTO_AIM_ENERGY_RUNE_TARGET_SELECTOR_HPP

// 选 target（该打哪片）：照搬 sp_vision_25 `tasks/auto_buff/buff_type.cpp` 里
// `PowerRune` 构造的三条规则，只用"亮片数 + 像素位置"，**不依赖网络类别**：
//   1) 只检到 1 片 ⇒ 它就是 target；
//   2) 亮片数与上一帧相同 ⇒ 取**离上一帧 target 最近**的那片（连续性）；
//   3) 亮片数 = 上一帧 + 1（有新片亮起）⇒ 取**与上一帧已见亮片的最小距离最大**的那片
//      （= 新出现的那片）。
// 为什么不用类别：实测网络对"已激活/未激活"的判别在关键时刻（class0 可用率 17~46%）
// 不可靠；sp_vision 全程只用几何，反而稳。

#include <vector>

#include <opencv2/core.hpp>

namespace auto_aim::energy
{
    class RuneTargetSelector
    {
    public:
        struct Result
        {
            bool valid = false;
            int index = -1;            // 选中的亮片索引
            cv::Point2f center{0.0f, 0.0f};
            int rule = 0;              // 1=单帧 2=最近 3=新片（诊断用）
        };

        /// @brief 选 target。@param lit_centers 本帧检出的亮片靶心（像素）
        Result select(const std::vector<cv::Point2f>& lit_centers);
        void reset();

    private:
        int last_num_ = -1;
        cv::Point2f last_target_{0.0f, 0.0f};
        std::vector<cv::Point2f> last_seen_;
    };
} // namespace auto_aim::energy

#endif // AUTO_AIM_ENERGY_RUNE_TARGET_SELECTOR_HPP
