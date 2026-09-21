#include "auto_buff/rune_target_selector.hpp"

#include <limits>

namespace auto_aim::energy
{
    void RuneTargetSelector::reset()
    {
        last_num_ = -1;
        last_seen_.clear();
    }

    RuneTargetSelector::Result RuneTargetSelector::select(
        const std::vector<cv::Point2f>& lit_centers)
    {
        Result result;
        const int count = static_cast<int>(lit_centers.size());
        if (count <= 0) {
            return result;   // 本帧没有亮片：不更新选择，等下一帧
        }

        int index = 0;
        int rule = 1;
        if (count == 1 || last_num_ < 0) {
            index = 0;
            rule = 1;
        } else if (count == last_num_) {
            // 规则 2：与上一帧 target 最近的那片。
            double best = std::numeric_limits<double>::max();
            for (int i = 0; i < count; ++i) {
                const double distance = cv::norm(lit_centers[static_cast<std::size_t>(i)] -
                                                 last_target_);
                if (distance < best) {
                    best = distance;
                    index = i;
                }
            }
            rule = 2;
        } else if (count == last_num_ + 1 && !last_seen_.empty()) {
            // 规则 3：新亮起的那片 = 与上一帧已见亮片的最小距离最大者。
            double best = -1.0;
            for (int i = 0; i < count; ++i) {
                double nearest = std::numeric_limits<double>::max();
                for (const cv::Point2f& seen : last_seen_) {
                    nearest = std::min(nearest,
                                       cv::norm(lit_centers[static_cast<std::size_t>(i)] - seen));
                }
                if (nearest > best) {
                    best = nearest;
                    index = i;
                }
            }
            rule = 3;
        } else {
            // 亮片数跳变（掉帧/合并/误检）：退化为"离上帧 target 最近"，保证有输出。
            double best = std::numeric_limits<double>::max();
            for (int i = 0; i < count; ++i) {
                const double distance = cv::norm(lit_centers[static_cast<std::size_t>(i)] -
                                                 last_target_);
                if (distance < best) {
                    best = distance;
                    index = i;
                }
            }
            rule = 4;
        }

        result.valid = true;
        result.index = index;
        result.center = lit_centers[static_cast<std::size_t>(index)];
        result.rule = rule;

        // 记住本帧：target 位置 + 全部亮片位置（供下帧"新片"判据）
        last_num_ = count;
        last_target_ = result.center;
        last_seen_ = lit_centers;
        return result;
    }
} // namespace auto_aim::energy
