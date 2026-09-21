#pragma once
//
// 让影子头文件（img_viz.hpp）里的落盘命名带上"当前仿真帧号"，
// 这样同一帧的 NN结果/二值图/轮廓/重投影能够互相对应，也能和 CSV 对齐。

#include <atomic>

namespace rp26_sim
{
inline std::atomic<long long> g_current_frame_index{0};
} // namespace rp26_sim
