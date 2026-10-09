#pragma once

namespace rm
{
    // 多装甲板联合 + 慢层收敛 + 车型切换 + 退化(Cz≈2R) + 单帧 PnP 对照。
    // 结果写 data/multi_slow.csv、data/model_switch.csv、data/pnp_compare.csv。
    void runMultiPlateExperiments();
}  // namespace rm
