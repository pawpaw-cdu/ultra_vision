# 深大 RP-26Rune 算法直接接仿真器：接入方法、坑与实测结论

> 目的：把"深大算法在我们平台上效果如何"从印象变成可复现的数据。
> 数据流按要求**直接接 `simulator_system` 仿真器**（7666 帧 / 7667 指令 / 7668 遥测），
> 不跑他们的 demo、不接 foxglove。
> 代码：`tools/rp26_sim/`（harness，约 900 行，含桩头文件与两个"只加日志"的副本）。
> 日期：2026-09-23；模型：`models/openvino/RuneDetectionModel-master/model/model-0624.onnx`。

## 0. 一句话结论

他们的算法**能在仿真器上跑通并开火**，但要先做 4 处"平台适配"；即便如此，
在同样 30 s / 小符 / `preset low` 下，命中计分次数仍低于我们现有链路
（我们 3 轮：4/0/3 次计分命中，他们 3 轮：1/0/1 次）。
**共同瓶颈是"检测流的连续性"**：他们的火控设计里有多个硬阈值
（观测间隔 0.1 s、数据寿命 0.2 s、换靶 0.3 rad、初始冷却 0.3 s）全都假设
"相机 + 网络能连续给观测"，而本平台给不出——这一点反过来印证了我们自己
路线图里"提高检测占空比"才是主线。

## 1. 为什么不能直接跑他们的工程

| 障碍 | 现象 | harness 的做法 |
|---|---|---|
| 缺 `libfoxglove.a` | 公开仓库没有预编译库，`output/app` 断链 | 影子头文件 `include/foxglove_viz/foxglove_viz.hpp`（空实现，接口同名） |
| 只用 HighGUI 看结果 | 我们不需要窗口 | 影子 `include/img_viz.hpp`，但它会把每级图像落盘（`RP26_DUMP_DIR`） |
| `std::filesystem::canonical("/proc/self/exe")` | macOS 上没这个路径，静态初始化直接 terminate | 影子 `include/json.hpp`：配置目录改成环境变量 + 编译期默认值 |
| `timetool::Timestamp(std::chrono::nanoseconds(n))` | libc++ 的 `system_clock` 是微秒，编译不过 | 影子 `include/common/power_rune_function.hpp`：`duration_cast` |
| flatbuffers 运行时缺失 | `Time_generated.h` 要 flatbuffers 头 | 影子 `include/Time_generated.h`：只提供 `foxglove::Time{sec,nsec}` |
| 插件框架（`app::Context`、`REGISTER_PLUGIN`） | 与仿真器无关 | `rp26_detector.cpp`：复制他们的 `NNDetector.cpp`，去掉插件外壳，**推理细节逐行保留** |

算法本体（`src/core/algorithm/power_rune/**`、`transform_tools`、`function`、
`ReJson`）**原样编译，不改他们的仓库**；只有 `RuneObservationRefiner.cpp`
用了"只加日志"的副本（`src/rp26_refiner_probe.cpp`，判断逻辑一行未改），
用来定位他们的描述子卡在哪。

## 2. 接入时踩到并修掉的坑

### 坑 1（最致命）：云台姿态填给他们 TFTree 的映射

仿真器的相机位姿是 `home * Ry(-yaw) * Rx(pitch)`（位置固定，只有转动），
把它换算成"相机→车系"旋转就是 `Ry(yaw) * Rx(pitch)`（车系 = home 朝向的
OpenCV 坐标系：x 右 / y 下 / z 前）。他们的 `RuneTrackerManager` 里
`vehicle_from_ecs = V`、`gimbal_from_camera = G`，且 **恰好 `G = V^T`**。

按"`R_car_from_camera = V*M*G`"的直觉去设 gimbal 旋转，实测他们读到的却是
`M*G`（差一个固定旋转）。harness 的启动自检（多组姿态数值比对）显示：

```
最大 ||他们读到的 R_car_from_camera − 期望的 Ry(yaw)Rx(pitch)|| = 7.4e-16  (PASS)
```

正确映射是 `M = Ry(yaw) * Rx(pitch) * V`。**映射写错时会得到 ~95° 的偏航偏差**
（我们第一版就是），此时任何"效果评价"都没有意义——所以这一步的数值自检
放在接入的第一位。

### 坑 2：内参/分辨率与他们的实车配置不一致

他们配置是 1440×1080、fx≈2334、带畸变；仿真器是 640×480、垂直 FOV 45°（fx=579.4）、
无畸变。harness 按实际帧尺寸现算内参并覆盖他们的全局 `CAM/DIS`；
`project.max_img_*` 只被用作 ROI 裁剪上界，置成 4096×3072 与分辨率解耦。
另外 `offset.yaw`（他们实车枪管-相机 0.5° 偏置）在仿真器里弹丸就是沿光轴发射，
所以置 0。

## 3. 量化：他们的传统链路卡在哪（配置扫描）

`tools/rp26_sim/sweep.py`（每个配置跑一次仿真，从**他们的原始配置**出发改键）：

| 配置 | 可投影比例（装甲板+灯臂+R标都可用） | 确认旋转方向 | 有目标帧 | 30 s 命中计分 |
|---|---|---|---|---|
| 原样（R−B 阈值 50，他们实车值） | **10.9%** | 0 | 0 | 0 |
| 只把掩膜阈值 50→30 | 95.7% | 1 | 14 | 0 |
| 只放宽灯臂描述子（solidity 0.45） | 30.3% | 0 | 0 | 0 |
| 只放宽长宽比容差（0.9） | 25.0% | 0 | 0 | 0 |
| 只对齐弹道（25 m/s、阻力 0.043、无马格努斯） | — | — | — | 0 |
| 掩膜+弹道 | 86..96% | 0 | 0 | 0 |
| + 观测率放宽（0.25 s / 10 帧） | 93% | 1 | 143 | 0 |
| + 数据寿命放宽（0.6 s） | 96% | 1 | 154 | 5（不计分） |
| + 开火状态机放宽 | 98% | 1 | 98 | 0 |

定位过程：

1. 用"只加日志"的 refiner 副本打出每个候选的描述子数值：
   `armor=1(usable=1,...) lightarm=?(usable=?,solidity=?,aspect=?) centerR=1`。
   **装甲板与 R 标几乎总是可用，卡的是灯臂**：仿真器上灯臂掩膜实心度 0.38~0.88
   （阈值 0.66）、长宽比 1.6~7.8（允许 [2.9, 7.1]）。
2. 掩膜阈值是主因：30 之后灯臂轮廓变完整，可投影率 11% → 96%。
3. 即使可投影率 96%，他们的相位滤波仍然锁不上：`图片相位估计器`要求
   **相邻观测 ≤0.1 s 且连续 20 帧**才能确认旋转方向；本平台一轮点灯窗口
   只有 2.5 s（`ACTIVATION_PRIMARY_TIMEOUT`），而观测间隙常有 0.5 s
   （网络不是每帧都看到点亮扇叶）→ 每次都清空缓冲。放宽到 0.25 s / 10 帧之后才锁定。
4. 锁定之后仍然不开火：`temp.data_life=0.2 s` 一到就退化为"瞄符心 + 禁止开火"，
   而本平台观测间隙 >0.2 s → 大部分时间在"回符心"。
5. 弹道模型：他们实车参数（阻力系数 0.42、马格努斯 -0.008、初速 24.5）在仿真器上
   会多抬 **+6° 俯仰**（仿真器是 25 m/s + `LinearDamping 0.05`，等效阻力系数约 0.043）。
   对齐后，开环下他们输出俯仰与图像所需角的中位差从 +5.0° 降到 **−0.25°**。
6. 开火状态机：初始冷却 0.3 s + 连发阈值 0.04 s 是给 100~200 Hz 火控 + 连续观测设计的；
   稀疏观测下每次重建目标都会重置冷却。harness 补上"100 Hz 火控线程"（他们真实部署形态）
   并把三个键改到 0.05/0.2/0.3 s 后，30 s 开火数 2 → 8~10 发。

## 4. 同平台闭环对照（30 s、小符、`tools/sim/rune_camera_low.yaml`）

| 轮次 | Ultra_Vision（现状） |  |  | RP-26（平台适配） |  |  |
|---|---|---|---|---|---|---|
| | 开火 | 命中 | 计分命中 | 开火 | 命中 | 计分命中 |
| 1 | 11 | 25 | 4 | 10 | 6 | 1 |
| 2 | 9 | 0 | 0 | 9 | 0 | 0 |
| 3 | 15 | 15 | 3 | 8 | 5 | 1 |
| 激活 | 0 | | | 0 | | |

（RP-26 用**他们原始参数**时 3 轮全部 0 发、0 命中：掩膜阈值让他们的
未激活扇叶在 90% 帧上被判为不可用，相位滤波因此永远不锁定。）

结论：

* 目标级精度不是差距来源——把平台参数对齐后，他们的瞄准/预测是合理的
  （开环俯仰中位差 −0.25°，偏航中位差 0.3~2.2°，取决于扇叶当时在哪个相位）。
* 差距来自**观测连续性**：他们的火控在观测断流时不是"滑行"，而是"回符心 + 禁火"，
  所以同一轮里能开的火比我们少，且更容易打在半路。
* 单轮方差很大（两边都有 0 命中轮），不要用单轮下结论。

## 4.1 大符（`--preset low_large --mode large`，30 s）

实测：`frames_with_detection=174`、refiner 探针 240 条、**可投影 225 条（94%）**，
但 `find_frames=0`、`fire_sent=0`、`hits=0`、`activated=0`。

即：大符卡的不是轮廓/掩膜（可投影率和大符模式无关），而是他们的大符运动拟合
`BigRunePhaseMotionFilter`：它要 `min_data_size_to_build_motion_model=100` 条观测
先建正弦相位模型、再用 LM-IRLS 迭代，且确认旋转方向有 `acceptable_time_interval_to_confirm_rotation=0.15 s`
的连续性要求（`big_phase_motion_estimate` 段）。本平台 30 s 内的观测是"一阵一阵"的，
没能进入建模型分支（该判断点没有日志，所以这一条属于**推断**，未逐项验证）。
对照：我们自己链路的大符是"能打到、激活不了"，他们则是**连目标都没建立**。

## 5. 可以引入我们项目的部分（按优先级）

1. **候选几何的轮廓精修（他们的核心思想）**：用"颜色差 → 语义分割（装甲板/灯臂/R标）
   → 距离场拟合"取代裸关键点做位姿。前提是按我们的相机重新标定掩膜阈值
   （他们的 50 在我们仿真器画面只剩 11% 可用），并且要接受它在低分辨率/暗光下不稳。
   落地方式建议：先只在"我们用关键点算出的候选"上做**验证**（装甲板轮廓是否椭圆、
   实心度是否够），把明显误检挡掉，而不是一上来就替换位姿来源。
2. **描述子阈值当成"候选可信度"**：`solidity`、长宽比、贴边判据都是便宜的可解释特征，
   可以并到我们 `rune_detector` 的评分里（我们目前只有亮点裕度 + 类别）。
3. **观测断流的处理策略**：他们的 `recover2rune_center` 与我们"丢目标停符心"是同一思路，
   但他们把"多久算丢"做成配置（0.2 s）。建议我们用**实测观测间隙分布**来定这个数
   （我们的间隙 p90 已到 0.3~0.5 s，照抄 0.2 s 会一直在回中）。
4. **开火状态机语义**：初始冷却 + 连发 + 冷却 + 换靶重置，比我们现在固定
   `fire_gap_time` 更完整；但初冷却必须按本平台观测间隔定（他们 0.3 s 对本平台太长）。
5. **不要照抄**：`data_life`、`max_data_interval`、`min_size_to_confirm_rotation`、
   `target_switch_threshold`、描述子阈值、弹道阻力系数——这些全是"他们相机+网络+枪"的
   标定值，直接搬到我们平台会同时踩上面 4 个坑。

## 6. 复现方式

```bash
# 1) 构建 harness（不动 RP-26Rune-main）
cmake -S tools/rp26_sim -B build_rp26_sim \
      -DRP26_ROOT=/Users/a/Desktop/RM_source/RP-26Rune-main \
      -DCMAKE_MODULE_PATH=/tmp/cmake_modules
cmake --build build_rp26_sim -j 4

# 2) 一键跑（自动起停仿真器，默认用"平台适配"配置）
tools/rp26_sim/run.sh --preset low --mode small --seconds 30 --out /tmp/rp26_test

# 3) 配置扫描（从他们的原始配置出发，逐个键看影响）
python3 tools/rp26_sim/sweep.py --seconds 12 --open-loop --out /tmp/rp26_sweep

# 4) 落盘他们的各级中间图（网络结果/二值掩膜/轮廓/重投影）
RP26_DUMP_DIR=/tmp/rp26_imgs RP26_DUMP_STRIDE=1 \
  tools/rp26_sim/run.sh --preset low --seconds 8 --out /tmp/rp26_dump -- --open-loop
```

产物：`rp26.csv`（逐帧姿态/下发角/像素残差/自洽检查）、`rp26.csv.ground_truth`
（仿真器 RUNE 遥测原文）、`rp26.log`（含他们的 glog 与 refiner 探针）、`sim.log`。
