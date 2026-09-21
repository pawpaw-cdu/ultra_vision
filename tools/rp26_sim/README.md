# tools/rp26_sim —— 深大 RP-26Rune 算法直接接仿真器

把 `RP-26Rune-main` 的**算法本体**（检测、轮廓精修、平面/相位估计、火控决策）
接到 `simulator_system/simulator` 的实时数据流上，用于"同平台对照"与"误差来源定位"。

```
daedalus ──7666 TCP 帧──▶ rp26_sim ──(他们的检测+算法)──▶ 目标 yaw/pitch
         ◀──7667 GIMBAL/FIRE/RESET──┘
         ──7668 遥测 RUNE 事件──▶ 打分（命中/计分/激活）
```

## 目录

| 文件 | 作用 |
|---|---|
| `src/main.cpp` | harness 主程序：收帧、填 TF、调用他们算法、下发/开火、写 CSV |
| `src/rp26_detector.cpp/.hpp` | 复制自他们的 `NNDetector.cpp`，去掉插件外壳；推理细节逐行保留 |
| `src/rp26_refiner_probe.cpp` | 复制自他们的 `RuneObservationRefiner.cpp`，**只加日志**（描述子数值探针） |
| `src/sim_link.cpp/.hpp` | 7666/7667/7668 协议（与仓库现有 `sim_receiver` 同协议） |
| `include/*` | 影子头文件：`json.hpp`、`PnPVariable.hpp`、`img_viz.hpp`、`foxglove_viz/`、`Time_generated.h`、`common/power_rune_function.hpp` |
| `config/power_rune.json` | **平台适配版**（仿真器相机/弹道/观测率/开火阈值，每处都有 `_sim_adapt` 说明） |
| `config/power_rune.rp26_origin.json` | 他们仓库原始配置的副本（扫描脚本以它为 baseline） |
| `config/rp26_sim.json` | 五点模型检测参数（格式同他们的 `detect.json`） |
| `run.sh` | 一键：起仿真器 → 跑 harness → 收结果 |
| `sweep.py` | 配置扫描：逐个键看"可投影率 / 锁定 / 有目标帧 / 开火命中" |

## 用法

```bash
cmake -S tools/rp26_sim -B build_rp26_sim \
      -DRP26_ROOT=/Users/a/Desktop/RM_source/RP-26Rune-main \
      -DCMAKE_MODULE_PATH=/tmp/cmake_modules
cmake --build build_rp26_sim -j 4

tools/rp26_sim/run.sh --preset low --mode small --seconds 30 --out /tmp/rp26_test
tools/rp26_sim/run.sh --preset low_large --mode large --seconds 30 --out /tmp/rp26_large
tools/rp26_sim/run.sh --preset low --seconds 8 --out /tmp/rp26_dump -- --open-loop   # 只看不控
python3 tools/rp26_sim/sweep.py --seconds 12 --open-loop --out /tmp/rp26_sweep
```

harness 参数（`--` 之后透传）：

| 参数 | 说明 |
|---|---|
| `--mode small\|large` | 大小符（决定类别映射与他们的决策配置） |
| `--open-loop` | 不下发他们的角、不开火，只记录（用于标定/对照） |
| `--control-hz N` | 火控线程频率（默认 100 Hz，0 = 逐帧调用 `get_rune_data`） |
| `--yaw-sign/--pitch-sign/--yaw-offset-deg/--pitch-offset-deg` | 输出角到仿真器指令的映射（默认恒等，因为映射已在启动自检里验证） |
| `--fire-interval` | 我们的开火限速（仿真器弹丸冷却 0.05 s） |

## 环境变量

| 变量 | 作用 |
|---|---|
| `RP26_CONFIG_DIR` | 配置目录（默认 `tools/rp26_sim/config`） |
| `RP26_MODEL_PATH` | 覆盖模型路径（默认按配置目录相对解析，兜底到 `models/openvino/...`） |
| `RP26_DUMP_DIR` / `RP26_DUMP_STRIDE` | 落盘他们的中间图（NN结果/二值掩膜/轮廓/重投影），按帧号命名 |

## 注意事项

1. 启动自检必须 PASS：`TFTree 姿态映射自检 max ||actual-expected|| ≈ 0`。
   搞错这一步会得到 ~95° 的偏航偏差，结论会完全错。
2. 内参按**实际帧尺寸**现算并覆盖他们的 `CAM/DIS`；`--width/--height` 改仿真器分辨率时不需要改配置。
3. `config/power_rune.json` 是**平台适配版**：掩膜阈值、弹道、观测率、数据寿命、
   开火阈值都按本平台改过，每处都写了 `_sim_adapt` 与原始值对照。
   想跑他们的原参数：`RP26_CONFIG_DIR=<自己> ` 或看 `sweep.py` 的 baseline。
4. 单轮方差很大（两边都会出现 0 命中轮），结论至少看 3 轮。

详细分析见 `docs/energy_rune_rp26_sim.md`。
