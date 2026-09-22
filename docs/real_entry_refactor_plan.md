# 真机入口 `node.cpp` 重构：进度与后续计划（A2）

> 目标（A 组）：`node.cpp` 变薄，真机专属逻辑进 `app/`，与 `app/sim_runner` 同一套结构。
> `auto_aim_real` 需要大恒 Galaxy / 海康 MVS SDK，**本机（macOS）不编译**，只能在 NUC 上验证。

## 已完成（本机可验证部分）

1. **入口变薄**：`node.cpp` 由 1640 行 → **17 行**，只留
   `RealRunner runner(resolveConfigDir(...)); return runner.run();`。
2. **真机逻辑进组合层**：原 `main()` 整段（配置装载、相机初始化、下位机链路、世界系
   下发换算、图像闭环、轴符号探针、运行时自标定、曲线、数据集、状态行）**原样搬进**
   `app/real_runner.cpp` 的 `RealRunner::run()`（未改一行算法/参数）。
3. **本机静态验证**：`node.cpp` 与 `app/real_runner.cpp` 都能用仓库内自带的 Galaxy SDK
   头文件做 `-fsyntax-only` 通过（含 `ULTRA_VISION_USE_OPENVINO=1`）：
   ```bash
   c++ -std=c++17 -fsyntax-only -I src/auto_aim -I . -I common -I tools -I io \
       -I io/GalaxySDK/include -DULTRA_VISION_USE_OPENVINO=1 \
       $(pkg-config --cflags opencv4 yaml-cpp) \
       -DULTRA_VISION_CONFIG_DIR='"configs"' src/auto_aim/app/real_runner.cpp
   ```
   CMake、`ctest 18/18`、仿真冒烟均不受影响（真机目标本机跳过）。
4. **分层 lint**：`tools/dep_lint.sh` 现在对 `node.cpp`/`node_sim.cpp` 都判"薄入口 ✓"。

## 为什么没有把 `RealRunner::run()` 继续拆成阶段方法（下一步，NUC）

`run()` 里是**安全攸关的实时控制核心**：图像闭环的积分/微分/死区/步长限幅、权限看门狗、
世界系下发换算、轴符号探针、运行时 2×2 自标定。这些逻辑与 ~90 个跨帧状态量
（`servo_*`、`cmd_abs_*`、`aim_off_*`、`probe_*`、`image_err_*`、`sent_*` 等）以及
下发的 `pipe_send` 回调**强耦合**。把它们从 `run()` 的局部量改成 `RealRunner` 成员、
把大段状态机从内联改成方法，是"变量作用域搬迁"，**语法能过但语义只能上台架看出来**
（一个符号/清零时机写错就是实车甩飞）。本机无法运行验证，因此这一步留给 NUC。

## 后续计划（NUC 上逐步做，每步 `tools/build_real.sh` + 真机 smoke）

分三步，每步都保持"可编译 + 上台架不退化"，做完一步再动下一步：

1. **`app/image_servo.{hpp,cpp}`：把图像闭环整块内聚成一个类。**
   - 该类拥有：`servo_*` 全部状态、`probe_off_*`/`minv_*`/`probe_ready`、`aim_off_*`、
     `cmd_abs_*`、`image_err_*`、`fire_gate_blocked`、`servo_auth_*`、`servo_sign_*`、
     `ff_*`、`cmd_rate_*`。
   - 暴露两个方法：`pipeSend(...)`（原 `pipe_send` lambda，喂给 `AimPipeline`）与
     `update(...)`（原主循环里的 image-servo 段）。
   - 依赖：只吃 `io::Gimbal`、`Tracker` 的最新观测、相机内参；其余自持有。
   - 这一步做完，`run()` 里第一个大块（~400 行）消失。
2. **`app/axis_probe.{hpp,cpp}` / `app/self_calib.{hpp,cpp}`：** 把"7.1 轴符号探针"与
   "运行时响应自标定"两段独立状态机搬出去（它们只读写 `probe_*`/`minv_*` 与曲线记录）。
   这两段本来就是**诊断工具**，不参与正常控制，风险最低，可以先做。
3. **`app/real_runner` 拆成 `initialize()` + 主循环：** 把 76~671 行的配置/装配
   改成成员 + `initialize()`，`run()` 只留"取帧 → 检测 → 姿态 → 瞄准 → 图像闭环 →
   可视化 → 状态"的循环骨架（目标 ~200 行）。

三步之后，`node.cpp`（~20 行）+ `app/real_runner`（~400 行骨架）+ `app/image_servo`
（~1000 行核心）就与 `node_sim` + `app/sim_runner` 完全同构。

## 可并行的无风险项（本机可做，已做）

- 配置装载收敛进 `config_loader.hpp`/`config/`（已完成，两入口共用）。
- 曲线/叠加显示走 `app/overlay.hpp`、`visualization/curve_plotter.hpp`（已完成）。
- 时间基契约见 `docs/io_driver_and_timebase.md`（C7）。
