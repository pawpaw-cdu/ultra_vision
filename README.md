# Ultra Vision

RoboMaster vision project for armor auto-aim and energy-buff detection.

## Build

```bash
cmake -S . -B build
cmake --build build -j
```

Build only one module if needed:

```bash
cmake --build build --target auto_aim -j
cmake --build build --target energy_demo -j
cmake --build build --target hsv_tuner -j
```

The top-level CMake searches for OpenCV, yaml-cpp, and the bundled Daheng
Galaxy SDK. The SDK architecture defaults to `armv8` on ARM Linux and `amd64`
elsewhere; override it with:

```bash
cmake -S . -B build -DULTRA_VISION_DAHENG_ARCH=amd64
```

`auto_aim` uses the bundled Galaxy SDK on Linux by default. On macOS it uses
the TCP simulator receiver because the Galaxy SDK is Linux-only.

## Run

```bash
./build/auto_aim
./build/src/auto_buff/energy_demo
```

`auto_aim` reads configs from `ULTRA_VISION_CONFIG_DIR` when set, or from the
first command-line argument, and otherwise uses the source `configs/` directory.

### 代码结构（对齐 sp_vision：公共能力下沉，应用层只留 IO）

两个可执行文件（`auto_aim` 真机 / `auto_aim` 仿真）**共用同一套算法**，差别只在 IO 与显示：

```
src/auto_aim/perception/armor_source.{hpp,cpp}   检测前端（共用）
    · 传统灯条检测 / 神经网络检测（异步）二选一
    · 动态 ROI（跟住估计出的四块板）+ 未锁定时居中阶梯重捕
    · 每块检测做 PnP，产出可直接进估计器的 Armor
src/auto_aim/control/aim_pipeline.{hpp,cpp}      估计→选板→提前量→云台→开火（共用）
    · 内部持有 Tracker / TargetSelector / GimbalAimer / AimSignalFilter /
      GimbalController / Shooter，以及反跑飞守卫与"无目标心跳"
    · 输入一帧观测 + 观测时刻 + 该时刻相机姿态 + 弹速；输出决策/瞄稳/开火
src/auto_aim/node.cpp        真机应用层：海康相机 + 串口协议 + 显示
src/auto_aim/node_sim.cpp    仿真应用层：TCP 帧源 + 曝光→到手时间基 + 曲线/CSV + 显示
io/gimbal/ + io/serial/      下位机链路（真机）；io/camera/ 相机驱动
```

以前的坑：识别与控制只在 `node_sim` 里写了一遍，真机入口还在跑老的灯条流程 ——
识别/控制上的每个修复都得抄两遍（反跑飞守卫就漏抄过）。现在两个入口只负责
"喂帧 + 显示 + 各自时间基"，决策逻辑只有一份。

### 实车链路（下位机串口 + 云台控制 + 开火）

协议与下位机 `visual_task.h` 的 `TJ_T_t`/`TJ_R_t` 逐字段一致（= sp_vision `io/gimbal`
的 `GimbalToVision`/`VisionToGimbal`）：

| 方向 | 长度 | 内容 |
|---|---|---|
| 下位机 → 视觉 | 43 B | `'S','P'` + mode(0 空闲/1 自瞄/2 小符/3 大符) + q(wxyz) + yaw/yaw_vel/pitch/pitch_vel + bullet_speed + bullet_count + CRC16 |
| 视觉 → 下位机 | 29 B | `'S','P'` + mode(0 不控制/1 控制不开火/**2 控制且开火**) + yaw/yaw_vel/yaw_acc + pitch/pitch_vel/pitch_acc + CRC16 |

- CRC16 = RM 标准 = **CRC-16/MCRF4XX**（poly 0x1021 反射、init 0xffff、无异或输出），
  校验值 `"123456789" -> 0x6F91`，覆盖除最后两字节外的全部内容（`io/serial/crc16.cpp`）。
- **开火决策在视觉端**：下位机的 `fire_ctrl()` 只看 mode，mode=2 才真开火；
  我们这边由 `ShootEvaluator`（`control/shoot_evaluator.hpp`）判据决定。
- 只有下位机把 mode 置成"自瞄(1)"时我们才接管云台（`allow_control`），否则发 mode=0。

代码：`io/serial/`（termios 串口 + CRC）、`io/gimbal/`（协议 + 收发线程 + IMU 队列）、
`configs/serial.yaml`（设备/超时/重连/`use_imu_world_frame`）。硬件入口 `node.cpp` 用
**和仿真入口同一套** `config_loader` + `TargetSelector`/`GimbalController`/`Shooter`，
只是输入换成相机 SDK + 下位机帧。

```bash
# 目标机（有 MVS SDK 的 x86_64/aarch64）：
cmake -S . -B build_hw -DULTRA_VISION_AUTO_AIM_SIMULATOR=OFF \
      -DULTRA_VISION_USE_HIK_CAMERA=ON -DULTRA_VISION_HIK_SDK_ROOT=/opt/MVS
cmake --build build_hw -j8 --target auto_aim
./build/auto_aim            # configs/serial.yaml 里 device: auto
```

不需要硬件的自检（CI 里也在跑）：

```bash
ctest --test-dir build_sim -R gimbal      # 协议帧 + 伪终端整链路
```

**上真机前必须确认的两件事**（代码里都留了开关/打印）：
1. `configs/serial.yaml` 的 `use_imu_world_frame`：打开后估计活在 IMU 世界系（底盘小陀螺
   时不会跟着转），但**前提是下位机四元数的轴向约定和 tracker 的 `base_to_world` 一致**
   —— 转一下底盘看估计是否跟着转即可判定；不确定就先留 false。
2. 下位机 yaw/pitch 的正方向与零位要和 `Tracker::cameraToBaseRotation` 的约定一致
   （同样的零位 = 车头正前方）。状态行里的 `mcu: ... yaw= ... pitch= ...` 与画面上的
   估计框可以直接对照。

### 实车 bring-up（三步）

```bash
# 1) 下位机链路：收帧率 / CRC / 模式（--send 再下发一条 mode=1）
./build/gimbal_probe configs 8 --send

# 2) 相机 + 内参：<文件夹> 里放 15~30 张不同角度的标定板照片，直接解
./build/calibrate_camera <图片文件夹> --cols 11 --rows 8 --size 15
#    没现成照片就先采：空格存一张（只在该帧找到板时才存），q 退出后**自动接着解算**
./build/calibrate_camera ~/calib_board --live --cols 11 --rows 8 --size 15
#    结果直接写 camera_intrinsics.yaml（旧照片不会丢，重复跑会接着编号），
#    把输出粘到 configs/camera.yaml 的 hikcamera 段

# 3) 手眼（相机→云台外参）：一个进程内开相机 + 开串口 + 找标定板 + 解算
./build/hand_eye_calibrate --live
#    把标定板**静止**摆在画面里；云台 yaw 大范围扫、pitch 分组单独变（至少 8 组，
#    必须绕两个不同轴，只在一条线上扫数学上退化）
#    空格=采一组（同时打印板子重投影 rms，>1px 说明板子/内参有问题）
#    s=存盘（hand_eye.yaml）→ 粘到 configs/camera.yaml 的 hikcamera 段
```

标定板参数在 `configs/calibration.yaml`（棋盘格或圆点阵、行列数、方格边长）。
内参不准，手眼解出来的外参会把内参误差一起吸收进去，所以顺序不能反。

**为什么需要这两个外参**：相机光心相对云台旋转中心有偏转/平移，不补的话瞄点随距离
和偏角系统性偏（1° 在 7 m 处 = 12 cm），底盘或云台转动时偏差还会跟着转。
仿真里相机在光心上，外参是单位阵/零平移，所以仿真不受影响。

### 调试窗口（仿真入口）

`node_sim` 只开**一个**窗口 `Ultra Vision`：**上面**是检测/跟踪叠加图，**下面**是实时曲线
（两者同宽；曲线区高度按 window 自适应，默认整窗不超过 900 px，笔记本也放得下）。
曲线四个 panel 分别是 yaw（命令 vs 观测）、pitch（命令 vs 观测）、瞄点像素误差、
距离（观测 vs 估计）；同一 panel 的两条线共用一条纵轴、图例各占一行。
`ULTRA_VISION_CURVES=0` 可关曲线，`ULTRA_VISION_CURVE_CSV=<path>` 同时落 CSV
（配合 `tools/rune_curve.py` 之类离线重画），`ULTRA_VISION_NO_DISPLAY=1` 完全不出窗口
（跑回归/看真实帧率时用，HighGUI 每帧要 ~20 ms）。
曲线现在默认"按需采集"：只有开窗口或给了 `ULTRA_VISION_CURVE_CSV` 才采，
显式 `ULTRA_VISION_CURVES=1/0` 仍然可以强制开/关。

`ULTRA_VISION_PROF=1` 会每秒打印一次**分段耗时**（roi / submit / pnp+ekf /
csv / control / overlay / resize），并对 pnp+ekf 同时给出**线程 CPU 时间**：
墙钟远大于 CPU 时间说明是机器在抢 CPU，不是算法变慢 —— 看性能数据前先过这一关。

Both `auto_aim` entries -- the hardware camera build (`node.cpp`) and the TCP
simulator build (`node_sim.cpp`) -- share one whole-chassis `Tracker`, the same
`config_loader.hpp` parsing of `detector.yaml`/`tracker.yaml`, and the same
`visualization/projection.hpp` overlay of the estimated chassis model. The
hardware entry has no gimbal or IMU feedback yet, so it keeps the default
identity `CameraPose` and estimates in the camera frame; inject
`tracker.setCameraPose()` before `update()` to move the same estimator into the
gimbal/world frame. Frames are processed at native resolution, so the PnP
intrinsics stay consistent with the image.

### 配置放哪里（和 sp_vision 一致）

`configs/*.yaml` 只留**会随机器人 / 工况改**的值：标定值（相机内参、装甲板等效
尺寸）、模型与设备、机器人物理限制（云台速度/加速度/加加速度/俯仰范围）、
以及随战术调的增益与容差（速度阈值、延迟档、过程噪声、射击间隔/容差）。
其余"设一次就不动"的内部常数（窗口长度、NIS 门限、初始协方差、handoff 平滑、
云台跟踪增益、UV 实验参数、传统灯条检测器的整套阈值）都是 `hpp` 里的结构体
默认值：`Kalman/tracker.hpp`、`control/*.hpp`、`perception/detector.hpp`。
`config_loader.hpp` 里没写的项一律回落到那些默认值，所以想临时覆盖某个内部
参数，直接往 yaml 里写回同名键即可。

改这两边任何一处，都用 `config_dump` 核账（它打印真正加载进结构体的每个值）：

```bash
./build_sim/config_dump > /tmp/before; ...改...; ./build_sim/config_dump > /tmp/after
diff /tmp/before /tmp/after      # 有效的值必须一行不差
```

`energy_demo` accepts an optional config directory followed by a video path:

```bash
./build/src/auto_buff/energy_demo src/auto_buff/config src/auto_buff/result.mp4
```

`hsv_tuner` opens camera 0 by default, or accepts a video path as its only
argument. `Show Red` and `Show Blue` select which colors contribute to the
mask, and pressing `e` prints the tuned red/blue HSV ranges as YAML:

```bash
./build/hsv_tuner
./build/hsv_tuner test_vid_pic/buff_test.mp4
```

## Tests

```bash
cmake -S . -B build -DULTRA_VISION_BUILD_TESTS=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Two of them exist specifically to catch the failures this project has actually
hit:

* `detector_golden_test` pins both detectors on recorded frames
  (`src/auto_aim/test/assets/`). Two blocking detector bugs — a light-bar tilt
  formula that scored vertical bars as 180 degrees, and an uninitialised
  `Light::color` read by the armor pairing filter — were invisible because no
  test ever asserted on detector output. Expectations live in
  `detector_expectations.csv`; the network classes are checked exactly, the
  classical counts with "at least" semantics.
* `truth_regression_selftest` checks the metric code of
  `tools/truth_regression.py`, which turns a joint-test recording into a
  per-stage accuracy report against the simulator's own `dataset.csv`:

```bash
# with a simulator run recorded to CSVs
python3 tools/truth_regression.py --truth truth.csv \
    --observation observation.csv --estimate estimate.csv --selector selector.csv
```

It reports observation error (detector + PnP), chassis-center error (estimator)
and plate radius (geometry model), and exits non-zero when a stage exceeds its
threshold.

* `armor_ekf_uv_test` pins the UV (pixel-space reprojection) observation added in
  `Kalman/armor_ekf.hpp`: pure math, no simulator. It checks that the predicted
  armour corners project onto the same geometry the detector labels, and that the
  filter converges (position/yaw/yaw-rate) from 1 px corner noise. The A/B switch
  is `tracker.kalman.uv_observation` (default off, see `docs/uv_observation.md`);
  `tools/auto_aim_sim_test.sh --uv 0|1` runs the simulator comparison and prints
  the per-stage errors.

Note for macOS/Homebrew: `opencv` there is the current major version (5.x) and
4.x lives in the keg-only `opencv@4`. The top-level CMake prefers `opencv@4`
when it is installed, because a stale `opencv` keg can reference an ffmpeg
soname that is no longer present, in which case every OpenCV-linked binary —
tests included — refuses to start with a missing `libavformat.*.dylib`. If
neither is usable:

```bash
brew install opencv@4          # keg-only, does not disturb an existing `opencv`
cmake -S . -B build -DOpenCV_DIR=$(brew --prefix opencv@4)/lib/cmake/opencv4
```

## Simulator

The auto-aim target can use the TCP simulator instead of the Galaxy camera.
macOS enables this mode by default; Linux and explicit builds use:

```bash
cmake -S . -B build_sim -DULTRA_VISION_AUTO_AIM_SIMULATOR=ON -DULTRA_VISION_BUILD_AUTO_BUFF=OFF
cmake --build build_sim --target auto_aim -j
```

For a two-host setup, start the simulator on the rendering host with the remote
profile. It keeps the calibrated `1440x1080` capture size while limiting the
TCP stream to JPEG quality 80 and 20 FPS:

```bash
cd /path/to/simulator_system/simulator
./run_remote.sh
```

On Windows, use `run_remote.ps1` from PowerShell or `run_remote.bat` from CMD.
For a local single-host run, use `run_host.sh` instead. The profile values can
still be overridden through `DAEDALUS_CAPTURE_FPS` and
`DAEDALUS_JPEG_QUALITY`, for example:

```bash
DAEDALUS_CAPTURE_FPS=15 DAEDALUS_JPEG_QUALITY=70 ./run_remote.sh
```

Then run the receiver on the algorithm host:

```bash

cd /path/to/Ultra_Vision
./build_sim/auto_aim
```

The simulator prints a periodic `[tcp]` line with `clients`, `encoded_fps`,
`frame_avg`, `bandwidth`, `encode_avg`, and `replaced`. This distinguishes a
network/encoder bottleneck from receiver-side decoding or algorithm load. The
stream only encodes frames while at least one client is connected; when the
link cannot keep up, it replaces stale queued frames instead of dropping the
client.

The simulator endpoint is configured in `configs/simulator.yaml`. After stable
tracking is established, `node_sim` aims at the filtered visible armor plate and
sends `GIMBAL <yaw> <pitch>` to TCP port 7667. Angles are absolute radians,
with positive yaw turning right and positive pitch turning up. The pitch limits
are configured under `tracker.gimbal` in `configs/tracker.yaml`. The receiver
enters tracking mode after stable detections and sends `FIRE` to the same port
at up to 10 Hz. Press `Esc` or `e` to exit.

The same configuration section also sets the visual-servo yaw/pitch gains,
maximum angle step per processed frame, and deadbands. The command state is
updated only after a newer image frame arrives, which prevents an asynchronous
TCP command from being mistaken for a camera pose that has already been applied.
Commands use one persistent TCP connection rather than reconnecting for every
message; if that connection breaks, the receiver reconnects it automatically.

Angle calculation lives in `src/auto_aim/control/gimbal_aimer.*`. It only
converts a target position and the current absolute gimbal angle into a new
absolute angle. TCP transmission remains in
`sim_receiver::VisionDateReceiver::sendGimbalCommand()`, so the solver can be
reused with a real gimbal control backend without simulator dependencies.
