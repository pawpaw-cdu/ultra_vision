# Ultra Vision

RoboMaster 视觉工程：装甲板自瞄 + 能量机关（buff）识别。

## 构建

```bash
cmake -S . -B build
cmake --build build -j
```

只编一个模块：

```bash
cmake --build build --target auto_aim -j
cmake --build build --target energy_demo -j
cmake --build build --target hsv_tuner -j
```

顶层 CMake 会去找 OpenCV、yaml-cpp，以及随仓库带的**大恒 Galaxy SDK**。SDK 架构默认：
ARM Linux 用 `armv8`，其它平台用 `amd64`；要改就显式指定：

```bash
cmake -S . -B build -DULTRA_VISION_DAHENG_ARCH=amd64
```

Linux 上 `auto_aim` 默认用随仓库的 Galaxy SDK；macOS 上因为 Galaxy SDK 只有 Linux 版，
改用 TCP 仿真接收端。

## 运行

```bash
./build/auto_aim
./build/src/auto_buff/energy_demo
```

`auto_aim` 读配置的顺序：环境变量 `ULTRA_VISION_CONFIG_DIR` → 第一个命令行参数 →
源码目录下的 `configs/`。

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

`auto_aim` 的两个入口——真机（`node.cpp`）和 TCP 仿真（`node_sim.cpp`）——共用同一套
**整车 Tracker**、同一套 `config_loader.hpp` 对 `detector.yaml`/`tracker.yaml` 的解析，
以及同一套 `visualization/projection.hpp` 的估计车体叠加显示。真机入口目前还没有云台/IMU
反馈，所以保留默认的单位阵 `CameraPose`，在**相机系**里做估计；要在云台系/世界系里估计，
在 `update()` 之前调用 `tracker.setCameraPose()` 注入即可。图像按原生分辨率处理，PnP
用的内参和画面始终一致。

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

`energy_demo` 可以跟一个配置目录 + 一个视频路径：

```bash
./build/src/auto_buff/energy_demo src/auto_buff/config src/auto_buff/result.mp4
```

`hsv_tuner` 默认打开 0 号相机，也可以只用一个视频路径当参数。界面上的 `Show Red` /
`Show Blue` 决定哪些颜色参与生成掩膜；按 `e` 会把调好的红/蓝 HSV 阈值按 YAML 打印出来：

```bash
./build/hsv_tuner
./build/hsv_tuner test_vid_pic/buff_test.mp4
```

## 测试

```bash
cmake -S . -B build -DULTRA_VISION_BUILD_TESTS=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
```

其中有两个测试是专门为**这个项目真踩过的坑**写的：

* `detector_golden_test` 把两个检测器钉死在录制帧上（`src/auto_aim/test/assets/`）。
  之前有两个卡死的检测器 bug——灯条倾斜角公式把竖直灯条算成 180°，以及 `Light::color`
  未初始化就被装甲板配对过滤器读到——之所以一直没被发现，就是因为没有任何测试对检测器
  输出做过断言。期望值放在 `detector_expectations.csv`：神经网络的类别**精确匹配**，
  传统检测的数量按"至少"比较。
* `truth_regression_selftest` 检查 `tools/truth_regression.py` 的度量代码，它把一次联调
  录制的数据，按阶段生成对仿真器自己 `dataset.csv` 的精度报告：

```bash
# 前提：仿真跑过一遍并录了 CSV
python3 tools/truth_regression.py --truth truth.csv \
    --observation observation.csv --estimate estimate.csv --selector selector.csv
```

它报告观测误差（检测器 + PnP）、车体中心误差（估计器）和板半径（几何模型），任一阶段
超出阈值就以非零码退出。

* `armor_ekf_uv_test` 钉住 `Kalman/armor_ekf.hpp` 里新增的 UV（像素域重投影）观测：
  纯数学、不需要仿真。它检查预测的装甲板角点投影回去是否和检测器标注的几何对齐，以及
  滤波器能否从 1 px 的角点噪声里收敛（位置/yaw/yaw 角速度）。A/B 开关是
  `tracker.kalman.uv_observation`（默认关，见 `docs/uv_observation.md`）；
  `tools/auto_aim_sim_test.sh --uv 0|1` 跑仿真对比并打印各阶段误差。

macOS/Homebrew 注意：那里的 `opencv` 是当前大版本（5.x），4.x 在 keg-only 的 `opencv@4`
里。顶层 CMake 在装了 `opencv@4` 时优先用它——因为旧的 `opencv` keg 可能引用一个已经不
存在的 ffmpeg soname，那种情况下所有链接 OpenCV 的程序（包括测试）都会因为找不到
`libavformat.*.dylib` 起不来。两个都用不了的话：

```bash
brew install opencv@4          # keg-only, does not disturb an existing `opencv`
cmake -S . -B build -DOpenCV_DIR=$(brew --prefix opencv@4)/lib/cmake/opencv4
```

## 仿真器

自瞄的目标源可以不用大恒相机，改走 TCP 仿真器。macOS 默认就是这个模式；Linux 或想显式
开启：

```bash
cmake -S . -B build_sim -DULTRA_VISION_AUTO_AIM_SIMULATOR=ON -DULTRA_VISION_BUILD_AUTO_BUFF=OFF
cmake --build build_sim --target auto_aim -j
```

双机模式：在渲染机上用 remote profile 起仿真器。它保持标定好的 `1440x1080` 采集尺寸，
同时把 TCP 流限制在 JPEG 质量 80、20 FPS：

```bash
cd /path/to/simulator_system/simulator
./run_remote.sh
```

Windows 上用 PowerShell 跑 `run_remote.ps1`，或 CMD 跑 `run_remote.bat`；单机本地跑用
`run_host.sh`。profile 里的数值仍可用 `DAEDALUS_CAPTURE_FPS` / `DAEDALUS_JPEG_QUALITY`
覆盖，例如：

```bash
DAEDALUS_CAPTURE_FPS=15 DAEDALUS_JPEG_QUALITY=70 ./run_remote.sh
```

然后在算法机上跑接收端：

```bash

cd /path/to/Ultra_Vision
./build_sim/auto_aim
```

仿真器会周期性打印一行 `[tcp]`，含 `clients`、`encoded_fps`、`frame_avg`、`bandwidth`、
`encode_avg`、`replaced`——用来区分"网络/编码瓶颈"和"接收端解码或算法负载"。只有至少连了
一个客户端时才会编码帧；链路跟不上时，它替换掉队列里过期的帧，而不是把客户端踢掉。

仿真端点配在 `configs/simulator.yaml`。跟踪稳定后，`node_sim` 会朝滤波后的可见装甲板
瞄准，并把 `GIMBAL <yaw> <pitch>` 发到 TCP 7667 端口。角度是**绝对弧度**，yaw 正方向
向右、pitch 正方向向上；pitch 限幅在 `configs/tracker.yaml` 的 `tracker.gimbal` 下配。
接收端在稳定检测后进入跟踪模式，并以最高 10 Hz 向同一端口发 `FIRE`。按 `Esc` 或 `e` 退出。

同一段配置里还配视觉伺服的 yaw/pitch 增益、每处理帧的最大角度步长和保护死区。命令状态只在
**有新图像帧到达之后**才更新，这样可以避免把一条异步 TCP 命令误当成"已经生效的相机姿态"。
命令走**一条常驻 TCP 连接**，不是每条消息重连；连接断了接收端会自动重连。

角度解算在 `src/auto_aim/control/gimbal_aimer.*`：它只做"目标位置 + 当前云台绝对角 →
新的绝对角"这一件事。TCP 发送仍在 `sim_receiver::VisionDateReceiver::sendGimbalCommand()`，
所以这套解算器可以不带仿真依赖地复用到真机云台后端。
