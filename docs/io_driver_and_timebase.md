# io/ 驱动层审计与时间基统一（C5 / C6 / C7）

> 本轮只做**执行/驱动**层面的清点与设计，不改算法。凡涉及真机时序行为、需要上台架
> 实测的条目，都标 **NUC 验证**，不在本机盲目改。

## C5. io/ 清点：三块各独立、无重复实现

| 模块 | 文件 | 职责 | 与其他块的关系 |
|---|---|---|---|
| 相机 | `io/camera/HikCamera.{hpp,cpp}`、`GalaxyCamera.{hpp,cpp}` | 取图 + 曝光/增益，边采边丢（只留最新几帧） | 靠 `camera.name` 二选一，接口对齐 |
| 串口 | `io/serial/serial_port.{hpp,cpp}`、`io/serial/crc16.{hpp,cpp}` | POSIX termios 打开/超时读/写；CRC16(MCRF4XX) | 只被 `io/gimbal` 用 |
| 云台 | `io/gimbal/gimbal.{hpp,cpp}`、`io/gimbal/protocol.{hpp,cpp}` | 43B 收 / 29B 发、后台读线程、姿态历史队列 | 依赖串口，不依赖任何上层 |

- `io_manager.hpp` 已删除；生命周期现由各驱动对象自己的构造/析构负责（`Gimbal`
  起/停读线程，`Camera` 起/停采集线程），没有中间层假装管三者生命周期。
- 相机两个驱动**接口同名同义**（`init/close/getImage/setExposureTime/setGain/...`），
  `node.cpp` 用 `using CameraType = ...` 按编译开关切换，**没有重复的调用逻辑**。
- 未发现三块之间的重复实现：串口只被云台用；相机不碰串口；云台不碰图像。

### 发现的差异（1 处，C7 相关）

- `HikCamera` 有带**到手时间戳**的重载 `getImage(image, FrameTime&, timeout)`；
  `GalaxyCamera` **没有**——只有 `getImage(image, timeout)`。
- `node.cpp` 当前调用的是无时间戳版本，所以两条相机路径都拿不到"曝光/到手时刻"，
  只能退回处理时刻。详见 C7。

## C6. 串口/云台下发的健壮性

现状（`io/gimbal/gimbal.cpp`）：

| 能力 | 实现 | 备注 |
|---|---|---|
| 读超时 | `SerialPort::read` 用 `poll()` + `read_timeout_ms`（默认 20ms），收不满一帧就下一轮 | 读线程**永不阻塞主循环** |
| 重连 | 累计读错误 ≥ `reconnect_after_errors`（默认 3000）→ `reconnect()`，重开设备 + `flushInput()` | 覆盖 USB CDC 掉线后 `/dev/gimbal` 重新枚举 |
| CRC 计数 | `Stats::crc_errors` | 帧尾 CRC16 校验失败即丢弃该帧 |
| 超时计数 | `Stats::timeouts` | 只统计"没读到数据"，与 CRC 失败区分 |
| 收帧率 | `Stats::fps`（最近一秒） | 状态行可看 |
| 设备探测 | `resolveDevice("auto")` 按 `/dev/ttyACM* → /dev/ttyUSB*` 顺序试 | 与 `configs/serial.yaml` 的 `device: auto` 对应 |

### 下发频率 100Hz → 200/500Hz 的可行性评估（**需与电控确认协议/固件**）

- 协议 29B/帧：200Hz → 5.8KB/s，500Hz → 14.5KB/s。USB CDC(115200 占位)实际是全速
  USB，带宽不是瓶颈；瓶颈在下位机**角度环步长固定 0.003 rad/1ms ≈ 172°/s** 与
  `CV_EV.yaw_vel` 前馈的滤波。
- 视觉侧：`AimPipeline` 的控制线程按 `command_rate_hz`（`loadGimbalCommandRateHz`）
  跑；提高它只是多发心跳/控制帧，**不加算法**。但**是否有效取决于下位机每帧是否都
  采信**（若固件按固定 1ms 步长内插，多发帧不会更快）。
- 结论：这是**协议字段 + 固件行为**问题，必须与电控确认：
  1) 下位机是否每收一帧就更新角度环目标、还是按固定步长自己内插？
  2) `yaw_vel/pitch_vel` 前馈的滤波时间常数是多少（决定更快的帧率会不会被滤掉）？
  3) 500Hz 下 43B 回传 + 29B 下发的收发时序是否还有余量？
  在三个问题有答案前，**不动** `command_rate_hz` 与协议。

## C7. 时间基统一：曝光 / 姿态 / 状态推进用同一单调时钟

### 三个时刻（必须区分）

1. **曝光时刻**：光子打进传感器。观测图像对应的物理时刻。
2. **姿态时刻**：把观测从相机系映射到世界系时，用的相机（云台）姿态对应的时刻。
3. **状态推进时刻**：估计器 `now` 与控制器当前的时刻。

### 现状

| 入口 | 曝光时刻 | 姿态时刻 | 状态推进 | 统一？ |
|---|---|---|---|---|
| 仿真 `node_sim` | 帧包回传的 `frame_local_timestamp_us`（本机单调钟） | 同帧 `frame_gimbal_yaw/pitch`（仿真器回传真实姿态） | `StandardClock::secondsFromUs(frame_local_timestamp_us)` | ✅ 三者同一单调钟 |
| 真机 `node.cpp` | **缺失**：用 `StandardClock::nowUs()`（处理时刻）当帧时间戳 | `imuAt(exposure_time)`，其中 `exposure_time = now - frame_age`，而 `frame_age = now - observation_time ≈ 0` | `StandardClock::nowSeconds()` | ⚠ 三者都塌到"处理时刻" |

真机的问题链：`camera.getImage()` 的无时间戳重载 → 检测用 `nowUs()` → `observation_time`
≈ 现在 → `frame_age ≈ 0` → `imuAt(≈now)` → 姿态仍是最新值。也就是说，代码里
"按曝光时刻插值姿态"的逻辑**在真机上没有生效**，因为缺一个可信的曝光时刻。

### 目标契约（执行层，不改算法）

> **入口只负责把"曝光时刻"这一个单调时钟量喂进来；姿态插值与状态推进都用它，
> 入口不得自己再造第二个时间戳。**

`CameraPose` 已经有 `timestamp`/`timestamp_valid` 字段，`Tracker`/`AimPipeline` 也已经
按它工作。缺的只是相机层把曝光时刻交出来。

### 落地步骤（**NUC 验证**）

1. `GalaxyCamera` 补齐 `getImage(image, FrameTime&, timeout)` 重载（对齐 HikCamera）：
   时间戳在 SDK 取到 buffer 后**立刻**打，不含队列等待。
2. `node.cpp` 改用带时间戳的重载，`frame_timestamp` 一路作为
   `observation_time = StandardClock::secondsFromUs(...)` 的来源；
   `frame_input.now` 仍用 `StandardClock::nowSeconds()`（状态推进=现在）。
3. `gimbal.imuAt(exposure_time)` 直接用上面的 `frame_timestamp`，删掉
   `frame_age = now - observation_time` 这段"猜"，`frame_age_seconds` 改为
   `now - frame_timestamp`（纯诊断）。
4. `EXPOSURE→到手` 延迟与仿真一致地进入观测年龄（`source_delay`），量法同仿真。

> 本机无相机、无法验证，故只落设计 + 契约，代码改动随 A2 的真机路径一起在 NUC 上验证。

### 入口不得自造时间戳（可 lint / 可评审）

- 所有时刻都从 `common/standard_clock.hpp` 的 `StandardClock` 派生；仿真用帧包里的
  本机单调钟、真机用相机驱动的 `FrameTime`。
- `std::chrono::steady_clock::now()` 只允许出现在：(a) 统计/计时（profiling），
  (b) 明确标注为"状态推进 now"的地方。**不得**用来代表曝光时刻。
