# Ultra Vision

分层重构版 RoboMaster 自瞄项目。当前版本只包含整车自瞄链路，能量机关仅保留扩展接口。

## 目录

```text
core/        时间、日志、基础类型
io/          TCP 图像输入和命令回发
perception/  装甲板检测和 PnP
estimation/  装甲板观测、EKF、Tracker、旋转拟合
selection/   目标选择和射击窗口
control/     瞄准滤波、云台轨迹、发射状态机
modules/     可插拔业务模块，目前只有能量机关接口
apps/        正常入口和调试入口
tests/       单元测试
```

## 构建

```bash
cmake -S . -B build -DULTRA_VISION_BUILD_TESTS=ON
cmake --build build -j4
ctest --test-dir build --output-on-failure
```

## 运行

```bash
./build/apps/auto_aim/ultra_vision_auto_aim configs
./build/apps/auto_aim/ultra_vision_debug configs
```

两个入口运行同一套业务逻辑。`ultra_vision_debug` 只额外启用 `DEBUG` 日志和调试 telemetry。

## 时序约定

- 所有估计、控制、滤波和命令历史时间戳均来自 `ultra_vision::StandardClock`。
- `StandardClock` 使用 `steady_clock` 测量时间，并在进程启动时与本机系统时间对齐一次。
- 模拟器帧头的源时间戳只保留为数据源元信息，不参与本机云台命令历史查询。
- 视频帧在自瞄进程内接收完成时打上本机单调时间戳，后续全链路只使用该时间戳。
