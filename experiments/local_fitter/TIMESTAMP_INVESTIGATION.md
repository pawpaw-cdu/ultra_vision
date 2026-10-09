# 时间戳排查：那 ~70 ms 到底在哪一段

## 结论

1. **视觉端没有把时间戳用错**。`runAim()` 用的是
   `observation_time = 本机到手时刻 − source_delay`，也就是**把曝光时刻还原出来**，
   不是直接用"收到帧的时刻"。
2. 两端的戳是**自洽**的：实测 `本机戳 − 仿真源戳` = **36.0 ms @1440x1080**、
   **17.0 ms @720x540**；仿真器自己报的 `pipeline_avg` 是 35.7 / 17.2 ms
   （其中 JPEG 编码 16.3 / 5.2 ms）。差值就是本机环回的网络+读包开销 ≈ 0。
3. **但图像内容比仿真端的曝光戳还老 ~34 ms**。这段在**仿真器内部**，
   视觉端把时间戳用对也消不掉。

也就是说：70 ms 里 ~36 ms 是"仿真→视觉"的传输（已被主链路补偿），
剩下 **~34 ms 是仿真器自己"打的戳 ↔ 像里的位姿"对不上**。

---

## 1. 三个时间戳分别在哪打的

| 戳 | 位置 | 文件 | 说明 |
|---|---|---|---|
| **仿真曝光戳** `source_timestamp_us` | 仿真器 Render 调度里、**GPU 回读之前** | `simulator/src/tcp.rs`，`capture_timestamp_us = SystemTime::now()`，写进 40 字节包头 | 注释说明是从"JPEG 编码之后"挪到"回读之前"的 |
| **本机到手戳** `local_timestamp_us` | 视觉端**把整包 payload 读完**之后 | `sim_receiver/src/tcp_input.cpp`，`StandardClock::nowUs()` | `StandardClock` = steady_clock 锚定到进程启动时的系统钟，不会随 NTP 漂 |
| **观测时刻** `observation_time` | 入口层 | `app/sim_runner.cpp` `runAim()`：`source_result_.timestamp_us − median(source_delay)` | 异步检测结果带的是"它真正处理的那一帧"的本机戳，再减掉跑动中位数 |

## 2. 实测

两次运行只改采集尺寸（其它一致：1.0 m、冻结云台、3 rad/s）：

| | 1440x1080 | 720x540 |
|---|---|---|
| 仿真实际出帧 | 16~17 fps（帧周期 ≈ 60~62 ms） | 21 fps（47.6 ms） |
| 仿真 `encode_avg` | 16.3 ms | 5.2 ms |
| 仿真 `pipeline_avg`（曝光→发出） | 35.7 ms | 17.2 ms |
| 视觉端 `source_delay`（本机−源） | **36.0 ms** | **17.0 ms** |
| **图像内容相对本机戳的最优偏移** | **−70 ms** | **−50 ms** |
| 该偏移下的角点残差 | 12.9 px | 6.0 px |
| 不修正（Δ=0）时的角点残差 | 60.4 px | 21.0 px |

角点残差 = "检测出的 4 个角点"对比"真值四板几何的投影"，扫描 Δ 得到；
曲线是干净单峰 V 形，四块板独立求最优 Δ 全部落在同一处。
另一条**不依赖我的几何模型**的复核（4 点 PnP 的 tvec 对真值板心）给出同样的
−70 ~ −80 ms。

## 3. 那 34 ms 是什么

| | 帧周期 | 内容滞后（总） | 其中传输 | **剩下（仿真内部）** |
|---|---|---|---|---|
| 1440x1080 | 62.5 ms | 70 ms | 36 ms | **34 ms** |
| 720x540 | 47.6 ms | 50 ms | 17 ms | **33 ms** |

帧率差了 1.3 倍，剩下这一段却**基本不变（33~34 ms）**——所以它**不是"整整一帧渲染的
滞后"**（那应该随帧周期 62→48 ms 变）。33.3 ms 恰好是 `DAEDALUS_CAPTURE_FPS=30` 的
一个周期，但采集节流（`interval = 1/capture_fps`）在这两档里都没触发（渲染比它慢），
所以还不能下结论。

**我没能在不插桩的情况下把它钉死。** 要定位，最直接的一步是在仿真器里给
`receive_tcp_frame` 的采集点同时打一个 **Bevy `Time::elapsed_secs()`**，
和 `export_armor_csv` 里那个 `elapsed_secs()` 对齐——同一条时间轴上一比，就知道
是"戳打晚了"还是"像用了旧帧"。

## 4. 顺带查出来的两件事

- **异步检测会返回旧帧**：检测 CSV 里 `time_us − source_time_us` 中位 **−14.4 ms**，
  而同一帧的 `source_delay` 是 **+36 ms**，说明这份检测来自**约 1 帧之前（~50 ms）**的帧。
  pipeline 已经正确处理（`source_result_.timestamp_us` 带的是旧帧自己的本机戳），
  但它确实又吃掉了 ~50 ms 的时效。
- **就算把时间戳修对，几何拟合仍然不收敛**：把单板轨迹按 −70 ms 平移后重跑，
  重投影 RMSE 仍是 33~36 px（灯条 74 px），α 依旧顶到 ±83° 的边界。
  说明除时间戳外，**估计器的多起点初始化对真实数据也不够**（自洽合成数据能到 1.45 px），
  这条要单独修。

## 5. 复现

```bash
# 1440x1080（默认）
ULTRA_VISION_DISABLE_CONTROL=1 ULTRA_VISION_DETECTIONS_CSV=/tmp/d1.csv \
tools/auto_aim_sim_test.sh --no-build --seconds 12 --camera armor_camera \
    --distance 1.0 --gyro 1 --gyro-speed 3.0 --out /tmp/r1

# 720x540（帧率不同，用来判断滞后是否随帧周期缩放）
ULTRA_VISION_DISABLE_CONTROL=1 ULTRA_VISION_DETECTIONS_CSV=/tmp/d2.csv \
tools/auto_aim_sim_test.sh --no-build --seconds 10 --camera armor_camera \
    --distance 1.0 --capture 720x540 --gyro 1 --gyro-speed 3.0 --out /tmp/r2
```

检测 CSV 现在同时带 `time_us`（本机到手）和 `source_time_us`（仿真曝光戳），
方便离线判"该用哪个"。
