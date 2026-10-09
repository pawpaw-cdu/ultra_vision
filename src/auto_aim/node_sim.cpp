// 仿真入口（auto_aim_sim）——薄组合根。
//
// 对齐 sp_vision 的 `tasks/ 能力模块 + src/*.cpp 薄组合根`：
//   · 能力层（detector / armor_source / aim_pipeline / tracker …）负责算法；
//   · app/sim_runner 负责仿真专属的装配与每帧阶段（TCP 帧源、曝光→到手时间基、
//     三条 CSV、曲线、估计→选板→开火）；
//   · 本文件只做三件事：**构造** SimRunner、跑一段可读的**主循环**、做**显示/上报**。
//
// 主循环就是整条数据流本身，读一遍就知道一帧都发生了什么：
//   fetchFrame → prepareFrame(检测) → recordDetections(CSV) → runAim(瞄准/开火)
//   → 叠画 → 每秒上报 → 曲线 + 显示。
// 真机入口 node.cpp 以后接的是同一套能力层，只把这里的 TCP 帧源/时间基换成相机 + 串口。

#include <chrono>
#include <cstdlib>
#include <thread>

#include <opencv2/opencv.hpp>

#include "app/config_paths.hpp"
#include "app/overlay.hpp"
#include "app/perf_report.hpp"
#include "app/perf_stats.hpp"
#include "app/sim_runner.hpp"
#include "control/aim_pipeline.hpp"
#include "perception/armor_source.hpp"
#include "sim_receiver/vision_date_receiver.hpp"
#include "visualization/curve_plotter.hpp"

int main(int argc, char** argv)
{
    auto_aim::app::SimRunner runner(auto_aim::app::resolveConfigDir(argc, argv));
    if (!runner.initialize()) {
        return 1;
    }

    // 记录/退出都由环境变量控制（详见 app/csv_sink.hpp 与 README）：
    //   ULTRA_VISION_ESTIMATE_CSV / OBSERVATION_CSV / SELECTOR_CSV → 三条 CSV 路径；
    //   ULTRA_VISION_TEST_SECONDS → 跑 N 秒后干净退出；ESC / e 键随时退出。
    // 组件引用（只是给下面循环少写几次 runner.xxx()，不改变所有权）。
    auto_aim::app::PerfStats& stats = runner.stats();
    auto_aim::visualization::CurvePlotter& curves = runner.curves();
    sim_receiver::VisionDateReceiver& receiver = runner.receiver();
    auto_aim::ArmorSource& armor_source = runner.armorSource();
    auto_aim::AimPipeline& pipeline = runner.pipeline();
    auto_aim::Tracker& tracker = runner.tracker();
    cv::Mat& image = runner.image();
    const cv::Mat& camera_matrix = runner.cameraMatrix();
    const cv::Mat& dist_coeffs = runner.distCoeffs();

    // 主循环分段耗时（ULTRA_VISION_PROF=1 时每秒打印一次）。只看
    // wait/work/display 三个数不够：work 里混着 ROI 投影、帧拷贝、PnP、EKF、
    // CSV 格式化、overlay 绘制，不拆开就不知道该优化谁。和 sp_vision 一样先量再改。
    // 每秒上报（reportPerSecond）里还有三类诊断，出问题时先看它们：
    //   · 进帧速率 net：net_fps 是每秒真正拿到的帧数，empty 是 getFrame 超时返回空的
    //     次数。自瞄需要 ≥15~20 fps，掉到个位数先看仿真器日志里的 [tcp] encoded_fps。
    //   · 关联层 assoc：det 是检测框数，nomatch/ambiguous/scale 分别是被"找不到对应板/
    //     模棱两可/尺度闸门"丢掉的，accepted 是真正进估计器的。lost 一直涨而 det 很多时，
    //     看这三个数就知道是关联层把观测全拒了，还是检测器在给假框。
    //   · 曝光→到手延迟 source_delay：每秒取中位，喂给选择器当观测年龄的固定附加项。
    while (!runner.exitRequested()) {
        // 定时退出（ULTRA_VISION_TEST_SECONDS）：仿真回归脚本要"跑 N 秒然后干净退出"，
        // 这样 CSV 的 ofstream 正常析构、最后几行不会因为 SIGTERM 丢掉。
        if (runner.testSecondsExpired()) {
            break;
        }

        const auto loop_start = std::chrono::steady_clock::now();
        if (!runner.fetchFrame(loop_start)) {
            continue;   // 没取到帧（超时返回空）：等下一帧
        }

        auto stage_mark = runner.afterFrame();
        double stage_ms[auto_aim::app::PerfStats::kStageCount] = {0.0};
        const auto lap = [&stage_mark, &stage_ms](int index) {
            const auto now = std::chrono::steady_clock::now();
            stage_ms[index] +=
                std::chrono::duration<double, std::milli>(now - stage_mark).count();
            stage_mark = now;
        };

        // 4. 时间基 + 处理尺寸缩放 + 内参 + 检测（共用组件：传统/神经网络 + 动态 ROI + PnP）
        runner.prepareFrame();
        lap(2);
        // 记录 observation.csv（含 PnP 角点，供离线重解）与 estimate.csv
        runner.recordDetections();
        lap(3);
        // 6. 估计 → 选板 → 瞄准 → 开火（和真机 node.cpp 跑同一份 AimPipeline）+ selector.csv
        runner.runAim();
        lap(4);

        // 瞄准结果叠画（整车轮廓 + 观测板高亮 + 提前点）；无显示时整段跳过。
        // ULTRA_VISION_SNAPSHOT_DIR=<dir>：不依赖 GUI，另存成对快照
        //   sim_XXXX.png = 算法收到的仿真原图（仿真器画面）
        //   proc_XXXX.png = 叠加检测/整车/提前点后的处理画面
        const char* snap_dir = std::getenv("ULTRA_VISION_SNAPSHOT_DIR");
        if (runner.showDisplay() || snap_dir != nullptr) {
            if (snap_dir != nullptr && tracker.hasObservation()) {
                static int snap_k = 0;
                ++snap_k;
                if (snap_k % 30 == 0 && snap_k <= 420) {
                    cv::imwrite(cv::format("%s/sim_%04d.png", snap_dir, snap_k), image);
                    cv::Mat overlayed = image.clone();
                    auto_aim::app::drawAimOverlay(overlayed, tracker, runner.lastDecision(),
                                                  runner.lastGimbalCommand(), runner.shooterStatus(),
                                                  camera_matrix, dist_coeffs);
                    cv::imwrite(cv::format("%s/proc_%04d.png", snap_dir, snap_k), overlayed);
                }
            }
            auto_aim::app::drawAimOverlay(image, tracker, runner.lastDecision(),
                                          runner.lastGimbalCommand(), runner.shooterStatus(),
                                          camera_matrix, dist_coeffs);
        }

        // 每秒一次的 FPS / 关联层 / 网络 / 观测延迟上报（到点才打印）。
        ++stats.fps_frames;
        const auto now = std::chrono::steady_clock::now();
        const double elapsed = std::chrono::duration<double>(now - runner.fpsStart()).count();
        if (elapsed >= 1.0) {
            auto_aim::app::reportPerSecond(stats, elapsed, tracker, armor_source,
                                           runner.lastDecision(), runner.shooterStatus(),
                                           pipeline, runner.profStages(),
                                           runner.sourceDelayEnabled(), runner.aimJumpLimit(),
                                           runner.aimJumpRejections(), receiver, now,
                                           runner.fpsStart(),
                                           runner.lightCount(), runner.armorCount(),
                                           runner.pnpCount());
        }
        lap(5);

        // 这一下采样**只为调试窗口**：ULTRA_VISION_NO_DISPLAY=1（回归全走这条路）时
        // 原来照样每帧做一次 1440x1080 → 640x480 的 INTER_AREA，实测 15~18 ms/帧，
        // 比整套 PnP+EKF+控制（~0.5 ms）贵 30 倍。没有窗口就整段跳过。
        if (runner.showDisplay()) {
            cv::resize(image, image, cv::Size(640, 480), 0.0, 0.0, cv::INTER_AREA);
        }
        lap(6);
        const auto before_display = std::chrono::steady_clock::now();

        // 实时数学曲线：装甲板位姿 vs 瞄准方向（滞后/超前/跳变只在时间序列上看得见）。
        runner.pushCurves(camera_matrix, dist_coeffs);
        // HighGUI 比整条处理链还贵（实测 ~15 ms/帧 vs ~0.9 ms），机器人上也没有显示器。
        // 检测图与曲线合成**一个**窗口（竖直：图在上、曲线在下），无显示时用 1ms 节流。
        if (runner.showDisplay()) {
            const cv::Mat display = runner.curvesEnabled() ? curves.renderStacked(image) : image;
            cv::imshow("Ultra Vision", display);
            runner.setKey(cv::waitKey(1));
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            runner.setKey(1);
        }
        const auto after_display = std::chrono::steady_clock::now();

        // 累计本帧耗时（每秒在 reportPerSecond 里打印并清零）。
        const auto ms = [](auto from, auto to) {
            return std::chrono::duration<double, std::milli>(to - from).count();
        };
        stats.prof_wait_ms += ms(loop_start, runner.afterFrame());
        stats.prof_work_ms += ms(runner.afterFrame(), before_display);
        stats.prof_display_ms += ms(before_display, after_display);
        for (int stage = 0; stage < auto_aim::app::PerfStats::kStageCount; ++stage) {
            stats.prof_stage_sum[stage] += stage_ms[stage];
        }
    }

    return 0;
}
