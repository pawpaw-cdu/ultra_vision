// 海康（Hikrobot MVS）相机探针：用**我们自己的 HikCamera 驱动**打开、抓帧、存图。
//
// 用途：上真机第一步 —— 确认 SDK 能看到相机、驱动能把 BayerRG8 正确转成 BGR、
// 分辨率/帧率是不是和标定一致（1440x1080 / 41°），并落一张 jpg 供肉眼确认。
// 顺带把每帧的通道均值打出来：发绿 / R/B 互换都能一眼看出来（Bayer 解码错的典型症状）。
//
//   tools/hik_probe <frames> [output.jpg] [exposure_ms] [gain]
//     exposure_ms <= 0 表示用自动曝光（并先等 2 s 让 AE 收敛，否则画面全黑）；
//     给了正数就切成手动曝光 + 手动增益（比赛口径）。

#include <chrono>
#include <thread>
#include <cstdlib>
#include <iostream>
#include <string>

#include <opencv2/opencv.hpp>

#include "io/camera/HikCamera.hpp"

int main(int argc, char* argv[])
{
    const int frames_wanted = argc > 1 ? std::atoi(argv[1]) : 30;
    const std::string output = argc > 2 ? argv[2] : "/tmp/hik_probe.jpg";
    const double exposure_ms = argc > 3 ? std::atof(argv[3]) : 0.0;
    const double gain = argc > 4 ? std::atof(argv[4]) : 0.0;

    rm_ultra::HikCamera camera;
    // device_index 从 1 开始（MVS 的枚举口径）；serial 为空表示按索引开。
    if (!camera.init("", 1)) {
        std::cerr << "[hik] 打不开相机（设备没插 / 被 MVS 客户端占用 / 权限）" << std::endl;
        return 1;
    }

    // 读回来确认设置是否真的生效（MVS 有些参数受"当前节点可用性/量程"限制，会静默钳位）
    auto reportSetting = [&camera]() {
        double exposure = -1.0, gain = -1.0;
        if (camera.exposureTimeUs(exposure) && camera.gain(gain)) {
            std::cout << "[hik] 当前曝光 " << exposure / 1000.0 << " ms，增益 " << gain
                      << " dB" << std::endl;
        }
    };

    if (exposure_ms > 0.0) {
        std::cout << "[hik] 设置前：" ; reportSetting();
        const bool ok_exposure = camera.setExposureTime(exposure_ms * 1000.0);
        const bool ok_gain = gain > 0.0 ? camera.setGain(gain) : true;
        std::cout << "[hik] 手动曝光 " << exposure_ms << "ms 增益 " << gain << "dB  设置"
                  << ((ok_exposure && ok_gain) ? "成功" : "失败") << std::endl;
    } else {
        camera.setAutoExposure(true);
        std::cout << "[hik] 自动曝光：先等 2 s 让 AE 收敛（否则画面会是黑的）" << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(2));
    }

    reportSetting();

    cv::Mat frame;
    const auto start = std::chrono::steady_clock::now();
    int got = 0;
    double sum_ms = 0.0;
    cv::Scalar mean;
    while (got < frames_wanted) {
        const auto before = std::chrono::steady_clock::now();
        if (!camera.getImage(frame, 1000) || frame.empty()) {
            std::cerr << "[hik] 取图失败（第 " << got << " 帧）" << std::endl;
            break;
        }
        sum_ms += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - before).count();
        ++got;
        mean = cv::mean(frame);
    }
    const double elapsed =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    if (got == 0) return 1;

    cv::imwrite(output, frame);
    std::cout << "[hik] 抓到 " << got << " 帧，用时 " << elapsed << " s → "
              << got / elapsed << " fps，取图平均 " << sum_ms / got << " ms" << std::endl;
    std::cout << "[hik] 分辨率 " << frame.cols << "x" << frame.rows << " 通道 " << frame.channels()
              << " 类型 " << frame.type() << std::endl;
    std::cout << "[hik] 通道均值 B/G/R = " << mean[0] << "/" << mean[1] << "/" << mean[2]
              << "（正常室内三者同量级；G 远高于 R/B=发绿，说明 Bayer 解码错）" << std::endl;
    reportSetting();
    std::cout << "[hik] 已存 " << output << std::endl;
    return 0;
}
