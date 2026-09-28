#ifndef HIK_CAMERA_HPP
#define HIK_CAMERA_HPP

// 海康（Hikvision / HIKROBOT）工业相机驱动：接口与 io/camera/GalaxyCamera.hpp 完全
// 一致，所以 auto_aim 的入口只需要按 configs/camera.yaml 的 camera.name 选择驱动。
//
// 依赖 MVS SDK：include/MvCameraControl.h + lib/<arch>/libMvCameraControl.so
// （CMake 用 ULTRA_VISION_HIK_SDK_ROOT 指定，默认 /opt/MVS；见顶层 CMakeLists）。
//
// 实测相机配置（MVS 客户端，2026-09-26）：**1440x1080 / Bayer RG 8**，所以驱动里
// 显式设置像素格式并用 MV_CC_ConvertPixelType 转成 BGR8_Packed（SDK 内部做去马赛克），
// 上层拿到的就是普通的 cv::Mat BGR。

#include <opencv2/opencv.hpp>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <string>
#include <thread>

namespace rm_ultra {

/// @brief 帧到手时刻（单调时钟）。用它去查同一时刻的 IMU 姿态，才不会有"处理耗时/队列
///        延迟"造成的姿态错配——手眼标定和自瞄的提前量都吃这个精度。
using FrameTime = std::chrono::steady_clock::time_point;

class HikCamera {
public:
    HikCamera();
    ~HikCamera();

    /// @param serialNum 非空则按序列号选设备，否则按 deviceIndex（**从 1 开始**，与 configs/camera.yaml 一致）
    bool init(const std::string& serialNum = "", int deviceIndex = 0);
    void close();
    bool getImage(cv::Mat& image, int timeoutMs = 1000);
    /// @brief 带**到手时间戳**取图。时间戳在采集线程 `MV_CC_GetImageBuffer` 返回后立刻打，
    ///        不包含队列等待和处理耗时（`captureThreadFunc` 只保留最新几帧，队满丢最旧）。
    bool getImage(cv::Mat& image, FrameTime& timestamp, int timeoutMs = 1000);
    bool isOpened() const { return m_isOpened.load(); }

    /// @brief 读回当前曝光（微秒）/增益（dB），用于确认设置有没有被 SDK 钳位。
    bool exposureTimeUs(double& exposureUs) const;
    bool gain(double& gainDb) const;

    bool setExposureTime(double exposureUs);
    bool setGain(double gain);
    bool setAutoExposure(bool enable);
    bool setAutoGain(bool enable);

private:
    void captureThreadFunc();

    void* m_handle = nullptr;              // MV_CC_HANDLE，避免在头文件里引入 SDK
    std::atomic<bool> m_isOpened{false};
    std::atomic<bool> m_quit{false};
    std::thread m_captureThread;

    std::queue<std::pair<cv::Mat, FrameTime>> m_queue;
    mutable std::mutex m_mutex;
    std::condition_variable m_condition;
    std::size_t m_maxQueue = 4;
};

} // namespace rm_ultra

#endif // HIK_CAMERA_HPP
