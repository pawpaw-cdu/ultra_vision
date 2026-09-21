#ifndef GALAXY_CAMERA_HPP
#define GALAXY_CAMERA_HPP

#include <opencv2/opencv.hpp>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <string>
#include <thread>

#include <GxIAPI.h>

namespace rm_ultra {

class GalaxyCamera {
public:
    GalaxyCamera();
    ~GalaxyCamera();

    bool init(const std::string& serialNum = "", int deviceIndex = 1);
    void close();
    bool getImage(cv::Mat& image, int timeoutMs = 1000);
    bool isOpened() const { return m_isOpened; }

    // 可选参数设置
    bool setExposureTime(double exposureUs);
    bool setGain(double gain);
    bool setAutoExposure(bool enable);
    bool setAutoGain(bool enable);

private:
    void captureThreadFunc();

    GX_DEV_HANDLE m_hDevice;
    std::atomic<bool> m_isOpened;
    std::atomic<bool> m_running;
    std::thread m_captureThread;

    std::queue<cv::Mat> m_imageQueue;
    std::mutex m_queueMutex;
    std::condition_variable m_queueCond;

    int64_t m_width, m_height;
    int64_t m_payloadSize;
    std::vector<char> m_buffer;
};

} // namespace rm_ultra

#endif
