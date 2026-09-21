#include "HikCamera.hpp"

#include <MvCameraControl.h>

#include <chrono>
#include <cstring>
#include <iostream>
#include <vector>

namespace rm_ultra {
namespace {
    /// 统一在这里改：MVS 里实测的输出（1440x1080 / Bayer RG 8）。
    constexpr int kWidth = 1440;
    constexpr int kHeight = 1080;
    constexpr int kBufferCount = 4;
    constexpr double kFrameRate = 100.0;   // 自瞄要快帧率；SDK 会按相机能力钳制

    bool check(int ret, const char* what)
    {
        if (ret == MV_OK) return true;
        std::cerr << "[HikCamera] " << what << " failed: 0x" << std::hex << ret << std::dec
                  << std::endl;
        return false;
    }
} // namespace

HikCamera::HikCamera() = default;

HikCamera::~HikCamera() { close(); }

bool HikCamera::init(const std::string& serialNum, int deviceIndex)
{
    if (m_isOpened.load()) return true;

    MV_CC_Initialize();
    MV_CC_DEVICE_INFO_LIST deviceList;
    std::memset(&deviceList, 0, sizeof(deviceList));
    if (!check(MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, &deviceList),
               "MV_CC_EnumDevices")) {
        return false;
    }
    if (deviceList.nDeviceNum == 0) {
        std::cerr << "[HikCamera] no device found" << std::endl;
        return false;
    }

    MV_CC_DEVICE_INFO* selected = nullptr;
    if (!serialNum.empty()) {
        for (unsigned i = 0; i < deviceList.nDeviceNum; ++i) {
            const MV_CC_DEVICE_INFO* info = deviceList.pDeviceInfo[i];
            const char* serial = nullptr;
            if (info->nTLayerType == MV_USB_DEVICE) {
                serial = reinterpret_cast<const char*>(info->SpecialInfo.stUsb3VInfo.chSerialNumber);
            } else if (info->nTLayerType == MV_GIGE_DEVICE) {
                serial = reinterpret_cast<const char*>(info->SpecialInfo.stGigEInfo.chSerialNumber);
            }
            if (serial != nullptr && serialNum == serial) {
                selected = deviceList.pDeviceInfo[i];
                break;
            }
        }
        if (selected == nullptr) {
            std::cerr << "[HikCamera] serial " << serialNum << " not found" << std::endl;
            return false;
        }
    } else {
        // 与外层约定一致：configs/camera.yaml 的 device_index **从 1 开始**
        // （GalaxyCamera 也是这个约定），所以这里减一。
        const unsigned index = deviceIndex <= 1 ? 0u : static_cast<unsigned>(deviceIndex - 1);
        if (index >= deviceList.nDeviceNum) {
            std::cerr << "[HikCamera] device index " << index << " out of range ("
                      << deviceList.nDeviceNum << ")" << std::endl;
            return false;
        }
        selected = deviceList.pDeviceInfo[index];
    }

    if (!check(MV_CC_CreateHandle(&m_handle, selected), "MV_CC_CreateHandle")) return false;
    if (!check(MV_CC_OpenDevice(m_handle), "MV_CC_OpenDevice")) {
        MV_CC_DestroyHandle(m_handle);
        m_handle = nullptr;
        return false;
    }

    // 分辨率与像素格式：与 MVS 实测一致。BayerRG8 进来，转 BGR8 出去（SDK 去马赛克）。
    check(MV_CC_SetEnumValue(m_handle, "PixelFormat", PixelType_Gvsp_BayerRG8), "set BayerRG8");
    check(MV_CC_SetIntValue(m_handle, "Width", kWidth), "set Width");
    check(MV_CC_SetIntValue(m_handle, "Height", kHeight), "set Height");
    check(MV_CC_SetEnumValue(m_handle, "AcquisitionMode", MV_ACQ_MODE_CONTINUOUS), "set mode");
    check(MV_CC_SetBoolValue(m_handle, "AcquisitionFrameRateEnable", true), "enable fps");
    check(MV_CC_SetFloatValue(m_handle, "AcquisitionFrameRate",
                              static_cast<float>(kFrameRate)), "set fps");
    check(MV_CC_SetEnumValue(m_handle, "ExposureAuto", MV_EXPOSURE_AUTO_MODE_CONTINUOUS),
          "auto exposure");
    check(MV_CC_SetEnumValue(m_handle, "GainAuto", MV_GAIN_MODE_CONTINUOUS), "auto gain");

    if (!check(MV_CC_StartGrabbing(m_handle), "MV_CC_StartGrabbing")) {
        MV_CC_CloseDevice(m_handle);
        MV_CC_DestroyHandle(m_handle);
        m_handle = nullptr;
        return false;
    }

    m_quit.store(false);
    m_isOpened.store(true);
    m_captureThread = std::thread(&HikCamera::captureThreadFunc, this);
    std::cout << "[HikCamera] opened " << kWidth << "x" << kHeight << " BayerRG8 -> BGR8"
              << std::endl;
    return true;
}

void HikCamera::captureThreadFunc()
{
    // 目标缓冲：与源同尺寸的 BGR8。
    std::vector<unsigned char> destination(static_cast<std::size_t>(kWidth) * kHeight * 3);
    MV_CC_PIXEL_CONVERT_PARAM convertParam;
    while (!m_quit.load()) {
        MV_FRAME_OUT frameOut;
        std::memset(&frameOut, 0, sizeof(frameOut));
        const int ret = MV_CC_GetImageBuffer(m_handle, &frameOut, 1000);
        if (ret != MV_OK) continue;   // 超时/丢帧：继续取
        // 到手时刻：紧跟取图调用，尽量贴近曝光结束时刻（不含后续转换与队列等待）。
        const FrameTime frameTime = std::chrono::steady_clock::now();

        const unsigned width = frameOut.stFrameInfo.nWidth;
        const unsigned height = frameOut.stFrameInfo.nHeight;
        const std::size_t needed = static_cast<std::size_t>(width) * height * 3;
        if (destination.size() < needed) destination.resize(needed);
        std::memset(&convertParam, 0, sizeof(convertParam));
        convertParam.nWidth = width;
        convertParam.nHeight = height;
        convertParam.pSrcData = frameOut.pBufAddr;
        convertParam.nSrcDataLen = frameOut.stFrameInfo.nFrameLen;
        convertParam.enSrcPixelType = frameOut.stFrameInfo.enPixelType;
        convertParam.enDstPixelType = PixelType_Gvsp_BGR8_Packed;
        convertParam.pDstBuffer = destination.data();
        convertParam.nDstBufferSize = static_cast<unsigned>(destination.size());

        if (MV_CC_ConvertPixelType(m_handle, &convertParam) == MV_OK) {
            // 拷贝一份：SDK 的缓冲在 FreeImageBuffer 之后失效。
            cv::Mat image(static_cast<int>(height), static_cast<int>(width), CV_8UC3);
            std::memcpy(image.data, destination.data(), needed);
            std::lock_guard<std::mutex> lock(m_mutex);
            while (m_queue.size() >= m_maxQueue) m_queue.pop();   // 永远只保留最新几帧
            m_queue.emplace(std::move(image), frameTime);
            m_condition.notify_one();
        }
        MV_CC_FreeImageBuffer(m_handle, &frameOut);
    }
}

bool HikCamera::getImage(cv::Mat& image, int timeoutMs)
{
    FrameTime timestamp;
    return getImage(image, timestamp, timeoutMs);
}

bool HikCamera::getImage(cv::Mat& image, FrameTime& timestamp, int timeoutMs)
{
    if (!m_isOpened.load()) return false;
    std::unique_lock<std::mutex> lock(m_mutex);
    if (!m_condition.wait_for(lock, std::chrono::milliseconds(timeoutMs),
                              [this] { return !m_queue.empty() || m_quit.load(); })) {
        return false;
    }
    if (m_queue.empty()) return false;
    image = std::move(m_queue.front().first);
    timestamp = m_queue.front().second;
    m_queue.pop();
    return !image.empty();
}

bool HikCamera::exposureTimeUs(double& exposureUs) const
{
    MVCC_FLOATVALUE value{};
    if (m_handle == nullptr ||
        MV_CC_GetFloatValue(m_handle, "ExposureTime", &value) != MV_OK) {
        return false;
    }
    exposureUs = value.fCurValue;
    return true;
}

bool HikCamera::gain(double& gainDb) const
{
    MVCC_FLOATVALUE value{};
    if (m_handle == nullptr || MV_CC_GetFloatValue(m_handle, "Gain", &value) != MV_OK) {
        return false;
    }
    gainDb = value.fCurValue;
    return true;
}

bool HikCamera::setExposureTime(double exposureUs)
{
    if (m_handle == nullptr) return false;
    return check(MV_CC_SetEnumValue(m_handle, "ExposureAuto", MV_EXPOSURE_AUTO_MODE_OFF),
                 "manual exposure") &&
        check(MV_CC_SetFloatValue(m_handle, "ExposureTime", static_cast<float>(exposureUs)),
              "set ExposureTime");
}

bool HikCamera::setGain(double gain)
{
    if (m_handle == nullptr) return false;
    return check(MV_CC_SetEnumValue(m_handle, "GainAuto", MV_GAIN_MODE_OFF), "manual gain") &&
        check(MV_CC_SetFloatValue(m_handle, "Gain", static_cast<float>(gain)), "set Gain");
}

bool HikCamera::setAutoExposure(bool enable)
{
    if (m_handle == nullptr) return false;
    return check(MV_CC_SetEnumValue(m_handle, "ExposureAuto",
                                   enable ? MV_EXPOSURE_AUTO_MODE_CONTINUOUS
                                          : MV_EXPOSURE_AUTO_MODE_OFF),
                 "set ExposureAuto");
}

bool HikCamera::setAutoGain(bool enable)
{
    if (m_handle == nullptr) return false;
    return check(MV_CC_SetEnumValue(m_handle, "GainAuto",
                                   enable ? MV_GAIN_MODE_CONTINUOUS : MV_GAIN_MODE_OFF),
                 "set GainAuto");
}

void HikCamera::close()
{
    m_quit.store(true);
    m_condition.notify_all();
    if (m_captureThread.joinable()) m_captureThread.join();
    if (m_handle != nullptr) {
        MV_CC_StopGrabbing(m_handle);
        MV_CC_CloseDevice(m_handle);
        MV_CC_DestroyHandle(m_handle);
        m_handle = nullptr;
    }
    m_isOpened.store(false);
    MV_CC_Finalize();
}

} // namespace rm_ultra
