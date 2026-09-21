#include "GalaxyCamera.hpp"
#include <iostream>
#include <cstring>

namespace rm_ultra {

GalaxyCamera::GalaxyCamera()
    : m_hDevice(nullptr)
    , m_isOpened(false)
    , m_running(false)
    , m_width(0)
    , m_height(0)
    , m_payloadSize(0) {
}

GalaxyCamera::~GalaxyCamera() {
    close();
}

bool GalaxyCamera::init(const std::string& serialNum, int deviceIndex) {
    if (m_isOpened) return true;

    GX_STATUS status = GXInitLib();
    if (status != GX_STATUS_SUCCESS) {
        std::cerr << "[GalaxyCamera] GXInitLib failed, status=" << status << std::endl;
        return false;
    }

    uint32_t deviceCount = 0;
    status = GXUpdateDeviceList(&deviceCount, 1000);
    if (status != GX_STATUS_SUCCESS || deviceCount == 0) {
        std::cerr << "[GalaxyCamera] No device found, status=" << status << std::endl;
        GXCloseLib();
        return false;
    }
    std::cout << "[GalaxyCamera] Found " << deviceCount << " device(s)." << std::endl;

    status = GXOpenDeviceByIndex(deviceIndex, &m_hDevice);
    if (status != GX_STATUS_SUCCESS || m_hDevice == nullptr) {
        std::cerr << "[GalaxyCamera] GXOpenDeviceByIndex failed, status=" << status << std::endl;
        GXCloseLib();
        return false;
    }

    // 获取图像参数
    GXGetInt(m_hDevice, GX_INT_WIDTH, &m_width);
    GXGetInt(m_hDevice, GX_INT_HEIGHT, &m_height);
    GXGetInt(m_hDevice, GX_INT_PAYLOAD_SIZE, &m_payloadSize);
    std::cout << "[GalaxyCamera] Image size: " << m_width << "x" << m_height
              << ", payload size: " << m_payloadSize << std::endl;

    m_buffer.resize(static_cast<std::size_t>(m_payloadSize));

    status = GXSendCommand(m_hDevice, GX_COMMAND_ACQUISITION_START);
    if (status != GX_STATUS_SUCCESS) {
        std::cerr << "[GalaxyCamera] GXSendCommand START failed, status=" << status << std::endl;
        GXCloseDevice(m_hDevice);
        GXCloseLib();
        return false;
    }

    m_isOpened = true;
    m_running = true;
    m_captureThread = std::thread(&GalaxyCamera::captureThreadFunc, this);

    std::cout << "[GalaxyCamera] Camera initialized successfully." << std::endl;
    return true;
}

void GalaxyCamera::close() {
    if (!m_isOpened) return;

    m_running = false;
    if (m_captureThread.joinable()) {
        m_captureThread.join();
    }

    if (m_hDevice) {
        GXSendCommand(m_hDevice, GX_COMMAND_ACQUISITION_STOP);
        GXCloseDevice(m_hDevice);
        m_hDevice = nullptr;
    }
    GXCloseLib();
    m_isOpened = false;
    m_queueCond.notify_all();
    std::cout << "[GalaxyCamera] Camera closed." << std::endl;
}

bool GalaxyCamera::getImage(cv::Mat& image, int timeoutMs) {
    if (!m_isOpened) return false;

    std::unique_lock<std::mutex> lock(m_queueMutex);
    if (!m_queueCond.wait_for(lock, std::chrono::milliseconds(timeoutMs),
                              [this] { return !m_imageQueue.empty(); })) {
        return false;
    }
    image = m_imageQueue.front();
    m_imageQueue.pop();
    return true;
}

void GalaxyCamera::captureThreadFunc() {
    GX_FRAME_DATA frameData = {0};
    frameData.pImgBuf = m_buffer.data();

    while (m_running && m_isOpened) {
        GX_STATUS status = GXGetImage(m_hDevice, &frameData, 500);
        if (status != GX_STATUS_SUCCESS) {
            continue;
        }

        // 根据像素格式确定 OpenCV 转换码
        // int conversionCode = -1;
        // switch (frameData.nPixelFormat) {
        //     case GX_PIXEL_FORMAT_BAYER_GR8: conversionCode = cv::COLOR_BayerGR2BGR; break;
        //     case GX_PIXEL_FORMAT_BAYER_RG8: conversionCode = cv::COLOR_BayerRG2BGR; break;
        //     case GX_PIXEL_FORMAT_BAYER_GB8: conversionCode = cv::COLOR_BayerGB2BGR; break;
        //     case GX_PIXEL_FORMAT_BAYER_BG8: conversionCode = cv::COLOR_BayerBG2BGR; break;
        //     case GX_PIXEL_FORMAT_MONO8:    conversionCode = cv::COLOR_GRAY2BGR; break;
        //     default:
        //         std::cerr << "[GalaxyCamera] Unsupported pixel format: " << frameData.nPixelFormat << std::endl;
        //         continue;
        // }

        int width = static_cast<int>(frameData.nWidth);
        int height = static_cast<int>(frameData.nHeight);
        cv::Mat raw(height, width, CV_8UC1, frameData.pImgBuf);
        cv::Mat bgrImage;
        cv::cvtColor(raw, bgrImage, cv::COLOR_BayerBG2BGR);

        // 放入队列
        {
            std::lock_guard<std::mutex> lock(m_queueMutex);
            if (m_imageQueue.size() < 10) {
                m_imageQueue.push(bgrImage.clone());
                m_queueCond.notify_one();
            } else {
                m_imageQueue.pop();
                m_imageQueue.push(bgrImage.clone());
            }
        }
    }
}

bool GalaxyCamera::setExposureTime(double exposureUs) {
    if (!m_hDevice) return false;
    return GXSetFloat(m_hDevice, GX_FLOAT_EXPOSURE_TIME, exposureUs) == GX_STATUS_SUCCESS;
}

bool GalaxyCamera::setGain(double gain) {
    if (!m_hDevice) return false;
    return GXSetFloat(m_hDevice, GX_FLOAT_GAIN, gain) == GX_STATUS_SUCCESS;
}

bool GalaxyCamera::setAutoExposure(bool enable) {
    if (!m_hDevice) return false;
    int64_t value = enable ? GX_EXPOSURE_AUTO_CONTINUOUS : GX_EXPOSURE_AUTO_OFF;
    return GXSetEnum(m_hDevice, GX_ENUM_EXPOSURE_AUTO, value) == GX_STATUS_SUCCESS;
}

bool GalaxyCamera::setAutoGain(bool enable) {
    if (!m_hDevice) return false;
    int64_t value = enable ? GX_GAIN_AUTO_CONTINUOUS : GX_GAIN_AUTO_OFF;
    return GXSetEnum(m_hDevice, GX_ENUM_GAIN_AUTO, value) == GX_STATUS_SUCCESS;
}

} // namespace rm_ultra