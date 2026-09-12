#ifndef SIM_RECEIVER_VISION_DATE_RECEVIER_HPP
#define SIM_RECEIVER_VISION_DATE_RECEVIER_HPP

#include <cstdint>
#include <memory>
#include <string>

#include <opencv2/core.hpp>

namespace sim_receiver {

class VisionDateRecevier {
public:
    explicit VisionDateRecevier(const std::string& config_path);
    ~VisionDateRecevier();

    VisionDateRecevier(const VisionDateRecevier&) = delete;
    VisionDateRecevier& operator=(const VisionDateRecevier&) = delete;
    VisionDateRecevier(VisionDateRecevier&&) noexcept = default;
    VisionDateRecevier& operator=(VisionDateRecevier&&) noexcept = default;

    void overrideEndpoint(const std::string& host, uint16_t port);
    bool connect();
    bool connected() const;
    bool sendFireCommand();

    // Returns an empty Mat when the stream is closed or a frame is invalid.
    cv::Mat getFrame();

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

using VisionDateReceiver = VisionDateRecevier;
using VisionDataReceiver = VisionDateRecevier;

} // namespace sim_receiver

#endif
