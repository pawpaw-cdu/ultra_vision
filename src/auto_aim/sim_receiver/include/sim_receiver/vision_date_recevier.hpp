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
    void overrideCommandEndpoint(uint16_t port);
    bool connect();
    bool connected() const;
    bool sendFireCommand();
    // Puts the simulator scene back into its startup state (camera pose, chassis
    // position and yaw, gimbal, small gyro). Used to make back-to-back test runs
    // comparable.
    bool sendResetCommand();
    bool sendGimbalCommand(double yaw, double pitch);

    // Returns an empty Mat when the stream is closed or a frame is invalid.
    cv::Mat getFrame(std::uint64_t* sequence = nullptr,
                     std::uint64_t* source_timestamp_us = nullptr,
                     std::uint64_t* local_timestamp_us = nullptr);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

using VisionDateReceiver = VisionDateRecevier;
using VisionDataReceiver = VisionDateRecevier;

} // namespace sim_receiver

#endif
