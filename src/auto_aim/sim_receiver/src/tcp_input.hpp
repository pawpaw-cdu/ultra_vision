#ifndef SIM_RECEIVER_TCP_INPUT_HPP
#define SIM_RECEIVER_TCP_INPUT_HPP

#include <array>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/opencv.hpp>

namespace sim_receiver {

struct SimFrame {
    cv::Mat bgr;
    uint64_t seq = 0;
    uint64_t source_timestamp_us = 0;
    uint64_t local_timestamp_us = 0;
};

class TcpInput {
public:
    explicit TcpInput(std::string host, uint16_t port);
    ~TcpInput();

    TcpInput(const TcpInput&) = delete;
    TcpInput& operator=(const TcpInput&) = delete;
    TcpInput(TcpInput&& other) noexcept;
    TcpInput& operator=(TcpInput&& other) noexcept;

    bool connect();
    bool connected() const;
    bool sendLine(uint16_t command_port, const std::string& line);
    bool readFrame(SimFrame& frame);
    void close();

private:
    static bool writeAll(int fd, const std::string& data);
    bool connectOnce();
    bool connectCommandOnce();
    void closeImage();
    void closeCommand();
    bool readExact(void* buffer, size_t size);

    std::string host_;
    uint16_t port_;
    int sock_ = -1;
    int command_sock_ = -1;
    uint16_t command_port_ = 7667;
    std::array<uint8_t, 32> header_{};
    std::vector<uint8_t> payload_;
};

} // namespace sim_receiver

#endif
