#ifndef SIM_RECEIVER_TCP_INPUT_HPP
#define SIM_RECEIVER_TCP_INPUT_HPP

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace sim_receiver {

// Reusable byte-buffer pool.
//
// The receiver hands a freshly received JPEG payload to a decoder thread.  If the
// payload were owned by a plain std::vector the reader would either have to copy
// it (a full 100 KB+ memcpy per frame) or allocate a new buffer per frame.  This
// pool instead lends the storage out as a shared_ptr whose deleter returns the
// buffer once the last owner (reader -> decoder) drops it, so the pixel bytes
// travel by pointer and the allocation happens once per concurrent frame.
class BufferPool : public std::enable_shared_from_this<BufferPool> {
public:
    using Buffer = std::shared_ptr<std::vector<uint8_t>>;

    // A buffer of exactly `size` bytes.  Storage is recycled, never shrunk in
    // capacity, so steady-state operation performs no allocation.
    Buffer acquire(std::size_t size);

    std::size_t freeCount() const;

private:
    void release(std::shared_ptr<std::vector<uint8_t>> buffer);

    mutable std::mutex mutex_;
    std::vector<std::shared_ptr<std::vector<uint8_t>>> free_;
};

// One wire frame: the (v2) 40-byte header decoded, plus the JPEG payload still
// owned by the pool.  Nothing is copied between the socket and the decoder
// threads.  v2 在 v1 的 32 字节后追加了**真实云台姿态** `gimbal_yaw/pitch`
// （f32, big-endian），视觉端用它当 CameraPose —— 等价实车的 IMU 反馈。
struct SimPacket {
    BufferPool::Buffer payload;
    uint64_t seq = 0;
    uint64_t source_timestamp_us = 0;   // exposure time stamped by the simulator
    uint64_t local_timestamp_us = 0;    // when the last payload byte arrived
    uint32_t width = 0;
    uint32_t height = 0;
    float gimbal_yaw = 0.0f;
    float gimbal_pitch = 0.0f;
};

struct SimFrame {
    cv::Mat bgr;
    uint64_t seq = 0;
    uint64_t source_timestamp_us = 0;
    uint64_t local_timestamp_us = 0;
    float gimbal_yaw = 0.0f;
    float gimbal_pitch = 0.0f;
};

// TCP client for the simulator's image channel.
//
// The socket lifecycle is RAII: the file descriptors are closed by the
// destructor (and by close()), and every early-return path in readPacket()
// tears the image socket down instead of leaking it.  readPacket() only
// receives; decoding is deliberately left to the caller so the socket can be
// drained on its own thread.
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

    // Receive one frame's header and payload.  No decode, no copy: the payload
    // is a pooled buffer that the caller passes on to a decode thread.
    bool readPacket(SimPacket& packet);

    // Convenience wrapper for synchronous callers: readPacket + decode.
    bool readFrame(SimFrame& frame);

    // Decode a packet's JPEG payload.  Empty Mat when the payload is invalid.
    static cv::Mat decode(const SimPacket& packet);

    // Unblocks a reader parked in recv() so stop() does not wait for the socket
    // timeout.  Safe to call from another thread; the reader still owns close().
    void shutdownImage();

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
    // Touched by the reader thread and by shutdownImage() from the owner thread,
    // so the handle is atomic: a stop request can unblock recv() without the two
    // threads racing on the descriptor.
    std::atomic<int> sock_{-1};
    int command_sock_ = -1;
    uint16_t command_port_ = 7667;
    std::array<uint8_t, 40> header_{};
    std::shared_ptr<BufferPool> pool_;
};

} // namespace sim_receiver

#endif // SIM_RECEIVER_TCP_INPUT_HPP
