#ifndef SIM_RECEIVER_SIM_PACKET_HPP
#define SIM_RECEIVER_SIM_PACKET_HPP

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
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

// One image frame on the wire: the header decoded, plus the JPEG payload still
// owned by the pool.  Nothing is copied between the socket and the decoder
// threads.  TCP v2 (40B header) and the UDP fragment header both append the
// **真实云台姿态** `gimbal_yaw/pitch` (f32, big-endian); the vision side uses it
// as CameraPose, i.e. the simulator's equivalent of real-car IMU feedback.
struct SimPacket {
    BufferPool::Buffer payload;
    uint64_t seq = 0;
    uint64_t source_timestamp_us = 0;   // exposure time stamped by the simulator
    uint64_t local_timestamp_us = 0;    // when the last payload byte arrived
    uint32_t width = 0;                 // 0 when the transport does not carry it (UDP)
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

// Result of one read attempt.
//   Frame — a complete frame is in `packet`.
//   Idle  — nothing to deliver *yet*; the stream is still healthy, keep reading.
//           (UDP poll timeout.  Must not be treated as a disconnect: reconnecting
//           would drop the partially reassembled frames and re-subscribe.)
//   Error — the stream failed; the caller should back off and retry.
enum class ReadStatus { Frame, Idle, Error };

// Image-stream interface, implemented by TcpInput and UdpInput so the receive
// pipeline (reader thread -> decode pool -> latest-frame slot) is transport
// agnostic.  Commands always stay on TCP 7667.
class PacketSource {
public:
    virtual ~PacketSource() = default;
    virtual bool connect() = 0;
    virtual bool connected() const = 0;
    virtual ReadStatus read(SimPacket& packet) = 0;
    // Unblock a reader parked in a blocking syscall, if the implementation needs
    // it.  Must be safe to call from the owner thread while the reader runs.
    virtual void shutdown() = 0;
    virtual void close() = 0;
};

// Decode a packet's JPEG payload.  Empty Mat when the payload is invalid.
cv::Mat decodePacket(const SimPacket& packet);

} // namespace sim_receiver

#endif // SIM_RECEIVER_SIM_PACKET_HPP
