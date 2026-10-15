#ifndef SIM_RECEIVER_TCP_INPUT_HPP
#define SIM_RECEIVER_TCP_INPUT_HPP

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "sim_packet.hpp"

namespace sim_receiver {

// TCP client for the simulator's image channel (default transport), and for the
// command channel (`FIRE` / `GIMBAL` / `RESET`, port 7667) in every transport.
//
// The socket lifecycle is RAII: the file descriptors are closed by the
// destructor (and by close()), and every early-return path in read() tears the
// image socket down instead of leaking it.  read() only receives; decoding is
// deliberately left to the caller so the socket can be drained on its own
// thread.
class TcpInput : public PacketSource {
public:
    explicit TcpInput(std::string host, uint16_t port);
    ~TcpInput() override;

    TcpInput(const TcpInput&) = delete;
    TcpInput& operator=(const TcpInput&) = delete;
    TcpInput(TcpInput&& other) noexcept;
    TcpInput& operator=(TcpInput&& other) noexcept;

    bool connect() override;
    bool connected() const override;

    bool sendLine(uint16_t command_port, const std::string& line);

    // Receive one frame's header and payload.  No decode, no copy: the payload
    // is a pooled buffer that the caller passes on to a decode thread.
    // Accepts both 32-byte (v1) and 40-byte (v2, +真实云台姿态) headers.
    ReadStatus read(SimPacket& packet) override;

    // Convenience wrapper for synchronous callers: read + decode.
    bool readFrame(SimFrame& frame);

    // Unblocks a reader parked in recv() so stop() does not wait for the socket
    // timeout.  Safe to call from another thread; the reader still owns close().
    void shutdown() override;

    void close() override;

    const std::string& host() const { return host_; }
    uint16_t port() const { return port_; }

private:
    static bool writeAll(int fd, const std::string& data);
    bool connectOnce();
    bool connectCommandOnce();
    void closeImage();
    void closeCommand();
    bool readExact(void* buffer, size_t size);

    std::string host_;
    uint16_t port_;
    // Touched by the reader thread and by shutdown() from the owner thread, so
    // the handle is atomic: a stop request can unblock recv() without the two
    // threads racing on the descriptor.
    std::atomic<int> sock_{-1};
    int command_sock_ = -1;
    uint16_t command_port_ = 7667;
    std::array<uint8_t, 40> header_{};
    std::shared_ptr<BufferPool> pool_;
};

} // namespace sim_receiver

#endif // SIM_RECEIVER_TCP_INPUT_HPP
