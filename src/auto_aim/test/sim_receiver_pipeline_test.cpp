// Regression test for the sim_receiver frame pipeline.
//
// The receiver used to do recv() + cv::imdecode() back to back on one thread.
// While it decoded it was not reading the socket, so a fast sender was throttled
// by the client's decode rate over TCP back-pressure — which is why making the
// simulator faster could leave the observed frame rate unchanged.
//
// This test drives a synthetic sender that pushes 1440x1080 JPEG frames as fast
// as the loopback will take them and checks that:
//   * frames still arrive and carry their sequence/timestamps,
//   * the sender is NOT throttled to the decode rate (it free-runs, the receiver
//     drops stale frames instead), which only holds if recv and decode are on
//     separate stages.

#include <sim_receiver/vision_date_receiver.hpp>

#include "tcp_input.hpp"   // sim_receiver::BufferPool

#include <arpa/inet.h>
#include <cstring>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstring>
#include <future>
#include <iostream>
#include <memory>
#include <thread>
#include <vector>

#include <opencv2/opencv.hpp>

namespace {

bool expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAILED: " << message << std::endl;
    }
    return condition;
}

void writeU32(std::vector<uint8_t>& out, uint32_t value) {
    out.push_back(static_cast<uint8_t>(value >> 24));
    out.push_back(static_cast<uint8_t>(value >> 16));
    out.push_back(static_cast<uint8_t>(value >> 8));
    out.push_back(static_cast<uint8_t>(value));
}

void writeU64(std::vector<uint8_t>& out, uint64_t value) {
    for (int shift = 56; shift >= 0; shift -= 8) {
        out.push_back(static_cast<uint8_t>(value >> shift));
    }
}

void writeF32(std::vector<uint8_t>& out, float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    writeU32(out, bits);
}

bool sendAll(int fd, const uint8_t* data, std::size_t size) {
    std::size_t sent = 0;
    while (sent < size) {
#ifdef MSG_NOSIGNAL
        constexpr int kFlags = MSG_NOSIGNAL;
#else
        constexpr int kFlags = 0;
#endif
        const ssize_t n = ::send(fd, data + sent, size - sent, kFlags);
        if (n <= 0) {
            return false;
        }
        sent += static_cast<std::size_t>(n);
    }
    return true;
}

} // namespace

int main() {
    bool passed = true;

    // ---- buffer pool: pooled storage is reused, concurrent borrows differ ----
    {
        auto pool = std::make_shared<sim_receiver::BufferPool>();
        auto first = pool->acquire(4096);
        const void* address = first->data();
        passed &= expect(first->size() == 4096, "pooled buffer has the requested size");
        first.reset();   // sole owner gone -> storage returns to the pool
        passed &= expect(pool->freeCount() == 1, "released buffer returns to the pool");
        auto reused = pool->acquire(2048);
        passed &= expect(reused->data() == address, "pooled storage is reused");
        auto second = pool->acquire(2048);
        passed &= expect(second->data() != reused->data(),
                         "two live borrows never alias");
    }

    // ---- synthetic high-rate sender -------------------------------------------
    cv::Mat scene(1080, 1440, CV_8UC3, cv::Scalar(20, 20, 25));
    cv::putText(scene, "sim receiver pipeline test", cv::Point(80, 540),
                cv::FONT_HERSHEY_SIMPLEX, 3.0, cv::Scalar(240, 240, 240), 6);
    std::vector<uint8_t> jpeg;
    {
        std::vector<uint8_t> encoded;
        const std::vector<int> params{cv::IMWRITE_JPEG_QUALITY, 95};
        passed &= expect(cv::imencode(".jpg", scene, encoded, params),
                         "test scene encodes to JPEG");
        jpeg = std::move(encoded);
    }

    std::promise<uint16_t> port_promise;
    std::atomic<bool> stop{false};
    std::atomic<uint64_t> sent_frames{0};
    double send_seconds = 0.0;

    std::thread server([&] {
        const int listen_fd = ::socket(AF_INET, SOCK_STREAM, 0);
        sockaddr_in address{};
        address.sin_family = AF_INET;
        address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        address.sin_port = 0;
        int enabled = 1;
        setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &enabled, sizeof(enabled));
        if (::bind(listen_fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0 ||
            ::listen(listen_fd, 1) != 0) {
            port_promise.set_value(0);
            ::close(listen_fd);
            return;
        }
        socklen_t length = sizeof(address);
        ::getsockname(listen_fd, reinterpret_cast<sockaddr*>(&address), &length);
        port_promise.set_value(ntohs(address.sin_port));

        const int conn = ::accept(listen_fd, nullptr, nullptr);
        if (conn < 0) {
            ::close(listen_fd);
            return;
        }

        std::vector<uint8_t> packet;
        const auto start = std::chrono::steady_clock::now();
        while (!stop.load()) {
            packet.clear();
            packet.reserve(40 + jpeg.size());
            writeU32(packet, 0x44414544);           // magic "DAED"
            writeU32(packet, 2);                    // version 2: + 真实云台姿态
            writeU32(packet, 1440);                 // width
            writeU32(packet, 1080);                 // height
            writeU32(packet, static_cast<uint32_t>(jpeg.size()));
            writeU64(packet, 1700000000000000ull);  // source (exposure) timestamp
            writeU32(packet, static_cast<uint32_t>(sent_frames.load() + 1));
            writeF32(packet, 0.25f);                // gimbal_yaw
            writeF32(packet, -0.10f);               // gimbal_pitch
            packet.insert(packet.end(), jpeg.begin(), jpeg.end());
            if (!sendAll(conn, packet.data(), packet.size())) {
                break;   // client went away
            }
            sent_frames.fetch_add(1);
        }
        send_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start).count();
        ::close(conn);
        ::close(listen_fd);
    });

    const uint16_t port = port_promise.get_future().get();
    passed &= expect(port != 0, "test server bound a port");
    if (port == 0) {
        stop.store(true);
        server.join();
        return 1;
    }

    sim_receiver::VisionDateReceiver receiver("/nonexistent/simulator.yaml");
    receiver.overrideEndpoint("127.0.0.1", port);
    passed &= expect(receiver.connect(), "receiver connects to the test server");

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(1500);
    uint64_t delivered = 0;
    uint64_t last_sequence = 0;
    bool sequence_monotonic = true;
    bool geometry_ok = true;
    bool gimbal_ok = true;
    while (std::chrono::steady_clock::now() < deadline) {
        uint64_t sequence = 0;
        uint64_t source_us = 0;
        uint64_t local_us = 0;
        float gimbal_yaw = 0.0f;
        float gimbal_pitch = 0.0f;
        const cv::Mat frame = receiver.getFrame(&sequence, &source_us, &local_us,
                                                &gimbal_yaw, &gimbal_pitch);
        if (frame.empty()) {
            continue;
        }
        ++delivered;
        // v2 包头必须把真实云台姿态原样带出来（测试服务端写的是 0.25 / -0.10）。
        if (std::abs(gimbal_yaw - 0.25f) > 1e-6f ||
            std::abs(gimbal_pitch + 0.10f) > 1e-6f) {
            gimbal_ok = false;
        }
        if (sequence < last_sequence) {
            sequence_monotonic = false;
        }
        last_sequence = sequence;
        if (frame.cols != 1440 || frame.rows != 1080 || source_us == 0 || local_us == 0) {
            geometry_ok = false;
        }
    }

    stop.store(true);
    server.join();

    const uint64_t sent = sent_frames.load();
    const double sent_fps = send_seconds > 0.0 ? sent / send_seconds : 0.0;
    std::cout << "sent=" << sent << " (" << sent_fps << " fps) delivered=" << delivered
              << " dropped=" << receiver.droppedPackets() << std::endl;

    passed &= expect(delivered > 0, "receiver delivers decoded frames");
    passed &= expect(geometry_ok, "frames keep their size and timestamps");
    passed &= expect(gimbal_ok, "v2 header carries the real gimbal pose");
    passed &= expect(sequence_monotonic, "the consumer never sees an older frame after a newer one");
    // The sender free-runs (loopback is far faster than the decoders), while the
    // consumer is bounded by decode throughput.  A serial recv+decode client
    // would instead have throttled the sender down to what it could decode.
    passed &= expect(sent > 3 * delivered,
                     "sender is not throttled to the client's decode rate");
    passed &= expect(receiver.droppedPackets() > 0,
                     "stale frames are dropped instead of queued");

    if (passed) {
        std::cout << "sim_receiver_pipeline_test: OK" << std::endl;
    }
    return passed ? 0 : 1;
}
