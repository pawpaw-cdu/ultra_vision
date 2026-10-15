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

#include "sim_packet.hpp"   // sim_receiver::BufferPool

#include <arpa/inet.h>
#include <cstring>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <cstdio>
#include <fstream>
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

// ---- UDP 分片流（协议见 simulator_system/API.md §1.1）----------------------

constexpr std::size_t kUdpHeaderLen = 40;
constexpr std::size_t kUdpChunk = 1400;

void appendUdpFragment(std::vector<uint8_t>& out, uint32_t frame_id, uint64_t timestamp_us,
                       uint32_t total_len, uint32_t offset, const uint8_t* chunk,
                       uint32_t chunk_len, float gimbal_yaw, float gimbal_pitch) {
    out.clear();
    writeU32(out, 0x44414544);                          // magic "DAED"
    out.push_back(0);                                   // version = 1 (u16 BE)
    out.push_back(1);
    out.push_back(static_cast<uint8_t>(kUdpHeaderLen >> 8));   // header_len = 40
    out.push_back(static_cast<uint8_t>(kUdpHeaderLen));
    writeU32(out, frame_id);
    writeU64(out, timestamp_us);
    writeU32(out, total_len);
    writeU32(out, offset);
    writeU32(out, chunk_len);
    writeF32(out, gimbal_yaw);
    writeF32(out, gimbal_pitch);
    out.insert(out.end(), chunk, chunk + chunk_len);
}

bool sendUdpFrame(int fd, const sockaddr_in& target, const std::vector<uint8_t>& jpeg,
                  uint32_t frame_id, uint64_t timestamp_us, float gimbal_yaw,
                  float gimbal_pitch, bool drop_middle_fragment) {
    std::vector<uint8_t> datagram;
    const uint32_t total_len = static_cast<uint32_t>(jpeg.size());
    std::size_t fragment = 0;
    for (std::size_t offset = 0; offset < jpeg.size(); offset += kUdpChunk) {
        const auto len = static_cast<uint32_t>(std::min(kUdpChunk, jpeg.size() - offset));
        ++fragment;
        if (drop_middle_fragment && fragment == 2) {
            continue;   // 故意丢一片：整帧拼不齐，接收端必须整帧丢弃
        }
        appendUdpFragment(datagram, frame_id, timestamp_us, total_len,
                          static_cast<uint32_t>(offset), jpeg.data() + offset, len,
                          gimbal_yaw, gimbal_pitch);
        if (::sendto(fd, datagram.data(), datagram.size(), 0,
                     reinterpret_cast<const sockaddr*>(&target), sizeof(target)) < 0) {
            return false;
        }
    }
    return true;
}

// 合成一个 UDP "仿真器"：先等接收端的订阅报文，再推 3 帧（第 2 帧缺一片）。
bool runUdpStreamTest() {
    bool ok = true;

    cv::Mat scene(1080, 1440, CV_8UC3, cv::Scalar(25, 20, 20));
    cv::putText(scene, "udp fragment test", cv::Point(80, 540),
                cv::FONT_HERSHEY_SIMPLEX, 3.0, cv::Scalar(240, 240, 240), 6);
    std::vector<uint8_t> jpeg;
    {
        std::vector<uint8_t> encoded;
        const std::vector<int> params{cv::IMWRITE_JPEG_QUALITY, 95};
        ok &= expect(cv::imencode(".jpg", scene, encoded, params), "udp test scene encodes");
        jpeg = std::move(encoded);
    }

    const int sim_fd = ::socket(AF_INET, SOCK_DGRAM, 0);
    sockaddr_in sim_addr{};
    sim_addr.sin_family = AF_INET;
    sim_addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    sim_addr.sin_port = 0;
    if (sim_fd < 0 ||
        ::bind(sim_fd, reinterpret_cast<sockaddr*>(&sim_addr), sizeof(sim_addr)) != 0) {
        if (sim_fd >= 0) ::close(sim_fd);
        return expect(false, "udp test server bound a port");
    }
    socklen_t sim_len = sizeof(sim_addr);
    ::getsockname(sim_fd, reinterpret_cast<sockaddr*>(&sim_addr), &sim_len);
    const uint16_t sim_port = ntohs(sim_addr.sin_port);

    // 临时配置：走 UDP 那条路（端口用刚抢占的临时端口）。
    const std::string config_path =
        "/tmp/sim_receiver_udp_test_" + std::to_string(::getpid()) + ".yaml";
    {
        std::ofstream cfg(config_path, std::ios::out | std::ios::trunc);
        cfg << "host: 127.0.0.1\n"
            << "port: 7666\n"
            << "protocol: udp\n"
            << "udp_port: " << sim_port << '\n';
    }

    std::atomic<bool> subscribed{false};
    std::thread sender([&] {
        // 1) 等订阅报文：接收端 connect() 会立刻发一个（内容任意，只认来源地址）。
        timeval tv{};
        tv.tv_sec = 3;
        setsockopt(sim_fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
        char buffer[64];
        sockaddr_in client{};
        socklen_t client_len = sizeof(client);
        if (::recvfrom(sim_fd, buffer, sizeof(buffer), 0,
                       reinterpret_cast<sockaddr*>(&client), &client_len) >= 0) {
            subscribed.store(true);
        }
        // 2) 帧 1 完整 / 帧 2 缺片 / 帧 3 完整且时间戳更新。
        const auto pace = [](int ms) {
            std::this_thread::sleep_for(std::chrono::milliseconds(ms));
        };
        sendUdpFrame(sim_fd, client, jpeg, 1, 1700000000000000ull, 0.30f, -0.20f, false);
        pace(250);
        sendUdpFrame(sim_fd, client, jpeg, 2, 1700000000100000ull, 0.31f, -0.21f, true);
        pace(250);
        sendUdpFrame(sim_fd, client, jpeg, 3, 1700000000200000ull, 0.32f, -0.22f, false);
        pace(250);
    });

    std::vector<uint64_t> sequences;
    bool geometry_ok = true;
    {
        sim_receiver::VisionDateReceiver receiver(config_path);
        ok &= expect(receiver.connect(), "udp receiver binds a socket and subscribes");

        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(1200);
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
            sequences.push_back(sequence);
            if (frame.cols != 1440 || frame.rows != 1080 || source_us == 0 || local_us == 0) {
                geometry_ok = false;
            }
            // 每片都带云台姿态，帧 3 写的是 0.32 / -0.22。
            if (sequence == 3 &&
                (std::abs(gimbal_yaw - 0.32f) > 1e-6f ||
                 std::abs(gimbal_pitch + 0.22f) > 1e-6f)) {
                geometry_ok = false;
            }
        }
    }

    sender.join();
    ::close(sim_fd);
    std::remove(config_path.c_str());

    std::cout << "udp delivered seqs=";
    for (uint64_t sequence : sequences) std::cout << sequence << ' ';
    std::cout << std::endl;

    ok &= expect(subscribed.load(), "the simulator sees the subscription datagram");
    ok &= expect(geometry_ok, "udp frames keep their size, timestamps and gimbal pose");
    ok &= expect(std::find(sequences.begin(), sequences.end(), 2) == sequences.end(),
                 "a frame with a missing fragment is never delivered");
    ok &= expect(std::find(sequences.begin(), sequences.end(), 3) != sequences.end(),
                 "the newest complete udp frame is delivered");
    ok &= expect(!sequences.empty() && sequences.back() == 3,
                 "udp delivery never goes backwards");
    return ok;
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

    // ---- UDP 分片图像流（protocol: udp）---------------------------------------
    passed &= runUdpStreamTest();

    if (passed) {
        std::cout << "sim_receiver_pipeline_test: OK" << std::endl;
    }
    return passed ? 0 : 1;
}
