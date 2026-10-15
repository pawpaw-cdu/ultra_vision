#include "sim_packet.hpp"

#include <opencv2/imgcodecs.hpp>

namespace sim_receiver {

BufferPool::Buffer BufferPool::acquire(std::size_t size) {
    std::shared_ptr<std::vector<uint8_t>> storage;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!free_.empty()) {
            storage = std::move(free_.back());
            free_.pop_back();
        }
    }
    if (!storage) {
        storage = std::make_shared<std::vector<uint8_t>>();
    }
    // Exact size: the decoder consumes the whole buffer, so it must not carry
    // stale bytes from a larger previous frame.
    if (storage->size() != size) {
        storage->resize(size);
    }

    // The deleter owns one reference and returns the storage to the pool when the
    // last borrower drops it.  weak_ptr keeps this correct even if the pool is
    // destroyed while a buffer is still in flight (endpoint reconfiguration).
    std::weak_ptr<BufferPool> weak = shared_from_this();
    return Buffer(storage.get(), [weak, storage](std::vector<uint8_t>*) mutable {
        if (auto pool = weak.lock()) {
            pool->release(std::move(storage));
        }
    });
}

std::size_t BufferPool::freeCount() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return free_.size();
}

void BufferPool::release(std::shared_ptr<std::vector<uint8_t>> buffer) {
    std::lock_guard<std::mutex> lock(mutex_);
    free_.push_back(std::move(buffer));
}

cv::Mat decodePacket(const SimPacket& packet) {
    if (!packet.payload || packet.payload->empty()) {
        return cv::Mat();
    }
    return cv::imdecode(*packet.payload, cv::IMREAD_COLOR);
}

} // namespace sim_receiver
