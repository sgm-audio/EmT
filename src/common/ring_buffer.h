#ifndef RING_BUFFER_H
#define RING_BUFFER_H

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace emt {

template<typename T, size_t Capacity>
class SPSCRingBuffer {
public:
    static_assert((Capacity & (Capacity - 1)) == 0, "Capacity must be power of 2");
    static_assert(Capacity <= 1024, "Capacity too large for lock-free implementation");

    SPSCRingBuffer() : head_(0), tail_(0) {
        std::memset(buffer_, 0, sizeof(buffer_));
    }

    bool push(const T& item) {
        uint64_t head = head_.load(std::memory_order_relaxed);
        uint64_t next_head = (head + 1) & (Capacity - 1);

        if (next_head == tail_.load(std::memory_order_acquire)) {
            return false;
        }

        buffer_[head] = item;
        head_.store(next_head, std::memory_order_release);
        return true;
    }

    bool pop(T& item) {
        uint64_t tail = tail_.load(std::memory_order_relaxed);

        if (tail == head_.load(std::memory_order_acquire)) {
            return false;
        }

        item = buffer_[tail];
        tail_.store((tail + 1) & (Capacity - 1), std::memory_order_release);
        return true;
    }

    bool is_empty() const {
        return head_.load(std::memory_order_acquire) == tail_.load(std::memory_order_acquire);
    }

    bool is_full() const {
        uint64_t head = head_.load(std::memory_order_acquire);
        uint64_t next_head = (head + 1) & (Capacity - 1);
        return next_head == tail_.load(std::memory_order_acquire);
    }

    size_t size() const {
        uint64_t head = head_.load(std::memory_order_acquire);
        uint64_t tail = tail_.load(std::memory_order_acquire);
        return (head - tail) & (Capacity - 1);
    }

private:
    alignas(64) T buffer_[Capacity];
    alignas(64) std::atomic<uint64_t> head_;
    alignas(64) std::atomic<uint64_t> tail_;
};

} // namespace emt

#endif