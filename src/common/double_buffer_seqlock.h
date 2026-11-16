#ifndef DOUBLE_BUFFER_SEQLOCK_H
#define DOUBLE_BUFFER_SEQLOCK_H

#include <atomic>
#include <cstdint>
#include <cstring>
#include <new>
#include <type_traits>

namespace mamba {

// Triple-buffer lock-free snapshot exchange per ADR-001.
//
// The producer writes to an exclusive non-front slot returned by
// begin_write() and publishes it atomically on end_write().
// Readers pin the current front slot while copying it so the producer
// never overwrites a slot a reader is actively reading, even if the
// producer laps or swap() is omitted.
//
// Pattern:
//   - producer calls begin_write() to get an exclusive slot pointer,
//     populates it, then calls end_write() (swap() is kept for API
//     compatibility and is an optional no-op after end_write()).
//   - consumer calls read(out) to obtain a consistent, tear-free
//     snapshot without blocking.
class DoubleBufferSeqLockCore {
public:
    static constexpr uint32_t kSlots = 3;

    // Construct in caller-provided storage. slot_storage must be at
    // least kSlots * slot_size bytes and aligned to alignof(T).
    void init(void* slot_storage, size_t slot_size, size_t align) {
        slot_size_ = slot_size;
        slots_ = static_cast<std::byte*>(slot_storage);
        std::memset(slots_, 0, slot_size * kSlots);
        (void)align;
        front_.store(0, std::memory_order_relaxed);
        writer_ = 1;
        writing_.store(false, std::memory_order_relaxed);
        for (uint32_t i = 0; i < kSlots; ++i) {
            readers_[i].store(0, std::memory_order_relaxed);
        }
        generation_.store(0, std::memory_order_relaxed);
    }

    void* begin_write() {
        bool expected = false;
        if (!writing_.compare_exchange_strong(expected, true, std::memory_order_acquire)) {
            return nullptr;
        }
        const uint32_t f = front_.load(std::memory_order_seq_cst);
        for (uint32_t off = 0; off < kSlots; ++off) {
            const uint32_t candidate = (writer_ + off) % kSlots;
            if (candidate != f && readers_[candidate].load(std::memory_order_seq_cst) == 0) {
                writer_ = candidate;
                return &slots_[candidate * slot_size_];
            }
        }
        writing_.store(false, std::memory_order_release);
        return nullptr;
    }

    void end_write() {
        if (!writing_.load(std::memory_order_acquire)) {
            return;
        }
        const uint32_t published = writer_;
        writer_ = (published + 1) % kSlots;
        front_.store(published, std::memory_order_seq_cst);
        generation_.fetch_add(1, std::memory_order_release);
        writing_.store(false, std::memory_order_release);
    }

    bool read(void* out) const {
        if (!out) return false;
        uint32_t f = front_.load(std::memory_order_seq_cst);
        for (int attempt = 0; attempt < 64; ++attempt) {
            readers_[f].fetch_add(1, std::memory_order_seq_cst);
            const uint32_t f2 = front_.load(std::memory_order_seq_cst);
            if (f2 == f) {
                std::memcpy(out, &slots_[f * slot_size_], slot_size_);
                readers_[f].fetch_sub(1, std::memory_order_release);
                return true;
            }
            readers_[f].fetch_sub(1, std::memory_order_release);
            f = f2;
        }
        return false;
    }

    void swap() {
        // Publication and slot rotation happen atomically in end_write().
    }

    uint64_t generation() const {
        return generation_.load(std::memory_order_acquire);
    }

    size_t slot_size() const { return slot_size_; }

private:
    std::byte* slots_;
    size_t slot_size_;
    uint32_t writer_{1};
    alignas(64) std::atomic<bool> writing_{false};
    alignas(64) std::atomic<uint32_t> front_{0};
    alignas(64) mutable std::atomic<uint32_t> readers_[kSlots]{};
    alignas(64) std::atomic<uint64_t> generation_{0};
};

// Typed wrapper. Storage is a fixed-size byte array embedded in the
// caller struct, so no heap allocation is needed.
template<typename T>
class DoubleBufferSeqLock {
    static_assert(std::is_trivially_copyable_v<T>,
        "DoubleBufferSeqLock<T> requires T to be trivially copyable");

public:
    DoubleBufferSeqLock() {
        core_.init(slots_, sizeof(T), alignof(T));
    }

    T* begin_write() {
        return reinterpret_cast<T*>(core_.begin_write());
    }

    void end_write() {
        core_.end_write();
    }

    bool read(T& out) const {
        return core_.read(&out);
    }

    void swap() { core_.swap(); }

    uint64_t generation() const { return core_.generation(); }

private:
    alignas(64) std::byte slots_[sizeof(T) * DoubleBufferSeqLockCore::kSlots];
    DoubleBufferSeqLockCore core_;
};

} // namespace mamba

#endif
