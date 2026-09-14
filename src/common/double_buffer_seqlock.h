#ifndef DOUBLE_BUFFER_SEQLOCK_H
#define DOUBLE_BUFFER_SEQLOCK_H

#include <atomic>
#include <cstdint>
#include <cstring>
#include <new>
#include <type_traits>

namespace mamba {

// Double-buffer seqlock per ADR-001.
//
// The audio (producer) thread writes to a "back" slot while readers
// observe the "front" slot. A swap is performed by the main thread
// (or any non-audio thread) so that the audio thread never blocks.
//
// Pattern:
//   - producer (audio thread) calls begin_write() to get a reference
//     to the writable slot, copies/fills it, then calls end_write().
//   - consumer (main thread) calls read() to get a stable snapshot
//     of the most recently completed slot. read() is wait-free and
//     does not block.
//   - swap() is called by the main thread; it atomically exchanges
//     the front/back indices and bumps a generation counter so the
//     audio thread can detect the swap and rebind to the new back.
//   - On the audio thread, between calls to process(), the producer
//     is exclusive owner of the back slot, so no synchronization
//     beyond a release/acquire on the generation is needed inside
//     the begin/end pair.
class DoubleBufferSeqLockCore {
public:
    static constexpr uint32_t kSlots = 2;

    // Construct in caller-provided storage. slot_storage must be at
    // least slot_count * slot_size bytes and aligned to alignof(T).
    void init(void* slot_storage, size_t slot_size, size_t align) {
        slot_size_ = slot_size;
        slots_ = static_cast<std::byte*>(slot_storage);
        // zero storage
        for (size_t i = 0; i < slot_size * kSlots; ++i) {
            slots_[i] = std::byte{0};
        }
        (void)align;
        front_.store(0, std::memory_order_relaxed);
        writer_.store(0, std::memory_order_relaxed);
        generation_.store(0, std::memory_order_relaxed);
    }

    void* begin_write() {
        return &slots_[writer_.load(std::memory_order_acquire) * slot_size_];
    }

    void end_write() {
        uint32_t cur = writer_.load(std::memory_order_relaxed);
        front_.store(cur, std::memory_order_release);
        generation_.fetch_add(1, std::memory_order_acq_rel);
    }

    bool read(void* out) const {
        uint64_t g1 = generation_.load(std::memory_order_acquire);
        uint32_t f = front_.load(std::memory_order_acquire);
        std::memcpy(out, &slots_[f * slot_size_], slot_size_);
        uint64_t g2 = generation_.load(std::memory_order_acquire);
        return g1 == g2;
    }

    void swap() {
        uint32_t cur = writer_.load(std::memory_order_acquire);
        uint32_t next = cur ^ 1u;
        writer_.store(next, std::memory_order_release);
    }

    uint64_t generation() const {
        return generation_.load(std::memory_order_acquire);
    }

    size_t slot_size() const { return slot_size_; }

private:
    std::byte* slots_;
    size_t slot_size_;
    alignas(64) std::atomic<uint32_t> front_;
    alignas(64) std::atomic<uint32_t> writer_;
    alignas(64) std::atomic<uint64_t> generation_;
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
