#ifndef SEQLOCK_H
#define SEQLOCK_H

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

namespace mamba {

class SeqLock {
public:
    SeqLock() : seq_(0) {}

    void write_lock() {
        seq_.fetch_add(1, std::memory_order_seq_cst);
    }

    void write_unlock() {
        seq_.fetch_add(1, std::memory_order_seq_cst);
    }

    bool read_begin(uint64_t& start_seq) const {
        start_seq = seq_.load(std::memory_order_seq_cst);
        return (start_seq & 1) == 0;
    }

    bool read_retry(uint64_t start_seq) const {
        uint64_t end_seq = seq_.load(std::memory_order_seq_cst);
        return start_seq != end_seq;
    }

    uint64_t load(std::memory_order order = std::memory_order_acquire) const {
        return seq_.load(order);
    }

private:
    alignas(64) std::atomic<uint64_t> seq_;
};

template<typename T>
struct Seqlocked {
    static_assert(std::is_trivially_copyable_v<T>,
                  "Seqlocked<T> requires T to be trivially copyable");
    static constexpr size_t kWords = (sizeof(T) + sizeof(uint64_t) - 1) / sizeof(uint64_t);

    alignas(64) std::atomic<uint64_t> words[kWords]{};
    SeqLock seqlock;

#if defined(__SANITIZE_THREAD__)
    void write(const T& value) {
        uint64_t buf[kWords]{};
        std::memcpy(buf, &value, sizeof(T));
        seqlock.write_lock();
        for (size_t i = 0; i < kWords; ++i) {
            words[i].store(buf[i], std::memory_order_release);
        }
        seqlock.write_unlock();
    }

    bool read(T& out) const {
        uint64_t buf[kWords];
        for (int attempt = 0; attempt < 4096; ++attempt) {
            uint64_t start_seq = 0;
            if (seqlock.read_begin(start_seq)) {
                for (size_t i = 0; i < kWords; ++i) {
                    buf[i] = words[i].load(std::memory_order_acquire);
                }
                if (!seqlock.read_retry(start_seq)) {
                    std::memcpy(&out, buf, sizeof(T));
                    return true;
                }
            }
        }
        return false;
    }
#else
    void write(const T& value) {
        uint64_t buf[kWords]{};
        std::memcpy(buf, &value, sizeof(T));
        seqlock.write_lock();
        std::atomic_thread_fence(std::memory_order_release);
        for (size_t i = 0; i < kWords; ++i) {
            words[i].store(buf[i], std::memory_order_relaxed);
        }
        seqlock.write_unlock();
    }

    bool read(T& out) const {
        uint64_t buf[kWords];
        for (int attempt = 0; attempt < 4096; ++attempt) {
            uint64_t start_seq = 0;
            if (seqlock.read_begin(start_seq)) {
                std::atomic_thread_fence(std::memory_order_acquire);
                for (size_t i = 0; i < kWords; ++i) {
                    buf[i] = words[i].load(std::memory_order_relaxed);
                }
                std::atomic_thread_fence(std::memory_order_acquire);
                if (!seqlock.read_retry(start_seq)) {
                    std::memcpy(&out, buf, sizeof(T));
                    return true;
                }
            }
        }
        return false;
    }
#endif
};

} // namespace mamba

#endif