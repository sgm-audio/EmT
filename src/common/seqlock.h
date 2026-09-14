#ifndef SEQLOCK_H
#define SEQLOCK_H

#include <atomic>
#include <cstdint>
#include <cstring>

namespace emt {

class SeqLock {
public:
    SeqLock() : seq_(0) {}

    void write_lock() {
        seq_.fetch_add(1, std::memory_order_acq_rel);
    }

    void write_unlock() {
        seq_.fetch_add(1, std::memory_order_acq_rel);
    }

    bool read_begin(uint64_t& start_seq) const {
        start_seq = seq_.load(std::memory_order_acquire);
        return (start_seq & 1) == 0;
    }

    bool read_retry(uint64_t start_seq) const {
        uint64_t end_seq = seq_.load(std::memory_order_acquire);
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
    alignas(64) T data;
    SeqLock seqlock;

    void write(const T& value) {
        seqlock.write_lock();
        data = value;
        seqlock.write_unlock();
    }

    bool read(T& out) const {
        uint64_t start_seq;
        for (;;) {
            if (seqlock.read_begin(start_seq)) {
                out = data;
                if (!seqlock.read_retry(start_seq)) {
                    return true;
                }
            }
        }
    }
};

} // namespace emt

#endif