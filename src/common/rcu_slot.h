#ifndef MAMBA_RCU_SLOT_H
#define MAMBA_RCU_SLOT_H

#include <array>
#include <atomic>
#include <cstddef>
#include <new>
#include "common/ring_buffer.h"

namespace mamba {

// Single-writer / single-reader RCU pointer swap between the main thread
// (writer) and the audio thread (reader).
//
//   Writer (main thread): publish(T*), reclaim(), destroy(), published()
//   Reader (audio thread): adopt(), current()
//
// publish() makes a fully-built heap object visible; the audio thread
// adopt()s the latest object once per process() block and, when it moves to a
// new object, reports the one it just left via a retired SPSC queue. The writer
// frees reported objects in reclaim() (called before every publish and on
// shutdown). No lock, no allocation, no free ever runs on the reader.
//
// The published object must be immutable once published: the writer builds
// each new object from scratch and never mutates one that has been published.
// Uses fixed inline storage plus nothrow linked nodes so publish() never throws
// under -fno-exceptions and any skipped pointers are freed by destroy().
template <typename T>
class RcuSlot {
public:
    RcuSlot() = default;
    ~RcuSlot() { destroy(); }
    RcuSlot(const RcuSlot&) = delete;
    RcuSlot& operator=(const RcuSlot&) = delete;

    // Writer (main thread): publish the next object.
    void publish(T* next) {
        reclaim();
        T* old = published_.exchange(next, std::memory_order_acq_rel);
        if (old) push_hold(old);
    }

    // Reader (audio thread): return the current object (stable across the
    // whole block). Reports the previous object for reclamation.
    T* adopt() {
        T* p = published_.load(std::memory_order_acquire);
        if (p != active_) {
            if (active_) {
                (void)retired_queue_.push(active_);
            }
            active_ = p;
        }
        return active_;
    }

    T* current() const { return active_; }

    // Writer (main thread): the currently published object (null before the
    // first publish).
    T* published() const { return published_.load(std::memory_order_acquire); }

    // Writer (main thread): free objects the reader has reported. Call before
    // each publish and on any main-thread hook (e.g. on_main_thread).
    void reclaim() {
        T* p = nullptr;
        while (retired_queue_.pop(p)) {
            if (p) {
                remove_from_hold(p);
                delete p;
            }
        }
    }

    // Writer (main thread): free everything. The audio thread must be stopped.
    void destroy() {
        reclaim();
        if (T* p = published_.exchange(nullptr, std::memory_order_acq_rel)) delete p;
        for (size_t i = 0; i < hold_count_; ++i) {
            delete hold_[i];
            hold_[i] = nullptr;
        }
        hold_count_ = 0;
        while (overflow_head_) {
            HoldNode* n = overflow_head_;
            overflow_head_ = n->next;
            delete n->ptr;
            delete n;
        }
        active_ = nullptr;
    }

private:
    static constexpr size_t kInlineHoldCapacity = 64;

    struct HoldNode {
        T* ptr;
        HoldNode* next;
    };

    void push_hold(T* old) {
        if (hold_count_ < kInlineHoldCapacity) {
            hold_[hold_count_++] = old;
            return;
        }
        if (auto* node = new (std::nothrow) HoldNode{old, overflow_head_}) {
            overflow_head_ = node;
        }
    }

    void remove_from_hold(T* p) {
        for (size_t i = 0; i < hold_count_; ++i) {
            if (hold_[i] == p) {
                hold_[i] = hold_[--hold_count_];
                hold_[hold_count_] = nullptr;
                return;
            }
        }
        HoldNode** curr = &overflow_head_;
        while (*curr) {
            if ((*curr)->ptr == p) {
                HoldNode* victim = *curr;
                *curr = victim->next;
                delete victim;
                return;
            }
            curr = &((*curr)->next);
        }
    }

    alignas(64) std::atomic<T*> published_{nullptr};
    alignas(64) SPSCRingBuffer<T*, 64> retired_queue_;
    alignas(64) T* active_ = nullptr;   // reader only
    std::array<T*, kInlineHoldCapacity> hold_{}; // writer only
    size_t hold_count_ = 0;             // writer only
    HoldNode* overflow_head_ = nullptr; // writer only
};

} // namespace mamba

#endif
