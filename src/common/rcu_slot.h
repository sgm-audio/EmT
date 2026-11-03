#ifndef MAMBA_RCU_SLOT_H
#define MAMBA_RCU_SLOT_H

#include <atomic>
#include <cstddef>
#include <vector>

namespace mamba {

// Single-writer / single-reader RCU pointer swap between the main thread
// (writer) and the audio thread (reader).
//
//   Writer (main thread): publish(T*), reclaim(), destroy(), published()
//   Reader (audio thread): adopt(), current()
//
// publish() makes a fully-built heap object visible; the audio thread
// adopt()s the latest object once per process() block and, when it moves to a
// new object, reports the one it just left via a retired slot. The writer
// frees reported objects in reclaim() (called before every publish and on
// shutdown). No lock, no allocation, no free ever runs on the reader.
//
// The published object must be immutable once published: the writer builds
// each new object from scratch and never mutates one that has been published.
// An object the reader skips (two publishes within one reader interval) stays
// in hold_ and is freed by destroy(); with weight loads that is a bounded,
// short-lived case (one engine object per skipped reload).
template <typename T>
class RcuSlot {
public:
    RcuSlot() = default;
    RcuSlot(const RcuSlot&) = delete;
    RcuSlot& operator=(const RcuSlot&) = delete;

    // Writer (main thread): publish the next object.
    void publish(T* next) {
        T* old = published_.exchange(next, std::memory_order_release);
        if (old) hold_.push_back(old);
    }

    // Reader (audio thread): return the current object (stable across the
    // whole block). Reports the previous object for reclamation.
    T* adopt() {
        T* p = published_.load(std::memory_order_acquire);
        if (p != active_) {
            if (active_) retired_.store(active_, std::memory_order_release);
            active_ = p;
        }
        return active_;
    }

    T* current() const { return active_; }

    // Writer (main thread): the currently published object (null before the
    // first publish).
    T* published() const { return published_.load(std::memory_order_relaxed); }

    // Writer (main thread): free objects the reader has reported. Call before
    // each publish and on any main-thread hook (e.g. on_main_thread).
    void reclaim() {
        T* p = retired_.exchange(nullptr, std::memory_order_acquire);
        while (p) {
            remove_from_hold(p);
            delete p;
            p = retired_.exchange(nullptr, std::memory_order_acquire);
        }
    }

    // Writer (main thread): free everything. The audio thread must be stopped.
    void destroy() {
        reclaim();
        if (T* p = published_.exchange(nullptr, std::memory_order_acq_rel)) delete p;
        for (T* p : hold_) delete p;
        hold_.clear();
        active_ = nullptr;
    }

private:
    void remove_from_hold(T* p) {
        for (auto it = hold_.begin(); it != hold_.end(); ++it) {
            if (*it == p) {
                hold_.erase(it);
                return;
            }
        }
    }

    std::atomic<T*> published_{nullptr};
    std::atomic<T*> retired_{nullptr};
    T* active_ = nullptr;   // reader only
    std::vector<T*> hold_;  // writer only
};

} // namespace mamba

#endif
