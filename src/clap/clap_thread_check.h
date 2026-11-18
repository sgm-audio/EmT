#ifndef CLAP_THREAD_CHECK_H
#define CLAP_THREAD_CHECK_H

#include <clap/clap.h>
#include <array>
#include <atomic>
#include <cstdint>
#include <functional>
#include <thread>

namespace mamba {

// Helper around the host-provided thread-check extension (clap.thread-check).
//
// clap.thread-check is a HOST extension, not a plugin extension: there is no
// clap_plugin_thread_check_t. Plugins that need to verify they are running on
// the audio thread (or not on it) query the host's extension and call
// is_audio_thread() / is_main_thread().
//
// The recommended pattern is to cache the extension pointer at activation
// time on the main thread, and re-check on each interesting call. We do that
// by stashing the host extension pointer in the plugin struct, then offering
// small inline helpers used from process() and the various flush hooks.
//
// We also expose a lock-free multi-thread fallback set for hosts that process
// audio across a worker pool without implementing clap.thread-check.
class ThreadCheck {
public:
    static constexpr size_t kMaxAudioThreads = 32;
    static_assert(std::atomic<uint64_t>::is_always_lock_free,
                  "ThreadCheck requires lock-free 64-bit atomics");

    ThreadCheck() : host_(nullptr), ext_(nullptr) {
        main_thread_token_ = current_thread_token();
    }

    // Must be called from the main thread (e.g. during plugin_init or first
    // safe point). The host is allowed to change its extension pointer, but
    // typically does not.
    void set_host(const clap_host_t* host) {
        host_ = host;
        if (host && host->get_extension) {
            ext_ = static_cast<const clap_host_thread_check_t*>(
                host->get_extension(host, CLAP_EXT_THREAD_CHECK));
        } else {
            ext_ = nullptr;
        }
    }

    // Main-thread predicate. Falls back to thread token comparison if
    // the host does not provide the extension.
    bool is_main_thread() const {
        if (ext_ && ext_->is_main_thread) {
            return ext_->is_main_thread(host_);
        }
        return current_thread_token() == main_thread_token_;
    }

    // Audio-thread predicate. We use the host's extension as the source of
    // truth; the lock-free audio thread token set is used as a fallback
    // (recorded whenever process() runs on a thread).
    bool is_audio_thread() const {
        if (ext_ && ext_->is_audio_thread) {
            return ext_->is_audio_thread(host_);
        }
        const uint64_t token = current_thread_token();
        for (size_t i = 0; i < kMaxAudioThreads; ++i) {
            const uint64_t slot = audio_threads_[i].load(std::memory_order_acquire);
            if (slot == 0) break;
            if (slot == token) return true;
        }
        return false;
    }

    // Called from process() to record audio worker thread tokens when the host
    // does not provide the extension. Lock-free and idempotent.
    void mark_audio_thread() {
        const uint64_t token = current_thread_token();
        for (size_t i = 0; i < kMaxAudioThreads; ++i) {
            uint64_t cur = audio_threads_[i].load(std::memory_order_acquire);
            if (cur == token) return;
            if (cur == 0) {
                uint64_t expected = 0;
                if (audio_threads_[i].compare_exchange_strong(
                        expected, token, std::memory_order_acq_rel, std::memory_order_acquire)) {
                    return;
                }
                if (expected == token) return;
            }
        }
    }

    const clap_host_thread_check_t* host_ext() const { return ext_; }

private:
    static uint64_t current_thread_token() noexcept {
        uint64_t h = static_cast<uint64_t>(std::hash<std::thread::id>{}(std::this_thread::get_id()));
        return h == 0 ? 1u : h;
    }

    const clap_host_t* host_;
    const clap_host_thread_check_t* ext_;
    uint64_t main_thread_token_{0};
    std::array<std::atomic<uint64_t>, kMaxAudioThreads> audio_threads_{};
};

} // namespace mamba

#endif
