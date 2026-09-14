#ifndef CLAP_THREAD_CHECK_H
#define CLAP_THREAD_CHECK_H

#include <clap/clap.h>
#include <atomic>
#include <thread>

namespace emt {

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
// We also expose a static-std::thread::id cache that the plugin can fall back
// to if the host does not implement thread-check.
class ThreadCheck {
public:
    ThreadCheck() : host_(nullptr), ext_(nullptr) {
        main_thread_id_ = std::this_thread::get_id();
        audio_thread_id_.store(std::thread::id{}, std::memory_order_relaxed);
        has_audio_id_.store(false, std::memory_order_relaxed);
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

    // Main-thread predicate. Falls back to std::thread::id comparison if
    // the host does not provide the extension.
    bool is_main_thread() const {
        if (ext_ && ext_->is_main_thread) {
            return ext_->is_main_thread(host_);
        }
        return std::this_thread::get_id() == main_thread_id_;
    }

    // Audio-thread predicate. We use the host's extension as the source of
    // truth; the cached audio thread id is only used as a coarse fallback
    // (recorded the first time we observe a process() call).
    bool is_audio_thread() const {
        if (ext_ && ext_->is_audio_thread) {
            return ext_->is_audio_thread(host_);
        }
        if (has_audio_id_.load(std::memory_order_acquire)) {
            return std::this_thread::get_id() ==
                audio_thread_id_.load(std::memory_order_acquire);
        }
        return false;
    }

    // Called from process() to record the audio thread id when the host
    // does not provide the extension. Idempotent.
    void mark_audio_thread() {
        bool expected = false;
        if (has_audio_id_.compare_exchange_strong(
                expected, true, std::memory_order_acq_rel)) {
            audio_thread_id_.store(std::this_thread::get_id(),
                std::memory_order_release);
        }
    }

    const clap_host_thread_check_t* host_ext() const { return ext_; }

private:
    const clap_host_t* host_;
    const clap_host_thread_check_t* ext_;
    std::thread::id main_thread_id_;
    std::atomic<std::thread::id> audio_thread_id_;
    std::atomic<bool> has_audio_id_;
};

} // namespace emt

#endif
