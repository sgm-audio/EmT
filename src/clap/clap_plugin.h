#ifndef MAMBA_CLAP_PLUGIN_H
#define MAMBA_CLAP_PLUGIN_H

#include <clap/clap.h>
#include <atomic>
#include <cstdint>
#include <array>
#include <cstring>
#include <cmath>

#include "core/ssm_inference_core.h"
#include "core/weight_loader.h"
#include "core/arena_allocator.h"
#include "common/ring_buffer.h"
#include "common/seqlock.h"
#include "common/conditioning_frame.h"
#include "common/double_buffer_seqlock.h"
#include "clap/clap_thread_check.h"
#include "ara/ara_document_controller.h"
#include "config.h"

namespace mamba {

enum ParamIds : clap_id {
    PARAM_GAIN = 0,
    PARAM_ATTACK = 1,
    PARAM_RELEASE = 2,
    PARAM_MIX = 3,
    PARAM_BYPASS = 4,
    PARAM_COUNT = 5
};

enum ParamCurve : uint8_t {
    PARAM_CURVE_LINEAR = 0,
    PARAM_CURVE_LOG    = 1,
    PARAM_CURVE_STEPPED = 2,
};

struct ParamAtom {
    clap_id id;
    double value;
    uint32_t frame_offset;
};

// Per-parameter smoothing state. The target is the latest value the
// host automation asked for; the current is what the audio thread is
// reading. Both are atomic so the host thread (via flush) can update
// targets and the audio thread can read them.
struct alignas(64) ParamSmooth {
    std::atomic<double> target{0.0};
    std::atomic<double> current{0.0};
    double min_value{0.0};
    double max_value{1.0};
    double default_value{0.0};
    ParamCurve curve{PARAM_CURVE_LINEAR};
    float smooth_time_s{0.010f};
    double coeff{0.0};
};

enum BypassState : uint8_t {
    BYPASS_PROCESSING    = 0,
    BYPASS_FADING_OUT    = 1,
    BYPASS_BYPASSED      = 2,
    BYPASS_FADING_IN     = 3,
};

enum VoiceState : uint8_t {
    VOICE_IDLE     = 0,
    VOICE_ATTACK   = 1,
    VOICE_SUSTAIN  = 2,
    VOICE_RELEASE  = 3,
    VOICE_SILENT   = 4,
};

static constexpr uint32_t MAX_VOICES = 8;

// Per-voice state (audio thread only, not atomic).
// Each voice holds its own SSM state snapshot for save/restore with
// the shared SSMInferenceCore.
struct alignas(64) Voice {
    MambaState ssm_state;       // SSM state snapshot
    float env;                  // envelope amplitude 0..1
    float attack_rate;          // envelope attack increment per sample
    float release_rate;         // envelope release decrement per sample
    uint8_t stage;              // VoiceState: IDLE/ATTACK/SUSTAIN/RELEASE/SILENT
    float velocity;             // note velocity 0..1
    int32_t key;                // MIDI key
    int32_t channel;            // MIDI channel
    uint64_t note_id;           // unique note ID for matching note-off
};

// Per-voice snapshot for state save (trivially copyable for seqlock).
struct alignas(64) VoiceSnapshot {
    MambaState ssm_state;
    float env;
    uint8_t stage;  // VoiceState
    uint8_t pad[3]; // padding
};

constexpr size_t kMaxWeightPath = 512;

// Snapshot used for state save. The audio thread writes to the back
// slot of a double-buffer seqlock each process() call; the main thread
// reads from the front slot when the host asks for a state save.
// Must be trivially copyable for DoubleBufferSeqLock.
struct alignas(64) StateMirror {
    VoiceSnapshot voice_snapshots[MAX_VOICES];
    double param_values[PARAM_COUNT];
    bool bypass_target;
    float bypass_fade;
    char weight_path[kMaxWeightPath];
};

// Per-voice atomic flags visible to the main thread. NOT part of
// StateMirror (to keep it trivially copyable). Updated by audio thread,
// read by main thread for UI.
struct alignas(64) VoiceAtomics {
    std::atomic<uint8_t> stage{VOICE_IDLE};
    std::atomic<float> env{0.0f};
    std::atomic<int32_t> key{-1};
    std::atomic<int32_t> channel{-1};
    std::atomic<float> velocity{0.0f};
};

struct MambaSynth {
    const clap_host_t* host;

    alignas(64) std::byte core_storage[sizeof(SSMInferenceCore)];
    SSMInferenceCore* core;
    MambaConfig core_cfg;

    double sample_rate;
    uint32_t block_size;
    bool is_active;
    bool is_processing;

    std::atomic<bool> weights_loaded;

    // SPSC queue from main thread to audio thread for parameter changes
    // delivered with sample-accurate offsets via params_flush().
    SPSCRingBuffer<ParamAtom, 256> param_queue;

    // Per-parameter smoothing state, with cache-line alignment on the
    // first one. The audio thread reads target/current atomically.
    alignas(64) ParamSmooth params[PARAM_COUNT];
    std::atomic<double> param_values[PARAM_COUNT];

    // Bypass state machine.
    std::atomic<bool> bypass_target;
    alignas(64) std::atomic<uint32_t> bypass_state;
    alignas(64) std::atomic<float> bypass_fade;
    static constexpr float BYPASS_FADE_RATE = 1.0f / 64.0f;

    // Voice pool: N voices sharing one SSMInferenceCore.
    // Each voice has its own SSM state (saved/restored per-sample).
    // The atomic mirrors are for the main thread (state save, UI).
    alignas(64) Voice voices[MAX_VOICES];
    alignas(64) VoiceAtomics voice_atomics[MAX_VOICES];
    uint64_t next_note_id;      // monotonic, for matching note-on/off

    static constexpr float VOICE_ATTACK_RATE  = 1.0f / 1024.0f;
    static constexpr float VOICE_RELEASE_RATE = 1.0f / 4096.0f;

    // Conditioning (main thread writes, audio thread reads via seqlock).
    Seqlocked<ConditioningFrame> conditioning;
    Seqlocked<ARAConditioningFrame> ara_conditioning;

    // Snapshot of h_state for state save. The audio thread writes to
    // the back slot; main thread reads from front.
    alignas(64) DoubleBufferSeqLock<StateMirror> state_mirror;

    uint32_t latency;

    static constexpr size_t MAX_WEIGHT_PATH = kMaxWeightPath;
    alignas(64) char weight_path[kMaxWeightPath];
    WeightLoader weight_loader;

    // Host thread-check extension cache.
    alignas(64) ThreadCheck thread_check;

    bool load_weights(const char* path);
};

extern const clap_plugin_descriptor_t mamba_synth_descriptor;

extern const clap_plugin_audio_ports_t s_audio_ports;
extern const clap_plugin_note_ports_t s_note_ports;
extern const clap_plugin_params_t s_params;
extern const clap_plugin_state_t s_state;
extern const clap_plugin_tail_t s_tail;
extern const clap_plugin_latency_t s_latency;
extern const clap_plugin_gui_t s_gui;

} // namespace mamba

#endif
