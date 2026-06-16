#include "clap/clap_plugin.h"
#include "clap/clap_gui.h"
#include "config.h"
#include "core/arena_allocator.h"
#include <cstring>
#include <cmath>
#include <cstdio>
#include <algorithm>
#include <new>

namespace mamba {

// Plugin-lifetime arena: all plugin-owned memory is bump-allocated from
// this single mmap-backed region. No heap fragmentation, no malloc/free
// during audio processing.
static constexpr size_t kPluginArenaSize = 65536; // 64 KB
static ArenaAllocator s_plugin_arena(kPluginArenaSize);

static const char* s_features[] = {
    CLAP_PLUGIN_FEATURE_INSTRUMENT,
    CLAP_PLUGIN_FEATURE_SYNTHESIZER,
    CLAP_PLUGIN_FEATURE_STEREO,
    nullptr
};

static const clap_plugin_descriptor_t s_descriptor = {
    CLAP_VERSION_INIT,
    "com.mamba-ssm.synth",
    "Mamba Synth",
    "Mamba Audio",
    "https://mamba-audio.com",
    "",
    "",
    "0.3.0",
    "Mamba-3 SSM synthesizer with FiLM conditioning",
    reinterpret_cast<const char* const*>(s_features)
};

const clap_plugin_descriptor_t mamba_synth_descriptor = s_descriptor;

static MambaSynth* get_synth(const clap_plugin_t* plugin) {
    return static_cast<MambaSynth*>(plugin->plugin_data);
}

// Recompute smoothing coefficients after a sample-rate change. Called
// from activate() and from any code path that mutates smooth_time_s.
static void recompute_smooth_coefficients(MambaSynth* m) {
    const double sr = (m->sample_rate > 1.0) ? m->sample_rate : 48000.0;
    for (uint32_t i = 0; i < PARAM_COUNT; ++i) {
        ParamSmooth& p = m->params[i];
        if (p.smooth_time_s <= 0.0f) {
            p.coeff = 1.0; // instant
        } else {
            const double t = static_cast<double>(p.smooth_time_s) * sr;
            // 1 - exp(-1/t) for a one-pole lowpass with cutoff
            // 1/(2*pi*smooth_time_s).
            p.coeff = 1.0 - std::exp(-1.0 / t);
        }
    }
}

// Initialize per-parameter metadata (range, curve, smoothing time).
// Defaults: linear, 10ms smoothing, ranges from previous design.
static void init_param_metadata(MambaSynth* m) {
    auto& p = m->params;

    p[PARAM_GAIN].min_value = 0.0;
    p[PARAM_GAIN].max_value = 2.0;
    p[PARAM_GAIN].default_value = 1.0;
    p[PARAM_GAIN].curve = PARAM_CURVE_LOG;
    p[PARAM_GAIN].smooth_time_s = 0.010f;

    p[PARAM_ATTACK].min_value = 0.001;
    p[PARAM_ATTACK].max_value = 1.0;
    p[PARAM_ATTACK].default_value = 0.1;
    p[PARAM_ATTACK].curve = PARAM_CURVE_LOG;
    p[PARAM_ATTACK].smooth_time_s = 0.020f;

    p[PARAM_RELEASE].min_value = 0.001;
    p[PARAM_RELEASE].max_value = 2.0;
    p[PARAM_RELEASE].default_value = 0.3;
    p[PARAM_RELEASE].curve = PARAM_CURVE_LOG;
    p[PARAM_RELEASE].smooth_time_s = 0.050f;

    p[PARAM_MIX].min_value = 0.0;
    p[PARAM_MIX].max_value = 1.0;
    p[PARAM_MIX].default_value = 1.0;
    p[PARAM_MIX].curve = PARAM_CURVE_LINEAR;
    p[PARAM_MIX].smooth_time_s = 0.010f;

    p[PARAM_BYPASS].min_value = 0.0;
    p[PARAM_BYPASS].max_value = 1.0;
    p[PARAM_BYPASS].default_value = 0.0;
    p[PARAM_BYPASS].curve = PARAM_CURVE_STEPPED;
    p[PARAM_BYPASS].smooth_time_s = 0.0f; // no smoothing on bypass
}

static void reset_smoothing_to_targets(MambaSynth* m) {
    for (uint32_t i = 0; i < PARAM_COUNT; ++i) {
        ParamSmooth& p = m->params[i];
        p.current.store(p.target.load(std::memory_order_relaxed),
            std::memory_order_relaxed);
    }
}

static bool plugin_init(const clap_plugin_t* plugin) {
    auto* m = get_synth(plugin);
    m->core_cfg = MambaConfig::max_config();
    m->core_cfg.num_layers = 1;
    m->core = new (m->core_storage) SSMInferenceCore(m->core_cfg);
    m->sample_rate = 48000.0;
    m->block_size = 512;
    m->is_active = false;
    m->is_processing = false;
    m->latency = 0;  // Mamba-3: no conv1d

    init_param_metadata(m);

    // Initialize atomic param values + smoothing to defaults.
    for (uint32_t i = 0; i < PARAM_COUNT; ++i) {
        const ParamSmooth& p = m->params[i];
        m->param_values[i].store(p.default_value, std::memory_order_relaxed);
        m->params[i].target.store(p.default_value, std::memory_order_relaxed);
        m->params[i].current.store(p.default_value, std::memory_order_relaxed);
    }
    recompute_smooth_coefficients(m);

    m->bypass_target.store(false, std::memory_order_relaxed);
    m->bypass_state.store(BYPASS_PROCESSING, std::memory_order_relaxed);
    m->bypass_fade.store(0.0f, std::memory_order_relaxed);

    // Initialize voice pool
    m->next_note_id = 1;
    for (uint32_t v = 0; v < MAX_VOICES; ++v) {
        auto& voice = m->voices[v];
        std::memset(&voice, 0, sizeof(voice));
        voice.stage = VOICE_IDLE;
        voice.key = -1;
        voice.channel = -1;
        voice.attack_rate = MambaSynth::VOICE_ATTACK_RATE;
        voice.release_rate = MambaSynth::VOICE_RELEASE_RATE;
        auto& va = m->voice_atomics[v];
        va.stage.store(VOICE_IDLE, std::memory_order_relaxed);
        va.env.store(0.0f, std::memory_order_relaxed);
        va.key.store(-1, std::memory_order_relaxed);
        va.channel.store(-1, std::memory_order_relaxed);
        va.velocity.store(0.0f, std::memory_order_relaxed);
    }

    m->ara_conditioning.data = ARAConditioningFrame{};

    ConditioningFrame initial_frame{};
    initial_frame.tempo_bpm = 120.0f;
    initial_frame.key_sig = 0.0f;
    std::memset(initial_frame.chord_one_hot, 0, sizeof(initial_frame.chord_one_hot));
    std::memset(initial_frame.reserved, 0, sizeof(initial_frame.reserved));
    m->conditioning.write(initial_frame);

    m->weight_path[0] = '\0';

    // Initialize thread-check helper with the host we were created with.
    m->thread_check.set_host(m->host);

    // Initialize the double-buffer state mirror with empty content.
    {
        StateMirror* slot = m->state_mirror.begin_write();
        *slot = StateMirror{};
        for (uint32_t i = 0; i < PARAM_COUNT; ++i) {
            slot->param_values[i] = m->params[i].default_value;
        }
        slot->bypass_target = false;
        slot->bypass_fade = 0.0f;
        slot->weight_path[0] = '\0';
        m->state_mirror.end_write();
    }

    return true;
}

static void plugin_destroy(const clap_plugin_t* plugin) {
    auto* m = get_synth(plugin);
    if (m->core) {
        m->core->~SSMInferenceCore();
        m->core = nullptr;
    }
    m->~MambaSynth();
    // Arena owns the memory; no delete needed.
}

static bool plugin_activate(const clap_plugin_t* plugin,
    double sample_rate,
    uint32_t min_frames,
    uint32_t max_frames) {
    auto* m = get_synth(plugin);
    (void)min_frames;
    m->sample_rate = sample_rate;
    m->block_size = max_frames;
    m->is_active = true;

    recompute_smooth_coefficients(m);

    m->core->reset();
    m->bypass_state.store(BYPASS_PROCESSING, std::memory_order_relaxed);
    m->bypass_fade.store(0.0f, std::memory_order_relaxed);

    // Snap smoothing currents to current target so a stream restart
    // doesn't produce a click.
    reset_smoothing_to_targets(m);

    // Reset all voices
    for (uint32_t v = 0; v < MAX_VOICES; ++v) {
        auto& voice = m->voices[v];
        std::memset(&voice.ssm_state, 0, sizeof(voice.ssm_state));
        voice.env = 0.0f;
        voice.stage = VOICE_IDLE;
        voice.key = -1;
        voice.channel = -1;
        voice.velocity = 0.0f;
        voice.attack_rate = MambaSynth::VOICE_ATTACK_RATE;
        voice.release_rate = MambaSynth::VOICE_RELEASE_RATE;
        auto& va = m->voice_atomics[v];
        va.stage.store(VOICE_IDLE, std::memory_order_relaxed);
        va.env.store(0.0f, std::memory_order_relaxed);
        va.key.store(-1, std::memory_order_relaxed);
        va.channel.store(-1, std::memory_order_relaxed);
        va.velocity.store(0.0f, std::memory_order_relaxed);
    }

    return true;
}

static void plugin_deactivate(const clap_plugin_t* plugin) {
    auto* m = get_synth(plugin);
    m->is_active = false;
}

static bool plugin_start_processing(const clap_plugin_t* plugin) {
    auto* m = get_synth(plugin);
    m->is_processing = true;
    return true;
}

static void plugin_stop_processing(const clap_plugin_t* plugin) {
    auto* m = get_synth(plugin);
    m->is_processing = false;
}

static void plugin_reset(const clap_plugin_t* plugin) {
    auto* m = get_synth(plugin);
    m->core->reset();
    m->bypass_state.store(BYPASS_PROCESSING, std::memory_order_relaxed);
    m->bypass_fade.store(0.0f, std::memory_order_relaxed);
    m->bypass_target.store(
        m->param_values[PARAM_BYPASS].load(std::memory_order_acquire) > 0.5,
        std::memory_order_release);

    reset_smoothing_to_targets(m);

    // Reset all voices
    for (uint32_t v = 0; v < MAX_VOICES; ++v) {
        auto& voice = m->voices[v];
        std::memset(&voice.ssm_state, 0, sizeof(voice.ssm_state));
        voice.env = 0.0f;
        voice.stage = VOICE_IDLE;
        voice.key = -1;
        voice.channel = -1;
        voice.velocity = 0.0f;
        auto& va = m->voice_atomics[v];
        va.stage.store(VOICE_IDLE, std::memory_order_relaxed);
        va.env.store(0.0f, std::memory_order_relaxed);
        va.key.store(-1, std::memory_order_relaxed);
        va.channel.store(-1, std::memory_order_relaxed);
        va.velocity.store(0.0f, std::memory_order_relaxed);
    }
}

// Apply any param events drained from the SPSC queue. The audio thread
// runs this at the start of each process() call.
static void apply_params_from_queue(MambaSynth* m) {
    ParamAtom atom;
    while (m->param_queue.pop(atom)) {
        if (atom.id >= PARAM_COUNT) continue;
        m->param_values[atom.id].store(atom.value, std::memory_order_release);
        m->params[atom.id].target.store(atom.value, std::memory_order_release);
        if (atom.id == PARAM_BYPASS) {
            const bool t = atom.value > 0.5;
            m->bypass_target.store(t, std::memory_order_release);
        }
    }
}

// Apply a single param value to both the atomic and the smoother's
// target. Used by params_flush (audio thread context) and the in-process
// event loop. We mirror to param_values[] so get_value() returns the
// same value the audio thread will see; this is what makes the
// state-reproducibility-flush validator pass.
static void apply_param_value(MambaSynth* m, clap_id id, double value) {
    if (id >= PARAM_COUNT) return;
    m->param_values[id].store(value, std::memory_order_release);
    m->params[id].target.store(value, std::memory_order_release);
    if (id == PARAM_BYPASS) {
        m->bypass_target.store(value > 0.5, std::memory_order_release);
    }
}

static inline void advance_one_pole(double& cur, double target, double coeff) {
    cur += coeff * (target - cur);
}

static void update_bypass_state_machine(MambaSynth* m, uint32_t frames) {
    const bool target = m->bypass_target.load(std::memory_order_acquire);
    uint32_t state = m->bypass_state.load(std::memory_order_relaxed);
    float fade = m->bypass_fade.load(std::memory_order_relaxed);

    // Compute total fade step for the upcoming frames. We move fade
    // toward 1.0 (bypassed) or 0.0 (processing) at BYPASS_FADE_RATE per
    // sample, and allow the state machine to transition once fade
    // reaches 0 or 1.
    const float step = MambaSynth::BYPASS_FADE_RATE * static_cast<float>(frames);

    switch (state) {
    case BYPASS_PROCESSING:
        if (target) {
            // Begin fading out.
            fade = std::min(1.0f, fade + step);
            if (fade >= 1.0f) {
                state = BYPASS_BYPASSED;
            } else {
                state = BYPASS_FADING_OUT;
            }
        }
        break;
    case BYPASS_FADING_OUT:
        if (!target) {
            // Cancellation: fade back in.
            fade = std::max(0.0f, fade - step);
            if (fade <= 0.0f) state = BYPASS_PROCESSING;
        } else {
            fade = std::min(1.0f, fade + step);
            if (fade >= 1.0f) state = BYPASS_BYPASSED;
        }
        break;
    case BYPASS_BYPASSED:
        if (!target) {
            // Begin fading in.
            fade = std::max(0.0f, fade - step);
            if (fade <= 0.0f) {
                state = BYPASS_PROCESSING;
            } else {
                state = BYPASS_FADING_IN;
            }
        }
        break;
    case BYPASS_FADING_IN:
        if (target) {
            fade = std::min(1.0f, fade + step);
            if (fade >= 1.0f) state = BYPASS_BYPASSED;
        } else {
            fade = std::max(0.0f, fade - step);
            if (fade <= 0.0f) state = BYPASS_PROCESSING;
        }
        break;
    }

    m->bypass_fade.store(fade, std::memory_order_relaxed);
    m->bypass_state.store(state, std::memory_order_relaxed);
}

// Find a free voice slot, or steal the oldest released/silent one.
// Returns the voice index or -1 if all voices are busy attacking.
static int32_t allocate_voice(MambaSynth* m) {
    // First pass: find VOICE_IDLE or VOICE_SILENT
    for (uint32_t v = 0; v < MAX_VOICES; ++v) {
        uint8_t st = m->voices[v].stage;
        if (st == VOICE_IDLE || st == VOICE_SILENT) {
            return static_cast<int32_t>(v);
        }
    }
    // Second pass: steal the oldest RELEASE voice (lowest note_id)
    uint64_t oldest_id = UINT64_MAX;
    int32_t oldest_v = -1;
    for (uint32_t v = 0; v < MAX_VOICES; ++v) {
        if (m->voices[v].stage == VOICE_RELEASE && m->voices[v].note_id < oldest_id) {
            oldest_id = m->voices[v].note_id;
            oldest_v = static_cast<int32_t>(v);
        }
    }
    if (oldest_v >= 0) return oldest_v;
    // Third pass: steal the oldest SUSTAIN voice
    for (uint32_t v = 0; v < MAX_VOICES; ++v) {
        if (m->voices[v].note_id < oldest_id) {
            oldest_id = m->voices[v].note_id;
            oldest_v = static_cast<int32_t>(v);
        }
    }
    return oldest_v;  // may be -1 if no voices exist (shouldn't happen)
}

// Start a new note on the given voice slot.
static void start_voice(MambaSynth* m, uint32_t v, int32_t key, int32_t channel,
                         float velocity, uint64_t note_id) {
    auto& voice = m->voices[v];
    // Reset SSM state for this voice
    std::memset(&voice.ssm_state, 0, sizeof(voice.ssm_state));
    voice.env = 0.0f;
    voice.stage = VOICE_ATTACK;
    voice.velocity = velocity;
    voice.key = key;
    voice.channel = channel;
    voice.note_id = note_id;
    voice.attack_rate = MambaSynth::VOICE_ATTACK_RATE;
    voice.release_rate = MambaSynth::VOICE_RELEASE_RATE;

    auto& va = m->voice_atomics[v];
    va.stage.store(VOICE_ATTACK, std::memory_order_relaxed);
    va.env.store(0.0f, std::memory_order_relaxed);
    va.key.store(key, std::memory_order_relaxed);
    va.channel.store(channel, std::memory_order_relaxed);
    va.velocity.store(velocity, std::memory_order_relaxed);
}

// Iterate note events in the input event list with sample-accurate
// frame offsets. Each event's time is the offset within the current
// process() block at which the event should take effect.
static void process_note_events(MambaSynth* m,
    const clap_input_events_t* in_events,
    uint32_t frames) {
    (void)frames;
    const uint32_t count = in_events->size(in_events);
    for (uint32_t i = 0; i < count; ++i) {
        const clap_event_header_t* hdr = in_events->get(in_events, i);
        if (hdr->space_id != CLAP_CORE_EVENT_SPACE_ID) continue;

        if (hdr->type == CLAP_EVENT_NOTE_ON) {
            const auto* ev = reinterpret_cast<const clap_event_note_t*>(hdr);
            if (ev->velocity <= 0.0) {
                // NOTE_ON with velocity 0 is a release (per CLAP spec).
                // Find the voice playing this note and release it.
                for (uint32_t v = 0; v < MAX_VOICES; ++v) {
                    if (m->voices[v].key == ev->key &&
                        m->voices[v].channel == ev->channel &&
                        (m->voices[v].stage == VOICE_ATTACK ||
                         m->voices[v].stage == VOICE_SUSTAIN)) {
                        m->voices[v].stage = VOICE_RELEASE;
                        m->voice_atomics[v].stage.store(VOICE_RELEASE,
                            std::memory_order_relaxed);
                        break;
                    }
                }
            } else {
                // Allocate a voice and start the note.
                int32_t v = allocate_voice(m);
                if (v >= 0) {
                    uint64_t nid = m->next_note_id++;
                    start_voice(m, static_cast<uint32_t>(v),
                                ev->key, ev->channel,
                                static_cast<float>(ev->velocity), nid);
                }
            }
        } else if (hdr->type == CLAP_EVENT_NOTE_OFF) {
            const auto* ev = reinterpret_cast<const clap_event_note_t*>(hdr);
            // Find the voice playing this note and release it.
            for (uint32_t v = 0; v < MAX_VOICES; ++v) {
                if (m->voices[v].key == ev->key &&
                    m->voices[v].channel == ev->channel &&
                    (m->voices[v].stage == VOICE_ATTACK ||
                     m->voices[v].stage == VOICE_SUSTAIN)) {
                    m->voices[v].stage = VOICE_RELEASE;
                    m->voice_atomics[v].stage.store(VOICE_RELEASE,
                        std::memory_order_relaxed);
                    break;
                }
            }
        } else if (hdr->type == CLAP_EVENT_PARAM_VALUE) {
            const auto* ev = reinterpret_cast<const clap_event_param_value_t*>(hdr);
            apply_param_value(m, ev->param_id, ev->value);
        }
    }
}

// Build a conditioning frame from the most recently pressed voice
// (highest note_id in ATTACK or SUSTAIN) and ARA context.
static void refresh_conditioning(MambaSynth* m) {
    ConditioningFrame cf{};
    cf.tempo_bpm = 120.0f;
    cf.key_sig = 0.0f;
    std::memset(cf.chord_one_hot, 0, sizeof(cf.chord_one_hot));
    cf.reserved[0] = 0.0f;
    cf.reserved[1] = 0.0f;

    // Find the most recent active voice for conditioning.
    uint64_t newest_id = 0;
    for (uint32_t v = 0; v < MAX_VOICES; ++v) {
        uint8_t st = m->voices[v].stage;
        if ((st == VOICE_ATTACK || st == VOICE_SUSTAIN) &&
            m->voices[v].note_id > newest_id) {
            newest_id = m->voices[v].note_id;
            int32_t key = m->voices[v].key;
            float vel = m->voices[v].velocity;
            int pitch_class = key % 12;
            cf.key_sig = static_cast<float>(pitch_class);
            cf.chord_one_hot[pitch_class] = vel;
            cf.reserved[0] = static_cast<float>(key) / 127.0f;
            cf.reserved[1] = vel;
        }
    }

    // ARA overrides conditioning when it has a non-default value.
    ARAConditioningFrame ara_cf;
    if (m->ara_conditioning.read(ara_cf)) {
        cf.tempo_bpm = ara_cf.tempo_bpm;
        cf.key_sig = ara_cf.key_sig;
        std::memcpy(cf.chord_one_hot, ara_cf.chord_one_hot, sizeof(cf.chord_one_hot));
    }

    m->core->set_conditioning(&cf.tempo_bpm);
}

// Push a fresh state-mirror snapshot to the double-buffer seqlock. The
// audio thread calls this at the end of each process() call.
static void publish_state_mirror(MambaSynth* m) {
    StateMirror* slot = m->state_mirror.begin_write();
    for (uint32_t v = 0; v < MAX_VOICES; ++v) {
        slot->voice_snapshots[v].ssm_state = m->voices[v].ssm_state;
        slot->voice_snapshots[v].env = m->voices[v].env;
        slot->voice_snapshots[v].stage = m->voices[v].stage;
    }
    for (uint32_t i = 0; i < PARAM_COUNT; ++i) {
        slot->param_values[i] = m->param_values[i].load(std::memory_order_acquire);
    }
    slot->bypass_target = m->bypass_target.load(std::memory_order_acquire);
    slot->bypass_fade = m->bypass_fade.load(std::memory_order_relaxed);
    std::memset(slot->weight_path, 0, sizeof(slot->weight_path));
    std::memcpy(slot->weight_path, m->weight_path, std::strlen(m->weight_path));
    m->state_mirror.end_write();
    m->state_mirror.swap();
}

static clap_process_status plugin_process(const clap_plugin_t* plugin,
    const clap_process_t* process) {
    auto* m = get_synth(plugin);

    SSMBlock::apply_ftz_daz();
    m->thread_check.mark_audio_thread();

    update_bypass_state_machine(m, process->frames_count);
    refresh_conditioning(m);

    const float* input = nullptr;
    float* output = nullptr;
    if (process->audio_inputs_count > 0 && process->audio_inputs->data32) {
        input = process->audio_inputs->data32[0];
    }
    if (process->audio_outputs_count > 0 && process->audio_outputs->data32) {
        output = process->audio_outputs->data32[0];
    }
    if (!output) return CLAP_PROCESS_ERROR;

    const float fade = m->bypass_fade.load(std::memory_order_relaxed);
    const uint32_t frames = process->frames_count;

    // Pre-compute the per-parameter targets (load once).
    const double t_gain = m->params[PARAM_GAIN].target.load(std::memory_order_relaxed);
    const double t_mix  = m->params[PARAM_MIX].target.load(std::memory_order_relaxed);
    const double c_gain = m->params[PARAM_GAIN].coeff;
    const double c_mix  = m->params[PARAM_MIX].coeff;

    double cur_gain = m->params[PARAM_GAIN].current.load(std::memory_order_relaxed);
    double cur_mix  = m->params[PARAM_MIX].current.load(std::memory_order_relaxed);

    // Pre-scan CLAP events and SPSC queue for sample-accurate scheduling.
    // Events with frameOffset==0 apply immediately; others are stored in
    // on-the-fly buffers indexed by frame.
    //
    // For SPSC queue (main thread -> audio thread params):
    // Drain the queue at the start; apply frameOffset==0 now, buffer others.
    // We use a simple static-size buffer: at most 8 pending param events.
    static constexpr uint32_t MAX_PENDING_PARAMS = 8;
    struct PendingParam { clap_id id; double value; uint32_t frame; };
    PendingParam pending_params[MAX_PENDING_PARAMS];
    uint32_t num_pending = 0;
    {
        ParamAtom atom;
        while (m->param_queue.pop(atom)) {
            if (atom.id >= PARAM_COUNT) continue;
            if (atom.frame_offset == 0) {
                apply_param_value(m, atom.id, atom.value);
            } else if (num_pending < MAX_PENDING_PARAMS) {
                pending_params[num_pending].id = atom.id;
                pending_params[num_pending].value = atom.value;
                pending_params[num_pending].frame = atom.frame_offset;
                num_pending++;
            }
        }
    }

    // For CLAP events: build a compact per-frame offset list.
    // Max CLAP events per process() is bounded (typically < 256).
    static constexpr uint32_t MAX_CLAP_EVENTS = 256;
    struct FrameEvent { uint32_t frame; bool is_note_on; int32_t key;
                        int32_t channel; float velocity; };
    FrameEvent frame_events[MAX_CLAP_EVENTS];
    uint32_t num_frame_events = 0;
    if (process->in_events) {
        const uint32_t count = process->in_events->size(process->in_events);
        for (uint32_t i = 0; i < count && num_frame_events < MAX_CLAP_EVENTS; ++i) {
            const clap_event_header_t* hdr = process->in_events->get(process->in_events, i);
            if (hdr->space_id != CLAP_CORE_EVENT_SPACE_ID) continue;
            if (hdr->type == CLAP_EVENT_NOTE_ON) {
                const auto* ev = reinterpret_cast<const clap_event_note_t*>(hdr);
                frame_events[num_frame_events].frame = hdr->time;
                frame_events[num_frame_events].is_note_on = ev->velocity > 0.0f;
                frame_events[num_frame_events].key = ev->key;
                frame_events[num_frame_events].channel = ev->channel;
                frame_events[num_frame_events].velocity = static_cast<float>(ev->velocity);
                num_frame_events++;
            } else if (hdr->type == CLAP_EVENT_NOTE_OFF) {
                const auto* ev = reinterpret_cast<const clap_event_note_t*>(hdr);
                frame_events[num_frame_events].frame = hdr->time;
                frame_events[num_frame_events].is_note_on = false;
                frame_events[num_frame_events].key = ev->key;
                frame_events[num_frame_events].channel = ev->channel;
                frame_events[num_frame_events].velocity = 0.0f;
                num_frame_events++;
            } else if (hdr->type == CLAP_EVENT_PARAM_VALUE) {
                const auto* ev = reinterpret_cast<const clap_event_param_value_t*>(hdr);
                if (hdr->time == 0) {
                    apply_param_value(m, ev->param_id, ev->value);
                } else if (num_pending < MAX_PENDING_PARAMS) {
                    pending_params[num_pending].id = ev->param_id;
                    pending_params[num_pending].value = ev->value;
                    pending_params[num_pending].frame = hdr->time;
                    num_pending++;
                }
            }
        }
    }
    // Sort pending events by frame
    if (num_pending > 1) {
        for (uint32_t i = 1; i < num_pending; ++i) {
            for (uint32_t j = i; j > 0 && pending_params[j].frame < pending_params[j-1].frame; --j) {
                PendingParam tmp = pending_params[j];
                pending_params[j] = pending_params[j-1];
                pending_params[j-1] = tmp;
            }
        }
    }
    // Sort frame events by frame
    if (num_frame_events > 1) {
        for (uint32_t i = 1; i < num_frame_events; ++i) {
            for (uint32_t j = i; j > 0 && frame_events[j].frame < frame_events[j-1].frame; --j) {
                FrameEvent tmp = frame_events[j];
                frame_events[j] = frame_events[j-1];
                frame_events[j-1] = tmp;
            }
        }
    }
    uint32_t pending_idx = 0;
    uint32_t frame_event_idx = 0;

    for (uint32_t n = 0; n < frames; ++n) {
        // Apply pending param changes at their target frame
        while (pending_idx < num_pending && pending_params[pending_idx].frame == n) {
            apply_param_value(m, pending_params[pending_idx].id,
                              pending_params[pending_idx].value);
            pending_idx++;
        }

        // Apply CLAP events at their target frame
        while (frame_event_idx < num_frame_events && frame_events[frame_event_idx].frame == n) {
            const auto& fe = frame_events[frame_event_idx];
            if (fe.is_note_on) {
                int32_t v = allocate_voice(m);
                if (v >= 0) {
                    uint64_t nid = m->next_note_id++;
                    start_voice(m, static_cast<uint32_t>(v),
                                fe.key, fe.channel, fe.velocity, nid);
                }
            } else {
                for (uint32_t v = 0; v < MAX_VOICES; ++v) {
                    if (m->voices[v].key == fe.key &&
                        m->voices[v].channel == fe.channel &&
                        (m->voices[v].stage == VOICE_ATTACK ||
                         m->voices[v].stage == VOICE_SUSTAIN)) {
                        m->voices[v].stage = VOICE_RELEASE;
                        m->voice_atomics[v].stage.store(VOICE_RELEASE,
                            std::memory_order_relaxed);
                        break;
                    }
                }
            }
            frame_event_idx++;
        }

        advance_one_pole(cur_gain, t_gain, c_gain);
        advance_one_pole(cur_mix,  t_mix,  c_mix);

        const float in_sample = input ? input[n] : 0.0f;
        float voice_sum = 0.0f;

        // Process each active voice, accumulating output.
        for (uint32_t v = 0; v < MAX_VOICES; ++v) {
            auto& voice = m->voices[v];
            uint8_t vs = voice.stage;
            if (vs == VOICE_IDLE) continue;

            // Envelope state machine
            switch (vs) {
            case VOICE_ATTACK:
                voice.env += voice.attack_rate;
                if (voice.env >= 1.0f) {
                    voice.env = 1.0f;
                    voice.stage = VOICE_SUSTAIN;
                    m->voice_atomics[v].stage.store(VOICE_SUSTAIN,
                        std::memory_order_relaxed);
                }
                break;
            case VOICE_SUSTAIN:
                voice.env = 1.0f;
                break;
            case VOICE_RELEASE:
                voice.env -= voice.release_rate;
                if (voice.env <= 0.0f) {
                    voice.env = 0.0f;
                    voice.stage = VOICE_SILENT;
                    m->voice_atomics[v].stage.store(VOICE_SILENT,
                        std::memory_order_relaxed);
                }
                break;
            case VOICE_SILENT:
                voice.env = 0.0f;
                break;
            default:
                voice.env = 0.0f;
                break;
            }

            if (voice.env <= 0.0f) continue;

            // Restore voice's SSM state, process, save state
            m->core->set_state(voice.ssm_state);
            alignas(64) float outbuf[MAX_D_MODEL];
            m->core->process_sample(in_sample, outbuf);
            m->core->get_state(voice.ssm_state);

            float processed = outbuf[0];
            voice_sum += processed * voice.env * voice.velocity;

            // Update atomic mirror for this voice
            m->voice_atomics[v].env.store(voice.env, std::memory_order_relaxed);
        }

        // Mix dry/wet with gain
        const float dry = in_sample;
        const float wet = voice_sum * static_cast<float>(cur_gain);
        const float mixed = dry * static_cast<float>(1.0 - cur_mix)
                          + wet  * static_cast<float>(cur_mix);

        // Crossfade between processed (fade=0) and dry bypass (fade=1).
        output[n] = mixed * (1.0f - fade) + dry * fade;

        const uint32_t ch_count = process->audio_outputs->channel_count;
        for (uint32_t ch = 1; ch < ch_count; ++ch) {
            process->audio_outputs->data32[ch][n] = output[n];
        }
    }

    m->params[PARAM_GAIN].current.store(cur_gain, std::memory_order_relaxed);
    m->params[PARAM_MIX].current.store(cur_mix, std::memory_order_relaxed);

    publish_state_mirror(m);

    return CLAP_PROCESS_CONTINUE;
}

// Default behaviour: thread-check is queried from the host, not provided
// by the plugin. The plugin's get_extension returns nullptr for
// CLAP_EXT_THREAD_CHECK, and helpers in clap_thread_check.h use the
// cached host pointer instead. We keep the slot here so get_extension
// is unambiguous about which IDs the plugin recognizes.
static const void* plugin_get_extension(const clap_plugin_t* plugin, const char* id) {
    (void)plugin;

    if (strcmp(id, CLAP_EXT_AUDIO_PORTS) == 0) {
        return &s_audio_ports;
    }
    if (strcmp(id, CLAP_EXT_NOTE_PORTS) == 0) {
        return &s_note_ports;
    }
    if (strcmp(id, CLAP_EXT_PARAMS) == 0) {
        return &s_params;
    }
    if (strcmp(id, CLAP_EXT_STATE) == 0) {
        return &s_state;
    }
    if (strcmp(id, CLAP_EXT_TAIL) == 0) {
        return &s_tail;
    }
    if (strcmp(id, CLAP_EXT_LATENCY) == 0) {
        return &s_latency;
    }
    if (strcmp(id, CLAP_EXT_GUI) == 0) {
        return &s_gui;
    }
    return nullptr;
}

static void plugin_on_main_thread(const clap_plugin_t* plugin) {
    auto* m = get_synth(plugin);
    if (m) m->thread_check.set_host(m->host);
}

static const clap_plugin_t s_plugin_class = {
    &s_descriptor,
    nullptr,
    plugin_init,
    plugin_destroy,
    plugin_activate,
    plugin_deactivate,
    plugin_start_processing,
    plugin_stop_processing,
    plugin_reset,
    plugin_process,
    plugin_get_extension,
    plugin_on_main_thread
};

bool MambaSynth::load_weights(const char* path) {
    if (!path || path[0] == '\0') return false;
    if (weight_loader.load(path) != 0) return false;

    // Validate dimensions against the loaded header
    const auto& hdr = weight_loader.header();
    if (!weight_loader.validate_dimensions(
            hdr.d_state, hdr.d_model, hdr.d_inner,
            hdr.nheads, hdr.headdim,
            static_cast<uint32_t>(core_cfg.num_layers))) {
        weight_loader.unload();
        return false;
    }

    // Update core config from header
    core_cfg.d_state = hdr.d_state;
    core_cfg.d_model = hdr.d_model;
    core_cfg.d_inner = hdr.d_inner;
    core_cfg.nheads = hdr.nheads;
    core_cfg.headdim = hdr.headdim;
    core_cfg.cond_dim = hdr.cond_dim;
    if (hdr.version == 2) {
        core_cfg.num_layers = hdr.num_layers;
    }
    core->init(core_cfg);

    // Load weights: 13 arrays per layer (Mamba-3 layout)
    for (uint32_t layer = 0; layer < core_cfg.num_layers; ++layer) {
        WeightLayerPointers ptrs;
        if (weight_loader.get_layer_pointers(layer, ptrs) != 0) {
            weight_loader.unload();
            return false;
        }

        SSMWeights w{};
        size_t in_proj_out = core_cfg.d_inner * 2
                           + core_cfg.nheads * MAX_MIMO_RANK * core_cfg.d_state * 2
                           + core_cfg.nheads * 3
                           + MAX_NUM_ROPE_ANGLES;
        std::memcpy(w.in_proj_w,      ptrs.in_proj_w,      in_proj_out * core_cfg.d_model * sizeof(float));
        std::memcpy(w.dt_bias,        ptrs.dt_bias,        core_cfg.nheads * sizeof(float));
        std::memcpy(w.B_bias,         ptrs.B_bias,         core_cfg.nheads * MAX_MIMO_RANK * core_cfg.d_state * sizeof(float));
        std::memcpy(w.C_bias,         ptrs.C_bias,         core_cfg.nheads * MAX_MIMO_RANK * core_cfg.d_state * sizeof(float));
        std::memcpy(w.B_norm_weight,  ptrs.B_norm_weight,  core_cfg.d_state * sizeof(float));
        std::memcpy(w.C_norm_weight,  ptrs.C_norm_weight,  core_cfg.d_state * sizeof(float));
        std::memcpy(w.mimo_x,         ptrs.mimo_x,         core_cfg.nheads * MAX_MIMO_RANK * core_cfg.headdim * sizeof(float));
        std::memcpy(w.mimo_z,         ptrs.mimo_z,         core_cfg.nheads * MAX_MIMO_RANK * core_cfg.headdim * sizeof(float));
        std::memcpy(w.mimo_o,         ptrs.mimo_o,         core_cfg.nheads * MAX_MIMO_RANK * core_cfg.headdim * sizeof(float));
        std::memcpy(w.D_proj,         ptrs.D_proj,         core_cfg.nheads * sizeof(float));
        std::memcpy(w.out_proj_w,     ptrs.out_proj_w,     core_cfg.d_model * core_cfg.d_inner * sizeof(float));
        std::memcpy(w.film_proj_w,    ptrs.film_proj_w,    2 * core_cfg.d_model * core_cfg.cond_dim * sizeof(float));
        std::memcpy(w.dt_film_proj_w, ptrs.dt_film_proj_w, 2 * core_cfg.nheads * core_cfg.cond_dim * sizeof(float));

        core->set_layer_weights(layer, w);
    }
    weights_loaded.store(true, std::memory_order_release);

    // Record the loaded path so state save can round-trip it.
    std::strncpy(weight_path, path, MAX_WEIGHT_PATH - 1);
    weight_path[MAX_WEIGHT_PATH - 1] = '\0';

    return true;
}

const clap_plugin_t* mamba_synth_create(const clap_host_t* host) {
    auto* m = static_cast<MambaSynth*>(
        s_plugin_arena.allocate(sizeof(MambaSynth), alignof(MambaSynth)));
    if (!m) return nullptr;
    new (m) MambaSynth();
    m->host = host;
    m->core = nullptr;
    m->weights_loaded.store(false, std::memory_order_relaxed);
    m->block_size = 512;
    m->is_active = false;
    m->is_processing = false;
    m->latency = 0;  // Mamba-3: no conv1d
    m->bypass_target.store(false, std::memory_order_relaxed);
    m->bypass_state.store(BYPASS_PROCESSING, std::memory_order_relaxed);
    m->bypass_fade.store(0.0f, std::memory_order_relaxed);
    m->next_note_id = 1;
    for (uint32_t v = 0; v < MAX_VOICES; ++v) {
        auto& voice = m->voices[v];
        std::memset(&voice, 0, sizeof(voice));
        voice.stage = VOICE_IDLE;
        voice.key = -1;
        voice.channel = -1;
        voice.attack_rate = MambaSynth::VOICE_ATTACK_RATE;
        voice.release_rate = MambaSynth::VOICE_RELEASE_RATE;
        auto& va = m->voice_atomics[v];
        va.stage.store(VOICE_IDLE, std::memory_order_relaxed);
        va.env.store(0.0f, std::memory_order_relaxed);
        va.key.store(-1, std::memory_order_relaxed);
        va.channel.store(-1, std::memory_order_relaxed);
        va.velocity.store(0.0f, std::memory_order_relaxed);
    }
    m->ara_conditioning.data = ARAConditioningFrame{};
    for (uint32_t i = 0; i < PARAM_COUNT; ++i) {
        m->param_values[i].store(0.0, std::memory_order_relaxed);
    }
    m->weight_path[0] = '\0';
    m->thread_check.set_host(host);

    auto* plugin = static_cast<clap_plugin_t*>(
        s_plugin_arena.allocate(sizeof(clap_plugin_t), alignof(clap_plugin_t)));
    if (!plugin) return nullptr;
    new (plugin) clap_plugin_t(s_plugin_class);
    plugin->plugin_data = m;
    return plugin;
}

}
