#include "clap/clap_plugin.h"
#include "clap/clap_common_utils.h"
#include "clap/clap_gui.h"
#include "clap/param_table.h"
#include "core/weight_loader.h"
#include "config.h"
#include <cstring>
#include <cmath>
#include <cstdio>
#include <algorithm>
#include <clap/ext/timer-support.h>
#include <memory>
#include <new>

namespace mamba {

// Plugin instances (MambaSynth is tens of MB once sized for
// MambaConfig::max_config()) are allocated with plain new/delete.
// Creation/destruction only ever happens on the main thread, so
// heap-allocation RT-safety concerns don't apply here; a fixed-size
// bump arena cannot hold even one instance and never reclaims memory
// across create/destroy cycles, which made multi-instance hosting and
// plugin-scan rescans impossible.

static const char* s_features[] = {
    CLAP_PLUGIN_FEATURE_INSTRUMENT,
    CLAP_PLUGIN_FEATURE_SYNTHESIZER,
    CLAP_PLUGIN_FEATURE_STEREO,
    nullptr
};

static const clap_plugin_descriptor_t s_descriptor = {
    CLAP_VERSION_INIT,
    "com.emt.synth",
    "EmT Synth",
    "EmT Audio",
    "https://emt-audio.com",
    "",
    "",
    "0.3.0",
    "EmT Mamba-3 SSM synthesizer with FiLM conditioning",
    reinterpret_cast<const char* const*>(s_features)
};

const clap_plugin_descriptor_t emt_synth_descriptor = s_descriptor;
const clap_plugin_descriptor_t mamba_synth_descriptor = s_descriptor;

static MambaSynth* get_synth(const clap_plugin_t* plugin) {
    return plugin ? static_cast<MambaSynth*>(plugin->plugin_data) : nullptr;
}

// Recompute smoothing coefficients after a sample-rate change. Called
// from activate() and from any code path that mutates smooth_time_s.
static void recompute_smooth_coefficients(MambaSynth* m) {
    for (uint32_t i = 0; i < PARAM_COUNT; ++i) {
        m->params[i].coeff = compute_one_pole_coeff(m->params[i].smooth_time_s, m->sample_rate);
    }
}

// Initialize per-parameter metadata (range, curve, smoothing time) from the
// shared table. Array-size deduction in init_param_smooth fails to compile
// if the smoothing array and the table ever disagree on PARAM_COUNT.
static void init_param_metadata(MambaSynth* m) {
    init_param_smooth(m->params, kSynthParamTable);
}

static void reset_smoothing_to_targets(MambaSynth* m) {
    for (uint32_t i = 0; i < PARAM_COUNT; ++i) {
        ParamSmooth& p = m->params[i];
        p.current.store(p.target.load(std::memory_order_relaxed),
            std::memory_order_relaxed);
    }
}

static bool plugin_init(const clap_plugin_t* plugin) {
    if (!plugin) return false;
    auto* m = get_synth(plugin);
    if (!m) return false;
    m->core_cfg = MambaConfig::max_config();
    m->core_cfg.num_layers = 1;
    auto* core = new (std::nothrow) SSMInferenceCore(m->core_cfg);
    if (!core) return false;
    m->core.publish(core);
    m->sample_rate = 48000.0;
    m->block_size = 512;
    m->is_active = false;
    m->is_processing.store(false, std::memory_order_relaxed);
    m->state_mirror_idle_published = false;
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
    m->steal_residual[0] = 0.0f;
    m->steal_residual[1] = 0.0f;
    for (uint32_t v = 0; v < MAX_VOICES; ++v) {
        auto& voice = m->voices[v];
        std::memset(&voice, 0, sizeof(voice));
        voice.stage = VOICE_IDLE;
        voice.key = -1;
        voice.channel = -1;
        voice.clap_note_id = -1;
        voice.attack_rate = MambaSynth::VOICE_ATTACK_RATE;
        voice.release_rate = MambaSynth::VOICE_RELEASE_RATE;
        voice.last_out[0] = 0.0f;
        voice.last_out[1] = 0.0f;
        m->mirror_zeroed_count[v] = 2;
        auto& va = m->voice_atomics[v];
        va.stage.store(VOICE_IDLE, std::memory_order_relaxed);
        va.env.store(0.0f, std::memory_order_relaxed);
        va.key.store(-1, std::memory_order_relaxed);
        va.channel.store(-1, std::memory_order_relaxed);
        va.velocity.store(0.0f, std::memory_order_relaxed);
    }

    m->ara_conditioning.write(ARAConditioningFrame{});

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
        if (slot) {
            std::memset(slot, 0, sizeof(StateMirror));
            for (uint32_t i = 0; i < PARAM_COUNT; ++i) {
                slot->param_values[i] = m->params[i].default_value;
            }
            slot->bypass_target = false;
            slot->bypass_fade = 0.0f;
            m->state_mirror.end_write();
        }
    }

    return true;
}

static void plugin_destroy(const clap_plugin_t* plugin) {
    if (!plugin) return;
    auto* m = get_synth(plugin);
    if (m) {
        m->core.destroy();
        delete m;
    }
    delete plugin;
}

static bool plugin_activate(const clap_plugin_t* plugin,
    double sample_rate,
    uint32_t min_frames,
    uint32_t max_frames) {
    if (!plugin) return false;
    auto* m = get_synth(plugin);
    if (!m) return false;
    (void)min_frames;
    m->sample_rate = sample_rate;
    m->block_size = max_frames;
    m->is_active = true;
    m->state_mirror_idle_published = false;
    m->steal_residual[0] = 0.0f;
    m->steal_residual[1] = 0.0f;

    recompute_smooth_coefficients(m);

    if (auto* c = m->core.published()) c->reset();
    const bool bypassed = m->bypass_target.load(std::memory_order_acquire);
    m->bypass_state.store(bypassed ? BYPASS_BYPASSED : BYPASS_PROCESSING, std::memory_order_relaxed);
    m->bypass_fade.store(bypassed ? 1.0f : 0.0f, std::memory_order_relaxed);

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
        voice.clap_note_id = -1;
        voice.velocity = 0.0f;
        voice.last_out[0] = 0.0f;
        voice.last_out[1] = 0.0f;
        m->mirror_zeroed_count[v] = 2;
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
    if (!m) return;
    m->is_active = false;
}

static bool plugin_start_processing(const clap_plugin_t* plugin) {
    auto* m = get_synth(plugin);
    if (!m) return false;
    m->is_processing.store(true, std::memory_order_release);
    return true;
}

static void plugin_stop_processing(const clap_plugin_t* plugin) {
    auto* m = get_synth(plugin);
    if (!m) return;
    m->is_processing.store(false, std::memory_order_release);
}

static void plugin_reset(const clap_plugin_t* plugin) {
    auto* m = get_synth(plugin);
    if (!m) return;
    if (auto* c = m->core.published()) c->reset();
    const bool bypassed = m->param_values[PARAM_BYPASS].load(std::memory_order_acquire) > 0.5;
    m->bypass_target.store(bypassed, std::memory_order_release);
    m->bypass_state.store(bypassed ? BYPASS_BYPASSED : BYPASS_PROCESSING, std::memory_order_relaxed);
    m->bypass_fade.store(bypassed ? 1.0f : 0.0f, std::memory_order_relaxed);
    m->state_mirror_idle_published = false;
    m->steal_residual[0] = 0.0f;
    m->steal_residual[1] = 0.0f;

    reset_smoothing_to_targets(m);

    // Reset all voices
    for (uint32_t v = 0; v < MAX_VOICES; ++v) {
        auto& voice = m->voices[v];
        std::memset(&voice.ssm_state, 0, sizeof(voice.ssm_state));
        voice.env = 0.0f;
        voice.stage = VOICE_IDLE;
        voice.key = -1;
        voice.channel = -1;
        voice.clap_note_id = -1;
        voice.velocity = 0.0f;
        voice.last_out[0] = 0.0f;
        voice.last_out[1] = 0.0f;
        m->mirror_zeroed_count[v] = 2;
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

// Apply a single param value to both the atomic and the smoother's
// target. Used by params_flush (audio thread context) and the in-process
// event loop. We mirror to param_values[] so get_value() returns the
// same value the audio thread will see; this is what makes the
// state-reproducibility-flush validator pass.
static void apply_param_value(MambaSynth* m, clap_id id, double value) {
    if (id >= PARAM_COUNT || !std::isfinite(value)) return;
    const ParamSmooth& p = m->params[id];
    value = std::clamp(value, p.min_value, p.max_value);
    if (p.curve == PARAM_CURVE_STEPPED) {
        value = value >= 0.5 ? 1.0 : 0.0;
    }
    m->param_values[id].store(value, std::memory_order_release);
    m->params[id].target.store(value, std::memory_order_release);
    if (id == PARAM_BYPASS) {
        m->bypass_target.store(value > 0.5, std::memory_order_release);
    }
}

static float compute_envelope_rate(double time_s, double min_s, double max_s, double sample_rate) {
    const double sr = sample_rate > 1.0 ? sample_rate : 48000.0;
    const double clamped = std::clamp(time_s, min_s, max_s);
    return static_cast<float>(1.0 / std::max(1.0, clamped * sr));
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
                         int32_t clap_note_id, float velocity, uint64_t note_id) {
    auto& voice = m->voices[v];
    if (voice.stage != VOICE_IDLE && voice.stage != VOICE_SILENT && voice.env > 0.0f) {
        const float amp = voice.env * voice.velocity;
        if (std::isfinite(voice.last_out[0])) m->steal_residual[0] += voice.last_out[0] * amp;
        if (std::isfinite(voice.last_out[1])) m->steal_residual[1] += voice.last_out[1] * amp;
    }
    voice.last_out[0] = 0.0f;
    voice.last_out[1] = 0.0f;
    // Reset SSM state for this voice
    std::memset(&voice.ssm_state, 0, sizeof(voice.ssm_state));
    voice.env = 0.0f;
    voice.stage = VOICE_ATTACK;
    voice.velocity = velocity;
    voice.key = key;
    voice.channel = channel;
    voice.clap_note_id = clap_note_id;
    voice.note_id = note_id;
    voice.attack_rate = compute_envelope_rate(
        m->params[PARAM_ATTACK].current.load(std::memory_order_relaxed),
        m->params[PARAM_ATTACK].min_value, m->params[PARAM_ATTACK].max_value, m->sample_rate);
    voice.release_rate = compute_envelope_rate(
        m->params[PARAM_RELEASE].current.load(std::memory_order_relaxed),
        m->params[PARAM_RELEASE].min_value, m->params[PARAM_RELEASE].max_value, m->sample_rate);

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

static uint64_t next_monotonic_note_id(MambaSynth* m) {
    uint64_t id = (m->next_note_id == 0) ? 1u : m->next_note_id;
    m->next_note_id = (id == UINT64_MAX) ? 1u : (id + 1u);
    return id;
}

static bool cf_any_nonzero(const ConditioningFrame& cf) {
    const float* p = &cf.tempo_bpm;
    for (size_t i = 0; i < COND_DIM; ++i) {
        if (std::isfinite(p[i]) && std::fabs(p[i]) > 0.0f) return true;
    }
    return false;
}

static void sanitize_conditioning_frame(ConditioningFrame& cf) {
    float* p = &cf.tempo_bpm;
    for (size_t i = 0; i < COND_DIM; ++i) {
        if (!std::isfinite(p[i])) p[i] = 0.0f;
    }
}

// Push a fresh state-mirror snapshot to the double-buffer seqlock. The
// audio thread calls this at the end of each process() call only when
// weights are loaded and voice state changed (avoiding multi-MB idle copies).
static void publish_state_mirror(MambaSynth* m, uint32_t active_layers, bool any_voice_active) {
    const bool has_weights = m->weights_loaded.load(std::memory_order_acquire);
    if (!has_weights) return;
    if (!any_voice_active && m->state_mirror_idle_published) return;
    StateMirror* slot = m->state_mirror.begin_write();
    if (!slot) return;
    const uint32_t nlayers = std::min<uint32_t>(active_layers, MAX_LAYERS);
    bool all_zeroed = true;
    for (uint32_t v = 0; v < MAX_VOICES; ++v) {
        if (m->voices[v].stage != VOICE_IDLE) {
            for (uint32_t l = 0; l < nlayers; ++l) {
                slot->voice_snapshots[v].ssm_state.layers[l] = m->voices[v].ssm_state.layers[l];
            }
            slot->voice_snapshots[v].env = m->voices[v].env;
            slot->voice_snapshots[v].attack_rate = m->voices[v].attack_rate;
            slot->voice_snapshots[v].release_rate = m->voices[v].release_rate;
            slot->voice_snapshots[v].velocity = m->voices[v].velocity;
            slot->voice_snapshots[v].key = m->voices[v].key;
            slot->voice_snapshots[v].channel = m->voices[v].channel;
            slot->voice_snapshots[v].note_id = m->voices[v].note_id;
            slot->voice_snapshots[v].stage = m->voices[v].stage;
            m->mirror_zeroed_count[v] = 0;
            all_zeroed = false;
        } else if (m->mirror_zeroed_count[v] < 2) {
            std::memset(&slot->voice_snapshots[v], 0, sizeof(VoiceSnapshot));
            slot->voice_snapshots[v].key = -1;
            slot->voice_snapshots[v].channel = -1;
            slot->voice_snapshots[v].stage = VOICE_IDLE;
            ++m->mirror_zeroed_count[v];
            if (m->mirror_zeroed_count[v] < 2) all_zeroed = false;
        }
    }
    for (uint32_t i = 0; i < PARAM_COUNT; ++i) {
        slot->param_values[i] = m->param_values[i].load(std::memory_order_acquire);
    }
    slot->bypass_target = m->bypass_target.load(std::memory_order_acquire);
    slot->bypass_fade = m->bypass_fade.load(std::memory_order_relaxed);
    m->state_mirror.end_write();
    m->state_mirror.swap();
    m->state_mirror_idle_published = !any_voice_active && all_zeroed;
}

static bool match_voice(const Voice& voice, int32_t key, int32_t channel, int32_t clap_note_id) {
    if (clap_note_id != -1) {
        return voice.clap_note_id == clap_note_id;
    }
    if (key != -1 && voice.key != key) return false;
    if (channel != -1 && voice.channel != channel) return false;
    return true;
}

static void release_voice(MambaSynth* m, int32_t key, int32_t channel, int32_t clap_note_id) {
    for (uint32_t v = 0; v < MAX_VOICES; ++v) {
        Voice& voice = m->voices[v];
        if ((voice.stage == VOICE_ATTACK || voice.stage == VOICE_SUSTAIN) &&
            match_voice(voice, key, channel, clap_note_id)) {
            voice.stage = VOICE_RELEASE;
            m->voice_atomics[v].stage.store(VOICE_RELEASE, std::memory_order_relaxed);
        }
    }
}

static void choke_voice(MambaSynth* m, int32_t key, int32_t channel, int32_t clap_note_id) {
    for (uint32_t v = 0; v < MAX_VOICES; ++v) {
        Voice& voice = m->voices[v];
        if (voice.stage != VOICE_IDLE &&
            match_voice(voice, key, channel, clap_note_id)) {
            voice.stage = VOICE_IDLE;
            voice.env = 0.0f;
            voice.key = -1;
            voice.channel = -1;
            voice.clap_note_id = -1;
            auto& va = m->voice_atomics[v];
            va.stage.store(VOICE_IDLE, std::memory_order_relaxed);
            va.env.store(0.0f, std::memory_order_relaxed);
            va.key.store(-1, std::memory_order_relaxed);
            va.channel.store(-1, std::memory_order_relaxed);
            va.velocity.store(0.0f, std::memory_order_relaxed);
        }
    }
}

static bool apply_process_event(MambaSynth* m, const clap_event_header_t* hdr) {
    if (!hdr || hdr->space_id != CLAP_CORE_EVENT_SPACE_ID) return false;
    if (hdr->type == CLAP_EVENT_PARAM_VALUE && hdr->size >= sizeof(clap_event_param_value_t)) {
        const auto* ev = reinterpret_cast<const clap_event_param_value_t*>(hdr);
        apply_param_value(m, ev->param_id, ev->value);
        return false;
    } else if (hdr->type == CLAP_EVENT_NOTE_ON && hdr->size >= sizeof(clap_event_note_t)) {
        const auto* ev = reinterpret_cast<const clap_event_note_t*>(hdr);
        if (ev->velocity <= 0.0) {
            release_voice(m, ev->key, ev->channel, ev->note_id);
        } else {
            const int32_t v = allocate_voice(m);
            if (v >= 0) {
                start_voice(m, static_cast<uint32_t>(v), ev->key, ev->channel,
                            ev->note_id, static_cast<float>(ev->velocity),
                            next_monotonic_note_id(m));
            }
        }
        return true;
    } else if (hdr->type == CLAP_EVENT_NOTE_OFF && hdr->size >= sizeof(clap_event_note_t)) {
        const auto* ev = reinterpret_cast<const clap_event_note_t*>(hdr);
        release_voice(m, ev->key, ev->channel, ev->note_id);
        return true;
    } else if (hdr->type == CLAP_EVENT_NOTE_CHOKE && hdr->size >= sizeof(clap_event_note_t)) {
        const auto* ev = reinterpret_cast<const clap_event_note_t*>(hdr);
        choke_voice(m, ev->key, ev->channel, ev->note_id);
        return true;
    } else if (hdr->type == CLAP_EVENT_MIDI && hdr->size >= sizeof(clap_event_midi_t)) {
        const auto* ev = reinterpret_cast<const clap_event_midi_t*>(hdr);
        const uint8_t status = ev->data[0] & 0xF0u;
        const int32_t channel = static_cast<int32_t>(ev->data[0] & 0x0Fu);
        const int32_t d1 = static_cast<int32_t>(ev->data[1] & 0x7Fu);
        const uint8_t d2 = ev->data[2] & 0x7Fu;
        if (status == 0x90u && d2 > 0) {
            const int32_t v = allocate_voice(m);
            if (v >= 0) {
                start_voice(m, static_cast<uint32_t>(v), d1, channel, -1,
                            static_cast<float>(d2) / 127.0f, next_monotonic_note_id(m));
            }
            return true;
        } else if (status == 0x80u || (status == 0x90u && d2 == 0)) {
            release_voice(m, d1, channel, -1);
            return true;
        } else if (status == 0xB0u && d1 == 120) {
            choke_voice(m, -1, channel, -1);
            return true;
        } else if (status == 0xB0u && d1 == 123) {
            release_voice(m, -1, channel, -1);
            return true;
        }
    }
    return false;
}

static clap_process_status plugin_process(const clap_plugin_t* plugin,
    const clap_process_t* process) {
    if (!plugin || !process || !process->audio_outputs || process->audio_outputs_count == 0)
        return CLAP_PROCESS_ERROR;
    auto* m = get_synth(plugin);
    if (!m) return CLAP_PROCESS_ERROR;
    ScopedFtzDaz ftz_guard;  // FTZ/DAZ for the block; MXCSR restored on exit (E8/E19)
    m->thread_check.mark_audio_thread();

    clap_audio_buffer_t& out_bus = process->audio_outputs[0];
    const clap_audio_buffer_t* in_bus = process->audio_inputs && process->audio_inputs_count
        ? &process->audio_inputs[0] : nullptr;
    if ((!out_bus.data32 && !out_bus.data64) || out_bus.channel_count == 0)
        return CLAP_PROCESS_ERROR;

    // Drain any GUI parameter edits queued via param_queue and emit to host.
    {
        ParamAtom queued{};
        while (m->param_queue.pop(queued)) {
            apply_param_value(m, queued.id, queued.value);
            if (process->out_events && process->out_events->try_push && queued.id < PARAM_COUNT) {
                clap_event_param_value_t ev{};
                ev.header.size = sizeof(ev);
                ev.header.time = 0;
                ev.header.space_id = CLAP_CORE_EVENT_SPACE_ID;
                ev.header.type = CLAP_EVENT_PARAM_VALUE;
                ev.header.flags = 0;
                ev.param_id = queued.id;
                ev.cookie = &m->params[queued.id];
                ev.note_id = -1;
                ev.port_index = -1;
                ev.channel = -1;
                ev.key = -1;
                ev.value = m->param_values[queued.id].load(std::memory_order_acquire);
                process->out_events->try_push(process->out_events, &ev.header);
            }
        }
    }

    // Adopt the latest published core for the whole block (stable across the
    // per-sample loop); the main thread may publish a newer one mid-block and
    // we simply pick it up next block (E12).
    SSMInferenceCore* core = m->core.adopt();
    const size_t core_d_model = core ? core->config().d_model : 1;
    const uint32_t core_num_layers = core ? static_cast<uint32_t>(core->config().num_layers) : 0u;
    const bool loaded = m->weights_loaded.load(std::memory_order_acquire);

    // Read base conditioning once per block (preserves m->conditioning and
    // full 128-float ARA frame including f0_probs and semantic embedding).
    ConditioningFrame base_cf{};
    if (!m->conditioning.read(base_cf)) {
        std::memset(&base_cf, 0, sizeof(base_cf));
    } else {
        sanitize_conditioning_frame(base_cf);
    }

    ARAConditioningFrame ara_cf{};
    const bool ara_active = m->ara_conditioning.read(ara_cf) && ara_cf.tempo_bpm > 0.0f;
    if (ara_active) {
        sanitize_conditioning_frame(ara_cf);
        base_cf = ara_cf;
    }

    ConditioningFrame voice_cf[MAX_VOICES]{};
    bool voice_cond_active[MAX_VOICES]{};
    auto rebuild_voice_conditioning = [&]() {
        for (uint32_t v = 0; v < MAX_VOICES; ++v) {
            const Voice& voice = m->voices[v];
            if (voice.stage == VOICE_IDLE) {
                voice_cond_active[v] = false;
                continue;
            }
            voice_cf[v] = base_cf;
            if (voice.key >= 0) {
                const int pitch_class = voice.key % 12;
                if (!ara_active) {
                    voice_cf[v].key_sig = static_cast<float>(pitch_class);
                    std::memset(voice_cf[v].chord_one_hot, 0, sizeof(voice_cf[v].chord_one_hot));
                    voice_cf[v].chord_one_hot[pitch_class] = voice.velocity;
                }
                voice_cf[v].reserved[0] = static_cast<float>(voice.key) / 127.0f;
                voice_cf[v].reserved[1] = voice.velocity;
            }
            voice_cond_active[v] = cf_any_nonzero(voice_cf[v]);
        }
    };
    rebuild_voice_conditioning();

    const uint32_t frames = process->frames_count;
    uint32_t bypass_state = m->bypass_state.load(std::memory_order_relaxed);
    float fade = m->bypass_fade.load(std::memory_order_relaxed);
    const double c_gain = m->params[PARAM_GAIN].coeff;
    const double c_attack = m->params[PARAM_ATTACK].coeff;
    const double c_release = m->params[PARAM_RELEASE].coeff;
    const double c_mix = m->params[PARAM_MIX].coeff;
    double cur_gain = m->params[PARAM_GAIN].current.load(std::memory_order_relaxed);
    double cur_attack = m->params[PARAM_ATTACK].current.load(std::memory_order_relaxed);
    double cur_release = m->params[PARAM_RELEASE].current.load(std::memory_order_relaxed);
    double cur_mix = m->params[PARAM_MIX].current.load(std::memory_order_relaxed);
    double prev_attack = cur_attack;
    double prev_release = cur_release;
    float live_attack_rate = compute_envelope_rate(
        cur_attack, m->params[PARAM_ATTACK].min_value,
        m->params[PARAM_ATTACK].max_value, m->sample_rate);
    float live_release_rate = compute_envelope_rate(
        cur_release, m->params[PARAM_RELEASE].min_value,
        m->params[PARAM_RELEASE].max_value, m->sample_rate);

    const uint32_t event_count = process->in_events ? process->in_events->size(process->in_events) : 0;
    uint32_t event_index = 0;
    uint32_t sample_pos = 0;

    while (sample_pos < frames) {
        bool cond_dirty = false;
        // Process all events scheduled at or before the current sample_pos
        while (event_index < event_count) {
            const clap_event_header_t* hdr = process->in_events->get(process->in_events, event_index);
            if (!hdr) {
                ++event_index;
                continue;
            }
            if (hdr->time > sample_pos) {
                break;
            }
            if (apply_process_event(m, hdr)) cond_dirty = true;
            ++event_index;
        }

        // Determine the boundary for this sub-block (next event time or end of block)
        uint32_t sub_block_end = frames;
        while (event_index < event_count) {
            const clap_event_header_t* next_hdr = process->in_events->get(process->in_events, event_index);
            if (!next_hdr) {
                ++event_index;
                continue;
            }
            if (next_hdr->time <= sample_pos) {
                if (apply_process_event(m, next_hdr)) cond_dirty = true;
                ++event_index;
                continue;
            }
            if (next_hdr->time < sub_block_end) {
                sub_block_end = next_hdr->time;
            }
            break;
        }
        if (cond_dirty) {
            rebuild_voice_conditioning();
        }

        const bool bypass_target = m->bypass_target.load(std::memory_order_acquire);
        const double target_gain = m->params[PARAM_GAIN].target.load(std::memory_order_relaxed);
        const double target_attack = m->params[PARAM_ATTACK].target.load(std::memory_order_relaxed);
        const double target_release = m->params[PARAM_RELEASE].target.load(std::memory_order_relaxed);
        const double target_mix = m->params[PARAM_MIX].target.load(std::memory_order_relaxed);
        const uint32_t in_channels = in_bus ? std::min<uint32_t>(in_bus->channel_count, 2) : 0;

        for (uint32_t n = sample_pos; n < sub_block_end; ++n) {
            advance_one_pole(cur_gain, target_gain, c_gain);
            advance_one_pole(cur_attack, target_attack, c_attack);
            advance_one_pole(cur_release, target_release, c_release);
            advance_one_pole(cur_mix, target_mix, c_mix);
            advance_bypass_machine(bypass_target, fade, bypass_state, 1u, MambaSynth::BYPASS_FADE_RATE);

            if (std::fabs(cur_attack - prev_attack) > 1e-12) {
                live_attack_rate = compute_envelope_rate(
                    cur_attack, m->params[PARAM_ATTACK].min_value,
                    m->params[PARAM_ATTACK].max_value, m->sample_rate);
                prev_attack = cur_attack;
            }
            if (std::fabs(cur_release - prev_release) > 1e-12) {
                live_release_rate = compute_envelope_rate(
                    cur_release, m->params[PARAM_RELEASE].min_value,
                    m->params[PARAM_RELEASE].max_value, m->sample_rate);
                prev_release = cur_release;
            }

            float dry[2] = {0.0f, 0.0f};
            for (uint32_t ch = 0; ch < in_channels; ++ch) {
                float raw = 0.0f;
                if (in_bus->data32 && in_bus->data32[ch]) raw = in_bus->data32[ch][n];
                else if (in_bus->data64 && in_bus->data64[ch]) raw = static_cast<float>(in_bus->data64[ch][n]);
                dry[ch] = std::isfinite(raw) ? std::clamp(raw, -1e6f, 1e6f) : 0.0f;
            }
            const float mono = in_channels > 1 ? 0.5f * (dry[0] + dry[1]) : dry[0];
            alignas(64) float voice_sum[MAX_D_MODEL]{};
            const bool run_ssm = loaded && core && !(bypass_state == BYPASS_BYPASSED && fade >= 1.0f);
            for (uint32_t v = 0; v < MAX_VOICES; ++v) {
                Voice& voice = m->voices[v];
                if (voice.stage == VOICE_IDLE) continue;
                voice.attack_rate = live_attack_rate;
                voice.release_rate = live_release_rate;
                voice_envelope_step(voice.env, voice.stage, live_attack_rate, live_release_rate);
                if (voice.env <= 0.0f || !run_ssm) continue;
                alignas(64) float outbuf[MAX_D_MODEL]{};
                core->process_sample(mono, outbuf, voice.ssm_state,
                                     &voice_cf[v].tempo_bpm, voice_cond_active[v]);
                voice.last_out[0] = outbuf[0];
                voice.last_out[1] = (core_d_model > 1) ? outbuf[1] : outbuf[0];
                const float voice_amp = voice.env * voice.velocity;
                for (size_t d = 0; d < core_d_model; ++d) {
                    voice_sum[d] += outbuf[d] * voice_amp;
                }
            }
            float stereo_wet[2] = {0.0f, 0.0f};
            if (core_d_model == 1) {
                stereo_wet[0] = voice_sum[0];
                stereo_wet[1] = voice_sum[0];
            } else if (core_d_model == 2) {
                stereo_wet[0] = voice_sum[0];
                stereo_wet[1] = voice_sum[1];
            } else {
                for (size_t d = 0; d < core_d_model; ++d) {
                    stereo_wet[d & 1u] += voice_sum[d];
                }
                const float norm = 1.0f / static_cast<float>(core_d_model / 2);
                stereo_wet[0] *= norm;
                stereo_wet[1] *= norm;
            }
            stereo_wet[0] += m->steal_residual[0];
            stereo_wet[1] += m->steal_residual[1];
            m->steal_residual[0] *= 0.96f;
            m->steal_residual[1] *= 0.96f;

            for (uint32_t ch = 0; ch < out_bus.channel_count; ++ch) {
                const float channel_dry = ch < in_channels ? dry[ch] : mono;
                const float wet = stereo_wet[ch & 1u] * static_cast<float>(cur_gain);
                float sample = compute_mixed_sample(
                    channel_dry, wet, static_cast<float>(cur_mix), fade);
                if (!std::isfinite(sample)) sample = 0.0f;
                if (out_bus.data32 && out_bus.data32[ch]) out_bus.data32[ch][n] = sample;
                if (out_bus.data64 && out_bus.data64[ch]) out_bus.data64[ch][n] = sample;
            }
        }
        sample_pos = sub_block_end;
    }

    // Process any trailing events (e.g. at or beyond the last frame)
    while (event_index < event_count) {
        const clap_event_header_t* hdr = process->in_events->get(process->in_events, event_index);
        if (hdr) {
            apply_process_event(m, hdr);
        }
        ++event_index;
    }

    bool any_voice_active = false;
    for (uint32_t v = 0; v < MAX_VOICES; ++v) {
        const Voice& voice = m->voices[v];
        m->voice_atomics[v].stage.store(voice.stage, std::memory_order_relaxed);
        m->voice_atomics[v].env.store(voice.env, std::memory_order_relaxed);
        if (voice.stage != VOICE_IDLE) any_voice_active = true;
    }

    m->params[PARAM_GAIN].current.store(cur_gain, std::memory_order_relaxed);
    m->params[PARAM_ATTACK].current.store(cur_attack, std::memory_order_relaxed);
    m->params[PARAM_RELEASE].current.store(cur_release, std::memory_order_relaxed);
    m->params[PARAM_MIX].current.store(cur_mix, std::memory_order_relaxed);
    m->bypass_fade.store(fade, std::memory_order_relaxed);
    m->bypass_state.store(bypass_state, std::memory_order_relaxed);
    publish_state_mirror(m, core_num_layers, any_voice_active);
    return CLAP_PROCESS_CONTINUE;
}
// Default behaviour: thread-check is queried from the host, not provided
// by the plugin. The plugin's get_extension returns nullptr for
// CLAP_EXT_THREAD_CHECK, and helpers in clap_thread_check.h use the
// cached host pointer instead. We keep the slot here so get_extension
// is unambiguous about which IDs the plugin recognizes.
static const void* plugin_get_extension(const clap_plugin_t* plugin, const char* id) {
    if (!plugin || !id) return nullptr;

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
    if (strcmp(id, CLAP_EXT_TIMER_SUPPORT) == 0) {
        return &s_timer_support;
    }
    return nullptr;
}

static void plugin_on_main_thread(const clap_plugin_t* plugin) {
    auto* m = get_synth(plugin);
    if (m) {
        m->thread_check.set_host(m->host);
        m->core.reclaim();  // free cores the audio thread has retired (E12)
    }
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

    // Load into a local WeightLoader so failed loads leave the currently
    // active engine, config, and weight_path completely untouched.
    WeightLoader tmp_loader;
    if (tmp_loader.load(path) != LoadError::Ok) return false;

    const auto& hdr = tmp_loader.header();
    if (!WeightLoader::fits_in_static_weights_with(hdr)) {
        return false;
    }

    MambaConfig new_cfg = core_cfg;
    new_cfg.d_state = hdr.d_state;
    new_cfg.d_model = hdr.d_model;
    new_cfg.d_inner = hdr.d_inner;
    new_cfg.nheads = hdr.nheads;
    new_cfg.headdim = hdr.headdim;
    new_cfg.mimo_rank = 1;  // D3 enforces rank-1 SISO; step_in_proj reads MAX_MIMO_RANK stride
    new_cfg.cond_dim = hdr.cond_dim;
    new_cfg.num_layers = (hdr.version >= 2) ? hdr.num_layers : 1;
    new_cfg.num_rope_angles = hdr.d_state / 2;

    // Build the new engine fully OFF the audio path, then publish it (E12).
    auto* new_core = new (std::nothrow) SSMInferenceCore(new_cfg);
    auto w = std::unique_ptr<SSMWeights>(new (std::nothrow) SSMWeights());
    if (!new_core || !w) {
        delete new_core;
        return false;
    }

    for (uint32_t layer = 0; layer < new_cfg.num_layers; ++layer) {
        if (tmp_loader.dequantize_layer_to(layer, *w) != LoadError::Ok) {
            delete new_core;
            return false;
        }
        const float* raw_w = reinterpret_cast<const float*>(w.get());
        const size_t n_floats = sizeof(SSMWeights) / sizeof(float);
        for (size_t idx = 0; idx < n_floats; ++idx) {
            if (!std::isfinite(raw_w[idx])) {
                delete new_core;
                return false;
            }
        }
        new_core->set_layer_weights(layer, *w);
    }

    core_cfg = new_cfg;
    core.reclaim();   // free any previously retired core (main thread)
    core.publish(new_core);
    weights_loaded.store(true, std::memory_order_release);
    state_mirror_idle_published = false;

    // Record the loaded path so state save can round-trip it.
    std::strncpy(weight_path, path, MAX_WEIGHT_PATH - 1);
    weight_path[MAX_WEIGHT_PATH - 1] = '\0';

    return true;
}

const clap_plugin_t* mamba_synth_create(const clap_host_t* host) {
    auto* m = new (std::nothrow) MambaSynth();
    if (!m) return nullptr;
    m->host = host;
    m->weights_loaded.store(false, std::memory_order_relaxed);
    m->block_size = 512;
    m->is_active = false;
    m->is_processing.store(false, std::memory_order_relaxed);
    m->state_mirror_idle_published = false;
    m->steal_residual[0] = 0.0f;
    m->steal_residual[1] = 0.0f;
    for (uint32_t v = 0; v < MAX_VOICES; ++v) m->mirror_zeroed_count[v] = 2;
    m->latency = 0;  // Mamba-3: no conv1d
    m->bypass_target.store(false, std::memory_order_relaxed);
    m->bypass_state.store(BYPASS_PROCESSING, std::memory_order_relaxed);
    m->bypass_fade.store(0.0f, std::memory_order_relaxed);
    m->next_note_id = 1;
    m->gui_data = nullptr;
    for (uint32_t v = 0; v < MAX_VOICES; ++v) {
        auto& voice = m->voices[v];
        std::memset(&voice, 0, sizeof(voice));
        voice.stage = VOICE_IDLE;
        voice.key = -1;
        voice.channel = -1;
        voice.clap_note_id = -1;
        voice.attack_rate = MambaSynth::VOICE_ATTACK_RATE;
        voice.release_rate = MambaSynth::VOICE_RELEASE_RATE;
        auto& va = m->voice_atomics[v];
        va.stage.store(VOICE_IDLE, std::memory_order_relaxed);
        va.env.store(0.0f, std::memory_order_relaxed);
        va.key.store(-1, std::memory_order_relaxed);
        va.channel.store(-1, std::memory_order_relaxed);
        va.velocity.store(0.0f, std::memory_order_relaxed);
    }
    m->ara_conditioning.write(ARAConditioningFrame{});
    for (uint32_t i = 0; i < PARAM_COUNT; ++i) {
        m->param_values[i].store(0.0, std::memory_order_relaxed);
    }
    m->weight_path[0] = '\0';
    m->thread_check.set_host(host);

    auto* plugin = new (std::nothrow) clap_plugin_t(s_plugin_class);
    if (!plugin) { delete m; return nullptr; }
    plugin->plugin_data = m;
    return plugin;
}

}
