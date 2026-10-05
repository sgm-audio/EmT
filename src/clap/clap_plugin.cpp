#include "clap/clap_plugin.h"
#include "clap/clap_common_utils.h"
#include "clap/clap_gui.h"
#include "clap/param_table.h"
#include "config.h"
#include <cstring>
#include <cmath>
#include <cstdio>
#include <algorithm>
#include <clap/ext/timer-support.h>
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
    return static_cast<MambaSynth*>(plugin->plugin_data);
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
    auto* m = get_synth(plugin);
    m->core_cfg = MambaConfig::max_config();
    m->core_cfg.num_layers = 1;
    m->core = new (m->core_storage.data) SSMInferenceCore(m->core_cfg);
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
        std::memset(slot, 0, sizeof(StateMirror));
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
    delete m;
    delete plugin;
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

static void update_bypass_state_machine(MambaSynth* m, uint32_t frames) {
    const bool target = m->bypass_target.load(std::memory_order_acquire);
    uint32_t state = m->bypass_state.load(std::memory_order_relaxed);
    float fade = m->bypass_fade.load(std::memory_order_relaxed);

    advance_bypass_machine(target, fade, state, frames, MambaSynth::BYPASS_FADE_RATE);

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
        slot->voice_snapshots[v].attack_rate = m->voices[v].attack_rate;
        slot->voice_snapshots[v].release_rate = m->voices[v].release_rate;
        slot->voice_snapshots[v].velocity = m->voices[v].velocity;
        slot->voice_snapshots[v].key = m->voices[v].key;
        slot->voice_snapshots[v].channel = m->voices[v].channel;
        slot->voice_snapshots[v].note_id = m->voices[v].note_id;
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

static void release_voice(MambaSynth* m, int32_t key, int32_t channel) {
    for (uint32_t v = 0; v < MAX_VOICES; ++v) {
        Voice& voice = m->voices[v];
        if (voice.key == key && voice.channel == channel &&
            (voice.stage == VOICE_ATTACK || voice.stage == VOICE_SUSTAIN)) {
            voice.stage = VOICE_RELEASE;
            m->voice_atomics[v].stage.store(VOICE_RELEASE, std::memory_order_relaxed);
            return;
        }
    }
}

static void apply_process_event(MambaSynth* m, const clap_event_header_t* hdr) {
    if (!hdr || hdr->space_id != CLAP_CORE_EVENT_SPACE_ID) return;
    if (hdr->type == CLAP_EVENT_PARAM_VALUE && hdr->size >= sizeof(clap_event_param_value_t)) {
        const auto* ev = reinterpret_cast<const clap_event_param_value_t*>(hdr);
        apply_param_value(m, ev->param_id, ev->value);
    } else if ((hdr->type == CLAP_EVENT_NOTE_ON || hdr->type == CLAP_EVENT_NOTE_OFF) &&
               hdr->size >= sizeof(clap_event_note_t)) {
        const auto* ev = reinterpret_cast<const clap_event_note_t*>(hdr);
        if (hdr->type == CLAP_EVENT_NOTE_OFF || ev->velocity <= 0.0) {
            release_voice(m, ev->key, ev->channel);
        } else {
            const int32_t v = allocate_voice(m);
            if (v >= 0) start_voice(m, static_cast<uint32_t>(v), ev->key, ev->channel,
                                    static_cast<float>(ev->velocity), m->next_note_id++);
        }
    }
}

static clap_process_status plugin_process(const clap_plugin_t* plugin,
    const clap_process_t* process) {
    if (!plugin || !process || !process->audio_outputs || process->audio_outputs_count == 0)
        return CLAP_PROCESS_ERROR;
    auto* m = get_synth(plugin);
    SSMBlock::apply_ftz_daz();
    m->thread_check.mark_audio_thread();

    clap_audio_buffer_t& out_bus = process->audio_outputs[0];
    const clap_audio_buffer_t* in_bus = process->audio_inputs && process->audio_inputs_count
        ? &process->audio_inputs[0] : nullptr;
    if ((!out_bus.data32 && !out_bus.data64) || out_bus.channel_count == 0)
        return CLAP_PROCESS_ERROR;

    update_bypass_state_machine(m, process->frames_count);
    refresh_conditioning(m);

    const uint32_t frames = process->frames_count;
    const float fade = m->bypass_fade.load(std::memory_order_relaxed);
    const double c_gain = m->params[PARAM_GAIN].coeff;
    const double c_mix = m->params[PARAM_MIX].coeff;
    double cur_gain = m->params[PARAM_GAIN].current.load(std::memory_order_relaxed);
    double cur_mix = m->params[PARAM_MIX].current.load(std::memory_order_relaxed);
    const uint32_t event_count = process->in_events ? process->in_events->size(process->in_events) : 0;
    uint32_t event_index = 0;
    uint32_t sample_pos = 0;

    while (sample_pos < frames) {
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
            apply_process_event(m, hdr);
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
                apply_process_event(m, next_hdr);
                ++event_index;
                continue;
            }
            if (next_hdr->time < sub_block_end) {
                sub_block_end = next_hdr->time;
            }
            break;
        }

        for (uint32_t n = sample_pos; n < sub_block_end; ++n) {
            advance_one_pole(cur_gain, m->params[PARAM_GAIN].target.load(std::memory_order_relaxed), c_gain);
            advance_one_pole(cur_mix, m->params[PARAM_MIX].target.load(std::memory_order_relaxed), c_mix);

            float dry[2] = {0.0f, 0.0f};
            const uint32_t in_channels = in_bus ? std::min<uint32_t>(in_bus->channel_count, 2) : 0;
            for (uint32_t ch = 0; ch < in_channels; ++ch) {
                if (in_bus->data32 && in_bus->data32[ch]) dry[ch] = in_bus->data32[ch][n];
                else if (in_bus->data64 && in_bus->data64[ch]) dry[ch] = static_cast<float>(in_bus->data64[ch][n]);
            }
            const float mono = in_channels > 1 ? 0.5f * (dry[0] + dry[1]) : dry[0];
            float voice_sum = 0.0f;
            for (uint32_t v = 0; v < MAX_VOICES; ++v) {
                Voice& voice = m->voices[v];
                switch (voice.stage) {
                case VOICE_IDLE: continue;
                case VOICE_ATTACK:
                    voice.env += voice.attack_rate;
                    if (voice.env >= 1.0f) { voice.env = 1.0f; voice.stage = VOICE_SUSTAIN; }
                    break;
                case VOICE_SUSTAIN: voice.env = 1.0f; break;
                case VOICE_RELEASE:
                    voice.env -= voice.release_rate;
                    if (voice.env <= 0.0f) { voice.env = 0.0f; voice.stage = VOICE_SILENT; }
                    break;
                case VOICE_SILENT: voice.env = 0.0f; break;
                default: voice.env = 0.0f; break;
                }
                m->voice_atomics[v].stage.store(voice.stage, std::memory_order_relaxed);
                m->voice_atomics[v].env.store(voice.env, std::memory_order_relaxed);
                if (voice.env <= 0.0f) continue;
                alignas(64) float outbuf[MAX_D_MODEL];
                m->core->process_sample(mono, outbuf, voice.ssm_state);
                voice_sum += outbuf[0] * voice.env * voice.velocity;
            }
            const float wet = voice_sum * static_cast<float>(cur_gain);
            for (uint32_t ch = 0; ch < out_bus.channel_count; ++ch) {
                const float channel_dry = ch < in_channels ? dry[ch] : mono;
                const float mixed = channel_dry * static_cast<float>(1.0 - cur_mix)
                                  + wet * static_cast<float>(cur_mix);
                const float sample = mixed * (1.0f - fade) + channel_dry * fade;
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
    if (strcmp(id, CLAP_EXT_TIMER_SUPPORT) == 0) {
        return &s_timer_support;
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
    if (weight_loader.load(path) != LoadError::Ok) return false;

    // Validate dimensions against the loaded header
    const auto& hdr = weight_loader.header();
    // The file may describe any shape that fits the compiled MAX_* limits;
    // core_cfg is re-initialised from the header below. Do NOT compare the
    // header against core_cfg.num_layers here: at this point that is still
    // the 1-layer placeholder set in the constructor, so such a comparison
    // rejects every multi-layer model.
    if (!WeightLoader::fits_in_static_weights_with(hdr)) {
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
        if (weight_loader.get_layer_pointers(layer, ptrs) != LoadError::Ok) {
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
    auto* m = new (std::nothrow) MambaSynth();
    if (!m) return nullptr;
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
    m->gui_data = nullptr;
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

    auto* plugin = new (std::nothrow) clap_plugin_t(s_plugin_class);
    if (!plugin) { delete m; return nullptr; }
    plugin->plugin_data = m;
    return plugin;
}

}
