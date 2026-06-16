#include "clap/clap_plugin.h"
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

    m->active_note.key.store(-1, std::memory_order_relaxed);
    m->active_note.channel.store(-1, std::memory_order_relaxed);
    m->active_note.velocity.store(0.0f, std::memory_order_relaxed);
    m->active_note.note_id.store(0, std::memory_order_relaxed);
    m->active_note.is_pressed.store(false, std::memory_order_relaxed);

    m->voice_state.store(VOICE_IDLE, std::memory_order_relaxed);
    m->voice_env.store(0.0f, std::memory_order_relaxed);
    m->voice_sample_count = 0;
    m->voice_attack_rate = 1.0f / 1024.0f;
    m->voice_release_rate = 1.0f / 4096.0f;
    m->voice_velocity = 0.0f;
    m->voice_key = -1;

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

    m->active_note.key.store(-1, std::memory_order_relaxed);
    m->active_note.channel.store(-1, std::memory_order_relaxed);
    m->active_note.velocity.store(0.0f, std::memory_order_relaxed);
    m->active_note.is_pressed.store(false, std::memory_order_relaxed);
    m->voice_state.store(VOICE_IDLE, std::memory_order_relaxed);
    m->voice_env.store(0.0f, std::memory_order_relaxed);
    m->voice_sample_count = 0;
    m->voice_velocity = 0.0f;
    m->voice_key = -1;

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

    m->active_note.key.store(-1, std::memory_order_relaxed);
    m->active_note.channel.store(-1, std::memory_order_relaxed);
    m->active_note.velocity.store(0.0f, std::memory_order_relaxed);
    m->active_note.is_pressed.store(false, std::memory_order_relaxed);
    m->voice_state.store(VOICE_IDLE, std::memory_order_relaxed);
    m->voice_env.store(0.0f, std::memory_order_relaxed);
    m->voice_sample_count = 0;
    m->voice_velocity = 0.0f;
    m->voice_key = -1;
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
                uint8_t st = m->voice_state.load(std::memory_order_relaxed);
                if (st == VOICE_ATTACK || st == VOICE_SUSTAIN) {
                    m->voice_state.store(VOICE_RELEASE, std::memory_order_relaxed);
                    m->voice_sample_count = 0;
                }
                m->active_note.is_pressed.store(false, std::memory_order_relaxed);
            } else {
                // Steal voice: reset SSM state, start attack.
                m->core->reset();
                m->voice_state.store(VOICE_ATTACK, std::memory_order_relaxed);
                m->voice_sample_count = 0;
                m->voice_velocity = static_cast<float>(ev->velocity);
                m->voice_key = ev->key;
                m->active_note.key.store(ev->key, std::memory_order_relaxed);
                m->active_note.channel.store(ev->channel, std::memory_order_relaxed);
                m->active_note.velocity.store(
                    static_cast<float>(ev->velocity), std::memory_order_relaxed);
                m->active_note.is_pressed.store(true, std::memory_order_relaxed);
                m->active_note.note_id.fetch_add(1, std::memory_order_relaxed);
            }
        } else if (hdr->type == CLAP_EVENT_NOTE_OFF) {
            const auto* ev = reinterpret_cast<const clap_event_note_t*>(hdr);
            int32_t cur = m->active_note.key.load(std::memory_order_relaxed);
            if (ev->key == cur) {
                uint8_t st = m->voice_state.load(std::memory_order_relaxed);
                if (st == VOICE_ATTACK || st == VOICE_SUSTAIN) {
                    m->voice_state.store(VOICE_RELEASE, std::memory_order_relaxed);
                    m->voice_sample_count = 0;
                }
                m->active_note.key.store(-1, std::memory_order_relaxed);
                m->active_note.channel.store(-1, std::memory_order_relaxed);
                m->active_note.velocity.store(0.0f, std::memory_order_relaxed);
                m->active_note.is_pressed.store(false, std::memory_order_relaxed);
            }
        } else if (hdr->type == CLAP_EVENT_PARAM_VALUE) {
            const auto* ev = reinterpret_cast<const clap_event_param_value_t*>(hdr);
            apply_param_value(m, ev->param_id, ev->value);
        }
    }
}

// Update the conditioning vector based on the active note (if any) and
// the latest ARA conditioning frame. Called once per process() block
// since the note is monophonic and the ARA frame changes slowly.
static void refresh_conditioning(MambaSynth* m) {
    ConditioningFrame cf{};
    cf.tempo_bpm = 120.0f;
    cf.key_sig = 0.0f;
    std::memset(cf.chord_one_hot, 0, sizeof(cf.chord_one_hot));
    cf.reserved[0] = 0.0f;
    cf.reserved[1] = 0.0f;

    const int32_t key = m->active_note.key.load(std::memory_order_relaxed);
    const float vel = m->active_note.velocity.load(std::memory_order_relaxed);
    if (key >= 0) {
        const int pitch_class = key % 12;
        cf.key_sig = static_cast<float>(pitch_class);
        cf.chord_one_hot[pitch_class] = vel;
        cf.reserved[0] = static_cast<float>(key) / 127.0f;
        cf.reserved[1] = vel;
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
    m->core->get_state(slot->h_state);
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

    apply_params_from_queue(m);
    process_note_events(m, process->in_events, process->frames_count);

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

    for (uint32_t n = 0; n < frames; ++n) {
        // Smooth only audio-rate params; bypass/attack/release don't
        // affect the per-sample coefficient here.
        advance_one_pole(cur_gain, t_gain, c_gain);
        advance_one_pole(cur_mix,  t_mix,  c_mix);

        // Voice envelope state machine (per-sample).
        float env = m->voice_env.load(std::memory_order_relaxed);
        uint8_t vs = m->voice_state.load(std::memory_order_relaxed);
        switch (vs) {
        case VOICE_ATTACK:
            env += m->voice_attack_rate;
            if (env >= 1.0f) {
                env = 1.0f;
                m->voice_state.store(VOICE_SUSTAIN, std::memory_order_relaxed);
            }
            break;
        case VOICE_SUSTAIN:
            env = 1.0f;
            break;
        case VOICE_RELEASE:
            env -= m->voice_release_rate;
            if (env <= 0.0f) {
                env = 0.0f;
                m->voice_state.store(VOICE_SILENT, std::memory_order_relaxed);
            }
            break;
        case VOICE_SILENT:
            env = 0.0f;
            break;
        case VOICE_IDLE:
        default:
            env = 0.0f;
            break;
        }
        m->voice_env.store(env, std::memory_order_relaxed);
        m->voice_sample_count++;

        const float in_sample = input ? input[n] : 0.0f;
        alignas(64) float outbuf[MAX_D_MODEL];
        m->core->process_sample(in_sample, outbuf);
        const float processed = outbuf[0];

        const float dry = in_sample;
        const float wet = processed * static_cast<float>(cur_gain) * env * m->voice_velocity;
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
    m->active_note.key.store(-1, std::memory_order_relaxed);
    m->active_note.channel.store(-1, std::memory_order_relaxed);
    m->active_note.velocity.store(0.0f, std::memory_order_relaxed);
    m->active_note.is_pressed.store(false, std::memory_order_relaxed);
    m->voice_state.store(VOICE_IDLE, std::memory_order_relaxed);
    m->voice_env.store(0.0f, std::memory_order_relaxed);
    m->voice_sample_count = 0;
    m->voice_attack_rate = 1.0f / 1024.0f;
    m->voice_release_rate = 1.0f / 4096.0f;
    m->voice_velocity = 0.0f;
    m->voice_key = -1;
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
