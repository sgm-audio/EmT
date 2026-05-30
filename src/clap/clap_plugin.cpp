#include "clap/clap_plugin.h"
#include "config.h"
#include <cstring>
#include <cmath>
#include <cstdio>

namespace mamba {

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
    "0.1.0",
    "Mamba SSM synthesizer with ARA conditioning",
    reinterpret_cast<const char* const*>(s_features)
};

const clap_plugin_descriptor_t mamba_synth_descriptor = s_descriptor;

static MambaSynth* get_synth(const clap_plugin_t* plugin) {
    return static_cast<MambaSynth*>(plugin->plugin_data);
}

static bool plugin_init(const clap_plugin_t* plugin) {
    auto* m = get_synth(plugin);
    m->core_cfg = {MAX_D_STATE, MAX_D_MODEL, MAX_D_INNER, MAX_D_CONV, 1, COND_DIM};
    m->core = new (m->core_storage) SSMInferenceCore(m->core_cfg);
    m->sample_rate = 48000.0;
    m->block_size = 512;
    m->is_active = false;
    m->is_processing = false;
    m->latency = MAX_D_CONV - 1;

    m->param_values[PARAM_GAIN].store(1.0, std::memory_order_relaxed);
    m->param_values[PARAM_ATTACK].store(0.1, std::memory_order_relaxed);
    m->param_values[PARAM_RELEASE].store(0.3, std::memory_order_relaxed);
    m->param_values[PARAM_MIX].store(1.0, std::memory_order_relaxed);
    m->param_values[PARAM_BYPASS].store(0.0, std::memory_order_relaxed);

    m->bypass_target.store(false, std::memory_order_relaxed);
    m->bypass_fade.store(0.0f, std::memory_order_relaxed);
    m->active_note = -1;
    m->active_velocity = 0.0f;

    m->smoothed_gain.store(1.0f, std::memory_order_relaxed);
    m->smoothed_mix.store(1.0f, std::memory_order_relaxed);

    m->ara_conditioning.data = ARAConditioningFrame{};

    ConditioningFrame initial_frame{};
    initial_frame.tempo_bpm = 120.0f;
    initial_frame.key_sig = 0.0f;
    std::memset(initial_frame.chord_one_hot, 0, sizeof(initial_frame.chord_one_hot));
    std::memset(initial_frame.reserved, 0, sizeof(initial_frame.reserved));
    m->conditioning.write(initial_frame);

    m->weight_path[0] = '\0';

    return true;
}

static void plugin_destroy(const clap_plugin_t* plugin) {
    auto* m = get_synth(plugin);
    if (m->core) {
        m->core->~SSMInferenceCore();
        m->core = nullptr;
    }
    delete m;
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
    m->core->reset();
    m->bypass_fade.store(0.0f, std::memory_order_relaxed);
    m->smoothed_gain.store(static_cast<float>(m->param_values[PARAM_GAIN].load(std::memory_order_acquire)), std::memory_order_relaxed);
    m->smoothed_mix.store(static_cast<float>(m->param_values[PARAM_MIX].load(std::memory_order_acquire)), std::memory_order_relaxed);
    m->active_note = -1;
    m->active_velocity = 0.0f;
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
    m->bypass_fade.store(0.0f, std::memory_order_relaxed);
    m->smoothed_gain.store(static_cast<float>(m->param_values[PARAM_GAIN].load(std::memory_order_acquire)), std::memory_order_relaxed);
    m->smoothed_mix.store(static_cast<float>(m->param_values[PARAM_MIX].load(std::memory_order_acquire)), std::memory_order_relaxed);
    m->active_note = -1;
    m->active_velocity = 0.0f;
}

static void apply_params_from_queue(MambaSynth* m) {
    ParamAtom atom;
    while (m->param_queue.pop(atom)) {
        if (atom.id < PARAM_COUNT) {
            m->param_values[atom.id].store(atom.value, std::memory_order_release);
            if (atom.id == PARAM_BYPASS) {
                m->bypass_target.store(atom.value > 0.5, std::memory_order_release);
            }
        }
    }
}

static clap_process_status plugin_process(const clap_plugin_t* plugin,
    const clap_process_t* process) {
    auto* m = get_synth(plugin);

    SSMBlock::apply_ftz_daz();

    apply_params_from_queue(m);

    bool bypass = m->bypass_target.load(std::memory_order_acquire);
    float fade = m->bypass_fade.load(std::memory_order_relaxed);
    if (bypass && fade < 1.0f) {
        fade = std::min(1.0f, fade + MambaSynth::BYPASS_FADE_RATE);
        m->bypass_fade.store(fade, std::memory_order_relaxed);
    } else if (!bypass && fade > 0.0f) {
        fade = std::max(0.0f, fade - MambaSynth::BYPASS_FADE_RATE);
        m->bypass_fade.store(fade, std::memory_order_relaxed);
    }

    ConditioningFrame cf;
    if (m->conditioning.read(cf)) {
        m->core->set_conditioning(&cf.tempo_bpm);
    }

    {
        ARAConditioningFrame ara_cf;
        if (m->ara_conditioning.read(ara_cf)) {
            float cond[COND_DIM];
            std::memset(cond, 0, sizeof(cond));
            cond[0] = ara_cf.tempo_bpm;
            cond[1] = ara_cf.key_sig;
            for (int i = 0; i < 12; ++i) {
                cond[2 + i] = ara_cf.chord_one_hot[i];
            }
            m->core->set_conditioning(cond);
        }
    }

    const uint32_t frames = process->frames_count;

    for (uint32_t i = 0; i < process->in_events->size(process->in_events); ++i) {
        const clap_event_header_t* hdr = process->in_events->get(process->in_events, i);
        if (hdr->type == CLAP_EVENT_NOTE_ON) {
            auto* ev = reinterpret_cast<const clap_event_note_t*>(hdr);
            m->active_note = ev->key;
            m->active_velocity = static_cast<float>(ev->velocity);
        } else if (hdr->type == CLAP_EVENT_NOTE_OFF) {
            auto* ev = reinterpret_cast<const clap_event_note_t*>(hdr);
            if (ev->key == m->active_note) {
                m->active_note = -1;
                m->active_velocity = 0.0f;
            }
        } else if (hdr->type == CLAP_EVENT_PARAM_VALUE) {
            auto* ev = reinterpret_cast<const clap_event_param_value_t*>(hdr);
            if (ev->param_id < PARAM_COUNT) {
                m->param_values[ev->param_id].store(ev->value, std::memory_order_release);
                if (ev->param_id == PARAM_BYPASS) {
                    m->bypass_target.store(ev->value > 0.5, std::memory_order_release);
                }
            }
        }
    }

    const float* input = nullptr;
    float* output = nullptr;

    if (process->audio_inputs_count > 0 && process->audio_inputs->data32) {
        input = process->audio_inputs->data32[0];
    }
    if (process->audio_outputs_count > 0 && process->audio_outputs->data32) {
        output = process->audio_outputs->data32[0];
    }

    if (!output) return CLAP_PROCESS_ERROR;

    float coeff = 1.0f - expf(-1.0f / (0.01f * static_cast<float>(m->sample_rate)));
    float target_gain = static_cast<float>(m->param_values[PARAM_GAIN].load(std::memory_order_acquire));
    float target_mix = static_cast<float>(m->param_values[PARAM_MIX].load(std::memory_order_acquire));

    for (uint32_t n = 0; n < frames; ++n) {
        float sg = m->smoothed_gain.load(std::memory_order_relaxed);
        sg += coeff * (target_gain - sg);
        m->smoothed_gain.store(sg, std::memory_order_relaxed);

        float sm = m->smoothed_mix.load(std::memory_order_relaxed);
        sm += coeff * (target_mix - sm);
        m->smoothed_mix.store(sm, std::memory_order_relaxed);

        float in_sample = input ? input[n] : 0.0f;

        if (m->active_note >= 0) {
            int pitch_class = m->active_note % 12;
            ConditioningFrame note_cf{};
            note_cf.tempo_bpm = 120.0f;
            note_cf.key_sig = static_cast<float>(pitch_class);
            std::memset(note_cf.chord_one_hot, 0, sizeof(note_cf.chord_one_hot));
            note_cf.chord_one_hot[pitch_class] = m->active_velocity;
            note_cf.reserved[0] = static_cast<float>(m->active_note) / 127.0f;
            note_cf.reserved[1] = m->active_velocity;
            m->core->set_conditioning(&note_cf.tempo_bpm);
            in_sample += m->active_velocity * 0.3f;
        }

        float processed = m->core->process_sample(in_sample);

        float dry = in_sample;
        float wet = processed * sg;
        float mixed = dry * (1.0f - sm) + wet * sm;

        output[n] = mixed * (1.0f - fade) + dry * fade;

        uint32_t ch_count = process->audio_outputs->channel_count;
        for (uint32_t ch = 1; ch < ch_count; ++ch) {
            process->audio_outputs->data32[ch][n] = output[n];
        }
    }

    return CLAP_PROCESS_CONTINUE;
}

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
    (void)plugin;
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
    if (!weight_loader.validate_dimensions(
            MAX_D_STATE, MAX_D_MODEL, MAX_D_INNER, MAX_D_CONV,
            static_cast<uint32_t>(core_cfg.num_layers))) {
        weight_loader.unload();
        return false;
    }
    for (uint32_t layer = 0; layer < core_cfg.num_layers; ++layer) {
        WeightLayerPointers ptrs;
        if (weight_loader.get_layer_pointers(layer, ptrs) != 0) {
            weight_loader.unload();
            return false;
        }
        SSMWeights w{};
        std::memcpy(w.A_log, ptrs.A_log, MAX_D_STATE * sizeof(float));
        std::memcpy(w.D_proj, ptrs.D_proj, MAX_D_STATE * sizeof(float));
        std::memcpy(w.dt_bias, ptrs.dt_bias, MAX_D_STATE * sizeof(float));
        std::memcpy(w.dt_proj_w, ptrs.dt_proj_w, MAX_D_STATE * MAX_D_INNER * sizeof(float));
        std::memcpy(w.B_proj_w, ptrs.B_proj_w, MAX_D_INNER * MAX_D_STATE * sizeof(float));
        std::memcpy(w.C_proj_w, ptrs.C_proj_w, MAX_D_INNER * MAX_D_STATE * sizeof(float));
        std::memcpy(w.conv1d_w, ptrs.conv1d_w, MAX_D_CONV * MAX_D_INNER * sizeof(float));
        std::memcpy(w.conv1d_b, ptrs.conv1d_b, MAX_D_INNER * sizeof(float));
        std::memcpy(w.W_in, ptrs.W_in, MAX_D_INNER * MAX_D_MODEL * sizeof(float));
        std::memcpy(w.W_out, ptrs.W_out, MAX_D_MODEL * MAX_D_INNER * sizeof(float));
        std::memcpy(w.W_ssm, ptrs.W_ssm, MAX_D_MODEL * MAX_D_STATE * sizeof(float));
        std::memcpy(w.b_in, ptrs.b_in, MAX_D_INNER * sizeof(float));
        core->set_layer_weights(layer, w);
    }
    weights_loaded.store(true, std::memory_order_release);
    return true;
}

const clap_plugin_t* mamba_synth_create(const clap_host_t* host) {
    auto* m = new MambaSynth();
    m->host = host;
    m->core = nullptr;
    m->weights_loaded.store(false, std::memory_order_relaxed);
    m->block_size = 512;
    m->is_active = false;
    m->is_processing = false;
    m->latency = MAX_D_CONV - 1;
    m->bypass_target.store(false, std::memory_order_relaxed);
    m->bypass_fade.store(0.0f, std::memory_order_relaxed);
    m->active_note = -1;
    m->active_velocity = 0.0f;
    m->smoothed_gain.store(1.0f, std::memory_order_relaxed);
    m->smoothed_mix.store(1.0f, std::memory_order_relaxed);
    m->ara_conditioning.data = ARAConditioningFrame{};
    for (uint32_t i = 0; i < PARAM_COUNT; ++i) {
        m->param_values[i].store(0.0, std::memory_order_relaxed);
    }
    m->weight_path[0] = '\0';

    auto* plugin = new clap_plugin_t(s_plugin_class);
    plugin->plugin_data = m;
    return plugin;
}

}