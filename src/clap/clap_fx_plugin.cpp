#include "clap/clap_fx_plugin.h"
#include "clap/clap_plugin.h"    // reuse ParamSmooth, ParamAtom, etc.
#include "clap/clap_common_utils.h"
#include "clap/clap_gui.h"
#include "core/ssm_inference_core.h"
#include "core/weight_loader.h"
#include "common/ring_buffer.h"
#include "common/conditioning_frame.h"
#include "config.h"
#include <cstring>
#include <cmath>
#include <algorithm>
#include <new>
#include <cstdlib>
#include <cstdio>

// ponytail: Mamba FX reuses the synth's SSM core with minimal wrapper.
// If the FX needs different SSM behavior (e.g., no FiLM, different
// projection), add FX-specific SSM paths here.

namespace mamba {


static FxState* get_fx(const clap_plugin_t* plugin) {
    return static_cast<FxState*>(plugin->plugin_data);
}

static void recompute_smooth_coefficients(FxState* f) {
    for (uint32_t i = 0; i < FX_PARAM_COUNT; ++i) {
        f->params[i].coeff = compute_one_pole_coeff(f->params[i].smooth_time_s, f->sample_rate);
    }
}

static void init_param_metadata(FxState* f) {
    f->params[FX_PARAM_GAIN].min_value = 0.0;
    f->params[FX_PARAM_GAIN].max_value = 2.0;
    f->params[FX_PARAM_GAIN].default_value = 1.0;
    f->params[FX_PARAM_GAIN].curve = PARAM_CURVE_LOG;
    f->params[FX_PARAM_GAIN].smooth_time_s = 0.010f;

    f->params[FX_PARAM_MIX].min_value = 0.0;
    f->params[FX_PARAM_MIX].max_value = 1.0;
    f->params[FX_PARAM_MIX].default_value = 1.0;
    f->params[FX_PARAM_MIX].curve = PARAM_CURVE_LINEAR;
    f->params[FX_PARAM_MIX].smooth_time_s = 0.010f;

    f->params[FX_PARAM_BYPASS].min_value = 0.0;
    f->params[FX_PARAM_BYPASS].max_value = 1.0;
    f->params[FX_PARAM_BYPASS].default_value = 0.0;
    f->params[FX_PARAM_BYPASS].curve = PARAM_CURVE_STEPPED;
    f->params[FX_PARAM_BYPASS].smooth_time_s = 0.0f;
}

// ---- CLAP plugin callbacks ----

static bool fx_init(const clap_plugin_t* plugin) {
    auto* f = get_fx(plugin);
    f->core_cfg = MambaConfig::max_config();
    f->core_cfg.num_layers = 1;
    f->core = new (f->core_storage) SSMInferenceCore(f->core_cfg);
    f->sample_rate = 48000.0;
    f->block_size = 512;
    f->is_active = false;
    f->is_processing = false;

    init_param_metadata(f);
    for (uint32_t i = 0; i < FX_PARAM_COUNT; ++i) {
        f->param_values[i].store(f->params[i].default_value, std::memory_order_relaxed);
        f->params[i].target.store(f->params[i].default_value, std::memory_order_relaxed);
        f->params[i].current.store(f->params[i].default_value, std::memory_order_relaxed);
    }
    recompute_smooth_coefficients(f);

    f->bypass_target.store(false, std::memory_order_relaxed);
    f->bypass_state.store(BYPASS_PROCESSING, std::memory_order_relaxed);
    f->bypass_fade.store(0.0f, std::memory_order_relaxed);
    f->weight_path[0] = '\0';
    f->thread_check.set_host(f->host);
    return true;
}

static void fx_destroy(const clap_plugin_t* plugin) {
    auto* f = get_fx(plugin);
    if (f->core) { f->core->~SSMInferenceCore(); f->core = nullptr; }
    delete f;
    delete plugin;
}

static bool fx_activate(const clap_plugin_t* plugin, double sample_rate,
                         uint32_t min_frames, uint32_t max_frames) {
    auto* f = get_fx(plugin);
    (void)min_frames;
    f->sample_rate = sample_rate;
    f->block_size = max_frames;
    f->is_active = true;
    recompute_smooth_coefficients(f);
    f->core->reset();
    return true;
}

static void fx_deactivate(const clap_plugin_t* plugin) {
    auto* f = get_fx(plugin);
    f->is_active = false;
}

static bool fx_start_processing(const clap_plugin_t* plugin) {
    auto* f = get_fx(plugin);
    f->is_processing = true;
    return true;
}

static void fx_stop_processing(const clap_plugin_t* plugin) {
    auto* f = get_fx(plugin);
    f->is_processing = false;
}

static void fx_reset(const clap_plugin_t* plugin) {
    auto* f = get_fx(plugin);
    f->core->reset();
    f->bypass_state.store(BYPASS_PROCESSING, std::memory_order_relaxed);
    f->bypass_fade.store(0.0f, std::memory_order_relaxed);
}

static void apply_param_value(FxState* f, clap_id id, double value) {
    if (id >= FX_PARAM_COUNT) return;
    f->param_values[id].store(value, std::memory_order_release);
    f->params[id].target.store(value, std::memory_order_release);
}

static uint32_t fx_params_count(const clap_plugin_t*) { return FX_PARAM_COUNT; }
static bool fx_params_get_info(const clap_plugin_t* plugin, uint32_t index, clap_param_info_t* info) {
    if (!info || index >= FX_PARAM_COUNT) return false;
    auto* f = get_fx(plugin); std::memset(info, 0, sizeof(*info));
    info->id = index; info->cookie = &f->params[index];
    info->min_value = f->params[index].min_value; info->max_value = f->params[index].max_value;
    info->default_value = f->params[index].default_value; info->flags = CLAP_PARAM_IS_AUTOMATABLE;
    if (index == FX_PARAM_BYPASS) info->flags |= CLAP_PARAM_IS_STEPPED | CLAP_PARAM_IS_BYPASS;
    const char* name = index == FX_PARAM_GAIN ? "Gain" : index == FX_PARAM_MIX ? "Mix" : "Bypass";
    std::strncpy(info->name, name, CLAP_NAME_SIZE - 1); return true;
}
static bool fx_params_get_value(const clap_plugin_t* plugin, clap_id id, double* value) {
    if (!value || id >= FX_PARAM_COUNT) return false;
    *value = get_fx(plugin)->param_values[id].load(std::memory_order_acquire); return true;
}
static bool fx_params_value_to_text(const clap_plugin_t*, clap_id id, double value, char* out, uint32_t size) {
    if (!out || size == 0 || id >= FX_PARAM_COUNT || !std::isfinite(value)) return false;
    if (id == FX_PARAM_BYPASS) std::snprintf(out, size, "%s", value >= 0.5 ? "On" : "Off");
    else std::snprintf(out, size, "%.3f", value); return true;
}
static bool fx_params_text_to_value(const clap_plugin_t*, clap_id id, const char* text, double* value) {
    if (!text || !value || id >= FX_PARAM_COUNT) return false;
    if (id == FX_PARAM_BYPASS) { *value = (!std::strcmp(text,"On") || !std::strcmp(text,"1")) ? 1.0 : 0.0; return true; }
    char* end = nullptr; const double parsed = std::strtod(text, &end);
    if (end == text || !std::isfinite(parsed)) return false; *value = parsed; return true;
}
static void fx_params_flush(const clap_plugin_t* plugin, const clap_input_events_t* in, const clap_output_events_t*) {
    if (!in) return; auto* f = get_fx(plugin);
    const uint32_t count = in->size(in);
    for (uint32_t i = 0; i < count; ++i) {
        const clap_event_header_t* hdr = in->get(in, i);
        if (!hdr || hdr->space_id != CLAP_CORE_EVENT_SPACE_ID || hdr->type != CLAP_EVENT_PARAM_VALUE ||
            hdr->size < sizeof(clap_event_param_value_t)) continue;
        const auto* ev = reinterpret_cast<const clap_event_param_value_t*>(hdr);
        if (ev->param_id < FX_PARAM_COUNT) apply_param_value(f, ev->param_id, ev->value);
    }
}
static const clap_plugin_params_t s_fx_params = {fx_params_count, fx_params_get_info, fx_params_get_value,
    fx_params_value_to_text, fx_params_text_to_value, fx_params_flush};
// FX: process audio in -> SSM -> audio out with dry/wet mix
static clap_process_status fx_process(const clap_plugin_t* plugin, const clap_process_t* process) {
    if (!process || !process->audio_outputs || process->audio_outputs_count == 0)
        return CLAP_PROCESS_ERROR;
    auto* f = get_fx(plugin); SSMBlock::apply_ftz_daz(); f->thread_check.mark_audio_thread();
    clap_audio_buffer_t& out = process->audio_outputs[0];
    const clap_audio_buffer_t* in = process->audio_inputs && process->audio_inputs_count ? &process->audio_inputs[0] : nullptr;
    if ((!out.data32 && !out.data64) || out.channel_count == 0) return CLAP_PROCESS_ERROR;
    double gain = f->params[FX_PARAM_GAIN].current.load(std::memory_order_relaxed);
    double mix = f->params[FX_PARAM_MIX].current.load(std::memory_order_relaxed);
    float fade = f->bypass_fade.load(std::memory_order_relaxed);
    const uint32_t event_count = process->in_events ? process->in_events->size(process->in_events) : 0;
    uint32_t event_index = 0;
    for (uint32_t n = 0; n < process->frames_count; ++n) {
        while (event_index < event_count) {
            const clap_event_header_t* hdr = process->in_events->get(process->in_events, event_index);
            if (!hdr) { ++event_index; continue; } if (hdr->time > n) break;
            if (hdr->space_id == CLAP_CORE_EVENT_SPACE_ID && hdr->type == CLAP_EVENT_PARAM_VALUE &&
                hdr->size >= sizeof(clap_event_param_value_t)) {
                const auto* ev = reinterpret_cast<const clap_event_param_value_t*>(hdr);
                apply_param_value(f, ev->param_id, ev->value);
            }
            ++event_index;
        }
        const bool bypass = f->params[FX_PARAM_BYPASS].target.load(std::memory_order_relaxed) > 0.5;
        f->bypass_target.store(bypass, std::memory_order_relaxed);
        fade = std::clamp(fade + (bypass ? FxState::BYPASS_FADE_RATE : -FxState::BYPASS_FADE_RATE), 0.0f, 1.0f);
        advance_one_pole(gain, f->params[FX_PARAM_GAIN].target.load(std::memory_order_relaxed), f->params[FX_PARAM_GAIN].coeff);
        advance_one_pole(mix, f->params[FX_PARAM_MIX].target.load(std::memory_order_relaxed), f->params[FX_PARAM_MIX].coeff);
        float dry[2]{}; const uint32_t nc = in ? std::min<uint32_t>(in->channel_count, 2) : 0;
        for (uint32_t ch=0; ch<nc; ++ch) {
            if (in->data32 && in->data32[ch]) dry[ch]=in->data32[ch][n];
            else if (in->data64 && in->data64[ch]) dry[ch]=static_cast<float>(in->data64[ch][n]);
        }
        const float mono = nc > 1 ? 0.5f*(dry[0]+dry[1]) : dry[0];
        alignas(64) float processed[MAX_D_MODEL]; f->core->process_sample(mono, processed);
        for (uint32_t ch=0; ch<out.channel_count; ++ch) {
            const float d = ch < nc ? dry[ch] : mono;
            const float wet = processed[0] * static_cast<float>(gain);
            const float sample = (d*static_cast<float>(1.0-mix)+wet*static_cast<float>(mix))*(1.0f-fade)+d*fade;
            if (out.data32 && out.data32[ch]) out.data32[ch][n]=sample;
            if (out.data64 && out.data64[ch]) out.data64[ch][n]=sample;
        }
    }
    f->params[FX_PARAM_GAIN].current.store(gain,std::memory_order_relaxed);
    f->params[FX_PARAM_MIX].current.store(mix,std::memory_order_relaxed);
    f->bypass_fade.store(fade,std::memory_order_relaxed);
    return CLAP_PROCESS_CONTINUE;
}
static uint32_t fx_tail_get(const clap_plugin_t* plugin) {
    const auto* f = get_fx(plugin);
    return static_cast<uint32_t>(std::clamp(0.1 * f->sample_rate, 0.0, 4294967294.0));
}
static const clap_plugin_tail_t s_fx_tail = {fx_tail_get};

// ---------------------------------------------------------------------------
// Stream helpers & state serialization
// ---------------------------------------------------------------------------

// Shared stream serialization helpers provided by clap/clap_common_utils.h

static bool fx_load_weights(FxState* f, const char* path) {
    if (!path || path[0] == '\0') return false;
    if (f->weight_loader.load(path) != 0) return false;

    const auto& hdr = f->weight_loader.header();
    if (!f->weight_loader.validate_dimensions(
            hdr.d_state, hdr.d_model, hdr.d_inner,
            hdr.nheads, hdr.headdim,
            static_cast<uint32_t>(f->core_cfg.num_layers))) {
        f->weight_loader.unload();
        return false;
    }

    f->core_cfg.d_state = hdr.d_state;
    f->core_cfg.d_model = hdr.d_model;
    f->core_cfg.d_inner = hdr.d_inner;
    f->core_cfg.nheads = hdr.nheads;
    f->core_cfg.headdim = hdr.headdim;
    f->core_cfg.cond_dim = hdr.cond_dim;
    if (hdr.version == 2) {
        f->core_cfg.num_layers = hdr.num_layers;
    }
    f->core->init(f->core_cfg);

    auto w_ptr = std::make_unique<SSMWeights>();
    SSMWeights& w = *w_ptr;

    for (uint32_t layer = 0; layer < f->core_cfg.num_layers; ++layer) {
        WeightLayerPointers ptrs;
        if (f->weight_loader.get_layer_pointers(layer, ptrs) != 0) {
            f->weight_loader.unload();
            return false;
        }

        std::memset(&w, 0, sizeof(SSMWeights));
        size_t in_proj_out = f->core_cfg.d_inner * 2
                           + f->core_cfg.nheads * MAX_MIMO_RANK * f->core_cfg.d_state * 2
                           + f->core_cfg.nheads * 3
                           + MAX_NUM_ROPE_ANGLES;
        std::memcpy(w.in_proj_w, ptrs.in_proj_w, in_proj_out * f->core_cfg.d_model * sizeof(float));
        std::memcpy(w.dt_bias, ptrs.dt_bias, f->core_cfg.nheads * sizeof(float));
        std::memcpy(w.B_bias, ptrs.B_bias, f->core_cfg.nheads * MAX_MIMO_RANK * f->core_cfg.d_state * sizeof(float));
        std::memcpy(w.C_bias, ptrs.C_bias, f->core_cfg.nheads * MAX_MIMO_RANK * f->core_cfg.d_state * sizeof(float));
        std::memcpy(w.B_norm_weight, ptrs.B_norm_weight, f->core_cfg.d_state * sizeof(float));
        std::memcpy(w.C_norm_weight, ptrs.C_norm_weight, f->core_cfg.d_state * sizeof(float));
        std::memcpy(w.mimo_x, ptrs.mimo_x, f->core_cfg.nheads * MAX_MIMO_RANK * f->core_cfg.headdim * sizeof(float));
        std::memcpy(w.mimo_z, ptrs.mimo_z, f->core_cfg.nheads * MAX_MIMO_RANK * f->core_cfg.headdim * sizeof(float));
        std::memcpy(w.mimo_o, ptrs.mimo_o, f->core_cfg.nheads * MAX_MIMO_RANK * f->core_cfg.headdim * sizeof(float));
        std::memcpy(w.D_proj, ptrs.D_proj, f->core_cfg.nheads * sizeof(float));
        std::memcpy(w.out_proj_w, ptrs.out_proj_w, f->core_cfg.d_model * f->core_cfg.d_inner * sizeof(float));
        std::memcpy(w.film_proj_w, ptrs.film_proj_w, 2 * f->core_cfg.d_model * f->core_cfg.cond_dim * sizeof(float));
        std::memcpy(w.dt_film_proj_w, ptrs.dt_film_proj_w, 2 * f->core_cfg.nheads * f->core_cfg.cond_dim * sizeof(float));

        f->core->set_layer_weights(layer, w);
    }
    f->weights_loaded.store(true, std::memory_order_release);
    std::strncpy(f->weight_path, path, sizeof(f->weight_path) - 1);
    f->weight_path[sizeof(f->weight_path) - 1] = '\0';
    return true;
}

static constexpr uint32_t kFxStateMagic   = 0x4D465853u; // 'MFXS'
static constexpr uint32_t kFxStateVersion = 1u;

static bool fx_state_save(const clap_plugin_t* plugin, const clap_ostream_t* stream) {
    if (!plugin || !stream) return false;
    auto* f = get_fx(plugin);

    if (!stream_write_u32(stream, kFxStateMagic)) return false;
    if (!stream_write_u32(stream, kFxStateVersion)) return false;

    const double gain = f->param_values[FX_PARAM_GAIN].load(std::memory_order_acquire);
    if (!stream_write_f64(stream, gain)) return false;

    const double mix = f->param_values[FX_PARAM_MIX].load(std::memory_order_acquire);
    if (!stream_write_f64(stream, mix)) return false;

    const double bypass = (f->bypass_target.load(std::memory_order_acquire) ||
                           f->param_values[FX_PARAM_BYPASS].load(std::memory_order_acquire) >= 0.5) ? 1.0 : 0.0;
    if (!stream_write_f64(stream, bypass)) return false;

    uint32_t path_len = 0;
    if (f->weight_path[0] != '\0') {
        path_len = static_cast<uint32_t>(std::strlen(f->weight_path));
        if (path_len >= sizeof(f->weight_path)) {
            path_len = static_cast<uint32_t>(sizeof(f->weight_path) - 1);
        }
    }
    if (!stream_write_u32(stream, path_len)) return false;
    if (path_len > 0) {
        if (!stream_write_all(stream, f->weight_path, path_len)) return false;
    }

    return true;
}

static bool fx_state_load(const clap_plugin_t* plugin, const clap_istream_t* stream) {
    if (!plugin || !stream) return false;
    auto* f = get_fx(plugin);

    uint32_t magic = 0;
    if (!stream_read_u32(stream, magic)) return false;
    if (magic != kFxStateMagic) return false;

    uint32_t version = 0;
    if (!stream_read_u32(stream, version)) return false;
    if (version == 0 || version > kFxStateVersion) return false;

    double gain = 0.0;
    if (!stream_read_f64(stream, gain) || !std::isfinite(gain) ||
        gain < f->params[FX_PARAM_GAIN].min_value || gain > f->params[FX_PARAM_GAIN].max_value) {
        return false;
    }

    double mix = 0.0;
    if (!stream_read_f64(stream, mix) || !std::isfinite(mix) ||
        mix < f->params[FX_PARAM_MIX].min_value || mix > f->params[FX_PARAM_MIX].max_value) {
        return false;
    }

    double bypass = 0.0;
    if (!stream_read_f64(stream, bypass) || !std::isfinite(bypass) ||
        bypass < f->params[FX_PARAM_BYPASS].min_value || bypass > f->params[FX_PARAM_BYPASS].max_value) {
        return false;
    }

    uint32_t path_len = 0;
    if (!stream_read_u32(stream, path_len) || path_len >= sizeof(f->weight_path)) {
        return false;
    }

    char path_buf[sizeof(f->weight_path)] = {0};
    if (path_len > 0) {
        if (!stream_read_all(stream, path_buf, path_len)) return false;
    }
    path_buf[path_len] = '\0';

    // Apply parameters
    apply_param_value(f, FX_PARAM_GAIN, gain);
    f->params[FX_PARAM_GAIN].current.store(gain, std::memory_order_release);

    apply_param_value(f, FX_PARAM_MIX, mix);
    f->params[FX_PARAM_MIX].current.store(mix, std::memory_order_release);

    const bool bypass_bool = (bypass >= 0.5);
    const double bypass_val = bypass_bool ? 1.0 : 0.0;
    apply_param_value(f, FX_PARAM_BYPASS, bypass_val);
    f->params[FX_PARAM_BYPASS].current.store(bypass_val, std::memory_order_release);

    f->bypass_target.store(bypass_bool, std::memory_order_release);
    f->bypass_fade.store(bypass_bool ? 1.0f : 0.0f, std::memory_order_relaxed);
    f->bypass_state.store(
        bypass_bool ? BYPASS_BYPASSED : BYPASS_PROCESSING,
        std::memory_order_relaxed);

    std::memset(f->weight_path, 0, sizeof(f->weight_path));
    if (path_len > 0) {
        std::memcpy(f->weight_path, path_buf, path_len);
        f->weight_path[path_len] = '\0';
        fx_load_weights(f, path_buf);
    }

    return true;
}

const clap_plugin_state_t s_fx_state = { fx_state_save, fx_state_load };

static const void* fx_get_extension(const clap_plugin_t* plugin, const char* id) {
    (void)plugin;
    if (std::strcmp(id, CLAP_EXT_AUDIO_PORTS) == 0) {
        // Reuse synth's audio port extension (stereo I/O)
        extern const clap_plugin_audio_ports_t s_audio_ports;
        return &s_audio_ports;
    }
    if (std::strcmp(id, CLAP_EXT_PARAMS) == 0) {
        return &s_fx_params;
    }
    if (std::strcmp(id, CLAP_EXT_STATE) == 0) {
        return &s_fx_state;
    }
    if (std::strcmp(id, CLAP_EXT_TAIL) == 0) {
        return &s_fx_tail;
    }
    if (std::strcmp(id, CLAP_EXT_LATENCY) == 0) {
        extern const clap_plugin_latency_t s_latency;
        return &s_latency;
    }
    return nullptr;
}

static void fx_on_main_thread(const clap_plugin_t* plugin) {
    auto* f = get_fx(plugin);
    if (f) f->thread_check.set_host(f->host);
}

// ---- Descriptor and plugin class ----

static const char* fx_features[] = {
    CLAP_PLUGIN_FEATURE_AUDIO_EFFECT,
    CLAP_PLUGIN_FEATURE_STEREO,
    nullptr
};

static const clap_plugin_descriptor_t fx_descriptor = {
    CLAP_VERSION_INIT,
    "com.emt.fx",
    "EmT FX",
    "EmT Audio",
    "https://emt-audio.com",
    "",
    "",
    "0.3.0",
    "EmT Mamba-3 SSM neural audio effect with FiLM conditioning",
    reinterpret_cast<const char* const*>(fx_features)
};

const clap_plugin_descriptor_t emt_fx_descriptor = fx_descriptor;
const clap_plugin_descriptor_t mamba_fx_descriptor = fx_descriptor;

static const clap_plugin_t fx_plugin_class = {
    &fx_descriptor,
    nullptr,
    fx_init,
    fx_destroy,
    fx_activate,
    fx_deactivate,
    fx_start_processing,
    fx_stop_processing,
    fx_reset,
    fx_process,
    fx_get_extension,
    fx_on_main_thread
};

const clap_plugin_t* mamba_fx_create(const clap_host_t* host) {
    auto* f = new (std::nothrow) FxState();
    if (!f) return nullptr;
    f->host = host;
    f->core = nullptr;
    f->weights_loaded.store(false, std::memory_order_relaxed);
    f->block_size = 512;
    f->is_active = false;
    f->is_processing = false;
    f->bypass_target.store(false, std::memory_order_relaxed);
    f->bypass_state.store(BYPASS_PROCESSING, std::memory_order_relaxed);
    f->bypass_fade.store(0.0f, std::memory_order_relaxed);
    f->weight_path[0] = '\0';
    f->thread_check.set_host(host);
    for (uint32_t i = 0; i < FX_PARAM_COUNT; ++i)
        f->param_values[i].store(0.0, std::memory_order_relaxed);

    auto* plugin = new (std::nothrow) clap_plugin_t(fx_plugin_class);
    if (!plugin) { delete f; return nullptr; }
    plugin->plugin_data = f;
    return plugin;
}

} // namespace mamba
