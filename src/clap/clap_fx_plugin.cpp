#include "clap/clap_fx_plugin.h"
#include "clap/clap_plugin.h"    // reuse ParamSmooth, ParamAtom, etc.
#include "clap/clap_gui.h"
#include "core/ssm_inference_core.h"
#include "core/weight_loader.h"
#include "core/arena_allocator.h"
#include "common/ring_buffer.h"
#include "common/conditioning_frame.h"
#include "config.h"
#include <cstring>
#include <cmath>
#include <algorithm>
#include <new>

// ponytail: Mamba FX reuses the synth's SSM core with minimal wrapper.
// If the FX needs different SSM behavior (e.g., no FiLM, different
// projection), add FX-specific SSM paths here.

namespace mamba {

static ArenaAllocator s_fx_arena(65536);

// FX-specific param IDs (subset of synth params)
enum FxParamId : clap_id {
    FX_PARAM_GAIN = 0,
    FX_PARAM_MIX  = 1,
    FX_PARAM_BYPASS = 2,
    FX_PARAM_COUNT = 3
};

struct alignas(64) FxState {
    const clap_host_t* host;
    std::byte core_storage[sizeof(SSMInferenceCore)];
    SSMInferenceCore* core;
    MambaConfig core_cfg;
    double sample_rate;
    uint32_t block_size;
    bool is_active;
    bool is_processing;
    std::atomic<bool> weights_loaded;

    // Param smoothing (shared with synth's infrastructure for simplicity)
    alignas(64) ParamSmooth params[FX_PARAM_COUNT];
    std::atomic<double> param_values[FX_PARAM_COUNT];

    // Bypass state
    std::atomic<bool> bypass_target;
    alignas(64) std::atomic<uint32_t> bypass_state;
    alignas(64) std::atomic<float> bypass_fade;
    static constexpr float BYPASS_FADE_RATE = 1.0f / 64.0f;

    // Weight loader
    WeightLoader weight_loader;
    char weight_path[512];

    // Thread check
    alignas(64) ThreadCheck thread_check;
};

static FxState* get_fx(const clap_plugin_t* plugin) {
    return static_cast<FxState*>(plugin->plugin_data);
}

static void recompute_smooth_coefficients(FxState* f) {
    const double sr = (f->sample_rate > 1.0) ? f->sample_rate : 48000.0;
    for (uint32_t i = 0; i < FX_PARAM_COUNT; ++i) {
        if (f->params[i].smooth_time_s <= 0.0f)
            f->params[i].coeff = 1.0;
        else
            f->params[i].coeff = 1.0 - std::exp(-1.0 / (f->params[i].smooth_time_s * sr));
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
    f->~FxState();
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

static void advance_one_pole(double& cur, double target, double coeff) {
    cur += coeff * (target - cur);
}

static void apply_param_value(FxState* f, clap_id id, double value) {
    if (id >= FX_PARAM_COUNT) return;
    f->param_values[id].store(value, std::memory_order_release);
    f->params[id].target.store(value, std::memory_order_release);
}

// FX: process audio in -> SSM -> audio out with dry/wet mix
static clap_process_status fx_process(const clap_plugin_t* plugin,
                                       const clap_process_t* process) {
    auto* f = get_fx(plugin);
    SSMBlock::apply_ftz_daz();
    f->thread_check.mark_audio_thread();

    if (!process->audio_inputs || !process->audio_outputs)
        return CLAP_PROCESS_CONTINUE;

    const float* in_l = process->audio_inputs->data32[0];
    float* out_l = process->audio_outputs->data32[0];
    uint32_t channels = process->audio_outputs->channel_count;
    if (!out_l) return CLAP_PROCESS_ERROR;

    const uint32_t frames = process->frames_count;
    const float fade = f->bypass_fade.load(std::memory_order_relaxed);

    // Pre-load param targets
    const double t_gain = f->params[FX_PARAM_GAIN].target.load(std::memory_order_relaxed);
    const double t_mix  = f->params[FX_PARAM_MIX].target.load(std::memory_order_relaxed);
    const double c_gain = f->params[FX_PARAM_GAIN].coeff;
    const double c_mix  = f->params[FX_PARAM_MIX].coeff;
    double cur_gain = f->params[FX_PARAM_GAIN].current.load(std::memory_order_relaxed);
    double cur_mix  = f->params[FX_PARAM_MIX].current.load(std::memory_order_relaxed);

    for (uint32_t n = 0; n < frames; ++n) {
        advance_one_pole(cur_gain, t_gain, c_gain);
        advance_one_pole(cur_mix,  t_mix,  c_mix);

        const float in_sample = in_l ? in_l[n] : 0.0f;
        alignas(64) float outbuf[MAX_D_MODEL];

        // Process through SSM core
        f->core->process_sample(in_sample, outbuf);
        const float processed = outbuf[0];

        // Dry/wet mix with gain
        const float mixed = in_sample * static_cast<float>(1.0 - cur_mix)
                          + processed * static_cast<float>(cur_gain * cur_mix);

        // Bypass crossfade
        out_l[n] = mixed * (1.0f - fade) + in_sample * fade;

        // Copy to other channels
        for (uint32_t ch = 1; ch < channels; ++ch) {
            process->audio_outputs->data32[ch][n] = out_l[n];
        }
    }

    f->params[FX_PARAM_GAIN].current.store(cur_gain, std::memory_order_relaxed);
    f->params[FX_PARAM_MIX].current.store(cur_mix, std::memory_order_relaxed);

    return CLAP_PROCESS_CONTINUE;
}

static const void* fx_get_extension(const clap_plugin_t* plugin, const char* id) {
    (void)plugin;
    if (std::strcmp(id, CLAP_EXT_AUDIO_PORTS) == 0) {
        // Reuse synth's audio port extension (stereo I/O)
        extern const clap_plugin_audio_ports_t s_audio_ports;
        return &s_audio_ports;
    }
    if (std::strcmp(id, CLAP_EXT_PARAMS) == 0) {
        // FX needs its own params. For now return nullptr (no adjustable params).
        return nullptr;
    }
    if (std::strcmp(id, CLAP_EXT_STATE) == 0) {
        return nullptr; // state not yet implemented for FX
    }
    if (std::strcmp(id, CLAP_EXT_TAIL) == 0) {
        extern const clap_plugin_tail_t s_tail;
        return &s_tail;
    }
    if (std::strcmp(id, CLAP_EXT_LATENCY) == 0) {
        extern const clap_plugin_latency_t s_latency;
        return &s_latency;
    }
    if (std::strcmp(id, CLAP_EXT_GUI) == 0) {
        return &s_gui;
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
    "com.mamba-ssm.fx",
    "Mamba FX",
    "Mamba Audio",
    "https://mamba-audio.com",
    "",
    "",
    "0.1.0",
    "Mamba-3 SSM neural audio effect with FiLM conditioning",
    reinterpret_cast<const char* const*>(fx_features)
};

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
    auto* f = static_cast<FxState*>(
        s_fx_arena.allocate(sizeof(FxState), alignof(FxState)));
    if (!f) return nullptr;
    new (f) FxState();
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

    auto* plugin = static_cast<clap_plugin_t*>(
        s_fx_arena.allocate(sizeof(clap_plugin_t), alignof(clap_plugin_t)));
    if (!plugin) return nullptr;
    new (plugin) clap_plugin_t(fx_plugin_class);
    plugin->plugin_data = f;
    return plugin;
}

} // namespace mamba
