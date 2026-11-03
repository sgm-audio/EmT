#ifndef MAMBA_FX_PLUGIN_H
#define MAMBA_FX_PLUGIN_H

#include <clap/clap.h>
#include "clap/clap_plugin.h"
#include "common/aligned_storage.h"
#include "common/rcu_slot.h"
#include "core/ssm_inference_core.h"
#include "core/weight_loader.h"
#include <atomic>
#include <cstddef>

namespace mamba {

// FX-specific param IDs (subset of synth params)
enum FxParamId : clap_id {
    FX_PARAM_GAIN = 0,
    FX_PARAM_MIX  = 1,
    FX_PARAM_BYPASS = 2,
    FX_PARAM_COUNT = 3
};

struct alignas(64) FxState {
    const clap_host_t* host;
    // Inference engine, swapped RCU-style (see RcuSlot in common/rcu_slot.h):
    // the main thread builds a fresh heap core and publish()es it; the audio
    // thread adopt()s the latest per block; the old core is reclaimed on the
    // main thread (E12, E7).
    RcuSlot<SSMInferenceCore> core;
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

// Load a weight file into an FX instance (the plugin's real model-load path).
// Exposed for tests: this is where the 'does this file fit the compiled
// limits?' decision is made, and it must accept every multi-layer model
// within those limits. Returns false (and unloads) on any rejection.
bool fx_load_weights(FxState* f, const char* path);

// Factory function for the Mamba FX plugin.
const clap_plugin_t* mamba_fx_create(const clap_host_t* host);

// Descriptor for the FX plugin.
extern const clap_plugin_descriptor_t emt_fx_descriptor;
extern const clap_plugin_descriptor_t mamba_fx_descriptor;

// State extension for Mamba FX.
extern const clap_plugin_state_t s_fx_state;

} // namespace mamba

#endif

