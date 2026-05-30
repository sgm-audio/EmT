#ifndef MAMBA_CLAP_PLUGIN_H
#define MAMBA_CLAP_PLUGIN_H

#include <clap/clap.h>
#include <atomic>
#include <cstdint>
#include <array>
#include <cstring>

#include "core/ssm_inference_core.h"
#include "core/weight_loader.h"
#include "common/ring_buffer.h"
#include "common/seqlock.h"
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

struct ParamAtom {
    clap_id id;
    double value;
    uint32_t frame_offset;
};

struct alignas(64) ConditioningFrame {
    float tempo_bpm;
    float key_sig;
    float chord_one_hot[12];
    float reserved[2];
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

    SPSCRingBuffer<ParamAtom, 256> param_queue;

    std::atomic<double> param_values[PARAM_COUNT];

    std::atomic<bool> bypass_target;
    alignas(64) std::atomic<float> bypass_fade;
    static constexpr float BYPASS_FADE_RATE = 1.0f / 256.0f;

    int32_t active_note;
    float active_velocity;

    Seqlocked<ConditioningFrame> conditioning;

    Seqlocked<ARAConditioningFrame> ara_conditioning;

    alignas(64) std::atomic<float> smoothed_gain;
    alignas(64) std::atomic<float> smoothed_mix;

    uint32_t latency;

    static constexpr size_t MAX_WEIGHT_PATH = 512;
    alignas(64) char weight_path[MAX_WEIGHT_PATH];
    WeightLoader weight_loader;

    bool load_weights(const char* path);
};

extern const clap_plugin_descriptor_t mamba_synth_descriptor;

extern const clap_plugin_audio_ports_t s_audio_ports;
extern const clap_plugin_note_ports_t s_note_ports;
extern const clap_plugin_params_t s_params;
extern const clap_plugin_state_t s_state;
extern const clap_plugin_tail_t s_tail;
extern const clap_plugin_latency_t s_latency;

} // namespace mamba

#endif
