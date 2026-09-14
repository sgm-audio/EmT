#include "core/ssm_inference_core.h"
#include <cstring>
#include <algorithm>
#include <new>

namespace mamba {

SSMInferenceCore::SSMInferenceCore()
    : cfg_(), layers_(nullptr), num_constructed_(0) {
    std::memset(layers_storage_, 0, sizeof(layers_storage_));
    std::memset(cond_.data(), 0, cond_.size() * sizeof(float));
}

SSMInferenceCore::SSMInferenceCore(const MambaConfig& cfg)
    : cfg_(cfg), layers_(nullptr), num_constructed_(0) {
    init(cfg);
}

void SSMInferenceCore::init(const MambaConfig& cfg) {
    if (layers_ != nullptr) {
        for (size_t i = 0; i < num_constructed_; ++i) {
            layers_[i].~SSMBlock();
        }
        num_constructed_ = 0;
    }

    cfg_ = cfg;
    // Clamp to MAX limits
    if (cfg_.num_layers > MAX_LAYERS) cfg_.num_layers = MAX_LAYERS;
    if (cfg_.d_state > MAX_D_STATE) cfg_.d_state = MAX_D_STATE;
    if (cfg_.d_model > MAX_D_MODEL) cfg_.d_model = MAX_D_MODEL;
    if (cfg_.d_inner > MAX_D_INNER) cfg_.d_inner = MAX_D_INNER;
    if (cfg_.nheads > MAX_NHEADS) cfg_.nheads = MAX_NHEADS;
    if (cfg_.headdim > MAX_HEADDIM) cfg_.headdim = MAX_HEADDIM;
    if (cfg_.mimo_rank > MAX_MIMO_RANK) cfg_.mimo_rank = MAX_MIMO_RANK;
    if (cfg_.num_bc_heads > MAX_NUM_BC_HEADS) cfg_.num_bc_heads = MAX_NUM_BC_HEADS;
    if (cfg_.num_rope_angles > MAX_NUM_ROPE_ANGLES) cfg_.num_rope_angles = MAX_NUM_ROPE_ANGLES;
    if (cfg_.cond_dim > COND_DIM) cfg_.cond_dim = COND_DIM;

    layers_ = reinterpret_cast<SSMBlock*>(layers_storage_);
    for (size_t i = 0; i < cfg_.num_layers; ++i) {
        new (&layers_[i]) SSMBlock(
            cfg_.d_state, cfg_.d_model, cfg_.d_inner,
            cfg_.nheads, cfg_.headdim, cfg_.mimo_rank,
            cfg_.num_bc_heads, cfg_.cond_dim);
        num_constructed_++;
    }

    std::memset(cond_.data(), 0, cond_.size() * sizeof(float));
}

SSMInferenceCore::~SSMInferenceCore() {
    for (size_t i = 0; i < num_constructed_; ++i) {
        layers_[i].~SSMBlock();
    }
}

void SSMInferenceCore::reset() {
    for (size_t i = 0; i < cfg_.num_layers; ++i) {
        layers_[i].reset();
    }
}

void SSMInferenceCore::process_sample(float input, float* output, MambaState& state) {
    if (cfg_.num_layers == 0) {
        output[0] = input;
        return;
    }
    alignas(64) float buf[MAX_D_MODEL];
    layers_[0].process_sample(input, buf, state.layers[0]);
    for (size_t i = 1; i < cfg_.num_layers; ++i) {
        // Sum d_model outputs to mono for next layer's scalar input
        float next = 0.0f;
        for (size_t j = 0; j < cfg_.d_model; ++j) {
            next += buf[j];
        }
        layers_[i].process_sample(next, buf, state.layers[i]);
    }
    std::copy(buf, buf + cfg_.d_model, output);
}

void SSMInferenceCore::process_sample(float input, float* output) {
    if (cfg_.num_layers == 0) {
        output[0] = input;
        return;
    }
    alignas(64) float buf[MAX_D_MODEL];
    layers_[0].process_sample(input, buf);
    for (size_t i = 1; i < cfg_.num_layers; ++i) {
        // Sum d_model outputs to mono for next layer's scalar input
        float next = 0.0f;
        for (size_t j = 0; j < cfg_.d_model; ++j) {
            next += buf[j];
        }
        layers_[i].process_sample(next, buf);
    }
    std::copy(buf, buf + cfg_.d_model, output);
}

void SSMInferenceCore::process_block(const float* input, float* output, size_t size, MambaState& state) {
    if (cfg_.num_layers == 0) {
        if (output != input) {
            std::memcpy(output, input, size * sizeof(float));
        }
        return;
    }
    if (cfg_.num_layers == 1) {
        layers_[0].process_block(input, output, size, state.layers[0]);
        return;
    }
    alignas(64) float buf_dm[MAX_BLOCK_SIZE * MAX_D_MODEL];
    alignas(64) float buf_mono[MAX_BLOCK_SIZE];
    layers_[0].process_block(input, buf_dm, size, state.layers[0]);
    for (size_t i = 1; i < cfg_.num_layers; ++i) {
        for (size_t t = 0; t < size; ++t) {
            float sum = 0.0f;
            for (size_t j = 0; j < cfg_.d_model; ++j) {
                sum += buf_dm[t * cfg_.d_model + j];
            }
            buf_mono[t] = sum;
        }
        if (i == cfg_.num_layers - 1) {
            layers_[i].process_block(buf_mono, output, size, state.layers[i]);
        } else {
            layers_[i].process_block(buf_mono, buf_dm, size, state.layers[i]);
        }
    }
}

void SSMInferenceCore::process_block(const float* input, float* output, size_t size) {
    if (cfg_.num_layers == 0) {
        if (output != input) {
            std::memcpy(output, input, size * sizeof(float));
        }
        return;
    }
    if (cfg_.num_layers == 1) {
        layers_[0].process_block(input, output, size);
        return;
    }
    alignas(64) float buf_dm[MAX_BLOCK_SIZE * MAX_D_MODEL];
    alignas(64) float buf_mono[MAX_BLOCK_SIZE];
    layers_[0].process_block(input, buf_dm, size);
    for (size_t i = 1; i < cfg_.num_layers; ++i) {
        for (size_t t = 0; t < size; ++t) {
            float sum = 0.0f;
            for (size_t j = 0; j < cfg_.d_model; ++j) {
                sum += buf_dm[t * cfg_.d_model + j];
            }
            buf_mono[t] = sum;
        }
        if (i == cfg_.num_layers - 1) {
            layers_[i].process_block(buf_mono, output, size);
        } else {
            layers_[i].process_block(buf_mono, buf_dm, size);
        }
    }
}

void SSMInferenceCore::set_layer_weights(size_t layer, const SSMWeights& w) {
    if (layer < cfg_.num_layers) {
        layers_[layer].set_weights(w);
    }
}

void SSMInferenceCore::set_layer_params(size_t layer, const SSMBlockParams& params) {
    if (layer < cfg_.num_layers) {
        layers_[layer].set_params(params);
    }
}

void SSMInferenceCore::set_conditioning(const float* cond_vector) {
    if (cond_vector) {
        size_t n = std::min(static_cast<size_t>(COND_DIM), cond_.size());
        std::memcpy(cond_.data(), cond_vector, n * sizeof(float));
    }
    for (size_t i = 0; i < cfg_.num_layers; ++i) {
        layers_[i].set_conditioning(cond_.data());
    }
}

void SSMInferenceCore::get_state(MambaState& out) const {
    for (size_t i = 0; i < cfg_.num_layers; ++i) {
        std::memcpy(&out.layers[i], &layers_[i].state(), sizeof(SSMState));
    }
}

void SSMInferenceCore::set_state(const MambaState& in) {
    for (size_t i = 0; i < cfg_.num_layers; ++i) {
        std::memcpy(&layers_[i].state(), &in.layers[i], sizeof(SSMState));
    }
}

} // namespace mamba
