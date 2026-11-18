#include "core/ssm_inference_core.h"
#include <cstring>
#include <cmath>
#include <algorithm>
#include <new>

namespace mamba {

SSMInferenceCore::SSMInferenceCore()
    : cfg_(), layers_(nullptr), num_constructed_(0) {
    std::memset(layers_storage_, 0, sizeof(layers_storage_));
    std::memset(cond_.data(), 0, cond_.size() * sizeof(float));
    std::memset(block_scratch_dm_.data(), 0, block_scratch_dm_.size() * sizeof(float));
    std::memset(block_scratch_mono_.data(), 0, block_scratch_mono_.size() * sizeof(float));
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
    cond_active_ = false;
    std::memset(block_scratch_dm_.data(), 0, block_scratch_dm_.size() * sizeof(float));
    std::memset(block_scratch_mono_.data(), 0, block_scratch_mono_.size() * sizeof(float));
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

void SSMInferenceCore::process_sample(float input, float* output, MambaState& state,
                                      const float* cond_vector, bool cond_active) {
    if (!std::isfinite(input)) input = 0.0f;
    if (cfg_.num_layers == 0) {
        if (cfg_.d_model > 1) {
            std::memset(output, 0, cfg_.d_model * sizeof(float));
        }
        output[0] = input;
        return;
    }
    alignas(64) float buf[MAX_D_MODEL];
    layers_[0].process_sample(input, buf, state.layers[0], cond_vector, cond_active);
    for (size_t i = 1; i < cfg_.num_layers; ++i) {
        // Sum d_model outputs to mono for next layer's scalar input
        float next = 0.0f;
        for (size_t j = 0; j < cfg_.d_model; ++j) {
            next += buf[j];
        }
        layers_[i].process_sample(next, buf, state.layers[i], cond_vector, cond_active);
    }
    std::copy(buf, buf + cfg_.d_model, output);
}

void SSMInferenceCore::process_sample(float input, float* output, MambaState& state) {
    process_sample(input, output, state, cond_.data(), cond_active_);
}

void SSMInferenceCore::process_sample(float input, float* output) {
    if (!std::isfinite(input)) input = 0.0f;
    if (cfg_.num_layers == 0) {
        if (cfg_.d_model > 1) {
            std::memset(output, 0, cfg_.d_model * sizeof(float));
        }
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
        if (cfg_.d_model == 1) {
            if (output != input) {
                std::memcpy(output, input, size * sizeof(float));
            }
        } else {
            for (size_t t = 0; t < size; ++t) {
                float* dst = output + t * cfg_.d_model;
                std::memset(dst, 0, cfg_.d_model * sizeof(float));
                dst[0] = input[t];
            }
        }
        return;
    }
    if (cfg_.num_layers == 1) {
        layers_[0].process_block(input, output, size, state.layers[0]);
        return;
    }
    float* buf_dm = block_scratch_dm_.data();
    float* buf_mono = block_scratch_mono_.data();
    size_t offset = 0;
    while (offset < size) {
        size_t chunk = size - offset;
        if (chunk > MAX_BLOCK_SIZE) chunk = MAX_BLOCK_SIZE;

        const float* in_chunk = input + offset;
        float* out_chunk = output + offset * cfg_.d_model;

        layers_[0].process_block(in_chunk, buf_dm, chunk, state.layers[0]);
        for (size_t i = 1; i < cfg_.num_layers; ++i) {
            for (size_t t = 0; t < chunk; ++t) {
                float sum = 0.0f;
                for (size_t j = 0; j < cfg_.d_model; ++j) {
                    sum += buf_dm[t * cfg_.d_model + j];
                }
                buf_mono[t] = sum;
            }
            if (i == cfg_.num_layers - 1) {
                layers_[i].process_block(buf_mono, out_chunk, chunk, state.layers[i]);
            } else {
                layers_[i].process_block(buf_mono, buf_dm, chunk, state.layers[i]);
            }
        }
        offset += chunk;
    }
}

void SSMInferenceCore::process_block(const float* input, float* output, size_t size) {
    if (cfg_.num_layers == 0) {
        if (cfg_.d_model == 1) {
            if (output != input) {
                std::memcpy(output, input, size * sizeof(float));
            }
        } else {
            for (size_t t = 0; t < size; ++t) {
                float* dst = output + t * cfg_.d_model;
                std::memset(dst, 0, cfg_.d_model * sizeof(float));
                dst[0] = input[t];
            }
        }
        return;
    }
    if (cfg_.num_layers == 1) {
        layers_[0].process_block(input, output, size);
        return;
    }
    float* buf_dm = block_scratch_dm_.data();
    float* buf_mono = block_scratch_mono_.data();
    size_t offset = 0;
    while (offset < size) {
        size_t chunk = size - offset;
        if (chunk > MAX_BLOCK_SIZE) chunk = MAX_BLOCK_SIZE;

        const float* in_chunk = input + offset;
        float* out_chunk = output + offset * cfg_.d_model;

        layers_[0].process_block(in_chunk, buf_dm, chunk);
        for (size_t i = 1; i < cfg_.num_layers; ++i) {
            for (size_t t = 0; t < chunk; ++t) {
                float sum = 0.0f;
                for (size_t j = 0; j < cfg_.d_model; ++j) {
                    sum += buf_dm[t * cfg_.d_model + j];
                }
                buf_mono[t] = sum;
            }
            if (i == cfg_.num_layers - 1) {
                layers_[i].process_block(buf_mono, out_chunk, chunk);
            } else {
                layers_[i].process_block(buf_mono, buf_dm, chunk);
            }
        }
        offset += chunk;
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
        const size_t n = std::min(cfg_.cond_dim, cond_.size());
        std::memset(cond_.data(), 0, cond_.size() * sizeof(float));
        bool any_nonzero = false;
        for (size_t i = 0; i < n; ++i) {
            const float v = std::isfinite(cond_vector[i]) ? cond_vector[i] : 0.0f;
            cond_[i] = v;
            if (std::fabs(v) > 0.0f) any_nonzero = true;
        }
        cond_active_ = any_nonzero;
        for (size_t i = 0; i < cfg_.num_layers; ++i) {
            layers_[i].set_conditioning(any_nonzero ? cond_.data() : nullptr);
        }
    } else {
        std::memset(cond_.data(), 0, cond_.size() * sizeof(float));
        cond_active_ = false;
        for (size_t i = 0; i < cfg_.num_layers; ++i) {
            layers_[i].set_conditioning(nullptr);
        }
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
