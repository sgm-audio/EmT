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
    if (cfg_.num_layers > MAX_LAYERS) cfg_.num_layers = MAX_LAYERS;
    if (cfg_.d_state > MAX_D_STATE) cfg_.d_state = MAX_D_STATE;
    if (cfg_.d_model > MAX_D_MODEL) cfg_.d_model = MAX_D_MODEL;
    if (cfg_.d_inner > MAX_D_INNER) cfg_.d_inner = MAX_D_INNER;
    if (cfg_.d_conv > MAX_D_CONV) cfg_.d_conv = MAX_D_CONV;
    if (cfg_.cond_dim > COND_DIM) cfg_.cond_dim = COND_DIM;

    layers_ = reinterpret_cast<SSMBlock*>(layers_storage_);
    for (size_t i = 0; i < cfg_.num_layers; ++i) {
        new (&layers_[i]) SSMBlock(cfg_.d_state, cfg_.d_model, cfg_.d_inner, cfg_.d_conv);
        num_constructed_++;
    }

    std::memset(cond_.data(), 0, cond_.size() * sizeof(float));
}

SSMInferenceCore::~SSMInferenceCore() {
    for (size_t i = 0; i < num_constructed_;++i) {
        layers_[i].~SSMBlock();
    }
}

void SSMInferenceCore::reset() {
    for (size_t i = 0; i < cfg_.num_layers; ++i) {
        layers_[i].reset();
    }
}

float SSMInferenceCore::process_sample(float input) {
    float x = input;
    for (size_t i = 0; i < cfg_.num_layers; ++i) {
        x = layers_[i].process_sample(x);
    }
    return x;
}

void SSMInferenceCore::process_block(const float* input, float* output, size_t size) {
    for (size_t n = 0; n < size; ++n) {
        output[n] = process_sample(input[n]);
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
