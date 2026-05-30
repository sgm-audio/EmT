#ifndef SSM_INFERENCE_CORE_H
#define SSM_INFERENCE_CORE_H

#include "core/ssm_block.h"
#include "config.h"
#include <cstddef>
#include <array>
#include <new>

namespace mamba {

struct MambaConfig {
    size_t d_state;
    size_t d_model;
    size_t d_inner;
    size_t d_conv;
    size_t num_layers;
    size_t cond_dim;
};

struct alignas(64) MambaState {
    SSMState layers[MAX_LAYERS];
};

class SSMInferenceCore {
public:
    SSMInferenceCore();
    explicit SSMInferenceCore(const MambaConfig& cfg);
    ~SSMInferenceCore();

    SSMInferenceCore(const SSMInferenceCore&) = delete;
    SSMInferenceCore& operator=(const SSMInferenceCore&) = delete;

    void init(const MambaConfig& cfg);

    bool is_initialized() const { return layers_ != nullptr; }

    void reset();

    float process_sample(float input);
    void process_block(const float* input, float* output, size_t size);

    void set_layer_weights(size_t layer, const SSMWeights& w);
    void set_layer_params(size_t layer, const SSMBlockParams& params);
    void set_conditioning(const float* cond_vector);

    const MambaConfig& config() const { return cfg_; }

    SSMBlock& layer(size_t i) { return layers_[i]; }
    const SSMBlock& layer(size_t i) const { return layers_[i]; }

    void get_state(MambaState& out) const;
    void set_state(const MambaState& in);

private:
    MambaConfig cfg_;
    alignas(64) std::byte layers_storage_[sizeof(SSMBlock) * MAX_LAYERS];
    SSMBlock* layers_;
    size_t num_constructed_;
    alignas(64) std::array<float, COND_DIM> cond_;
};

} // namespace mamba

#endif
