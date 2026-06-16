#ifndef SSM_INFERENCE_CORE_H
#define SSM_INFERENCE_CORE_H

#include "core/ssm_block.h"
#include "config.h"
#include <cstddef>
#include <array>
#include <new>

namespace mamba {

// Mamba-3 configuration: no d_conv, adds multi-head / MIMO / RoPE params
struct MambaConfig {
    size_t d_state;
    size_t d_model;
    size_t d_inner;
    size_t nheads;
    size_t headdim;
    size_t mimo_rank;
    size_t num_bc_heads;
    size_t num_rope_angles;
    size_t num_layers;
    size_t cond_dim;

    // Convenience defaults for MAX config
    static MambaConfig max_config() {
        return {
            MAX_D_STATE, MAX_D_MODEL, MAX_D_INNER,
            MAX_NHEADS, MAX_HEADDIM, MAX_MIMO_RANK,
            MAX_NUM_BC_HEADS, MAX_NUM_ROPE_ANGLES,
            1, COND_DIM
        };
    }
};

// Multi-layer SSM state snapshot
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

    void process_sample(float input, float* output);
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
