#ifndef SSM_BLOCK_H
#define SSM_BLOCK_H

#include "config.h"
#include <cstddef>
#include <cstdint>
#include <array>

namespace mamba {

struct alignas(64) SSMState {
    float h[MAX_D_STATE];
    float conv_state[MAX_D_CONV * MAX_D_INNER];
};

struct SSMWeights {
    float A_log[MAX_D_STATE];
    float D_proj[MAX_D_STATE];
    float dt_bias[MAX_D_STATE];
    float dt_proj_w[MAX_D_STATE][MAX_D_INNER];
    float B_proj_w[MAX_D_INNER][MAX_D_STATE];
    float C_proj_w[MAX_D_INNER][MAX_D_STATE];
    float conv1d_w[MAX_D_CONV][MAX_D_INNER];
    float conv1d_b[MAX_D_INNER];
    float W_in[MAX_D_INNER][MAX_D_MODEL];
    float W_out[MAX_D_MODEL][MAX_D_INNER];
    float W_ssm[MAX_D_MODEL][MAX_D_STATE];
    float b_in[MAX_D_INNER];
    float cond_proj_dt[MAX_D_STATE][COND_DIM];
    float cond_proj_B[MAX_D_STATE][COND_DIM];
};

struct SSMBlockParams {
    float gain;
    float mix;
};

class SSMBlock {
public:
    SSMBlock(size_t d_state, size_t d_model, size_t d_inner, size_t d_conv);
    ~SSMBlock() = default;

    SSMBlock(const SSMBlock&) = delete;
    SSMBlock& operator=(const SSMBlock&) = delete;

    void reset();

    float process_sample(float input);

    void process_block(const float* input, float* output, size_t size);

    void set_weights(const SSMWeights& w);

    void set_params(const SSMBlockParams& params);

    void set_conditioning(const float* cond_vector);

    static void apply_ftz_daz();

    size_t get_d_state() const { return d_state_; }
    size_t get_d_model() const { return d_model_; }

    const SSMState& state() const { return state_; }
    SSMState& state() { return state_; }

private:
    size_t d_state_;
    size_t d_model_;
    size_t d_inner_;
    size_t d_conv_;

    SSMState state_;
    SSMWeights weights_;
    SSMBlockParams params_;

    alignas(64) std::array<float, MAX_D_INNER> x_inner_;
    alignas(64) std::array<float, MAX_D_INNER> conv_out_;
    alignas(64) std::array<float, MAX_D_STATE> dt_;
    alignas(64) std::array<float, MAX_D_STATE> B_;
    alignas(64) std::array<float, MAX_D_STATE> C_;
    alignas(64) std::array<float, MAX_D_STATE> ss_;
    alignas(64) std::array<float, MAX_D_MODEL> y_out_;

    size_t conv_pos_;

    alignas(64) std::array<float, COND_DIM> cond_;
    bool conditioning_active_;
    bool weights_loaded_;

    static constexpr uint32_t STATE_NORM_INTERVAL = 4096;
    uint32_t sample_count_;
};

} // namespace mamba

#endif
