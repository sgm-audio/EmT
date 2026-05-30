#include "core/ssm_block.h"
#include "config.h"
#include <cstring>
#include <cmath>
#include <algorithm>

namespace {

void set_mxcsr_ftz_daz_impl() {
#if defined(__x86_64__) || defined(_M_X64)
    unsigned int mxcsr;
    __asm__ __volatile__ ("stmxcsr %0" : "=m"(mxcsr));
    mxcsr |= 0x8040;
    __asm__ __volatile__ ("ldmxcsr %0" : : "m"(mxcsr));
#endif
}

inline float softplus(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return 0.0f;
    return log1pf(expf(x));
}

inline float sigmoid(float x) {
    if (x >= 0.0f) {
        float z = expf(-x);
        return 1.0f / (1.0f + z);
    } else {
        float z = expf(x);
        return z / (1.0f + z);
    }
}

inline float silu(float x) {
    return x * sigmoid(x);
}

} // namespace

namespace mamba {

SSMBlock::SSMBlock(size_t d_state, size_t d_model, size_t d_inner, size_t d_conv)
    : d_state_(d_state), d_model_(d_model), d_inner_(d_inner), d_conv_(d_conv),
    conv_pos_(0), sample_count_(0) {
    reset();
    std::memset(&weights_, 0, sizeof(weights_));
    for (size_t i = 0; i < d_state_ && i < MAX_D_STATE; ++i) {
        weights_.A_log[i] = 0.0f;
        weights_.D_proj[i] = 0.0f;
        weights_.dt_bias[i] = 0.0f;
    }
    for (size_t j = 0; j < d_model_ && j < MAX_D_MODEL; ++j) {
        for (size_t k = 0; k < d_state_ && k < MAX_D_STATE; ++k) {
            if (j == k) weights_.W_ssm[j][k] = 1.0f;
        }
    }
    params_.gain = 1.0f;
    params_.mix = 1.0f;
    std::memset(cond_.data(), 0, cond_.size() * sizeof(float));
    weights_loaded_ = false;
}

void SSMBlock::reset() {
    std::memset(&state_, 0, sizeof(state_));
    std::memset(x_inner_.data(), 0, x_inner_.size() * sizeof(float));
    std::memset(conv_out_.data(), 0, conv_out_.size() * sizeof(float));
    std::memset(dt_.data(), 0, dt_.size() * sizeof(float));
    std::memset(B_.data(), 0, B_.size() * sizeof(float));
    std::memset(C_.data(), 0, C_.size() * sizeof(float));
    std::memset(ss_.data(), 0, ss_.size() * sizeof(float));
    std::memset(y_out_.data(), 0, y_out_.size() * sizeof(float));
    conv_pos_ = 0;
    sample_count_ = 0;
}

float SSMBlock::process_sample(float input) {
    if (!weights_loaded_) return 0.0f;

    // === Step 1: Input projection ===
    // x_inner = W_in @ x + b_in
    for (size_t j = 0; j < d_inner_; ++j) {
        float sum = weights_.b_in[j];
        for (size_t k = 0; k < d_model_; ++k) {
            sum += weights_.W_in[j][k] * input;
        }
        x_inner_[j] = sum;
    }

    // === Step 2: Causal conv1d ===
    // Shift conv_state right, insert new x_inner at position 0
    for (size_t j = 0; j < d_inner_; ++j) {
        for (size_t k = d_conv_ - 1; k > 0; --k) {
            state_.conv_state[k * d_inner_ + j] = state_.conv_state[(k - 1) * d_inner_ + j];
        }
        state_.conv_state[j] = x_inner_[j];
    }

    // Convolution: conv_out = conv1d_w * conv_state + conv1d_b, then SiLU
    for (size_t j = 0; j < d_inner_; ++j) {
        float sum = weights_.conv1d_b[j];
        for (size_t k = 0; k < d_conv_; ++k) {
            sum += weights_.conv1d_w[k][j] * state_.conv_state[k * d_inner_ + j];
        }
        conv_out_[j] = silu(sum);
    }

    // === Step 3: Compute input-dependent dt, B, C ===
    // dt = softplus(dt_proj_w @ conv_out + dt_bias)
    // B = B_proj_w @ conv_out
    // C = C_proj_w @ conv_out
    for (size_t i = 0; i < d_state_; ++i) {
        float dt_raw = weights_.dt_bias[i];
        float b_raw = 0.0f;
        float c_raw = 0.0f;
        for (size_t j = 0; j < d_inner_; ++j) {
            dt_raw += weights_.dt_proj_w[i][j] * conv_out_[j];
            b_raw += weights_.B_proj_w[j][i] * conv_out_[j];
            c_raw += weights_.C_proj_w[j][i] * conv_out_[j];
        }
        dt_[i] = softplus(dt_raw);
        B_[i] = b_raw;
        C_[i] = c_raw;
    }

    // === Step 4: Conditioning modulation (learned projection) ===
    // Uses dedicated cond_proj weights (d_state, COND_DIM) per ADR-003
    if (conditioning_active_) {
        for (size_t i = 0; i < d_state_; ++i) {
            float dt_cond = 0.0f;
            float b_cond = 0.0f;
            for (size_t c = 0; c < COND_DIM; ++c) {
                float cv = cond_[c];
                dt_cond += weights_.cond_proj_dt[i][c] * cv;
                b_cond += weights_.cond_proj_B[i][c] * cv;
            }
            dt_[i] = softplus(dt_[i] + dt_cond);
            B_[i] += b_cond;
        }
    }

    // === Step 5: Selective scan with proper Mamba discretization ===
    // ZOH discretization: h_t = A_bar * h_{t-1} + B_bar * u_t
    // where A_bar = exp(A*delta), B_bar = (exp(A*delta)-1)/A * B
    // B_bar already incorporates the B projection — do not re-multiply.
    for (size_t i = 0; i < d_state_; ++i) {
        float A = -expf(weights_.A_log[i]);
        float delta = dt_[i];

        float A_bar = expf(A * delta);

        float B_bar;
        if (fabsf(A) > 1e-6f) {
            B_bar = (A_bar - 1.0f) / A * B_[i];
        } else {
            B_bar = delta * B_[i];
        }

        state_.h[i] = A_bar * state_.h[i] + B_bar * conv_out_[i % d_inner_];
        ss_[i] = C_[i] * state_.h[i];
    }

    // === Step 6: Output projection + D skip ===
    // SSM path: W_ssm maps d_state -> d_model (no zero-padding)
    // Conv/D-skip path: W_out maps d_inner -> d_model
    float d_skip_total = 0.0f;
    for (size_t k = 0; k < d_inner_; ++k) {
        d_skip_total += weights_.D_proj[k % d_state_] * conv_out_[k];
    }

    for (size_t j = 0; j < d_model_; ++j) {
        float ssm_sum = 0.0f;
        for (size_t k = 0; k < d_state_; ++k) {
            ssm_sum += weights_.W_ssm[j][k] * ss_[k];
        }
        float conv_sum = 0.0f;
        for (size_t k = 0; k < d_inner_; ++k) {
            conv_sum += weights_.W_out[j][k] * conv_out_[k];
        }
        y_out_[j] = ssm_sum + conv_sum;
    }

    float y = d_skip_total;
    for (size_t j = 0; j < d_model_; ++j) {
        y += y_out_[j];
    }

    y *= params_.gain;

    // === Step 7: Periodic state normalization ===
    sample_count_++;
    if (sample_count_ >= STATE_NORM_INTERVAL) {
        sample_count_ = 0;
        float max_h = 0.0f;
        for (size_t i = 0; i < d_state_; ++i) {
            float v = fabsf(state_.h[i]);
            if (v > max_h) max_h = v;
        }
        if (max_h > 1.0f) {
            float scale = 1.0f / max_h;
            for (size_t i = 0; i < d_state_; ++i) {
                state_.h[i] *= scale;
            }
        }
    }

    return y;
}

void SSMBlock::process_block(const float* input, float* output, size_t size) {
    for (size_t n = 0; n < size; ++n) {
        output[n] = process_sample(input[n]);
    }
}

void SSMBlock::set_weights(const SSMWeights& w) {
    std::memcpy(&weights_, &w, sizeof(weights_));
    weights_loaded_ = true;
}

void SSMBlock::set_params(const SSMBlockParams& params) {
    params_ = params;
}

void SSMBlock::set_conditioning(const float* cond_vector) {
    if (cond_vector) {
        size_t n = std::min(static_cast<size_t>(COND_DIM), cond_.size());
        std::memcpy(cond_.data(), cond_vector, n * sizeof(float));
        bool any_nonzero = false;
        for (size_t i = 0; i < n; ++i) {
            if (cond_[i] != 0.0f) { any_nonzero = true; break; }
        }
        conditioning_active_ = any_nonzero;
    }
}

void SSMBlock::apply_ftz_daz() {
    set_mxcsr_ftz_daz_impl();
}

}