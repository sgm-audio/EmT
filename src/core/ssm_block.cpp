#include "core/ssm_block.h"
#include "core/simd_helpers.h"
#include "config.h"
#include <cstring>
#include <cmath>
#include <algorithm>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

namespace {

static constexpr float PI_f = 3.14159265358979323846f;

void set_mxcsr_ftz_daz_impl() {
#if defined(_MSC_VER) && (defined(__x86_64__) || defined(_M_X64) || defined(__i386__))
    unsigned int mxcsr = _mm_getcsr();
    mxcsr |= 0x8040;
    _mm_setcsr(mxcsr);
#elif defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
    unsigned int mxcsr;
    __asm__ __volatile__ ("stmxcsr %0" : "=m"(mxcsr));
    mxcsr |= 0x8040;
    __asm__ __volatile__ ("ldmxcsr %0" : "=m"(mxcsr));
#elif defined(__aarch64__) || defined(_M_ARM64)
    // ARM: Flush-to-Zero (FZ) bit in FPCR, bit 24.
    // DAZ is not needed on ARM — FZ alone flushes all denormals.
    uint64_t fpcr;
    __asm__ __volatile__ ("mrs %0, fpcr" : "=r"(fpcr));
    fpcr |= (uint64_t{1} << 24);  // FZ: flush-to-zero
    __asm__ __volatile__ ("msr fpcr, %0" : : "r"(fpcr));
#endif
}

inline float softplus(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return 0.0f;
    return log1pf(expf(x));
}

} // namespace

namespace mamba {

SSMBlock::SSMBlock(size_t d_state, size_t d_model, size_t d_inner,
                   size_t nheads, size_t headdim, size_t mimo_rank,
                   size_t num_bc_heads, size_t d_cond)
    : d_state_(d_state), d_model_(d_model), d_inner_(d_inner),
      nheads_(nheads), headdim_(headdim), mimo_rank_(mimo_rank),
      num_bc_heads_(num_bc_heads), num_rope_angles_(d_state / 2),
      d_cond_(d_cond), sample_count_(0) {
    // Clamp to MAX limits
    if (d_state_ > MAX_D_STATE) d_state_ = MAX_D_STATE;
    if (d_model_ > MAX_D_MODEL) d_model_ = MAX_D_MODEL;
    if (d_inner_ > MAX_D_INNER) d_inner_ = MAX_D_INNER;
    if (nheads_ > MAX_NHEADS) nheads_ = MAX_NHEADS;
    if (headdim_ > MAX_HEADDIM) headdim_ = MAX_HEADDIM;
    if (mimo_rank_ > MAX_MIMO_RANK) mimo_rank_ = MAX_MIMO_RANK;
    if (num_bc_heads_ > MAX_NUM_BC_HEADS) num_bc_heads_ = MAX_NUM_BC_HEADS;
    if (num_rope_angles_ > MAX_NUM_ROPE_ANGLES) num_rope_angles_ = MAX_NUM_ROPE_ANGLES;
    if (d_cond_ > COND_DIM) d_cond_ = COND_DIM;

    reset();
    std::memset(&weights_, 0, sizeof(weights_));

    // Default: identity-like output projection
    for (size_t j = 0; j < d_model_ && j < MAX_D_MODEL; ++j) {
        if (j < d_inner_) weights_.out_proj_w[j][j] = 1.0f;
    }
    // Default: neutral RMSNorm (scale=1)
    for (size_t s = 0; s < d_state_ && s < MAX_D_STATE; ++s) {
        weights_.B_norm_weight[s] = 1.0f;
        weights_.C_norm_weight[s] = 1.0f;
    }
    // Default: neutral FiLM (zero gamma, zero beta)
    std::memset(weights_.film_proj_w, 0, sizeof(weights_.film_proj_w));
    std::memset(weights_.dt_film_proj_w, 0, sizeof(weights_.dt_film_proj_w));

    params_.gain = 1.0f;
    params_.mix = 1.0f;
    conditioning_active_ = false;
    weights_loaded_ = false;
}

void SSMBlock::reset() {
    std::memset(&state_, 0, sizeof(state_));
    std::memset(u_film_.data(), 0, u_film_.size() * sizeof(float));
    std::memset(in_proj_out_.data(), 0, in_proj_out_.size() * sizeof(float));
    std::memset(DT_t_.data(), 0, DT_t_.size() * sizeof(float));
    std::memset(A_t_.data(), 0, A_t_.size() * sizeof(float));
    std::memset(rope_buf_.data(), 0, rope_buf_.size() * sizeof(float));
    std::memset(y_head_.data(), 0, y_head_.size() * sizeof(float));
    std::memset(out_.data(), 0, out_.size() * sizeof(float));
    sample_count_ = 0;
}

// === Mamba-3 per-sample forward pass ===
//
// Eliminates Mamba-1's conv1d entirely. Uses:
//   - Exponential-trapezoidal discretization (2nd-order)
//   - Multi-head SSM with per-head decay (A) and step size (Δ)
//   - RoPE rotary position embedding on B and C
//   - RMSNorm on B and C
//   - FiLM conditioning (optional)
//   - MIMO rank coupling (simplified to rank=1 initially)

void SSMBlock::step_film_conditioning(float input) {
    if (conditioning_active_) {
        const float* cond_ptr = cond_.data();
        for (size_t j = 0; j < d_model_; ++j) {
            float gamma = simd::dot_product(weights_.film_proj_w[2 * j], cond_ptr, d_cond_);
            float beta  = simd::dot_product(weights_.film_proj_w[2 * j + 1], cond_ptr, d_cond_);
            u_film_[j] = input * (1.0f + gamma) + beta;
        }
    } else {
        for (size_t j = 0; j < d_model_; ++j) {
            u_film_[j] = input;
        }
    }
}

void SSMBlock::step_in_proj(const float*& z_ptr, const float*& x_ptr,
                            const float*& B_ptr, const float*& C_ptr,
                            const float*& angle_ptr) {
    size_t in_proj_out_dim = d_inner_ * 2
                           + nheads_ * mimo_rank_ * d_state_ * 2
                           + nheads_ * 3
                           + num_rope_angles_;
    for (size_t out_idx = 0; out_idx < in_proj_out_dim; ++out_idx) {
        in_proj_out_[out_idx] = simd::dot_product(
            weights_.in_proj_w[out_idx], u_film_.data(), d_model_);
    }

    z_ptr     = in_proj_out_.data();
    x_ptr     = z_ptr + d_inner_;
    B_ptr     = x_ptr + d_inner_;
    size_t bc_size = nheads_ * mimo_rank_ * d_state_;
    C_ptr     = B_ptr + bc_size;
    const float* dd_dt_ptr = C_ptr + bc_size;
    const float* dd_A_ptr  = dd_dt_ptr + nheads_;
    const float* trap_ptr  = dd_A_ptr + nheads_;
    angle_ptr = trap_ptr + nheads_;
}

void SSMBlock::step_preprocess_decay_and_dt(const float* C_ptr) {
    size_t bc_size = nheads_ * mimo_rank_ * d_state_;
    const float* dd_dt_ptr = C_ptr + bc_size;
    const float* dd_A_ptr  = dd_dt_ptr + nheads_;

    static constexpr float A_FLOOR = 1.0f;
    for (size_t h = 0; h < nheads_; ++h) {
        float a_raw = dd_A_ptr[h];
        float A_t_val = -softplus(a_raw);
        if (A_t_val > -A_FLOOR) A_t_val = -A_FLOOR;

        float dt_raw = dd_dt_ptr[h];
        DT_t_[h] = softplus(dt_raw + weights_.dt_bias[h]);
        A_t_[h] = A_t_val;
    }
}

void SSMBlock::step_rmsnorm_bc(const float* B_ptr, const float* C_ptr,
                               float* B_norm, float* C_norm) {
    // RMSNorm on B
    for (size_t g = 0; g < num_bc_heads_; ++g) {
        for (size_t r = 0; r < mimo_rank_; ++r) {
            for (size_t h = 0; h < nheads_; ++h) {
                float rms = 0.0f;
                for (size_t s = 0; s < d_state_; ++s) {
                    size_t idx = (g * nheads_ + h) * mimo_rank_ * d_state_
                               + r * d_state_ + s;
                    float val = B_ptr[idx];
                    rms += val * val;
                }
                rms = std::sqrt(rms / static_cast<float>(d_state_) + 1e-5f);
                float inv_rms = 1.0f / rms;
                for (size_t s = 0; s < d_state_; ++s) {
                    size_t idx = (g * nheads_ + h) * mimo_rank_ * d_state_
                               + r * d_state_ + s;
                    B_norm[idx] = B_ptr[idx] * inv_rms * weights_.B_norm_weight[s]
                                + weights_.B_bias[h][r][s];
                }
            }
        }
    }
    // RMSNorm on C
    for (size_t g = 0; g < num_bc_heads_; ++g) {
        for (size_t r = 0; r < mimo_rank_; ++r) {
            for (size_t h = 0; h < nheads_; ++h) {
                float rms = 0.0f;
                for (size_t s = 0; s < d_state_; ++s) {
                    size_t idx = (g * nheads_ + h) * mimo_rank_ * d_state_
                               + r * d_state_ + s;
                    float val = C_ptr[idx];
                    rms += val * val;
                }
                rms = std::sqrt(rms / static_cast<float>(d_state_) + 1e-5f);
                float inv_rms = 1.0f / rms;
                for (size_t s = 0; s < d_state_; ++s) {
                    size_t idx = (g * nheads_ + h) * mimo_rank_ * d_state_
                               + r * d_state_ + s;
                    C_norm[idx] = C_ptr[idx] * inv_rms * weights_.C_norm_weight[s]
                                + weights_.C_bias[h][r][s];
                }
            }
        }
    }
}

void SSMBlock::step_rope_bc(SSMState& state, float* B_norm, float* C_norm, const float* angle_ptr) {
    for (size_t h = 0; h < nheads_; ++h) {
        for (size_t a = 0; a < num_rope_angles_; ++a) {
            float angle = state.angle_state[h][a] + angle_ptr[a];
            angle = std::fmod(angle, 2.0f * PI_f);
            state.angle_state[h][a] = angle;

            float cos_a = std::cos(angle);
            float sin_a = std::sin(angle);

            for (size_t g = 0; g < num_bc_heads_; ++g) {
                for (size_t r = 0; r < mimo_rank_; ++r) {
                    size_t base = (g * nheads_ + h) * mimo_rank_ * d_state_
                                + r * d_state_;

                    size_t s0 = a;
                    size_t s1 = a + num_rope_angles_;
                    if (s1 >= d_state_) continue;

                    float B0 = B_norm[base + s0];
                    float B1 = B_norm[base + s1];
                    B_norm[base + s0] = B0 * cos_a - B1 * sin_a;
                    B_norm[base + s1] = B0 * sin_a + B1 * cos_a;

                    float C0 = C_norm[base + s0];
                    float C1 = C_norm[base + s1];
                    C_norm[base + s0] = C0 * cos_a + C1 * sin_a;
                    C_norm[base + s1] = -C0 * sin_a + C1 * cos_a;
                }
            }
        }
    }
}

void SSMBlock::step_recurrence(SSMState& state, const float* x_ptr,
                               const float* B_norm, const float* C_norm) {
    std::memset(y_head_.data(), 0, nheads_ * headdim_ * sizeof(float));

    for (size_t h = 0; h < nheads_; ++h) {
        float A_decay = A_t_[h];
        float dt = DT_t_[h];

        float half_a = A_decay * dt * 0.5f;
        float inv_denom = 1.0f / (1.0f - half_a);
        float A_bar = (1.0f + half_a) * inv_denom;
        float B_bar_scale = dt * inv_denom;

        for (size_t d = 0; d < headdim_; ++d) {
            float x_hd = x_ptr[h * headdim_ + d];
            for (size_t s = 0; s < d_state_; ++s) {
                size_t bc_idx = (0 * nheads_ + h) * mimo_rank_ * d_state_
                              + 0 * d_state_ + s;
                float B_val = B_norm[bc_idx];
                float C_val = C_norm[bc_idx];

                state.ssm_real[h][d][s] = A_bar * state.ssm_real[h][d][s]
                                         + B_bar_scale * B_val * x_hd;

                y_head_[h * headdim_ + d] += C_val * state.ssm_real[h][d][s];
            }
        }
    }
}

void SSMBlock::step_silu_gate(const float* z_ptr) {
    for (size_t h = 0; h < nheads_; ++h) {
        for (size_t d = 0; d < headdim_; ++d) {
            float z_val = z_ptr[h * headdim_ + d];
            float z_sig = 1.0f / (1.0f + std::exp(-z_val));
            float z_gated = z_val * z_sig;
            y_head_[h * headdim_ + d] *= z_gated;
        }
    }
}

void SSMBlock::step_out_proj_residual(const float* x_ptr, float* output) {
    for (size_t j = 0; j < d_model_; ++j) {
        float sum = simd::dot_product(weights_.out_proj_w[j], y_head_.data(), d_inner_);
        if (j < nheads_) {
            sum += weights_.D_proj[j] * x_ptr[j * headdim_];
        }
        sum += u_film_[j];
        out_[j] = sum;
    }
    simd::scale(output, out_.data(), params_.gain, d_model_);
}

void SSMBlock::step_periodic_normalization(SSMState& state) {
    sample_count_++;
    if (sample_count_ >= STATE_NORM_INTERVAL) {
        sample_count_ = 0;
        float max_h = 0.0f;
        for (size_t h = 0; h < nheads_; ++h) {
            for (size_t d = 0; d < headdim_; ++d) {
                for (size_t s = 0; s < d_state_; ++s) {
                    float v = std::fabs(state.ssm_real[h][d][s]);
                    if (v > max_h) max_h = v;
                }
            }
        }
        if (max_h > 1.0f) {
            float scale = 1.0f / max_h;
            for (size_t h = 0; h < nheads_; ++h) {
                for (size_t d = 0; d < headdim_; ++d) {
                    for (size_t s = 0; s < d_state_; ++s) {
                        state.ssm_real[h][d][s] *= scale;
                        state.ssm_imag[h][d][s] *= scale;
                    }
                }
            }
        }
    }
}

void SSMBlock::process_sample(float input, float* output, SSMState& state) {
    if (!weights_loaded_) {
        std::memset(output, 0, d_model_ * sizeof(float));
        return;
    }

    step_film_conditioning(input);

    const float* z_ptr = nullptr;
    const float* x_ptr = nullptr;
    const float* B_ptr = nullptr;
    const float* C_ptr = nullptr;
    const float* angle_ptr = nullptr;
    step_in_proj(z_ptr, x_ptr, B_ptr, C_ptr, angle_ptr);

    step_preprocess_decay_and_dt(C_ptr);

    const size_t bc_size = nheads_ * mimo_rank_ * d_state_;
    float* B_norm = rope_buf_.data();
    float* C_norm = B_norm + bc_size;
    step_rmsnorm_bc(B_ptr, C_ptr, B_norm, C_norm);

    step_rope_bc(state, B_norm, C_norm, angle_ptr);
    step_recurrence(state, x_ptr, B_norm, C_norm);
    step_silu_gate(z_ptr);
    step_out_proj_residual(x_ptr, output);
    step_periodic_normalization(state);
}

void SSMBlock::process_sample(float input, float* output) {
    process_sample(input, output, state_);
}

// process_block: falls back to per-sample loop until a block-optimized
// version (with batched in_proj and multi-head parallelism) is implemented.
void SSMBlock::process_block(const float* input, float* output, size_t size, SSMState& state) {
    if (!weights_loaded_) {
        std::memset(output, 0, size * d_model_ * sizeof(float));
        return;
    }

    for (size_t t = 0; t < size; ++t) {
        process_sample(input[t], output + t * d_model_, state);
    }
}

void SSMBlock::process_block(const float* input, float* output, size_t size) {
    process_block(input, output, size, state_);
}

void SSMBlock::set_weights(const SSMWeights& w) {
    // Row-by-row copy for in_proj_w: flat memcpy won't work because
    // SSMWeights.in_proj_w has stride MAX_D_MODEL, not d_model.
    size_t in_proj_rows = d_inner_ * 2 + nheads_ * mimo_rank_ * d_state_ * 2
                        + nheads_ * 3 + num_rope_angles_;
    for (size_t r = 0; r < in_proj_rows; ++r) {
        std::memcpy(weights_.in_proj_w[r], w.in_proj_w[r], d_model_ * sizeof(float));
    }

    // 1D arrays (flat memcpy is fine)
    std::memcpy(weights_.dt_bias,       w.dt_bias,        nheads_ * sizeof(float));
    std::memcpy(weights_.B_norm_weight, w.B_norm_weight,  d_state_ * sizeof(float));
    std::memcpy(weights_.C_norm_weight, w.C_norm_weight,  d_state_ * sizeof(float));
    std::memcpy(weights_.D_proj,        w.D_proj,         nheads_ * sizeof(float));

    // Multi-D arrays: per-(head,rank) copy to avoid MAX_* stride corruption
    for (size_t h = 0; h < nheads_; ++h) {
        for (size_t r = 0; r < mimo_rank_; ++r) {
            std::memcpy(weights_.B_bias[h][r],  w.B_bias[h][r],  d_state_ * sizeof(float));
            std::memcpy(weights_.C_bias[h][r],  w.C_bias[h][r],  d_state_ * sizeof(float));
            std::memcpy(weights_.mimo_x[h][r],  w.mimo_x[h][r],  headdim_ * sizeof(float));
            std::memcpy(weights_.mimo_z[h][r],  w.mimo_z[h][r],  headdim_ * sizeof(float));
            std::memcpy(weights_.mimo_o[h][r],  w.mimo_o[h][r],  headdim_ * sizeof(float));
        }
    }

    // Row-by-row for out_proj_w [MAX_D_MODEL, MAX_D_INNER]
    for (size_t r = 0; r < d_model_; ++r) {
        std::memcpy(weights_.out_proj_w[r], w.out_proj_w[r], d_inner_ * sizeof(float));
    }
    // Row-by-row for film_proj_w [2*MAX_D_MODEL, COND_DIM]
    for (size_t r = 0; r < 2 * d_model_; ++r) {
        std::memcpy(weights_.film_proj_w[r], w.film_proj_w[r], d_cond_ * sizeof(float));
    }
    // dt_film_proj_w: flat copy works (inner dim is COND_DIM, which matches)
    std::memcpy(weights_.dt_film_proj_w, w.dt_film_proj_w, 2 * nheads_ * d_cond_ * sizeof(float));

    weights_loaded_ = true;
}

void SSMBlock::set_params(const SSMBlockParams& params) {
    params_ = params;
}

// FiLM conditioning: stores the conditioning vector and toggles
// conditioning_active_ if any value is non-zero.
void SSMBlock::set_conditioning(const float* cond_vector) {
    if (cond_vector) {
        size_t n = std::min(d_cond_, cond_.size());
        std::memcpy(cond_.data(), cond_vector, n * sizeof(float));
        bool any_nonzero = false;
        for (size_t i = 0; i < n; ++i) {
            if (cond_[i] != 0.0f) { any_nonzero = true; break; }
        }
        conditioning_active_ = any_nonzero;
    } else {
        std::memset(cond_.data(), 0, cond_.size() * sizeof(float));
        conditioning_active_ = false;
    }
}

void SSMBlock::apply_ftz_daz() {
    set_mxcsr_ftz_daz_impl();
}

} // namespace mamba
