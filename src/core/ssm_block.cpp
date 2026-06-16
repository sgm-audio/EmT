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

void SSMBlock::process_sample(float input, float* output) {
    if (!weights_loaded_) {
        std::memset(output, 0, d_model_ * sizeof(float));
        return;
    }

    // ===== Step 0: FiLM conditioning on input =====
    // Broadcast scalar input to d_model channels, optionally modulated
    // by conditioning-dependent gamma and beta.
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

    // ===== Step 1: in_proj — single fused linear projection =====
    // u_film (d_model) → [z, x, B, C, dd_dt, dd_A, trap, angles]
    //   z:     [d_inner]
    //   x:     [d_inner]
    //   B:     [nheads * mimo_rank * d_state]
    //   C:     [nheads * mimo_rank * d_state]
    //   dd_dt: [nheads]
    //   dd_A:  [nheads]
    //   trap:  [nheads]
    //   angles:[num_rope_angles]
    size_t in_proj_out_dim = d_inner_ * 2
                           + nheads_ * mimo_rank_ * d_state_ * 2
                           + nheads_ * 3
                           + num_rope_angles_;
    for (size_t out_idx = 0; out_idx < in_proj_out_dim; ++out_idx) {
        in_proj_out_[out_idx] = simd::dot_product(
            weights_.in_proj_w[out_idx], u_film_.data(), d_model_);
    }

    // Split in_proj output into components
    const float* z_ptr     = in_proj_out_.data();
    const float* x_ptr     = z_ptr + d_inner_;
    const float* B_ptr     = x_ptr + d_inner_;
    size_t bc_size         = nheads_ * mimo_rank_ * d_state_;
    const float* C_ptr     = B_ptr + bc_size;
    const float* dd_dt_ptr = C_ptr + bc_size;
    const float* dd_A_ptr  = dd_dt_ptr + nheads_;
    const float* trap_ptr  = dd_A_ptr + nheads_;
    const float* angle_ptr = trap_ptr + nheads_;

    // ===== Step 2: Preprocess A and Δ (per head) =====
    // A_t[h] = clamp(-softplus(dd_A[h]), max=-A_FLOOR)
    // DT_t[h] = softplus(dd_dt[h] + dt_bias[h])
    static constexpr float A_FLOOR = 1.0f;  // limit decay: A_t ∈ (-∞, -1.0]
    for (size_t h = 0; h < nheads_; ++h) {
        float a_raw = dd_A_ptr[h];
        float A_t_val = -softplus(a_raw);
        // Clamp to maximum (least negative) decay
        if (A_t_val > -A_FLOOR) A_t_val = -A_FLOOR;

        float dt_raw = dd_dt_ptr[h];
        DT_t_[h] = softplus(dt_raw + weights_.dt_bias[h]);
        A_t_[h] = A_t_val;
    }

    // ===== Step 3: RMSNorm on B and C ====
    // B = B * rms_norm_scale + B_bias    where rms_norm_scale = B_norm_weight / rms(B)
    // C = C * rms_norm_scale + C_bias
    // We compute in-place into rope_buf_ (first half = B, second half = C)
    float* B_norm = rope_buf_.data();
    float* C_norm = B_norm + bc_size;

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
    // RMSNorm on C (same pattern)
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

    // ===== Step 4: RoPE on B and C (rotary position embedding) =====
    // Accumulate angles, apply rotation to pairs (s, s+num_rope_angles)
    for (size_t h = 0; h < nheads_; ++h) {
        for (size_t a = 0; a < num_rope_angles_; ++a) {
            // Update angle state
            float angle = state_.angle_state[h][a] + angle_ptr[a];
            angle = std::fmod(angle, 2.0f * static_cast<float>(M_PI));
            state_.angle_state[h][a] = angle;

            float cos_a = std::cos(angle);
            float sin_a = std::sin(angle);

            for (size_t g = 0; g < num_bc_heads_; ++g) {
                for (size_t r = 0; r < mimo_rank_; ++r) {
                    size_t base = (g * nheads_ + h) * mimo_rank_ * d_state_
                                + r * d_state_;

                    size_t s0 = a;
                    size_t s1 = a + num_rope_angles_;
                    if (s1 >= d_state_) continue;  // safety

                    // B rotation: (B0 + i·B1) * exp(i·angle)
                    float B0 = B_norm[base + s0];
                    float B1 = B_norm[base + s1];
                    B_norm[base + s0] = B0 * cos_a - B1 * sin_a;
                    B_norm[base + s1] = B0 * sin_a + B1 * cos_a;

                    // C rotation (conjugate): (C0 + i·C1) * exp(-i·angle)
                    float C0 = C_norm[base + s0];
                    float C1 = C_norm[base + s1];
                    C_norm[base + s0] = C0 * cos_a + C1 * sin_a;   // C0' = C0*cos + C1*sin
                    C_norm[base + s1] = -C0 * sin_a + C1 * cos_a;  // C1' = -C0*sin + C1*cos
                }
            }
        }
    }

    // ===== Step 5: Multi-head SSM recurrence (trapezoidal discretization) =====
    // Trapezoidal: A_bar = (1 + A*t/2) / (1 - A*t/2)
    //              B_bar_scale = t / (1 - A*t/2)
    // For each head h, head dimension d:
    //   h_{h,d}[s] = A_bar[h] * h_{h,d}[s] + B_bar_scale[h] * B[h,d,s] * x[h,d]

    // Zero y_head accumulation buffer
    std::memset(y_head_.data(), 0, nheads_ * headdim_ * sizeof(float));

    for (size_t h = 0; h < nheads_; ++h) {
        float A_decay = A_t_[h];
        float dt = DT_t_[h];

        // Trapezoidal discretization
        float half_a = A_decay * dt * 0.5f;
        float inv_denom = 1.0f / (1.0f - half_a);  // always > 0 for A<0
        float A_bar = (1.0f + half_a) * inv_denom;
        float B_bar_scale = dt * inv_denom;

        for (size_t d = 0; d < headdim_; ++d) {
            for (size_t s = 0; s < d_state_; ++s) {
                // x value for this head: x_ptr[h * headdim + d]
                float x_hd = x_ptr[h * headdim_ + d];

                // B value from normalized + RoPE'd B (use rank 0, group 0)
                size_t bc_idx = (0 * nheads_ + h) * mimo_rank_ * d_state_
                              + 0 * d_state_ + s;
                float B_val = B_norm[bc_idx];

                // C value (same indexing as B)
                float C_val = C_norm[bc_idx];

                // State update
                state_.ssm_real[h][d][s] = A_bar * state_.ssm_real[h][d][s]
                                         + B_bar_scale * B_val * x_hd;

                // Output contribution (real part only for now)
                y_head_[h * headdim_ + d] += C_val * state_.ssm_real[h][d][s];
            }
        }
    }

    // ===== Step 6: MIMO combine (z gating) ====
    // Apply SiLU gating: z_silu = x * sigmoid(z)
    // (mimo_z applied if mimo_rank > 1 — simplified to rank=1 for now)
    for (size_t h = 0; h < nheads_; ++h) {
        for (size_t d = 0; d < headdim_; ++d) {
            float z_val = z_ptr[h * headdim_ + d];
            // SiLU gating on z
            float z_sig = 1.0f / (1.0f + std::exp(-z_val));  // sigmoid
            float z_gated = z_val * z_sig;

            // Element-wise combine (simplified: z * y for rank=1)
            y_head_[h * headdim_ + d] *= z_gated;
        }
    }

    // ===== Step 7: Output projection + residual =====
    // out = out_proj(concat_head(y)) + residual (identity skip)
    for (size_t j = 0; j < d_model_; ++j) {
        float sum = simd::dot_product(weights_.out_proj_w[j], y_head_.data(), d_inner_);
        // D skip connection (per-head, broadcast to d_model)
        if (j < nheads_) {
            sum += weights_.D_proj[j] * x_ptr[j * headdim_];  // simplified skip
        }
        // Residual from input
        sum += u_film_[j];
        out_[j] = sum;
    }

    // Apply gain and copy to output
    simd::scale(output, out_.data(), params_.gain, d_model_);

    // ===== Step 8: Periodic state normalization =====
    sample_count_++;
    if (sample_count_ >= STATE_NORM_INTERVAL) {
        sample_count_ = 0;
        float max_h = 0.0f;
        for (size_t h = 0; h < nheads_; ++h) {
            for (size_t d = 0; d < headdim_; ++d) {
                for (size_t s = 0; s < d_state_; ++s) {
                    float v = std::fabs(state_.ssm_real[h][d][s]);
                    if (v > max_h) max_h = v;
                }
            }
        }
        if (max_h > 1.0f) {
            float scale = 1.0f / max_h;
            for (size_t h = 0; h < nheads_; ++h) {
                for (size_t d = 0; d < headdim_; ++d) {
                    for (size_t s = 0; s < d_state_; ++s) {
                        state_.ssm_real[h][d][s] *= scale;
                        state_.ssm_imag[h][d][s] *= scale;
                    }
                }
            }
        }
    }
}

// process_block: falls back to per-sample loop until a block-optimized
// version (with batched in_proj and multi-head parallelism) is implemented.
void SSMBlock::process_block(const float* input, float* output, size_t size) {
    if (!weights_loaded_) {
        std::memset(output, 0, size * d_model_ * sizeof(float));
        return;
    }

    for (size_t t = 0; t < size; ++t) {
        process_sample(input[t], output + t * d_model_);
    }
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
