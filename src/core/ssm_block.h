#ifndef SSM_BLOCK_H
#define SSM_BLOCK_H

#include "config.h"
#include <cstddef>
#include <cstdint>
#include <array>

namespace mamba {

// === Mamba-3 Complex Multi-Head State ===
//
// Per-head complex-valued SSM state (real + imag), RoPE angle state,
// and cached B projection.  alignas(64) prevents false sharing.
//
// DEVIATION D2 (open): the recurrence currently updates `ssm_real` only;
// `ssm_imag` is allocated and scaled but never written by a rotational state
// update.  Until D2 lands, describe the state as real with a data-dependent
// rotary rotation applied to B and C — never as a complex-valued state.
//
// Memory footprint at MAX config:
//   ssm_real/imag: 8 * 64 * 32 * 2 = 32,768 floats ≈ 131 KB
//   k_state:       4 * 8 * 32 = 1,024 floats ≈ 4 KB
//   angle_state:   8 * 16 = 128 floats ≈ 0.5 KB
//   Total: ~135 KB per layer

struct alignas(64) SSMState {
    float ssm_real[MAX_NHEADS][MAX_HEADDIM][MAX_D_STATE];
    float ssm_imag[MAX_NHEADS][MAX_HEADDIM][MAX_D_STATE];
    float k_state[MAX_MIMO_RANK][MAX_NHEADS][MAX_D_STATE];  // cached B
    float angle_state[MAX_NHEADS][MAX_NUM_ROPE_ANGLES];      // RoPE phase
};

// === Mamba-3 Weight Struct ===
//
// 13 weight arrays (down from 16 in Mamba-1). No conv1d arrays.
// All projections are stacked into a single in_proj matrix.
// FiLM conditioning replaces per-state cond_proj_dt/B.

struct SSMWeights {
    // --- Input projection (single fused matmul) ---
    // Layout: [in_proj_out, d_model]
    // in_proj_out = z(d_inner) + x(d_inner) + B(d_state*mimo_rank*nheads)
    //               + C(same) + dd_dt(nheads) + dd_A(nheads)
    //               + trap(nheads) + angles(nheads * num_rope_angles)
    static constexpr size_t IN_PROJ_OUT =
        MAX_D_INNER * 2
        + MAX_NHEADS * MAX_MIMO_RANK * MAX_D_STATE * 2
        + MAX_NHEADS * 3
        + MAX_NUM_ROPE_ANGLES;

    float in_proj_w[IN_PROJ_OUT][MAX_D_MODEL];

    // --- Δ bias (per-head): learned initial step size ---
    float dt_bias[MAX_NHEADS];

    // --- B and C biases (per-head, per-rank, per-state) ---
    float B_bias[MAX_NHEADS][MAX_MIMO_RANK][MAX_D_STATE];
    float C_bias[MAX_NHEADS][MAX_MIMO_RANK][MAX_D_STATE];

    // --- B and C RMSNorm weights (learned scale per state dim) ---
    float B_norm_weight[MAX_D_STATE];
    float C_norm_weight[MAX_D_STATE];

    // --- MIMO projection matrices (rank coupling per head) ---
    // mimo_x: V projection [nheads, mimo_rank, headdim]
    // mimo_z: Z projection [nheads, mimo_rank, headdim]
    // mimo_o: Output projection [nheads, mimo_rank, headdim]
    float mimo_x[MAX_NHEADS][MAX_MIMO_RANK][MAX_HEADDIM];
    float mimo_z[MAX_NHEADS][MAX_MIMO_RANK][MAX_HEADDIM];
    float mimo_o[MAX_NHEADS][MAX_MIMO_RANK][MAX_HEADDIM];

    // --- D skip connection (per-head scalar) ---
    float D_proj[MAX_NHEADS];

    // --- Output projection: d_inner → d_model ---
    float out_proj_w[MAX_D_MODEL][MAX_D_INNER];

    // --- FiLM conditioning projections ---
    // film_proj_w: maps COND_DIM → γ,β per d_model channel
    //   Layout: [2 * d_model, d_cond]
    //   Row 2*j produces γ for channel j; row 2*j+1 produces β
    float film_proj_w[2 * MAX_D_MODEL][COND_DIM];

    // dt_film_proj_w: maps COND_DIM → γ,β per head (modulates Δ)
    //   Layout: [2 * nheads, d_cond]
    float dt_film_proj_w[2 * MAX_NHEADS][COND_DIM];
};

// === Runtime parameters (gain, mix) ===
struct SSMBlockParams {
    float gain;
    float mix;
};

// === Single Mamba-3 SSM Layer ===
//
// No conv1d — 0 samples algorithmic latency.
// Multi-head SSM with a real (rank-1) state, bilinear/trapezoidal
// discretization, RoPE rotary embeddings on B/C, and FiLM conditioning.
//
// Claim policy (see ARCHITECTURE_FIDELITY.md):
//   D1 open — two-term update, not the paper's exponential-trapezoidal
//             three-term recurrence.
//   D2 open — no rotational (complex) dynamics inside the state.
//   D4 closed (documented) — A is input-derived, not a learned parameter.

class SSMBlock {
public:
    SSMBlock(size_t d_state, size_t d_model, size_t d_inner,
             size_t nheads, size_t headdim, size_t mimo_rank,
             size_t num_bc_heads, size_t d_cond);
    ~SSMBlock() = default;

    SSMBlock(const SSMBlock&) = delete;
    SSMBlock& operator=(const SSMBlock&) = delete;

    void reset();

    // Process one sample: input is scalar (from previous layer sum), output is d_model floats.
    void process_sample(float input, float* output);
    void process_sample(float input, float* output, SSMState& state);

    // Process a block of scalar samples.
    void process_block(const float* input, float* output, size_t size);
    void process_block(const float* input, float* output, size_t size, SSMState& state);

    void set_weights(const SSMWeights& w);
    void set_params(const SSMBlockParams& params);

    // Mamba-3 conditioning: FiLM feature-wise linear modulation.
    // cond_vector is d_cond_ floats.
    void set_conditioning(const float* cond_vector);

    static void apply_ftz_daz();

    // Accessors
    size_t get_d_state() const { return d_state_; }
    size_t get_d_model() const { return d_model_; }
    size_t get_nheads() const { return nheads_; }
    size_t get_headdim() const { return headdim_; }

    const SSMState& state() const { return state_; }
    SSMState& state() { return state_; }

private:
    // === Architecture dimensions ===
    size_t d_state_;
    size_t d_model_;
    size_t d_inner_;
    size_t nheads_;
    size_t headdim_;
    size_t mimo_rank_;
    size_t num_bc_heads_;
    size_t num_rope_angles_;
    size_t d_cond_;

    // === Persistent state (cache-aligned) ===
    SSMState state_;
    SSMWeights weights_;
    SSMBlockParams params_;

    // === Audio-thread scratch buffers (cache-aligned, no heap) ===
    static constexpr size_t IN_PROJ_DIM =
        MAX_D_INNER * 2
        + MAX_NHEADS * MAX_MIMO_RANK * MAX_D_STATE * 2
        + MAX_NHEADS * 3
        + MAX_NUM_ROPE_ANGLES;

    alignas(64) std::array<float, MAX_D_MODEL> u_film_;       // FiLM-modulated input
    alignas(64) std::array<float, IN_PROJ_DIM> in_proj_out_;  // in_proj output buffer
    alignas(64) std::array<float, MAX_NHEADS> DT_t_;          // per-head delta
    alignas(64) std::array<float, MAX_NHEADS> A_t_;           // per-head decay
    alignas(64) std::array<float, MAX_NHEADS * MAX_MIMO_RANK * MAX_D_STATE * 2> rope_buf_; // B/C RoPE
    alignas(64) std::array<float, MAX_NHEADS * MAX_HEADDIM> y_head_; // per-head output
    alignas(64) std::array<float, MAX_D_MODEL> out_;          // final output

    // Conditioning vector (FiLM input, d_cond_ floats)
    alignas(64) std::array<float, COND_DIM> cond_;

    bool conditioning_active_;
    bool weights_loaded_; 

    // === Decomposed execution steps (inlined in Release) ===
    void step_film_conditioning(float input);
    void step_in_proj(const float*& z_ptr, const float*& x_ptr,
                      const float*& B_ptr, const float*& C_ptr,
                      const float*& angle_ptr);
    void step_preprocess_decay_and_dt(const float* C_ptr);
    void step_rmsnorm_bc(const float* B_ptr, const float* C_ptr,
                         float* B_norm, float* C_norm);
    void step_rope_bc(SSMState& state, float* B_norm, float* C_norm, const float* angle_ptr);
    void step_recurrence(SSMState& state, const float* x_ptr,
                         const float* B_norm, const float* C_norm);
    void step_silu_gate(const float* z_ptr);
    void step_out_proj_residual(const float* x_ptr, float* output);
    // NOTE: there is no step_periodic_normalization().  The periodic state
    // renormalization was removed on 2026-10-04 (deviation D5): it was an
    // untrained non-linearity that stepped the output gain by 14.8 % every
    // SSM_STATE_NORM_INTERVAL samples.  Stability now comes from the
    // discretization itself (A_stab = (1 + A*dt/2) / (1 - A*dt/2), |A_stab| < 1
    // for the clamped A <= -1), which keeps the state bounded without mutating
    // it.  See ARCHITECTURE_FIDELITY.md D5.
};

} // namespace mamba

namespace emt = mamba;

#endif
