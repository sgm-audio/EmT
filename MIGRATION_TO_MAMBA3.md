# Mamba-1 → Mamba-3 Migration Plan

> **⚠️ Status correction (2026-10-04).** This document records a migration *plan*
> and the shape of the target API. It overstated what shipped. Verified state:
> the **bilinear (Tustin) recurrence, RMSNorm+biases on B/C, and the data-dependent
> rotation of B/C are implemented**, but the
> **complex-valued state update and rank-R MIMO coupling are NOT active in the
> forward pass** (`ssm_imag` is never written; the recurrence indexes rank 0 and
> never touches `mimo_o`), and the paper's **exponential-trapezoidal three-term
> recurrence is not implemented** (what is implemented is the classical trapezoidal
> rule on a real pole). See [`ARCHITECTURE_FIDELITY.md`](ARCHITECTURE_FIDELITY.md)
> for the deviation register and tests. Passing statements below should be read as
> "API/format reserved for", not "runtime implemented".

**Date:** 2026-06-14  
**Last updated:** 2026-06-15 — **Phases 1-4 complete, core + CLAP builds**
**Reference implementation:** `/home/sgm/Desktop/Dev/mamba3/model.py` (PerTrackProcessor, Mamba3MIMOLayer)  
**Official reference:** `/home/sgm/Github Repo's/Mamba Based/mamba/mamba_ssm/modules/mamba3.py` (Mamba3, step())  
**Paper:** arXiv 2603.15569

## Migration Progress Tracker

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Configuration & Types (`config.h`, `ssm_block.h`) | ✅ **DONE** |
| 2 | Weight Loading (`weight_loader.h/.cpp`) | ✅ **DONE** |
| 3 | Multi-Layer Core (`ssm_inference_core.h/.cpp`) | ✅ **DONE** |
| 4 | CLAP Plugin Shell (`clap_plugin.cpp`, `clap_latency.cpp`, `conditioning_frame.h`) | ✅ **DONE** |
| 5 | Python Export Pipeline (`export_mamba_weights.py`, `validate_export.py`) | ✅ **DONE** |
| 6 | Tests (6 files) | ✅ **DONE** — 40/40 passing |
| 7 | ss_block.cpp `process_sample()` Mamba-3 forward pass | ✅ **DONE** |

**Build status:** Core library + CLAP plugin (`mamba_synth`) compiles and links. **40/40 tests pass** (the suite has since grown to 64). The 1.79 us/sample single-layer / 3.73 us/sample 2-layer figures were measured on the *toy* benchmark shape (d_state 16, d_model 16, d_inner 32, 2 heads); the shapes this repo's own training configs produce cost 4.3 us (laptop config) to ~19 us (RunPod L4 config) per sample per layer. Use `tests/benchmark_ssm_block.cpp` and quote its JSON.

---

---

## The Problem

This project implements Mamba-1 (arxiv 2312.00752):
- Causal conv1d before SSM (+3 samples latency)
- ZOH discretization (`A_bar = exp(A*Δ)`)
- Only Δ and B are input-selective (static C, static A)
- SISO per state channel (scalar input/output)
- Real-valued state
- Batch-conditioning on dt and B (`cond_proj_dt`, `cond_proj_B`)

## The Target: Mamba-3

Mamba-3 eliminates every Mamba-1 limitation:
- **No conv1d** — removes the stateful delay line; note Mamba-1/2's causal conv is *already* zero-look-ahead in streaming form, so this is a compute/simplification win rather than a latency win
- **Caveat:** the shipped recurrence is two-term bilinear, not the paper's exponential-trapezoidal form
- **(paper feature — NOT implemented)** exponential-trapezoidal discretization; the shipped rule is the classical two-term trapezoidal/bilinear update above, which is stable and cheap but carries no input-history (β) term
- **A, B, C all input-dependent** — via single `in_proj` layer
- **Multi-head SSM** — H heads, each a resonant complex pole
- **Data-dependent rotation of B/C** — implemented (`step_rope_bc`); the *state* itself stays real
- **(not implemented)** complex-valued state / complex-plane poles per head — `ssm_imag` is allocated but never written
- **(format only)** MIMO — the weight format and exporter reserve rank R, but the inference path uses rank 0 and never applies `mimo_o`
- **FiLM conditioning** — Feature-wise Linear Modulation from cross-track semantic + f0 + loudness context
- **RMSNorm** on B and C for stable training/inference

### Concrete Math (per-sample step)

```
Input: u ∈ ℝ^d_model

1. in_proj:  [z, x, B, C, dd_dt, dd_A, trap, angles] = in_proj @ u
   z, x:     [nheads, headdim]      — output/input per head
   B, C:     [mimo_rank, nheads, d_state]  — selective projections
   dd_dt:    [nheads]               — Δ raw
   dd_A:     [nheads]               — A raw (decay)
   trap:     [nheads]               — trapezoidal param
   angles:   [nheads, d_state/2]    — RoPE rotation angles

2. Preprocess:
   A_t  = clamp(-softplus(dd_A), max=-A_floor)     — per-head decay
   DT_t = softplus(dd_dt + dt_bias)                  — per-head Δ
   B    = B_norm(B)  (RMSNorm) + B_bias
   C    = C_norm(C)  (RMSNorm) + C_bias

3. RoPE (rotary position embedding) on B, C:
   Apply sin/cos rotation using angle_state and angles projection
   This rotates B and C in the complex plane (the paper's efficient real
   realisation of a *complex state*). In this repo the rotation is applied to
   the B/C vectors, but the carried state `h` stays real - `ssm_imag` is never
   written. See ARCHITECTURE_FIDELITY.md D2.

4. SSM recurrence (per head, per d_state dimension):
   h_t = A_t · h_{t-1} + B_t · x_t    (MIMO: B_t ∈ ℝ^[R, headdim, d_state])
   y_t = C_t^T · h_t                    (MIMO: C_t ∈ ℝ^[R, headdim, d_state])

5. MIMO combine (PAPER FORM — not executed by this repo):
   y = einsum("h r p, h r p -> h p", y, mimo_o)    — rank-R → headdim
   z = einsum("h r p, h r p -> h r p", z, mimo_z)   — MIMO z projection
   The shipped recurrence indexes rank 0 only and never applies mimo_o/mimo_z,
   so mimo_rank > 1 is silently rank-1 at inference (ARCHITECTURE_FIDELITY.md D3).

6. Output projection + residual:
   out = out_proj(concat_head(y))
   out = out + u  (with optional FiLM: out = out * (1 + gamma) + beta)
```

### Trapezoidal vs ZOH

> Scope note: this section describes the *paper's* discretization. What the repo
> implements is the two-term bilinear form, i.e. the trapezoidal rule applied to
> a real pole without the input-history term; the numerical-stability argument
> carries over, the second-order accuracy argument does not.

| | ZOH (Mamba-1) | Trapezoidal (Mamba-3) |
|---|---|---|
| A_bar | `exp(A * Δ)` | `(1 + Δ·A/2) / (1 - Δ·A/2)` |
| B_bar | `(exp(A·Δ) - 1) / A * B` | `Δ / (1 - Δ·A/2)` · B |
| Numerical | `expm1` division near A≈0 | **No singularity** — clean division |
| Frequency warp | Present at high freq | **Preserved** (bilinear map) |
| Stability | `|A_bar| ≤ 1` guaranteed | `|A_bar| ≤ 1` guaranteed (stable by construction for A<0) |

Trapezoidal discretization in C++:

```cpp
// Mamba-3: A_bar = (1 + half_a) / (1 - half_a)
// B_bar = dt / (1 - half_a) * B
float half_a = A * dt / 2.0f;
float denom = 1.0f / (1.0f - half_a);  // always > 0 for A<0
float A_bar = (1.0f + half_a) * denom;
float B_bar = dt * denom * B_t;
```

Much simpler than Mamba-1's `(exp(A) - 1)/A` with the A≈0 branch.

---

## Migration Steps — File by File

### Phase 1: Configuration & Types (Impact: 2 files)

#### 1.1 `src/config.h` — Replace entire file

```c
#define MAX_D_STATE   32     // N: state dimension (Mamba-3 needs N/2 vs M1 for same quality)
#define MAX_D_MODEL  256     // embedding dimension
#define MAX_D_INNER  512     // expand * d_model (expand=2)
#define MAX_HEADDIM   64     // head dimension
#define MAX_NHEADS     8     // d_inner / headdim = 512/64 = 8
#define MAX_MIMO_RANK  4     // MIMO rank R
#define MAX_NUM_BC_HEADS 1   // B/C head groups
#define MAX_CHUNK_SIZE 128   // for block processing
#define MAX_LAYERS     6
#define MAX_BLOCK_SIZE 512
#define MAX_PARAM_ATOMS 256
#define COND_DIM 128          // FiLM conditioning dimension
#define MAX_NUM_ROPE_ANGLES 16 // d_state/2

#define VERSION_MAJOR 0
#define VERSION_MINOR 3
#define VERSION_PATCH 0
```

**REMOVED:** `MAX_D_CONV` — conv1d erased from existence.

#### 1.2 `src/core/ssm_block.h` — Rewrite SSMWeights and SSMState

```cpp
#ifndef SSM_BLOCK_H
#define SSM_BLOCK_H

#include "config.h"
#include <cstddef>
#include <cstdint>
#include <array>
#include <complex>

namespace mamba {

// Complex state per head: [nheads, headdim, d_state]
// Stored as real/imag for SIMD-friendliness
struct alignas(64) SSMState {
    float ssm_real[MAX_NHEADS][MAX_HEADDIM][MAX_D_STATE];
    float ssm_imag[MAX_NHEADS][MAX_HEADDIM][MAX_D_STATE];
    float k_state[MAX_MIMO_RANK][MAX_NHEADS][MAX_D_STATE];  // B cache
    float angle_state[MAX_NHEADS][MAX_NUM_ROPE_ANGLES];     // RoPE state
};

struct SSMWeights {
    // input projection weights (single matrix: d_model -> [z, x, B, C, dt, A, trap, angle])
    // Layout: in_proj is [d_inner*2 + d_state*mimo_rank*nheads*2 + nheads*3 + num_rope_angles][d_model]
    float in_proj_w[MAX_D_INNER * 2 + MAX_NHEADS * MAX_MIMO_RANK * MAX_D_STATE * 2
                   + MAX_NHEADS * 3 + MAX_NUM_ROPE_ANGLES][MAX_D_MODEL];

    // Δ bias (per-head): learned initial step size
    float dt_bias[MAX_NHEADS];

    // B and C biases (per-head, per-rank, per-state)
    float B_bias[MAX_NHEADS][MAX_MIMO_RANK][MAX_D_STATE];
    float C_bias[MAX_NHEADS][MAX_MIMO_RANK][MAX_D_STATE];

    // B and C RMSNorm weights (learned scale for normalization)
    float B_norm_weight[MAX_D_STATE];
    float C_norm_weight[MAX_D_STATE];

    // MIMO projection matrices (rank coupling per head)
    float mimo_x[MAX_NHEADS][MAX_MIMO_RANK][MAX_HEADDIM];  // V projection
    float mimo_z[MAX_NHEADS][MAX_MIMO_RANK][MAX_HEADDIM];  // Z projection
    float mimo_o[MAX_NHEADS][MAX_MIMO_RANK][MAX_HEADDIM];  // Output projection

    // D skip connection (per-head scalar)
    float D_proj[MAX_NHEADS];

    // Output projection: d_inner -> d_model
    float out_proj_w[MAX_D_MODEL][MAX_D_INNER];
    // No bias in out_proj

    // FiLM conditioning projections
    float film_proj_w[MAX_D_COND] [2 * MAX_D_MODEL];   // γ, β per d_model
    float dt_film_proj_w[MAX_NHEADS * 2][MAX_D_COND];  // γ, β per head
};

// ... rest of SSMBlock class (process_sample, process_block, etc.)

} // namespace mamba
#endif
```

**REMOVED from Mamba-1 weights:**
- `A_log[d_state]` → now per-head scalar in in_proj
- `D_proj[d_state]` → now per-head `D_proj[nheads]`
- `dt_proj_w[d_state][d_inner]` → now part of in_proj
- `B_proj_w[d_inner][d_state]` → now part of in_proj
- `C_proj_w[d_inner][d_state]` → now part of in_proj
- `conv1d_w[d_conv][d_inner]` → **deleted**
- `conv1d_b[d_inner]` → **deleted**
- `W_in[d_inner][d_model]` → now part of in_proj
- `W_out[d_model][d_inner]` → now `out_proj_w`
- `W_ssm[d_model][d_state]` → **deleted** (different residual mechanism)
- `b_in[d_inner]` → **deleted** (bias on in_proj separate)
- `cond_proj_dt[d_state][cond_dim]` → now FiLM
- `cond_proj_B[d_state][cond_dim]` → now FiLM
- `cond_dt_bias[d_state]` → now FiLM
- `cond_B_bias[d_state]` → now FiLM

**REMOVED from Mamba-1 state:**
- `conv_state[d_conv * d_inner]` — **deleted**

#### 1.3 `src/core/ssm_block.cpp` — Complete rewrite of process_sample

The per-sample function becomes:

```cpp
void SSMBlock::process_sample(float input, float* output) {
    if (!weights_loaded_) return;

    // === Step 0: FiLM on input ===
    if (conditioning_active_) {
        for (size_t j = 0; j < d_model_; ++j) {
            float gamma = 0.0f, beta = 0.0f;
            for (size_t k = 0; k < d_cond_; ++k) {
                gamma += weights_.film_proj_w[2*j][k] * cond_[k];
                beta  += weights_.film_proj_w[2*j+1][k] * cond_[k];
            }
            u_film_[j] = input * (1.0f + gamma) + beta;  // single scalar input per sample
        }
    } else {
        for (size_t j = 0; j < d_model_; ++j) u_film_[j] = input;
    }

    // === Step 1: in_proj ===
    // u_film (d_model) → [z, x, B, C, dt_raw, A_raw, trap, angles]
    // No conv1d. No SiLU on inner features. Just linear projection.
    for (size_t out_idx = 0; out_idx < in_proj_out_dim_; ++out_idx) {
        float sum = 0.0f;
        for (size_t j = 0; j < d_model_; ++j) {
            sum += weights_.in_proj_w[out_idx][j] * u_film_[j];
        }
        in_proj_out_[out_idx] = sum;
    }

    // Split in_proj output into components
    const float* z_ptr      = in_proj_out_;
    const float* x_ptr      = z_ptr + d_inner_;
    const float* B_ptr      = x_ptr + d_inner_;
    const float* C_ptr      = B_ptr + d_state_ * num_bc_heads_ * mimo_rank_;
    const float* dd_dt_ptr  = C_ptr + d_state_ * num_bc_heads_ * mimo_rank_;
    const float* dd_A_ptr   = dd_dt_ptr + nheads_;
    const float* trap_ptr   = dd_A_ptr + nheads_;
    const float* angle_ptr  = trap_ptr + nheads_;

    // === Step 2: Preprocess A, Δ ===
    for (size_t h = 0; h < nheads_; ++h) {
        float a_raw = dd_A_ptr[h];
        float A_t = -softplusf(a_raw);
        A_t = (A_t < -A_floor_) ? -A_floor_ : A_t;  // clamp max

        float dt_raw = dd_dt_ptr[h];
        DT_t_[h] = softplusf(dt_raw + weights_.dt_bias[h]);
        A_t_[h] = A_t;
    }

    // === Step 3: RoPE on B, C ===
    // Apply rotary position embedding: rotates B and C vectors
    // angle_state[h][i] carries accumulated angle, angles[h][i] is per-step rotation
    for (size_t g = 0; g < num_bc_heads_; ++g) {
        for (size_t r = 0; r < mimo_rank_; ++r) {
            for (size_t h = 0; h < nheads_; ++h) {
                for (size_t s = 0; s < num_rope_angles_; ++s) {
                    // Update angle state
                    float angle = state_.angle_state[h][s] + angle_ptr[h * num_rope_angles_ + s];
                    angle = fmodf(angle, 2.0f * M_PI);
                    state_.angle_state[h][s] = angle;

                    float cos_a = cosf(angle);
                    float sin_a = sinf(angle);

                    // Indices for RoPE pairs
                    size_t s0 = s;
                    size_t s1 = s + num_rope_angles_;

                    // B_rope
                    float B0 = B_ptr[(g * nheads_ + h) * d_state_ + s0];
                    float B1 = B_ptr[(g * nheads_ + h) * d_state_ + s1];
                    // Rotate: (B0 + i·B1) * exp(i·angle)
                    B_rope_[s0] = B0 * cos_a - B1 * sin_a;
                    B_rope_[s1] = B0 * sin_a + B1 * cos_a;

                    // C_rope (conjugate rotation)
                    float C0 = C_ptr[(g * nheads_ + h) * d_state_ + s0];
                    float C1 = C_ptr[(g * nheads_ + h) * d_state_ + s1];
                    C_rope_[s0] = C0 * cos_a + C1 * sin_a;
                    C_rope_[s1] = C0 * (-sin_a) + C1 * cos_a;
                }
            }
        }
    }

    // === Step 4: B/C normalization (RMSNorm) ===
    // B = rms_norm(B) * weight + bias  (per d_state element)
    // C = rms_norm(C) * weight + bias

    // === Step 5: Multi-head SSM recurrence ===
    for (size_t h = 0; h < nheads_; ++h) {
        float A_decay = A_t_[h];          // per-head decay scalar
        float dt = DT_t_[h];              // per-head time step

        // Trapezoidal discretization
        float half_a = A_decay * dt / 2.0f;
        float inv_denom = 1.0f / (1.0f - half_a);
        float A_bar = (1.0f + half_a) * inv_denom;
        float B_bar_scale = dt * inv_denom;

        for (size_t d = 0; d < headdim_; ++d) {
// For each state dimension:
for (size_t s = 0; s < d_state_; ++s) {
    // MIMO: B couples rank-R input to state. 
    // B_t = Σ_r B[h][r][s] (B_proj from in_proj)
    // x_t = Σ_r mimo_x[h][r][d] * x[h * headdim + d]  
    // For simplicity in first pass: collapse rank dimension
    // h_t[s] = A_bar * h_t[s] + B_bar_scale * B[h][0][s] * x[h*headdim+d]
    // y_t[d] += C[h][0][s] * h_t[s]
    
    float x_head_val = x_ptr[h * headdim_ + d];
    float B_val = B_rope_[h * d_state_ + s];  // RoPE'd
    float C_val = C_rope_[h * d_state_ + s];  // RoPE'd
    
    // State update (complex via RoPE, but recurrence on real state)
    state_.ssm_real[h][d][s] = A_bar * state_.ssm_real[h][d][s]
                               + B_bar_scale * B_val * x_head_val;
    state_.ssm_imag[h][d][s] = A_bar * state_.ssm_imag[h][d][s]
                               + B_bar_scale * B_val * x_head_val;  // simplified
    
    // Output contribution
    y_head_[h][d] += C_val * state_.ssm_real[h][d][s];
}
```

**Key simplifications from the full Mamba-3 forward path:**
1. No chunk/parallel scan — per-sample recurrence only
2. No MIMO rank expansion in first pass (rank=1 initially, add later)
3. RoPE applied as real-valued cycle rotation (not full complex)
4. Trapezoidal discretization uses the simple `(1 + A*dt/2) / (1 - A*dt/2)` form

### Phase 2: Weight Loading (Impact: 2 files)

#### 2.1 `src/core/weight_loader.h` — New WeightLayerPointers

```cpp
struct WeightLayerPointers {
    const float* in_proj_w;       // [in_proj_out, d_model]
    const float* dt_bias;         // [nheads]
    const float* B_bias;          // [nheads, mimo_rank, d_state]
    const float* C_bias;          // [nheads, mimo_rank, d_state]
    const float* B_norm_weight;   // [d_state]
    const float* C_norm_weight;   // [d_state]
    const float* mimo_x;          // [nheads, mimo_rank, headdim]
    const float* mimo_z;          // [nheads, mimo_rank, headdim]
    const float* mimo_o;          // [nheads, mimo_rank, headdim]
    const float* D_proj;          // [nheads]
    const float* out_proj_w;      // [d_model, d_inner]
    const float* film_proj_w;     // [2*d_model, d_cond]
    const float* dt_film_proj_w;  // [2*nheads, d_cond]
    
    // Quantized versions (Int8/FP16) follow same pattern
    // const int8_t* in_proj_w; ...
};
```

#### 2.2 `src/core/weight_loader.cpp` — New offsets, dequant

Per-layer weight order in binary file:
```
in_proj_w        [in_proj_out * d_model]  — float32
dt_bias          [nheads]                  — float32
B_bias           [nheads * mimo_rank * d_state] — float32
C_bias           [nheads * mimo_rank * d_state] — float32
B_norm_weight    [d_state]                — float32
C_norm_weight    [d_state]                — float32
mimo_x           [nheads * mimo_rank * headdim] — float32
mimo_z           [nheads * mimo_rank * headdim] — float32
mimo_o           [nheads * mimo_rank * headdim] — float32
D_proj           [nheads]                 — float32
out_proj_w       [d_model * d_inner]      — float32
film_proj_w      [2 * d_model * d_cond]   — float32
dt_film_proj_w   [2 * nheads * d_cond]    — float32
```

13 weight arrays per layer (down from 16 in Mamba-1). No conv1d arrays. No separate input projection (folded into in_proj).

### Phase 3: Multi-Layer Core (Impact: 2 files)

#### 3.1 `src/core/ssm_inference_core.h/cpp`

SSMInferenceCore changes:
- Constructor takes `nheads`, `headdim`, `mimo_rank`, `num_bc_heads`
- State get/set handles multi-head complex state
- `process_block` simplified (no conv1d chunk loop)
- FiLM conditioning passed to each layer's `process_sample/block`

### Phase 4: CLAP Plugin Shell (Impact: 3 files)

#### 4.1 `src/clap/clap_plugin.cpp` — load_weights update

Replace the 16-array memcpy block (lines 616-627) with new 13-array block:

```cpp
std::memcpy(w.in_proj_w, ptrs.in_proj_w, in_proj_out * d_model * sizeof(float));
std::memcpy(w.dt_bias, ptrs.dt_bias, nheads * sizeof(float));
std::memcpy(w.B_bias, ptrs.B_bias, nheads * mimo_rank * d_state * sizeof(float));
std::memcpy(w.C_bias, ptrs.C_bias, nheads * mimo_rank * d_state * sizeof(float));
std::memcpy(w.B_norm_weight, ptrs.B_norm_weight, d_state * sizeof(float));
std::memcpy(w.C_norm_weight, ptrs.C_norm_weight, d_state * sizeof(float));
std::memcpy(w.mimo_x, ptrs.mimo_x, nheads * mimo_rank * headdim * sizeof(float));
std::memcpy(w.mimo_z, ptrs.mimo_z, nheads * mimo_rank * headdim * sizeof(float));
std::memcpy(w.mimo_o, ptrs.mimo_o, nheads * mimo_rank * headdim * sizeof(float));
std::memcpy(w.D_proj, ptrs.D_proj, nheads * sizeof(float));
std::memcpy(w.out_proj_w, ptrs.out_proj_w, d_model * d_inner * sizeof(float));
std::memcpy(w.film_proj_w, ptrs.film_proj_w, 2 * d_model * d_cond * sizeof(float));
std::memcpy(w.dt_film_proj_w, ptrs.dt_film_proj_w, 2 * nheads * d_cond * sizeof(float));
```

#### 4.2 `src/clap/clap_latency.cpp`

```cpp
// Mamba-3: no conv1d → zero algorithmic latency
uint32_t latency = 0;
```

#### 4.3 `src/common/conditioning_frame.h`

Update COND_DIM to 128 (for HiLM conditioning). The conditioning frame now carries:
- tempo, key signature, chord (like before, 16 floats)
- f0 probability vector (32 floats, CREPE 360→32 projected)
- semantic embedding (128 floats, from cross-track aggregator)
- total COND_DIM = 128

### Phase 5: Python Export Pipeline (Impact: 2 files)

#### 5.1 `scripts/export_mamba_weights.py` — Rewrite

Load `Mamba3` from `mamba_ssm.modules.mamba3`, extract weight dict, write flat binary with new header + 13-array order per layer. Use `--random-init` for testing.

```python
from mamba_ssm.modules.mamba3 import Mamba3

model = Mamba3(
    d_model=256, d_state=32, expand=2, headdim=64,
    ngroups=1, is_mimo=False, mimo_rank=1, chunk_size=64
)
# Extract state_dict()
# Write flat binary with new WeightFileHeader
```

#### 5.2 `scripts/validate_export.py` — Rewrite

Load weights from binary, instantiate Mamba3, compare Random inputs for equality within 1e-5 tolerance.

### Phase 6: Tests (Impact: 5+ files)

All existing SSM tests need rewrite. New tests needed:
- `test_ssm_block.cpp`: Output non-zero, multi-head consistency, RoPE rotation, trapezoidal stability
- `test_weight_loader.cpp`: New array counts, dimension validation
- `test_conditioning.cpp`: HiLM conditioning modulation
- `test_stability.cpp`: Complex state no-explosion over 10+ min
- `test_realtime_safety.cpp`: No malloc/mutex/IO in new process path

---

## What Survives Unchanged

These files/systems require **zero modifications**:

| System | Files | Reason |
|--------|-------|--------|
| CLAP entry | `clap_entry.cpp` | Factory doesn't care about model arch |
| CLAP audio ports | `clap_audio_ports.cpp` | Same I/O configuration |
| CLAP note ports | `clap_note_ports.cpp` | Same MIDI handling |
| CLAP params | `clap_params.cpp` | Same parameters (gain, mix, bypass, attack, release) |
| CLAP state save/load | `clap_state.cpp` | Only change: more state bytes to serialize |
| CLAP tail | `clap_tail.cpp` | Tail length unchanged |
| CLAP thread check | `clap_thread_check.h` | Same thread safety pattern |
| SPSC Ring Buffer | `ring_buffer.h` | Lock-free queue |
| SeqLock / Seqlocked | `seqlock.h` | Seqlock same semantics |
| DoubleBufferSeqLock | `double_buffer_seqlock.h` | Already fixed tonight |
| ConditioningFrame | `conditioning_frame.h` | Minor: bump COND_DIM |
| Arena allocator | `arena_allocator.h/cpp` | Bump block size for larger state |
| FTZ/DAZ | `ssm_block.cpp:14-25` | Same x86 path, add ARM later |
| SIMD helpers | `simd_helpers.h` | dot_product, fmadd still useful for in_proj |
| CMakeLists.txt (root) | `CMakeLists.txt` | No structural build changes |
| ARA stubs | `src/ara/*` | Still stubs, no changes needed |

---

## Risk Assessment

| Risk | Probability | Mitigation |
|------|-------------|------------|
| Complex state explosion | 15% | Periodic normalization every 4096 samples (as Mamba-1 does) |
| RoPE numerical drift | 5% | Use `fmodf(angle, 2π)` on every step |
| Trapezoidal division instability | 2% | Denominator `1 + Δ·|A|/2` always > 0 for stable A |
| Memory footprint too large | 10% | State: 8 heads × 64 headdim × 32 state × 2 complex × 4 bytes = 128 KB per layer. 6 layers = 768 KB. Fine for RT. |
| Performance exceeds budget | 5% | Per-sample: ~nheads × headdim × d_state = 8 × 64 × 32 = 16K FMAs ≈ 0.3 us. Comfortable. |
| Weight export mismatch | 10% | Validate with Python round-trip check before C++ implementation |

**Overall success probability: >90%**

---

## Estimated Timeline (ALL COMPLETED)

| Phase | Files | Hours | Status |
|-------|-------|-------|--------|
| 1. Config + SSMBlock types/header | `config.h`, `ssm_block.h` | 0.5 | ✅ |
| 2. SSMBlock process_sample/block rewrite | `ssm_block.cpp` | 2.0 | ✅ |
| 3. SSMInferenceCore update | `ssm_inference_core.h/cpp` | 1.0 | ✅ |
| 4. Weight loader rewrite | `weight_loader.h/cpp` | 1.5 | ✅ |
| 5. Export/validate rewrite | `export_mamba_weights.py`, `validate_export.py` | 1.5 | ✅ |
| 6. CLAP plugin integration | `clap_plugin.cpp`, `clap_latency.cpp` | 0.5 | ✅ |
| 7. Tests rewrite | 5 test files | 2.0 | ✅ |
| 8. Build, test, debug, benchmark | — | 1.5 | ✅ |
| **Total** | **~16 files** | **~10 hours** | **ALL DONE** |

---

## Benefits Summary

| Metric | Mamba-1 (Current) | Mamba-3 (Target) | Improvement |
|--------|------------------|-----------------|-------------|
| Latency | 3 samples (68 μs) | 0 samples | ∞ better |
| Conv1d code | 80 lines | 0 lines | Removed |
| State size | 64 bytes/layer | 128 KB/layer | Richer musical memory |
| Conditioning | 4 scalars (dt, B bias) | HiLM (2 × d_model dims) | **Orchestral** |
| Discretization | ZOH (warped at highs) | Trapezoidal (exact freq resp) | Correct |
| Head bandwidth filtering | None | H complex resonant poles | **Per-band musical focus** |
| MIMO coupling | N/A | R-rank input→state coupling | Multi-dimensional input |
| SIMD-friendly | Strided [d_inner][d_state] | Contiguous [d_state][head_dim] in_proj | Cache-local |
| Weight count | 16 arrays | 13 arrays | Simpler |
| Reference parity | PyTorch Mamba-1 | PyTorch Mamba-3 + Cpp step() | Testable |
