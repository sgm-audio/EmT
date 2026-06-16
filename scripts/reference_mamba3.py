#!/usr/bin/env python3
"""
Mamba-3 C++ Parity Reference

Standalone Python reference that matches the C++ SSMBlock::process_sample()
step-for-step.  Uses only numpy (no PyTorch) so the logic is transparent
and the test can run without a GPU.

Run with --generate to create test vectors for C++ parity tests.
"""

import argparse
import math
import struct
import os
import sys
import numpy as np

HEADER_FMT = '<12IQQ'
HEADER_SIZE = 64
WEIGHT_MAGIC = 0x4D424D41
WEIGHT_VERSION = 2


# === Step-by-step match for C++ SSMBlock::process_sample() ===

def softplus(x: np.ndarray) -> np.ndarray:
    """Matches C++ softplus: smooth approximation with stability cutoffs."""
    y = np.where(x > 20.0, x, np.where(x < -20.0, 0.0, np.log1p(np.exp(x))))
    return y


def mamba3_process_sample(
    input_scalar: float,            # C++: float input
    d_state: int, d_model: int, d_inner: int,
    nheads: int, headdim: int, mimo_rank: int,
    num_bc_heads: int, d_cond: int,
    # Weights
    in_proj_w: np.ndarray,          # [in_proj_out, d_model]
    dt_bias: np.ndarray,            # [nheads]
    B_bias: np.ndarray,             # [nheads, mimo_rank, d_state]
    C_bias: np.ndarray,             # [nheads, mimo_rank, d_state]
    B_norm_weight: np.ndarray,      # [d_state]
    C_norm_weight: np.ndarray,      # [d_state]
    D_proj: np.ndarray,             # [nheads]
    out_proj_w: np.ndarray,         # [d_model, d_inner]
    film_proj_w: np.ndarray,        # [2*d_model, d_cond]
    # State (mutated)
    ssm_real: np.ndarray,           # [nheads, headdim, d_state]
    angle_state: np.ndarray,        # [nheads, num_rope_angles]
    # Conditioning
    cond: np.ndarray,               # [d_cond]
    conditioning_active: bool,
    sample_count: int,
):
    num_rope_angles = d_state // 2
    bc_size = nheads * mimo_rank * d_state
    in_proj_out = (d_inner * 2 + nheads * mimo_rank * d_state * 2
                   + nheads * 3 + num_rope_angles)

    # === Step 0: FiLM conditioning ===
    u_film = np.zeros(d_model, dtype=np.float32)
    if conditioning_active:
        for j in range(d_model):
            gamma = 0.0
            beta = 0.0
            for k in range(d_cond):
                gamma += film_proj_w[2 * j, k] * cond[k]
                beta  += film_proj_w[2 * j + 1, k] * cond[k]
            u_film[j] = input_scalar * (1.0 + gamma) + beta
    else:
        u_film[:] = input_scalar

    # === Step 1: in_proj ===
    in_proj_out_arr = np.zeros(in_proj_out, dtype=np.float32)
    for out_idx in range(in_proj_out):
        s = 0.0
        for j in range(d_model):
            s += in_proj_w[out_idx, j] * u_film[j]
        in_proj_out_arr[out_idx] = s

    # Split components
    idx = 0
    z = in_proj_out_arr[idx:idx + d_inner]; idx += d_inner
    x = in_proj_out_arr[idx:idx + d_inner]; idx += d_inner
    B_flat = in_proj_out_arr[idx:idx + bc_size]; idx += bc_size
    C_flat = in_proj_out_arr[idx:idx + bc_size]; idx += bc_size
    dd_dt = in_proj_out_arr[idx:idx + nheads]; idx += nheads
    dd_A = in_proj_out_arr[idx:idx + nheads]; idx += nheads
    trap = in_proj_out_arr[idx:idx + nheads]; idx += nheads
    angles = in_proj_out_arr[idx:idx + num_rope_angles]

    # Reshape B, C: flat → [nheads, mimo_rank, d_state]
    # C++ layout: (g * nheads + h) * mimo_rank * d_state + r * d_state + s
    # with g=0 (num_bc_heads=1)
    B = np.zeros((nheads, mimo_rank, d_state), dtype=np.float32)
    C = np.zeros((nheads, mimo_rank, d_state), dtype=np.float32)
    for h in range(nheads):
        for r in range(mimo_rank):
            for s in range(d_state):
                flat_idx = h * mimo_rank * d_state + r * d_state + s
                B[h, r, s] = B_flat[flat_idx]
                C[h, r, s] = C_flat[flat_idx]

    # === Step 2: Preprocess A and Δ ===
    A_FLOOR = 1.0  # clamp: A_t ∈ (-∞, -A_FLOOR]
    A_t = np.zeros(nheads, dtype=np.float32)
    DT_t = np.zeros(nheads, dtype=np.float32)
    for h in range(nheads):
        a_raw = dd_A[h]
        A_t_val = -softplus(np.array([a_raw]))[0]
        if A_t_val > -A_FLOOR:
            A_t_val = -A_FLOOR
        A_t[h] = A_t_val
        dt_raw = dd_dt[h]
        DT_t[h] = softplus(np.array([dt_raw + dt_bias[h]]))[0]

    # === Step 3: RMSNorm on B and C ===
    B_norm = np.zeros((nheads, mimo_rank, d_state), dtype=np.float32)
    C_norm = np.zeros((nheads, mimo_rank, d_state), dtype=np.float32)

    for h in range(nheads):
        for r in range(mimo_rank):
            # B RMSNorm
            rms = 0.0
            for s in range(d_state):
                rms += B[h, r, s] ** 2
            rms = math.sqrt(rms / d_state + 1e-5)
            inv_rms = 1.0 / rms
            for s in range(d_state):
                B_norm[h, r, s] = (B[h, r, s] * inv_rms * B_norm_weight[s]
                                    + B_bias[h, r, s])
            # C RMSNorm
            rms = 0.0
            for s in range(d_state):
                rms += C[h, r, s] ** 2
            rms = math.sqrt(rms / d_state + 1e-5)
            inv_rms = 1.0 / rms
            for s in range(d_state):
                C_norm[h, r, s] = (C[h, r, s] * inv_rms * C_norm_weight[s]
                                    + C_bias[h, r, s])

    # === Step 4: RoPE on B and C ===
    for h in range(nheads):
        for a in range(num_rope_angles):
            angle = angle_state[h, a] + angles[a]
            angle = math.fmod(angle, 2.0 * math.pi)
            angle_state[h, a] = angle

            cos_a = math.cos(angle)
            sin_a = math.sin(angle)

            s0 = a
            s1 = a + num_rope_angles
            if s1 >= d_state:
                continue

            for r in range(mimo_rank):
                B0 = B_norm[h, r, s0]
                B1 = B_norm[h, r, s1]
                B_norm[h, r, s0] = B0 * cos_a - B1 * sin_a
                B_norm[h, r, s1] = B0 * sin_a + B1 * cos_a

                C0 = C_norm[h, r, s0]
                C1 = C_norm[h, r, s1]
                C_norm[h, r, s0] = C0 * cos_a + C1 * sin_a
                C_norm[h, r, s1] = -C0 * sin_a + C1 * cos_a

    # === Step 5: Multi-head SSM recurrence (trapezoidal) ===
    y_head = np.zeros(d_inner, dtype=np.float32)

    for h in range(nheads):
        half_a = A_t[h] * DT_t[h] * 0.5
        inv_denom = 1.0 / (1.0 - half_a)
        A_bar = (1.0 + half_a) * inv_denom
        B_bar_scale = DT_t[h] * inv_denom

        for d in range(headdim):
            x_hd = x[h * headdim + d]

            for s in range(d_state):
                # Use rank 0
                B_val = B_norm[h, 0, s]
                C_val = C_norm[h, 0, s]

                ssm_real[h, d, s] = (A_bar * ssm_real[h, d, s]
                                     + B_bar_scale * B_val * x_hd)
                y_head[h * headdim + d] += C_val * ssm_real[h, d, s]

    # === Step 6: MIMO combine (SiLU z-gating) ===
    for h in range(nheads):
        for d in range(headdim):
            z_val = z[h * headdim + d]
            z_sig = 1.0 / (1.0 + math.exp(-z_val))
            z_gated = z_val * z_sig
            y_head[h * headdim + d] *= z_gated

    # === Step 7: Output projection + D skip + residual ===
    out = np.zeros(d_model, dtype=np.float32)
    for j in range(d_model):
        s = 0.0
        for i in range(d_inner):
            s += out_proj_w[j, i] * y_head[i]
        if j < nheads:
            s += D_proj[j] * x[j * headdim]
        s += u_film[j]
        out[j] = s

    # === Step 8: State norm (simplified — caller handles) ===
    sample_count += 1
    STATE_NORM_INTERVAL = 4096
    if sample_count >= STATE_NORM_INTERVAL:
        sample_count = 0
        max_h = max(abs(ssm_real).max(), 1e-10)
        if max_h > 1.0:
            ssm_real /= max_h

    return out, ssm_real, angle_state, sample_count


# === Test vector generation ===

def generate_test_vectors(output_dir: str, d_state=16, d_model=16, d_inner=32,
                          nheads=2, headdim=None, mimo_rank=1):
    """Generate weight file and expected outputs for C++ parity test."""
    if headdim is None:
        headdim = d_inner // nheads
    num_bc_heads = 1
    d_cond = 128
    num_rope_angles = d_state // 2
    MAX_MIMO_RANK = 4  # Must match C++ config.h for weight file layout
    MAX_NUM_ROPE_ANGLES = 16  # Must match C++ config.h

    # Fixed seed for reproducibility
    rng = np.random.RandomState(42)

    # Generate random weights for actual mimo_rank (used by forward pass)
    in_proj_out = (d_inner * 2 + nheads * mimo_rank * d_state * 2
                   + nheads * 3 + num_rope_angles)

    in_proj_w = rng.randn(in_proj_out, d_model).astype(np.float32) * 0.1
    dt_bias = rng.randn(nheads).astype(np.float32) * 0.1
    B_bias = rng.randn(nheads, mimo_rank, d_state).astype(np.float32) * 0.1
    C_bias = rng.randn(nheads, mimo_rank, d_state).astype(np.float32) * 0.1
    B_norm_weight = (1.0 + rng.randn(d_state).astype(np.float32) * 0.1)
    C_norm_weight = (1.0 + rng.randn(d_state).astype(np.float32) * 0.1)
    D_proj = rng.randn(nheads).astype(np.float32) * 0.1
    out_proj_w = rng.randn(d_model, d_inner).astype(np.float32) * 0.1
    film_proj_w = rng.randn(2 * d_model, d_cond).astype(np.float32) * 0.01

    # For weight file layout, expand B_bias, C_bias, mimo_x/z/o to MAX_MIMO_RANK
    # (C++ expects MAX_MIMO_RANK slots; we only use rank 0 in forward pass)
    B_bias_file = np.zeros((nheads, MAX_MIMO_RANK, d_state), dtype=np.float32)
    B_bias_file[:, :mimo_rank, :] = B_bias
    C_bias_file = np.zeros((nheads, MAX_MIMO_RANK, d_state), dtype=np.float32)
    C_bias_file[:, :mimo_rank, :] = C_bias
    mimo_x = np.zeros((nheads, MAX_MIMO_RANK, headdim), dtype=np.float32)
    mimo_z = np.zeros((nheads, MAX_MIMO_RANK, headdim), dtype=np.float32)
    mimo_o = np.zeros((nheads, MAX_MIMO_RANK, headdim), dtype=np.float32)
    dt_film_proj_w = np.zeros((2 * nheads, d_cond), dtype=np.float32)

    # Write weight file (with MAX_MIMO_RANK layout)
    os.makedirs(output_dir, exist_ok=True)
    weight_path = os.path.join(output_dir, 'parity_test_weights.bin')

    in_proj_out_file = (d_inner * 2 + nheads * MAX_MIMO_RANK * d_state * 2
                        + nheads * 3 + MAX_NUM_ROPE_ANGLES)
    fpl = (in_proj_out_file * d_model + nheads + nheads * MAX_MIMO_RANK * d_state
           + nheads * MAX_MIMO_RANK * d_state + d_state + d_state
           + nheads * MAX_MIMO_RANK * headdim + nheads * MAX_MIMO_RANK * headdim
           + nheads * MAX_MIMO_RANK * headdim + nheads + d_model * d_inner
           + 2 * d_model * d_cond + 2 * nheads * d_cond)

    # Expand in_proj_w to MAX_MIMO_RANK + MAX_NUM_ROPE_ANGLES layout for weight file
    # Layout: [z(d_inner), x(d_inner), B(nh*MAX_MIMO_RANK*ds), C(nh*MAX_MIMO_RANK*ds), dd_dt(nh), dd_A(nh), trap(nh), angles(MAX_NUM_ROPE_ANGLES)]
    # B/C stride: (g*nheads+h)*MAX_MIMO_RANK*d_state + r*d_state + s  (g=0, r varies)
    #   head h, rank r: h*MAX_MIMO_RANK*d_state + r*d_state + s
    # Original (mimo_rank=1):  h*1*d_state + 0*d_state + s = h*d_state + s
    in_proj_w_file = np.zeros((in_proj_out_file, d_model), dtype=np.float32)
    # z and x portions are the same size (d_inner each)
    in_proj_w_file[:d_inner * 2, :] = in_proj_w[:d_inner * 2, :]

    # B portion: expand padding between heads (rank 0 data, ranks 1-3 are zeros)
    b_start = d_inner * 2
    for h_idx in range(nheads):
        old_offset = b_start + h_idx * d_state  # h*d_state (mimo_rank=1 layout)
        new_offset = b_start + h_idx * MAX_MIMO_RANK * d_state  # h*MAX_MIMO_RANK*d_state
        in_proj_w_file[new_offset:new_offset + d_state, :] = in_proj_w[old_offset:old_offset + d_state, :]

    # C portion: same expansion (zero-padded ranks 1-3 between heads)
    c_start = b_start + nheads * MAX_MIMO_RANK * d_state
    c_start_orig = b_start + nheads * d_state  # in original (mimo_rank=1), C starts after B
    for h_idx in range(nheads):
        old_offset = c_start_orig + h_idx * d_state
        new_offset = c_start + h_idx * MAX_MIMO_RANK * d_state
        in_proj_w_file[new_offset:new_offset + d_state, :] = in_proj_w[old_offset:old_offset + d_state, :]

    # dd_dt, dd_A, trap, angles portions (contiguous, same sizes)
    rest_start_new = c_start + nheads * MAX_MIMO_RANK * d_state
    rest_start_old = c_start_orig + nheads * d_state
    dt_a_trap_size = nheads * 3
    angles_size = num_rope_angles
    in_proj_w_file[rest_start_new:rest_start_new + dt_a_trap_size, :] = in_proj_w[rest_start_old:rest_start_old + dt_a_trap_size, :]
    angles_start_new = rest_start_new + dt_a_trap_size
    angles_start_old = rest_start_old + dt_a_trap_size
    in_proj_w_file[angles_start_new:angles_start_new + angles_size, :] = in_proj_w[angles_start_old:angles_start_old + angles_size, :]

    with open(weight_path, 'wb') as f:
        # Header: 12 uint32 + 2 uint64
        hdr = struct.pack(HEADER_FMT,
                          WEIGHT_MAGIC, WEIGHT_VERSION,
                          d_state, d_model, d_inner, nheads, 1, headdim,
                          d_cond, 0, 0, 0,  # magic, version, d_state, d_model, d_inner, nheads, num_layers, headdim, cond_dim, quantization, q_metadata_size, flags
                          fpl * 4, 0)  # weight_data_size, reserved
        f.write(hdr)
        assert len(hdr) == HEADER_SIZE

        # Write weights in C++ order (MAX_MIMO_RANK layout)
        def w(data):
            f.write(data.astype(np.float32).tobytes())

        w(in_proj_w_file)
        w(dt_bias)
        w(B_bias_file)
        w(C_bias_file)
        w(B_norm_weight)
        w(C_norm_weight)
        w(mimo_x)
        w(mimo_z)
        w(mimo_o)
        w(D_proj)
        w(out_proj_w)
        w(film_proj_w)
        w(dt_film_proj_w)

    print(f"Wrote {weight_path} ({fpl} floats per layer, MAX_MIMO_RANK={MAX_MIMO_RANK})")

    # === Run Python reference on multiple samples ===
    ssm_real = np.zeros((nheads, headdim, d_state), dtype=np.float32)
    angle_state = np.zeros((nheads, num_rope_angles), dtype=np.float32)
    cond = np.zeros(d_cond, dtype=np.float32)
    cond[0] = 0.5  # non-zero conditioning
    cond[1] = -0.25
    sample_count = 0

    # Test inputs: [0.0, 1.0, -1.0, 0.5, -0.5, ...]
    test_inputs = [0.0, 1.0, -1.0, 0.5, -0.5, 0.25, -0.25, 0.75,
                   0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    results = {}
    for i, val in enumerate(test_inputs):
        out, ssm_real, angle_state, sample_count = mamba3_process_sample(
            val, d_state, d_model, d_inner, nheads, headdim, mimo_rank,
            num_bc_heads, d_cond, in_proj_w, dt_bias, B_bias, C_bias,
            B_norm_weight, C_norm_weight, D_proj, out_proj_w, film_proj_w,
            ssm_real, angle_state, cond, True, sample_count)
        results[f'out_{i}'] = out.astype(np.float32)

    # Write expected outputs
    expect_path = os.path.join(output_dir, 'parity_expected.bin')
    with open(expect_path, 'wb') as f:
        for i in range(len(test_inputs)):
            f.write(results[f'out_{i}'].tobytes())

    print(f"Wrote {expect_path} ({len(test_inputs)} outputs)")
    return weight_path, expect_path, test_inputs


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--generate', action='store_true',
                   help='Generate test vectors for C++ parity test')
    p.add_argument('--output-dir', default='build',
                   help='Output directory for test vectors')
    p.add_argument('--state-dim', type=int, default=16)
    p.add_argument('--model-dim', type=int, default=16)
    p.add_argument('--inner-dim', type=int, default=32)
    p.add_argument('--nheads', type=int, default=2)
    args = p.parse_args()

    if args.generate:
        generate_test_vectors(args.output_dir,
                              args.state_dim, args.model_dim, args.inner_dim,
                              args.nheads)
        return 0

    print("Run with --generate to create test vectors")
    return 0


if __name__ == '__main__':
    sys.exit(main())
