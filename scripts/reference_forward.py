#!/usr/bin/env python3
"""
reference_forward.py — Bit-exact Python reference for C++ SSMBlock::process_sample()

Loads the same flat-binary weight file that C++ WeightLoader consumes,
implements the identical math (same order, same ops, same precision),
and produces bit-exact results against the C++ forward pass.

This is the GOLD STANDARD for C++ parity testing.

Usage:
    python3 scripts/reference_forward.py --weights build/parity_test_weights.bin --generate-vectors
    python3 scripts/reference_forward.py --weights models/test_model.bin --input 0.5
"""

import argparse
import math
import struct
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from emt_format import (  # noqa: E402
    HEADER_FMT,
    HEADER_SIZE,
    MAX_MIMO_RANK,
    MAX_NUM_BC_HEADS,
    MAX_NUM_ROPE_ANGLES,
    WEIGHT_MAGIC,
    WEIGHT_VERSION,
)

# ---------------------------------------------------------------------------
# Math helpers — exact C++ equivalents
# ---------------------------------------------------------------------------

def softplus(x):
    """C++: if (x > 20) return x; if (x < -20) return 0; return log1p(exp(x))"""
    if x > 20.0:
        return x
    if x < -20.0:
        return 0.0
    return math.log1p(math.exp(x))


def softplus_np(x):
    """Vectorized softplus matching C++ logic."""
    return np.where(x > 20.0, x, np.where(x < -20.0, 0.0, np.log1p(np.exp(x))))


# ---------------------------------------------------------------------------
# Weight file loader (matches C++ WeightLoader)
# ---------------------------------------------------------------------------

class WeightFileHeader:
    def __init__(self, data):
        assert len(data) == HEADER_SIZE
        # Little-endian unpack
        vals = struct.unpack(HEADER_FMT, data)
        self.magic = vals[0]
        self.version = vals[1]
        self.d_state = vals[2]
        self.d_model = vals[3]
        self.d_inner = vals[4]
        self.nheads = vals[5]
        self.num_layers = vals[6]
        self.headdim = vals[7]
        self.cond_dim = vals[8]
        self.quantization = vals[9]
        self.q_metadata_size = vals[10]
        self.flags = vals[11]
        self.weight_data_size = vals[12]
        # Deviation D3: 0 = legacy file (pre-mimo_rank field, treated as rank 1),
        # 1 = rank-1 SISO.  The C++ runtime refuses values > 1.
        self.mimo_rank = vals[13]

    def validate(self):
        assert self.magic == WEIGHT_MAGIC, f"Bad magic: {self.magic:08X}"
        assert self.version == WEIGHT_VERSION, f"Bad version: {self.version}"
        assert self.d_state > 0
        assert self.d_model > 0
        assert self.d_inner > 0
        assert self.nheads > 0
        assert self.headdim > 0
        assert self.cond_dim > 0
        assert self.num_layers > 0


class SSMWeights:
    """Python mirror of C++ SSMWeights struct."""
    def __init__(self):
        self.in_proj_w = None       # [in_proj_out, d_model]
        self.dt_bias = None         # [nheads]
        self.B_bias = None          # [nheads, mimo_rank, d_state]
        self.C_bias = None          # [nheads, mimo_rank, d_state]
        self.B_norm_weight = None   # [d_state]
        self.C_norm_weight = None   # [d_state]
        self.mimo_x = None          # [nheads, mimo_rank, headdim]
        self.mimo_z = None          # [nheads, mimo_rank, headdim]
        self.mimo_o = None          # [nheads, mimo_rank, headdim]
        self.D_proj = None          # [nheads]
        self.out_proj_w = None      # [d_model, d_inner]
        self.film_proj_w = None     # [2*d_model, d_cond]
        self.dt_film_proj_w = None  # [2*nheads, d_cond]


class WeightLoader:
    """Python equivalent of C++ WeightLoader — mmap not needed, just read."""
    def __init__(self, path):
        with open(path, 'rb') as f:
            self.data = f.read()

        self.header = WeightFileHeader(self.data[:HEADER_SIZE])
        self.header.validate()

        self.weight_data = self.data[HEADER_SIZE + self.header.q_metadata_size:]
        self._validate_size()

    def _validate_size(self):
        h = self.header
        in_proj_out = (h.d_inner * 2
                       + h.nheads * MAX_MIMO_RANK * h.d_state * 2
                       + h.nheads * 3
                       + MAX_NUM_ROPE_ANGLES)

        elems = 0
        elems += in_proj_out * h.d_model
        elems += h.nheads
        elems += h.nheads * MAX_MIMO_RANK * h.d_state
        elems += h.nheads * MAX_MIMO_RANK * h.d_state
        elems += h.d_state
        elems += h.d_state
        elems += h.nheads * MAX_MIMO_RANK * h.headdim
        elems += h.nheads * MAX_MIMO_RANK * h.headdim
        elems += h.nheads * MAX_MIMO_RANK * h.headdim
        elems += h.nheads
        elems += h.d_model * h.d_inner
        elems += 2 * h.d_model * h.cond_dim
        elems += 2 * h.nheads * h.cond_dim

        expected_bytes = elems * 4 * h.num_layers
        actual_bytes = len(self.weight_data)
        if actual_bytes < expected_bytes:
            raise ValueError(
                f"Weight data too small: expected {expected_bytes} bytes, "
                f"got {actual_bytes}"
            )

    def get_layer(self, layer_idx, mimo_rank=1):
        """Extract a single layer's weights as SSMWeights (using MAX_* file layout)."""
        h = self.header
        if layer_idx >= h.num_layers:
            raise IndexError(f"Layer {layer_idx} >= {h.num_layers}")

        in_proj_out = (h.d_inner * 2
                       + h.nheads * MAX_MIMO_RANK * h.d_state * 2
                       + h.nheads * 3
                       + MAX_NUM_ROPE_ANGLES)

        # Compute bytes per layer (MAX_* layout)
        elems = 0
        elems += in_proj_out * h.d_model
        elems += h.nheads
        elems += h.nheads * MAX_MIMO_RANK * h.d_state
        elems += h.nheads * MAX_MIMO_RANK * h.d_state
        elems += h.d_state
        elems += h.d_state
        elems += h.nheads * MAX_MIMO_RANK * h.headdim
        elems += h.nheads * MAX_MIMO_RANK * h.headdim
        elems += h.nheads * MAX_MIMO_RANK * h.headdim
        elems += h.nheads
        elems += h.d_model * h.d_inner
        elems += 2 * h.d_model * h.cond_dim
        elems += 2 * h.nheads * h.cond_dim
        bytes_per_layer = elems * 4

        base = layer_idx * bytes_per_layer
        data = self.weight_data[base:base + bytes_per_layer]
        floats = np.frombuffer(data, dtype=np.float32)

        w = SSMWeights()
        off = 0

        def take(n):
            nonlocal off
            arr = floats[off:off + n]
            off += n
            return arr

        w.in_proj_w = take(in_proj_out * h.d_model).reshape(in_proj_out, h.d_model)
        w.dt_bias = take(h.nheads)
        w.B_bias = take(h.nheads * MAX_MIMO_RANK * h.d_state).reshape(h.nheads, MAX_MIMO_RANK, h.d_state)
        w.C_bias = take(h.nheads * MAX_MIMO_RANK * h.d_state).reshape(h.nheads, MAX_MIMO_RANK, h.d_state)
        w.B_norm_weight = take(h.d_state)
        w.C_norm_weight = take(h.d_state)
        w.mimo_x = take(h.nheads * MAX_MIMO_RANK * h.headdim).reshape(h.nheads, MAX_MIMO_RANK, h.headdim)
        w.mimo_z = take(h.nheads * MAX_MIMO_RANK * h.headdim).reshape(h.nheads, MAX_MIMO_RANK, h.headdim)
        w.mimo_o = take(h.nheads * MAX_MIMO_RANK * h.headdim).reshape(h.nheads, MAX_MIMO_RANK, h.headdim)
        w.D_proj = take(h.nheads)
        w.out_proj_w = take(h.d_model * h.d_inner).reshape(h.d_model, h.d_inner)
        w.film_proj_w = take(2 * h.d_model * h.cond_dim).reshape(2 * h.d_model, h.cond_dim)
        w.dt_film_proj_w = take(2 * h.nheads * h.cond_dim).reshape(2 * h.nheads, h.cond_dim)

        return w


# ---------------------------------------------------------------------------
# Reference forward pass — bit-exact to C++ SSMBlock::process_sample()
# ---------------------------------------------------------------------------

class ReferenceForward:
    """
    Bit-exact Python reference for C++ SSMBlock::process_sample().

    Loads weights from the same flat-binary file that C++ WeightLoader uses,
    implements the identical math, and produces the same outputs.
    """

    def __init__(self, weight_path, layer_idx=0, mimo_rank=1):
        self.loader = WeightLoader(weight_path)
        self.h = self.loader.header
        self.w = self.loader.get_layer(layer_idx, MAX_MIMO_RANK)
        self.mimo_rank = MAX_MIMO_RANK  # Must match C++ for weight layout stride
        self.runtime_mimo_rank = mimo_rank  # Kept for reference (actual model rank)

        # State (mutable, like C++ SSMState)
        self.ssm_real = np.zeros((self.h.nheads, self.h.headdim, self.h.d_state), dtype=np.float32)
        self.ssm_imag = np.zeros((self.h.nheads, self.h.headdim, self.h.d_state), dtype=np.float32)
        self.k_state = np.zeros((mimo_rank, self.h.nheads, self.h.d_state), dtype=np.float32)
        self.angle_state = np.zeros((self.h.nheads, self.h.d_state // 2), dtype=np.float32)

        # Parameters (like C++ SSMBlockParams)
        self.gain = 1.0
        self.mix = 1.0

        # Conditioning
        self.cond = np.zeros(self.h.cond_dim, dtype=np.float32)
        self.conditioning_active = False

    def set_conditioning(self, cond_vector):
        """Set FiLM conditioning vector (like C++ set_conditioning)."""
        if cond_vector is not None and len(cond_vector) > 0:
            n = min(len(cond_vector), self.h.cond_dim)
            self.cond[:n] = cond_vector[:n]
            self.conditioning_active = np.any(self.cond != 0.0)
        else:
            self.cond.fill(0.0)
            self.conditioning_active = False

    def reset(self):
        """Reset all state to zero."""
        self.ssm_real.fill(0.0)
        self.ssm_imag.fill(0.0)
        self.k_state.fill(0.0)
        self.angle_state.fill(0.0)

    def process_sample(self, input_scalar):
        """
        Process one sample through the Mamba-3 SSM layer.
        Returns: d_model output vector (np.ndarray, float32)
        """
        h = self.h
        w = self.w
        d_state = h.d_state
        d_model = h.d_model
        d_inner = h.d_inner
        nheads = h.nheads
        headdim = h.headdim
        mimo_rank = self.mimo_rank
        num_rope_angles = d_state // 2
        num_bc_heads = 1  # C++ MAX_NUM_BC_HEADS = 1
        A_FLOOR = 1.0

        # ===== Step 0: FiLM conditioning on input =====
        u_film = np.zeros(d_model, dtype=np.float32)
        if self.conditioning_active:
            for j in range(d_model):
                gamma = 0.0
                beta = 0.0
                for k in range(h.cond_dim):
                    gamma += w.film_proj_w[2 * j, k] * self.cond[k]
                    beta += w.film_proj_w[2 * j + 1, k] * self.cond[k]
                u_film[j] = input_scalar * (1.0 + gamma) + beta
        else:
            u_film[:] = input_scalar

        # ===== Step 1: in_proj — single fused linear projection =====
        in_proj_out_dim = (d_inner * 2
                           + nheads * mimo_rank * d_state * 2
                           + nheads * 3
                           + num_rope_angles)

        in_proj_out_arr = np.zeros(in_proj_out_dim, dtype=np.float32)
        for out_idx in range(in_proj_out_dim):
            s = 0.0
            for j in range(d_model):
                s += w.in_proj_w[out_idx, j] * u_film[j]
            in_proj_out_arr[out_idx] = s

        # Split components
        idx = 0
        z = in_proj_out_arr[idx:idx + d_inner]; idx += d_inner
        x = in_proj_out_arr[idx:idx + d_inner]; idx += d_inner
        bc_size = nheads * mimo_rank * d_state
        B_flat = in_proj_out_arr[idx:idx + bc_size]; idx += bc_size
        C_flat = in_proj_out_arr[idx:idx + bc_size]; idx += bc_size
        dd_dt = in_proj_out_arr[idx:idx + nheads]; idx += nheads
        dd_A = in_proj_out_arr[idx:idx + nheads]; idx += nheads
        trap = in_proj_out_arr[idx:idx + nheads]; idx += nheads
        angles = in_proj_out_arr[idx:idx + num_rope_angles]

        # Reshape B, C: flat -> [nheads, mimo_rank, d_state]
        B = np.zeros((nheads, mimo_rank, d_state), dtype=np.float32)
        C = np.zeros((nheads, mimo_rank, d_state), dtype=np.float32)
        for h_idx in range(nheads):
            for r in range(mimo_rank):
                for s in range(d_state):
                    flat_idx = h_idx * mimo_rank * d_state + r * d_state + s
                    B[h_idx, r, s] = B_flat[flat_idx]
                    C[h_idx, r, s] = C_flat[flat_idx]

        # ===== Step 2: Preprocess A and Δ (per head) =====
        A_t = np.zeros(nheads, dtype=np.float32)
        DT_t = np.zeros(nheads, dtype=np.float32)
        for h_idx in range(nheads):
            a_raw = dd_A[h_idx]
            A_t_val = -softplus(a_raw)
            if A_t_val > -A_FLOOR:
                A_t_val = -A_FLOOR
            A_t[h_idx] = A_t_val

            # D6 (closed 2026-10-04): Δ FiLM, same convention as the input FiLM
            # path: v -> v * (1 + γ) + β, per head, only when conditioning is on.
            dt_pre = dd_dt[h_idx] + w.dt_bias[h_idx]
            if self.conditioning_active:
                gamma = 0.0
                beta = 0.0
                for k in range(w.d_cond):
                    gamma += w.dt_film_proj_w[2 * h_idx, k] * self.cond[k]
                    beta += w.dt_film_proj_w[2 * h_idx + 1, k] * self.cond[k]
                dt_pre = dt_pre * (1.0 + gamma) + beta
            DT_t[h_idx] = softplus(dt_pre)

        # ===== Step 3: RMSNorm on B and C =====
        B_norm = np.zeros((nheads, mimo_rank, d_state), dtype=np.float32)
        C_norm = np.zeros((nheads, mimo_rank, d_state), dtype=np.float32)

        for h_idx in range(nheads):
            for r in range(mimo_rank):
                # B RMSNorm
                rms = 0.0
                for s in range(d_state):
                    rms += B[h_idx, r, s] ** 2
                rms = math.sqrt(rms / d_state + 1e-5)
                inv_rms = 1.0 / rms
                for s in range(d_state):
                    B_norm[h_idx, r, s] = (B[h_idx, r, s] * inv_rms * w.B_norm_weight[s]
                                            + w.B_bias[h_idx, r, s])
                # C RMSNorm
                rms = 0.0
                for s in range(d_state):
                    rms += C[h_idx, r, s] ** 2
                rms = math.sqrt(rms / d_state + 1e-5)
                inv_rms = 1.0 / rms
                for s in range(d_state):
                    C_norm[h_idx, r, s] = (C[h_idx, r, s] * inv_rms * w.C_norm_weight[s]
                                            + w.C_bias[h_idx, r, s])

        # ===== Step 4: RoPE on B and C =====
        for h_idx in range(nheads):
            for a in range(num_rope_angles):
                angle = self.angle_state[h_idx, a] + angles[a]
                angle = math.fmod(angle, 2.0 * math.pi)
                self.angle_state[h_idx, a] = angle

                cos_a = math.cos(angle)
                sin_a = math.sin(angle)

                s0 = a
                s1 = a + num_rope_angles
                if s1 >= d_state:
                    continue

                for r in range(mimo_rank):
                    B0 = B_norm[h_idx, r, s0]
                    B1 = B_norm[h_idx, r, s1]
                    B_norm[h_idx, r, s0] = B0 * cos_a - B1 * sin_a
                    B_norm[h_idx, r, s1] = B0 * sin_a + B1 * cos_a

                    C0 = C_norm[h_idx, r, s0]
                    C1 = C_norm[h_idx, r, s1]
                    C_norm[h_idx, r, s0] = C0 * cos_a + C1 * sin_a
                    C_norm[h_idx, r, s1] = -C0 * sin_a + C1 * cos_a

        # ===== Step 5: Multi-head SSM recurrence (trapezoidal) =====
        y_head = np.zeros(d_inner, dtype=np.float32)

        for h_idx in range(nheads):
            half_a = A_t[h_idx] * DT_t[h_idx] * 0.5
            inv_denom = 1.0 / (1.0 - half_a)
            A_bar = (1.0 + half_a) * inv_denom
            B_bar_scale = DT_t[h_idx] * inv_denom

            for d in range(headdim):
                x_hd = x[h_idx * headdim + d]
                for s in range(d_state):
                    B_val = B_norm[h_idx, 0, s]
                    C_val = C_norm[h_idx, 0, s]

                    self.ssm_real[h_idx, d, s] = (A_bar * self.ssm_real[h_idx, d, s]
                                                   + B_bar_scale * B_val * x_hd)
                    y_head[h_idx * headdim + d] += C_val * self.ssm_real[h_idx, d, s]

        # ===== Step 6: MIMO combine (SiLU z-gating) =====
        for h_idx in range(nheads):
            for d in range(headdim):
                z_val = z[h_idx * headdim + d]
                z_sig = 1.0 / (1.0 + math.exp(-z_val))
                z_gated = z_val * z_sig
                y_head[h_idx * headdim + d] *= z_gated

        # ===== Step 7: Output projection + D skip + residual =====
        out = np.zeros(d_model, dtype=np.float32)
        for j in range(d_model):
            s = 0.0
            for i in range(d_inner):
                s += w.out_proj_w[j, i] * y_head[i]
            if j < nheads:
                s += w.D_proj[j] * x[j * headdim]
            s += u_film[j]
            out[j] = s

        # Apply gain
        out *= self.gain

        # NOTE (deviation D5, closed 2026-10-04): no periodic state
        # renormalization.  The trapezoidal update is A-stable for the clamped
        # A <= -1, so the state is bounded without a non-linear guard.

        return out


# ---------------------------------------------------------------------------
# CLI / Test vector generation
# ---------------------------------------------------------------------------

def generate_test_vectors(weight_path, output_dir, num_samples=16):
    """Generate expected outputs for C++ parity test."""
    os.makedirs(output_dir, exist_ok=True)

    rf = ReferenceForward(weight_path, layer_idx=0, mimo_rank=1)

    # Set conditioning (matching C++ test)
    cond = np.zeros(rf.h.cond_dim, dtype=np.float32)
    cond[0] = 0.5
    cond[1] = -0.25
    rf.set_conditioning(cond)

    # Test inputs matching C++
    test_inputs = [0.0, 1.0, -1.0, 0.5, -0.5, 0.25, -0.25, 0.75,
                   0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    outputs = []
    for val in test_inputs[:num_samples]:
        out = rf.process_sample(val)
        outputs.append(out)

    # Write expected outputs
    expect_path = os.path.join(output_dir, 'parity_expected.bin')
    with open(expect_path, 'wb') as f:
        for out in outputs:
            f.write(out.tobytes())

    print(f"Wrote {expect_path} ({len(outputs)} outputs x {rf.h.d_model} channels)")
    return expect_path


def main():
    p = argparse.ArgumentParser(description="Mamba-3 reference forward pass")
    p.add_argument('--weights', required=True, help='Path to flat-binary weight file')
    p.add_argument('--layer', type=int, default=0, help='Layer index (default: 0)')
    p.add_argument('--mimo-rank', type=int, default=1, help='MIMO rank (default: 1)')
    p.add_argument('--input', type=float, default=None, help='Single input value to process')
    p.add_argument('--generate-vectors', action='store_true',
                   help='Generate test vectors for C++ parity test')
    p.add_argument('--output-dir', default='build', help='Output directory for test vectors')
    p.add_argument('--num-samples', type=int, default=16, help='Number of test samples')
    args = p.parse_args()

    if args.generate_vectors:
        generate_test_vectors(args.weights, args.output_dir, args.num_samples)
        return 0

    rf = ReferenceForward(args.weights, args.layer, args.mimo_rank)

    if args.input is not None:
        out = rf.process_sample(args.input)
        print(f"Input: {args.input}")
        print(f"Output: {out}")
    else:
        print("Use --input <value> to process a single sample")
        print("Use --generate-vectors to create C++ parity test vectors")

    return 0


if __name__ == '__main__':
    sys.exit(main())
