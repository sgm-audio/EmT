#!/usr/bin/env python3
"""
Mamba-3 Weight Export Tool

Exports a trained PyTorch Mamba-3 model to the flat-binary weight format
consumed by the C++ WeightLoader.  Also generates config.h with matching
MAX_* constants.

Usage:
    python scripts/export_mamba_weights.py --state-dim 16 --model-dim 16 \\
        --inner-dim 32 --nheads 2 --layers 1 --random-init \\
        --output models/test_model.bin --config-h src/config.h
"""

import argparse
import struct
import os
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Reference Mamba-3 block (PyTorch)
#
# Matches the C++ SSMBlock architecture:
#   - No conv1d (zero latency)
#   - Fused in_proj: all projections in one linear layer
#   - Per-head A (decay) and dt (step size)
#   - RMSNorm on B and C
#   - RoPE rotary embedding
#   - Exponential-trapezoidal discretization
#   - Multi-head SSM recurrence
#   - SiLU z-gating
#   - FiLM conditioning
#   - MIMO rank (simplified to rank=1 for initial export)
# ---------------------------------------------------------------------------

class Mamba3Block(nn.Module):
    """Single Mamba-3 SSM layer."""

    def __init__(self, d_state: int, d_model: int, d_inner: int,
                 nheads: int, headdim: int, mimo_rank: int = 1,
                 cond_dim: int = 128):
        super().__init__()
        self.d_state = d_state
        self.d_model = d_model
        self.d_inner = d_inner
        self.nheads = nheads
        self.headdim = headdim
        self.mimo_rank = mimo_rank
        self.cond_dim = cond_dim
        self.num_rope_angles = d_state // 2

        # in_proj output dimension
        self.in_proj_out = (d_inner * 2                    # z + x
                            + nheads * mimo_rank * d_state * 2  # B + C
                            + nheads * 3                    # dd_dt + dd_A + trap
                            + self.num_rope_angles)         # angles

        # Single fused input projection
        self.in_proj = nn.Linear(d_model, self.in_proj_out, bias=False)

        # Δ bias (per-head)
        self.dt_bias = nn.Parameter(torch.zeros(nheads))

        # B and C biases (per-head, per-rank, per-state)
        self.B_bias = nn.Parameter(torch.zeros(nheads, mimo_rank, d_state))
        self.C_bias = nn.Parameter(torch.zeros(nheads, mimo_rank, d_state))

        # B and C RMSNorm weights (learned scale per state dimension)
        self.B_norm_weight = nn.Parameter(torch.ones(d_state))
        self.C_norm_weight = nn.Parameter(torch.ones(d_state))

        # MIMO projection matrices
        self.mimo_x = nn.Parameter(torch.zeros(nheads, mimo_rank, headdim))
        self.mimo_z = nn.Parameter(torch.zeros(nheads, mimo_rank, headdim))
        self.mimo_o = nn.Parameter(torch.zeros(nheads, mimo_rank, headdim))

        # D skip connection (per-head scalar)
        self.D_proj = nn.Parameter(torch.zeros(nheads))

        # Output projection: d_inner -> d_model
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

        # FiLM conditioning projections
        self.film_proj = nn.Linear(cond_dim, 2 * d_model, bias=False)
        self.dt_film_proj = nn.Linear(cond_dim, 2 * nheads, bias=False)

    def forward(self, x: torch.Tensor, cond: torch.Tensor = None) -> torch.Tensor:
        """
        x: (batch, seq_len, d_model)
        cond: (batch, d_cond) or None — FiLM conditioning vector
        Per-sample recurrence matching C++ SSMBlock::process_sample().
        """
        batch, seq_len, _ = x.shape
        device = x.device

        # Step 0: FiLM conditioning
        if cond is not None and cond.any():
            # film_proj: (d_cond, 2*d_model) → cond @ film_proj^T → (batch, 2*d_model)
            film_out = F.linear(cond, self.film_proj.weight, None)  # (B, 2*d_model)
            gamma = film_out[:, 0::2]  # (B, d_model)
            beta  = film_out[:, 1::2]  # (B, d_model)
            x_film = x * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
        else:
            x_film = x

        # Step 1: in_proj — single matmul
        proj = self.in_proj(x_film)  # (B, S, in_proj_out)

        # Split components (same as before)
        idx = 0
        z = proj[..., idx:idx + self.d_inner]; idx += self.d_inner
        x_proj = proj[..., idx:idx + self.d_inner]; idx += self.d_inner
        bc_size = self.nheads * self.mimo_rank * self.d_state
        B_raw = proj[..., idx:idx + bc_size]; idx += bc_size
        C_raw = proj[..., idx:idx + bc_size]; idx += bc_size
        dd_dt = proj[..., idx:idx + self.nheads]; idx += self.nheads
        dd_A = proj[..., idx:idx + self.nheads]; idx += self.nheads
        trap = proj[..., idx:idx + self.nheads]; idx += self.nheads
        angles = proj[..., idx:idx + self.num_rope_angles]

        # Reshape B, C and clone so that in-place RoPE updates are safe
        # during PyTorch training (backward through in-place views of in_proj
        # would otherwise raise an error).
        B = B_raw.view(batch, seq_len, self.nheads, self.mimo_rank, self.d_state).clone()
        C = C_raw.view(batch, seq_len, self.nheads, self.mimo_rank, self.d_state).clone()

        # Step 2: Preprocess A and dt (with A clamping)
        A_floor = 1.0
        A_t = -F.softplus(dd_A)  # (B, S, nheads)
        A_t = torch.clamp(A_t, max=-A_floor)  # C++: clamp max to -A_floor
        dt = F.softplus(dd_dt + self.dt_bias)  # (B, S, nheads)

        # Step 3: RMSNorm on B and C
        B_rms = B.pow(2).mean(dim=-1, keepdim=True).add(1e-5).rsqrt()
        B = B * B_rms * self.B_norm_weight.view(1, 1, 1, 1, -1) + self.B_bias
        C_rms = C.pow(2).mean(dim=-1, keepdim=True).add(1e-5).rsqrt()
        C = C * C_rms * self.C_norm_weight.view(1, 1, 1, 1, -1) + self.C_bias

        # Step 4: RoPE on B and C (vectorized, no in-place ops for training)
        # angles: (B, S, num_rope_angles); cumulative angle per head.
        angle_state = torch.cumsum(angles, dim=1)                 # (B, S, nra)
        angle_state = torch.fmod(angle_state, 2.0 * math.pi)

        # Reshape so the last dim is (real, imag) pairs.
        # B/C: (B, S, nheads, mimo_rank, d_state)
        B_pairs = B.view(batch, seq_len, self.nheads, self.mimo_rank,
                         self.num_rope_angles, 2)
        C_pairs = C.view(batch, seq_len, self.nheads, self.mimo_rank,
                         self.num_rope_angles, 2)

        # Broadcast angles across the mimo_rank and pair dimensions.
        # angle_state: (B, S, nra)  ->  (B, S, 1, 1, nra, 1) to broadcast over head/mimo/pair
        cos_a = torch.cos(angle_state).unsqueeze(2).unsqueeze(3).unsqueeze(5)
        sin_a = torch.sin(angle_state).unsqueeze(2).unsqueeze(3).unsqueeze(5)

        # Keep the pair dim explicit for broadcasting: (B,S,nh,mr,nra,1)
        B0, B1 = B_pairs[..., :1], B_pairs[..., 1:]
        C0, C1 = C_pairs[..., :1], C_pairs[..., 1:]

        B_rot = torch.cat([
            B0 * cos_a - B1 * sin_a,
            B0 * sin_a + B1 * cos_a,
        ], dim=-1)
        C_rot = torch.cat([
            C0 * cos_a + C1 * sin_a,
           -C0 * sin_a + C1 * cos_a,
        ], dim=-1)

        B = B_rot.view(batch, seq_len, self.nheads, self.mimo_rank, self.d_state)
        C = C_rot.view(batch, seq_len, self.nheads, self.mimo_rank, self.d_state)

        # State: (B, nheads, headdim, d_state)
        h = torch.zeros(batch, self.nheads, self.headdim, self.d_state,
                        device=device)

        y_heads = []
        for t in range(seq_len):
            at = A_t[:, t, :]   # (B, nheads)
            dtt = dt[:, t, :]
            xt = x_proj[:, t, :]
            xt_h = xt.view(batch, self.nheads, self.headdim)

            # Step 5: Trapezoidal discretization
            half_a = at * dtt / 2.0
            inv_denom = 1.0 / (1.0 - half_a)
            A_bar = (1.0 + half_a) * inv_denom
            B_bar_scale = dtt * inv_denom

            B_t = B[:, t, :, 0, :]  # rank 0
            C_t = C[:, t, :, 0, :]

            h = (A_bar.unsqueeze(2).unsqueeze(3) * h +
                 B_bar_scale.unsqueeze(2).unsqueeze(3) *
                 B_t.unsqueeze(2) * xt_h.unsqueeze(3))

            y_t = (C_t.unsqueeze(2) * h).sum(dim=-1)  # (B, nheads, headdim)

            # Step 6: SiLU z-gating
            z_t = z[:, t, :].view(batch, self.nheads, self.headdim)
            z_gated = z_t * torch.sigmoid(z_t)
            y_t = y_t * z_gated

            y_heads.append(y_t.view(batch, self.d_inner))

        y_concat = torch.stack(y_heads, dim=1)

        # Step 7: Output projection + D skip + residual
        out = self.out_proj(y_concat)

        # D skip: per-head scalar * x_proj per head
        x_h = x_proj.view(batch, seq_len, self.nheads, self.headdim)
        d_skip = (self.D_proj.view(1, 1, -1, 1)
                  * x_h).view(batch, seq_len, self.nheads * self.headdim)
        # Match d_model: truncate or pad as needed
        if d_skip.size(-1) < self.d_model:
            d_skip = F.pad(d_skip, (0, self.d_model - d_skip.size(-1)))
        elif d_skip.size(-1) > self.d_model:
            d_skip = d_skip[..., :self.d_model]
        out = out + d_skip

        # Residual from FiLM'd input
        out = out + x_film

        return out


class Mamba3Model(nn.Module):
    def __init__(self, d_state: int, d_model: int, d_inner: int,
                 nheads: int, headdim: int, num_layers: int,
                 mimo_rank: int = 1, cond_dim: int = 128):
        super().__init__()
        self.layers = nn.ModuleList(
            [Mamba3Block(d_state, d_model, d_inner, nheads, headdim,
                        mimo_rank, cond_dim)
             for _ in range(num_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


# ---------------------------------------------------------------------------
# Flat-binary format constants (Mamba-3, version 2)
# ---------------------------------------------------------------------------

WEIGHT_MAGIC = 0x4D424D41
WEIGHT_VERSION = 2  # Mamba-3 format
HEADER_FMT = '<12IQQ'
HEADER_SIZE = 64

# Compile-time layout constants (must match C++ config.h).
# The weight file uses these fixed-size slots regardless of actual model dims.
MAX_MIMO_RANK = 4
MAX_NUM_ROPE_ANGLES = 16


def floats_per_layer(ds: int, dm: int, di: int, nh: int, mr: int,
                     hd: int, cd: int) -> int:
    """Compute number of float32 elements per layer (MAX_*-padded layout)."""
    nra = MAX_NUM_ROPE_ANGLES  # padded to MAX for file layout
    in_proj_out = di * 2 + nh * MAX_MIMO_RANK * ds * 2 + nh * 3 + nra
    return (in_proj_out * dm +       # in_proj_w
            nh +                      # dt_bias
            nh * MAX_MIMO_RANK * ds + # B_bias
            nh * MAX_MIMO_RANK * ds + # C_bias
            ds +                      # B_norm_weight
            ds +                      # C_norm_weight
            nh * MAX_MIMO_RANK * hd + # mimo_x
            nh * MAX_MIMO_RANK * hd + # mimo_z
            nh * MAX_MIMO_RANK * hd + # mimo_o
            nh +                      # D_proj
            dm * di +                 # out_proj_w
            2 * dm * cd +             # film_proj_w
            2 * nh * cd)              # dt_film_proj_w


def make_header(ds: int, dm: int, di: int, nh: int, hd: int,
                nl: int, cd: int, wds: int) -> bytes:
    """Build 64-byte WeightFileHeader (version 2, 12IQQ layout)."""
    data = struct.pack(HEADER_FMT,
                       WEIGHT_MAGIC, WEIGHT_VERSION,
                       ds, dm, di, nh, nl, hd, cd,
                       0, 0, 0,  # quantization, q_metadata_size, flags
                       wds, 0)   # weight_data_size, reserved
    assert len(data) == HEADER_SIZE
    return data


def extract_layer_weights(block: Mamba3Block) -> bytes:
    """Extract per-layer weights in MAX_*-padded C++ WeightLoader order (13 arrays)."""
    ds, dm, di = block.d_state, block.d_model, block.d_inner
    nh, mr, hd = block.nheads, block.mimo_rank, block.headdim
    cd = block.cond_dim
    nra = MAX_NUM_ROPE_ANGLES  # file-layout stride

    # Pad model arrays to MAX_* layout (insert zero-filled slots)
    import numpy as np

    # 1. in_proj_w: pad to MAX_MIMO_RANK and MAX_NUM_ROPE_ANGLES layout
    in_proj_model = block.in_proj.weight.detach().cpu().numpy().astype(np.float32)
    in_proj_out_file = di * 2 + nh * MAX_MIMO_RANK * ds * 2 + nh * 3 + nra
    in_proj_file = np.zeros((in_proj_out_file, dm), dtype=np.float32)

    # z and x portions (same sizes)
    in_proj_file[:di * 2, :] = in_proj_model[:di * 2, :]

    # B portion: expand from mimo_rank to MAX_MIMO_RANK stride
    b_start_model = di * 2
    b_start_file = di * 2
    for h_idx in range(nh):
        src = b_start_model + h_idx * mr * ds
        dst = b_start_file + h_idx * MAX_MIMO_RANK * ds
        in_proj_file[dst:dst + mr * ds, :] = in_proj_model[src:src + mr * ds, :]

    # C portion: expand from mimo_rank to MAX_MIMO_RANK stride
    c_start_model = b_start_model + nh * mr * ds
    c_start_file = b_start_file + nh * MAX_MIMO_RANK * ds
    for h_idx in range(nh):
        src = c_start_model + h_idx * mr * ds
        dst = c_start_file + h_idx * MAX_MIMO_RANK * ds
        in_proj_file[dst:dst + mr * ds, :] = in_proj_model[src:src + mr * ds, :]

    # dd_dt, dd_A, trap (same sizes, contiguous after padded B/C)
    rest_src = c_start_model + nh * mr * ds
    rest_dst = c_start_file + nh * MAX_MIMO_RANK * ds
    rest_size = nh * 3  # dd_dt + dd_A + trap
    in_proj_file[rest_dst:rest_dst + rest_size, :] = in_proj_model[rest_src:rest_src + rest_size, :]

    # angles portion: pad to MAX_NUM_ROPE_ANGLES
    angles_src = rest_src + rest_size
    angles_dst = rest_dst + rest_size
    model_angles = ds // 2
    in_proj_file[angles_dst:angles_dst + model_angles, :] = in_proj_model[angles_src:angles_src + model_angles, :]

    parts = [in_proj_file.tobytes()]

    # 2. dt_bias: (nheads,)
    parts.append(block.dt_bias.detach().cpu().numpy().astype(np.float32).tobytes())

    # 3-4. B_bias, C_bias: pad to MAX_MIMO_RANK
    for arr_name in ['B_bias', 'C_bias']:
        model_arr = getattr(block, arr_name).detach().cpu().numpy().astype(np.float32)
        padded = np.zeros((nh, MAX_MIMO_RANK, ds), dtype=np.float32)
        padded[:, :mr, :] = model_arr
        parts.append(padded.tobytes())

    # 5. B_norm_weight: (d_state,)
    parts.append(block.B_norm_weight.detach().cpu().numpy().astype(np.float32).tobytes())
    # 6. C_norm_weight: (d_state,)
    parts.append(block.C_norm_weight.detach().cpu().numpy().astype(np.float32).tobytes())

    # 7-9. mimo_x, mimo_z, mimo_o: pad to MAX_MIMO_RANK
    for arr_name in ['mimo_x', 'mimo_z', 'mimo_o']:
        model_arr = getattr(block, arr_name).detach().cpu().numpy().astype(np.float32)
        padded = np.zeros((nh, MAX_MIMO_RANK, hd), dtype=np.float32)
        padded[:, :mr, :] = model_arr
        parts.append(padded.tobytes())

    # 10. D_proj: (nheads,)
    parts.append(block.D_proj.detach().cpu().numpy().astype(np.float32).tobytes())

    # 11. out_proj_w: (d_model, d_inner)
    parts.append(block.out_proj.weight.detach().cpu().numpy().astype(np.float32).tobytes())
    # 12. film_proj_w: (2*d_model, d_cond)
    parts.append(block.film_proj.weight.detach().cpu().numpy().astype(np.float32).tobytes())
    # 13. dt_film_proj_w: (2*nheads, d_cond)
    parts.append(block.dt_film_proj.weight.detach().cpu().numpy().astype(np.float32).tobytes())

    return b''.join(parts)


def export_model(model: Mamba3Model, output_path: str,
                 cond_dim: int = 128, random_init: bool = False):
    b = model.layers[0]
    ds, dm, di = b.d_state, b.d_model, b.d_inner
    nh, mr, hd = b.nheads, b.mimo_rank, b.headdim
    nl = len(model.layers)

    if random_init:
        _random_init(model)

    fpl = floats_per_layer(ds, dm, di, nh, mr, hd, cond_dim)
    wds = fpl * nl * 4

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'wb') as f:
        f.write(make_header(ds, dm, di, nh, hd, nl, cond_dim, wds))
        for layer in model.layers:
            f.write(extract_layer_weights(layer))

    print(f"Exported {nl} layers "
          f"(ds={ds} dm={dm} di={di} nh={nh} hd={hd}) -> {output_path}"
          f"  [{wds} bytes]")


def _random_init(model: Mamba3Model, std: float = 0.02):
    for b in model.layers:
        nn.init.normal_(b.in_proj.weight, std=std)
        nn.init.normal_(b.dt_bias, std=0.1)
        nn.init.normal_(b.B_bias, std=std)
        nn.init.normal_(b.C_bias, std=std)
        nn.init.ones_(b.B_norm_weight)
        nn.init.ones_(b.C_norm_weight)
        nn.init.normal_(b.mimo_x, std=std)
        nn.init.normal_(b.mimo_z, std=std)
        nn.init.normal_(b.mimo_o, std=std)
        nn.init.normal_(b.D_proj, std=0.1)
        nn.init.normal_(b.out_proj.weight, std=std)
        nn.init.normal_(b.film_proj.weight, std=std)
        nn.init.normal_(b.dt_film_proj.weight, std=std)


def generate_config_h(ds: int, dm: int, di: int, nh: int, hd: int,
                      nl: int, cd: int, mr: int, path: Optional[str] = None) -> str:
    nra = ds // 2
    content = f'''#ifndef CONFIG_H
#define CONFIG_H

// Auto-generated by export_mamba_weights.py (Mamba-3)
#define MAX_D_STATE {ds}
#define MAX_D_MODEL {dm}
#define MAX_D_INNER {di}
#define MAX_HEADDIM {hd}
#define MAX_NHEADS {nh}
#define MAX_MIMO_RANK {mr}
#define MAX_NUM_BC_HEADS 1
#define MAX_NUM_ROPE_ANGLES {nra}
#define MAX_CHUNK_SIZE 128
#define MAX_LAYERS {nl}
#define MAX_BLOCK_SIZE 512
#define MAX_PARAM_ATOMS 256
#define COND_DIM {cd}
#define SSM_STATE_NORM_INTERVAL 4096

#define VERSION_MAJOR 0
#define VERSION_MINOR 3
#define VERSION_PATCH 0

#endif
'''
    if path:
        with open(path, 'w') as f:
            f.write(content)
        print(f"Generated {path}")
    return content


# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Export Mamba-3 weights")
    p.add_argument('--state-dim', type=int, default=16)
    p.add_argument('--model-dim', type=int, default=16)
    p.add_argument('--inner-dim', type=int, default=32)
    p.add_argument('--nheads', type=int, default=2)
    p.add_argument('--headdim', type=int, default=None)
    p.add_argument('--layers', type=int, default=1)
    p.add_argument('--cond-dim', type=int, default=128)
    p.add_argument('--output', default='models/test_model.bin')
    p.add_argument('--config-h', default=None)
    p.add_argument('--load-checkpoint', default=None)
    p.add_argument('--random-init', action='store_true')
    args = p.parse_args()

    headdim = args.headdim if args.headdim else args.inner_dim // args.nheads
    if args.inner_dim % args.nheads != 0:
        print(f"WARNING: d_inner ({args.inner_dim}) not divisible by "
              f"nheads ({args.nheads})")
        headdim = args.inner_dim // args.nheads

    model = Mamba3Model(args.state_dim, args.model_dim,
                        args.inner_dim, args.nheads, headdim,
                        args.layers, cond_dim=args.cond_dim)

    if args.load_checkpoint and os.path.exists(args.load_checkpoint):
        model.load_state_dict(torch.load(args.load_checkpoint,
                                         map_location='cpu'))
        print(f"Loaded checkpoint: {args.load_checkpoint}")
    elif args.random_init:
        print("Random-init weights")
    else:
        print("WARNING: zero-initialized (will produce silence)")

    export_model(model, args.output, args.cond_dim, args.random_init)

    if args.config_h:
        generate_config_h(args.state_dim, args.model_dim,
                          args.inner_dim, args.nheads, headdim,
                          args.layers, args.cond_dim, MAX_MIMO_RANK,
                          args.config_h)


if __name__ == '__main__':
    main()
