#!/usr/bin/env python3
"""
Mamba-SSM Weight Export Tool

Exports a trained PyTorch Mamba-SSM model to the flat-binary weight format
consumed by the C++ WeightLoader.  Also generates config.h with matching
MAX_* constants.

Usage:
    python scripts/export_mamba_weights.py --state-dim 16 --model-dim 16 \\
        --inner-dim 32 --conv-dim 4 --layers 2 --random-init \\
        --output models/test_model.bin --config-h src/config.h
"""

import argparse
import struct
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Reference Mamba block (PyTorch, matches C++ SSMBlock architecture)
# ---------------------------------------------------------------------------

class MambaBlock(nn.Module):
    """Single Mamba-SSM layer."""

    def __init__(self, d_state: int, d_model: int, d_inner: int, d_conv: int):
        super().__init__()
        self.d_state = d_state
        self.d_model = d_model
        self.d_inner = d_inner
        self.d_conv = d_conv

        # Input projection: d_model -> d_inner
        self.W_in = nn.Linear(d_model, d_inner, bias=True)

        # Causal depthwise conv1d
        self.conv1d = nn.Conv1d(d_inner, d_inner, d_conv,
                                padding=d_conv - 1, groups=d_inner, bias=True)

        # SSM parameters
        self.A_log = nn.Parameter(torch.zeros(d_state))
        self.D_proj = nn.Parameter(torch.zeros(d_state))
        self.dt_bias = nn.Parameter(torch.zeros(d_state))
        self.dt_proj_w = nn.Parameter(torch.zeros(d_state, d_inner))
        self.B_proj_w = nn.Parameter(torch.zeros(d_inner, d_state))
        self.C_proj_w = nn.Parameter(torch.zeros(d_inner, d_state))

        # Conditioning projections (d_state, cond_dim)
        self.cond_proj_dt = nn.Parameter(torch.zeros(d_state, 16))
        self.cond_proj_B = nn.Parameter(torch.zeros(d_state, 16))

        # Output projections
        self.W_out = nn.Linear(d_inner, d_model, bias=False)
        self.W_ssm = nn.Linear(d_state, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        x = self.W_in(x)
        x = x.transpose(1, 2)
        x = self.conv1d(x)[:, :, :seq_len]
        x = F.silu(x)
        x = x.transpose(1, 2)

        A = -torch.exp(self.A_log)
        dt = F.softplus(torch.matmul(x, self.dt_proj_w.t()) + self.dt_bias)
        B = torch.matmul(x, self.B_proj_w)
        C = torch.matmul(x, self.C_proj_w)

        h = torch.zeros(batch, self.d_state, device=x.device)
        ys = []
        for t in range(seq_len):
            d = dt[:, t, :]
            b = B[:, t, :]
            c = C[:, t, :]
            u = x[:, t, :]
            A_bar = torch.exp(A * d)
            mask = A.abs() > 1e-6
            B_bar = torch.where(
                mask, (A_bar - 1.0) / A * b, d * b)
            h = A_bar * h + B_bar * u[:, :self.d_state]
            ys.append((c * h).sum(dim=-1))

        y_ssm = torch.stack(ys, dim=1)
        y = self.W_out(x) + y_ssm.unsqueeze(-1).expand(-1, -1, self.d_model)
        return y


class SimpleMamba(nn.Module):
    def __init__(self, d_state: int, d_model: int, d_inner: int, d_conv: int,
                 num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList(
            [MambaBlock(d_state, d_model, d_inner, d_conv)
             for _ in range(num_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


# ---------------------------------------------------------------------------
# Flat-binary format constants
# ---------------------------------------------------------------------------

WEIGHT_MAGIC = 0x4D424D41
WEIGHT_VERSION = 1
HEADER_FMT = '<9I4I4x Q'
HEADER_SIZE = 64


def floats_per_layer(ds: int, dm: int, di: int, dc: int, cd: int = 16) -> int:
    return (3 * ds + 3 * ds * di + dc * di + 2 * di +
            di * dm + dm * di + dm * ds + 2 * ds * cd)


def make_header(ds: int, dm: int, di: int, dc: int,
                nl: int, cd: int, wds: int) -> bytes:
    data = struct.pack(HEADER_FMT,
                       WEIGHT_MAGIC, WEIGHT_VERSION,
                       ds, dm, di, dc, nl, di // dm, cd,
                       0, 0, 0, 0, wds)
    assert len(data) == HEADER_SIZE
    return data


def extract_layer_weights(block: MambaBlock) -> bytes:
    """Extract per-layer weights in the exact C++ WeightLoader order."""
    parts = []
    # 1. A_log: (d_state,)
    parts.append(block.A_log.detach().cpu().numpy().tobytes())
    # 2. D_proj: (d_state,)
    parts.append(block.D_proj.detach().cpu().numpy().tobytes())
    # 3. dt_bias: (d_state,)
    parts.append(block.dt_bias.detach().cpu().numpy().tobytes())
    # 4. dt_proj_w: (d_state, d_inner)
    parts.append(block.dt_proj_w.detach().cpu().numpy().tobytes())
    # 5. B_proj_w: (d_inner, d_state)
    parts.append(block.B_proj_w.detach().cpu().numpy().tobytes())
    # 6. C_proj_w: (d_inner, d_state)
    parts.append(block.C_proj_w.detach().cpu().numpy().tobytes())
    # 7. conv1d_w: (d_conv, d_inner) from depthwise weight (d_inner,1,d_conv)
    cw = block.conv1d.weight.detach().cpu()
    cw = cw.view(block.d_inner, block.d_conv).t().contiguous()
    parts.append(cw.numpy().tobytes())
    # 8. conv1d_b: (d_inner,)
    parts.append(block.conv1d.bias.detach().cpu().numpy().tobytes())
    # 9. W_in: (d_inner, d_model)
    parts.append(block.W_in.weight.detach().cpu().numpy().tobytes())
    # 10. W_out: (d_model, d_inner)
    parts.append(block.W_out.weight.detach().cpu().numpy().tobytes())
    # 11. W_ssm: (d_model, d_state)
    parts.append(block.W_ssm.weight.detach().cpu().numpy().tobytes())
    # 12. b_in: (d_inner,)
    parts.append(block.W_in.bias.detach().cpu().numpy().tobytes())
    # 13. cond_proj_dt: (d_state, cond_dim)
    parts.append(block.cond_proj_dt.detach().cpu().numpy().tobytes())
    # 14. cond_proj_B: (d_state, cond_dim)
    parts.append(block.cond_proj_B.detach().cpu().numpy().tobytes())
    return b''.join(parts)


def export_model(model: SimpleMamba, output_path: str,
                 cond_dim: int = 16, random_init: bool = False):
    b = model.layers[0]
    ds, dm, di, dc = b.d_state, b.d_model, b.d_inner, b.d_conv
    nl = len(model.layers)

    if random_init:
        _random_init(model)

    fpl = floats_per_layer(ds, dm, di, dc)
    wds = fpl * nl * 4

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'wb') as f:
        f.write(make_header(ds, dm, di, dc, nl, cond_dim, wds))
        for layer in model.layers:
            f.write(extract_layer_weights(layer))

    print(f"Exported {nl} layers "
          f"(ds={ds} dm={dm} di={di} dc={dc}) -> {output_path}"
          f"  [{wds} bytes]")


def _random_init(model: SimpleMamba, std: float = 0.02):
    for b in model.layers:
        nn.init.normal_(b.W_in.weight, std=std)
        nn.init.zeros_(b.W_in.bias)
        nn.init.normal_(b.conv1d.weight, std=std)
        nn.init.zeros_(b.conv1d.bias)
        nn.init.normal_(b.A_log, std=0.1)
        nn.init.normal_(b.D_proj, std=0.1)
        nn.init.normal_(b.dt_bias, std=0.1)
        nn.init.normal_(b.dt_proj_w, std=std)
        nn.init.normal_(b.B_proj_w, std=std)
        nn.init.normal_(b.C_proj_w, std=std)
        nn.init.normal_(b.W_out.weight, std=std)
        nn.init.normal_(b.W_ssm.weight, std=std)
        nn.init.normal_(b.cond_proj_dt, std=std)
        nn.init.normal_(b.cond_proj_B, std=std)


def generate_config_h(ds: int, dm: int, di: int, dc: int,
                      nl: int, cd: int, path: Optional[str] = None) -> str:
    content = f'''#ifndef CONFIG_H
#define CONFIG_H

#define MAX_D_STATE {ds}
#define MAX_D_MODEL {dm}
#define MAX_D_INNER {di}
#define MAX_D_CONV {dc}
#define MAX_LAYERS {nl}
#define MAX_BLOCK_SIZE 512
#define MAX_PARAM_ATOMS 256
#define COND_DIM {cd}
#define SSM_STATE_NORM_INTERVAL 4096

#define VERSION_MAJOR 0
#define VERSION_MINOR 1
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
    p = argparse.ArgumentParser(description="Export Mamba-SSM weights")
    p.add_argument('--state-dim', type=int, default=16)
    p.add_argument('--model-dim', type=int, default=16)
    p.add_argument('--inner-dim', type=int, default=32)
    p.add_argument('--conv-dim', type=int, default=4)
    p.add_argument('--layers', type=int, default=1)
    p.add_argument('--cond-dim', type=int, default=16)
    p.add_argument('--output', default='models/test_model.bin')
    p.add_argument('--config-h', default=None)
    p.add_argument('--load-checkpoint', default=None)
    p.add_argument('--random-init', action='store_true')
    args = p.parse_args()

    model = SimpleMamba(args.state_dim, args.model_dim,
                        args.inner_dim, args.conv_dim, args.layers)

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
                          args.inner_dim, args.conv_dim,
                          args.layers, args.cond_dim, args.config_h)


if __name__ == '__main__':
    main()
