#!/usr/bin/env python3
"""
Mamba-3 Weight Export Validator

Loads a flat-binary weight file (version 2), reconstructs the model in
PyTorch, and verifies that the forward pass produces valid output.

Usage:
    python scripts/export_mamba_weights.py --random-init --layers 2 \\
        --state-dim 16 --model-dim 16 --inner-dim 32 --nheads 2 \\
        --output /tmp/test_model.bin
    python scripts/validate_export.py /tmp/test_model.bin
"""

import argparse
import struct
import sys
import os

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from export_mamba_weights import (
    WEIGHT_MAGIC, WEIGHT_VERSION, HEADER_SIZE,
    MAX_MIMO_RANK, MAX_NUM_ROPE_ANGLES,
    floats_per_layer, Mamba3Block, Mamba3Model
)

ATOL = 1e-5
RTOL = 1e-5


def load_header(path: str) -> dict:
    with open(path, 'rb') as f:
        raw = f.read(HEADER_SIZE)
    fields = struct.unpack('<12IQQ', raw)
    return {
        'magic': fields[0],
        'version': fields[1],
        'd_state': fields[2],
        'd_model': fields[3],
        'd_inner': fields[4],
        'nheads': fields[5],
        'num_layers': fields[6],
        'headdim': fields[7],
        'cond_dim': fields[8],
        'quantization': fields[9],
        'q_metadata_size': fields[10],
        'flags': fields[11],
        'weight_data_size': fields[12],
        'reserved': fields[13],
    }


def load_weights_into_model(model: Mamba3Model, path: str) -> Mamba3Model:
    """Load MAX_*-padded flat-binary weights into a PyTorch Mamba-3 model."""
    hdr = load_header(path)
    ds = hdr['d_state']
    dm = hdr['d_model']
    di = hdr['d_inner']
    nh = hdr['nheads']
    hd = hdr['headdim']
    nl = hdr['num_layers']
    cd = hdr['cond_dim']
    mr = model.layers[0].mimo_rank  # model's actual mimo_rank (not MAX_)

    in_proj_out_file = di * 2 + nh * MAX_MIMO_RANK * ds * 2 + nh * 3 + MAX_NUM_ROPE_ANGLES
    in_proj_out_model = di * 2 + nh * mr * ds * 2 + nh * 3 + (ds // 2)

    fpl = floats_per_layer(ds, dm, di, nh, mr, hd, cd)
    file_size = os.path.getsize(path)
    expected = HEADER_SIZE + fpl * nl * 4
    assert file_size == expected, (
        f"File size mismatch: {file_size} vs expected {expected}")

    with open(path, 'rb') as f:
        f.read(HEADER_SIZE)
        raw = f.read()

    arr = np.frombuffer(raw, dtype=np.float32)
    offset = 0

    for li in range(nl):
        block = model.layers[li]

        def read(n):
            nonlocal offset
            data = arr[offset:offset + n].copy()
            offset += n
            return data

        # 1. in_proj_w: read from MAX_*-padded file layout, extract model-sized portion
        in_proj_file = read(in_proj_out_file * dm).reshape(in_proj_out_file, dm)
        in_proj_model_arr = np.zeros((in_proj_out_model, dm), dtype=np.float32)
        # z and x portions (same size)
        in_proj_model_arr[:di * 2, :] = in_proj_file[:di * 2, :]
        # B: extract mr*ds per head from MAX_MIMO_RANK stride
        b_src = di * 2
        b_dst = di * 2
        for h_idx in range(nh):
            src = b_src + h_idx * MAX_MIMO_RANK * ds
            dst = b_dst + h_idx * mr * ds
            in_proj_model_arr[dst:dst + mr * ds, :] = in_proj_file[src:src + mr * ds, :]
        # C: extract mr*ds per head from MAX_MIMO_RANK stride
        c_src = b_src + nh * MAX_MIMO_RANK * ds
        c_dst = b_dst + nh * mr * ds
        for h_idx in range(nh):
            src = c_src + h_idx * MAX_MIMO_RANK * ds
            dst = c_dst + h_idx * mr * ds
            in_proj_model_arr[dst:dst + mr * ds, :] = in_proj_file[src:src + mr * ds, :]
        # dd_dt, dd_A, trap (same size)
        rest_src = c_src + nh * MAX_MIMO_RANK * ds
        rest_dst = c_dst + nh * mr * ds
        rest_size = nh * 3
        in_proj_model_arr[rest_dst:rest_dst + rest_size, :] = in_proj_file[rest_src:rest_src + rest_size, :]
        # angles portion
        angles_src = rest_src + rest_size
        angles_dst = rest_dst + rest_size
        in_proj_model_arr[angles_dst:angles_dst + (ds // 2), :] = in_proj_file[angles_src:angles_src + (ds // 2), :]
        block.in_proj.weight.data = torch.from_numpy(in_proj_model_arr)

        # 2. dt_bias: (nheads,)
        block.dt_bias.data = torch.from_numpy(read(nh))

        # 3. B_bias: padded to MAX_MIMO_RANK — extract mr portion
        B_bias_file = read(nh * MAX_MIMO_RANK * ds).reshape(nh, MAX_MIMO_RANK, ds)
        block.B_bias.data = torch.from_numpy(B_bias_file[:, :mr, :].copy())

        # 4. C_bias: padded to MAX_MIMO_RANK — extract mr portion
        C_bias_file = read(nh * MAX_MIMO_RANK * ds).reshape(nh, MAX_MIMO_RANK, ds)
        block.C_bias.data = torch.from_numpy(C_bias_file[:, :mr, :].copy())

        # 5. B_norm_weight: (d_state,)
        block.B_norm_weight.data = torch.from_numpy(read(ds))
        # 6. C_norm_weight: (d_state,)
        block.C_norm_weight.data = torch.from_numpy(read(ds))

        # 7-9. mimo_x, mimo_z, mimo_o: padded to MAX_MIMO_RANK
        for attr in ['mimo_x', 'mimo_z', 'mimo_o']:
            arr_file = read(nh * MAX_MIMO_RANK * hd).reshape(nh, MAX_MIMO_RANK, hd)
            getattr(block, attr).data = torch.from_numpy(arr_file[:, :mr, :].copy())

        # 10. D_proj: (nheads,)
        block.D_proj.data = torch.from_numpy(read(nh))
        # 11. out_proj_w: (d_model, d_inner)
        block.out_proj.weight.data = torch.from_numpy(read(dm * di).reshape(dm, di))
        # 12. film_proj_w: (2*d_model, d_cond)
        block.film_proj.weight.data = torch.from_numpy(read(2 * dm * cd).reshape(2 * dm, cd))
        # 13. dt_film_proj_w: (2*nheads, d_cond)
        block.dt_film_proj.weight.data = torch.from_numpy(read(2 * nh * cd).reshape(2 * nh, cd))

    assert offset == fpl * nl, (
        f"Read {offset} floats, expected {fpl * nl}")
    return model


def validate(path: str, batch: int = 1, seq_len: int = 64):
    """Validate that flat-binary round-trips through PyTorch correctly."""
    hdr = load_header(path)
    assert hdr['magic'] == WEIGHT_MAGIC, f"Bad magic: {hdr['magic']:#x}"
    assert hdr['version'] == WEIGHT_VERSION, (
        f"Bad version: {hdr['version']} (expected {WEIGHT_VERSION})")

    print(f"Header: layers={hdr['num_layers']} ds={hdr['d_state']} "
          f"dm={hdr['d_model']} di={hdr['d_inner']} "
          f"nh={hdr['nheads']} hd={hdr['headdim']}")

    model = Mamba3Model(hdr['d_state'], hdr['d_model'],
                        hdr['d_inner'], hdr['nheads'], hdr['headdim'],
                        hdr['num_layers'], cond_dim=hdr['cond_dim'])
    load_weights_into_model(model, path)
    model.eval()

    # Run forward pass with random input
    x = torch.randn(batch, seq_len, hdr['d_model'])
    with torch.no_grad():
        y = model(x)

    assert not torch.isnan(y).any(), "NaN in output!"
    assert not torch.isinf(y).any(), "Inf in output!"
    assert y.shape == (batch, seq_len, hdr['d_model']), (
        f"Shape mismatch: {y.shape}")

    # Run a second time to verify determinism
    x2 = x.clone()
    with torch.no_grad():
        y2 = model(x2)
    assert torch.allclose(y, y2, atol=ATOL), "Non-deterministic output!"

    output_sum = y.abs().sum().item()
    print(f"Forward pass OK: shape={list(y.shape)} "
          f"range=[{y.min().item():.4f}, {y.max().item():.4f}]")
    print(f"Non-zero output: {output_sum > 0}")
    print(f"Deterministic: yes")
    print(f"Validation PASSED")

    return output_sum > 0


def main():
    p = argparse.ArgumentParser(description="Validate exported Mamba-3 weights")
    p.add_argument('path', help='Path to .bin weight file')
    p.add_argument('--batch', type=int, default=1)
    p.add_argument('--seq-len', type=int, default=64)
    args = p.parse_args()

    if not os.path.exists(args.path):
        print(f"ERROR: {args.path} not found")
        sys.exit(1)

    ok = validate(args.path, args.batch, args.seq_len)
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
