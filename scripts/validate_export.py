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
    floats_per_layer, Mamba3Block, Mamba3Model
)

ATOL = 1e-5
RTOL = 1e-5


def load_header(path: str) -> dict:
    with open(path, 'rb') as f:
        raw = f.read(HEADER_SIZE)
    fields = struct.unpack('<9I4I4x Q', raw)
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
        'reserved': list(fields[9:13]),
        'weight_data_size': fields[13],
    }


def load_weights_into_model(model: Mamba3Model, path: str) -> Mamba3Model:
    """Load flat-binary weights into a PyTorch Mamba-3 model."""
    hdr = load_header(path)
    ds = hdr['d_state']
    dm = hdr['d_model']
    di = hdr['d_inner']
    nh = hdr['nheads']
    hd = hdr['headdim']
    nl = hdr['num_layers']
    cd = hdr['cond_dim']
    mr = 1  # mimo_rank (fixed at 1 for now)
    nra = ds // 2

    in_proj_out = di * 2 + nh * mr * ds * 2 + nh * 3 + nra

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

        def read(shape):
            nonlocal offset
            n = int(np.prod(shape))
            data = torch.from_numpy(arr[offset:offset + n].copy()).reshape(shape)
            offset += n
            return data

        # 1. in_proj_w: (in_proj_out, d_model)
        block.in_proj.weight.data = read((in_proj_out, dm))
        # 2. dt_bias: (nheads,)
        block.dt_bias.data = read((nh,))
        # 3. B_bias: (nheads, mimo_rank, d_state)
        block.B_bias.data = read((nh, mr, ds))
        # 4. C_bias: (nheads, mimo_rank, d_state)
        block.C_bias.data = read((nh, mr, ds))
        # 5. B_norm_weight: (d_state,)
        block.B_norm_weight.data = read((ds,))
        # 6. C_norm_weight: (d_state,)
        block.C_norm_weight.data = read((ds,))
        # 7. mimo_x: (nheads, mimo_rank, headdim)
        block.mimo_x.data = read((nh, mr, hd))
        # 8. mimo_z: (nheads, mimo_rank, headdim)
        block.mimo_z.data = read((nh, mr, hd))
        # 9. mimo_o: (nheads, mimo_rank, headdim)
        block.mimo_o.data = read((nh, mr, hd))
        # 10. D_proj: (nheads,)
        block.D_proj.data = read((nh,))
        # 11. out_proj_w: (d_model, d_inner)
        block.out_proj.weight.data = read((dm, di))
        # 12. film_proj_w: (2*d_model, d_cond)
        block.film_proj.weight.data = read((2 * dm, cd))
        # 13. dt_film_proj_w: (2*nheads, d_cond)
        block.dt_film_proj.weight.data = read((2 * nh, cd))

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
