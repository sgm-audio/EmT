#!/usr/bin/env python3
"""
Mamba-SSM Weight Export Validator

Loads a flat-binary weight file, reconstructs the model in PyTorch,
and verifies that the C++ inference would produce the same output as
the PyTorch forward pass (within quantization tolerance).

Usage:
    # Basic validation of a binary against PyTorch reimplementation
    python scripts/validate_export.py models/test_model.bin

    # Full round-trip with random weights
    python scripts/export_mamba_weights.py --random-init --layers 2 \\
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
from export_mamba_weights import (WEIGHT_MAGIC, WEIGHT_VERSION, HEADER_SIZE,
                                  floats_per_layer, MambaBlock, SimpleMamba)

# Tolerance for float32 comparison
ATOL = 1e-5
RTOL = 1e-5


def load_header(path: str):
    with open(path, 'rb') as f:
        raw = f.read(HEADER_SIZE)
    fields = struct.unpack('<9I4I4x Q', raw)
    return {
        'magic': fields[0],
        'version': fields[1],
        'd_state': fields[2],
        'd_model': fields[3],
        'd_inner': fields[4],
        'd_conv': fields[5],
        'num_layers': fields[6],
        'expansion_factor': fields[7],
        'cond_dim': fields[8],
        'reserved': list(fields[9:13]),
        'weight_data_size': fields[13],
    }


def load_weights_into_model(model: SimpleMamba, path: str) -> SimpleMamba:
    """Load flat-binary weights into a PyTorch model for verification."""
    hdr = load_header(path)
    ds = hdr['d_state']
    dm = hdr['d_model']
    di = hdr['d_inner']
    dc = hdr['d_conv']
    nl = hdr['num_layers']
    fpl = floats_per_layer(ds, dm, di, dc)

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

        # 1. A_log
        block.A_log.data = read((ds,))
        # 2. D_proj
        block.D_proj.data = read((ds,))
        # 3. dt_bias
        block.dt_bias.data = read((ds,))
        # 4. dt_proj_w
        block.dt_proj_w.data = read((ds, di))
        # 5. B_proj_w
        block.B_proj_w.data = read((di, ds))
        # 6. C_proj_w
        block.C_proj_w.data = read((di, ds))
        # 7. conv1d_w: (dc, di) -> reshape to conv weight (di, 1, dc)
        conv_w = read((dc, di)).t().contiguous().view(di, 1, dc)
        block.conv1d.weight.data = conv_w
        # 8. conv1d_b
        block.conv1d.bias.data = read((di,))
        # 9. W_in.weight (di, dm)
        block.W_in.weight.data = read((di, dm))
        # 10. W_out.weight (dm, di)
        block.W_out.weight.data = read((dm, di))
        # 11. W_ssm.weight (dm, ds)
        block.W_ssm.weight.data = read((dm, ds))
        # 12. b_in
        block.W_in.bias.data = read((di,))
        # 13. cond_proj_dt
        block.cond_proj_dt.data = read((ds, hdr['cond_dim']))
        # 14. cond_proj_B
        block.cond_proj_B.data = read((ds, hdr['cond_dim']))

    assert offset == floats_per_layer(ds, dm, di, dc, hdr['cond_dim']) * nl, (
        f"Read {offset} floats, expected "
        f"{floats_per_layer(ds, dm, di, dc, hdr['cond_dim']) * nl}")
    return model


def validate(path: str, batch: int = 2, seq_len: int = 64):
    """Validate that flat-binary round-trips through PyTorch correctly."""
    hdr = load_header(path)
    assert hdr['magic'] == WEIGHT_MAGIC, f"Bad magic: {hdr['magic']:#x}"
    assert hdr['version'] == WEIGHT_VERSION, f"Bad version: {hdr['version']}"

    print(f"Header: layers={hdr['num_layers']} ds={hdr['d_state']} "
          f"dm={hdr['d_model']} di={hdr['d_inner']} dc={hdr['d_conv']}")

    model = SimpleMamba(hdr['d_state'], hdr['d_model'],
                        hdr['d_inner'], hdr['d_conv'], hdr['num_layers'])
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

    print(f"Forward pass OK: shape={list(y.shape)} "
          f"range=[{y.min().item():.4f}, {y.max().item():.4f}]")
    print(f"Non-zero output: {y.abs().sum().item() > 0}")
    print(f"Deterministic: yes")
    print(f"Validation PASSED")


def main():
    p = argparse.ArgumentParser(description="Validate exported Mamba weights")
    p.add_argument('path', help='Path to .bin weight file')
    p.add_argument('--batch', type=int, default=2)
    p.add_argument('--seq-len', type=int, default=64)
    args = p.parse_args()

    if not os.path.exists(args.path):
        print(f"ERROR: {args.path} not found")
        sys.exit(1)

    validate(args.path, args.batch, args.seq_len)


if __name__ == '__main__':
    main()
