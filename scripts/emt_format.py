#!/usr/bin/env python3
"""
emt_format.py — single source of truth for the EmT flat-binary weight format.

Everything that reads or writes ``.bin`` weight files on the Python side
(exporter, validator, NumPy references, trainer) imports the constants and
layout helpers from this module instead of re-declaring them.  The C++ mirror
is ``src/core/weight_loader.h`` (``WeightFileHeader``, ``QuantFormat``) plus
the file-layout strides in ``src/config.h`` (``MAX_MIMO_RANK``,
``MAX_NUM_ROPE_ANGLES``); if the format ever changes, those two files and
this module are the complete list of places to touch.

Layout recap (Mamba-3, version 2 — see WEIGHT_FORMAT_SPEC.md):
  * 64-byte little-endian header: 12 x uint32, one uint64
    (``weight_data_size``), one uint32 (``mimo_rank``), one uint32 (reserved).
  * 13 float arrays per layer in a fixed order, padded to the ``MAX_*``
    file-layout strides regardless of the model's actual dims.
"""

import struct
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Header constants (must match src/core/weight_loader.h)
# ---------------------------------------------------------------------------

WEIGHT_MAGIC = 0x4D424D41  # "MBMA"
WEIGHT_VERSION = 2  # Mamba-3 format
HEADER_FMT = '<12IQII'  # 12 x uint32, uint64 wds, uint32 mimo_rank, uint32 reserved
HEADER_SIZE = 64


class QuantFormat:
    """Mirror of C++ ``QuantFormat`` (src/core/weight_loader.h)."""

    FLOAT32 = 0
    INT8_PER_TENSOR = 1
    INT8_PER_LAYER = 2
    FP16 = 3


# ---------------------------------------------------------------------------
# File-layout strides (must match src/config.h)
# ---------------------------------------------------------------------------
# The weight file pads these slots to the compile-time maxima regardless of
# the model's actual dims, so the C++ loader can compute fixed offsets.

MAX_MIMO_RANK = 4
MAX_NUM_ROPE_ANGLES = 16
MAX_NUM_BC_HEADS = 1


@dataclass(frozen=True)
class WeightHeader:
    """Parsed 64-byte file header (field order = ``HEADER_FMT``)."""

    magic: int
    version: int
    d_state: int
    d_model: int
    d_inner: int
    nheads: int
    num_layers: int
    headdim: int
    cond_dim: int
    quantization: int
    q_metadata_size: int
    flags: int
    weight_data_size: int
    # Deviation D3: 0 = legacy file written before the field existed
    # (treated as rank 1), 1 = rank-1 SISO.  The runtime refuses > 1.
    mimo_rank: int
    reserved: int

    @classmethod
    def from_bytes(cls, data: bytes) -> "WeightHeader":
        if len(data) < HEADER_SIZE:
            raise ValueError(
                f"header truncated: {len(data)} bytes, need {HEADER_SIZE}")
        return cls(*struct.unpack(HEADER_FMT, data[:HEADER_SIZE]))

    def validate(self) -> "WeightHeader":
        if self.magic != WEIGHT_MAGIC:
            raise ValueError(f"bad magic: {self.magic:#010x}")
        if self.version != WEIGHT_VERSION:
            raise ValueError(f"bad version: {self.version}")
        for name in ("d_state", "d_model", "d_inner", "nheads",
                     "headdim", "cond_dim", "num_layers"):
            if getattr(self, name) <= 0:
                raise ValueError(f"bad {name}: {getattr(self, name)}")
        return self


def parse_header(data: bytes) -> WeightHeader:
    """Unpack + validate the 64-byte header at the start of ``data``."""
    return WeightHeader.from_bytes(data).validate()


def in_proj_out_for(d_inner: int, nheads: int, d_state: int) -> int:
    """Fused-projection row count in the *file* layout (``MAX_*``-padded).

    Row groups: z + x + B + C + (dd_dt, dd_A, trap) + angles.
    """
    return (d_inner * 2
            + nheads * MAX_MIMO_RANK * d_state * 2
            + nheads * 3
            + MAX_NUM_ROPE_ANGLES)


def floats_per_layer(ds: int, dm: int, di: int, nh: int,
                     hd: int, cd: int) -> int:
    """Float32 element count per layer in the ``MAX_*``-padded file layout."""
    in_proj_out = in_proj_out_for(di, nh, ds)
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
                nl: int, cd: int, wds: int, mr: int = 1) -> bytes:
    """Build the 64-byte ``WeightFileHeader`` (version 2, 12I Q I I layout).

    ``mr`` (mimo_rank) is read by ``WeightLoader::load()``: values > 1 are
    refused because the forward pass implements rank-1 SISO only
    (deviation D3).
    """
    data = struct.pack(HEADER_FMT,
                       WEIGHT_MAGIC, WEIGHT_VERSION,
                       ds, dm, di, nh, nl, hd, cd,
                       0, 0, 0,        # quantization, q_metadata_size, flags
                       wds, mr, 0)     # weight_data_size, mimo_rank, reserved
    assert len(data) == HEADER_SIZE
    return data
