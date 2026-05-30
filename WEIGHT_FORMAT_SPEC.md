# Mamba-SSM Weight File Format Specification

**Version**: 1.0  
**Status**: Active  
**File extension**: `.bin`  
**Endianness**: Little-endian  

## Overview

Flat binary weight file read via memory-mapped I/O. No compression, no
pickling, no variable-length encoding.  The file is laid out as a
`WeightFileHeader` followed by concatenated per-layer weight data.

## Header (`WeightFileHeader` — 64 bytes)

| Offset | Size | Field | Type | Description |
|--------|------|-------|------|-------------|
| 0 | 4 | `magic` | `uint32_t` | `0x4D424D41` (`"MBMA"`) |
| 4 | 4 | `version` | `uint32_t` | Format version (currently 1) |
| 8 | 4 | `d_state` | `uint32_t` | SSM state dimension (N) |
| 12 | 4 | `d_model` | `uint32_t` | Model input/output dimension (D) |
| 16 | 4 | `d_inner` | `uint32_t` | Inner expansion dimension |
| 20 | 4 | `d_conv` | `uint32_t` | Causal conv1d kernel size |
| 24 | 4 | `num_layers` | `uint32_t` | Number of SSM layers |
| 28 | 4 | `expansion_factor` | `uint32_t` | `d_inner / d_model` |
| 32 | 4 | `cond_dim` | `uint32_t` | Conditioning vector dimension |
| 36 | 16 | `reserved` | `uint32_t[4]` | Reserved, zero-filled |
| 52 | 8 | `weight_data_size` | `uint64_t` | Total weight data bytes (after header) |

**Total header size**: 64 bytes (cache-line aligned).

## Per-Layer Weight Layout

Each layer contains the following weight tensors in **exactly this order**,
contiguously as `float32` values:

| # | Name | Shape | Elements | Purpose |
|---|------|-------|----------|---------|
| 1 | `A_log` | `(d_state,)` | `d_state` | Logged A matrix diagonal |
| 2 | `D_proj` | `(d_state,)` | `d_state` | D skip connection |
| 3 | `dt_bias` | `(d_state,)` | `d_state` | Delta bias |
| 4 | `dt_proj_w` | `(d_state, d_inner)` | `d_state * d_inner` | Delta projection weights |
| 5 | `B_proj_w` | `(d_inner, d_state)` | `d_inner * d_state` | B projection weights |
| 6 | `C_proj_w` | `(d_inner, d_state)` | `d_inner * d_state` | C projection weights |
| 7 | `conv1d_w` | `(d_conv, d_inner)` | `d_conv * d_inner` | Causal conv1d kernel |
| 8 | `conv1d_b` | `(d_inner,)` | `d_inner` | Causal conv1d bias |
| 9 | `W_in` | `(d_inner, d_model)` | `d_inner * d_model` | Input projection matrix |
| 10 | `W_out` | `(d_model, d_inner)` | `d_model * d_inner` | Output projection matrix |
| 11 | `W_ssm` | `(d_model, d_state)` | `d_model * d_state` | SSM state→output mapping |
| 12 | `b_in` | `(d_inner,)` | `d_inner` | Input projection bias |
| 13 | `cond_proj_dt` | `(d_state, cond_dim)` | `d_state * cond_dim` | Conditioning → delta bias (learned) |
| 14 | `cond_proj_B` | `(d_state, cond_dim)` | `d_state * cond_dim` | Conditioning → B bias (learned) |

**Floats per layer**: `3*d_state + 3*d_state*d_inner + d_conv*d_inner + 2*d_inner + 2*d_inner*d_model + d_model*d_state + 2*d_state*cond_dim`

## Layer Count

Layers are concatenated layer 0, layer 1, ..., layer `num_layers-1`.

Total file size = `sizeof(WeightFileHeader) + num_layers * floats_per_layer * sizeof(float)`.

## Quantization (Future)

Version 2 will add support for INT8 affine quantization with per-tensor
scale + zero-point metadata stored after the header.
