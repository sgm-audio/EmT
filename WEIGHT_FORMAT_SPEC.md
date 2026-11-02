# Mamba-SSM Weight File Format Specification

**Version**: 1.0 (header version 1, FLOAT32 layout) / 1.1 (header version 1, optional INT8/FP16)
**Status**: Active
**File extension**: `.bin`
**Endianness**: Little-endian
**Pickling**: NONE — flat binary, no Python pickling, no protobuf, no JSON

## Overview

Flat binary weight file read via memory-mapped I/O. No compression, no
pickling, no variable-length encoding. The file is laid out as a
`WeightFileHeader` followed by concatenated per-layer weight data plus
quantization metadata (v1.1+).

The file is **read-only** and **memory-mapped** at load time. The C++
`WeightLoader` extracts per-tensor pointers directly into the mmap region;
**no per-weight allocation or copy** is performed in the hot path.

## Header (`WeightFileHeader` — 64 bytes)

Cache-line aligned. All multi-byte fields are little-endian.

| Offset | Size | Field | Type | Description |
|--------|------|-------|------|-------------|
| 0 | 4 | `magic` | `uint32_t` | `0x4D424D41` (`"MBMA"`) |
| 4 | 4 | `version` | `uint32_t` | Format version (currently 1) |
| 8 | 4 | `d_state` | `uint32_t` | SSM state dimension (N) |
| 12 | 4 | `d_model` | `uint32_t` | Model input/output dimension (D) |
| 16 | 4 | `d_inner` | `uint32_t` | Inner expansion dimension (E) |
| 20 | 4 | `d_conv` | `uint32_t` | Causal conv1d kernel size |
| 24 | 4 | `num_layers` | `uint32_t` | Number of SSM layers |
| 28 | 4 | `expansion_factor` | `uint32_t` | `d_inner / d_model` |
| 32 | 4 | `cond_dim` | `uint32_t` | Conditioning vector dimension |
| 36 | 4 | `quantization` | `uint32_t` | `0` = FLOAT32, `1` = INT8 per-tensor, `2` = INT8 per-layer, `3` = FP16 |
| 40 | 4 | `q_metadata_size` | `uint32_t` | Size in bytes of quantization metadata block (0 if FLOAT32) |
| 44 | 4 | `flags` | `uint32_t` | Reserved bitfield (set to 0) |
| 48 | 8 | `weight_data_size` | `uint64_t` | Total weight data bytes (after header + q-metadata) |
| 56 | 8 | `reserved` | `uint64_t` | Reserved (zero) |

**Total header size**: 64 bytes (cache-line aligned).

### Magic Number

`0x4D424D41` is `"MBMA"` in ASCII (Mamba). The C++ `WeightLoader`
rejects any file whose first 4 bytes do not match this value (error code
`-11`).

### Versioning

- Version 1, `quantization=0`: original FLOAT32 layout, no metadata block.
- Version 1, `quantization!=0`: `q_metadata_size` bytes of quantization
  metadata follow the header, then the quantized weight blobs.
- Future versions (`version >= 2`) are reserved for breaking format
  changes. The C++ loader rejects any `version != 1` (error code `-12`).

## Per-Layer Weight Layout (FLOAT32, `quantization = 0`)

Each layer contains 16 weight tensors in **exactly this order**, contiguous
as `float32` values:

| # | Name | Shape | Elements | C++ Field |
|---|------|-------|----------|-----------|
| 1 | `A_log` | `(d_state,)` | `d_state` | `SSMWeights::A_log` |
| 2 | `D_proj` | `(d_state,)` | `d_state` | `SSMWeights::D_proj` |
| 3 | `dt_bias` | `(d_state,)` | `d_state` | `SSMWeights::dt_bias` |
| 4 | `dt_proj_w` | `(d_state, d_inner)` | `d_state * d_inner` | `SSMWeights::dt_proj_w` |
| 5 | `B_proj_w` | `(d_inner, d_state)` | `d_inner * d_state` | `SSMWeights::B_proj_w` |
| 6 | `C_proj_w` | `(d_inner, d_state)` | `d_inner * d_state` | `SSMWeights::C_proj_w` |
| 7 | `conv1d_w` | `(d_conv, d_inner)` | `d_conv * d_inner` | `SSMWeights::conv1d_w` |
| 8 | `conv1d_b` | `(d_inner,)` | `d_inner` | `SSMWeights::conv1d_b` |
| 9 | `W_in` | `(d_inner, d_model)` | `d_inner * d_model` | `SSMWeights::W_in` |
| 10 | `W_out` | `(d_model, d_inner)` | `d_model * d_inner` | `SSMWeights::W_out` |
| 11 | `W_ssm` | `(d_model, d_state)` | `d_model * d_state` | `SSMWeights::W_ssm` |
| 12 | `b_in` | `(d_inner,)` | `d_inner` | `SSMWeights::b_in` |
| 13 | `cond_proj_dt` | `(d_state, cond_dim)` | `d_state * cond_dim` | `SSMWeights::cond_proj_dt` |
| 14 | `cond_proj_B` | `(d_state, cond_dim)` | `d_state * cond_dim` | `SSMWeights::cond_proj_B` |
| 15 | `cond_dt_bias` | `(d_state,)` | `d_state` | `SSMWeights::cond_dt_bias` |
| 16 | `cond_B_bias` | `(d_state,)` | `d_state` | `SSMWeights::cond_B_bias` |

**Floats per layer**:

```
fpl = 3*d_state                           # A_log, D_proj, dt_bias
     + 3*d_state*d_inner                  # dt_proj_w, B_proj_w, C_proj_w
     + d_conv*d_inner                     # conv1d_w
     + 2*d_inner                          # conv1d_b, b_in
     + 2*d_inner*d_model                  # W_in, W_out
     + d_model*d_state                    # W_ssm
     + 2*d_state*cond_dim                 # cond_proj_dt, cond_proj_B
     + 2*d_state                          # cond_dt_bias, cond_B_bias
```

**Bytes per layer** (FLOAT32): `fpl * 4`.

Layers are concatenated in order: layer 0, layer 1, ..., layer `num_layers-1`.

**Total file size** (FLOAT32): `64 + num_layers * fpl * 4`.

## Quantization (v1.1, `quantization != 0`)

When `quantization != 0`, the file layout is:

```
[WeightFileHeader (64 bytes)] [Q-metadata block (q_metadata_size bytes)] [Quantized weights]
```

### Quantization Modes

| Value | Mode | Element Type | Metadata |
|-------|------|--------------|----------|
| 0 | `QFLOAT32` | `float32` (4 B) | none |
| 1 | `QINT8_TENSOR` | `int8` (1 B) | 1 scale (f32) + 1 zero_point (i32), per file |
| 2 | `QINT8_LAYER` | `int8` (1 B) | 1 scale + 1 zero_point per layer, in metadata block |
| 3 | `QFP16` | `uint16` (FP16, 2 B) | none (FP16 is lossless from FP32 within 11-bit mantissa) |

### Q-INT8 Affine Dequantization

```
dequant(q) = scale * (float(q) - zero_point)
```

### Q-INT8 Per-Layer Metadata Block

For `quantization = QINT8_LAYER`, the metadata block contains
`num_layers` records of:

```c
struct LayerQMeta {
    float scale;          // 4 bytes
    int32_t zero_point;   // 4 bytes
    uint8_t pad[8];       // 8 bytes reserved (zero)
};
// sizeof(LayerQMeta) = 16 bytes
// total = num_layers * 16
```

Records are stored in layer order.

### Q-INT8 Per-Tensor Metadata Block

For `quantization = QINT8_TENSOR`, the metadata block is a single record:

```c
struct TensorQMeta {
    float scale;          // 4 bytes
    int32_t zero_point;   // 4 bytes
    uint8_t pad[8];       // 8 bytes reserved
};
// sizeof(TensorQMeta) = 16 bytes
```

### FP16 Storage

`quantization = QFP16` stores each weight as a 16-bit IEEE 754 half
float. The C++ `WeightLoader` provides a `dequantize_layer_to()` helper
that converts the FP16 blobs to FLOAT32 in-place into the `SSMWeights`
struct (caller-allocated). FP16 → FP32 conversion follows the IEEE 754
binary16 standard (10-bit implicit, 5-bit exponent, 10-bit mantissa).

### Quantized Weight Layout

The 14 tensors appear in the same order as FLOAT32; only the element
type differs. The byte size of each tensor is `num_elements * elem_size`,
where `elem_size` is 1 for INT8 and 2 for FP16.

## Round-Trip Test Vector

The `models/test_model.bin` file is a known-good FLOAT32 weight file
with dimensions matching `config.h` at the time of generation. It is
produced by `scripts/export_mamba_weights.py --random-init`.

The corresponding expected output for a known input sequence is
generated by `scripts/reference_ssm.py` and checked by
`tests/test_weight_loader_integration.cpp`.

## Error Codes (C++ `WeightLoader::load`)

| Code | Meaning |
|------|---------|
| 0 | Success |
| -1 | File open failed (not found, permission denied) |
| -2 | File size query failed (`fstat` / `GetFileSizeEx`) |
| -3 | `mmap` / `MapViewOfFile` failed |
| -4 | `MapViewOfFile` returned null (Windows) |
| -10 | File too small (< `sizeof(WeightFileHeader)`) |
| -11 | Bad magic (≠ `0x4D424D41`) |
| -12 | Bad version (≠ 1) |
| -13 | Weight data size mismatch (smaller than `fpl * num_layers`) |
| -14 | Out-of-bounds dimension (e.g. d_state > MAX_D_STATE) |

## Configuration

The weight file path is resolved in this order:

1. Constructor `WeightLoader(path)` (if provided)
2. Environment variable `MAMBA_WEIGHTS`
3. Default: `models/test_model.bin` (relative to the executable)

The C++ `SSMWeights` struct is **fixed-size** (sized by `MAX_*`
constants in `config.h`); the loader checks that the file's dimensions
do not exceed the compiled-in maxima and returns `-14` on overflow.
