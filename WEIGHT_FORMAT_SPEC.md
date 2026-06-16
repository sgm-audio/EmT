# Weight File Format Specification

**Version:** 1.0  
**Date:** 2026-06-14  
**Magic:** `0x4D424D41` ("MBMA" in little-endian)

---

## 1. Overview

The Mamba-SSM weight file is a flat binary format designed for zero-copy mmap loading.
A single file contains a fixed-size header (64 bytes), an optional quantization metadata
block, and the concatenated weight data for all layers in a deterministic order.

All multi-byte fields are **little-endian**. All floating-point values use **IEEE 754**.

---

## 2. File Layout

```
+------------------+  offset 0
|  Header (64 B)   |
+------------------+  offset 64
|  Q-metadata      |  (0 bytes for Float32; variable for quantized formats)
+------------------+  offset 64 + q_metadata_size
|  Layer 0 weights |
+------------------+
|  Layer 1 weights |
+------------------+
|       ...        |
+------------------+
|  Layer L-1 wts   |
+------------------+  offset 64 + q_metadata_size + weight_data_size
```

### 2.1 Header (64 bytes, cache-line aligned)

| Offset | Size | C Type | Field | Description |
|--------|------|--------|-------|-------------|
| 0 | 4 | `uint32_t` | `magic` | `0x4D424D41` ("MBMA") |
| 4 | 4 | `uint32_t` | `version` | Format version (currently `1`) |
| 8 | 4 | `uint32_t` | `d_state` | State dimension (N) |
| 12 | 4 | `uint32_t` | `d_model` | Model dimension (D) |
| 16 | 4 | `uint32_t` | `d_inner` | Inner/expansion dimension (E) |
| 20 | 4 | `uint32_t` | `d_conv` | Conv1d kernel width |
| 24 | 4 | `uint32_t` | `num_layers` | Number of layers (L) |
| 28 | 4 | `uint32_t` | `expansion_factor` | `d_inner / d_model` |
| 32 | 4 | `uint32_t` | `cond_dim` | Conditioning vector dimension |
| 36 | 4 | `uint32_t` | `quantization` | Quantization format (see §3) |
| 40 | 4 | `uint32_t` | `q_metadata_size` | Size of quantization metadata block |
| 44 | 4 | `uint32_t` | `flags` | Reserved (must be 0) |
| 48 | 8 | `uint64_t` | `weight_data_size` | Total weight data bytes (after q-metadata) |
| 56 | 8 | `uint64_t` | `reserved` | Reserved (must be 0) |

**Total:** 64 bytes (`static_assert(sizeof(WeightFileHeader) == 64)`).

The header struct in C++:
```cpp
struct WeightFileHeader {
    uint32_t magic;
    uint32_t version;
    uint32_t d_state;
    uint32_t d_model;
    uint32_t d_inner;
    uint32_t d_conv;
    uint32_t num_layers;
    uint32_t expansion_factor;
    uint32_t cond_dim;
    uint32_t quantization;
    uint32_t q_metadata_size;
    uint32_t flags;
    uint64_t weight_data_size;
    uint64_t reserved;
};
```

---

## 3. Quantization Formats

| `quantization` value | Name | Description |
|----------------------|------|-------------|
| `0` | **Float32** | IEEE 754 single-precision, raw float array |
| `1` | **Int8 (per-tensor)** | Single scale + zero_point for all weights |
| `2` | **Int8 (per-layer)** | Per-layer scale + zero_point |
| `3` | **FP16** | IEEE 754 half-precision (`binary16`) |

### 3.1 Float32
- No quantization metadata block (`q_metadata_size = 0`).
- Weights stored as native `float[4]` arrays.

### 3.2 Int8 Quantization
- Weight values stored as `int8_t`.
- Dequantization formula: `float_value = (int8_value - zero_point) * scale`
- **Per-tensor:** One `scale` (float) + `zero_point` (int32) for the entire file.
  `q_metadata_size = 8` bytes.
- **Per-layer:** One `scale` (float) + `zero_point` (int32) per layer.
  `q_metadata_size = L * 8` bytes.
- Q-metadata block layout:
  ```
  [scale_0: f32, zp_0: i32] [scale_1: f32, zp_1: i32] ...
  ```
  (or just one pair for per-tensor)

### 3.3 FP16 Quantization
- Weight values stored as `uint16_t` (IEEE 754 `binary16`).
- No quantization metadata block (`q_metadata_size = 0`).
- Convert to float32 using IEEE 754 `binary16` → `binary32` conversion.
  Subnormals are flushed to zero (FTZ) to match the audio thread's denormal policy.
- Dequantization code: `float fp16_to_fp32(uint16_t h)` — rebias exponent from 15 to 127,
  preserve sign, shift mantissa to upper 23 bits.

---

## 4. Per-Layer Weight Ordering

Each layer's weight block contains 16 arrays concatenated in this **deterministic order**:

| # | Array | Shape | Size (floats) |
|---|-------|-------|---------------|
| 1 | `A_log` | `(d_state,)` | `d_state` |
| 2 | `D_proj` | `(d_state,)` | `d_state` |
| 3 | `dt_bias` | `(d_state,)` | `d_state` |
| 4 | `dt_proj_w` | `(d_state, d_inner)` | `d_state × d_inner` |
| 5 | `B_proj_w` | `(d_inner, d_state)` | `d_inner × d_state` |
| 6 | `C_proj_w` | `(d_inner, d_state)` | `d_inner × d_state` |
| 7 | `conv1d_w` | `(d_conv, d_inner)` | `d_conv × d_inner` |
| 8 | `conv1d_b` | `(d_inner,)` | `d_inner` |
| 9 | `W_in` | `(d_inner, d_model)` | `d_inner × d_model` |
| 10 | `W_out` | `(d_model, d_inner)` | `d_model × d_inner` |
| 11 | `W_ssm` | `(d_model, d_state)` | `d_model × d_state` |
| 12 | `b_in` | `(d_inner,)` | `d_inner` |
| 13 | `cond_proj_dt` | `(d_state, cond_dim)` | `d_state × cond_dim` |
| 14 | `cond_proj_B` | `(d_state, cond_dim)` | `d_state × cond_dim` |
| 15 | `cond_dt_bias` | `(d_state,)` | `d_state` |
| 16 | `cond_B_bias` | `(d_state,)` | `d_state` |

### 4.1 Total Floats Per Layer

```
floats_per_layer = 3·ds + 3·ds·di + dc·di + 2·di + di·dm + dm·di + dm·ds + 2·ds·cd + 2·ds
```
Where `ds = d_state`, `dm = d_model`, `di = d_inner`, `dc = d_conv`, `cd = cond_dim`.

### 4.2 Total File Size

```
file_size = 64 + q_metadata_size + (floats_per_layer × L × elem_size)
```
Where `elem_size = 4` (Float32), `1` (Int8), or `2` (FP16).

---

## 5. Loader Algorithm

The C++ `WeightLoader` class implements the following:

1. **Open** the file via `open()` + `mmap()` (read-only, private).
2. **Validate** the header:
   - `magic == 0x4D424D41`
   - `version == 1`
   - `d_state > 0`, `d_model > 0`, `d_inner > 0`, `d_conv > 0`, `num_layers > 0`, `cond_dim > 0`
   - File size matches `64 + q_metadata_size + weight_data_size`
3. **Check** dimensions fit within `config.h` `MAX_*` bounds (file dimensions must be ≤ `MAX_*`):
   ```cpp
   d_state <= MAX_D_STATE && d_model <= MAX_D_MODEL &&
   d_inner <= MAX_D_INNER && d_conv <= MAX_D_CONV &&
   num_layers <= MAX_LAYERS && cond_dim <= COND_DIM
   ```
4. **Compute** per-layer pointers using `layer_base(layer)`:
   - Points past header + q-metadata block + `layer * bytes_per_layer`.
   - Each of the 16 arrays is offset from `layer_base` by the cumulative size of preceding arrays.
5. **Dequantize** if INT8/FP16 → convert to float32 before copying into `SSMWeights`.

### 5.1 Weight Path Resolution

```
1. Explicit path argument       (highest priority)
2. MAMBA_WEIGHTS env variable
3. "models/test_model.bin"     (relative to cwd, lowest priority)
```

---

## 6. C++ Data Structures

The mmap'd data populates the `SSMWeights` struct (stack-allocated, fixed-size using `MAX_*`):

```cpp
struct SSMWeights {
    float A_log[MAX_D_STATE];
    float D_proj[MAX_D_STATE];
    float dt_bias[MAX_D_STATE];
    float dt_proj_w[MAX_D_STATE][MAX_D_INNER];
    float B_proj_w[MAX_D_INNER][MAX_D_STATE];
    float C_proj_w[MAX_D_INNER][MAX_D_STATE];
    float conv1d_w[MAX_D_CONV][MAX_D_INNER];
    float conv1d_b[MAX_D_INNER];
    float W_in[MAX_D_INNER][MAX_D_MODEL];
    float W_out[MAX_D_MODEL][MAX_D_INNER];
    float W_ssm[MAX_D_MODEL][MAX_D_STATE];
    float b_in[MAX_D_INNER];
    float cond_proj_dt[MAX_D_STATE][COND_DIM];
    float cond_proj_B[MAX_D_STATE][COND_DIM];
    float cond_dt_bias[MAX_D_STATE];
    float cond_B_bias[MAX_D_STATE];
};
```

Array layout is **row-major**. For `float[d_state][d_inner]`, elements are stored as:
`[0,0], [0,1], ..., [0,di-1], [1,0], ...`

---

## 7. Config.h Generation

The export tool (`scripts/export_mamba_weights.py`) can auto-generate `src/config.h` with
`MAX_*` constants matching the weight file dimensions:

```bash
python scripts/export_mamba_weights.py \
    --state-dim 16 --model-dim 16 --inner-dim 32 --conv-dim 4 --layers 2 \
    --random-init --output models/test_model.bin --config-h src/config.h
```

**Rule:** `config.h` must be regenerated whenever the weight file dimensions change.
The `MAX_*` constants must be **≥** all actual weight file dimensions, or the loader
will reject the file (`fits_in_static_weights()` returns `false`).

### 7.1 Generated Config.h Template

```c
#define MAX_D_STATE <d_state>
#define MAX_D_MODEL <d_model>
#define MAX_D_INNER <d_inner>
#define MAX_D_CONV  <d_conv>
#define MAX_LAYERS  <num_layers>
#define MAX_BLOCK_SIZE 512
#define MAX_PARAM_ATOMS 256
#define COND_DIM    <cond_dim>
#define SSM_STATE_NORM_INTERVAL 4096
```

---

## 8. Validation

```bash
# Export random-init weights
python scripts/export_mamba_weights.py --random-init --layers 2 --output models/test_model.bin

# Validate round-trip (binary → PyTorch → verify)
python scripts/validate_export.py models/test_model.bin

# The C++ test suite validates loading, dimensions, and bad-file rejection:
./build/tests/test_weight_loader
```

---

## 9. Binary Compatibility

| Version | Date | Changes |
|---------|------|---------|
| 1 | 2026-05 | Initial format: 16 arrays per layer, 64-byte header, Float32/Int8/FP16 |

- **Backward compatibility:** Not guaranteed across format version bumps.
- **Endianness:** Little-endian only. Big-endian platforms must byteswap on load.
- **Alignment:** Header is 64-byte aligned (cache-line). Weight arrays are tightly packed
  (no padding between arrays).
