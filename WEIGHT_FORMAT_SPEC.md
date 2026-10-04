# EmT Weight File Format Specification (Version 2 — Mamba-3)

**Specification Version:** 2.0  
**Status:** Current Production Standard  
**Magic:** `0x4D424D41` ("MBMA" in little-endian)  
**Target Architecture:** Mamba-3 Zero-Conv State Space Model (CLAP/ARA Engine)

---

## 1. Executive Overview

The EmT weight format is a cache-aligned, flat binary serialization designed specifically for high-throughput, **zero-copy memory-mapped (`mmap`) loading** in real-time audio environments.

Unlike traditional weight formats (e.g. SafeTensors, PyTorch `.pt`, ONNX) which impose runtime parsing, string lookup dictionaries, nested graphs, or heap allocation overheads, the EmT format guarantees:
- **Zero Heap Allocations on Load:** Mapped directly into virtual memory; layer tensors are addressed via deterministic byte pointer arithmetic.
- **Cache-Line Alignment:** The file header is 64 bytes (`alignas(64)`), guaranteeing that the weight payload begins on a CPU cache-line boundary.
- **Zero Algorithmic Latency:** Specifically formatted for the Mamba-3 zero-conv architecture, eliminating the 1D causal convolution and fusing all projections into `in_proj_w`.
- **Hardware Portability:** Little-endian IEEE 754 floating-point with support for 8-bit integer quantization and FP16 half-precision.

---

## 2. File Binary Layout

```
+-------------------------------------------------------------------------------+  Offset 0
| Header: WeightFileHeader (64 bytes, cache-line aligned)                       |
+-------------------------------------------------------------------------------+  Offset 64
| Quantization Metadata (Optional)                                              |
| - Float32 / FP16: 0 bytes                                                     |
| - Int8 (Tensor):  8 bytes (scale: f32, zero_point: i32)                       |
| - Int8 (Layer):   num_layers × 8 bytes                                        |
+-------------------------------------------------------------------------------+  Offset 64 + q_metadata_size
| Layer 0 Weights (13 contiguous arrays, tightly packed)                        |
+-------------------------------------------------------------------------------+
| Layer 1 Weights (13 contiguous arrays, tightly packed)                        |
+-------------------------------------------------------------------------------+
| ...                                                                           |
+-------------------------------------------------------------------------------+
| Layer L-1 Weights                                                             |
+-------------------------------------------------------------------------------+  EOF
```

---

## 3. Header Specification (`WeightFileHeader`)

The header is fixed at exactly **64 bytes** (`static_assert(sizeof(WeightFileHeader) == 64)`).

| Offset | Size (Bytes) | Type | Field Name | Description & Invariants |
|:---|:---|:---|:---|:---|
| `0` | 4 | `uint32_t` | `magic` | Magic identifier: `0x4D424D41` (ASCII `"MBMA"`). |
| `4` | 4 | `uint32_t` | `version` | Format version: `2` (Mamba-3). Legacy `1` files are still loadable; any other value is rejected with `-12`. |
| `8` | 4 | `uint32_t` | `d_state` | State dimension ($N$). Must satisfy $1 \le d\_state \le 	ext{MAX\_D\_STATE}$ (32). |
| `12` | 4 | `uint32_t` | `d_model` | Embedding / channel dimension ($D$). Must satisfy $1 \le d\_model \le 	ext{MAX\_D\_MODEL}$ (256). |
| `16` | 4 | `uint32_t` | `d_inner` | Expanded inner dimension ($E = 2 	imes D$). Must satisfy $1 \le d\_inner \le 	ext{MAX\_D\_INNER}$ (512). |
| `20` | 4 | `uint32_t` | `nheads` | Number of SSM heads ($H$). Must satisfy $1 \le nheads \le 	ext{MAX\_NHEADS}$ (8). |
| `24` | 4 | `uint32_t` | `num_layers`| Total layer count ($L$). Must satisfy $1 \le num\_layers \le 	ext{MAX\_LAYERS}$ (6). |
| `28` | 4 | `uint32_t` | `headdim` | Head dimension ($P = E / H$). Must satisfy $1 \le headdim \le 	ext{MAX\_HEADDIM}$ (64). |
| `32` | 4 | `uint32_t` | `cond_dim` | Musical conditioning dimension ($C$). Must satisfy $1 \le cond\_dim \le 	ext{COND\_DIM}$ (128). |
| `36` | 4 | `uint32_t` | `quantization`| Quantization format enum: `0` (Float32), `1` (Int8 Tensor), `2` (Int8 Layer), `3` (FP16). |
| `40` | 4 | `uint32_t` | `q_metadata_size`| Size in bytes of quantization block (0 for Float32/FP16). |
| `44` | 4 | `uint32_t` | `flags` | Reserved bitfield. Must be `0`. |
| `48` | 8 | `uint64_t` | `weight_data_size`| Total byte size of the weight block across all layers. |
| `56` | 4 | `uint32_t` | `mimo_rank` | MIMO rank. `1` = rank-1 SISO per head (the only rank this runtime executes). `0` = legacy file written before the field existed, treated as rank 1. **Values `> 1` are refused at load with `-16`** (deviation D3) — the format reserves the slots, the forward pass does not execute them. |
| `60` | 4 | `uint32_t` | `reserved` | Reserved for future expansion. Writers must emit `0`; readers ignore it. |

The `mimo_rank` field occupies the low half of what was a single `reserved`
`uint64_t` in earlier revisions of this spec; the byte layout (and the 64-byte
total) is unchanged, so legacy files with a zero `reserved` double-word read back
as rank 1.

### C++ Header Definition

```cpp
struct WeightFileHeader {
    uint32_t magic;             // 0x4D424D41 ("MBMA")
    uint32_t version;           // 2 (Mamba-3 format)
    uint32_t d_state;           // N (state dimension)
    uint32_t d_model;           // D (embedding dimension)
    uint32_t d_inner;           // E (expanded dimension = 2*D)
    uint32_t nheads;            // H (number of heads)
    uint32_t num_layers;        // L
    uint32_t headdim;           // P (head dimension = E/H)
    uint32_t cond_dim;          // Conditioning vector dimension
    uint32_t quantization;      // 0 = Float32, 1 = Int8 per-tensor, 2 = Int8 per-layer, 3 = FP16
    uint32_t q_metadata_size;   // Size in bytes of q-metadata block (0 for Float32)
    uint32_t flags;             // Reserved (must be 0)
    uint64_t weight_data_size;  // Bytes of weight data after header (and q-metadata)
    uint32_t mimo_rank;         // version 2: 1 = rank-1 SISO (only rank implemented).
                                // 0 = legacy file (treated as 1).  > 1 refused by load() (-16).
    uint32_t reserved;          // Reserved (writers emit 0)
};
static_assert(sizeof(WeightFileHeader) == 64, "Header must be 64 bytes");
```

---

## 4. Quantization Formats & Metadata

| `quantization` Code | Enumeration | Quantization Mode | Element Size | `q_metadata_size` | Dequantization Formula |
|:---:|:---|:---|:---:|:---:|:---|
| `0` | `QuantFormat::Float32` | Raw Single Precision | 4 bytes | `0` bytes | None (identity) |
| `1` | `QuantFormat::Int8Tensor` | Per-Tensor Signed Int8 | 1 byte | `8` bytes | $v = (i_8 - 	ext{zp}) 	imes 	ext{scale}$ |
| `2` | `QuantFormat::Int8PerLayer` | Per-Layer Signed Int8 | 1 byte | $L 	imes 8$ bytes | $v = (i_8 - 	ext{zp}_l) 	imes 	ext{scale}_l$ |
| `3` | `QuantFormat::FP16` | IEEE 754 Half Precision | 2 bytes | `0` bytes | IEEE 754 binary16 $	o$ binary32 conversion with FTZ |

---

## 5. Per-Layer Weight Ordering (Mamba-3 Tensor Layout)

Each layer contains **13 contiguous arrays**, serialized strictly in row-major layout without inter-tensor padding:

| Index | Array Name | Tensor Shape | Float Count | Mathematical Role |
|:---:|:---|:---|:---|:---|
| **1** | `in_proj_w` | `[in_proj_out, d_model]` | `in_proj_out × d_model` | Fused input projection for $z, x, B, C, \Delta, A, 	ext{trap}, 	heta$ |
| **2** | `dt_bias` | `[nheads]` | `nheads` | Learnable log-scale time-step bias |
| **3** | `B_bias` | `[nheads, mimo_rank, d_state]` | `nheads × mimo_rank × d_state` | Additive bias for input projection $B$ |
| **4** | `C_bias` | `[nheads, mimo_rank, d_state]` | `nheads × mimo_rank × d_state` | Additive bias for state readout $C$ |
| **5** | `B_norm_weight` | `[d_state]` | `d_state` | RMSNorm gain parameters for $B$ |
| **6** | `C_norm_weight` | `[d_state]` | `d_state` | RMSNorm gain parameters for $C$ |
| **7** | `mimo_x` | `[nheads, mimo_rank, headdim]` | `nheads × mimo_rank × headdim` | MIMO input coupling weights |
| **8** | `mimo_z` | `[nheads, mimo_rank, headdim]` | `nheads × mimo_rank × headdim` | MIMO gate coupling weights |
| **9** | `mimo_o` | `[nheads, mimo_rank, headdim]` | `nheads × mimo_rank × headdim` | MIMO output combination weights |
| **10** | `D_proj` | `[nheads]` | `nheads` | Direct skip-connection feedthrough gains |
| **11** | `out_proj_w` | `[d_model, d_inner]` | `d_model × d_inner` | Final linear projection back to $D$ |
| **12** | `film_proj_w` | `[2 * d_model, cond_dim]` | `2 × d_model × cond_dim` | FiLM projection for output modulation $(\gamma, eta)$ |
| **13** | `dt_film_proj_w`| `[2 * nheads, cond_dim]` | `2 × nheads × cond_dim` | FiLM projection for $\Delta$ modulation. **Read by the forward pass since 2026-10-04 (deviation D6):** per head `h`, row `2h` is $\gamma$, row `2h+1` is $\beta$, applied as $(dd\_dt + dt\_bias) \mapsto v(1+\gamma) + \beta$ before `softplus`, only when the conditioning vector is non-zero |

### Fused Input Dimension (`in_proj_out`)

The dimension `in_proj_out` fuses all input-dependent projections into a single dense matrix multiply:
$$	ext{in\_proj\_out} = 2 \cdot d\_inner + 2 \cdot nheads \cdot mimo\_rank \cdot d\_state + 3 \cdot nheads + 	ext{MAX\_NUM\_ROPE\_ANGLES}$$

Where:
- $2 \cdot d\_inner$: Projections for gate $z$ and input $x$
- $2 \cdot nheads \cdot mimo\_rank \cdot d\_state$: Projections for $B$ and $C$
- $3 \cdot nheads$: Projections for $\Delta$, decay $A$, and trapezoidal parameter $lpha$
- $	ext{MAX\_NUM\_ROPE\_ANGLES}$: Angular velocity parameters for Rotary Position Embedding

---

## 6. Zero-Copy C++ Layer Views

In C++, `WeightLoader` constructs a `WeightLayerPointers` struct pointing directly into the `mmap` region:

```cpp
struct WeightLayerPointers {
    const float* in_proj_w;       // [in_proj_out, d_model]
    const float* dt_bias;         // [nheads]
    const float* B_bias;          // [nheads, mimo_rank, d_state]
    const float* C_bias;          // [nheads, mimo_rank, d_state]
    const float* B_norm_weight;   // [d_state]
    const float* C_norm_weight;   // [d_state]
    const float* mimo_x;          // [nheads, mimo_rank, headdim]
    const float* mimo_z;          // [nheads, mimo_rank, headdim]
    const float* mimo_o;          // [nheads, mimo_rank, headdim]
    const float* D_proj;          // [nheads]
    const float* out_proj_w;      // [d_model, d_inner]
    const float* film_proj_w;     // [2 * d_model, cond_dim]
    const float* dt_film_proj_w;  // [2 * nheads, cond_dim]
};
```

---

## 7. Path Resolution Hierarchy

When loading weights at runtime, the plugin resolves candidates in strict precedence:

1. **Explicit Path Argument:** Caller-provided absolute path.
2. **`EMT_WEIGHTS` Environment Variable:** System-wide or container-level override.
3. **`MAMBA_WEIGHTS` Environment Variable:** Backward-compatible fallback.
4. **`EMT_WEIGHTS_DIR` / `MAMBA_WEIGHTS_DIR`:** Search directory prepended to relative paths.
5. **Standard User Models Directory:**
   - Windows: `%APPDATA%/emt/models/model.bin`
   - Linux: `~/.local/share/emt/models/model.bin`
6. **Local CWD:** `models/test_model.bin` (fallback for test runners).

---

## 8. Validation Rules & Error Codes

The loader returns explicit negative error codes upon validation failure:

| Return Code | Cause | Description |
|:---:|:---|:---|
| `0` | Success | File mapped, validated, and layer pointers established. |
| `-1` | File Open Failure | Path does not exist or insufficient OS read permissions. |
| `-10` | Header Truncation | File size is smaller than `sizeof(WeightFileHeader)` (64 bytes). |
| `-11` | Invalid Magic | Header `magic != 0x4D424D41`. |
| `-12` | Version Incompatible | Header `version` is neither `1` (legacy) nor `2`. |
| `-13` | Corrupt Payload Size | File size does not match $64 + q\_metadata\_size + weight\_data\_size$, or weight data does not equal $L 	imes 	ext{bytes\_per\_layer}$. |
| `-14` | Dimension Overflow | One or more dimensions exceed the compile-time limits in `config.h`. |
| `-15` | Unsupported Quantization | `quantization > QuantFormat::FP16`. |
| `-16` | Unsupported MIMO Rank | Header `version >= 2` and `mimo_rank > 1`: rank-R MIMO is reserved in the format but not implemented by the forward pass (deviation D3). |
| `-17` | Invalid Metadata Size | *Documented but not implemented* — `q_metadata_size` is bounds-checked against the file size instead. |

Code that is documented above but not implemented in `WeightLoader::load()` today:
the `flags`/`reserved` zero checks (writers emit 0; readers ignore the values)
and `-17`.  The implemented set is `0`, `-1`, `-10` … `-16`; keep this table and
`src/core/weight_loader.h` in sync when adding a code.
