# EmT Mamba-3 Weight File Format

**Physical format:** version 2  
**Magic:** `0x4D424D41` (`MBMA` on little-endian systems)  
**Authority:** `src/core/weight_loader.*` and `scripts/export_mamba_weights.py` as audited 2026-08-29.

Version 1 is a different Mamba-1 layout and is rejected. This document specifies version 2 only.

## 1. Byte layout

```text
offset 0                         64-byte header
offset 64                        optional quantization metadata
offset 64 + q_metadata_size      packed layer 0
                                 packed layer 1 ... layer L-1
```

All integers/floats are little-endian IEEE representations. The loader does not byte-swap; big-endian hosts are unsupported. The file must contain exactly the declared layer-data bytes, although filesystem trailing-byte rejection remains a separate hardening item.

The mapped file is read-only, but runtime inference is not zero-copy: each packed layer is copied or dequantized into fixed-capacity Float32 `SSMWeights` using stride-aware loops.

## 2. Header

`WeightFileHeader` is exactly 64 bytes:

| Offset | Type | Field | Version-2 meaning |
|---:|---|---|---|
| 0 | `u32` | `magic` | `0x4D424D41` |
| 4 | `u32` | `version` | exactly `2` |
| 8 | `u32` | `d_state` | state dimension `ds` |
| 12 | `u32` | `d_model` | model dimension `dm` |
| 16 | `u32` | `d_inner` | expanded dimension `di` |
| 20 | `u32` | `nheads` | head count `nh`; this occupied `d_conv` in v1 |
| 24 | `u32` | `num_layers` | layer count `L` |
| 28 | `u32` | `headdim` | per-head dimension `hd` |
| 32 | `u32` | `cond_dim` | conditioning width `cd` |
| 36 | `u32` | `quantization` | format code 0–3 |
| 40 | `u32` | `q_metadata_size` | exact metadata byte count |
| 44 | `u32` | `flags` | must be zero |
| 48 | `u64` | `weight_data_size` | exact bytes for all packed layers |
| 56 | `u64` | `reserved` | must be zero |

Required bounds are the current `MAX_*` values. The loader also requires every stored dimension and layer count to be non-zero.

Structural invariants such as `di == nh * hd`, even `ds`, and `ds/2 <= MAX_NUM_ROPE_ANGLES` should be enforced in a future hardening change; exporters must already obey them.

## 3. Implicit version-2 constants

The header does **not** encode:

- MIMO rank;
- B/C group count;
- number of RoPE angles;
- padded file strides.

The current physical layout freezes these compile-time constants:

```text
file_mimo_rank_stride = MAX_MIMO_RANK = 4
file_rope_angle_stride = MAX_NUM_ROPE_ANGLES = 16
num_bc_heads = MAX_NUM_BC_HEADS = 1 (runtime convention)
active_rope_angles = d_state / 2
```

Models with a smaller active MIMO rank/angle count are zero-padded to those file strides. Changing either stride while retaining version 2 would silently break compatibility and is forbidden.

RoPE angles are shared across heads. There are `MAX_NUM_ROPE_ANGLES` input-projection rows, not one set per head.

## 4. Quantization metadata

| Code | Elements | Metadata |
|---:|---|---|
| 0 | Float32, 4 bytes | none; size 0 |
| 1 | signed Int8, 1 byte | one 16-byte tensor record |
| 2 | signed Int8, 1 byte | one 16-byte record per layer |
| 3 | IEEE binary16, 2 bytes | none; size 0 |

Each Int8 record is:

```text
float32 scale
int32   zero_point
uint8   padding[8]
```

Thus `q_metadata_size` is 16 for code 1 and `16 * L` for code 2. Older documentation incorrectly said 8 bytes.

Dequantization is `scale * (stored_int8 - zero_point)`. Validation of non-finite/non-positive scales and zero-point range is still a hardening TODO.

`fp16_to_fp32()` preserves zeros, normals, infinities, and NaNs but flushes FP16 subnormals to signed zero. That behavior is current implementation, not general IEEE-perfect conversion, and must remain stable or trigger a compatibility decision.

## 5. Packed layer layout

Define:

```text
ds = d_state
dm = d_model
di = d_inner
nh = nheads
hd = headdim
cd = cond_dim
MR = MAX_MIMO_RANK (4)
RA = MAX_NUM_ROPE_ANGLES (16)
IP = 2*di + 2*nh*MR*ds + 3*nh + RA
```

Each layer contains 13 tightly packed arrays in this exact order:

| # | Name | Packed shape | Elements |
|---:|---|---|---:|
| 1 | `in_proj_w` | `[IP, dm]` | `IP*dm` |
| 2 | `dt_bias` | `[nh]` | `nh` |
| 3 | `B_bias` | `[nh, MR, ds]` | `nh*MR*ds` |
| 4 | `C_bias` | `[nh, MR, ds]` | `nh*MR*ds` |
| 5 | `B_norm_weight` | `[ds]` | `ds` |
| 6 | `C_norm_weight` | `[ds]` | `ds` |
| 7 | `mimo_x` | `[nh, MR, hd]` | `nh*MR*hd` |
| 8 | `mimo_z` | `[nh, MR, hd]` | `nh*MR*hd` |
| 9 | `mimo_o` | `[nh, MR, hd]` | `nh*MR*hd` |
| 10 | `D_proj` | `[nh]` | `nh` |
| 11 | `out_proj_w` | `[dm, di]` | `dm*di` |
| 12 | `film_proj_w` | `[2*dm, cd]` | `2*dm*cd` |
| 13 | `dt_film_proj_w` | `[2*nh, cd]` | `2*nh*cd` |

All arrays are row-major with no inter-array padding.

### 5.1 `in_proj_w` row regions

The fused projection rows are ordered:

```text
z:       di
x:       di
B:       nh * MR * ds
C:       nh * MR * ds
dd_dt:   nh
dd_A:    nh
trap:    nh
angles:  RA (shared across heads; active first ds/2)
```

The Python exporter expands active MIMO ranks into the `MR` stride and zero-pads unused slots.

### 5.2 Layer and file sizes

```text
elements_per_layer =
    IP*dm
  + 2*nh*MR*ds
  + 2*ds
  + 3*nh*MR*hd
  + 2*nh
  + dm*di
  + 2*dm*cd
  + 2*nh*cd

bytes_per_layer = elements_per_layer * element_size
weight_data_size = bytes_per_layer * L
```

`element_size` is 4, 1, 1, or 2 for codes 0–3 respectively.

## 6. Loader validation and errors

Current loader checks:

- readable/map-able file of at least 64 bytes;
- magic and version 2;
- non-zero dimensions within compile-time maxima;
- quantization code 0–3;
- zero `flags`/`reserved`;
- exact metadata size for the selected code;
- declared region fits inside the mapped file;
- declared `weight_data_size` exactly equals computed layer bytes.

Current negative return codes:

| Code | Meaning |
|---:|---|
| -1…-4 | open/stat/map platform failures |
| -10 | smaller than header |
| -11 | bad magic |
| -12 | unsupported version |
| -13 | data bounds/size mismatch |
| -14 | zero/out-of-capacity dimension |
| -15 | unsupported quantization code |
| -16 | non-zero reserved field |
| -17 | metadata-size mismatch |

These integers are not yet a stable public API; an enum/error-string interface is planned.

## 7. Materialization

Packed rows cannot be flat-copied into `SSMWeights` because C++ arrays use compile-time inner strides (`MAX_D_MODEL`, `MAX_D_STATE`, `MAX_HEADDIM`, `COND_DIM`). `dequantize_layer_to()` therefore performs row/head/rank-aware copies for Float32 and conversions for quantized data. All plugin loaders must use this path.

`SSMBlock::set_weights()` performs another runtime-dimension-aware copy into the block's owned weights. Both layers of copying currently occur outside the audio callback.

## 8. Path resolution

For `WeightLoader::load(path)`:

1. non-empty explicit `path`;
2. `MAMBA_WEIGHTS` environment value;
3. `models/test_model.bin` relative to the process working directory.

Preset restoration has a separate resolver that also checks the preset directory, `MAMBA_WEIGHTS_DIR`, and `models/`. Do not conflate the two policies.

## 9. Export and validation

```powershell
python scripts/export_mamba_weights.py `
  --state-dim 16 --model-dim 16 --inner-dim 32 --layers 2 `
  --nheads 2 --headdim 16 --cond-dim 16 --random-init `
  --output models/test_model.bin --config-h src/config.h

python scripts/validate_export.py models/test_model.bin
ctest --test-dir build -C Release --output-on-failure
```

`--config-h` rewrites compile-time maxima and therefore can change ABI, memory use, and v2 padded strides. Review its diff before accepting it. Never regenerate production configuration casually from an arbitrary model.

## 10. Compatibility policy

- Version 2 is little-endian and tied to the fixed padded strides above.
- Version 1 is rejected; there is no implicit conversion.
- Future format changes require a new version, golden fixtures, loader tests, export validation, parity tests, and a migration story.
- CLAP IDs/state/preset compatibility is separate from model-file compatibility.
- Add integrity/model identity before accepting untrusted distributed weights.
