# Mamba-SSM CLAP/ARA Plugin Suite

## Build & Test

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4
ctest --test-dir build -C Release --output-on-failure    # 40/40 tests
```

CMake 3.20+, C++20. GoogleTest fetched via FetchContent. All subdirectories (`src/core`, `src/common`, `src/clap`, `src/ara`) wired into root CMakeLists.txt. Output is `MambaSynth.clap` (CLAP bundle).

### MSVC quirk
Test executables and the plugin itself need a **16 MB stack** (`/STACK:16777216`). `SSMWeights` at MAX config is ~4 MB. `tests/CMakeLists.txt` and `src/clap/CMakeLists.txt` already set this.

## Namespace & Key Types

Everything in `namespace mamba`. Core types:
- `SSMBlock` — single Mamba-3 layer: multi-head complex state, trapezoidal discretization, RoPE, RMSNorm, FiLM conditioning, MIMO rank coupling
- `SSMInferenceCore` — multi-layer wrapper, owns `SSMBlock[MAX_LAYERS]` via placement-new in pre-allocated byte storage
- `MambaConfig` / `MambaState` — runtime config (per-layer dims read from weight file) / state snapshot
- `SSMWeights` — fixed-size stack struct using `MAX_*` constants from `config.h`
- `WeightLoader` — mmap read-only weight loader (magic `0x4D424D41`, version 2 = Mamba-3)
- `SPSCRingBuffer<ParamAtom, 256>` — lock-free param queue (audio-thread safe)
- `Seqlocked<ConditioningFrame>` / `DoubleBufferSeqLock<StateMirror>` — lock-free frame exchange
- `MambaSynth` — CLAP plugin state struct (not the `clap_plugin_t` itself)
- `FxState` — Mamba FX plugin state (parallel struct, separate descriptor)

### Two CLAP plugins are exported
`clap_entry.cpp` factory returns **2** plugins: `com.mamba-ssm.synth` (Mamba Synth, instrument) and `com.mamba-ssm.fx` (Mamba FX, effect). Both use the same `SSMInferenceCore`.

## Three-Thread Architecture

| Thread | Role | Communication |
|--------|------|---------------|
| Audio | `plugin_process()`, SSM inference | Reads params via SPSC ring buffer, reads conditioning via seqlocked frames |
| Main | Host callbacks, preset save/load | Pushes `ParamAtom` to ring buffer, atomic param values; loads `WeightLoader` |
| ARA | Musical context, offline render | Writes `ARAConditioningFrame` via seqlock |

**Audio thread constraints**: zero `malloc`/`free`, zero `mutex`, zero blocking I/O. All working memory pre-allocated in `core_storage` byte arrays + `SSMState`/`SSMWeights` stack structs.

## config.h — Compile-Time Dimension Limits

`src/config.h` defines `MAX_*` constants that **must be >= the actual model dimensions** stored in the weight file header, or `WeightLoader::load()` rejects with error `-14`. The export script's `--config-h` flag regenerates this file.

Current values:
```
MAX_D_STATE=32  MAX_D_MODEL=256  MAX_D_INNER=512  MAX_NHEADS=8
MAX_HEADDIM=64  MAX_MIMO_RANK=4  MAX_NUM_BC_HEADS=1  MAX_NUM_ROPE_ANGLES=16
MAX_LAYERS=6    COND_DIM=128     MAX_BLOCK_SIZE=512  MAX_PARAM_ATOMS=256
MAX_CHUNK_SIZE=128  SSM_STATE_NORM_INTERVAL=4096
```

Mamba-3 has **no conv1d** — 0-sample algorithmic latency. `MAX_D_CONV` does not exist.

## Weight File Format (Mamba-3, Version 2)

- Magic: `0x4D424D41` ("MBMA"), version 2, 64-byte cache-aligned header
- mmap read-only (`PROT_READ`, `MAP_PRIVATE` on POSIX; `PAGE_READONLY` + `MapViewOfFile` on Windows)
- Weights never copied into writable memory; `dequantize_layer_to()` is the only exception (INT8/FP16 → float32)
- 13 arrays per layer, deterministic order: `in_proj_w`, `dt_bias`, `B_bias`, `C_bias`, `B_norm_weight`, `C_norm_weight`, `mimo_x`, `mimo_z`, `mimo_o`, `D_proj`, `out_proj_w`, `film_proj_w`, `dt_film_proj_w`
- Quantization: Float32(0), Int8 per-tensor(1), Int8 per-layer(2), FP16(3)
- Path resolution: explicit arg > `MAMBA_WEIGHTS` env var > `models/test_model.bin`

## RT-Safety & Concurrency Gotchas

- **Denormals**: `SSMBlock::apply_ftz_daz()` sets FTZ/DAZ in MXCSR — called at the top of every `plugin_process()`. ARM path (FPCR bit 24) is implemented.
- **Seqlock ordering**: `write_lock`/`write_unlock` use `memory_order_release`; `read_begin`/`read_retry` use `memory_order_acquire`
- **Cache alignment**: `alignas(64)` on `SSMState`, `MambaState`, `ConditioningFrame`, `ARAConditioningFrame`, `ParamSmooth`, `Voice`, `StateMirror`, scratch buffers. Failure to align causes false sharing between audio and main threads.
- **No exceptions/RTTI**: Release flags include `-fno-exceptions -fno-rtti -fvisibility=hidden -Wall -Wextra -Wpedantic`. Use error codes or assertions, never `throw`.
- **Mamba-3 FTZ/DAZ** is set per-`process()` call, not once — the host may restore MXCSR between calls.

## Known Limitations

- `SSMBlock::process_block()` is a **per-sample loop stub** — no vectorized block path
- ARA files are **custom stubs**, not `ARA::PlugIn` namespace — requires Celemony SDK
- GUI implemented only for Linux/X11 (`clap_gui.cpp`); returns false on macOS/Windows
- No SIMD in B/C projections (strided `[nheads][mimo_rank][d_state]` layout in `WeightLayerPointers`)
- Process events are handled per-sample in `clap_plugin.cpp:plugin_process()` — events iterated inside the sample loop, not batched
- Voices share one SSMInferenceCore; per-voice SSMState is save/restored each process() call

## Test Structure

| File | Tests | What |
|------|-------|------|
| `test_ssm_block.cpp` | 9 | SSMBlock multi-head, SSMInferenceCore multi-layer, state round-trip |
| `test_weight_loader.cpp` | 20 | Header validation, layer pointers, FP16/Int8 dequant, path resolution |
| `test_conditioning.cpp` | 4 | FiLM conditioning, seqlock tear protection |
| `test_stability.cpp` | 3 | Long-run no-NaN, extreme params, zero-input |
| `test_realtime_safety.cpp` | 3 | Process call-graph audit, no allocation, no lock in path |
| `test_cpp_parity.cpp` | 1 | C++ vs Python reference at 1e-5 tolerance |
| `benchmark_ssm_block` | — | Per-sample throughput (~0.65 µs/sample single layer) |

Run a single test binary: `./build/tests/Release/test_weight_loader.exe`

## Weight Export Pipeline

```bash
# Generate random-init test weights + regenerate config.h
python3 scripts/export_mamba_weights.py \
    --state-dim 16 --model-dim 16 --inner-dim 32 --layers 2 \
    --nheads 2 --headdim 16 --cond-dim 16 --random-init \
    --output models/test_model.bin --config-h src/config.h

# Validate round-trip (binary → PyTorch → compare)
python3 scripts/validate_export.py models/test_model.bin
```

## Reference Documents

- `TODO.md` — current gap analysis and phased plan
- `agent.md` — full task list with RT audio guidance (non-obvious considerations)
- `ARCHITECTURE.md` — deep architecture overview
- `WEIGHT_FORMAT_SPEC.md` — weight file format specification
- `PARAMETER_REFERENCE.md` — plugin parameter documentation
- `ADR's/` — ADR-001 (threading), ADR-002 (memory), ADR-003 (processing modes/ARA bridge)
- `Prompts/` — implementation prompts for each subsystem (T2A-T2D, T3A-T3C)
