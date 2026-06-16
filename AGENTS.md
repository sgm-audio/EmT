# Mamba-SSM CLAP/ARA Plugin Suite

## Build & Test

```bash
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release   # Debug adds -fsanitize=address
make -j4
ctest --output-on-failure
```

CMake 3.20+, C++20, GCC/Clang. GoogleTest fetched via FetchContent. All subdirectories (`src/core`, `src/common`, `src/clap`, `src/ara`) are wired into the root CMakeLists.txt. The `mamba_synth` shared library target produces the CLAP plugin (`.clap` on Linux).

## Namespace & Key Types

Everything is in `namespace mamba`. Core types:
- `SSMBlock` — single Mamba-SSM layer, per-sample processing with selective scan
- `SSMInferenceCore` — multi-layer wrapper, owns `SSMBlock[MAX_LAYERS]` via placement new
- `MambaConfig` / `MambaState` — config struct / state snapshot for all layers
- `SSMWeights` — fixed-size weight struct (all dimensions from `config.h`)
- `WeightLoader` — mmap read-only weight file loader (magic `0x4D424D41`, validates header)
- `SPSCRingBuffer<ParamAtom, 256>` — lock-free param queue
- `Seqlocked<ConditioningFrame>` / `Seqlocked<ARAConditioningFrame>` — seqlock-protected frames
- `MambaSynth` — CLAP plugin state struct (not the `clap_plugin_t` itself)

## Three-Thread Architecture

| Thread | Role | Communication |
|--------|------|---------------|
| **Audio** | `plugin_process()`, SSM inference | Reads params via SPSC ring buffer, reads conditioning via seqlocked frames |
| **Main** | UI, preset save/load | Pushes `ParamAtom` to ring buffer, atomic param values |
| **ARA** | Musical context, offline render | Writes `ARAConditioningFrame` via seqlock |

**Audio thread constraints**: zero `malloc`/`free`, zero `mutex`, zero blocking I/O. All working memory is pre-allocated in `activate()`.

## Current Completion State (~70-80%)

**Working**:
- ArenaAllocator, SPSCRingBuffer, SeqLock/Seqlocked, DoubleBufferSeqLock
- SSMBlock with proper Mamba discretization (`A_bar = exp(A*delta)`, selective scan, input/output projections, causal conv1d, SiLU)
- SSMInferenceCore multi-layer wrapper with state get/set
- WeightLoader with mmap, header validation, per-layer pointer extraction, dequantization (Float32, Int8, FP16)
- CLAP plugin with `clap_entry` factory, audio/note/params/state/tail/latency extensions
- Parameter smoothing (one-pole filter), bypass crossfade, conditioning read in `process()`
- Voice management (monophonic with attack/sustain/release envelope)
- Python export pipeline (`scripts/export_mamba_weights.py`, `scripts/validate_export.py`)
- SIMD helpers (SSE2/NEON) for dot product, fmadd, scale
- 18+ passing tests (SSM, weight loader, conditioning, stability, RT safety)

**Known Issues / Incomplete**:
- `SSMBlock::process_sample()` sums a `d_model` vector to a scalar, which is incorrect for `d_model > 1`.
- `SSMBlock::process_sample()` uses `conv_out_[i % d_inner_]` as the selective scan input `u_t`, causing state aliasing when `d_inner != d_state` and breaking PyTorch reference parity.
- `scripts/export_mamba_weights.py` has a scoping bug in `_random_init()` where `b.cond_proj_dt` (etc.) only targets the last layer.
- `scripts/export_mamba_weights.py` hardcodes `cond_dim=16` instead of using a parameter.
- ARA files are custom stubs, not `ARA::PlugIn` namespace (no real ARA SDK linked; set `ARA_PATH` env var + `ARA_INCLUDE_DIR` CMake var).
- No real `process_block()` vectorized path — `SSMInferenceCore::process_block()` uses ping-pong buffers, but the underlying `SSMBlock::process_block()` is not yet implemented (falls back to `process_sample()` loop).
- No GUI.

See `TODO.md` for the current, verified gap analysis and phased plan.

## Key Constraints

- **`config.h`**: hand-written `MAX_*` constants. Must match weight file dimensions or out-of-bounds access results. Currently: `MAX_D_STATE=16, MAX_D_MODEL=16, MAX_D_INNER=32, MAX_D_CONV=4, MAX_LAYERS=2, COND_DIM=16`
- **Cache alignment**: `alignas(64)` on all shared data structures (`SSMState`, `MambaState`, `ConditioningFrame`, `ARAConditioningFrame`, scratch buffers)
- **Denormals**: `SSMBlock::apply_ftz_daz()` sets FTZ/DAZ in MXCSR — called at start of `plugin_process()`
- **Seqlock ordering**: `write_lock`/`write_unlock` use `memory_order_release`; `read_begin`/`read_retry` use `memory_order_acquire`
- **Release compiler flags**: `-O3 -fno-math-errno -fno-trapping-math -fno-reciprocal-math -fno-signaling-nans -fcx-limited-range -fno-exceptions -fno-rtti -fvisibility=hidden -Wall -Wextra -Wpedantic` — no exceptions/RTTI, use error codes or assertions
- **Weight format**: mmap read-only, flat-binary with `WeightFileHeader` (magic `0x4D424D41`, version 1); per-layer weight order: A_log, D_proj, dt_bias, dt_proj_w, B_proj_w, C_proj_w, conv1d_w, conv1d_b, W_in, W_out, b_in, W_ssm, cond_proj_dt, cond_proj_B, cond_dt_bias, cond_B_bias
- **`SSMWeights` is fixed-size** (stack-allocated 2D arrays using `MAX_*` dims) — not compatible with arbitrary model sizes until `config.h` is regenerated

## Processing Modes

- **Mode A (Tiny)**: per-sample or 4-16 sample blocks, models <=2M params
- **Mode B (Block)**: 64-512 sample blocks, larger models; `SSMInferenceCore::process_block()` exists but the underlying `SSMBlock::process_block()` is a per-sample loop stub and needs a true block-optimized implementation.

## SDK Dependencies

| SDK | Location | Status |
|-----|----------|--------|
| CLAP 1.1+ | `extern/clap/include/` (vendored, official) | Working, `clap_entry` exports |
| ARA 2.0 | Not included — requires Celemony SDK download | Custom stubs only |
| GoogleTest | Fetched via FetchContent | Working |

## Plugin Types (All Share SSM Core)

1. **Mamba Synth** — generative audio from MIDI/ARA conditioning (mostly complete, see Known Issues)
2. **Mamba FX** — long-context neural audio effect (not started)
3. **ARA Generative Companion** — offline region generation (not started)

## Reference Documents

- `TODO.md` — current verified gap analysis and phased plan
- `agent.md` — full task list with RT audio guidance (non-obvious considerations section)
- `ADR's/` — ADR-001 (threading), ADR-002 (memory), ADR-003 (processing modes/ARA bridge), `integration_contract.h` (data structure layouts)
- `Prompts/` — implementation prompts for each subsystem (T2A-T2D, T3A-T3C)
