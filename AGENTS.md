# Mamba-SSM CLAP/ARA Plugin Suite

## Build & Test

```bash
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release   # Debug adds -fsanitize=address
make -j4
./tests/test_ssm_block               # 9 GTest cases (SSMBlock + SSMInferenceCore)
./tests/test_weight_loader            # 9 GTest cases (WeightLoader + integration)
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

**Audio thread constraints**: zero `malloc`/`free`, zero `mutex`, zero blocking I/O. All working memory pre-allocated in `activate()`.

## Current Completion State (~25-30%)

**Working**:
- ArenaAllocator, SPSCRingBuffer, SeqLock/Seqlocked
- SSMBlock with proper Mamba discretization (`A_bar = exp(A*delta)`, selective scan, input/output projections, causal conv1d, SiLU)
- SSMInferenceCore multi-layer wrapper with state get/set
- WeightLoader with mmap, header validation, per-layer pointer extraction
- CLAP plugin with `clap_entry` factory, audio/note/params/state/tail/latency extensions
- Parameter smoothing (one-pole filter), bypass crossfade, conditioning read in `process()`
- 18 passing tests (9 SSM + 9 weight loader)

**Not working / incomplete**:
- `set_conditioning()` applies a crude modulation hack (0.01/0.001 scale factors) — no learned projection layer
- `plugin_init()` uses `new` (heap alloc) — should use arena-allocated memory
- Weight loading in `process()` on note-off: calls `WeightLoader::load()` + `memcpy` per layer on audio thread (blocking I/O + alloc violation)
- No ML export pipeline (no Python scripts, no `config.h` generator)
- ARA files are custom stubs, not `ARA::PlugIn` namespace (no real ARA SDK linked; set `ARA_PATH` env var + `ARA_INCLUDE_DIR` CMake var)
- `models/` directory is empty
- `process_block()` is just a loop over `process_sample()` — no vectorized block path
- No SIMD, no voice management, no GUI

See `TODO.md` for full gap analysis and phased dependency chain.

## Key Constraints

- **`config.h`**: hand-written `MAX_*` constants. Must match weight file dimensions or out-of-bounds access results. Currently: `MAX_D_STATE=16, MAX_D_MODEL=16, MAX_D_INNER=32, MAX_D_CONV=4, MAX_LAYERS=4, COND_DIM=16`
- **Cache alignment**: `alignas(64)` on all shared data structures (`SSMState`, `MambaState`, `ConditioningFrame`, `ARAConditioningFrame`, scratch buffers)
- **Denormals**: `SSMBlock::apply_ftz_daz()` sets FTZ/DAZ in MXCSR — called at start of `plugin_process()`
- **Seqlock ordering**: `write_lock`/`write_unlock` use `memory_order_release`; `read_begin`/`read_retry` use `memory_order_acquire`
- **Release compiler flags**: `-O3 -fno-math-errno -fno-trapping-math -fno-reciprocal-math -fno-signaling-nans -fcx-limited-range -fno-exceptions -fno-rtti -fvisibility=hidden -Wall -Wextra -Wpedantic` — no exceptions/RTTI, use error codes or assertions
- **Weight format**: mmap read-only, flat-binary with `WeightFileHeader` (magic `0x4D424D41`, version 1); per-layer weight order: A_log, D_proj, dt_bias, dt_proj_w, B_proj_w, C_proj_w, conv1d_w, conv1d_b, W_in, W_out, b_in, W_ssm
- **`SSMWeights` is fixed-size** (stack-allocated 2D arrays using `MAX_*` dims) — not compatible with arbitrary model sizes until `config.h` is regenerated

## Processing Modes

- **Mode A (Tiny)**: per-sample or 4-16 sample blocks, models <=2M params
- **Mode B (Block)**: 64-512 sample blocks, larger models; `process_block()` currently just loops `process_sample()` — needs vectorized block path

## SDK Dependencies

| SDK | Location | Status |
|-----|----------|--------|
| CLAP 1.1+ | `extern/clap/include/` (vendored, official) | Working, `clap_entry` exports |
| ARA 2.0 | Not included — requires Celemony SDK download | Custom stubs only |
| GoogleTest | Fetched via FetchContent | Working |

## Plugin Types (All Share SSM Core)

1. **Mamba Synth** — generative audio from MIDI/ARA conditioning (in progress)
2. **Mamba FX** — long-context neural audio effect (not started)
3. **ARA Generative Companion** — offline region generation (not started)

## Reference Documents

- `TODO.md` — verified gap analysis, phased plan with dependency chain
- `agent.md` — full task list with RT audio guidance (non-obvious considerations section)
- `ADR's/` — ADR-001 (threading), ADR-002 (memory), ADR-003 (processing modes/ARA bridge), `integration_contract.h` (data structure layouts)
- `Prompts/` — implementation prompts for each subsystem (T2A-T2D, T3A-T3C)
