# Mamba-SSM CLAP/ARA Plugin Suite — TODO

**Last updated: 2026-06-15**
**Completion: ~90%** (critical parity bugs fixed, C++↔Python at 1e-5)

---

## Critical Bugs — FIXED

| # | Bug | Status | Files Changed |
|---|-----|--------|--------------|
| C1 | ~~selective scan input aliasing~~ | **FIXED** — was already correct (`conv_out_[i]` uses first `d_state` elements, matching PyTorch `u[:, :d_state]`) | — |
| C2 | `process_block()` summed `d_model` channels to scalar | **FIXED** — now writes `d_model` floats per sample | `ssm_block.cpp`, `ssm_inference_core.cpp`, `test_ssm_block.cpp`, `benchmark_ssm_block.cpp` |
| C3a | `export_mamba_weights.py` didn't pass `cond_dim` to `floats_per_layer()` | **FIXED** | `export_mamba_weights.py:188` |
| C3b | `validate_export.py` missing reads for `cond_dt_bias` / `cond_B_bias` | **FIXED** | `validate_export.py:113-118` |

### Bug also fixed (Jun 13 + Jun 15)

| # | Bug | Date | Files Changed |
|---|-----|------|---------------|
| B1 | `process_block()` conditioning applied `softplus(dt_raw)` before adding conditioning, not after | Jun 13 | `ssm_block.cpp` |
| B2 | `SSMInferenceCore::process_block()` ping-pong buffer aliasing | Jun 13 | `ssm_inference_core.cpp:74-100` |
| B3 | Tests + benchmark: stack buffer overflow in `process_block` output | Jun 13 | `test_ssm_block.cpp`, `benchmark_ssm_block.cpp` |
| B4 | **Flat `memcpy` corrupts multi-D weight arrays** — `SSMWeights` MAX_* strides differ from runtime dims; only first ~21 rows of `in_proj_w` and only head 0 of B/C/mimo arrays received real data | Jun 15 | `ssm_block.cpp`, `weight_loader.cpp`, `test_cpp_parity.cpp`, `reference_mamba3.py`, `reference_forward.py` |

---

## Remaining Work — By Priority

### P0 — Correctness

| # | Task | Status | Blocked By |
|---|------|--------|------------|
| P0-1 | Write NumPy reference Mamba forward pass for C++ parity test | `[x]` | — |
| P0-2 | Unit test: C++ output vs NumPy reference at 1e-5 tolerance | `[x]` | P0-1 |
| P0-3 | Long-run stability test (10+ minutes, verify no NaN/inf/state explosion) | `[ ]` | — |
| P0-4 | C++ round-trip test: load binary weights, run inference, compare vs PyTorch | `[ ]` | — |

### P1 — Features

| # | Task | Status | Blocked By |
|---|------|--------|------------|
| P1-1 | Voice polyphony (N-voice round-robin, currently monophonic only) | `[ ]` | — |
| P1-2 | Preset management beyond state serialization | `[ ]` | — |
| P1-3 | Mamba FX plugin (long-context neural audio effect) | `[ ]` | — |
| P1-4 | ARA Generative Companion plugin | `[ ]` | P1-3, ARA SDK |

### P2 — Performance

| # | Task | Status | Blocked By |
|---|------|--------|------------|
| P2-1 | SIMD in B_proj / C_proj dot-products (strided `[d_inner][d_state]` layout) | `[ ]` | **Layout refactor**: needs transposed `[d_state][d_inner]` in `SSMWeights`, weight loader, export script |
| P2-2 | SIMD in selective scan recurrence (inherently sequential, may not vectorize) | `[ ]` | P2-1 |
| P2-3 | Profile with `perf` — identify top hot spots | `[ ]` | — |
| P2-4 | Benchmark at 44.1kHz and 48kHz sample rates | `[ ]` | — |
| P2-5 | `apply_ftz_daz()` ARM path (currently x86 `ldmxcsr` only) | `[ ]` | ARM hardware or cross-compile |

### P3 — Integration / Tooling

| # | Task | Status | Blocked By |
|---|------|--------|------------|
| P3-1 | Run `clap-validator` against `.clap` bundle | **BLOCKED** | `clap-validator` tool not installed |
| P3-2 | Real-time safety: Valgrind/ASan on `process()` path | `[ ]` | — |
| P3-3 | Real-time safety: Helgrind/TSan on audio thread | `[ ]` | — |
| P3-4 | Cross-platform build (macOS arm64/x86_64, Windows) | `[ ]` | macOS/Windows hardware |
| P3-5 | CI script or Makefile packaging target | `[ ]` | — |

### P4 — ARA SDK (Full Phase 5)

| # | Task | Status | Blocked By |
|---|------|--------|------------|
| P4-1 | Obtain ARA SDK from Celemony | **BLOCKED** | **Celemony license / download** |
| P4-2 | Replace ARA stubs with real SDK (`ARA::PlugIn::PlugIn`, `DocumentController`, `PlaybackRenderer`, `EditorRenderer`) | **BLOCKED** | P4-1 |
| P4-3 | DocumentController: create/archive/restore, persist state | **BLOCKED** | P4-2 |
| P4-4 | PlaybackRenderer: offline bounce with ARA musical context | **BLOCKED** | P4-2 |
| P4-5 | EditorRenderer: waveform visualization, playback sync | **BLOCKED** | P4-2 |
| P4-6 | Wire ARA conditioning to SSM parameter modulation | **BLOCKED** | P4-2 |

### P5 — GUI

| # | Task | Status | Blocked By |
|---|------|--------|------------|
| P5-1 | Plugin GUI (parameter knobs, waveform, conditioning display) | `[ ]` | — |

### P6 — Documentation & Packaging

| # | Task | Status | Blocked By |
|---|------|--------|------------|
| P6-1 | README: build instructions, deps, installation | `[ ]` | — |
| P6-2 | Bundle packaging: Linux `.clap`, macOS bundle, `Info.plist` | `[ ]` | — |
| P6-3 | LICENSE file | `[ ]` | — |
| P6-4 | Parameter reference docs | `[ ]` | — |
| P6-5 | Architecture overview docs | `[ ]` | — |

---

## Blocked Items Summary

| Blocked Item | Blocker | Resolution Path |
|-------------|---------|-----------------|
| P3-1 (clap-validator) | Tool not installed | Install from [github.com/free-audio/clap](https://github.com/free-audio/clap) |
| P4-1–P4-6 (ARA SDK) | Celemony proprietary SDK | Request ARA SDK license from [celemony.com](https://celemony.com) |
| P2-1 (SIMD B/C projections) | Strided `[d_inner][d_state]` weight layout | Refactor to `[d_state][d_inner]` across `SSMWeights`, `WeightLoader`, `export_mamba_weights.py` |
| P2-5 (ARM FTZ/DAZ) | No ARM build target | Cross-compile or test on ARM hardware |
| P3-4 (cross-platform) | No macOS/Windows hardware | CI or VM |

---

## Completed Artifacts

```
src/core/arena_allocator.h/cpp       [done] — mmap-backed bump allocator
src/core/ring_buffer.h               [done] — SPSC lock-free ring buffer
src/core/seqlock.h                   [done] — Seqlock with acquire/release
src/core/ssm_block.h/cpp             [done] — SSM math, ZOH discretization, selective scan, d_model output
src/core/ssm_inference_core.h/cpp    [done] — Multi-layer wrapper, process_block with d_model buffers
src/core/simd_helpers.h              [done] — SSE2/NEON dot_product, fmadd, scale
src/core/weight_loader.h/cpp         [done] — mmap weight loader with FP16/INT8 dequant
src/clap/clap_plugin.h/cpp           [done] — CLAP plugin with audio/note/params/state/tail/thread-check
src/clap/clap_entry.cpp              [done] — clap_entry factory
src/ara/                             [stub] — Custom stubs (needs real ARA SDK)
scripts/export_mamba_weights.py      [done] — PyTorch → flat binary export
scripts/validate_export.py           [done] — Binary → PyTorch round-trip validation
models/test_model.bin                [done] — Generated test model
tests/test_ssm_block.cpp             [done] — 9 tests
tests/test_weight_loader.cpp         [done] — 21 tests
tests/test_conditioning.cpp          [done] — 3 tests
tests/benchmark_ssm_block.cpp        [done] — 0.65 us/sample (15x below 10us target)
```

---

## Test Status

```
33/33 passing
Benchmark: 0.651 us/sample single-layer, 1.33 us/sample 2-layer (target: <10 us/sample)
Export round-trip: PASSED (2 layers, ds=16 dm=16 di=32 dc=4)
```

## Quick Reference

```bash
# Build
cmake -B build && cmake --build build --parallel 4

# Test
./build/tests/test_ssm_block && ./build/tests/test_weight_loader && ./build/tests/test_conditioning

# Benchmark
./build/tests/benchmark_ssm_block

# Export weights
python3 scripts/export_mamba_weights.py --random-init --layers 2 --output models/test_model.bin --config-h src/config.h

# Validate export
python3 scripts/validate_export.py models/test_model.bin
```
