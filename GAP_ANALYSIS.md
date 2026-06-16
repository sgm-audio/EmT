# Mamba-SSM CLAP/ARA Plugin Suite — Comprehensive Gap Analysis

**Generated:** 2026-06-14
**Project Status:** ~85% complete per TODO.md
**Source:** Audit of all source files under `src/`, `tests/`, `scripts/`, `ADR's/`, `Prompts/`, `TODO.md`, `AGENTS.md`

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Explicit TODO/FIXME/HACK Markers](#2-explicit-todofixmehack-markers)
3. [Stub / Placeholder / Unimplemented Code](#3-stub--placeholder--unimplemented-code)
4. [Specification-Implementation Gaps](#4-specification-implementation-gaps)
5. [Missing Functionality and Components](#5-missing-functionality-and-components)
6. [Critical Bug: Conditioning Weights Not Loaded](#6-critical-bug-conditioning-weights-not-loaded)
7. [Code Quality Issues](#7-code-quality-issues)
8. [Blocked Items](#8-blocked-items)
9. [Completed (Verified Working)](#9-completed-verified-working)
10. [All Source Files — Complete Inventory](#10-all-source-files--complete-inventory)
11. [Tests — Status and Coverage](#11-tests--status-and-coverage)
12. [Priority Remediation Plan](#12-priority-remediation-plan)

---

## 1. Project Overview

Three-plugin suite (Mamba Synth, Mamba FX, ARA Generative Companion) powered by Mamba-SSM state space models. CLAP 1.1+ and ARA 2.0 plugin formats.

### Current Architecture

```
src/
├── config.h                    # Compile-time MAX_* constants
├── core/                       # SSM inference engine (static lib)
│   ├── ssm_block.h/.cpp        # Single SSM layer
│   ├── ssm_inference_core.h/.cpp # Multi-layer wrapper
│   ├── weight_loader.h/.cpp    # mmap weight loader
│   ├── arena_allocator.h/.cpp  # mmap-backed bump allocator
│   └── simd_helpers.h          # SSE2/NEON dot_product, fmadd, scale
├── common/                     # Lock-free primitives (interface lib)
│   ├── ring_buffer.h           # SPSC ring buffer
│   ├── seqlock.h               # SeqLock + Seqlocked<T>
│   ├── double_buffer_seqlock.h # Double-buffered seqlock
│   └── conditioning_frame.h    # MIDI/ARA conditioning data frame
├── clap/                       # CLAP plugin (shared lib: .clap)
│   ├── clap_entry.cpp          # clap_entry factory
│   ├── clap_plugin.h/.cpp      # Main plugin class, process(), bypass, voice
│   ├── clap_audio_ports.cpp    # CLAP_EXT_AUDIO_PORTS
│   ├── clap_note_ports.cpp     # CLAP_EXT_NOTE_PORTS
│   ├── clap_params.cpp         # CLAP_EXT_PARAMS
│   ├── clap_state.cpp          # CLAP_EXT_STATE
│   ├── clap_tail.cpp           # CLAP_EXT_TAIL
│   ├── clap_latency.cpp        # CLAP_EXT_LATENCY
│   └── clap_thread_check.h     # Host thread-check helper
└── ara/                        # ARA stubs (static lib)
    ├── ara_types.h             # Stub ARA types
    ├── ara_document_controller.h/.cpp # Stub document controller
    ├── ara_playback_renderer.cpp      # Renders silence
    ├── ara_editor_renderer.cpp        # Ring-buffer audio storage
    └── ara_renderer.h          # C function pointer API
```

---

## 2. Explicit TODO/FIXME/HACK Markers

| File | Line | Marker | Severity |
|------|------|--------|----------|
| `TODO.md` | 72 | `P4-2 Replace ARA stubs with real SDK -- BLOCKED` | **BLOCKED** |
| `TODO.md` | 120 | `src/ara/ [stub] -- Custom stubs (needs real ARA SDK)` | **STUB** |
| `TODO.md` | 33-36 | P0-1 through P0-4: NumPy reference, C++ parity test, long-run stability, weight round-trip | **P0** |
| `TODO.md` | 42-45 | P1-1 through P1-4: polyphony, presets, Mamba FX, ARA Companion | **P1** |
| `TODO.md` | 51-55 | P2-1 through P2-5: SIMD B/C proj, profiling, FTZ/DAZ ARM, benchmarks | **P2** |
| `TODO.md` | 61-65 | P3-1 through P3-5: clap-validator, ASan, TSan, cross-platform, CI | **P3** |
| `TODO.md` | 82 | P5-1: GUI | **P5** |
| `TODO.md` | 88-92 | P6-1 through P6-5: README, packaging, license, parameter docs, architecture docs | **P6** |
| `AGENTS.md` | 59 | `See TODO.md for the current, verified gap analysis` | info |
| `AGENTS.md` | 55 | `ARA files are custom stubs` | known |
| `AGENTS.md` | 74 | `process_block() is a per-sample loop stub` | known |
| `tests/test_conditioning.cpp` | 3 | `// Phase 3 verification per TODO.md lines 184-211` | info |
| `scripts/reference_conditioning.py` | 8 | `Per TODO.md Phase 3` | info |
| `scripts/export_mamba_weights.py` | 60 | `# (Phase 3 of TODO.md)` | info |

---

## 3. Stub / Placeholder / Unimplemented Code

### 3.1 ARA SDK — Complete Stub Layer

| File | Line | Issue |
|------|------|-------|
| `src/ara/ara_types.h` | 32-39 | Entire `ARA::PlugIn` namespace forward-declared as empty classes |
| `src/ara/ara_types.h` | 21-23 | `#define MAMBA_ARA_REAL_SDK` conditional requires `ARA_API/ARAPlugInBase.h` |
| `src/ara/ara_document_controller.h` | 24 | `ARADocumentControllerStub` — does not inherit from `ARA::PlugIn::DocumentController` |
| `src/ara/ara_playback_renderer.cpp` | 23-29 | **Renders silence**: `ssm_block->process_sample(0.0f, buf)` — hardcoded zero input |
| `src/ara/ara_editor_renderer.cpp` | 15-35 | No waveform visualization, no editor — just stores audio in cursor ring buffer |
| `src/ara/ara_renderer.h` | 11-19 | Uses C function pointers (`void (*render)(...)`) instead of ARA class hierarchy |
| `src/ara/CMakeLists.txt` | 9-11 | Conditional ARA include path (`if(ARA_INCLUDE_DIR)`) but never links real SDK |

### 3.2 Test Stubs (Empty Test Bodies)

| File | Line | Test | Body |
|------|------|------|------|
| `tests/test_realtime_safety.cpp` | 12 | `NoSTLContainersInProcessPath` | **Empty** — `{}` |
| `tests/test_realtime_safety.cpp` | 15 | `NoMutexInProcessPath` | **Empty** — `{}` |
| `tests/test_realtime_safety.cpp` | 18 | `NoFileIOInProcessPath` | **Empty** — `{}` |

### 3.3 Dead Code via Conditional Compilation

| File | Line | Code | Issue |
|------|------|------|-------|
| `tests/test_realtime_safety.cpp` | 30-57 | `#ifdef MAMBA_REALTIME_SAFE` block (operator new/delete override + allocation test) | **Macro never defined** — never compiled or run |
| `tests/test_realtime_safety.cpp` | 46-56 | `ZeroAllocationsInProcess` test | Never executed |

### 3.4 ARM FTZ/DAZ Not Implemented

| File | Line | Issue |
|------|------|-------|
| `src/core/ssm_block.cpp` | 14-25 | `set_mxcsr_ftz_daz_impl()` — x86/x64 `ldmxcsr` only. No `__aarch64__` path |
| `TODO.md` | 55 | P2-5: `apply_ftz_daz()` ARM path — blocked on ARM hardware |

### 3.5 ARA-Free Global Helper Functions (Backward Compatibility)

| File | Line | Function | Status |
|------|------|----------|--------|
| `src/ara/ara_document_controller.h` | 64-74 | `ARA_set_tempo`, `ARA_set_key_signature`, `ARA_set_chord`, `ARA_get_conditioning`, `ARA_get_conditioning_seqlock`, `ARA_write_conditioning_to`, `ARA_get_document_controller` | Functional stubs that delegate to global singleton |

---

## 4. Specification-Implementation Gaps

### 4.1 Gaps vs ADR-001 (Triple-Thread Boundary and Lock-Free Communication)

| ADR-001 Specifies | Current Implementation | Gap |
|---|---|---|
| 3 threads: Audio, Main, **ARA** | Audio + Main present; **ARA thread has only stubs** | **No real ARA thread** — blocked on ARA SDK |
| `Seqlock` for `MambaState` snapshot | Uses `DoubleBufferSeqLock<StateMirror>` instead (valid alternative) | Architecture change from ADR but functional |
| Double-buffer swap protocol | `DoubleBufferSeqLock::swap()` exists but **never called by any thread** | State mirror always writes to slot 0 — main thread reads slot 0 (self-referential) |
| `moodycamel::ReaderWriterQueue` referenced | Uses `SPSCRingBuffer<ParamAtom, 256>` (custom implementation) | Different implementation, same semantics |
| State save: seqlock on MambaState | State save uses DoubleBufferSeqLock with `state_mirror` snapshot | Works, but swap never happens so snapshot is always of the same buffer the audio thread just wrote to (racy) |

### 4.2 Gaps vs ADR-002 (Memory Architecture and Weight Storage)

| ADR-002 Specifies | Current Implementation | Gap |
|---|---|---|
| `config.h` generated by ML export tool | Config.h is hand-written; export tool CAN generate it | Both exist; must keep in sync manually |
| `mmap` read-only weight file + pointer casting | `WeightLoader` uses `mmap` but **copies into stack-allocated `SSMWeights`** | Could use mmap'd pointers directly instead of memcpy |
| Cache-aligned weight file | Header padded to 64 bytes | ok |
| Weight file magic + version validation | Implemented (`0x4D424D41`, version 1) | ok |

### 4.3 Gaps vs ADR-003 (Processing Modes and ARA Bridge)

| ADR-003 Specifies | Current Implementation | Gap |
|---|---|---|
| `process_tiny()` for Mode A (ultra-low latency) | Only `process_sample()` and `process_block()` exist | No `process_tiny()` function |
| Explicit Mode A vs Mode B selection | No selection — `process()` always calls per-sample | Mode B never used in plugin path |
| ARA thread computes Δ/B biases (precomputed) | Audio thread does the projection in `SSMBlock::process_sample()` | Different design; accepted as "cheap enough" per ADR-003 note |
| Conditioning modulates Δ, B, **and C** | Only Δ and B are modulated; C is **not conditioned** | `cond_proj_C` does not exist; `SSMWeights` has no field for it |
| `W_delta` and `W_bias` as matrices | `cond_proj_dt` and `cond_proj_B` are per-state vectors `[d_state, cond_dim]` | Different shape; each state gets its own projection |

### 4.4 Gaps vs T2/T3 Implementation Prompts

| Prompt Specifies | Current Implementation | Gap |
|---|---|---|
| T2b: "All three plugins (Mamba Synth, Mamba FX, ARA Companion)" | **Only Mamba Synth exists** | Mamba FX and ARA Companion are not started |
| T2b: "CLAP extension: gui" | `plugin_get_extension` returns **nullptr for `CLAP_EXT_GUI`** | No GUI at all |
| T2b: "Note-to-conditioning mapping via learned projections" | Simple key/velocity mapping in `refresh_conditioning()` | **No learned projections** for note->conditioning |
| T2d: "GUI framework: knobs, sliders, buttons, dropdowns" | **Nothing exists** | No GUI, no widget system, no layout |
| T2d: "Preset management with user-preset saving/loading, factory presets" | Only `clap_state.cpp` binary serialization | No preset manager, no factory presets |
| T2a: "Variable state dimensions for different model sizes" | `config.h` hardcodes `MAX_*` constants | Requires recompilation to change model size |
| T2a: "SIMD vectorization" | SIMD used only in `simd_helpers.h` (dot_product, fmadd, scale) | B_proj/C_proj use scalar loops — blocked on weight layout refactor |
| T2c: "Validation script shows numerical equivalence (PyTorch vs C++ inference)" | `validate_export.py` exists | Only validates weight round-trip, **not inference waveform comparison** |
| T3b: "Unit test file (test_clap_note_ports.cpp)" | **Does not exist** | No dedicated CLAP note ports test |
| T1.5: "Memory ownership matrix" | **Not documented** | No ownership document |
| T3c: "WEIGHT_FORMAT_SPEC.md" | **Does not exist** | Weight format is only implicitly defined |
| T2a: "Bit-exact reproducibility across platforms" | No cross-platform CI | Not verified |
| T3a: "Benchmark both block and per-sample modes" | `benchmark_ssm_block.cpp` exists | ok |

### 4.5 CLAP Extension Status

| Extension | Required by T2b | Status | File |
|-----------|---------------|--------|------|
| `CLAP_EXT_AUDIO_PORTS` | Yes | ✅ Working | `clap_audio_ports.cpp` |
| `CLAP_EXT_NOTE_PORTS` | Yes | ✅ Working | `clap_note_ports.cpp` |
| `CLAP_EXT_PARAMS` | Yes | ✅ Working | `clap_params.cpp` |
| `CLAP_EXT_STATE` | Yes | ✅ Working | `clap_state.cpp` |
| `CLAP_EXT_TAIL` | Yes | ✅ Working | `clap_tail.cpp` |
| `CLAP_EXT_LATENCY` | Yes | ✅ Working | `clap_latency.cpp` |
| `CLAP_EXT_THREAD_CHECK` | Yes | ✅ Working (host extension) | `clap_thread_check.h` |
| `CLAP_EXT_GUI` | Yes | ❌ **Not implemented** | Returns nullptr |

### 4.6 ARA 2.0 Interface Status

| Interface | Required by T2b | Status |
|-----------|---------------|--------|
| `ARA::PlugIn::PlugIn` | Yes | ❌ **Stub** |
| `DocumentController` | Yes | ❌ **Stub** (`ARADocumentControllerStub`) |
| `PlaybackRenderer` | Yes | ❌ **Stub** (renders silence) |
| `EditorRenderer` | Yes | ❌ **Stub** (audio ring buffer only) |

---

## 5. Missing Functionality and Components

### 5.1 Completely Missing Components

| Component | Priority | Required By | Notes |
|-----------|----------|-------------|-------|
| **GUI/UI** (`CLAP_EXT_GUI`) | P5 | T2d, host requirement | No knobs, no sliders, no visualization |
| **Mamba FX plugin** (long-context neural audio effect) | P1-3 | T2b spec | Not started |
| **ARA Generative Companion plugin** (offline region generation) | P1-4 | T2b spec | Blocked on ARA SDK |
| **Preset management system** (user presets, factory presets, categories) | P1-2 | T2d spec | Only raw binary state save/load |
| **Polyphonic voice management** (currently monophonic) | P1-1 | T3b spec | `ActiveNote` struct supports only 1 voice |
| **NumPy/PyTorch reference forward pass** for C++ parity test | P0-1 | P0-2 | Would be in `scripts/reference_forward.py` |
| **C++ output vs NumPy reference unit test** at 1e-5 tolerance | P0-2 | P0-1 | Cannot exist without P0-1 |
| **Long-run stability test** (10+ min, no NaN/inf/state explosion) | P0-3 | — | Currently `test_stability.cpp` runs 300s at 44.1kHz — **DOES exist**, listed as done in tests |
| **C++ load + inference vs PyTorch round-trip test** | P0-4 | — | Missing |
| **clap-validator pass** | P3-1 | Release | Tool not installed |
| **Cross-platform builds** (macOS, Windows) | P3-4 | Release goal | No CI |
| **macOS bundle packaging** (`Info.plist`) | P6-2 | Release | Not started |
| **Windows packaging** | P6-2 | Release | Not started |
| **LICENSE file** | P6-3 | Release | Missing |
| **Parameter reference documentation** | P6-4 | Release | Missing |
| **Architecture overview documentation** | P6-5 | Release | Missing |
| **`test_clap_note_ports.cpp`** | — | T3b output | Missing |
| **`WEIGHT_FORMAT_SPEC.md`** | — | T3c output | Missing |

### 5.2 Partial / Broken Components

| Component | Status | Issue |
|-----------|--------|-------|
| `SSMBlock::process_block()` | Works but **not vectorized** | Still a per-sample loop inside each chunk; no SIMD block processing |
| `SSMInferenceCore::process_block()` | Working | Stack allocates `buf_dm[MAX_BLOCK_SIZE * MAX_D_MODEL]` (32 KB) — safe but fragile |
| `DoubleBufferSeqLock::swap()` | **Never called** | Audio thread writes to slot 0, main thread reads slot 0 — swap should be called after every process() cycle to give main thread a consistent back-buffer snapshot |
| `plugin_process()` in `clap_plugin.cpp` | Always processes per-sample | Never calls `process_block()` — **Mode B is unused in the plugin** despite being implemented in the core |
| `MambaSynth::load_weights()` | **Conditioning weights not copied** | Lines 616-627 skip `cond_proj_dt`, `cond_proj_B`, `cond_dt_bias`, `cond_B_bias` |
| `WeightLoader` path resolution | No default path | `load()` without args works only via `MAMBA_WEIGHTS` env var; no compiled-in fallback |

### 5.3 Test Coverage Gaps

| Area | Test File | Tests | Gaps |
|------|-----------|-------|------|
| SSM Block | `test_ssm_block.cpp` | 9 | No NumPy reference comparison; no process_block output verification beyond NaN/Inf; no conditioning projection test at SSMBlock level (only at math level) |
| Weight Loader | `test_weight_loader.cpp` | 21 | None — comprehensive |
| Conditioning | `test_conditioning.cpp` | 3 | Seqlock atomicity, modulation detection, Python math reference — good |
| Stability | `test_stability.cpp` | 3 | Long-run, extreme params, zero-input — good |
| RT Safety | `test_realtime_safety.cpp` | 6 | **3 empty tests** (stubs); `MAMBA_REALTIME_SAFE` guard never defined |
| Benchmark | `benchmark_ssm_block.cpp` | 0 (main()) | Manual run, no assertions |
| CLAP note ports | **None** | 0 | `test_clap_note_ports.cpp` does not exist |
| CLAP state round-trip | **None** | 0 | No state save/load test |
| CLAP plugin integration | **None** | 0 | No end-to-end test |
| ARA bridge | **None** | 0 | No ARA test |

---

## 6. Critical Bug: Conditioning Weights Not Loaded

**Location:** `src/clap/clap_plugin.cpp`, lines 616-627

**The bug:** `MambaSynth::load_weights()` copies all weight arrays from `WeightLayerPointers` to `SSMWeights`, but **omits 4 conditioning arrays**:

```cpp
// Lines 616-627 — what IS copied:
std::memcpy(w.A_log, ptrs.A_log, ...);
std::memcpy(w.D_proj, ptrs.D_proj, ...);
std::memcpy(w.dt_bias, ptrs.dt_bias, ...);
std::memcpy(w.dt_proj_w, ptrs.dt_proj_w, ...);
std::memcpy(w.B_proj_w, ptrs.B_proj_w, ...);
std::memcpy(w.C_proj_w, ptrs.C_proj_w, ...);
std::memcpy(w.conv1d_w, ptrs.conv1d_w, ...);
std::memcpy(w.conv1d_b, ptrs.conv1d_b, ...);
std::memcpy(w.W_in, ptrs.W_in, ...);
std::memcpy(w.W_out, ptrs.W_out, ...);
std::memcpy(w.W_ssm, ptrs.W_ssm, ...);
std::memcpy(w.b_in, ptrs.b_in, ...);
// NOT COPIED:
// w.cond_proj_dt — always zero
// w.cond_proj_B  — always zero
// w.cond_dt_bias — always zero
// w.cond_B_bias  — always zero
```

**Impact:** Conditioning projections are **always zero** regardless of weight file contents. `conditioning_active_` will never trigger because `set_conditioning()` checks for non-zero vector, but the weights that would modulate it are zero. Conditioning is effectively non-functional when weights are loaded from file.

---

## 7. Code Quality Issues

### 7.1 Architecture & Safety

| File | Line | Issue | Severity |
|------|------|-------|----------|
| `clap_plugin.cpp` | 640-641 | Arena allocator `allocate()` has no matching `deallocate()` — relies on OS process cleanup | Medium |
| `ara_playback_renderer.cpp` | 19 | Placement new into `ssm_storage` within heap-allocated context — leak if create/destroy mismatch | Medium |
| `ara_editor_renderer.cpp` | 21-24 | `std::aligned_alloc` / `_aligned_malloc` mismatch: `std::free` may not work with aligned allocations | Medium |
| `ssm_inference_core.cpp` | 85-86 | Stack allocation of `buf_dm[MAX_BLOCK_SIZE * MAX_D_MODEL]` = 8192 floats (32 KB) | Low (safe with MAX=512) |
| `double_buffer_seqlock.h` | 60-64 | `read()` does not check generation counter change during memcpy — technically racy if swap happens mid-read | Medium |

### 7.2 Correctness

| File | Line | Issue | Severity |
|------|------|-------|----------|
| `clap_plugin.cpp` | 616-627 | Conditioning weights not copied in `load_weights()` | **High** |
| `clap_plugin.cpp` | 526 | `process()` always calls `process_sample()` — **never `process_block()`** | Medium (performance) |
| `ssm_block.cpp` | 94 | `W_in[j][k] * input` — same scalar for all d_model channels; degenerate for d_model>1 | Low (works for d_model=16) |
| `ssm_inference_core.cpp` | 62-69 | `process_sample()` sums d_model to mono between layers — discards d_model structure | Architecture choice |
| `ssm_block.cpp` | 169 | `conv_out_[i]` as SSM input for state i — takes first d_state elements of d_inner conv output | Correct per Mamba design |

### 7.3 DoubleBufferSeqLock Swap Bug

**The swap mechanism** in `DoubleBufferSeqLock` is designed so the main thread calls `swap()` to flip which slot the audio thread writes to, giving the main thread a consistent back-buffer to read. However:

1. The audio thread calls `begin_write()` / `end_write()` in `publish_state_mirror()` 
2. `swap()` is **never called** by any code path
3. Therefore the audio thread always writes to slot 0
4. The main thread always reads slot 0 — potentially reading data while the audio thread is mid-write

This is a **functional race condition** on state saves.

### 7.4 Python Script Issues

| File | Line | Issue |
|------|------|-------|
| `scripts/export_mamba_weights.py` | 88 | `u[:, :d_state]` hardcoded — takes first d_state channels of d_inner |
| `scripts/export_mamba_weights.py` | 68-97 | Selective scan implemented as Python loop — fine for export but extremely slow |
| `scripts/export_mamba_weights.py` | 179-180 | Variable `b` from `model.layers[0]` overwritten by loop variable in `_random_init` | Minor |

### 7.5 ArenaAllocator

| File | Line | Issue |
|------|------|-------|
| `src/core/arena_allocator.h/cpp` | — | Only used for `MambaSynth` and `clap_plugin_t` allocation. No deallocation support |

---

## 8. Blocked Items

| Blocked Item | Blocker | Resolution Path | Priority |
|-------------|---------|-----------------|----------|
| P4-1 through P4-6 (ARA SDK integration) | Celemony proprietary SDK license | Request ARA SDK from [celemony.com](https://celemony.com) | P4 |
| P3-1 (clap-validator) | Tool not installed | Clone from [github.com/free-audio/clap](https://github.com/free-audio/clap), build clap-validator | P3 |
| P2-1 (SIMD B/C projection refactor) | Strided `[d_inner][d_state]` weight layout | Refactor to `[d_state][d_inner]` across SSMWeights, WeightLoader, export_mamba_weights.py | P2 |
| P2-5 (ARM FTZ/DAZ) | No ARM build target | Cross-compile or test on ARM hardware | P2 |
| P3-4 (cross-platform builds) | No macOS/Windows hardware | CI (GitHub Actions) or VMs | P3 |

---

## 9. Completed (Verified Working)

### 9.1 Core Engine

- ✅ `SSMBlock` — correct Mamba math: ZOH discretization, selective scan, SiLU activation, causal conv1d, d_model output
- ✅ `SSMInferenceCore` — multi-layer wrapper with placement new, get/set state
- ✅ `SSMWeights` — all 16 weight arrays defined (A_log, D_proj, dt_bias, dt_proj_w, B_proj_w, C_proj_w, conv1d_w, conv1d_b, W_in, W_out, W_ssm, b_in, cond_proj_dt, cond_proj_B, cond_dt_bias, cond_B_bias)
- ✅ Periodic state normalization (every 4096 samples)
- ✅ Conditioning projection applied in SSM step

### 9.2 Lock-Free Primitives

- ✅ `SPSCRingBuffer<ParamAtom, 256>` — lock-free SPSC queue
- ✅ `SeqLock` / `Seqlocked<T>` — seqlock with acquire/release
- ✅ `DoubleBufferSeqLock<T>` — double-buffered snapshot (swap mechanism exists but unused)
- ✅ `ConditioningFrame` / `ARAConditioningFrame` — 16-float aligned frame

### 9.3 Weight Loading

- ✅ `WeightLoader` — mmap-based, magic/version validation, per-layer pointer extraction
- ✅ FP16 -> FP32 dequantization (IEEE 754)
- ✅ INT8 per-layer dequantization (scale + zero_point)
- ✅ Path resolution: explicit arg, MAMBA_WEIGHTS env var, default

### 9.4 CLAP Plugin

- ✅ `clap_entry` — factory with `CLAP_EXPORT`
- ✅ Audio ports — stereo + mono in/out pairs
- ✅ Note ports — MIDI + CLAP dialect
- ✅ Params — count, get_info, get_value, value_to_text, text_to_value, flush
- ✅ State — binary save/load with magic, version, schema, parameter/state/weight path
- ✅ Tail — 100ms at current sample rate
- ✅ Latency — `MAX_D_CONV - 1` samples
- ✅ Thread-check — host extension with thread::id fallback
- ✅ Bypass — 4-state crossfade machine (processing, fading out, bypassed, fading in)
- ✅ Voice envelope — ADSR-like (attack, sustain, release, silent)
- ✅ Parameter smoothing — one-pole lowpass
- ✅ Conditioning — reads ARA seqlock + MIDI note, writes to SSM core
- ✅ Arena allocation for plugin structs

### 9.5 Tests

- ✅ 33/33 tests passing
- ✅ `test_weight_loader.cpp` — 21 tests (load, validate, get pointers, multi-layer, bad magic/version/size, FP16 round-trip, INT8 dequant, path resolution, dimension bounds)
- ✅ `test_ssm_block.cpp` — 9 tests (reset, process_block output, dimensions, leak, weight differentiation, conditioning crash, multi-layer, state round-trip)
- ✅ `test_conditioning.cpp` — 3 tests (modulation, seqlock atomicity, Python reference)
- ✅ `test_stability.cpp` — 3 tests (300s long-run, extreme params, zero-input)

### 9.6 Benchmark

- ✅ `benchmark_ssm_block.cpp` — per-sample and block modes, single and multi-layer
- ✅ Single layer: 0.65 us/sample (15x below 10us target)
- ✅ 2 layers: 1.33 us/sample

### 9.7 ML Pipeline

- ✅ `export_mamba_weights.py` — PyTorch -> flat binary with `--random-init`, `--load-checkpoint`, config.h generation
- ✅ `validate_export.py` — binary -> PyTorch round-trip validation
- ✅ `reference_conditioning.py` — standalone conditioning projection reference

---

## 10. All Source Files — Complete Inventory

### 10.1 Source

| File | Lines | Status | Purpose |
|------|-------|--------|---------|
| `src/config.h` | 18 | ✅ Done | Compile-time MAX_* constants |
| `src/core/ssm_block.h` | 105 | ✅ Done | SSMBlock class declaration |
| `src/core/ssm_block.cpp` | 358 | ✅ Done | SSM math, process_sample/block, conditioning |
| `src/core/ssm_inference_core.h` | 65 | ✅ Done | SSMInferenceCore class declaration |
| `src/core/ssm_inference_core.cpp` | 138 | ✅ Done | Multi-layer wrapper, process_sample/block |
| `src/core/weight_loader.h` | ~200 | ✅ Done | WeightFileHeader, WeightLayerPointers, FP16/INT8 |
| `src/core/weight_loader.cpp` | ~300 | ✅ Done | mmap, validate, dequantize |
| `src/core/arena_allocator.h` | ~50 | ✅ Done | Arena allocator declaration |
| `src/core/arena_allocator.cpp` | ~60 | ✅ Done | mmap-backed bump allocator |
| `src/core/simd_helpers.h` | ~100 | ✅ Done | SSE2/NEON intrinsics |
| `src/common/seqlock.h` | 66 | ✅ Done | SeqLock + Seqlocked<T> |
| `src/common/ring_buffer.h` | 70 | ✅ Done | SPSCRingBuffer |
| `src/common/double_buffer_seqlock.h` | 121 | ✅ Done | DoubleBufferSeqLock |
| `src/common/conditioning_frame.h` | 59 | ✅ Done | ConditioningFrame + ARAConditioningFrame |
| `src/clap/clap_entry.cpp` | 62 | ✅ Done | clap_entry factory |
| `src/clap/clap_plugin.h` | 168 | ✅ Done | MambaSynth struct, enum, ParamSmooth |
| `src/clap/clap_plugin.cpp` | 680 | ✅ Done | init, destroy, activate, process, bypass, voice, load_weights |
| `src/clap/clap_audio_ports.cpp` | 69 | ✅ Done | Stereo + mono ports |
| `src/clap/clap_note_ports.cpp` | 28 | ✅ Done | Note input port |
| `src/clap/clap_params.cpp` | 292 | ✅ Done | Parameter info, value, text, flush |
| `src/clap/clap_state.cpp` | 210 | ✅ Done | Binary state save/load |
| `src/clap/clap_tail.cpp` | 29 | ✅ Done | Tail length |
| `src/clap/clap_latency.cpp` | 21 | ✅ Done | Latency reporting |
| `src/clap/clap_thread_check.h` | 91 | ✅ Done | Host thread-check helper |
| `src/ara/ara_types.h` | 80 | 🟡 **Stub** | ARA SDK forward declarations |
| `src/ara/ara_renderer.h` | 31 | 🟡 **Stub** | C function pointer renderer API |
| `src/ara/ara_document_controller.h` | 78 | 🟡 **Stub** | ARADocumentControllerStub |
| `src/ara/ara_document_controller.cpp` | 141 | 🟡 **Stub** | Global singleton + free functions |
| `src/ara/ara_playback_renderer.cpp` | 43 | 🟡 **Stub** | Renders silence |
| `src/ara/ara_editor_renderer.cpp` | 52 | 🟡 **Stub** | Ring buffer storage |

### 10.2 Tests

| File | Lines | Status | Tests |
|------|-------|--------|-------|
| `tests/test_ssm_block.cpp` | 182 | ✅ Done | 9 |
| `tests/test_weight_loader.cpp` | 634 | ✅ Done | 21 |
| `tests/test_conditioning.cpp` | 319 | ✅ Done | 3 |
| `tests/test_stability.cpp` | 219 | ✅ Done | 3 |
| `tests/test_realtime_safety.cpp` | 67 | 🟡 **3 empty stubs** | 6 (3 empty) |
| `tests/benchmark_ssm_block.cpp` | 215 | ✅ Done | 0 (manual) |

### 10.3 Scripts

| File | Lines | Status | Purpose |
|------|-------|--------|---------|
| `scripts/export_mamba_weights.py` | 288 | ✅ Done | PyTorch -> flat binary |
| `scripts/validate_export.py` | ~150 | ✅ Done | Binary <-> PyTorch round-trip |
| `scripts/reference_conditioning.py` | 132 | ✅ Done | Standalone conditioning math ref |

### 10.4 Build System

| File | Purpose |
|------|---------|
| `CMakeLists.txt` | Root project (C++20, FetchContent googletest) |
| `src/core/CMakeLists.txt` | Static lib `core` |
| `src/common/CMakeLists.txt` | Interface lib `common` |
| `src/clap/CMakeLists.txt` | Shared lib `mamba_synth` (.clap) |
| `src/ara/CMakeLists.txt` | Static lib `ara_plugin` |
| `tests/CMakeLists.txt` | 4 test executables + benchmark |

---

## 11. Tests — Status and Coverage

### 11.1 Current Test Results

```
33/33 passing
Benchmark: 0.651 us/sample single-layer, 1.33 us/sample 2-layer
Export round-trip: PASSED
```

### 11.2 Test File Breakdown

| Test File | Test Count | Quality | Gaps |
|-----------|-----------|---------|-------|
| `test_weight_loader` | 21 | Excellent | None significant |
| `test_ssm_block` | 9 | Good | No NumPy reference comparison; no process_block d_model output verification beyond NaN/Inf |
| `test_conditioning` | 3 | Good | Seqlock atomicity, modulation detection, Python math ref |
| `test_stability` | 3 | Good | 300s long-run, extreme params, zero-input |
| `test_realtime_safety` | 6 | ❌ Bad | 3 empty stubs; 1 guarded by never-defined macro |
| `benchmark_ssm_block` | 0 | N/A | Manual run, pass/fail print |

### 11.3 Missing Test Coverage

| Area | Priority | Exists? |
|------|----------|---------|
| CLAP state save/load round-trip | High | ❌ |
| CLAP note ports event handling | High | ❌ |
| CLAP plugin integration (end-to-end) | High | ❌ |
| C++ vs PyTorch inference parity | High | ❌ |
| DoubleBufferSeqLock concurrent access | Medium | ❌ |
| Weight path resolution (env var) | Medium | ✅ |
| Conditioning projection in SSMBlock | Medium | ❌ (only at math level) |
| ARA conditioning bridge | Low | ❌ (ARA is stubs) |
| Polyphonic voice handling | Low | ❌ (not implemented yet) |
| Non-real-time safety (via static analysis) | Medium | ❌ (3 stubs) |

---

## 12. Priority Remediation Plan

### P0 — Immediate (Current Release Blockers)

1. **Fix `load_weights()` conditioning weights** — Add the 4 missing `memcpy` calls for `cond_proj_dt`, `cond_proj_B`, `cond_dt_bias`, `cond_B_bias`
2. **Fix `DoubleBufferSeqLock::swap()`** — Call `swap()` after `end_write()` in `publish_state_mirror()`, or restructure to use `Seqlocked<StateMirror>` directly without double-buffer
3. **Write NumPy reference forward pass** (`scripts/reference_forward.py`) for C++ numerical parity testing

### P1 — High (Core Functionality)

4. **Implement polyphonic voice management** — Multiple `ActiveNote` slots, round-robin SSM state per voice
5. **Use `process_block()` in plugin** — `plugin_process()` should call `core.process_block()` instead of per-sample loop
6. **Add `process_tiny()`** — Explicit Mode A implementation for ultra-low latency
7. **Fill 3 empty test stubs** in `test_realtime_safety.cpp` or remove them

### P2 — Performance

8. **Refactor `SSMWeights` layout** — Transpose `B_proj_w` and `C_proj_w` from `[d_inner][d_state]` to `[d_state][d_inner]` for SIMD-friendly access in `process_sample()`
9. **Update `WeightLoader` + export script** to match new layout
10. **Implement ARM FTZ/DAZ** in `set_mxcsr_ftz_daz_impl()`

### P3 — Integration & Tooling

11. **Install and run `clap-validator`**
12. **Add GitHub Actions CI** for Linux build + test
13. **Add Valgrind/ASan pass** on the process() path
14. **Write `test_clap_note_ports.cpp`** and `test_state_roundtrip.cpp`

### P4 — ARA SDK (Blocked)

15. Obtain ARA SDK from Celemony
16. Replace `ARADocumentControllerStub` with real `ARA::PlugIn::DocumentController`
17. Implement real `PlaybackRenderer` and `EditorRenderer`

### P5 — GUI

18. Build GUI plugin view — at minimum parameter knobs + bypass toggle
19. Implement `CLAP_EXT_GUI` returning a valid extension struct

### P6 — Documentation & Packaging

20. Create `WEIGHT_FORMAT_SPEC.md`
21. Create `LICENSE` file
22. Write proper README with build/install instructions
23. Add macOS `Info.plist` generation
24. Write parameter reference docs + architecture overview

---

*Generated by automated codebase audit. For questions, refer to the source files listed above or the `TODO.md` file.*
