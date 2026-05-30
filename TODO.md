# Mamba-SSM CLAP/ARA Plugin Suite — Verified End State TODO

## Gap Summary

**Current completion: ~15-20%**

Completed artifacts:
- `ArenaAllocator` (mmap-backed bump allocator)
- `SPSCRingBuffer` (lock-free SPSC queue with acquire/release)
- `SeqLock` / `Seqlocked<T>` (seqlock with write_lock/write_unlock/read_begin/read_retry)
- 6 smoke tests passing (NaN/inf checks, reset, non-zero output only)

**Everything else is stub, scaffold, or simplified approximation.**

### Critical Gaps

| # | Gap | Why Critical |
|---|-----|-------------|
| C1 | **SSM math is NOT the Mamba algorithm** | `process_sample()` uses crude Euler approximation `h_new = h*(1-dt) + Bx*dt`. No proper discretization of A,B with delta, no selective scan, no input/output projection. |
| C2 | **No weight loader** | No mmap-based flat-binary loader, no validation, no dequantization. SSM block uses hardcoded defaults with zero mechanism to load trained weights. |
| C3 | **No ML export pipeline** | No `export_mamba_weights.py`, no weight format spec, no `validate_export.py`, no `config.h` generator. No trained model can flow from PyTorch to C++. |
| C4 | **No real CLAP SDK** | Uses a hand-rolled `clap.h` that deviates from the actual CLAP spec (no `get_extension` lookup, wrong vtable layout). Even if compiled, no host could load it. |
| C5 | **No `clap_entry` factory** | No shared library entry point, no `.clap` bundle. Plugin cannot be loaded by any host. |
| C6 | **Conditioning pipeline is disconnected** | ARA thread can write conditioning data; audio thread never reads it. `set_conditioning()` is a no-op. Core value proposition (ARA modulating SSM params) is absent. |

### High Gaps

| # | Gap |
|---|-----|
| H1 | CLAP `activate`/`deactivate`/`start_processing`/`stop_processing` are all `nullptr` |
| H2 | No parameter smoothing (values applied as step functions, will produce clicks) |
| H3 | Bypass mutes output instead of crossfading (clicks on bypass) |
| H4 | `note_on_handler` doesn't produce conditioning vector; note events not processed in `process()` |
| H5 | State serialization saves only 16 floats + bypass; no h_state snapshot, no weight path |
| H6 | No voice management / polyphony (just an atomic counter) |
| H7 | No ARA SDK integration (custom stubs, not `ARA::PlugIn` namespace) |
| H8 | No double-buffer state snapshot for audio->main thread (ADR-001 spec) |

### Medium Gaps

| # | Gap |
|---|-----|
| M1 | `process_block()` is just a loop over `process_sample()` — no block-optimized path |
| M2 | No SIMD vectorization (scalar loops only) |
| M3 | No numerical stabilization / periodic state normalization |
| M4 | No mathematical correctness tests vs NumPy/PyTorch reference |
| M5 | No benchmarking (<10us/sample target) |
| M6 | No RT safety validation (Valgrind/ASan/TSan) |
| M7 | `config.h` is hand-written, not generated from model |
| M8 | `MambaState` missing batch dimension per integration contract |
| M9 | Conditioning frame seqlock used per-field instead of per-frame (not atomic read) |
| M10 | No FTZ/DAZ in `process()` callback (only in SSM block) |
| M11 | Mamba FX and ARA Generative Companion plugins not started |
| M12 | No GUI |
| M13 | No preset management |

---

## Dependency Chain

```
Phase 0 (CLAP SDK) ──> Phase 4 (CLAP completeness) ──> Phase 5 (ARA)
Phase 1 (SSM math) ──> Phase 2 (weights) ──> Phase 3 (conditioning) ──┘
Phase 1 ──────────────────────────────────────> Phase 6 (validation)
Phase 4 + Phase 5 ──────────────────────────> Phase 7 (packaging)
```

Phases 0, 1, and 2 can run in parallel.
Phase 3 depends on 1+2.
Phase 4 depends on 0+1+2.
Phase 5 depends on 4+3.
Phase 6 depends on 1+4.
Phase 7 depends on 4+5.

---

## Phase 0: Fix the Foundation

### 0.1 Replace hand-rolled CLAP headers with official SDK
- [ ] Clone CLAP SDK from `github.com/free-audio/clap`
- [ ] Set `CLAP_PATH` env var to SDK location
- [ ] Remove custom `clap.h` and any hand-rolled CLAP type definitions
- [ ] Update all `#include` paths to use official CLAP headers

### 0.2 Update CMakeLists.txt for real CLAP SDK
- [ ] Add CLAP SDK include directory from `CLAP_PATH`
- [ ] Add `.clap` bundle target (shared library + `entry` symbol)
- [ ] Add `clap/` and `ara/` subdirectories back into build
- [ ] Add macOS `.clap` bundle `Info.plist` generation

### 0.3 Fix all type/layout mismatches from SDK transition
- [ ] Replace direct extension pointers with `get_extension` lookup mechanism
- [ ] Fix `clap_plugin_t` vtable to match official CLAP 1.1+ layout
- [ ] Fix `clap_plugin_descriptor_t` fields (add `id`, `version`, `description`, etc.)
- [ ] Fix `clap_host_t` usage to match real host interface

### 0.4 Implement `clap_entry` plugin factory
- [ ] Implement `clap_entry` with `init()`, `deinit()`, `get_factory()` returning `CLAP_PLUGIN_FACTORY`
- [ ] Export `clap_entry` symbol from shared library
- [ ] Return plugin descriptor(s) from factory

### 0.5 Wire core lifecycle callbacks
- [ ] Implement `activate()` — allocate SSM state, set sample rate + block size
- [ ] Implement `deactivate()` — free SSM state
- [ ] Implement `start_processing()` — enable FTZ/DAZ in MXCSR
- [ ] Implement `stop_processing()` — restore MXCSR

**Acceptance**: `clap-info` or `clap-validator` can scan the `.clap` bundle and report the plugin descriptor without errors.

---

## Phase 1: SSM Core Correctness

### 1.1 Implement proper Mamba discretization
- [ ] Implement `A_bar = exp(delta * A)` discretization
- [ ] Implement `B_bar = delta * B` discretization
- [ ] Implement selective scan with input-dependent delta (core Mamba algorithm)
- [ ] Replace current Euler approximation `h_new = h*(1-dt) + Bx*dt` with correct recurrence

### 1.2 Add input/output projections
- [ ] Implement input projection: `X_tilde = W_in @ X + b_in`
- [ ] Implement output projection: `Y = W_out @ Y_tilde + D * X`

### 1.3 Implement multi-layer SSMInferenceCore
- [ ] Create `SSMInferenceCore` class wrapping N `SSMBlock` instances
- [ ] Implement `initialize(AudioConfig)` — pre-allocate all working memory
- [ ] Implement `process_sample()` / `process_block()` chaining layers
- [ ] Implement `set_layer_parameters(layer_idx, SSMParams)`
- [ ] Implement `get_state()` / `set_state()` for state snapshot/restore

### 1.4 Reference correctness verification
- [ ] Write NumPy reference implementation of the identical Mamba forward pass
- [ ] Generate reference test vectors (input + weights -> expected output) with NumPy
- [ ] Unit test: C++ output vs NumPy reference within 1e-5 tolerance
- [ ] Unit test: dequantized weights match PyTorch float32 output within quantization tolerance

### 1.5 Numerical stabilization
- [ ] Add periodic state normalization (every N samples, rescale h to prevent drift)
- [ ] Long-run test: process 10 minutes of audio, verify no NaN/inf/state explosion

### 1.6 Benchmarking
- [ ] Create `benchmark_ssm_block.cpp` with microsecond-precision timing
- [ ] Measure us/sample at 44.1kHz and 48kHz
- [ ] Verify <10us/sample target

**Acceptance**: `test_ssm_block` passes reference-comparison tests at 1e-5 tolerance; benchmark reports <10us/sample.

---

## Phase 2: Weight Loading Pipeline

### 2.1 Weight format specification
- [ ] Write `WEIGHT_FORMAT_SPEC.md`: header (magic `0x4D424D41`, version, dims) + per-layer weight blobs in deterministic order
- [ ] Define header struct: magic, version, state_dim, d_model, num_layers, expansion_factor, conv_kernel_size, total_weight_data_size
- [ ] Define per-layer weight order: A, B, C, D, dt_proj, W_in, b_in, W_out, conv1d

### 2.2 C++ weight loader
- [ ] Implement `WeightLoader` class: mmap read-only, validate header (magic + version + bounds)
- [ ] Extract per-layer weight pointers from mmap region
- [ ] Support configurable weight file path (env var + CLAP state)
- [ ] Return `const float*` / `const int8_t*` views into mmap — zero copy

### 2.3 Quantization support
- [ ] Implement INT8 dequantization: `float = scale * int8 + zero_point`
- [ ] Implement FP16 dequantization: `float = fp16_to_fp32(half)`
- [ ] Support per-layer and per-tensor quantization metadata in weight format

### 2.4 Python export pipeline
- [ ] Create `scripts/export_mamba_weights.py`: PyTorch Mamba model -> flat-binary file
- [ ] Implement INT8/FP16 quantization in export script (target >60dB SNR)
- [ ] Auto-generate `config.h` with MAX_* constants from exported model dimensions
- [ ] Generate deterministic binary output (fixed endianness, no Python pickling)
- [ ] Create `scripts/validate_export.py`: load binary in Python, compare against PyTorch forward pass

### 2.5 Round-trip integration test
- [ ] Export a small test model (e.g., d_state=16, d_model=32, 1 layer) to `models/test_model.bin`
- [ ] Load model in C++ via WeightLoader
- [ ] Run inference with same input as Python
- [ ] Verify C++ output matches PyTorch output within tolerance

**Acceptance**: Round-trip PyTorch export -> C++ load -> inference produces identical output (within quantization tolerance). `models/test_model.bin` exists and loads successfully.

---

## Phase 3: Conditioning System

### 3.1 Fix ConditioningFrame atomicity
- [ ] Wrap entire ConditioningFrame read/write in single seqlock transaction (not per-field)
- [ ] Fix ARA document controller to write all fields within one `write_lock`/`write_unlock` pair
- [ ] Fix audio thread to read entire frame with `read_begin`/copy/`read_retry` loop

### 3.2 Connect conditioning to audio thread
- [ ] Add ConditioningFrame reader in `process()` callback
- [ ] Pass conditioning data to `SSMBlock::set_conditioning()` (currently a no-op)

### 3.3 Implement learned projection layer
- [ ] Implement: `delta_effective = delta_base + W_delta @ cond`
- [ ] Implement: `B_effective = B_base + W_bias @ cond`
- [ ] Add projection weight matrices to weight file format
- [ ] Export projection weights from PyTorch as part of export pipeline

### 3.4 Wire conditioning into SSM parameter update
- [ ] Implement `set_conditioning()` in SSMBlock: apply projection, compute effective params
- [ ] Update effective params each block (or each sample for Mode A)

### 3.5 Unit tests
- [ ] Test: set conditioning frame -> verify SSM params change accordingly
- [ ] Test: conditioning frame seqlock atomicity (no torn reads under concurrent write)
- [ ] Test: projection math matches Python reference

**Acceptance**: Changing tempo/key/chord in ConditioningFrame produces measurable and correct change in SSM output. Seqlock reads never return torn frames.

---

## Phase 4: CLAP Plugin Completeness

### 4.1 Wire CLAP audio-ports extension
- [ ] Implement `get_extension` returning `clap_plugin_audio_ports_t`
- [ ] Define mono + stereo input/output port pairs
- [ ] Report latency to host (algorithmic delay from conv1d if applicable)

### 4.2 Wire CLAP note-ports extension
- [ ] Implement `get_extension` returning `clap_plugin_note_ports_t`
- [ ] Process note-on/note-off events in `process()` callback (iterate `in_events`)
- [ ] Map note-on: velocity + pitch -> conditioning signal
- [ ] Map note-off: release voice

### 4.3 Wire CLAP params extension
- [ ] Implement `get_extension` returning `clap_plugin_params_t`
- [ ] Implement `count()`, `get_info()`, `get_value()`, `value_to_text()`, `text_to_value()`
- [ ] Implement sample-accurate `frameOffset` handling in param flush
- [ ] Support linear/logarithmic/exponential parameter curves

### 4.4 Parameter smoothing
- [ ] Implement one-pole lowpass filter per parameter (coefficient from sample rate + smooth time)
- [ ] Apply smoothing in `process()` per-sample, respecting frameOffset
- [ ] No step changes on audio thread — all param transitions smoothed

### 4.5 Click-free bypass
- [ ] Replace hard mute (output = 0) with crossfade: short fade-out on bypass, fade-in on engage
- [ ] Crossfade duration: 64-256 samples (configurable)
- [ ] Bypass state machine: Processing -> FadingOut -> Bypassed -> FadingIn -> Processing

### 4.6 State extension
- [ ] Wire `clap_plugin_state_t` via `get_extension`
- [ ] Save: serialize params + weight file path + h_state snapshot (using double-buffer seqlock)
- [ ] Load: restore params + reload weight file + restore h_state
- [ ] Implement double-buffer seqlock for h_state snapshot (ADR-001 spec)
- [ ] Support JSON preset format with version field

### 4.7 Thread-check extension
- [ ] Implement `clap_plugin_thread_check_t` via `get_extension`
- [ ] `is_main_thread()` and `is_audio_thread()` implementations

### 4.8 Tail extension
- [ ] Implement `clap_plugin_tail_t` via `get_extension`
- [ ] `get_tail_ns()` returns estimated tail duration

### 4.9 Voice management
- [ ] Implement monophonic voice allocation: note-on steals current voice, note-off releases
- [ ] Track active note (pitch, velocity) for conditioning
- [ ] (Future: polyphonic with N-voice round-robin)

### 4.10 CLAP validator pass
- [ ] Run `clap-validator` against the `.clap` bundle
- [ ] Fix all required-check failures
- [ ] Fix all recommended-check failures

**Acceptance**: Plugin loads in clap-host/Carla/Bitwig, processes audio, responds to parameters, saves/loads state, passes `clap-validator` required checks.

---

## Phase 5: ARA 2.0 Integration

### 5.1 Obtain and integrate ARA SDK
- [ ] Obtain ARA SDK from Celemony
- [ ] Set `ARA_PATH` env var
- [ ] Add ARA SDK to CMakeLists.txt include paths
- [ ] Remove custom ARA stubs and type definitions

### 5.2 Replace ARA stubs with real SDK implementations
- [ ] Implement `ARA::PlugIn::PlugIn` factory
- [ ] Implement `ARA::PlugIn::DocumentController` with proper lifecycle
- [ ] Implement `ARA::PlugIn::PlaybackRenderer` using ARA SDK base classes
- [ ] Implement `ARA::PlugIn::EditorRenderer` using ARA SDK base classes

### 5.3 DocumentController
- [ ] Handle ARA document create/archiving/restoring
- [ ] Persist plugin state (params + weight path + h_state) in ARA document
- [ ] Integrate with CLAP state extension for consistency

### 5.4 PlaybackRenderer
- [ ] Implement offline rendering for bounce (read audio regions, process through SSM)
- [ ] Read ConditioningFrame from ARA analysis results
- [ ] Render SSM output with ARA-provided musical context
- [ ] Report rendering progress to host

### 5.5 EditorRenderer
- [ ] Implement analysis buffer for waveform visualization
- [ ] Provide audio data to GUI thread via lock-free ring buffer
- [ ] Handle playback position synchronization with host

### 5.6 Wire ARA -> conditioning pipeline
- [ ] ARA document changes (tempo, key, chords) -> ConditioningFrame update
- [ ] ConditioningFrame -> projection layer -> SSM parameter modulation
- [ ] Verify audible change when ARA context changes

**Acceptance**: Plugin loads in an ARA-compatible DAW (Reaper/Studio One), ARA analysis updates produce audible change in output.

---

## Phase 6: Validation & Hardening

### 6.1 Real-time safety verification
- [ ] Run Valgrind/ASan on `process()` path — verify zero allocations
- [ ] Run Helgrind/TSan — verify zero mutex locks on audio thread
- [ ] Verify no blocking syscalls in `process()` (no file I/O, no network, no sleep)
- [ ] Add `__attribute__((no_sanitize("thread")))` only where safe (seqlock reads)

### 6.2 Long-run stability
- [ ] Process 1 hour of audio, verify no NaN/inf/state explosion
- [ ] Verify periodic normalization keeps h_state bounded
- [ ] Test with extreme parameter values (max delta, min delta, zero input, DC input)

### 6.3 Performance optimization
- [ ] SIMD vectorize hot paths: matrix multiply in SSM step, projection layer
- [ ] Profile with `perf` or Instruments — identify top 3 hot spots
- [ ] Implement true block-optimized `process_block()` (vectorized conv1d, batched matmul)
- [ ] Verify <10us/sample at 44.1kHz and 48kHz

### 6.4 Cross-platform build verification
- [ ] Verify build on Linux x86_64 (gcc + clang)
- [ ] Verify build on macOS arm64 + x86_64 (Apple Silicon universal)
- [ ] Verify CMake works with both Release and Debug (ASan) configurations

**Acceptance**: Zero allocs/locks on audio thread; stable 1-hour run; benchmark at <10us/sample target; builds on Linux and macOS.

---

## Phase 7: Packaging & Documentation

### 7.1 README
- [ ] Build instructions (cmake, make, dependencies)
- [ ] Dependency acquisition guide (CLAP SDK, ARA SDK, GoogleTest)
- [ ] Plugin installation instructions (`.clap` bundle path per OS)

### 7.2 Bundle packaging
- [ ] Linux: `.clap` directory with `entry` symlink + shared library
- [ ] macOS: `.clap` bundle with `Info.plist` + universal binary
- [ ] CI script or Makefile target for packaging

### 7.3 License
- [ ] Choose license (GPLv3 for copyleft, MIT for permissive, or dual-license)
- [ ] Add LICENSE file
- [ ] Add license headers to source files

### 7.4 Reference documentation
- [ ] Parameter reference: name, id, range, default, curve type
- [ ] Weight export pipeline documentation for ML engineers
- [ ] Architecture overview (threading model, memory layout)

**Acceptance**: Fresh clone + SDK install -> build -> install -> plugin visible in host.

---

## Acceptance Summary

| Phase | Verifiable End State |
|-------|---------------------|
| 0 | `clap-validator` scans `.clap` bundle without errors |
| 1 | SSM math matches NumPy reference at 1e-5; benchmark <10us/sample |
| 2 | PyTorch export -> C++ load -> inference matches within quantization tolerance |
| 3 | Conditioning changes produce measurable + correct change in SSM output |
| 4 | Plugin loads in real CLAP host, processes audio, passes `clap-validator` |
| 5 | Plugin loads in ARA DAW, ARA context changes produce audible difference |
| 6 | Zero allocs/locks on audio thread; stable 1-hour run; cross-platform build |
| 7 | Fresh clone -> build -> install -> plugin visible in host |

---

## Completed Artifacts

```
src/core/arena_allocator.h/cpp    [done] — mmap-backed bump allocator
src/common/ring_buffer.h          [done] — SPSC lock-free ring buffer
src/common/seqlock.h              [done] — Seqlock with acquire/release
```

## File Checklist

```
src/core/
  ssm_block.h/cpp              [Phase 1 — needs rewrite]
  ssm_inference_core.h/cpp     [Phase 1 — new file]
  weight_loader.h/cpp          [Phase 2 — new file]
  arena_allocator.h/cpp        [done]

src/common/
  ring_buffer.h                [done]
  seqlock.h                    [done]
  conditioning_frame.h         [Phase 3 — new file, extract from clap_plugin.h]

src/clap/
  clap_plugin.h/cpp            [Phase 0+4 — needs SDK migration + completion]
  clap_audio_ports.cpp         [Phase 4 — needs SDK migration]
  clap_note_ports.cpp          [Phase 4 — needs SDK migration + wiring]
  clap_params.cpp              [Phase 4 — needs SDK migration + smoothing]
  clap_state.cpp               [Phase 4 — needs SDK migration + h_state snapshot]
  clap_entry.cpp               [Phase 0 — new file]

src/ara/
  ara_document_controller.h/cpp  [Phase 5 — needs ARA SDK migration]
  ara_playback_renderer.h/cpp    [Phase 5 — needs ARA SDK migration]
  ara_editor_renderer.h/cpp      [Phase 5 — needs ARA SDK migration]

scripts/
  export_mamba_weights.py      [Phase 2 — new file]
  validate_export.py           [Phase 2 — new file]

models/
  test_model.bin               [Phase 2 — generated by export script]

config.h                       [Phase 2 — auto-generated by export script]

WEIGHT_FORMAT_SPEC.md          [Phase 2 — new file]
```
