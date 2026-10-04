# CHANGELOG

All notable changes to the EmT CLAP/ARA Plugin Suite (EmT).

## [Unreleased] - 2026-10-04

Verification-driven fixes. Context: [`CLAIMS_VERIFICATION.md`](CLAIMS_VERIFICATION.md)
and the new [`ARCHITECTURE_FIDELITY.md`](ARCHITECTURE_FIDELITY.md).

### Fixed
- **The plugin rejected every multi-layer weight file.** `load_weights()` checked
  the header with `validate_dimensions(hdr…, core_cfg.num_layers)`, and that
  function compares for *equality* against the shape it is given. At that point
  `core_cfg.num_layers` is still the constructor's `1` placeholder, so **any model
  the training configs produce (3 layers by default) failed to load** — the
  exported `.bin` was silently refused. Both plugins now call
  `WeightLoader::fits_in_static_weights_with(hdr)`, which performs the intended
  `<=` comparison against the compiled `MAX_*` limits. `WeightLoader::fits_in_static_weights()`
  had the same equality bug (it returned `false` for files that plainly fit) and
  now forwards to that predicate. Verified end-to-end: a 3-layer model trained by
  `training/train.py` loads through the FX plugin's real `fx_load_weights()` path
  and processes an audio block.
- **`FxRejectsWeightsExceedingCompiledLimits`**: a header claiming
  `d_state = MAX_D_STATE + 1` is refused, so the new acceptance path cannot be
  widened into "anything loads".
- **Segfault in the FX plugin constructor (`FxState.core_storage`)**:
  the placement-new buffer followed a pointer member without its own alignment
  specifier, so `SSMInferenceCore` (64-byte aligned) was constructed 8 bytes off a
  64-byte boundary. `RealtimeSafety.FxStateSaveLoadRoundtrip` crashed the process;
  it now passes. Replaced the raw byte array with
  `mamba::AlignedStorage<T, N>` (`src/common/aligned_storage.h`) in both the synth
  and FX state structs so the requirement lives on the member and cannot drift.
- **Silent no-op training**: `training/train.py` skipped undecodable files with a
  warning and then exported untrained weights if nothing loaded. It now raises
  ("refusing to export untrained weights") when zero optimisation steps ran, and
  the empty-training guard is exercised in the CI workflow (see the Actions
  known issue below).

### Added
- **`ARCHITECTURE_FIDELITY.md`** - implemented-vs-paper ledger, deviation register
  (D1-D6) with `file:line` evidence and measured effects, path to compliance
  (F1-F5), and a claim policy for docs/demos/posts.
- **Task-matched objectives (`training/losses.py`)**: pre-emphasis L1, SI-SDR
  (with silent-target handling), multi-resolution STFT, and a `hybrid` composite;
  selectable via `--loss` / the `loss:` config key. Includes a runnable self-test
  (`python3 training/losses.py`) covering SI-SDR properties (including the
  noisy-reference caveat) and gradient flow for every variant.
- **Paired dry/wet training**: a `dry/` + `wet/` dataset layout (or `--paired`)
  trains the effect mapping an FX plugin actually has to perform, instead of
  next-sample prediction.
- **Dependency-light audio decoding**: `soundfile` -> `torchaudio` -> stdlib
  `wave` fallback, because recent torchaudio releases require TorchCodec.
- **Gating benchmark** (`tests/benchmark_ssm_block.cpp`, rewritten): reports
  microseconds per sample *and* percent of one core, takes `--config`, `--layers`,
  `--sample-rate(s)`, `--blocks`, `--budget-pct`, `--repeats`, `--json`, and
  **exits non-zero when a configuration misses the budget**. Adds `train4070` and
  `trainl4` shapes mirroring the training configs and a `max` stress shape.
- **CI**: parity vectors are generated before `ctest` (the C++/Python parity test
  previously skipped silently), the benchmark gate is invoked on every push, the
  training-shape benchmark is published as an artifact, and the objective
  self-test + trainer smoke test were added.

### Known issue (not fixable from the workflow file)
- **GitHub Actions does not start for this repository.** Every run since
  2026-10-02 is a `startup_failure` with no jobs, attributed to GitHub's
  `BuildFailed` placeholder workflow — including a push to `main` whose
  `ci.yml` is byte-identical to the version that ran on 2026-09-14, and the
  GitHub-managed `Dependency Graph` workflow. Both files validate against the
  official workflow schema (`check-jsonschema --builtin-schema
  github-workflows`). This is an account/repository-level Actions problem
  (billing/spending limit or Actions disabled) that needs an admin. Until it is
  resolved the gates are **wired, not enforced**; `ARCHITECTURE_FIDELITY.md` §4b
  records the evidence and the local equivalent of every CI step.

### Changed
- **Test suite: 67 tests, 0 failures** (was 62 + a segfaulting FX state test + a
  parity test that skipped without vectors). New coverage:
  `RealtimeSafety.FxAcceptsMultiLayerWeightsWithinLimits`,
  `RealtimeSafety.FxRejectsWeightsExceedingCompiledLimits`,
  `WeightLoaderPath.MultiLayerModelIsLoadable`.
- **`CppParity` can no longer skip silently**: setting `MAMBA_REQUIRE_PARITY=1`
  turns a missing vector file into a failure, and CI generates the vectors and a
  2-layer test export *before* `ctest` and points `MAMBA_TEST_WEIGHTS` at it.
- **`--steps` is now a global optimisation-step cap** (it defaulted to "only with
  `--smoke-test`", so `train.py --steps 20` silently ran all 20 epochs).
- **Correction to the 0.3.0 entry below.** That entry claims "AVX2 and SSE2
  vectorized operations for SSM recurrence update". SIMD is used for the FiLM
  projections, the `in_proj` dot products and the output projection
  (`src/core/ssm_block.cpp:112-131, 288-295`), but the **recurrence loop over
  `d_state` is still scalar** (`:245-273`) and `process_block()` remains a
  per-sample loop. Measured cost reflects that: see `ARCHITECTURE_FIDELITY.md` §3.
- **Documentation corrected to match the code.** README, `MIGRATION_TO_MAMBA3.md`,
  `TRAINING_PLAN.md` and the LoRA companion README no longer claim the
  exponential-trapezoidal recurrence, complex-valued state, active MIMO, "zero
  latency because conv1d was removed", or runtime LoRA. Measured CPU numbers
  replace the toy-shape 1.79 us/sample headline, and `TRAINING_PLAN.md` now warns
  that the RunPod L4 config is not real-time (~90 % of one core per layer).

### Deviation closures (second pass, 2026-10-04)

Closes four of the six entries in `ARCHITECTURE_FIDELITY.md` §2 (D3, D4, D5,
D6); D1/D2 remain open with prompts in `Prompts/DEVIATION_RESOLUTION/`.
`python3 scripts/verify_deviations.py` prints one verified status line per
deviation and fails if a closed one regresses.

- **D3 (MIMO) — refused instead of silently downgraded.** The 64-byte weight
  header now carries a real `mimo_rank` field (`'<12IQII'`; `0` = legacy file,
  treated as rank 1). `WeightLoader::load()` returns **-16** for `mimo_rank > 1`,
  `fits_in_static_weights_with()` rejects such headers, the exporter raises
  `SystemExit` instead of writing the file, and `training/train.py` refuses a
  rank-R config before training. Guards:
  `WeightLoaderTest.RejectsUnimplementedMimoRank`,
  `WeightLoaderTest.LegacyHeaderWithoutMimoFieldIsRankOne`.
- **D4 (input-derived `A`) — documented as a project variant.** D4 comments at
  the clamp site (C++, both Python references, the header) plus a prohibition in
  `ARCHITECTURE_FIDELITY.md` §1.2; no behaviour change.
- **D5 (periodic state renormalization) — removed.** Deleted from
  `src/core/ssm_block.{h,cpp}`, `src/config.h`, both Python references and the
  exporter's `config.h` generator. The old guard stepped the output gain by
  14.8 % every 4096 samples; stability now follows from the A-stable
  trapezoidal update. Guards:
  `Stability.NoRenormalizationGainStepAtFormerIntervalBoundary` (constructed
  `|h| = 4` case; the verifier also reproduces the legacy artifact as a
  counterfactual: 0.0000 % now vs 24.48 % with the old guard) and
  `Stability.HighStateLongRunStaysBounded`.
- **D6 (`dt_film_proj_w`) — wired into Δ.** Per-head `(γ, β)` rows `2h`/`2h+1`
  modulate the step size with the same convention as the input FiLM, applied
  only when conditioning is active (unconditioned inference is bit-identical).
  Implemented in C++, `reference_mamba3.py`, `reference_forward.py` and the
  PyTorch model used by the trainer; the parity generator now emits non-zero
  Δ-FiLM weights, so the C++ ↔ NumPy parity test (1.19e-07) covers the path.
  Behavioural guard: `ConditioningModulation.DtFilmModulatesDelta`.
- Verification: `ctest` **72/72** (was 67); real-time gate
  `--config small --layers 1,2 --budget-pct 40` still **8/8 PASS**
  (worst 4.80 us/sample = 23.0 % of one core on the CI-equivalent build).

## [0.3.0] - 2026-09-13

### Added
- **SIMD Block Processing (`SSMBlock::process_block()` & `simd_helpers.h`)**:
  - AVX2 and SSE2 vectorized operations for SSM recurrence update and dot-product output projections.
  - Portable intrinsics abstractions (`dot_product`, `fmadd_scalar`, `fmadd_elementwise`, `mul_elementwise`, `scale`) supporting x86-64 AVX2/SSE2 and ARM NEON with compiler auto-vectorization fallback.
  - Sub-microsecond execution achieving 1.985 µs/sample throughput (over 503,000 samples/sec), exceeding the 10 µs real-time threshold by 5×.
  - Verified numerical parity against scalar reference within $\le 10^{-5}$ tolerance.
- **Audio Thread Zero-Copy Voice State Isolation**:
  - Refactored `SSMInferenceCore::process_sample` and `SSMBlock::process_sample` to accept direct mutable `MambaState&` and `SSMState&` references.
  - Eliminated 8,192 multi-kilobyte `memcpy` operations per audio block in polyphonic synthesis by bypassing `set_state()`/`get_state()` ping-ponging.
  - Preserved backward-compatible non-state overloads for single-instance processing and unit tests.
  - Added `RealtimeSafety.ZeroCopyVoiceStateIsolation` and `SSMInferenceCore.ProcessSampleExplicitStateParity` unit tests verifying deterministic state separation and zero allocations.
- **CLAP Event Slicing & Sub-Sample Accurate Scheduling**:
  - Replaced linear per-sample event polling with sub-block event slicing in `clap_plugin.cpp`.
  - Incoming CLAP note and parameter events are sorted by timestamp at block start; processing is partitioned into contiguous slices between event timestamps ($t_0 \to t_1 \to \dots \to t_N$).
  - Sample-accurate parameter smoothing and envelope stage transitions applied at exact sub-sample boundaries.
- **Windows Native GUI Backend (`CLAP_WINDOW_API_WIN32`)**:
  - Complete Win32 window procedure (`WndProc`) and HWND embedding implementation in `clap_gui.cpp`.
  - Custom double-buffered GDI rendering of parameter knobs (Gain, Mix, Bypass), value labels, and interactive preset buttons.
  - Full mouse drag interaction with parameter normalization and clamped updates dispatched through host parameter flushes.
  - Window resizing (`WM_SIZE`), paint validation (`WM_PAINT`, `WM_ERASEBKGND`), timer-driven invalidation via `CLAP_EXT_TIMER_SUPPORT`.
  - Dual-platform support: automatically negotiates `CLAP_WINDOW_API_WIN32` on Windows and `CLAP_WINDOW_API_X11` on Linux.
- **EmT FX State Serialization (`CLAP_EXT_STATE`)**:
  - Implemented `clap_plugin_state_t` extension for `com.emt.fx` plugin (`clap_fx_plugin.cpp`).
  - Serializes and deserializes FX parameter states (Gain, Mix, Bypass) with binary magic `0x4D424658` (`MBFX`) and versioning schema.
  - Fully compatible with host DAW project saving, reloading, and undo/redo snapshots.
- **Stack Pressure Elimination & MSVC Safety**:
  - Converted stack-allocated `SSMWeights` (~4 MB) and `StateMirror` structs in tests, benchmarks, preset saving, and state serialization to heap-allocated `std::unique_ptr`.
  - Configured 16 MB stack flag (`/STACK:16777216`) in CMake for Windows test targets and CLAP binaries, preventing audio-thread stack overflow in DAWs with 1 MB thread stacks.
  - Resolved MSVC macro collisions (`NOMINMAX`) and CRT warnings cleanly.
- **CTest Automation & Complete Suite Discovery**:
  - Integrated `enable_testing()` into both project and monorepo root `CMakeLists.txt`.
  - Registered all test binaries with CTest: 64 test cases across 7 suites (`test_ssm_block`, `test_weight_loader`, `test_conditioning`, `test_stability`, `test_realtime_safety`, `test_cpp_parity`, `test_ara`).
  - Achieved 100% pass rate (64/64 tests passing in 2.67 seconds).
- **ARA 2.0 Engine Bridge (`ara_document_controller.cpp`, `ara_playback_renderer.cpp`)**:
  - Implemented complete ARA 2.0 DocumentController lifecycle state machine and analysis bounds negotiation with timeline region lookups.
  - Implemented offline chunked rendering with `SSMInferenceCore` driven by extracted musical context (tempo BPM, key signature, chord one-hot encoding).
  - Validated concurrent exchange over wait-free `DoubleBufferSeqLock<ARAConditioningFrame>` between ARA writer and real-time audio reader.
- **JSON Preset Engine & Factory Bank (`clap_presets.cpp`)**:
  - Added bank of 6 factory presets: "01 - Mamba Default", "02 - Dark Lead", "03 - Lush Pad", "04 - Resonant Sub", "05 - Fast Perc", "06 - Neural Delay".
  - Implemented JSON serialization and deserialization helpers complementing binary state snapshots.
- **Architectural Decomposition & Code Deduplication**:
  - Decomposed monolithic 250-line `SSMBlock::process_sample` into 9 modular, inline private step methods (`step_film_conditioning`, `step_in_proj`, `step_preprocess_decay_and_dt`, `step_rmsnorm_bc`, `step_rope_bc`, `step_recurrence`, `step_silu_gate`, `step_out_proj_residual`, `step_periodic_normalization`).
  - Extracted `src/clap/clap_common_utils.h` providing unified one-pole smoothing filter math, 4-state bypass crossfade state machine (`advance_bypass_machine`), buffer extract/mix utilities, and endian-safe CLAP stream I/O.
  - Eliminated hundreds of lines of duplicate code between `clap_plugin.cpp`, `clap_fx_plugin.cpp`, and `clap_state.cpp`.
  - Removed obsolete, dead `arena_allocator.h` and `arena_allocator.cpp`.
  - Synchronized refactored core into shared monorepo library `libs/cpp-inference-engine`.

### Fixed
- **State Mirror Initialization & MSVC Compiler OOM**: Replaced aggregate 6.5 MB stack value-initialization with `std::make_unique<StateMirror>()` and `std::memset`, resolving MSBuild C1060 compiler out-of-heap-memory errors.
- **SARIF Logger Crash**: Suppressed alignment padding warning `/wd4324` to prevent MSBuild 18 .NET SARIF output memory exhaustion.
- **Preset Path Resolution**: Added cross-platform fallback for Windows `%USERPROFILE%` and `%APPDATA%` when `HOME` environment variable is not defined.
- **MSVC Compiler Parity**: Added `/wd4324` (alignment padding) and `/wd4996` (deprecated POSIX/CRT functions) suppression flags.

---

## [0.2.0] - 2026-06-14

### Added
- Documentation suite: `WEIGHT_FORMAT_SPEC.md`, `PARAMETER_REFERENCE.md`, `ARCHITECTURE.md`, full `README.md` rewrite
- `LICENSE` file (Apache 2.0)
- CI workflow (`.github/workflows/ci.yml`): Ubuntu 24.04, build, test, benchmark, clap-validator
- `clap-validator` CMake custom target
- Python export script: `--cond-dim` flag now defaults to 16 (parameterized, not hardcoded)

### Fixed
- **Critical:** `load_weights()` now copies conditioning projection weights (`cond_proj_dt`, `cond_proj_B`, `cond_dt_bias`, `cond_B_bias`) from weight file into `SSMWeights` — previously skipped, causing conditioning to always use zero weights regardless of loaded file
- **Critical:** `SSMBlock::process_block()` now writes `d_model` floats per sample instead of summing to scalar — fixes incorrect output for `d_model > 1`
- **Critical:** `SSMInferenceCore::process_block()` ping-pong buffer aliasing fixed for 3+ layers — swap logic no longer corrupts intermediate mono/d_model buffers
- **Critical:** `export_mamba_weights.py` `_random_init()` scoping bug fixed — `b.cond_proj_dt` (etc.) now targets all layers, not just the last one
- **Critical:** `validate_export.py` now reads `cond_dt_bias` and `cond_B_bias` — previously omitted, causing validation to miss these arrays
- **Conditioning:** `process_block()` now applies `softplus()` after adding conditioning offset, matching `process_sample()` behavior
- `benchmark_ssm_block.cpp` and `test_ssm_block.cpp`: fixed stack buffer overflow in `process_block()` output allocation (was `size` floats, needs `size * d_model`)
- `export_mamba_weights.py`: `cond_dim` is now passed as a parameter to `floats_per_layer()` instead of being hardcoded to 16
- B double-counting in SSM recurrence fixed: `B_bar` already incorporates the B projection via ZOH discretization
- Weight file layout fixed: `b_in` vs `W_ssm` ordering now consistent between test data and `get_layer_pointers()`
- Blocking I/O in audio thread removed: `WeightLoader::load()` + per-layer `memcpy` moved out of `plugin_process()` into main-thread `load_weights()`
- Heap allocation in `plugin_init()` replaced with placement-new into pre-allocated `core_storage[]`

---

## [0.1.0] - 2026-05-30

### Added
- Python export pipeline: `export_mamba_weights.py` exports PyTorch MambaBlock → flat-binary weight loader format; `validate_export.py` validates round-trip correctness
- Proper conditioning projection layer (`cond_proj_dt`, `cond_proj_B` weights) replacing crude 0.01/0.001 modulation hack
- Benchmark infrastructure (`benchmark_ssm_block.cpp`) with per-sample and block processing benchmarks for single and multi-layer configurations
- Benchmark target registered in `tests/CMakeLists.txt`

### Fixed
- **Critical**: B double-counting in SSM recurrence (`ssm_block.cpp`)
- **Critical**: Weight file layout mismatch between test data and `get_layer_pointers()`
- Blocking I/O in audio thread
- Heap allocation in `plugin_init()`
- Python forward pass dimension error in `MambaBlock.forward()`

### Changed
- `SSMWeights` struct: added `cond_proj_dt[MAX_D_STATE][COND_DIM]` and `cond_proj_B[MAX_D_STATE][COND_DIM]`
- Weight file format v1: extended per-layer layout with conditioning projection weights
- `WeightLoader::floats_per_layer()` and `get_layer_pointers()`: account for conditioning projection weights
- `SSMInferenceCore`: added default constructor and `init()` method for two-phase placement-new construction
- `config.h`: auto-generated by export pipeline (regenerated at export time)
