# EmT Plugin Suite — Corrected Project Plan

**Audit date:** 2026-08-29  
**Authority:** source/tests in this checkout; refresh status after each verified build.

Status: `[x]` implemented with source/test support, `[ ]` open, `[~]` partial, `[B]` externally blocked. Implemented does not mean packaged, host-validated, or released.

## Corrections made in this audit

- [x] Reject format v1 rather than parse its Mamba-1 layout as Mamba-3.
- [x] Validate reserved header fields, exact quantization metadata size, and exact weight-data size.
- [x] Route Float32 synth loading through stride-aware materialization; flat packed-to-`MAX_*` copies corrupted non-maximal layouts.
- [x] Adopt `num_layers` from a valid file instead of comparing it to the one-layer placeholder core.
- [x] Change compatible display names to EmT Synth/EmT FX while retaining stable IDs/formats.
- [x] Enable top-level CTest registration; previously all test executables built but `ctest` reported no tests.
- [x] Make the parity test locate its source fixture under CTest and reuse canonical stride-aware materialization.
- [x] Correct documentation that claimed convolution, monophony, arena allocation, zero-copy inference, obsolete ports/extensions, and complete FX/ARA behavior.

## Phase 0 — trustworthy baseline and correctness

- [x] MSVC 19.51 Release build succeeded on 2026-08-29; CTest passed 43/43 after discovery/parity fixes.
- [ ] Reduce or explicitly classify current MSVC warnings (alignment padding, CRT portability, allocator declaration mismatch); the build is not zero-warning.
- [x] `validate_export.py models/test_model.bin` passed and CTest parity passed on 2026-08-29; fixture regeneration remains part of future format changes.
- [ ] Add a plugin-level test proving small Float32 and multi-layer files load without stride corruption.
- [ ] Test truncated metadata, trailing bytes, invalid structural relationships, non-finite quant scales, and invalid zero points.
- [ ] Enforce `d_inner == nheads * headdim`, even `d_state`, `d_state/2 <= MAX_NUM_ROPE_ANGLES`, and supported fixed MIMO/BC layout.
- [ ] Make failed reload transactional. A mid-load failure can currently destroy the previous core/config while leaving stale load state.
- [ ] Replace integer loader errors with a documented enum/error-string API.
- [ ] Reconcile `README.md`, `GAP_ANALYSIS.md`, `agent.md`, old prompts, ADRs, and parameter docs.

Exit: clean Release build, tests/export/parity pass, and every remaining correctness gap is explicit.

## Phase 1 — CLAP host correctness

- [ ] Test both descriptors, creation by stable ID, and repeated lifecycle sequences.
- [ ] Validate frame/port/channel counts against activation limits.
- [ ] Test Float32, Float64, mixed representations, in-place buffers, missing inputs, and stereo independence.
- [ ] Test event order/timestamps, invalid sizes/IDs, queue overflow, and parameter clamping.
- [ ] Decide whether EmT Synth should expose a stereo audio input.
- [ ] Replace the non-portable validator target; missing validation must fail visibly.
- [ ] Run official `clap-validator` and at least two hosts per supported platform.
- [ ] Verify thread-check fallback behavior without advertising a nonexistent plugin extension.

Exit: automated lifecycle/port/event coverage and current official validation for both plugins.

## Phase 2 — complete EmT Synth

- [~] Eight shared-core voice slots exist; musical behavior and cost are unvalidated.
- [ ] Define allocation/stealing for note IDs, channels, keys, overlaps, choke/end, and expression.
- [ ] Review signed CLAP note IDs versus stored `uint64_t` and wildcard matching.
- [ ] Test state/preset restore, corruption, partial streams, voice identity, and model-path restore.
- [ ] Decide whether state embeds a model hash/identity rather than only a path.
- [ ] Reconcile fixed per-voice rates with Attack/Release parameters.
- [ ] Define `d_model`-to-stereo mapping; current synth uses only `outbuf[0]`.
- [ ] Add finite-output, level-bound, rapid-note, and long-session tests through CLAP process.

Exit: documented musical semantics, deterministic restore, bounded voice cost, host-tested audio.

## Phase 3 — make EmT FX real

- [ ] Implement main-thread weight loading with transactional publication.
- [ ] Add FX state save/load for parameters and model identity/path.
- [ ] Define a dedicated effect model/input-output contract if scalar mono inference is insufficient.
- [ ] Preserve stereo; current FX sums to mono and duplicates `processed[0]`.
- [ ] Define no-weight/invalid-weight behavior. Current wet output is silence; safe passthrough needs an explicit decision.
- [ ] Add FX-specific lifecycle, ports, params, state, bypass, Float64, and validator tests.

Exit: non-silent model-backed stereo processing, persisted state, safe failure, independent validation.

## Phase 4 — weight and training pipeline

- [ ] Centralize/generate C++ and Python layout arithmetic from one schema.
- [ ] Encode implicit `mimo_rank`, BC grouping, and RoPE layout in a future header, or freeze them for v2.
- [ ] Decide whether `MAX_*`-padded disk strides are intentional; they couple files to builds and inflate small models.
- [ ] Add checksums/integrity metadata and separate model-semantic versioning.
- [ ] Test export-to-C++ parity for Float32, both Int8 modes, and FP16.
- [ ] Correct FP16 subnormal conversion or freeze flush-to-zero behavior in the contract.
- [ ] Add golden files/migration tooling for every supported version.

Exit: reproducible export, all formats parity-tested, integrity checked, compatibility policy documented.

## Phase 5 — real-time safety and performance

- [ ] Intercept allocations on actual synth/FX process call graphs.
- [ ] Run ASan/UBSan and concurrency tooling where supported.
- [ ] Prove bounded behavior and expose diagnostics when the 256-event queue fills.
- [ ] Review seqlock spinning under a stalled writer; lock-free is not wait-free.
- [ ] Measure p50/p95/p99/max callbacks, xruns, CPU, and memory at 44.1/48/96/192 kHz across model/block sizes.
- [ ] Profile before changing fixed-stride B/C/MIMO layout or adding SIMD.
- [ ] Optimize `process_block()` only with parity and regression benchmarks.
- [ ] Measure shared-core per-voice state-copy cost.
- [ ] Measure end-to-end latency separately from zero-sample algorithmic latency.

Exit: no tested real-time violations and hardware-qualified performance budgets.

## Phase 6 — GUI and user experience

- [~] Linux/X11 synth GUI exists; Windows/macOS and FX GUI do not.
- [ ] Choose/document a cross-platform GUI and ownership/thread model.
- [ ] Add accessible labels, model status/errors, metering, and conditioning state.
- [ ] Implement parameter gestures and correct host rescan/flush interactions.
- [ ] Test scale/resize/lifecycle and timer-registration failures.
- [ ] Rename preset/bundle identifiers only through an approved migration.

Exit: usable on supported platforms without affecting headless/audio operation.

## Phase 7 — ARA integration

- [B] Obtain/license the Celemony ARA SDK.
- [ ] Treat current classes as internal stubs, not compliance scaffolding.
- [ ] Design real CLAP/ARA factory/bindings and lifecycle from licensed SDK material.
- [ ] Implement controller, renderers, persistence, offline render, and conditioning bridge.
- [ ] Remove global singleton context where instance/document isolation requires it.
- [ ] Add SDK conformance and compatible-host tests.

Exit: SDK-linked ARA integration validated in hosts; blocked until then.

## Phase 8 — CI, packaging, and release

- [ ] Add Windows/macOS/Linux Release CI, tests, validation, and appropriate warnings gates.
- [ ] Produce correct CLAP bundles; current macOS rules are not verified bundle packaging.
- [ ] Add checksums, provenance, SBOM/license checks, signing/notarization, and installers.
- [ ] Add hardware-aware benchmark reporting.
- [ ] Complete LICENSE and third-party notices.
- [ ] Create an EmT compatibility migration matrix and release notes.
- [ ] Do not publish, sign, deploy, or upload without explicit approval.

Exit: green platform CI, validator/host evidence, recoverable signed packages, approved release.
