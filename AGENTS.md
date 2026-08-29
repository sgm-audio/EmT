# EmT Mamba-3 CLAP Plugin Suite

EmT (pronounced “empty”) is a C++20 neural-audio plugin suite built around a shared Mamba-3 state-space inference core. Read this file before making changes, then read the affected implementation and tests. Repository claims are not proof: build and rerun applicable checks before reporting them as current.

## Build and test

```powershell
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --parallel 4
ctest --test-dir build -C Release --output-on-failure
```

- CMake 3.20+, C++20, GoogleTest 1.14 through `FetchContent`.
- The shared library/bundle remains `MambaSynth.clap`; this compatibility-sensitive name has not migrated.
- The factory exports stable IDs `com.mamba-ssm.synth` and `com.mamba-ssm.fx`; display names are EmT Synth and EmT FX.
- MSVC tests use `/STACK:16777216`. The plugin target currently does not, so do not repeat the older claim that it does.
- The optional `clap-validator` target uses `bash`, is not portable to native Windows, and exits successfully when the validator is missing. A successful target invocation is not proof of validation.

## Product and compatibility boundary

Public brand: **EmT**. Existing CLAP IDs, bundle name, preset extension/directory, state magic/version, and weight magic/version remain compatibility identifiers. Do not change them without an approved migration plan covering saved DAW sessions, automation, presets, and state restore.

The checkout currently contains:

1. **EmT Synth** — exported CLAP instrument with eight voice slots, parameters, state, presets, and a Linux/X11 GUI.
2. **EmT FX** — exported CLAP effect with gain/mix/bypass, but no working weight-load path, persisted state, GUI, or independent host tests. Its wet path is currently silent because the inference core has no weights.
3. **ARA companion code** — internal custom stubs only. It is not an exported ARA plug-in and is not ARA 2 compliant.

## Core architecture

Everything remains in `namespace mamba`; the namespace is an internal compatibility detail.

- `SSMBlock`: one Mamba-3 layer with complex multi-head state, trapezoidal discretization, shared RoPE angles, RMS normalization, FiLM conditioning, and fixed-stride MIMO storage.
- `SSMInferenceCore`: placement-news up to `MAX_LAYERS` blocks into embedded byte storage.
- `SSMWeights`: fixed-capacity writable Float32 materialization used by inference.
- `WeightLoader`: maps a version-2 file read-only, validates it, then materializes each layer with stride-aware copying/dequantization.
- `MambaSynth`: synth instance state, allocated on the main thread with `new`.
- `FxState`: separate effect wrapper in `clap_fx_plugin.cpp`.

`ArenaAllocator` exists but is not used for current CLAP instance creation. Do not describe the current plugin as arena-allocated or inference as zero-copy.

## Threading and real-time rules

| Context | Current work | Rules |
|---|---|---|
| Audio | CLAP process callbacks, per-sample events, envelopes, SSM inference | No heap allocation, locks, waits, filesystem/environment access, logging I/O, exceptions, or unbounded work |
| Main/host | lifecycle, params flush when stopped, state/presets, weight loading | May allocate and perform I/O; must not race audio |
| GUI | Linux/X11 event/timer callbacks | Non-real-time; synth only |
| ARA stub | global musical-context helpers | Not a real host-created ARA thread or SDK integration |

Communication primitives:

- `SPSCRingBuffer<ParamAtom, 256>` for queued parameter events.
- `Seqlocked<ConditioningFrame>` and `Seqlocked<ARAConditioningFrame>` for conditioning.
- `DoubleBufferSeqLock<StateMirror>` for audio-to-main state snapshots.

The seqlock reader spins until stable and is lock-free but not wait-free. `DoubleBufferSeqLock::read()` can return `false`; callers must retry or fail safely. Preserve acquire/release ordering.

Call `SSMBlock::apply_ftz_daz()` in every process callback. x86 sets FTZ/DAZ; AArch64 sets FPCR.FZ.

## Compile-time and format limits

```text
MAX_D_STATE=32      MAX_D_MODEL=256      MAX_D_INNER=512
MAX_NHEADS=8        MAX_HEADDIM=64       MAX_MIMO_RANK=4
MAX_NUM_BC_HEADS=1  MAX_NUM_ROPE_ANGLES=16
MAX_LAYERS=6        COND_DIM=128         MAX_BLOCK_SIZE=512
MAX_CHUNK_SIZE=128  MAX_PARAM_ATOMS=256  SSM_STATE_NORM_INTERVAL=4096
```

The header stores no `mimo_rank`, `num_bc_heads`, or RoPE count. Version 2 uses fixed file strides `MAX_MIMO_RANK` and `MAX_NUM_ROPE_ANGLES`; the exporter zero-pads unused slots. Changing those constants changes the v2 physical layout and requires a format-version decision.

Weight rules:

- Magic `0x4D424D41`, version **2 only**, 64-byte little-endian header.
- Mamba-3 has no convolution field/weights and reports zero samples of algorithmic latency.
- Thirteen arrays per layer; see `WEIGHT_FORMAT_SPEC.md`.
- Quantization codes 0–3 are Float32, Int8 tensor, Int8 layer, and FP16.
- Int8 metadata records are 16 bytes; `flags` and `reserved` must be zero.
- Loader path order: explicit path, `MAMBA_WEIGHTS`, `models/test_model.bin`.
- Loading/materialization is main-thread work. The mapped file is read-only, but inference uses copied/dequantized `SSMWeights`.

## CLAP implementation status

- Both plugins expose one main stereo input and output with Float32/Float64 support.
- EmT Synth also exposes one note input, five parameters, state, tail, zero latency, GUI, and timer support.
- EmT FX exposes three parameters, tail, and zero latency; state returns `nullptr`, and GUI/weight loading are absent.
- The plugin does not expose `CLAP_EXT_THREAD_CHECK`; `clap_thread_check.h` consumes the host extension and falls back to recorded thread IDs.
- Presets use internal `.mambapreset` files, not the CLAP preset-discovery/load extension.
- Synth events are handled inside the sample loop.
- Eight voices share one core by restoring/saving a full `MambaState` per active voice per sample. This is implemented but not performance/host validated.

## Tests and evidence gaps

CTest discovers GoogleTest cases from six executables. Do not hard-code a passing total in durable documentation. Coverage includes core math, loader formats/errors, conditioning/seqlock behavior, stability, source-level real-time checks, and one C++/Python parity fixture.

Missing evidence:

- no automated CLAP factory/lifecycle/extension tests;
- no EmT FX behavior/state tests;
- no allocation interception on the actual plugin process path;
- “long-run” stability is a bounded synthetic loop, not ten wall-clock minutes;
- no current cross-platform CI or package evidence;
- no p50/p95/p99/max callback timing or xruns;
- narrow parity fixtures that must be regenerated when math/format changes.

## Required workflow

1. Inspect `git status` and preserve unrelated changes.
2. Read `TODO.md`, `ARCHITECTURE.md`, `WEIGHT_FORMAT_SPEC.md`, and affected code/tests.
3. State contract and compatibility impact.
4. Add focused regression tests.
5. Build narrow targets, then CTest and relevant Python validation.
6. Benchmark like-for-like when performance can change.
7. Report separately: edited, compiled, tested, benchmarked, validator-checked, packaged, and released.

## References

- `Prompts/EMT_SYSTEM_PROMPT.md` — execution framework.
- `TODO.md` — phased findings/work.
- `ARCHITECTURE.md` — current implementation/trust boundaries.
- `WEIGHT_FORMAT_SPEC.md` — authoritative v2 layout.
- `PARAMETER_REFERENCE.md`, `README.md`, `GAP_ANALYSIS.md`, `agent.md`, older prompts, and ADRs may still be stale; verify them before use.
