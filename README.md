# EmT CLAP/ARA Plugin Suite

[![C++20](https://img.shields.io/badge/C%2B%2B-20-blue.svg)](https://en.cppreference.com/w/cpp/20)
[![CLAP](https://img.shields.io/badge/CLAP-1.2+-purple.svg)](https://cleveraudio.org/)
[![ARA 2](https://img.shields.io/badge/ARA-2.0-orange.svg)](https://www.celemony.com/en/ara)
[![CTest](https://img.shields.io/badge/tests-67%20passing-brightgreen.svg)]()
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

Real-time neural audio synthesis and effect processing plugins powered by a **Mamba-3-inspired causal state-space model (SSM)**. Delivered as low-overhead, real-time safe **CLAP** plugins with optional **ARA 2.0** integration for DAW-hosted generative workflows.

> **Read this before quoting architecture or performance claims:**
> [`ARCHITECTURE_FIDELITY.md`](ARCHITECTURE_FIDELITY.md) is the authoritative
> record of what is implemented versus what the Mamba-3 paper (arXiv:2603.15569)
> specifies, with a claim policy and a deviation register. The short version: the
> core is a Mamba-family SSM with input-selective Δ/B/C, RMSNorm, B/C biases and
> data-dependent rotation of B/C — but the recurrence is a two-term bilinear
> update with a **real** state, not the paper's three-term exponential-trapezoidal
> recurrence, and the MIMO path is carried in the weight format without being
> used at inference. [`CLAIMS_VERIFICATION.md`](CLAIMS_VERIFICATION.md) is the
> independent third-party-style review that produced this position.

> **Project Identity:** This project is **EmT** (migrated from `mamba-ssm-plugin-suite-70317161`), a distinct Mamba-SSM CLAP/ARA plugin and training suite. It is separate from CLARA. No "0" in the name.

---

## Table of Contents

- [Plugin Suite Overview](#plugin-suite-overview)
- [SSM Core — implemented vs planned](#ssm-core-what-is-actually-implemented)
- [Architecture Fidelity & Claim Policy](ARCHITECTURE_FIDELITY.md)
- [Architecture](#architecture)
  - [Three-Thread Concurrency Model](#three-thread-concurrency-model)
  - [Zero-Copy Polyphonic Voice Management](#zero-copy-polyphonic-voice-management)
  - [Sub-Sample Event Scheduling](#sub-sample-event-scheduling)
  - [State Persistence & Preset Engine](#state-persistence--preset-engine)
  - [Dual-Platform GUI Backend](#dual-platform-gui-backend)
  - [Celemony ARA 2.0 Integration & Fallback](#celemony-ara-20-integration--fallback)
- [Performance & Benchmarks](#performance--benchmarks)
- [Training & Objectives](#training--objectives)
- [Build Instructions](#build-instructions)
  - [Prerequisites](#prerequisites)
  - [Windows Build (MSVC)](#windows-build-msvc)
  - [Linux Build (GCC / Clang)](#linux-build-gcc--clang)
  - [Running CTest & Benchmarks](#running-ctest--benchmarks)
  - [Plugin Installation Paths](#plugin-installation-paths)
- [Weight Export & Validation Pipeline](#weight-export--validation-pipeline)
- [Directory Structure](#directory-structure)
- [License](#license)

---

## Plugin Suite Overview

The suite exports two distinct CLAP plugins from a unified factory (`clap_entry.cpp`), sharing the high-performance `SSMInferenceCore`:

| Plugin | CLAP ID | Type | Description |
|--------|---------|------|-------------|
| **EmT Synth** | `com.emt.synth` | Instrument | Polyphonic neural synthesizer driven by MIDI note input, velocity, and ARA musical conditioning |
| **EmT FX** | `com.emt.fx` | Audio Effect | Low-latency neural effect processor with FiLM conditioning, parameter smoothing, and bypass crossfading |

Both plugins are delivered in a single unified bundle: `EmTSynth.clap`.

---

## SSM Core (what is actually implemented)

The core is a Mamba-family selective SSM adapted for real-time audio. The list
below is deliberately split into **implemented** and **not yet implemented**;
see [`ARCHITECTURE_FIDELITY.md`](ARCHITECTURE_FIDELITY.md) for `file:line`
evidence and the plan to close the gaps.

**Implemented:**

1. **Multi-head state recurrence** with a real-valued per-head state:
   $$h_t = \bar{A}\, h_{t-1} + \bar{B}\,(B_t \odot x_t), \qquad y_t = C_t^{\top} h_t + D\,x_t$$
   Each layer decomposes the hidden state across $N_H$ heads
   (`src/core/ssm_block.cpp:245-273`).
2. **Bilinear (Tustin) discretization**:
   $$\bar{A} = \frac{1 + \frac{1}{2} A \Delta}{1 - \frac{1}{2} A \Delta}, \quad \bar{B} = \frac{\Delta}{1 - \frac{1}{2} A \Delta}$$
   The bilinear map is A-stable, so the recurrence remains stable at 44.1–192 kHz.
   *(This is a classical trapezoidal rule applied to a real pole — not the paper's
   exponential-trapezoidal three-term form; see deviations D1/D2.)*
3. **Data-dependent rotation of $B$ and $C$ plus RMSNorm**: a cumulative angular
   state rotates each $(B, C)$ pair per head and dimension, with RMSNorm and
   learned bias terms on both projections (`src/core/ssm_block.cpp:163-243`).
4. **Feature-wise Linear Modulation (FiLM)** — an EmT addition, not a paper
   feature: conditions the layer input on MIDI pitch, velocity and DAW/ARA context.
5. **Zero algorithmic look-ahead**: the recurrence is causal and consumes one
   sample per step with no future reads, so the plugin reports 0 samples of
   latency. *(Mamba-1/2's causal conv is also streamable with no look-ahead —
   dropping it is a simplification and a compute saving, not a latency win.)*

**Not yet implemented** (do not claim these until F1–F3 in
`ARCHITECTURE_FIDELITY.md` are done):

- Exponential-trapezoidal (three-term, input-history) recurrence
- Complex-valued state update (the `ssm_imag` array is allocated but unused at
  runtime; only the B/C rotation is live)
- MIMO rank-R coupling at inference (weights are carried; rank 0 is used)

---

## Architecture

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                       HOST DAW                                         │
│  ┌────────────────────────┐   ┌────────────────────────┐   ┌────────────────────────┐  │
│  │      Audio Thread      │   │      Main Thread       │   │       ARA Thread       │  │
│  │ process() @ 48kHz RT   │   │ host callbacks / GUI   │   │ background analysis    │  │
│  └───────────┬────────────┘   └───────────┬────────────┘   └───────────┬────────────┘  │
└──────────────┼────────────────────────────┼────────────────────────────┼───────────────┘
               │                            │                            │
               ▼                            ▼                            ▼
     ┌──────────────────┐         ┌───────────────────┐        ┌───────────────────┐
     │   SPSC RingBuf   │         │ DoubleBufferSeq   │        │ DoubleBufferSeq   │
     │   <ParamAtom>    │         │ <StateMirror>     │        │ <ARAConditioning> │
     │ lock-free atomic │         │ lock-free publish │        │ lock-free acquire │
     └─────────┬────────┘         └─────────┬─────────┘        └─────────┬─────────┘
               │                            │                            │
               └────────────────────────────┼────────────────────────────┘
                                            ▼
                           ┌─────────────────────────────────┐
                           │        EmT CLAP Plugins         │
                           │      (EmT Synth & EmT FX)       │
                           │ ┌─────────────────────────────┐ │
                           │ │     SSMInferenceCore        │ │
                           │ │  ┌───────────────────────┐  │ │
                           │ │  │   SSMBlock (Layer 0)  │  │ │
                           │ │  ├───────────────────────┤  │ │
                           │ │  │   SSMBlock (Layer 1)  │  │ │
                           │ │  └───────────────────────┘  │ │
                           │ └─────────────────────────────┘ │
                           └─────────────────────────────────┘
```

### Three-Thread Concurrency Model

Strict real-time safety rules are enforced across the entire audio processing chain:

| Thread | Role | Communication Mechanisms | RT Safety Constraints |
|--------|------|--------------------------|-----------------------|
| **Audio** | Real-time sample/block inference | Reads params via SPSC ring buffer; reads conditioning via `DoubleBufferSeqLock`; writes state mirror | **Zero `malloc`/`free`**, zero mutexes, zero blocking I/O, FTZ/DAZ enabled |
| **Main** | Host callbacks, Win32/X11 GUI, preset serialization | Pushes parameter changes to SPSC ring buffer; loads weights via `mmap` | Non-real-time; manages heap allocations and OS window handles |
| **ARA** | Background musical context analysis | Updates `ARAConditioningFrame` via seqlock write lock | High-priority asynchronous worker |

### Zero-Copy Polyphonic Voice Management

In polyphonic mode (up to 8 voices), previous designs triggered thousands of multi-kilobyte `memcpy` operations per block by repeatedly swapping voice state buffers into the inference engine (`set_state()` / `get_state()`).

EmT refactors `SSMInferenceCore::process_sample` and `SSMBlock::process_sample` to accept direct mutable `MambaState&` references:
```cpp
// Direct execution on voice-isolated state — 0 copies, 0 allocations:
m->core->process_sample(mono_input, voice_out, voice.ssm_state);
```
This refactoring completely eliminates 8,192 `memcpy` calls per 512-frame block while preserving voice separation.

### Sub-Sample Event Scheduling

Incoming CLAP MIDI notes and parameter automation events are sorted upfront by sample offset. The audio engine partitions block processing into contiguous sub-blocks bounded by event timestamps:
$$t_0 \longrightarrow t_1 \longrightarrow \dots \longrightarrow t_N \longrightarrow \text{frames}$$

Parameter smoothing (one-pole lowpass filter, 10 ms time constant) and envelope stage transitions are calculated at exact sub-sample boundaries, ensuring sample-accurate automation without per-sample event polling branches.

### State Persistence & Preset Engine

- **EmT Synth**: State saving and loading conforms to Schema v3 (`kStateSchemaV3`), capturing voice allocation, active parameters, and bypass states.
- **EmT FX**: State saving and loading implements `CLAP_EXT_STATE` (`MBFX` magic `0x4D424658`, version 1), preserving Gain, Mix, and Bypass settings.
- **Preset File Format**: Human-readable `.mambapreset` files saved to standard user directories (`%APPDATA%/mamba-ssm/presets/` on Windows; `~/.local/share/mamba-ssm/presets/` on Linux).

### Dual-Platform GUI Backend

The plugin implements native window embedding without heavy third-party GUI framework dependencies:
- **Windows (`CLAP_WINDOW_API_WIN32`)**: Custom Win32 window procedure (`WndProc`) with double-buffered GDI rendering for parameter knobs, dynamic meters, and preset buttons.
- **Linux (`CLAP_WINDOW_API_X11`)**: Native Xlib rendering with event dispatching.
- **Decoupled Architecture**: GUI reads parameters asynchronously from atomic caches, ensuring host UI repaints never contend with the audio thread.

### Celemony ARA 2.0 Integration & Fallback

Modular ARA 2.0 components live in `src/ara/`:
- `ara_document_controller.cpp`: Extracts musical tempo, time signature, bar positions, and pitch trajectories from DAW host tracks.
- `ara_playback_renderer.cpp`: Renders offline neural audio aligned to timeline regions.
- `ara_editor_renderer.cpp`: Provides interactive audio editing hooks.
- **Transparent Fallback**: When loaded in hosts that do not negotiate ARA 2.0, the plugin runs in standard real-time CLAP mode with zero overhead or configuration changes.

---

## Performance & Benchmarks

Run the harness and quote its output — never a number from a previous machine.
(CI wiring exists for this, but see the note under *Latency & Safety Metrics* about
Actions not starting in this repository since 2026-09-14.)

```bash
./build/tests/benchmark_ssm_block                       # default CI gate
./build/tests/benchmark_ssm_block --budget-pct 10        # the strategy doc's target
./build/tests/benchmark_ssm_block --config train4070,trainl4 --json bench.json
./build/tests/benchmark_ssm_block --config max --layers 1  # stress shape
```

Cost is reported as **microseconds per sample and percent of one CPU core** at a
stated sample rate, gated against an explicit budget (`--budget-pct`), and the
process **exits non-zero when the gate fails** so CI can enforce it.

Measured 2026-10-04 on an Intel Xeon @ 2.60 GHz container (2 threads visible),
gcc 12.2.0, `-O3 -mavx2 -mfma`. Core-only; a modern desktop is faster. Re-measure
on your target machine before publishing anything.

| Model shape | Layers | µs/sample | % of one core @48 kHz | Gate (40 %) |
|---|---|---|---|---|
| `small` (d_state 16, d_model 16, d_inner 32) | 1 | 3.03 | 14.6 % | PASS |
| `small` | 2 | 5.42 | 26.0 % | PASS |
| `train4070` (mirrors `configs/laptop_rtx4070.yaml`) | 1 | 4.3 | 20.6 % | PASS |
| `train4070` | 3 | 8.4–9.6 | 40–46 % | borderline |
| `train4070` | 6 | 15.5–16.7 | 74–80 % | FAIL |
| `trainl4` (mirrors `configs/runpod_l4.yaml`) | 1 | 18.8–20.1 | **90–96 %** | FAIL |
| `trainl4` | 6 | 114–117 | 546–564 % | FAIL |
| `max` (all compile-time maxima — stress shape) | 1 | 319 | 1532 % | FAIL |

**The honest headline:** a single small layer is comfortably real-time; the shapes
this repo's own training configs produce are not. One layer of the RunPod L4
config already consumes ≈90 % of one core at 48 kHz, and the plugin instantiates
the compile-time maximum configuration until a weight file is loaded
(`src/clap/clap_fx_plugin.cpp:58-59`). Shrink `d_inner`/`nheads`/`headdim` and
re-measure before shipping a neural model.

The dominant costs are the per-(head, dimension, state) scalar recurrence loop
(not yet SIMD-vectorized), the dense `in_proj` executed once per sample, and
scalar FiLM/RMSNorm/SiLU work — see `ARCHITECTURE_FIDELITY.md` §3.

### Latency & Safety Metrics

- **Algorithmic look-ahead**: 0 samples — the recurrence is causal and consumes
  one sample per step, so the plugin reports 0 latency via the CLAP latency
  extension. (Reported, not asserted: verify with `clap-validator` and the host.)
- **Audio Thread Heap Allocations**: 0 bytes (`malloc`/`free`/`new` strictly forbidden on the audio path).
- **Denormal Prevention**: Automatic FTZ (Flush-to-Zero) and DAZ (Denormals-are-Zero) configured in MXCSR per audio block.
- **CTest Pass Rate**: **67/67 tests** across 7 suites. The two tests that need
  generated inputs — `CppParity` (parity vectors) and
  `RealtimeSafety.FxAcceptsMultiLayerWeightsWithinLimits` (an exported model) —
  are given them by CI before `ctest`, and `MAMBA_REQUIRE_PARITY=1` turns a
  missing vector file into a failure instead of a silent skip.
- **Real-time gate**: `benchmark_ssm_block` exits non-zero when any measured
  configuration exceeds `--budget-pct` of one core. It is wired into
  `.github/workflows/ci.yml`, **but this repository's Actions runs have not
  started since 2026-09-14** (every run is a `startup_failure` with no jobs,
  including on the unmodified `main` workflow) — so the gate is wired, not
  currently enforced by CI. Run it locally (command below) until that is fixed;
  the evidence is in `ARCHITECTURE_FIDELITY.md` §4b.

---

## Build Instructions

### Prerequisites

- **CMake**: Version 3.20 or newer
- **C++20 Compiler**:
  - Windows: Visual Studio 2022 (MSVC v143+)
  - Linux: GCC 13+ or Clang 17+
- **Python**: Version 3.10+ (optional, required for weight export script)
- **Linux Libraries** (Linux only): `libx11-dev`

---

### Windows Build (MSVC)

Open a PowerShell terminal:

```powershell
# Configure CMake with Release build
cmake -B build -DCMAKE_BUILD_TYPE=Release

# Build all targets (C++ core, CLAP plugin, test executables)
cmake --build build --config Release -j4

# Execute CTest verification suite
ctest --test-dir build -C Release --output-on-failure
```

> **MSVC Stack Size Notice:**  
> The `SSMWeights` configuration struct is ~4 MB at maximum dimension settings. Windows executables and the CLAP plugin are configured with a **16 MB stack** (`/STACK:16777216`) in `CMakeLists.txt` to guarantee stack safety in DAW hosts.

---

### Linux Build (GCC / Clang)

```bash
# Install X11 headers (Ubuntu/Debian)
sudo apt-get install -y libx11-dev

# Configure and compile
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# Run full test suite
ctest --test-dir build --output-on-failure
```

---

### Running CTest & Benchmarks

Run CTest across all 64 registered unit and stability tests:
```powershell
ctest --test-dir build -C Release --output-on-failure
```

Run the standalone throughput benchmark:
```powershell
./build/tests/Release/benchmark_ssm_block.exe
```

---

### Plugin Installation Paths

After compilation, copy `EmTSynth.clap` to your system's CLAP plugin directory:

| OS | Default CLAP Directory |
|----|------------------------|
| **Windows** | `C:\Program Files\Common Files\CLAP\` |
| **Linux** | `~/.clap/` or `/usr/lib/clap/` |
| **macOS** | `/Library/Audio/Plug-Ins/CLAP/` |

---

## Training & Objectives

`training/train.py` imports the same model definition the exporter and the C++
`WeightLoader` use, so there is no train/inference drift:

```bash
# Task-matched objective (pre-emphasis L1 + SI-SDR + multi-resolution STFT)
python training/train.py --config training/configs/laptop_rtx4070.yaml \
    --dataset-dir data/wav --output-dir runs/run1

# Learn an effect: dry/ and wet/ subfolders are paired sample-for-sample
python training/train.py --config training/configs/laptop_rtx4070.yaml \
    --dataset-dir data/drywet --output-dir runs/fx --loss hybrid

# Objective properties (scale invariance, silent-target handling, gradients)
python training/losses.py

# 2-step end-to-end check with no dataset
python training/train.py --smoke-test --steps 2 --loss hybrid
```

Objectives: `mse` (legacy next-sample MSE, kept for reproducibility), `l1`,
`sisdr`, `mrstft`, and `hybrid` (the recommended default in the shipped configs).
The trainer **refuses to export untrained weights** if no optimisation step ran.

> **Read before training:** the shape you pick decides whether the result can run
> in an audio callback. See the measured table above and the warning in
> [`TRAINING_PLAN.md`](TRAINING_PLAN.md) — the RunPod L4 config is not real-time
> at one layer on a 2-core machine, and the L4 is not the bottleneck there.

---

## Weight Export & Validation Pipeline

Export trained PyTorch checkpoints into flat-binary zero-copy weight files:

```bash
# 1. Generate test weights with random initialization and refresh config.h
python scripts/export_mamba_weights.py \
    --state-dim 32 --model-dim 256 --inner-dim 512 --layers 2 \
    --nheads 8 --headdim 64 --cond-dim 128 --random-init \
    --output models/test_model.bin --config-h src/config.h

# 2. Validate round-trip numerical accuracy
python scripts/validate_export.py models/test_model.bin

# 3. Export production checkpoint
python scripts/export_mamba_weights.py \
    --load-checkpoint checkpoints/mamba_synth_final.pt \
    --output models/production.bin
```

---

## Directory Structure
 
 ```
 .
 ├── CMakeLists.txt               # Root CMake orchestration with enable_testing()
 ├── README.md                    # Comprehensive project documentation
 ├── CHANGELOG.md                 # Complete release history and gap records
 ├── TODO.md                      # Master gap analysis & verification evidence (100% DONE)
 ├── ARCHITECTURE.md              # Deep technical architectural overview
 ├── PARAMETER_REFERENCE.md       # Plugin parameter specifications
 ├── WEIGHT_FORMAT_SPEC.md        # SSM weight binary format specification
 ├── ARCHITECTURE_FIDELITY.md     # implemented-vs-paper ledger and claim policy
 ├── CLAIMS_VERIFICATION.md       # independent review of strategy/architecture claims
 ├── LICENSE                      # Apache 2.0 open-source license
 ├── src/
 │   ├── config.h                 # Compile-time MAX_* dimension limits
 │   ├── core/                    # Core SSM inference engine (static library)
 │   │   ├── ssm_block.h/.cpp     # SSM layer (B/C rotation, RMSNorm, bilinear scan)
 │   │   ├── ssm_inference_core.h/.cpp # Multi-layer inference coordinator
 │   │   ├── weight_loader.h/.cpp # Memory-mapped zero-copy weight loader
 │   │   └── simd_helpers.h       # AVX2/SSE2/NEON portable vector intrinsics
 │   ├── common/                  # Real-time lock-free primitives (interface library)
 │   │   ├── ring_buffer.h        # SPSC lock-free parameter ring buffer
 │   │   ├── seqlock.h            # Single-writer multi-reader SeqLock
 │   │   ├── double_buffer_seqlock.h # Atomic snapshot double-buffer exchange
 │   │   └── conditioning_frame.h # Conditioning structures (MIDI / ARA)
 │   ├── clap/                    # CLAP plugin implementations (shared library)
 │   │   ├── clap_entry.cpp       # Plugin factory (exports synth and fx)
 │   │   ├── clap_plugin.h/.cpp   # EmT Synth instrument plugin
 │   │   ├── clap_fx_plugin.h/.cpp# EmT FX audio effect plugin
 │   │   ├── clap_gui.h/.cpp      # Dual Win32 HWND & Linux X11 native GUI
 │   │   ├── clap_params.cpp      # Parameter definitions & smoothing
 │   │   ├── clap_state.cpp       # Plugin state serialization (Schema v3)
 │   │   ├── clap_presets.h/.cpp  # Preset file save/load engine
 │   │   ├── clap_common_utils.h  # Common CLAP conversion & validation utilities
 │   │   ├── clap_audio_ports.cpp # Stereo audio port configuration
 │   │   ├── clap_note_ports.cpp  # MIDI note input port configuration
 │   │   ├── clap_tail.cpp        # Reverb/decay tail handling (100 ms)
 │   │   └── clap_latency.cpp     # Algorithmic latency reporter (0 samples)
 │   └── ara/                     # Celemony ARA 2.0 integration (static library)
 │       ├── ara_document_controller.h/.cpp # Musical context extraction
 │       ├── ara_playback_renderer.cpp     # Timeline region playback renderer
 │       ├── ara_editor_renderer.cpp       # Editor renderer
 │       └── ara_types.h          # ARA data structures & forward declarations
 ├── tests/                       # GoogleTest verification suite (67 tests)
 │   ├── test_ssm_block.cpp       # Recurrence, multi-layer, state parity tests
 │   ├── test_weight_loader.cpp   # mmap validation, quantization, dimension checks
 │   ├── test_conditioning.cpp    # FiLM modulation, seqlock tear resistance
 │   ├── test_stability.cpp       # Extreme parameter stability, zero-input decay
 │   ├── test_realtime_safety.cpp # Zero allocation audit, zero-copy voice test, FX state roundtrip
 │   ├── test_cpp_parity.cpp      # C++ vs Python numerical parity verification
 │   ├── test_ara.cpp             # Celemony ARA 2.0 document controller & playback renderer tests
 │   └── benchmark_ssm_block.cpp  # Real-time multi-rate throughput benchmark
 ├── scripts/                     # Python offline toolchain
 │   ├── export_mamba_weights.py  # Checkpoint exporter & config.h generator
 │   ├── validate_export.py       # Binary weight validator
 │   └── reference_mamba3.py      # NumPy reference implementation for parity vectors
 ├── training/                    # Training (PyTorch)
 │   ├── train.py                 # Trainer: --loss, paired dry/wet, export, smoke-test
 │   ├── losses.py                # Task-matched objectives + self-test
 │   └── configs/                 # laptop_rtx4070.yaml, runpod_l4.yaml (see the CPU warning in TRAINING_PLAN.md)
 └── models/                      # Binary model files (.bin)
 ```

---

## License

This project is licensed under the Apache 2.0 License. See the [LICENSE](LICENSE) file for details.
