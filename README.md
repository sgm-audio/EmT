# EmT CLAP/ARA Plugin Suite

[![C++20](https://img.shields.io/badge/C%2B%2B-20-blue.svg)](https://en.cppreference.com/w/cpp/20)
[![CLAP](https://img.shields.io/badge/CLAP-1.2+-purple.svg)](https://cleveraudio.org/)
[![ARA 2](https://img.shields.io/badge/ARA-2.0-orange.svg)](https://www.celemony.com/en/ara)
[![CTest](https://img.shields.io/badge/tests-64%2F64%20passing-brightgreen.svg)]()
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

Real-time neural audio synthesis and effect processing plugins powered by the **Mamba-3 state-space model (SSM)** architecture. Delivered as low-overhead, real-time safe **CLAP** plugins with optional **ARA 2.0** integration for DAW-hosted generative workflows.

> **Project Identity:** This project is **EmT** (migrated from `mamba-ssm-plugin-suite-70317161`), a distinct Mamba-SSM CLAP/ARA plugin and training suite. It is separate from CLARA. No "0" in the name.

---

## Table of Contents

- [Plugin Suite Overview](#plugin-suite-overview)
- [Mamba-3 Core Innovations](#mamba-3-core-innovations)
- [Architecture](#architecture)
  - [Three-Thread Concurrency Model](#three-thread-concurrency-model)
  - [Zero-Copy Polyphonic Voice Management](#zero-copy-polyphonic-voice-management)
  - [Sub-Sample Event Scheduling](#sub-sample-event-scheduling)
  - [State Persistence & Preset Engine](#state-persistence--preset-engine)
  - [Dual-Platform GUI Backend](#dual-platform-gui-backend)
  - [Celemony ARA 2.0 Integration & Fallback](#celemony-ara-20-integration--fallback)
- [Performance & Benchmarks](#performance--benchmarks)
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

## Mamba-3 Core Innovations

EmT utilizes the Mamba-3 state-space model formulation, optimized specifically for real-time digital signal processing:

1. **Multi-Head Complex State Recurrence**:
   Each layer decomposes the hidden state across $N_H$ heads with complex state representations ($h \in \mathbb{C}^{D_S}$):
   $$\mathbf{h}_t = \bar{\mathbf{A}} \mathbf{h}_{t-1} + \bar{\mathbf{B}} u_t$$
   $$y_t = \text{Re}\{\mathbf{C}^\dagger \mathbf{h}_t\} + \mathbf{D} u_t$$
2. **Trapezoidal (Bilinear) Discretization**:
   Replaces crude Euler or Zero-Order Hold (ZOH) discretization with trapezoidal integration:
   $$\bar{A} = \frac{1 + \frac{1}{2} A \Delta}{1 - \frac{1}{2} A \Delta}, \quad \bar{B} = \frac{\Delta}{1 - \frac{1}{2} A \Delta} B$$
   Guarantees A-stable discretization across high sample rates (44.1 kHz – 192 kHz).
3. **Rotary Position Embedding (RoPE) & RMSNorm**:
   Applies continuous angular rotation per head to encode pitch and temporal phase directly into recurrence states, stabilized by RMSNorm on input projections $B$ and $C$.
4. **Feature-wise Linear Modulation (FiLM)**:
   Modulates time-step $\Delta$ and projection matrix $B$ using conditioning vectors derived from MIDI pitch, velocity, and DAW musical context.
5. **Zero Algorithmic Latency**:
   Mamba-3 eliminates the causal 1D convolution (`conv1d`) present in Mamba-1/2, achieving **0 samples of algorithmic delay** for instantaneous live playing.

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

Benchmarked on AMD x86-64 hardware using AVX2 SIMD vectorization:

### Throughput Benchmarks (`benchmark_ssm_block.exe`)

| Configuration | Processing Mode | Throughput | Samples / Second | Real-Time Headroom |
|---------------|-----------------|------------|------------------|-------------------|
| **Single Layer** | Per-sample scalar | 2.047 µs / sample | 488,596 s/sec | **10.2× RT** (at 48 kHz) |
| **Single Layer** | Vectorized block (64) | **1.985 µs / sample** | **503,853 s/sec** | **10.5× RT** (at 48 kHz) |
| **Single Layer** | Vectorized block (512) | 2.008 µs / sample | 497,930 s/sec | **10.4× RT** (at 48 kHz) |
| **2 Layers** | Per-sample scalar | 3.820 µs / sample | 261,776 s/sec | **5.5× RT** (at 48 kHz) |
| **2 Layers** | Vectorized block (512) | 4.562 µs / sample | 219,223 s/sec | **4.6× RT** (at 48 kHz) |

*Target Latency Budget:* $< 10.0$ µs / sample (achieved $1.985$ µs, exceeding project target by **5×**).  
*CPU Budget at 48 kHz (20.83 µs budget):* Single layer uses only **9.5%** of available audio thread time.

### Latency & Safety Metrics

- **Algorithmic Latency**: 0 samples (zero algorithmic delay).
- **Audio Thread Heap Allocations**: 0 bytes (`malloc`/`free`/`new` strictly forbidden on the audio path).
- **Denormal Prevention**: Automatic FTZ (Flush-to-Zero) and DAZ (Denormals-are-Zero) configured in MXCSR per audio block.
- **CTest Pass Rate**: **64/64 tests passing** (100% pass rate in ~2.6 seconds).

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
 ├── WEIGHT_FORMAT_SPEC.md        # Mamba-3 binary format specification
 ├── LICENSE                      # Apache 2.0 open-source license
 ├── src/
 │   ├── config.h                 # Compile-time MAX_* dimension limits
 │   ├── core/                    # Core SSM inference engine (static library)
 │   │   ├── ssm_block.h/.cpp     # Mamba-3 layer (RoPE, RMSNorm, trapezoidal scan)
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
 ├── tests/                       # GoogleTest verification suite (64 tests)
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
 │   └── reference_mamba3.py      # NumPy Mamba-3 reference for parity vectors
 └── models/                      # Binary model files (.bin)
 ```

---

## License

This project is licensed under the Apache 2.0 License. See the [LICENSE](LICENSE) file for details.
