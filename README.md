# Mamba-SSM CLAP/ARA Plugin Suite

Real-time neural audio synthesis plugins powered by the Mamba state-space model architecture. Delivered as CLAP plugins with optional ARA 2.0 integration for DAW-hosted generative workflows.

## Plugin Suite

| Plugin | Status | Description |
|--------|--------|-------------|
| **Mamba Synth** | 85% complete | Generative audio synthesizer from MIDI/ARA conditioning |
| **Mamba FX** | Not started | Long-context neural audio effect processor |
| **ARA Companion** | Not started | Offline region generation with musical context |

All three plugins share the SSM inference core.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                         HOST DAW                              │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────────┐  │
│  │ Audio Thread  │   │  Main Thread  │   │    ARA Thread    │  │
│  │ process()     │   │  params/flush │   │ musical context  │  │
│  └───┬───────────┘   └──┬───────────┘   └────────┬─────────┘  │
└──────┼──────────────────┼────────────────────────┼────────────┘
       │                  │                        │
       ▼                  ▼                        ▼
┌──────────────┐  ┌──────────────────┐  ┌────────────────────┐
│ SPSC RingBuf │  │ Atomic params    │  │ Seqlocked           │
│ <ParamAtom>  │  │ (std::atomic)    │  │ ARAConditioningFrame│
│ lock-free    │  │ lock-free r/w    │  │ acquire/release     │
└──────┬───────┘  └────────┬─────────┘  └─────────┬──────────┘
       │                   │                      │
       └───────────────────┼──────────────────────┘
                           ▼
               ┌───────────────────────┐
               │   MambaSynth Plugin    │
               │  ┌─────────────────┐  │
               │  │ SSMInferenceCore │  │
               │  │  SSMBlock × L   │  │
               │  └─────────────────┘  │
               └───────────────────────┘
```

**Audio thread constraints:** zero `malloc`/`free`, zero `mutex`, zero blocking I/O. All working memory is pre-allocated in `activate()`.

### SSM Processing Pipeline

```
Input (MIDI/ARA) → W_in + b_in → causal conv1d → SiLU → selective scan → W_out + W_ssm·h → Output
                                    ↕                             ↕
                           dt/B/C projections            dt/B conditioned by
                         (learned per-step)             ARA musical context
```

The SSM recurrence uses ZOH-discretized selective scan (per the Mamba paper), computed elementwise per state dimension:
- **Ā** = exp(δ·A): discrete-time state transition
- **B̄** = (exp(δ·A) − 1)·A⁻¹·B: discrete-time input projection
- **State update:** hₜ = Ā·hₜ₋₁ + B̄·uₜ
- **Output:** yₜ = C·hₜ

Conditioning modulates δ and B via learned linear projections from a 16-dimensional conditioning vector (extracted from MIDI note/velocity and ARA musical analysis).

---

## Quick Start

### Prerequisites

- **CMake** 3.20+
- **C++20** compiler (GCC 13+ or Clang 17+)
- **Python** 3.10+ (for weight export; requires PyTorch + NumPy)

### Build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

### Test

```bash
ctest --test-dir build --output-on-failure
# 33/33 tests passing
```

### Benchmark

```bash
./build/tests/benchmark_ssm_block
# Single layer:  ~0.65 us/sample  (15x below 10us target)
# Two layers:    ~1.33 us/sample
```

### Plugin Installation

| Platform | Location |
|----------|----------|
| **Linux** | `cp build/src/clap/MambaSynth.clap ~/.clap/` |
| **macOS** | Place `MambaSynth.clap` bundle in `/Library/Audio/Plug-Ins/CLAP/` |
| **Windows** | Copy `MambaSynth.clap` to `C:\Program Files\Common Files\CLAP\` |

---

## Weight Export & Validation

```bash
# Generate test weights with random initialization
python3 scripts/export_mamba_weights.py \
    --state-dim 16 --model-dim 16 --inner-dim 32 --conv-dim 4 \
    --layers 2 --cond-dim 16 --random-init \
    --output models/test_model.bin --config-h src/config.h

# Validate round-trip correctness
python3 scripts/validate_export.py models/test_model.bin

# Export from trained PyTorch checkpoint
python3 scripts/export_mamba_weights.py \
    --load-checkpoint trained_model.pt --output models/production.bin
```

The `--config-h` flag auto-generates `src/config.h` with `MAX_*` constants matching the weight file dimensions, ensuring buffer sizes are correct.

---

## Current Status

| Subsystem | Status | Notes |
|-----------|--------|-------|
| **SSMBlock** (single-layer) | Done | ZOH discretization, selective scan, SiLU, causal conv1d, conditioning |
| **SSMInferenceCore** (multi-layer) | Done | Placement-new construction, get/set state, process_block ping-pong |
| **WeightLoader** (mmap) | Done | Magic/version validation, FP16/INT8 dequant, 3-tier path resolution |
| **Lock-free primitives** | Done | SPSCRingBuffer, SeqLock/Seqlocked, DoubleBufferSeqLock |
| **CLAP plugin** | Done | Entry factory, audio/note/params/state/tail/latency extensions |
| **Parameter smoothing** | Done | One-pole lowpass, 10ms time constant |
| **Bypass crossfade** | Done | 4-state machine, click-free transitions |
| **Voice management** | Done | Monophonic, attack/sustain/release envelope |
| **Python export pipeline** | Done | PyTorch → flat binary + round-trip validation |
| **SIMD helpers** | Done | SSE2/NEON dot product, fmadd, scale |
| **Tests** | Done | 33/33 passing (SSM, weight loader, conditioning, stability, RT safety) |
| **GUI** | Not started | `CLAP_EXT_GUI` returns nullptr |
| **Polyphonic voices** | Not started | Currently monophonic |
| **process_block() vectorized** | Not started | Falls back to per-sample loop |
| **ARA SDK** | Blocked | Custom stubs; needs Celemony SDK |
| **Mamba FX plugin** | Not started | — |
| **ARA Companion plugin** | Not started | Blocked on ARA SDK |

---

## Directory Structure

```
├── src/
│   ├── config.h                     Compile-time MAX_* constants
│   ├── core/                        SSM inference engine (static lib)
│   │   ├── ssm_block.h/.cpp         Single Mamba-SSM layer
│   │   ├── ssm_inference_core.h/.cpp Multi-layer wrapper
│   │   ├── weight_loader.h/.cpp     mmap weight loading + dequantization
│   │   ├── arena_allocator.h/.cpp   mmap-backed bump allocator
│   │   └── simd_helpers.h           SSE2/NEON intrinsics
│   ├── common/                      Lock-free primitives (interface lib)
│   │   ├── ring_buffer.h            SPSCRingBuffer<ParamAtom, 256>
│   │   ├── seqlock.h                SeqLock + Seqlocked<T>
│   │   ├── double_buffer_seqlock.h  DoubleBufferSeqLock<T>
│   │   └── conditioning_frame.h     MIDI/ARA conditioning data
│   ├── clap/                        CLAP plugin (shared lib)
│   │   ├── clap_entry.cpp           clap_entry factory
│   │   ├── clap_plugin.h/.cpp       Main plugin struct + process()
│   │   ├── clap_audio_ports.cpp     Audio port extension
│   │   ├── clap_note_ports.cpp      Note port extension
│   │   ├── clap_params.cpp          Parameter extension (5 params)
│   │   ├── clap_state.cpp           State save/load
│   │   ├── clap_tail.cpp            Tail extension (100ms)
│   │   ├── clap_latency.cpp         Latency extension (d_conv-1)
│   │   └── clap_thread_check.h      Host thread-check helper
│   └── ara/                         ARA stubs (static lib)
│       ├── ara_types.h              Forward declarations
│       ├── ara_document_controller.h/.cpp Musical context stubs
│       ├── ara_playback_renderer.cpp Offline render stub
│       ├── ara_editor_renderer.cpp   Editor stub
│       └── ara_renderer.h           C function pointer API
├── tests/                           GoogleTest suite
│   ├── test_ssm_block.cpp           9 tests
│   ├── test_weight_loader.cpp       21 tests
│   ├── test_conditioning.cpp        3 tests
│   ├── test_stability.cpp           3 tests
│   ├── test_realtime_safety.cpp     6 tests
│   └── benchmark_ssm_block.cpp      Throughput benchmark
├── scripts/                         Python export pipeline
│   ├── export_mamba_weights.py      PyTorch → flat binary
│   ├── validate_export.py           Binary → PyTorch round-trip
│   └── reference_conditioning.py    Conditioning math reference
├── models/                          Exported .bin weight files
├── extern/clap/                     Vendored CLAP 1.1+ headers
├── ADR's/                           Architecture Decision Records
├── Prompts/                         Implementation prompts
├── TODO.md                          Gap analysis + phased plan
├── GAP_ANALYSIS.md                  Comprehensive codebase audit
├── ARCHITECTURE.md                  Deep architecture overview
├── PARAMETER_REFERENCE.md           Plugin parameter documentation
├── WEIGHT_FORMAT_SPEC.md            Weight file format specification
├── CHANGELOG.md                     Release history
└── CMakeLists.txt                   Root build system
```

## License

Apache 2.0 — see [LICENSE](LICENSE).
