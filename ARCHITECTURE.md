# Architecture Overview

**Mamba-SSM CLAP/ARA Plugin Suite** — v0.1.0

---

## 1. System Architecture

### 1.1 Three-Thread Model

```
┌──────────────────────────────────────────────────────────┐
│                       HOST DAW                           │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────┐│
│  │   Audio Thread   │  │   Main Thread   │  │ ARA Thread││
│  │ plugin_process()  │  │ params_flush()  │  │  offline   ││
│  │ SSM inference    │  │ state save/load │  │  render   ││
│  └───────┬──────────┘  └───────┬─────────┘  └─────┬─────┘│
└──────────┼─────────────────────┼───────────────────┼──────┘
           │                     │                   │
           ▼                     ▼                   ▼
   ┌───────────────┐   ┌─────────────────┐   ┌──────────────┐
   │ SPSC Ring Buf │   │ Atomic params   │   │  Seqlocked    │
   │ <ParamAtom,256>│   │ (double)        │   │ ARAFrame     │
   │ lock-free push │   │ lock-free store │   │ write_lock/  │
   │     pop        │   │      load       │   │  unlock      │
   └───────────────┘   └─────────────────┘   └──────────────┘
           ▲                     ▲                   ▲
           │                     │                   │
   ┌───────┴─────────────────────┴───────────────────┴──────┐
   │                   MambaSynth Plugin                     │
   │  ┌─────────────────────────────────────────────────┐   │
   │  │              SSMInferenceCore                     │   │
   │  │  ┌──────────┐  ┌──────────┐      ┌──────────┐  │   │
   │  │  │ SSMBlock │→│ SSMBlock │→ ... →│ SSMBlock │  │   │
   │  │  │ Layer 0  │  │ Layer 1  │      │ Layer L-1│  │   │
   │  │  └──────────┘  └──────────┘      └──────────┘  │   │
   │  └─────────────────────────────────────────────────┘   │
   └────────────────────────────────────────────────────────┘
```

| Thread | Role | Communication Protocol | Constraints |
|--------|------|------------------------|-------------|
| **Audio** | `plugin_process()`, SSM inference, voice envelope | Reads `ParamAtom` from SPSC ring buffer; reads conditioning via seqlocked frames; publishes `StateMirror` via `DoubleBufferSeqLock` | **Zero** `malloc`/`free`, **zero** `mutex`, **zero** blocking I/O |
| **Main** | UI, preset save/load, parameter automation | Pushes `ParamAtom` to SPSC ring buffer; reads `StateMirror` for state save; atomic param stores | No real-time constraints |
| **ARA** | Musical context computation, offline region render | Writes `ARAConditioningFrame` via seqlock | Stub implementation (blocked on ARA SDK) |

---

## 2. Lock-Free Primitives

### 2.1 SPSCRingBuffer\<ParamAtom, 256\>

Single-Producer Single-Consumer ring buffer. The main thread pushes `ParamAtom` structs; the audio thread drains the queue in `process()`.

```cpp
struct ParamAtom {
    clap_id id;
    double value;
    uint32_t frame_offset;  // sample-accurate offset within block
};
```

Capacity: 256 atoms. Back-pressure: if full, the push is silently dropped (audio keeps previous value).

**File:** `src/common/ring_buffer.h`

### 2.2 Seqlocked\<T\>

Seqlock (sequential lock) pattern with acquire/release ordering. Writer holds a generation counter; reader spins until generation is even and stable.

```cpp
template<typename T>
class Seqlocked {
    void write_lock();    // odd generation → write in progress
    void write_unlock();  // release → even generation
    T read();             // retry loop until stable
};
```

- `write_lock()`: `memory_order_acquire`
- `write_unlock()`: `memory_order_release`
- Reader spin checks `generation % 2 == 0` and `generation == initial_gen`

Used for: `ConditioningFrame`, `ARAConditioningFrame`.

**File:** `src/common/seqlock.h`

### 2.3 DoubleBufferSeqLock\<T\>

Double-buffered seqlock. Two slots backed by `std::atomic<uint32_t>` write-index + per-slot seqlocks.
The writer writes to the "back" slot; `swap()` atomically flips which slot is back.
The reader reads from the "front" slot (guaranteed to be the last fully committed write).

Used for: `StateMirror` (state save snapshots). The audio thread writes to the back slot each `process()` call; the main thread reads the front slot when the host requests a state save.

**File:** `src/common/double_buffer_seqlock.h`

---

## 3. SSM Inference Pipeline

### 3.1 Processing Steps (per sample, per layer)

```
  Input (scalar)                     Conditioning (COND_DIM vector)
      │                                      │
      ▼                                      │
  ┌─────────┐                                │
  │  W_in    │  [d_model] weights row         │
  │  + b_in  │  → x_inner (d_inner vec)     │
  └────┬────┘                                │
       ▼                                      │
  ┌─────────┐                                │
  │ Conv1d   │  causal, depthwise, kernel=d_conv
  │ + SiLU   │  → conv_out (d_inner vec)
  └────┬────┘                                │
       ▼                                      │
  ┌─────────────────────────────────────────┐│
  │  Selective Scan (per state dim i):        ││
  │                                           ││
  │  dt_i = softplus(dt_proj_w[i]·conv_out   ││
  │                 + dt_bias[i]              ││
  │                 + cond_proj_dt[i]·cond    ││ ← conditioning
  │                 + cond_dt_bias[i])        ││
  │                                           ││
  │  Ā_i = exp(A_log[i] * dt_i)              ││
  │     = exp(-exp(A_log[i]) * dt_i)         ││  (A is negative)
  │                                           ││
  │  B̄_i = (Ā_i - 1) / A_i * B_i            ││  (ZOH discretization)
  │       where B_i = B_proj_w[i]·conv_out   ││
  │                   + cond_proj_B[i]·cond   ││ ← conditioning
  │                   + cond_B_bias[i]        ││
  │                                           ││
  │  h_i = Ā_i * h_i + B̄_i * u_i            ││  (state update)
  │       where u_i = conv_out[i]            ││
  │                                           ││
  │  C_i = C_proj_w[i]·conv_out              ││  (output gate)
  │  y_i = C_i * h_i                         ││
  └─────────────────────────────────────────┘│
       ▼
  ┌─────────┐
  │  W_out   │  output projection + skip
  │  W_ssm   │  y = W_out·conv_out + W_ssm·y_ssm
  └────┬────┘
       ▼
  Output (d_model floats)
```

### 3.2 ZOH Discretization

```
A_log ∈ ℝᵈˢᵗᵃᵗᵉ           (learned log of negative A)
A = -exp(A_log)                (A < 0 → stable eigenvalues)
δ = softplus(W_δ·x + b_δ)      (per-timestep step size)
Ā = exp(δ·A)                   (discrete A via matrix exponential)
B̄ = (exp(δ·A) - 1) · A⁻¹ · B  (discrete B via ZOH integral)

Numerically stable path:
  B̄ = where(|A| > 1e-6, (Ā - 1)/A * B, δ * B)  // δ*B when A→0
```

### 3.3 Conditioning Mechanism

A learned linear projection maps the conditioning vector to per-state biases on δ and B:

```
δ_effective = softplus(δ_base + W_cond_δ · cond + b_cond_δ)
B_effective = B_base + W_cond_B · cond + b_cond_B
```

The conditioning vector (16 floats) originates from:
- **ARA analysis:** tempo, key signature, chord, spectral content (stubs)
- **MIDI input:** note number, velocity (mapped to conditioning bins)

`C` is currently **not** conditioned (no `cond_proj_C` — future enhancement).

---

## 4. Memory Architecture

### 4.1 Weight File (mmap, read-only)

```
┌──────────────────────┐  0x0000
│  WeightFileHeader     │  64 bytes (cache-aligned)
│  - magic, version    │
│  - dimensions        │
│  - quantization info │
├──────────────────────┤  0x0040
│  Q-Metadata (opt)    │  0-8L bytes (for INT8/FP16)
├──────────────────────┤
│  Layer 0: 16 arrays  │  floats_per_layer × 4 bytes
├──────────────────────┤
│  Layer 1: 16 arrays  │
├──────────────────────┤
│  ...                 │
└──────────────────────┘
```

- **mmap** with `MAP_PRIVATE` → zero-copy load, pages loaded on-demand by kernel
- **Validation** at load time: magic check, version check, dimension fit check
- **Dequantization** (INT8/FP16) happens at load time (main thread), not in audio thread
- Weights copied into `SSMWeights` (stack-allocated struct on `MambaSynth`)

### 4.2 Plugin Allocation (Arena)

```cpp
// clap_plugin.cpp
static ArenaAllocator s_plugin_arena(/* mmap-backed */);

// MambaSynth and clap_plugin_t are allocated from the arena
auto* m = s_plugin_arena.allocate(sizeof(MambaSynth), alignof(MambaSynth));
auto* p = s_plugin_arena.allocate(sizeof(clap_plugin_t), alignof(clap_plugin_t));
```

- No `free()` — arena lives for process lifetime
- Prevents fragmentation, guarantees contiguous allocation
- All audio-thread scratch buffers are stack-allocated (no heap)

### 4.3 Stack-Allocated Audio Buffers

The audio thread never touches the heap. All working buffers are on the stack:

| Buffer | Size | Location |
|--------|------|----------|
| `SSMState` (h + conv_state) | ~d_state + d_conv·d_inner floats | Class member (cache-aligned) |
| `SSMWeights` | ~3K floats (at MAX config) | Class member |
| `x_inner_` | d_inner floats | Class member (alignas(64)) |
| `conv_out_` | d_inner floats | Class member (alignas(64)) |
| `dt_`, `B_`, `C_`, `ss_` | d_state floats each | Class member (alignas(64)) |
| `y_out_` | d_model floats | Class member (alignas(64)) |
| `cond_` | COND_DIM floats | Class member (alignas(64)) |
| `buf_dm[]` (in `process_block`) | MAX_BLOCK_SIZE × MAX_D_MODEL floats | Stack (32 KB at max) |

### 4.4 Cache Alignment

All shared/atomic data structures use `alignas(64)` to prevent false sharing:
- `SSMState`, `MambaState`, `ConditioningFrame`, `ARAConditioningFrame`
- `ParamSmooth[]`, `bypass_state`, `bypass_fade`, `voice_state`, `voice_env`
- All internal `SSMBlock` scratch arrays
- `StateMirror`, `ThreadCheck`, `weight_path`

---

## 5. CLAP Extension Status

| Extension | Implemented | Source File | Notes |
|-----------|-------------|-------------|-------|
| `CLAP_EXT_AUDIO_PORTS` | Yes | `clap_audio_ports.cpp` | 2 stereo + 2 mono ports (in/out) |
| `CLAP_EXT_NOTE_PORTS` | Yes | `clap_note_ports.cpp` | MIDI + CLAP dialect, 1 note input |
| `CLAP_EXT_PARAMS` | Yes | `clap_params.cpp` | 5 parameters (Gain, Attack, Release, Mix, Bypass) |
| `CLAP_EXT_STATE` | Yes | `clap_state.cpp` | Binary save/load with magic + version |
| `CLAP_EXT_TAIL` | Yes | `clap_tail.cpp` | 100ms tail |
| `CLAP_EXT_LATENCY` | Yes | `clap_latency.cpp` | `d_conv - 1` samples |
| `CLAP_EXT_THREAD_CHECK` | Yes | `clap_thread_check.h` | Host extension; `thread::id` fallback |
| `CLAP_EXT_GUI` | **No** | — | Returns nullptr |
| `CLAP_EXT_TIMER_SUPPORT` | **No** | — | Not needed yet |
| `CLAP_EXT_PRESET_LOAD` | **No** | — | Only raw state save/load |

---

## 6. ARA Integration

### 6.1 Current State (Stubs)

All ARA files are custom stubs that mimic the ARA 2.0 API without linking the real Celemony SDK:

| File | Purpose | Completeness |
|------|---------|-------------|
| `src/ara/ara_types.h` | Forward declarations for `ARA::PlugIn` namespace | Stub |
| `src/ara/ara_document_controller.h/.cpp` | Musical context management (tempo, key, chords) | Functional stubs |
| `src/ara/ara_playback_renderer.cpp` | Offline rendering (renders silence) | Stub |
| `src/ara/ara_editor_renderer.cpp` | Editor view (stores audio in ring buffer) | Stub |
| `src/ara/ara_renderer.h` | C function pointer API | Functional |

Global helper functions available:
- `ARA_set_tempo()`, `ARA_set_key_signature()`, `ARA_set_chord()`
- `ARA_get_conditioning()` — returns seqlocked ARAConditioningFrame
- `ARA_write_conditioning_to()` — populates a ConditioningFrame from ARA data

### 6.2 ARA SDK Block

Real ARA integration requires the Celemony ARA SDK (proprietary, available upon request from [celemony.com](https://celemony.com)). To enable:

1. Download and extract the ARA SDK
2. Set `ARA_PATH` env var to SDK root
3. Set `ARA_INCLUDE_DIR` CMake variable to SDK include path
4. Remove `#define MAMBA_ARA_REAL_SDK` guard from `ara_types.h`
5. Replace stubs with real `ARA::PlugIn::PlugIn`, `DocumentController`, etc.

---

## 7. config.h Relationship to Weight Files

`src/config.h` defines `MAX_*` compile-time constants that determine buffer sizes:

```c
#define MAX_D_STATE  16   // must be >= weight file d_state
#define MAX_D_MODEL  16   // must be >= weight file d_model
#define MAX_D_INNER  32   // must be >= weight file d_inner
#define MAX_D_CONV    4   // must be >= weight file d_conv
#define MAX_LAYERS    2   // must be >= weight file num_layers
#define COND_DIM     16   // must be >= weight file cond_dim
```

- **SSMWeights is fixed-size** (stack-allocated 2D arrays using `MAX_*` dims)
- `WeightLoader::fits_in_static_weights()` enforces file dims ≤ `MAX_*`
- Changing model size requires regenerating `config.h` and recompiling
- The export script (`--config-h`) auto-generates matching `config.h`

---

## 8. SSMInferenceCore Multi-Layer Wrapper

Manages `SSMBlock[MAX_LAYERS]` via placement new into pre-allocated storage.

```
SSMInferenceCore
  │
  ├── SSMBlock layer_0  (placement-new into core_storage[0])
  ├── SSMBlock layer_1  (placement-new into core_storage[1])
  └── ...

MambaConfig cfg: {.num_layers, .d_state, .d_model, .d_inner, .d_conv}
```

**Per-sample path:** Each layer processes a mono signal → d_model output vector → summed to scalar → next layer.
**Per-block path:** Ping-pong buffers between `d_model`-sized frames.

**State management:** `get_state(MambaState&)` / `set_state(const MambaState&)` serializes/deserializes
all layer SSM states (h + conv_state) for preset save/load.

---

## 9. Processing Modes

### Mode A (Tiny) — Per-Sample

- `process_sample(input, output[d_model])` — processes one sample through all layers
- Ultra-low latency (1 sample = 1 inference step)
- Used by the plugin in the current implementation
- Throughput: ~0.65 us/sample (single layer), ~1.33 us/sample (2 layers) at MAX config

### Mode B (Block) — Not Yet Used in Plugin

- `process_block(input[n], output[n*d_model], n)` — processes n samples
- Ping-pong d_model-dimensional buffers between layers
- The underlying `SSMBlock::process_block()` currently falls back to a per-sample loop
- Target: 64-512 sample blocks for SIMD-vectorized batch processing

---

## 10. Voice Management

Monophonic voice with envelope state machine:

```
  VOICE_IDLE
      │ note_on
      ▼
  VOICE_ATTACK  ──→ voice_env ramps 0→velocity over attack_time
      │ env >= velocity
      ▼
  VOICE_SUSTAIN ──→ voice_env = velocity (constant)
      │ note_off
      ▼
  VOICE_RELEASE ──→ voice_env ramps velocity→0 over release_time
      │ env <= 0
      ▼
  VOICE_SILENT  ──→ voice_env = 0, ready for new note
      │ note_on
      └──→ VOICE_ATTACK
```

- Envelope rates precomputed at activation time from attack/release parameters
- `voice_sample_count` tracks frames in each phase
- Output: `sample = voice_env * ssm_output`

---

## 11. Bypass Crossfade State Machine

```
  BYPASS_PROCESSING ──→ full wet signal
      │ bypass=true
      ▼
  BYPASS_FADING_OUT ──→ crossfade wet→dry over ~64 samples
      │ fade <= 0
      ▼
  BYPASS_BYPASSED   ──→ full dry passthrough
      │ bypass=false
      ▼
  BYPASS_FADING_IN  ──→ crossfade dry→wet over ~64 samples
      │ fade >= 1
      └──→ BYPASS_PROCESSING
```

- Fade rate: `1.0 / 64.0` per sample
- Guarantees click-free bypass transitions

---

## 12. Test Architecture

```
tests/
├── test_ssm_block.cpp        9 tests  — block reset, dimensions, multi-layer, state round-trip
├── test_weight_loader.cpp   21 tests  — load, validate, INT8/FP16, bad magic/version/size
├── test_conditioning.cpp     3 tests  — modulation, seqlock atomicity, Python reference
├── test_stability.cpp        3 tests  — 300s long-run, extreme params, zero-input
├── test_realtime_safety.cpp  6 tests  — (3 stubs, 3 functional)
└── benchmark_ssm_block.cpp   manual   — throughput: 0.65 us/sample single-layer
```

**Total:** 33/33 passing (as of 2026-06-14).

### Key Tests

| Test | What It Verifies |
|------|-----------------|
| `SSMBlock_ResetClearsState` | `reset()` zeroes h and conv_state |
| `SSMBlock_ProcessBlockOutputSizes` | `process_block()` writes `n * d_model` floats |
| `WeightLoader_MagicMismatchReturnsError` | File with wrong magic rejected |
| `WeightLoader_Int8Dequantization` | INT8 → FP32 conversion correct |
| `WeightLoader_FP16RoundTrip` | `fp16_to_fp32()` round-trip within FP16 precision |
| `Conditioning_ZeroDetectPreventsModulation` | Zero cond vector → no SSM parameter change |
| `Stability_LongRun10Minutes` | 300s @ 44.1kHz → no NaN/inf/state explosion |

---

## 13. Build Pipeline

```
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4
ctest --test-dir build --output-on-failure

# Components:
src/core/     → static lib 'core'       (SSMBlock, WeightLoader, ArenaAllocator, SIMD)
src/common/   → interface lib 'common' (RingBuffer, SeqLock, Conditioning)
src/ara/      → static lib 'ara_plugin'(ARA stubs)
src/clap/     → shared lib 'mamba_synth' (= MambaSynth.clap)
tests/        → 5 test executables + 1 benchmark
```

- **Compiler:** GCC 13+ or Clang 17+, C++20
- **Build system:** CMake 3.20+
- **Test framework:** GoogleTest (Fetched via `FetchContent`, v1.14.0)
- **Release flags:** `-O3 -fno-math-errno -fno-trapping-math -fno-reciprocal-math -fno-signaling-nans -fcx-limited-range -fno-exceptions -fno-rtti -fvisibility=hidden`
- **Python tools:** PyTorch, NumPy (for export/validate scripts)

---

## 14. Denormal Handling

Audio thread calls `SSMBlock::apply_ftz_daz()` at the start of `plugin_process()`:

```cpp
// x86/x64: set FTZ (bit 15) and DAZ (bit 6) in MXCSR
uint32_t mxcsr = _mm_getcsr();
mxcsr |= 0x8040;  // FTZ | DAZ
_mm_setcsr(mxcsr);
```

- FTZ (Flush to Zero): Subnormal results of arithmetic → zero
- DAZ (Denormals Are Zero): Subnormal inputs → zero
- ARM (`__aarch64__`) path: not yet implemented (blocked on ARM hardware)

---

## 15. Seqlock Memory Ordering

```
WRITER:                      READER:
write_lock()                 read_begin()
  gen += 1                     snapshot = gen
  // atomic store, release     if snapshot % 2 != 0 → retry
                               memcpy data (acquire)
write_unlock()               read_retry(snapshot)
  gen += 1                     // atomic load, acquire
  // release                   return gen == snapshot
```

- `write_lock`: `memory_order_release` (pairs with reader's acquire)
- `write_unlock`: `memory_order_release`
- Reader uses `memory_order_acquire` for the generation load

This guarantees the reader either sees a fully written copy or retries — never a torn write.
