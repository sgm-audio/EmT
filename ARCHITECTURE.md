# Architecture Overview

**EmT CLAP/ARA Plugin Suite** — Architecture Version: Mamba-3 (`v0.3.0`)  
**Targets:** `EmTSynth.clap` (CLAP Bundle exporting `com.emt.synth` and `com.emt.fx`)

---

## 1. Executive Summary & System Architecture

The EmT Plugin Suite is a high-performance, real-time neural audio synthesizer and effect processor built on the **Mamba-3** Structured State Space Model (SSM) architecture. It is engineered for low-latency digital audio workstations (DAWs) supporting the CLAP (CLever Audio Plug-in) 1.2+ specification and Celemony ARA 2.0.

Key architectural innovations include:
- **Zero-Conv Architecture:** Total elimination of causal 1D convolutions, achieving true **0-sample algorithmic latency**.
- **Exponential-Trapezoidal Discretization:** 2nd-order bilinear discretization preserving high-frequency response and guaranteeing strict numerical stability.
- **Complex Multi-Head Resonant Poles:** Multi-head state representation with Rotary Position Embedding (RoPE) creating frequency-selective resonant bands per head.
- **AVX2 / FMA Vector Acceleration:** Hardware SIMD primitives optimizing projection dot products and state updates without dynamic heap allocation.
- **Zero-Copy Multi-Voice State Handling:** Polyphonic voice pool (`MAX_VOICES = 8`) sharing a single inference engine via in-place state reference passing.
- **Strict 3-Thread Concurrency Boundary:** Lock-free, wait-free isolation between real-time Audio, host Main/UI, and offline ARA worker threads.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                       HOST DAW                                         │
│  ┌──────────────────────┐    ┌──────────────────────┐    ┌──────────────────────────┐  │
│  │     Audio Thread     │    │     Main Thread      │    │        ARA Thread        │  │
│  │   Real-Time Audio    │    │     UI & Presets     │    │   Musical Intelligence   │  │
│  │ • plugin_process()   │    │ • params_flush()     │    │ • DocumentController     │  │
│  │ • SSM recurrence     │    │ • GUI event loop     │    │ • Tempo & key tracking   │  │
│  │ • Voice envelopes    │    │ • Weight mmap loading│    │ • f0 & chord extraction  │  │
│  │ • 1-pole smoothing   │    │ • Preset save/load   │    │ • Offline render         │  │
│  └──────────┬───────────┘    └──────────┬───────────┘    └────────────┬─────────────┘  │
└─────────────┼───────────────────────────┼─────────────────────────────┼────────────────┘
              │                           │                             │
              ▼                           ▼                             ▼
      ┌───────────────┐           ┌───────────────┐             ┌───────────────┐
      │ SPSC Ring Buf │           │ Atomic Params │             │   Seqlocked   │
      │<ParamAtom,256>│           │   (double)    │             │   ARA Frame   │
      │ lock-free pop │           │ lock-free R/W │             │ release-store │
      └───────▲───────┘           └───────▲───────┘             └───────▲───────┘
              │                           │                             │
              └───────────────────────────┼─────────────────────────────┘
                                          │
                                          ▼
                      ┌───────────────────────────────────────┐
                      │               EmTSynth                │
                      │ ┌───────────────────────────────────┐ │
                      │ │         SSMInferenceCore          │ │
                      │ │   Layer 0 ──► Layer 1 ──► ...     │ │
                      │ │   (8 heads, RoPE, RMSNorm, FiLM)  │ │
                      │ └───────────────────────────────────┘ │
                      │ ┌───────────────────────────────────┐ │
                      │ │ Voice Pool (8 Voices, Zero-Copy)  │ │
                      │ └───────────────────────────────────┘ │
                      │ ┌───────────────────────────────────┐ │
                      │ │ DoubleBufferSeqLock<StateMirror>  │ │
                      │ └───────────────────────────────────┘ │
                      └───────────────────────────────────────┘
```

---

## 2. The 3-Thread Concurrency Model

Audio plugins must execute within strict real-time deadlines (typically \(\le 1.33\text{ ms}\) for a 64-sample buffer at \(48\text{ kHz}\)). Per **ADR-001**, the suite enforces strict thread boundaries across three asynchronous execution domains:

| Boundary | Thread | Priority | Role & Operations | Invariants & Constraints |
|---|---|---|---|---|
| **Domain 1** | **Audio Thread** | Real-Time / Priority Inversion Safe | `plugin_process()`, `fx_process()`, voice processing, SSM recurrence, one-pole smoothing, crossfading. | **Zero** `malloc`/`free`/`new`/`delete`<br>**Zero** mutexes / condition variables<br>**Zero** blocking I/O or sleep<br>**Zero** unbounded spinning |
| **Domain 2** | **Main Thread** | Standard User Priority | Host callbacks, UI rendering (`clap_gui`), preset save/load (`.mambapreset`), parameter flushes (`params_flush`), weight loading (`WeightLoader::load` via `mmap`). | Owns GUI handles, performs file I/O, decodes presets, parses human text. |
| **Domain 3** | **ARA Thread** | Background Worker Priority | Celemony ARA 2.0 Document Controller analysis, audio timeline scanning, tempo/chord detection, offline neural rendering. | Computes musical context asynchronously. Never interrupts the audio thread. |

### 2.1 Lock-Free Communication Matrix

Communication between threads uses dedicated lock-free primitives matching the producer-consumer topology:

```
Producer ───► Consumer      Primitive                               Payload
────────────────────────────────────────────────────────────────────────────────────────────
Main     ───► Audio         SPSCRingBuffer<ParamAtom, 256>          Sample-accurate param events
Main     ───► Audio         std::atomic<double> param_values[]      Instantaneous automation targets
ARA/Main ───► Audio         Seqlocked<ConditioningFrame>            128-float FiLM context vector
Audio    ───► Main          DoubleBufferSeqLock<StateMirror>        Complete voice & plugin state
Audio    ───► Main          VoiceAtomics voice_atomics[8]           UI metering (envelope & stage)
```

---

## 3. Lock-Free Synchronization Primitives

### 3.1 SPSCRingBuffer\<ParamAtom, 256\>

The single-producer single-consumer ring buffer (`src/common/ring_buffer.h`) transfers parameter automation atoms from the host's `params_flush()` to the real-time audio thread:

```cpp
struct ParamAtom {
    clap_id id;
    double value;
    uint32_t frame_offset;  // Sample-accurate offset within block
};
```

- **Power-of-Two Capacity:** Masking (`(head + 1) & (Capacity - 1)`) avoids division or conditional branches.
- **Cache Isolation:** `head_` and `tail_` atomics are aligned to 64 bytes (`alignas(64)`) to eliminate false sharing between CPU cores.
- **Memory Ordering:** `push()` uses relaxed load on head, acquire load on tail, and release store on head. `pop()` mirrors with release store on tail.
- **Back-Pressure Policy:** If the ring buffer fills (exceeds 256 pending automation events), excess events are dropped to guarantee that the producer never blocks the caller.

### 3.2 Seqlocked\<T\>

The sequential lock (`src/common/seqlock.h`) allows lock-free exchange of the 128-float conditioning frame (`ConditioningFrame`) from ARA/Main to Audio:

- **Writer (ARA/Main):** Increments sequence counter to an odd value (`memory_order_acq_rel`), writes plain data into the frame, and increments sequence counter to an even value (`memory_order_acq_rel`).
- **Reader (Audio):** Reads start sequence counter (`memory_order_acquire`). If odd (write in progress), it spins briefly. Copies data, then verifies the sequence counter has not changed (`read_retry`).
- **Bounded Reader Spin:** Because ARA updates occur at musical event boundaries (\(\sim 10\text{ Hz}\)), write collisions with the audio thread are statistically rare (\(< 0.001\%\)) and resolve within nanoseconds.

### 3.3 DoubleBufferSeqLock\<StateMirror\>

State serialization (DAW project saving and preset export) requires copying the entire plugin state—including 8 voice neural state snapshots (\(\sim 1.1\text{ MB}\)). A simple seqlock would cause the audio thread to stall or tear if the main thread took milliseconds to write to disk.

Per **ADR-001**, `DoubleBufferSeqLock<StateMirror>` (`src/common/double_buffer_seqlock.h`) implements wait-free double buffering:

```
Audio Thread (Writer):
  1. slot = begin_write()       // Writes to back slot (slots_[writer_ ^ 1])
  2. Copy voice snapshots & params
  3. end_write()                // Publishes to front_: front_.store(cur, release)
  4. swap()                     // Flips writer_ index atomically

Main Thread (Reader):
  1. g1 = generation_.load(acquire)
  2. f = front_.load(acquire)   // Stable front slot
  3. memcpy(out, &slots_[f], size)
  4. g2 = generation_.load(acquire)
  5. return g1 == g2            // Validation check
```

The audio thread is **100% wait-free**: it never spins, never blocks, and writes exclusively to its dedicated back buffer. The main thread reads the front buffer with optimistic retry.

---

## 4. Mamba-3 Zero-Conv Architecture

### 4.1 Elimination of Causal Conv1d (Zero Algorithmic Latency)

Prior SSM architectures (Mamba-1 / Mamba-2) inserted a causal 1D depthwise convolution (\(d\_conv = 4\)) before the selective scan to mix adjacent time-steps. This introduced:
1. An unavoidable algorithmic latency of \(d\_conv - 1 = 3\text{ samples}\) (\(\approx 68\ \mu\text{s}\) at \(44.1\text{ kHz}\)).
2. Significant state overhead (`conv_state[d_conv * d_inner]`) that had to be shifted and updated per sample.
3. Cache thrashing during multi-voice state save and restore.

**Mamba-3 eliminates causal conv1d entirely.** 
By expanding the input projection and introducing multi-head complex resonant poles, Mamba-3 captures local temporal dependencies directly in the continuous state recurrence. The plugin achieves **0 samples algorithmic latency** (`CLAP_EXT_LATENCY` returns `0`), making it suitable for live monitoring and real-time instrumental performance.

### 4.2 Exponential-Trapezoidal Discretization

Mamba-1 used Zero-Order Hold (ZOH) discretization:
$$\bar{A} = \exp(\Delta \cdot A), \quad \bar{B} = \frac{\exp(\Delta \cdot A) - 1}{A} \cdot B$$
ZOH suffers from high-frequency phase warping and requires branching to avoid division by zero near \(A \to 0\).

Mamba-3 replaces ZOH with **second-order exponential-trapezoidal discretization** (bilinear Cayley transform):

$$\bar{A} = \frac{1 + \frac{1}{2} \Delta A}{1 - \frac{1}{2} \Delta A}, \quad \bar{B}_{\text{scale}} = \frac{\Delta}{1 - \frac{1}{2} \Delta A}$$

#### Mathematical Guarantees:
1. **Unconditional Stability:** For any negative decay parameter \(A < 0\) and positive step size \(\Delta > 0\):
   $$1 - \frac{1}{2} \Delta A > 1 \implies |\bar{A}| = \left|\frac{1 - \frac{1}{2}\Delta |A|}{1 + \frac{1}{2}\Delta |A|}\right| < 1.0$$
   The poles are strictly guaranteed to lie inside the unit circle on the \(\mathcal{Z}\)-plane.
2. **Singularity-Free:** The denominator \(1 - \frac{1}{2}\Delta A\) is strictly positive and bounded away from zero. No conditional branching or \(\epsilon\)-thresholding is needed.
3. **Frequency Response Fidelity:** Preserves the analog continuous-time filter frequency response up to the Nyquist limit with natural bilinear warping.

C++ Implementation in `SSMBlock::process_sample()`:
```cpp
float half_a = A_decay * dt * 0.5f;
float inv_denom = 1.0f / (1.0f - half_a); // Always > 0 for A_decay < 0
float A_bar = (1.0f + half_a) * inv_denom;
float B_bar_scale = dt * inv_denom;
```

### 4.3 Multi-Head Complex Resonant State & RoPE

Each Mamba-3 layer decomposes the inner dimension into \(H\) independent heads (\(H \le 8\), `headdim` \(\le 64\)):

```
Input u (d_model)
   │
   ▼
[in_proj_w] ──► [ z, x, B, C, dd_dt, dd_A, trap, angles ]
                   │  │  │  │    │      │
                   │  │  │  │    │      └─► A_t = -softplus(dd_A) (per-head decay)
                   │  │  │  │    └────────► DT_t = softplus(dd_dt + dt_bias) (per-head step)
                   │  │  │  └─────────────► RMSNorm + RoPE(angles) ──► C_norm
                   │  │  └────────────────► RMSNorm + RoPE(angles) ──► B_norm
                   │  └───────────────────► x_hd [nheads, headdim]
                   └──────────────────────► z_silu [nheads, headdim]
```

- **Complex State Representation:** State is decomposed into real and imaginary matrices `ssm_real[H][P][N]` and `ssm_imag[H][P][N]`.
- **Rotary Position Embedding (RoPE):** Rotary angles are applied to pairs of state dimensions \((s, s + N/2)\):
  $$\begin{pmatrix} B'_0 \\ B'_1 \end{pmatrix} = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix} \begin{pmatrix} B_0 \\ B_1 \end{pmatrix}$$
  $$\begin{pmatrix} C'_0 \\ C'_1 \end{pmatrix} = \begin{pmatrix} \cos\theta & \sin\theta \\ -\sin\theta & \cos\theta \end{pmatrix} \begin{pmatrix} C_0 \\ C_1 \end{pmatrix}$$
  RoPE modulates the phase of the state recurrence on every sample, shifting the complex pole angles. Each head behaves as a resonant, time-varying bandpass filter with independent center frequency and bandwidth.
- **RMSNorm on B and C:** Root-Mean-Square normalization is applied along the state dimension \(N\) before RoPE rotation:
  $$\text{RMS}(B) = \sqrt{\frac{1}{N} \sum_{s=0}^{N-1} B_s^2 + 10^{-5}}, \quad B_{\text{norm}, s} = \frac{B_s}{\text{RMS}(B)} \cdot W_{\text{norm}, s} + \text{bias}_s$$
  This prevents gradient explosion during training and prevents state accumulation overflow during runtime inference.

### 4.4 FiLM Conditioning (Feature-wise Linear Modulation)

Conditioning data originates from musical analysis (tempo, key, chord) and neural analysis (CREPE pitch probability, semantic timbre embeddings):

$$\text{Conditioning Vector } \mathbf{c} \in \mathbb{R}^{128}$$
- \(\mathbf{c}[0..15]\): Musical context (tempo BPM, key signature, chord one-hot).
- \(\mathbf{c}[16..47]\): CREPE pitch probability distribution (32 bins).
- \(\mathbf{c}[48..127]\): Latent semantic timbre embedding (80 dimensions).

FiLM modulates the input vector \(\mathbf{u}\) and head step sizes \(\Delta\):
$$\gamma_j = \mathbf{W}_{\text{film}}[2j] \cdot \mathbf{c}, \quad \beta_j = \mathbf{W}_{\text{film}}[2j+1] \cdot \mathbf{c}$$
$$u_{\text{film}, j} = u_j \cdot (1.0 + \gamma_j) + \beta_j$$

### 4.5 MIMO Coupling & Output Gating

- **MIMO Coupling:** B and C matrices have rank \(R \le 4\), projecting between the multi-dimensional input/output and state space.
- **SiLU Gating on z Branch:** The \(z\) projection branch undergoes Sigmoid Linear Unit activation:
  $$z_{\text{silu}} = z \cdot \sigma(z) = \frac{z}{1 + e^{-z}}$$
  The head output \(y\) is multiplicatively gated: \(y_{\text{gated}} = y \cdot z_{\text{silu}}\).
- **Periodic State Normalization:** Every \(4096\text{ samples}\) (`SSM_STATE_NORM_INTERVAL`), the audio thread scans the state vector. If \(\max |h| > 1.0\), all state elements are scaled down proportionally. This guarantees that 24/7 continuous audio processing never accumulates numerical creep or drift.

---

## 5. AVX2 SIMD Optimizations & Low-Level DSP

The audio engine targets modern x86_64 architectures with AVX2 and FMA3 instruction sets, with transparent fallbacks to SSE2 and ARM NEON.

### 5.1 Compiler Configuration

Build configuration in `CMakeLists.txt`:
- **MSVC (Windows):** `/O2 /arch:AVX2 /DNDEBUG` with `/STACK:16777216` (16 MB stack allocation for large static weight structs).
- **GCC / Clang (Linux/macOS):** `-O3 -mavx2 -mfma -fno-math-errno -fno-trapping-math -fno-reciprocal-math -fno-signaling-nans -fcx-limited-range -fno-exceptions -fno-rtti -fvisibility=hidden`.

### 5.2 SIMD Helper Kernels (`src/core/simd_helpers.h`)

All heavy linear algebra operations in `SSMBlock` utilize hand-tuned SIMD intrinsics:

1. **Vector Dot Product (`simd::dot_product`):**
   Computes \(\sum_{i=0}^{n-1} a_i \cdot b_i\) unrolled in 4-wide / 8-wide blocks:
   ```cpp
   __m128 acc = _mm_setzero_ps();
   for (; i + 4 <= n; i += 4) {
       __m128 va = _mm_loadu_ps(a + i);
       __m128 vb = _mm_loadu_ps(b + i);
       acc = _mm_add_ps(acc, _mm_mul_ps(va, vb));
   }
   // Horizontal add reduction
   ```
   Accelerates the massive `in_proj_w` projection (\(3112 \times 256\) multiply-accumulates per sample) and `out_proj_w`.

2. **Vector Scalar Multiply-Accumulate (`simd::fmadd_scalar`):**
   Computes \(\mathbf{out} += \mathbf{a} \cdot \text{scalar}\) using broadcast loads (`_mm_set1_ps` / `vdupq_n_f32`).

3. **Element-wise Scaling & FMA (`simd::scale`, `simd::fmadd_elementwise`):**
   Operates across contiguous memory blocks for output scaling and residual mixing.

### 5.3 Denormal Prevention (FTZ / DAZ)

Subnormal (denormal) floating-point numbers can cause CPU execution units to fall back to microcode traps, increasing calculation time by up to 100x.

The engine applies hardware Flush-to-Zero (FTZ) and Denormals-Are-Zero (DAZ) at the start of **every** `plugin_process()` and `fx_process()` call via `SSMBlock::apply_ftz_daz()`:
- **x86 / x86_64:** Sets bits 15 (FTZ) and 6 (DAZ) in the MXCSR control register (`_mm_getcsr() | 0x8040`).
- **AArch64 (ARM):** Sets bit 24 (FZ) in the Floating-point Control Register (`FPCR`).

### 5.4 64-Byte Cache Alignment

All critical DSP structures enforce 64-byte alignment (`alignas(64)`), exactly matching the L1 cache line size on modern Intel, AMD, and Apple Silicon processors:
- `SSMState`, `MambaState`, `SSMWeights`
- Audio scratch buffers (`u_film_`, `in_proj_out_`, `rope_buf_`, `y_head_`, `out_`)
- Inter-thread queues (`SPSCRingBuffer`, `Seqlocked`, `DoubleBufferSeqLock`)

Alignment guarantees:
1. Zero cache-line splits on 256-bit AVX vector loads/stores.
2. Complete elimination of false sharing between cores executing audio vs main thread code.

---

## 6. Zero-Copy Polyphonic Voice Management

### 6.1 Multi-Voice Architecture (`MAX_VOICES = 8`)

`MambaSynth` supports 8-voice polyphony without instantiating 8 separate neural network engines. A single `SSMInferenceCore` is shared across all voices:

```cpp
struct alignas(64) Voice {
    MambaState ssm_state;       // Independent multi-layer neural state
    float env;                  // Current envelope amplitude [0.0, 1.0]
    float attack_rate;          // Per-sample attack increment
    float release_rate;         // Per-sample release decrement
    uint8_t stage;              // VOICE_IDLE, ATTACK, SUSTAIN, RELEASE, SILENT
    float velocity;             // Note-on velocity [0.0, 1.0]
    int32_t key;                // MIDI key (0 - 127)
    int32_t channel;            // MIDI channel (0 - 15)
    uint64_t note_id;           // Monotonic ID for note-off matching
};
```

### 6.2 Zero-Copy Reference Passing

During each audio block, active voices are evaluated sequentially:
```cpp
for (uint32_t v = 0; v < MAX_VOICES; ++v) {
    Voice& voice = m->voices[v];
    if (voice.env <= 0.0f) continue;

    alignas(64) float outbuf[MAX_D_MODEL];
    m->core->process_sample(mono_input, outbuf, voice.ssm_state);
    voice_sum += outbuf[0] * voice.env * voice.velocity;
}
```
`process_sample(input, output, MambaState& state)` takes the voice's state struct **by reference**. No state data is copied, serialized, or allocated during audio rendering.

### 6.3 Voice Allocation & Stealing State Machine

When a `CLAP_EVENT_NOTE_ON` event arrives, `allocate_voice()` selects a voice slot using a 3-tier priority heuristic:
1. **Free Voice:** Slot in `VOICE_IDLE` or `VOICE_SILENT` stage.
2. **Oldest Released Voice:** Voice currently in `VOICE_RELEASE` with the lowest `note_id`.
3. **Oldest Sustained Voice:** Voice currently in `VOICE_SUSTAIN` with the lowest `note_id`.

```
        note_on
  IDLE ────────► ATTACK ──────► SUSTAIN ──────► RELEASE ──────► SILENT
   ▲               │               │               │               │
   │               │ env >= 1.0    │ note_off      │ env <= 0.0    │
   └───────────────┴───────────────┴───────────────┴───────────────┘
```

### 6.4 Lock-Free UI Telemetry (`VoiceAtomics`)

The GUI thread requires visual metering of active voices and envelope levels. Reading `m->voices[v]` directly would violate thread safety.

`MambaSynth` maintains a parallel array of atomic structs (`VoiceAtomics`):
```cpp
struct alignas(64) VoiceAtomics {
    std::atomic<uint8_t> stage{VOICE_IDLE};
    std::atomic<float> env{0.0f};
    std::atomic<int32_t> key{-1};
    std::atomic<int32_t> channel{-1};
    std::atomic<float> velocity{0.0f};
};
```
The audio thread updates these atomics with relaxed ordering. The GUI thread polls them at 30–60 Hz for smooth animation without stalling the DSP pipeline.

---

## 7. Weight File Architecture & Memory Mapping

Weights are stored in flat, cache-aligned binary files (`models/*.bin`) adhering to **Weight Format Specification v2** (`WEIGHT_FORMAT_SPEC.md`).

### 7.1 64-Byte Header

```
Offset  Size  Field              Description
0x00    4     magic              0x4D424D41 ("MBMA")
0x04    4     version            2 (Mamba-3)
0x08    4     d_state            State dimension (N <= 32)
0x0C    4     d_model            Model dimension (D <= 256)
0x10    4     d_inner            Inner dimension (E <= 512)
0x14    4     nheads             Number of heads (H <= 8)
0x18    4     num_layers         Layers (L <= 6)
0x1C    4     headdim            Head dimension (P <= 64)
0x20    4     cond_dim           Conditioning dimension (COND_DIM = 128)
0x24    4     quantization       0=FP32, 1=Int8 Tensor, 2=Int8 Layer, 3=FP16
0x28    4     q_metadata_size    Size in bytes of quantization metadata
0x2C    4     flags              Reserved (0)
0x30    8     weight_data_size   Total weight payload size in bytes
0x38    8     reserved           Reserved (0)
```

### 7.2 Zero-Copy Memory Mapping

`WeightLoader::load()` maps the file directly into virtual address space:
- **POSIX:** `open()`, `mmap(..., PROT_READ, MAP_PRIVATE, fd, 0)`.
- **Windows:** `CreateFileA(..., GENERIC_READ)`, `CreateFileMappingA(..., PAGE_READONLY)`, `MapViewOfFile(..., FILE_MAP_READ)`.

The operating system pages weight data directly from disk into CPU cache on demand. No heap buffers are allocated for raw weights.

### 7.3 Deterministic Array Ordering (13 Arrays Per Layer)

Within each layer's block, weights are stored in 13 contiguous arrays:
1. `in_proj_w`: Fused projection `[in_proj_out, d_model]`
2. `dt_bias`: Per-head \(\Delta\) bias `[nheads]`
3. `B_bias`: Per-head, rank, state bias `[nheads, mimo_rank, d_state]`
4. `C_bias`: Per-head, rank, state bias `[nheads, mimo_rank, d_state]`
5. `B_norm_weight`: RMSNorm learned scale `[d_state]`
6. `C_norm_weight`: RMSNorm learned scale `[d_state]`
7. `mimo_x`: MIMO input coupling `[nheads, mimo_rank, headdim]`
8. `mimo_z`: MIMO gating coupling `[nheads, mimo_rank, headdim]`
9. `mimo_o`: MIMO output coupling `[nheads, mimo_rank, headdim]`
10. `D_proj`: Skip connection scalars `[nheads]`
11. `out_proj_w`: Output projection `[d_model, d_inner]`
12. `film_proj_w`: FiLM input modulation `[2 * d_model, cond_dim]`
13. `dt_film_proj_w`: FiLM step modulation `[2 * nheads, cond_dim]`

---

## 8. CLAP Extension Implementation Status

Both plugins (`com.mamba-ssm.synth` and `com.mamba-ssm.fx`) implement the CLAP 1.2 interface:

| CLAP Extension | Supported | Implementation Source | Description |
|---|---|---|---|
| `CLAP_EXT_AUDIO_PORTS` | **Yes** | `src/clap/clap_audio_ports.cpp` | Stereo I/O configuration (2 main channels). |
| `CLAP_EXT_NOTE_PORTS` | **Yes** (Synth) | `src/clap/clap_note_ports.cpp` | Single note input supporting CLAP and MIDI dialects. |
| `CLAP_EXT_PARAMS` | **Yes** | `src/clap/clap_params.cpp`, `clap_fx_plugin.cpp` | Parameter descriptors, text conversions, and lock-free flushing. |
| `CLAP_EXT_STATE` | **Yes** | `src/clap/clap_state.cpp` | Binary state save and restore for project persistence. |
| `CLAP_EXT_LATENCY` | **Yes** | `src/clap/clap_latency.cpp` | Reports `0 samples` (Zero-Conv architecture). |
| `CLAP_EXT_TAIL` | **Yes** | `src/clap/clap_tail.cpp` | Reports `100 ms` tail for SSM state dissipation. |
| `CLAP_EXT_THREAD_CHECK`| **Yes** | `src/clap/clap_thread_check.h` | Verifies execution context via host extension or fallback. |
| `CLAP_EXT_GUI` | **Partial** | `src/clap/clap_gui.cpp` | Native X11 UI implemented; macOS/Windows stubs. |
| `CLAP_EXT_TIMER_SUPPORT`| **Yes** | `src/clap/clap_plugin.cpp` | Timer support hook for periodic UI refresh. |

---

## 9. Verification & Test Suite

The codebase enforces strict correctness and real-time safety via automated GoogleTest executables (`tests/CMakeLists.txt`):

1. **`test_ssm_block.cpp` (9 tests):** Validates reset behavior, multi-head dimensions, RoPE phase wrapping, trapezoidal stability, and multi-layer ping-pong routing.
2. **`test_weight_loader.cpp` (20 tests):** Verifies header parsing, magic validation, dimension fit checks, FP16 dequantization, and INT8 dequantization.
3. **`test_conditioning.cpp` (4 tests):** Validates FiLM modulation, zero-input protection, and seqlock tear resistance.
4. **`test_stability.cpp` (3 tests):** Evaluates long-running audio generation (300 seconds continuous) to verify zero NaNs, no infinities, and no state explosion.
5. **`test_realtime_safety.cpp` (3 tests):** Real-time safety audit verifying zero allocations, zero mutex acquisitions, and zero blocking syscalls during `plugin_process()`.
6. **`test_cpp_parity.cpp` (1 test):** Validates bit-exact and numerical parity (\(\le 10^{-5}\) tolerance) between the C++ inference engine and reference Python Mamba-3 implementations.
