# Claude Prompt: Low-Level DSP & Voice Management for Mamba-SSM

You are an elite C++ Low-Level Systems and DSP Engineer specializing in real-time audio software, modern SIMD micro-architectures (AVX2, AVX-512, ARM NEON), lock-free concurrency, and plugin frameworks (CLAP, ARA 2.0).

Your task is to write the complete, production-ready C++ code to implement the missing core components of the **Mamba-SSM CLAP/ARA Plugin Suite** based on the architectural constraints and codebase states described below.

---

### I. ARCHITECTURAL CORE CONSTRAINTS

1. **Audio Thread Real-Time Safety**: Zero `malloc`/`free` (no heap allocation), zero `mutex`/blocking synchronization, zero blocking I/O, zero throwing of exceptions, and zero RTTI. All workspace memory must be pre-allocated.
2. **Three-Thread Boundary Rules**:
   - **Audio Thread**: High-priority. Runs `plugin_process()`. Reads parameters via `SPSCRingBuffer<ParamAtom, 256>`. Reads conditioning frames via `Seqlocked<ConditioningFrame>`. Does SSM inference.
   - **Main Thread**: GUI/Host thread. Pushes parameter changes to the ring buffer. Manages loading of weight binary files via memory-mapped I/O (`WeightLoader`).
   - **ARA Thread**: Musical context, analyzes timeline, writes `ARAConditioningFrame` via Seqlock.
3. **Cache Alignment**: Strict `alignas(64)` on all states, parameters, and workspace scratchpads to avoid false sharing and maximize L1/L2 cache hit rates.
4. **Denormals**: Flush-To-Zero (FTZ) and Denormals-Are-Zero (DAZ) set in the MXCSR register at the boundary of `process()`.
5. **Dimensions**: Static limits are hardcoded in `src/config.h`:
   `MAX_D_STATE=16`, `MAX_D_MODEL=16`, `MAX_D_INNER=32`, `MAX_D_CONV=4`, `MAX_LAYERS=2` (or 4), `COND_DIM=16`.
   Every multidimensional array layout MUST strictly use these constants for allocation bounds to remain compatible with `SSMWeights` flat weight offsets.

---

### II. FIXED COMPLETED COMPONENTS & PATTERNS

We have successfully implemented, debugged, and verified the following:
1. **SSM Block State-Space Math Bugfix**:
   The Selective State-Space discretization has been verified. The ZOH-discretization formulation is:
   $$A = -\exp(\text{A\_log})$$
   $$\Delta = \text{softplus}(\text{dt\_proj\_w} \times \text{conv\_out} + \text{dt\_bias} + \text{cond\_proj\_dt} \times \text{cond})$$
   $$A_{bar} = \exp(A \times \Delta)$$
   $$B_{bar} = \begin{cases} \frac{A_{bar} - 1}{A} \times B & \text{if } |A| > 10^{-6} \\ \Delta \times B & \text{otherwise} \end{cases}$$
   where:
   $$B = \text{B\_proj\_w} \times \text{conv\_out} + \text{cond\_proj\_B} \times \text{cond}$$
   $$h_t[i] = A_{bar}[i] \times h_{t-1}[i] + B_{bar}[i] \times \text{conv\_out}[i \pmod{d_{inner}}]$$
   $$y[i] = C[i] \times h_t[i]$$
   $$y_{out} = W_{ssm} \times ss + W_{out} \times \text{conv\_out}$$
   
   *CRITICAL FIX applied*: The previous code was double-counting $B$ by evaluating `Bx = B_[i] * conv_out_` during discretization and then multiplying again. The corrected code avoids this and maps `conv_out_[i % d_inner_]` cleanly.

2. **RT-Safe Weight Loading**:
   No blocking file operations or memory moves can happen inside the note-triggered handler on the audio thread. Weights are loaded entirely on the main/GUI thread using `WeightLoader::load()`, parsing headers, and calling a thread-safe helper `MambaSynth::load_weights()` which copies pointers to the active workspace memory using a `std::atomic<bool> weights_loaded` handshake.

3. **Inference Core Placement-New Initialization**:
   `SSMInferenceCore` has been refactored to support two-phase initialization (default constructor + explicit `.init(config)` method). The `MambaSynth` struct allocates a static byte array:
   `alignas(64) uint8_t core_storage[sizeof(SSMInferenceCore) * MAX_LAYERS]`
   and placement-constructs the inference core dynamically inside `activate()` using the host-provided sample rate, eliminating heap-allocation inside the plugin instance lifecycles.

4. **Conditioning Projection Weights**:
   The weight layout (V1 binary format) and `SSMWeights` struct now contain explicit learned projection weights to map conditioning inputs (16-dim) to $\Delta$ (dt) and $B$:
   - `float cond_proj_dt[MAX_D_STATE][COND_DIM]` (Offset 13)
   - `float cond_proj_B[MAX_D_STATE][COND_DIM]` (Offset 14)
   
   *Weights Binary File Order*: `A_log` (ds), `D_proj` (ds), `dt_bias` (ds), `dt_proj_w` (ds*di), `B_proj_w` (di*ds), `C_proj_w` (di*ds), `conv1d_w` (dc*di), `conv1d_b` (di), `W_in` (di*dm), `W_out` (dm*di), `W_ssm` (dm*ds), `b_in` (di), `cond_proj_dt` (ds*cd), `cond_proj_B` (ds*cd).

---

### III. CODEBASE REFERENCE HEADERS

#### 1. `src/core/ssm_block.h`
```cpp
#ifndef SSM_BLOCK_H
#define SSM_BLOCK_H
#include "config.h"
#include <cstddef>
#include <cstdint>
#include <array>

namespace mamba {
struct alignas(64) SSMState {
    float h[MAX_D_STATE];
    float conv_state[MAX_D_CONV * MAX_D_INNER];
};

struct SSMWeights {
    float A_log[MAX_D_STATE];
    float D_proj[MAX_D_STATE];
    float dt_bias[MAX_D_STATE];
    float dt_proj_w[MAX_D_STATE][MAX_D_INNER];
    float B_proj_w[MAX_D_INNER][MAX_D_STATE];
    float C_proj_w[MAX_D_INNER][MAX_D_STATE];
    float conv1d_w[MAX_D_CONV][MAX_D_INNER];
    float conv1d_b[MAX_D_INNER];
    float W_in[MAX_D_INNER][MAX_D_MODEL];
    float W_out[MAX_D_MODEL][MAX_D_INNER];
    float W_ssm[MAX_D_MODEL][MAX_D_STATE];
    float b_in[MAX_D_INNER];
    float cond_proj_dt[MAX_D_STATE][COND_DIM];
    float cond_proj_B[MAX_D_STATE][COND_DIM];
};

struct SSMBlockParams {
    float gain;
    float mix;
};

class SSMBlock {
public:
    SSMBlock(size_t d_state, size_t d_model, size_t d_inner, size_t d_conv);
    void reset();
    float process_sample(float input);
    void process_block(const float* input, float* output, size_t size);
    void set_weights(const SSMWeights& w);
    void set_params(const SSMBlockParams& params);
    void set_conditioning(const float* cond_vector);
    static void apply_ftz_daz();
    
    size_t get_d_state() const { return d_state_; }
    size_t get_d_model() const { return d_model_; }
    const SSMState& state() const { return state_; }
    SSMState& state() { return state_; }
private:
    size_t d_state_;
    size_t d_model_;
    size_t d_inner_;
    size_t d_conv_;
    SSMState state_;
    SSMWeights weights_;
    SSMBlockParams params_;
    alignas(64) std::array<float, MAX_D_INNER> x_inner_;
    alignas(64) std::array<float, MAX_D_INNER> conv_out_;
    alignas(64) std::array<float, MAX_D_STATE> dt_;
    alignas(64) std::array<float, MAX_D_STATE> B_;
    alignas(64) std::array<float, MAX_D_STATE> C_;
    alignas(64) std::array<float, MAX_D_STATE> ss_;
    alignas(64) std::array<float, MAX_D_MODEL> y_out_;
    size_t conv_pos_;
    alignas(64) std::array<float, COND_DIM> cond_;
    bool conditioning_active_;
    bool weights_loaded_;
    static constexpr uint32_t STATE_NORM_INTERVAL = 4096;
    uint32_t sample_count_;
};
}
#endif
```

#### 2. `src/core/ssm_inference_core.h`
```cpp
#ifndef SSM_INFERENCE_CORE_H
#define SSM_INFERENCE_CORE_H
#include "ssm_block.h"

namespace mamba {
struct MambaConfig {
    uint32_t d_state;
    uint32_t d_model;
    uint32_t d_inner;
    uint32_t d_conv;
    uint32_t num_layers;
    uint32_t cond_dim;
};

class SSMInferenceCore {
public:
    SSMInferenceCore();
    explicit SSMInferenceCore(const MambaConfig& cfg);
    ~SSMInferenceCore();
    void init(const MambaConfig& cfg);
    void reset();
    float process_sample(float input);
    void process_block(const float* input, float* output, size_t size);
    void set_layer_weights(uint32_t layer_idx, const SSMWeights& weights);
    void set_conditioning(const float* cond_vector);
    uint32_t get_num_layers() const { return num_layers_; }
private:
    MambaConfig config_;
    uint32_t num_layers_;
    SSMBlock* layers_[MAX_LAYERS];
    alignas(64) uint8_t block_memory_[sizeof(SSMBlock) * MAX_LAYERS];
};
}
#endif
```

#### 3. `src/clap/clap_plugin.h` (Abbreviated Workspace Layout)
```cpp
namespace mamba {
struct MambaSynth {
    clap_plugin_t plugin;
    const clap_host_t* host;
    // ...
    // PRE-ALLOCATED ARENA CORE STORAGE
    alignas(64) uint8_t core_storage[sizeof(SSMInferenceCore)];
    SSMInferenceCore* core;
    std::atomic<bool> weights_loaded{false};
    // ...
    bool load_weights(const char* path);
};
}
```

---

### IV. UNIMPLEMENTED COMPONENTS SPECIFICATIONS & TASKS

Write the complete implementations for the following targets.

#### 1. AVX2 Vectorized `process_block` for `SSMBlock`
Currently, `SSMBlock::process_block()` loops over `process_sample()`. You must replace this with a highly optimized, AVX2-intrinsics vectorized block path.
- **Requirements**:
  - Process samples in chunks of 8 floats using Intel AVX2 intrinsics (`__m256`, `_mm256_load_ps`, `_mm256_fmadd_ps`, etc.).
  - Vectorize the causal conv1d buffering window using vectorized ring-shifts or vectorized copying.
  - Implement a SIMD vectorization of the linear layers: $W_{in} \cdot x$, $\text{dt\_proj\_w} \cdot \text{conv\_out}$, $B_{proj\_w} \cdot \text{conv\_out}$, $C_{proj\_w} \cdot \text{conv\_out}$. Since `MAX_D_INNER` is 32 (multiple of 8) and `MAX_D_STATE` is 16 (multiple of 8), use unrolled 256-bit operations with 8-way parallel execution.
  - Compute the selective scan in parallel across state dimensions. Because states are updated elementwise ($h_i[t] = \bar{A}_i \cdot h_i[t-1] + \bar{B}_i \cdot u$), you can vectorize the parallel state update loop using 256-bit registers (processing 8 state dimensions at once).
  - Handle block sizes that are not multiples of 8 gracefully with a clean scalar fallback (reusing `process_sample()`).
  - Strict cash alignment (`alignas(64)`) on all local vector buffers inside the block process.

#### 2. Polyphonic Voice Manager (`MambaVoice` / `MambaVoiceManager`)
We need voice allocation and polyphony state tracking since the plugin acts as a synthesizer (`MambaSynth`), but currently lacks an explicit polyphonic framework.
- **Requirements**:
  - Implement a `MambaVoice` class containing its own `SSMInferenceCore` (constructed safely using placement-new on pre-allocated voice buffers) and ADSR envelope states.
  - Implement a thread-safe, lock-free `MambaVoiceManager` supporting up to 16/32-voice polyphony (bounded by a static compile-time value).
  - Voice Manager must handle:
    - Lock-free note-on/note-off scheduling.
    - Active voices tracking via a lock-free or atomic bitmask (`std::atomic<uint64_t> active_voices_mask`).
    - Note-stealing algorithm prioritizing killing the quietest voice (lowest ADSR value) or oldest voice if all voices are occupied.
    - Parameter smoothing per voice via simple lock-free exponential smoothing/one-pole filters.
  - ADSR Envelope: Implement a real-time safe linear-in-dB or exponential envelope generator running inside the voice processing loop to modulate output gains, preventing pops on note boundaries.

#### 3. Production Celemony ARA 2.0 SDK Bindings
`src/ara/` contains purely stub files and mock data structures. Replace them with proper, compile-ready classes mapped to the `ARA::PlugIn` API hierarchy.
- **Requirements**:
  - Map out the relationships between:
    - `ara_document_controller`: receives changes on host timeline states and regions.
    - `ara_playback_renderer`: processes host timeline contents and retrieves the backing audio file data.
    - `ara_editor_renderer`: manages visualization states.
  - Implement `get_ara_factory` exporting standard ARA factory descriptors.
  - Implement lock-free timeline positioning and thread-safe parsing of musical tempo/time signature context, then marshalling this data as conditioning vector packets using the `ARAConditioningFrame` structure.
  - Provide CMake configuration guidelines (via modern target-based syntax) indicating how to detect and link against the official Celemony ARA SDK using an external `ARA_SDK_PATH` environment parameter.

#### 4. Thread-Safe Parameter Interface and CLAP GUI Integration Shell
Integrate a real thread-safe parameter update strategy between CLAP host bindings and the core engine.
- **Requirements**:
  - Implement the thread safety barrier between the CLAP GUI main loop and the processing thread. Show the atomic parameter cache mechanics.
  - Implement the parameter queuing model: as parameters are modulated from the GUI/Host, they must be converted to parameter atom events, written to the `SPSCRingBuffer`, and drained on the audio thread before rendering.
  - Outline the registration skeleton for `clap_ext_gui` so the host can negotiate window handlers across operating systems (Win32/Cocoa/X11) safely.

---

### VI. DIRECTIVES FOR THE CODE DELIVERY

1. Provide **full, complete code listings** for each missing class or module. Do not omit function bodies, write "TODO" comments, or use pseudo-code.
2. Ensure variable naming conventions exactly match the existing codebase.
3. Every single optimization, voice allocation step, or pointer extraction must remain memory-safe and respect CPU cache alignment lines. Include inline comments explaining complex AVX2 register packing, lock-free ordering choices (`memory_order_release` / `memory_order_acquire`), or ARA SDK lifecycles.
