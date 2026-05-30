# Agent Task List: Mamba-SSM CLAP/ARA Plugin Suite

This file provides an extensive, thorough list of tasks for the next coding agent working on the Mamba-SSM CLAP/ARA plugin suite. It includes not only the implementation tasks but also critical non-obvious considerations that an AI agent might overlook when dealing with real-time audio systems, lock-free programming, and complex plugin architectures.

## Overview

The plugin suite consists of three plugins:
1. **Mamba Synth** – Generative audio from MIDI/ARA conditioning
2. **Mamba FX** – Long-context neural audio processing effect
3. **ARA Generative Companion** – Offline region generation via ARA renderers

All plugins share a common SSM inference core and must adhere to strict real-time constraints.

## Task List

### SSM Inference Core (High Priority)
- [ ] Implement `SSMInferenceCore` class with:
  - `initialize(int max_block_size)` – pre-allocates working memory
  - `reset()` – sets internal state to zero
  - `process_sample(float input, float& output)` – Mode A (tiny model, 4-16 sample blocks or per-sample)
  - `process_block(const float* input, float* output, int block_size)` – Mode B (block model, 64-512 sample blocks)
  - `set_layer_parameters(int layer, const float* delta, const float* B, const float* C)` – for conditioned SSM parameters
  - `set_conditioning_vector(const float* cond)` – receives conditioning vector from ARA thread
  - `get_state(float* state_buffer)` and `set_state(const float* state_buffer)` – for state snapshots (used by main thread with seqlock)
- [ ] Create arena allocator for pre-allocating all audio thread working memory (state vectors, temporary buffers) – no malloc/free in audio thread
- [ ] Implement numerical stabilization (periodic state normalization) to prevent drift in long runs
- [ ] Develop weight loader for flat-binary format (memory-mapped read-only) – weights never copied to writable memory after mmap
- [ ] Implement per-sample processing mode (optimize for latency)
- [ ] Implement block processing mode (optimize for throughput)
- [ ] Add parameter update interface for conditioned Δ, B, C matrices (called from main thread)
- [ ] Create unit tests for SSM block mathematical correctness against a reference NumPy/PyTorch implementation
- [ ] Benchmark SSM core performance – target <10μs per sample on a modern 2.5GHz+ CPU
- [ ] Verify that `process_sample` and `process_block` contain zero heap allocations and zero mutex locks

### CLAP/ARA Host Integration (High Priority)
- [ ] Implement CLAP plugin factory and entry points (`clap_entry`)
- [ ] Implement CLAP audio-ports extension (mono/stereo support, report latency and tail)
- [ ] Implement CLAP note-ports extension (handle note-on/note-off, translate to conditioning signals)
- [ ] Implement CLAP params extension with thread-safe smoothing (one-pole filter or linear interpolation)
- [ ] Implement CLAP gui extension (optional, for parameter visualization)
- [ ] Implement CLAP thread-check extension (to validate audio thread real-time safety)
- [ ] Implement CLAP state extension (for preset serialization, includes plugin params + weight file path + base64-encoded h_state)
- [ ] Implement CLAP tail extension (for FX plugins, compute and report tail time)
- [ ] Implement ARA 2.0 DocumentController interface (manage plugin instance lifecycle within ARA document)
- [ ] Implement ARA 2.0 PlaybackRenderer (offline rendering, e.g., bounce, render-in-place)
- [ ] Implement ARA 2.0 EditorRenderer (for GUI updates in ARA-enabled hosts like Studio One)
- [ ] Create ARA context-to-conditioning mapping: convert tempo (BPM), key signature, and chord one-hot (12-dim) into fixed-dim conditioning vector (`COND_DIM` floats)
- [ ] Implement learned projection system: conditioning vector → biases for SSM Δ and B per layer (small linear layer, precomputed or computed in ARA thread)
- [ ] Ensure all CLAP and ARA callbacks that can occur on the audio thread are lock-free and allocation-free (note events, param changes, etc.)

### ML Export & Weight Pipeline (High Priority)
- [ ] Develop PyTorch export script (`export_mamba_weights.py`):
  - Takes trained Mamba-SSM model
  - Applies quantization (INT8 or FP16) to weights
  - Packs weights and metadata into flat-binary format with deterministic ordering
  - Includes magic number, version, state dimension, number of layers, conditioning dimension, etc.
- [ ] Define flat-binary weight file format specification:
  - Header (cache-aligned): magic, version, STATE_DIM, NUM_LAYERS, COND_DIM, PARAM_COUNT, weight data size
  - Weight data: concatenated weights for each layer in a fixed order (e.g., input proj, recurrent proj, convolution, state update, output proj)
  - Quantization scales and zero-points stored per tensor (if affine quantization)
- [ ] Implement INT8/FP16 quantization in export pipeline – ensure SNR > 60dB for audio weights
- [ ] Create C++ weight loader:
  - Memory-map weight file read-only
  - Validate magic number and version
  - Extract metadata
  - Provide access to (dequantized) weights for each layer
  - Support both file-path and memory-buffer loading
- [ ] Build validation script (`validate_export.py`):
  - Load same weights in PyTorch and C++
  - Run identical random inputs through both
  - Compare outputs within quantization error tolerance
  - Verify that dequantization in C++ matches PyTorch's quantized→dequantized values
- [ ] Ensure export pipeline is deterministic: same PyTorch model + settings → identical weight binary

### UI/Parameter/State Management (High Priority)
- [ ] Design parameter system:
  - Support different parameter types (linear, logarithmic, exponential)
  - Smooth interpolation (one-pole low-pass or linear) to prevent zipper noise
  - Thread-safe updates via atomic float array or SPSC ring buffer (`ParamAtom` with sample-accurate `frameOffset`)
  - Mapping between normalized (0.0-1.0) and denormalized values
- [ ] Implement thread-safe parameter update mechanism:
  - Main thread pushes `ParamAtom` (id, value, frameOffset) to lock-free ring buffer
  - Audio thread pops atoms and applies smoothing/interpolation at the correct sample offset
- [ ] Create GUI abstraction layer (widgets: knobs, sliders, buttons, dropdowns) with CLAP gui extension support
- [ ] Implement state serialization/deserialization:
  - Serialize: plugin parameters (normalized values), absolute or relative path to weight file, base64-encoded h_state snapshot (concatenated state vectors)
  - Deserialize: restore parameters, locate weight file (search in preset directory or use absolute path), restore h_state
  - Use seqlock-protected double-buffer for h_state snapshots (audio thread writes to buffer A/B, main thread reads the non-writing buffer)
- [ ] Design preset file format (JSON recommended for readability, include versioning)
- [ ] Implement preset management: user preset save/load, factory preset bank, preset categorization/tagging
- [ ] Ensure GUI updates are offloaded from audio thread (use lock-free queues for parameter/state updates to UI thread if needed)

### Build System & Deployment (High Priority)
- [ ] Create cross-platform CMake build system:
  - Find and link against CLAP SDK and ARA 2.0 SDK
  - Set C++20 standard (`set(CMAKE_CXX_STANDARD 20)`)
  - Enable compiler flags for real-time safety:
    - `-fno-exceptions`
    - `-fno-rtti`
    - `-ffast-math` (use with caution, verify numeric stability)
    - `-march=native` or `-msse4.2 -mavx2` for SIMD
    - `-O3` (or `-O2` if `-ffast-math` causes issues)
    - `-fvisibility=hidden`
    - `-Wall -Wextra -Wpedantic`
  - Handle platform-specific settings:
    - macOS: universal builds (arm64+x86_64), set `CMAKE_OSX_ARCHITECTURES`
    - Windows: set runtime library to MultiThreadedDLL
    - Linux: default
  - Install plugins to appropriate locations (e.g., `/usr/local/lib/clap` for CLAP)
- [ ] Create config.h generation step in build (run ML export tool to produce config.h with `MAX_*` constants)
- [ ] Ensure all audio thread accessible data structures are cache-aligned (64-byte) to avoid false sharing
- [ ] Use aligned allocation (std::aligned_alloc, posix_memalign, or _aligned_malloc) for SIMD buffers
- [ ] Verify that weight file is memory-mapped read-only (use `mmap` with `PROT_READ`, `MAP_PRIVATE`)

### Testing & Validation (High Priority)
- [ ] Set up unit test framework (Catch2 or Google Test) for:
  - SSM core mathematical correctness
  - Parameter smoothing behavior
  - Weight loading and dequantization accuracy
  - State serialization round-trip
- [ ] Implement audio thread lock-free validation tests:
  - Run audio processing in a loop and check for any calls to malloc/free (use custom hooks or tools like Valgrind massif)
  - Verify no mutex/pthread_lock calls in audio thread (use Helgrind/TSAN or custom interlock detection)
  - Ensure all atomics use at least `memory_order_acquire/release` (seqlock uses acquire/release)
- [ ] Test CLAP plugin with official CLAP validator (`clap-validator`)
- [ ] Verify ARA 2.0 compatibility with at least one host (Studio One, Cubase, or Reaper via ARA bridge)
- [ ] Test in Reaper and Bitwig via CLAP wrapper (use `clap-wrapper` or similar)
- [ ] Measure DSP load: aim for <30% of real-time budget on a 4-core 2.5GHz CPU (use host's CPU meter or external tools like `perf` or `time`)
- [ ] Validate zero allocations in audio thread (run under Valgrind massif or AddressSanitizer with allocation hooks)
- [ ] Confirm zero mutex locks in audio path (run under Helgrind or ThreadSanitizer)
- [ ] Test model swapping: change only `config.h` and weight file to switch between 1M-param and 50M-param models, rebuild, and verify plugin loads and processes audio correctly
- [ ] Verify state serialization includes: plugin parameters (normalized), file path to weights, and base64-encoded h_state snapshot
- [ ] Test bypass click-free operation (for FX): engage/bypass automation should not produce audible clicks
- [ ] Test tail behavior: ensure tail time is reported correctly and audio decays to silence after input stops
- [ ] Test latency reporting: ensure plugin reports correct algorithmic latency to host
- [ ] Test denormal handling: ensure no performance penalties from denormals (consider using flush-to-zero or biasing)
- [ ] Test with silent input: ensure no NaNs or infinities
- [ ] Test with extreme parameters: ensure stability (no overflow/underflow)
- [ ] Test multi-channel support (if implemented): ensure correct processing for stereo/surround
- [ ] Test sample-accurate automation: verify that parameter changes occur at the exact sample offset specified in `ParamAtom.frameOffset`

### Documentation (Medium Priority)
- [ ] Create comprehensive API documentation (Doxygen or similar)
- [ ] Write build instructions (dependencies, CMake commands)
- [ ] Write user guide: how to install, use in DAWs, load presets, adjust parameters
- [ ] Write parameter reference guide: list all parameters, their ranges, modulation sources, smoothing times
- [ ] Write model export documentation for artists/researchers: how to train/export Mamba-SSM models for use in the plugins
- [ ] Create troubleshooting guide: common issues (missing SDKs, weight file not found, host compatibility)

### Release Preparation (Low Priority)
- [ ] Prepare release packages for:
  - CLAP (universal binary for macOS, Windows, Linux)
  - Optional VST3 wrappers (using CLAP-to-VST3 bridge if desired)
  - Optional AU wrappers (using CLAP-to-AU bridge)
  - Optional LV2 wrappers (for Linux)
- [ ] Ensure code signing for macOS (if distributing)
- [ ] Create installer scripts or package managers (Homebrew, Chocolatey, apt)

---

## NON-OBVIOUS CONSIDERATIONS FOR AI AGENTS

Please pay special attention to the following points, which are easy to overlook but critical for a real-time, lock-free audio plugin:

### Real-Time Audio Thread Safety
- **Zero heap allocations**: The audio thread (`process()` method) must **never** call `malloc`, `free`, `new`, `delete`, or any function that may internally allocate (e.g., `std::vector::push_back` if it triggers reallocation, `std::string`, `std::shared_ptr`). All memory used in the audio thread must be pre-allocated during `activate()` or the object's constructor.
- **Zero mutex locks**: Avoid `std::mutex`, `std::lock_guard`, `std::unique_lock`, `pthread_mutex_lock`, or any blocking synchronization primitive in the audio thread. Use lock-free/wait-free alternatives only.
- **No blocking OS calls**: Avoid `sleep`, `usleep`, `poll`, `select`, waiting on condition variables, or any file I/O (including reading/writing files) in the audio thread.
- **Avoid priority inversion**: Ensure that any shared resources accessed by lower-priority threads (main, ARA) do not block the audio thread for unbounded times. Use wait-free or lock-free data structures with bounded wait times.
- **Cache alignment and false sharing**: Align data structures to at least 64-byte cache lines (or 128-byte if targeting newer CPUs) to prevent false sharing between threads. Use `alignas(64)` or `__attribute__((aligned(64)))`.
- **Denormal numbers**: Denormal floating-point numbers can cause severe performance spikes. Either:
  - Enable flush-to-zero (FTZ) and denormals-are-zero (DAZ) in the MXCSR register at the start of each process block, or
  - Add a tiny bias to signals to keep them away from the denormal range, or
  - Use compiler flags like `-ftz -msse` (but runtime control is more portable).
- **SIMD alignment**: Ensure that any buffers passed to SIMD instructions (SSE/AVX) are aligned to 16-byte (SSE) or 32-byte (AVX) or 64-byte (AVX-512) boundaries. Use aligned allocation.
- **Instruction selection**: Avoid instructions that can cause exceptions or variable latency (e.g., integer division, some x87 FP instructions). Prefer SSE/AVX vector math.

### Lock-Free Programming Details
- **SPSC Ring Buffer**: If using a library like `moodycamel::ReaderWriterQueue`, ensure it is truly lock-free (it is). If implementing your own, use atomic indices with appropriate memory ordering (`memory_order_release` for producer, `memory_order_acquire` for consumer) and handle wrap-around correctly.
- **Double-Buffer Seqlock for State**: 
  - Audio thread writes to buffer `stateBuffers[writeIndex]` then flips `writeIndex` (atomic) to the other buffer **after** writing (so main thread always reads the non-writing buffer).
  - Main thread: 
      1. `uint32_t wi = stateWriteIndex.load(std::memory_order_acquire);`
      2. `const MambaState& state = stateBuffers[1 - wi];` // read the buffer not being written
    No seqlock needed on the state itself if we guarantee the audio thread only writes to the buffer indicated by `writeIndex` and flips only after finishing the write. However, if the audio thread may write multiple times to the same buffer before flipping, a seqlock on the state is safer. Either way, the main thread must verify consistency.
  - If using a seqlock on the `ConditioningFrame` (as described), the writer (ARA thread) does:
      ```
      frame.seqLock.fetch_add(1, std::memory_order_release); // make odd
      // write tempoBPM, keySig, chordOneHot
      frame.seqLock.fetch_add(1, std::memory_order_release); // make even
      ```
    The reader (audio thread) does:
      ```
      do {
          uint64_t start = frame.seqLoad(std::memory_order_acquire);
          if (start & 1) continue; // odd, retry
          // read data
          uint64_t end = frame.seqLoad(std::memory_order_acquire);
      } while (start != end);
      ```
- **Atomic Memory Ordering**: At a minimum, use `memory_order_release` for stores that need to be visible to other threads and `memory_order_acquire` for loads that need to see those stores. For simple flag communication, `memory_order_relaxed` may suffice if ordering doesn't matter, but be cautious.

### Memory Management & Weight Loading
- **Memory-Mapped Weights**: Use `mmap` (or `MapViewOfFile` on Windows) to map the weight file read-only. Do **not** `memcpy` the weights into writable memory; keep them read-only to allow OS paging and sharing. Ensure the weight file is cache-aligned (typically page-aligned, 4KB) for efficient mapping.
- **Weight File Format Determinism**: The export script must write weights in a **strict, deterministic order** (e.g., layer 0: input proj weight, recurrent proj weight, convolution weight, state update weight, output proj weight; then layer 1, etc.). Any non-determinism (like iterating over a hash map) will break reproducibility.
- **Quantization Handling**: If using affine quantization (scale + zero-point), the C++ loader must compute `real_value = (int8_value - zero_point) * scale` (or similar for FP16). Ensure the same computation is used in PyTorch during export validation.
- **config.h Generation**: The ML export tool should generate a `config.h` (or similar header) with `#define` constants for `MAX_BATCH`, `MAX_D_STATE`, `MAX_D_MODEL`, `MAX_PARAM_ATOMS`, `COND_DIM`, etc. These constants must match the actual dimensions in the weight file. If they don't, the plugin will either fail to load or read out-of-bounds.
- **Weight File Path in Presets**: Store either an absolute path or a path relative to the preset file. When loading a preset, first try the path as-is; if not found, look in the same directory as the preset file. Avoid hardcoding paths.

### Plugin-Specific Gotchas
- **CLAP Note Events**: In CLAP, note events are delivered on the audio thread (via `note_on`/`note_off` callbacks). Your implementation must be lock-free and allocation-free. Do not queue them to another thread unless you can guarantee the queue operations are lock-free and the audio thread will service them in time.
- **Parameter Automation Sample Accuracy**: The `frameOffset` in `ParamAtom` allows sample-accurate automation. Your smoothing/interpolation must respect this offset – do not simply apply the newest parameter value at the start of the block.
- **Bypass and Tail (FX)**: 
  - For bypass, ensure a smooth transition (e.g., crossfade or multiply by a ramp) to avoid clicks when engaging/bypassing.
  - For tail, compute the impulse response length (or estimate it) and report to the host via CLAP tail extension. The audio thread must continue processing for tail time after input goes silent.
- **Latency Reporting**: Report the algorithmic latency (in samples) of your plugin to the host via CLAP. This includes any internal buffering (e.g., for block processing, the latency is roughly half the block size if using overlap-add, but for causal SSM it's just the pipeline delay).
- **ARA Thread vs. Audio Thread**: The ARA thread (calling into `DocumentController`, `PlaybackRenderer`, `EditorRenderer`) is **not** the audio thread and may take arbitrarily long. However, it may still be called from a real-time context in some hosts (though not guaranteed). To be safe, keep ARA callbacks reasonably quick; defer heavy work (like weight loading or long analyses) to another thread if needed, but note that the ARA thread is allowed to block. The conditioning frame update should be quick (just a few float writes).
- **State Snapshots and Presets**: When serializing state for a preset, you must capture the current SSM state (`h_state`). Because the audio thread may be modifying it, use the double-buffer/seqlock mechanism to get a consistent snapshot without blocking the audio thread for long. The state snapshot can be base64-encoded (or raw binary) and stored in the preset.
- **Multi-Instance Safety**: Each plugin instance must have its own completely separate memory (SSM state, weights, parameters, etc.). No global static variables that are shared between instances (unless they are read-only constants). The weight file mapping can be shared (read-only mmap) but working buffers must be per-instance.
- **Thread Naming and Priority**: On platforms that allow it (e.g., pthreads), consider setting the audio thread to real-time priority (if the host grants it) and naming threads for easier debugging (e.g., "audio_thread", "main_thread", "ARA_thread").

### Build and Linking Specifics
- **CLAP and ARA SDKs**: Ensure you have the CLAP 1.1+ SDK and ARA 2.0 SDK installed and accessible to CMake. Use `find_path` or `find_library` or simply set paths via environment variables or CMake cache.
- **C++ Standard Library**: Avoid using parts of the STL that may allocate or throw in the audio thread (e.g., `std::vector` if it resizes, `std::string`, `std::shared_ptr`). Prefer raw arrays or fixed-size containers (`std::array` if size is known at compile time, or your own pre-allocated pool).
- **Exceptions and RTTI**: Disable exceptions (`-fno-exceptions`) and RTTI (`-fno-rtti`) because they can introduce hidden allocation and control flow overhead. If you disable exceptions, ensure your code does not throw (use error codes or assertions instead).
- **Link-Time Optimization (LTO)**: Consider using LTO (`-flto`) for better optimization, but verify it doesn't break real-time guarantees.
- **Visibility**: Use hidden visibility by default (`-fvisibility=hidden`) and explicitly export only the plugin entry points.

### Testing and Validation Specifics
- **Valgrind/ASan for Allocations**: Run the plugin under `valgrind --tool=massif` or `ASan` with allocation interceptors to ensure no allocations happen in the audio thread during a processing loop.
- **Helgrind/TSAN for Locks**: Run under Helgrind or ThreadSanitizer to detect any mutex locks or potential data races in the audio thread.
- **Real-Time Scheduling**: Use tools like `cyclictest` (Linux) or `rt-test` to verify that your plugin does not cause missed deadlines when under load.
- **Host Validation**:
  - CLAP: Run `clap-validator` on your `.clap` plugin.
  - VST3: Use the VST3 Plugin Test Host (if you wrap to VST3).
  - AU: Use `auval` (macOS).
  - ARA: Test with Studio One (free trial) or Cubase; verify that offline rendering and analysis work.
- **Audio Quality Tests**:
  - Feed a known signal (e.g., sine wave, impulse, pink noise) and compare output to a reference implementation (e.g., PyTorch offline) to ensure correctness.
  - Test bypass: engage/bypass with automation and ensure no clicks.
  - Test tail: trigger a short burst and verify output decays to silence after the reported tail time.
  - Test latency: use a delayed impulse and measure offset.
- **Fuzzing**: Consider fuzzing the plugin with random parameter values and random input signals to check for crashes or instabilities.

### Project Management and Documentation
- **Version Control**: Keep the weight file format versioned; increment the version number when making breaking changes.
- **Preset Versioning**: Include a version number in preset files to allow future evolution.
- **Build Reproducibility**: Ensure that given the same source and same weight file, the build produces bit-identical plugins (important for trust and validation).
- **Legal**: Verify that you have the right to distribute any SDKs (CLAP, ARA) or that users must obtain them separately. Typically, these SDKs are free to use but may require attribution.

---

## How to Use This Task List

1. **Pick a subsystem** (SSM Core, CLAP/ARA Integration, ML Pipeline, UI/State) and work through the high-priority tasks first.
2. **For each task**, write the implementation, then write unit tests that validate correctness and real-time safety.
3. **After completing a group of tasks**, run the validation steps (CLAP validator, allocation/lock checks, performance benchmarks).
4. **Iterate**: Real-time systems often require profiling and tweaking; use the suggested tools to identify and fix issues.
5. **Refer back to the "NON-OBVIOUS CONSIDERATIONS"** frequently – they are the difference between a plugin that works in ideal conditions and one that is robust in real-world host environments.

Good luck, and remember: in real-time audio, **correctness is not enough; it must also be deterministic and always fast**. 