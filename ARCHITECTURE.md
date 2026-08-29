# EmT Architecture

**Implementation snapshot:** 2026-08-29, source version 0.3.x. This document describes the current checkout, including limitations; it is not a claim of host validation or release readiness.

## 1. Product topology

One CLAP binary currently exports two stable plugin identities:

| Product | Stable CLAP ID | Current capability |
|---|---|---|
| EmT Synth | `com.mamba-ssm.synth` | Instrument wrapper, eight voice slots, five params, state/presets, Linux/X11 GUI |
| EmT FX | `com.mamba-ssm.fx` | Effect wrapper and three params; no weight-load/state/GUI path, so wet inference is silent |

The binary/bundle and internal namespace still use Mamba names. Those are compatibility-sensitive. ARA sources compile as a separate stub library but no real ARA factory/plugin is exported.

```text
Host
  -> clap_entry / factory
      -> EmT Synth wrapper ----+
      -> EmT FX wrapper -------+--> SSMInferenceCore --> SSMBlock[1..L]
                                      ^
Main-thread WeightLoader map/validate/materialize
```

## 2. Execution contexts

The older “three-thread architecture” was misleading: the repository does not create three threads. It supplies callbacks/data structures used in host-controlled contexts.

| Context | Work | Exchange |
|---|---|---|
| Host audio callback | event dispatch, smoothing, voices/bypass, SSM inference, output | drains SPSC params; reads conditioning; publishes state mirror |
| Host main callback | lifecycle, stopped params, state/presets, weight loading | publishes params/model state; reads mirror |
| Linux GUI callbacks | X11 drawing/input and timer pumping | atomic parameter fields and host flush request |
| ARA stub callers | update global context helpers | seqlocked `ARAConditioningFrame` |

Audio-path prohibitions: allocation/deallocation, mutexes, waits, blocking I/O, environment/filesystem access, logging I/O, exceptions, and unbounded operations.

### 2.1 SPSC parameters

`SPSCRingBuffer<ParamAtom, 256>` carries `{id, value, frame_offset}` from non-audio flush calls to the audio callback. Capacity is bounded. A full queue rejects pushes; loss behavior and diagnostics need explicit host tests.

### 2.2 Conditioning seqlocks

`Seqlocked<T>` wraps a payload and generation counter. The writer increments to odd, writes, then increments to even. Readers retry until they copy across a stable even generation. Atomic ordering is acquire/release/acq_rel, but the payload itself is plain memory; correctness assumes one writer and the implemented protocol. A reader can spin indefinitely if a writer stalls.

### 2.3 State mirror

`DoubleBufferSeqLock<StateMirror>` embeds two fixed slots. The audio writer fills its current slot and publishes it; the main reader copies the published slot and verifies that generation did not change. `read()` can fail and is not an ownership handoff. `swap()` selects the next producer slot.

## 3. Object and memory ownership

- `clap_plugin_t`, `MambaSynth`, and `FxState` are allocated with non-throwing `new` during main-thread creation and deleted at destruction.
- Each wrapper embeds storage for one `SSMInferenceCore`; placement new controls its lifetime.
- Each core embeds storage for `MAX_LAYERS` `SSMBlock` objects.
- Each block owns a fixed-capacity `SSMState`, `SSMWeights`, and scratch arrays. This is large but bounded.
- Each synth voice owns a full `MambaState` snapshot. Active voices swap that state into/out of one shared core for every processed sample.
- `ArenaAllocator` is compiled into `core` but is not used by current plugin creation.
- Weight files are mapped read-only. Every layer is then copied/dequantized into writable, fixed-stride `SSMWeights`; inference is not zero-copy.

MSVC test executables request a 16 MiB stack. Large plugin instances live on the heap. The plugin target itself does not currently set `/STACK:16777216`.

## 4. Mamba-3 layer

Runtime configuration contains state/model/inner dimensions, head count/dimension, MIMO rank, BC grouping, RoPE count, layer count, and conditioning dimension. File-provided fields are bounded by `src/config.h`; several architectural values are implicit fixed v2 conventions.

Per sample, a block conceptually performs:

1. optional FiLM modulation of the scalar-expanded model input;
2. fused input projection into `z`, `x`, padded B/C terms, per-head delta/A/trapezoid terms, and shared RoPE angles;
3. B/C normalization plus learned biases;
4. shared-angle RoPE rotation of B/C state pairs;
5. complex multi-head state recurrence with trapezoidal terms;
6. fixed-stride MIMO coupling, gating/skip path, output projection;
7. output FiLM and gain/mix parameters.

Mamba-3 in this repository has no `conv1d`; CLAP latency correctly returns zero samples. That is algorithmic latency only.

`SSMBlock::process_block()` is a per-sample fallback. `SSMInferenceCore::process_block()` uses fixed ping-pong buffers between layers but gains no batched kernel yet.

## 5. Weight-loading boundary

`WeightLoader::load()` resolves a path, maps the file read-only, validates header/format/bounds, and calculates layer views. It accepts version 2 only. `dequantize_layer_to()` is the canonical packed-file-to-fixed-stride materializer for all formats, including Float32.

Synth loading currently:

1. maps/validates the file;
2. adopts runtime dimensions and reinitializes the core;
3. materializes and installs each layer;
4. sets the loaded flag and records the path.

This sequence is not transactional. Failure after core reinitialization can discard the old working core. Fix before supporting live model replacement.

EmT FX owns loader/path fields but never invokes them. This is a functional gap, not merely missing UI.

## 6. CLAP wrappers

### 6.1 Shared ports

`s_audio_ports` advertises one main stereo input and output with 32/64-bit support and in-place pairing. Both plugins reuse it. Whether the synth should have an audio input is unresolved.

### 6.2 EmT Synth

- one CLAP/MIDI note input;
- Gain, Attack, Release, Mix, Bypass;
- sample-timed event handling inside the sample loop;
- eight voices, shared core, mono inference output duplicated to stereo;
- binary state and internal `.mambapreset` helpers;
- tail of approximately 100 ms and zero reported latency;
- Linux/X11 GUI and plugin timer callback.

The Attack/Release parameter contract needs review against per-voice fixed rates. State/preset I/O and weight loading perform non-real-time operations and must remain off the process path.

### 6.3 EmT FX

- Gain, Mix, Bypass;
- stereo input is summed to mono;
- only `processed[0]` is used and duplicated across outputs;
- no model weights are loaded, so `SSMBlock` writes zeros;
- no state extension or GUI.

No-weight safe behavior is undefined. A future implementation should choose explicit passthrough/error semantics and publish a fully constructed core transactionally.

### 6.4 Thread check and presets

The plugin consumes the host's `CLAP_EXT_THREAD_CHECK` when present; it does not expose a plugin thread-check extension. Presets are proprietary helper files, not CLAP preset-discovery/load support.

## 7. ARA boundary

`src/ara` uses hand-written types, a global controller singleton, and renderer function-pointer wrappers. Header presence can define `MAMBA_ARA_REAL_SDK`, but the classes do not become a complete SDK-derived integration merely because headers exist. No real ARA factory, binding, document lifecycle, or conformance evidence is present.

Real integration is blocked on the licensed Celemony SDK and must be designed/tested from that API. Global context must be reconsidered for multiple plugin instances/documents.

## 8. Build and packaging

- Root CMake builds `core`, `common`, `ara_plugin`, one shared CLAP binary, six test executables, and a manual benchmark.
- Linux requires X11 even for the plugin build.
- Windows produces a `.clap` file and installs to the common CLAP directory.
- Current macOS rules set a suffix/install destination but do not demonstrate a valid CLAP bundle structure or signing/notarization.
- The validator target depends on `bash`, treats missing validator as success, and is not a quality gate.

## 9. Verification model

Current unit tests cover core behavior, loader cases/formats, conditioning, bounded stability loops, source-oriented real-time checks, and one parity fixture. They do not prove host compliance, no-allocation plugin processing, FX correctness, ten-minute wall-clock stability, cross-platform packaging, or production performance.

Every phase should collect evidence separately:

```text
source-reviewed -> compiled -> unit-tested -> parity/export-tested
-> benchmarked -> validator-tested -> host-tested -> packaged -> released
```

Do not collapse these states in reports. See `TODO.md` for phase gates and missing tests.
