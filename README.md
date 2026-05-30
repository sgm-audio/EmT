# Mamba-SSM CLAP/ARA Plugin Suite

Real-time neural audio synthesis plugins powered by the Mamba state-space model architecture. Delivered as CLAP plugins with optional ARA 2.0 integration for DAW-hosted generative workflows.

## Architecture

Three-thread lock-free design with strict real-time guarantees on the audio thread:

| Thread | Role | Communication |
|--------|------|---------------|
| **Audio** | `plugin_process()`, SSM inference | SPSC ring buffer for params, seqlocked frames for conditioning |
| **Main** | UI, preset load/save | Atomically pushes `ParamAtom` to ring buffer |
| **ARA** | Musical context, offline render | Seqlock-guarded `ARAConditioningFrame` writes |

## Processing Pipeline

```
MIDI/conditioning → W_in → conv1d → SiLU → SSM scan → W_out + skip → output
                                     ↕
                           dt/B/C projections
```

The SSM recurrence uses ZOH-discretized selective scan (per Mamba paper), computed elementwise per state dimension without heap allocation.

## Directory Structure

```
src/
  core/          SSMBlock, SSMInferenceCore, arena allocator, WeightLoader
  clap/          CLAP plugin entry, extensions, parameter management
  common/        SPSCRingBuffer, SeqLock, lock-free utilities
  ara/           ARA stubs (requires Celemony ARA SDK)
tests/           GTest-based unit tests + benchmark
scripts/         Python export pipeline (PyTorch → flat-binary → validate)
models/          Exported .bin weight files
extern/clap/     Vendored CLAP 1.1+ headers
```

## Building

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4
./build/tests/test_ssm_block
./build/tests/test_weight_loader
./build/tests/benchmark_ssm_block
```

Requires CMake 3.20+, C++20, GCC or Clang.

## Weight Export

```bash
# Generate test weights with random init
python scripts/export_mamba_weights.py --random-init --layers 4 --output models/model.bin

# Validate round-trip through PyTorch
python scripts/validate_export.py models/model.bin

# Generate weights from trained checkpoint
python scripts/export_mamba_weights.py --load-checkpoint model.pt --config-h src/config.h
```

## Current Status

~25-30% complete. Core SSM inference, weight loading, CLAP plugin shell, lock-free primitives, and export pipeline are working. Missing: block-vectorized processing path, voice management, GUI, ARA SDK linkage, ML training pipeline.

## License

Apache 2.0 — see [LICENSE](LICENSE).
