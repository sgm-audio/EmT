# EmT

Two CLAP plugins, a synth (`com.emt.synth`) and an effect (`com.emt.fx`), built
around a small causal state-space model (SSM) in the Mamba family, written in
C++20 with no framework. There's a PyTorch trainer and exporter that share the
model definition with the C++ weight loader.

Status (2026-10-31): builds and passes its tests on Linux. No trained model is
published, so the plugins run on random or test weights. Not ready for use in a
session (see Known problems).

## What's implemented

- Multi-head SSM layer with input-selective Δ/B/C, RMSNorm, B/C biases and a
  data-dependent rotation of B and C (`src/core/ssm_block.cpp`). The recurrence
  is a two-term bilinear (Tustin) update with a real-valued state. That's not
  the Mamba-3 paper's three-term exponential-trapezoidal recurrence or its
  complex state (deviations D1/D2, open). Rank-R MIMO is refused at load (D3).
  [`docs/ARCHITECTURE_FIDELITY.md`](docs/ARCHITECTURE_FIDELITY.md) has the
  per-item ledger with `file:line` references, and
  `python3 scripts/verify_deviations.py` prints the status of each deviation.
- FiLM conditioning on MIDI pitch, velocity and (in principle) DAW context.
- Synth: 8 voices, sample-accurate event slicing, per-voice state passed by
  reference into the core.
- FX: gain, mix and bypass with smoothing and a bypass crossfade. The core runs
  even when bypassed, and the wet path is the mono sum of the input.
- State save/load for both plugins (FX magic `MFXS` = `0x4D465853`,
  `src/clap/clap_fx_plugin.cpp`), presets under
  `$XDG_DATA_HOME/clap/presets/com.mamba-ssm.synth/` (falls back to
  `~/.local/share/...`; `src/clap/preset_paths.cpp`).
- Plain Win32 and Xlib editors (`src/clap/gui_*.cpp`). The images in
  `assets/gui/` are mockups and aren't used by the code.
- 0 samples of reported latency; the recurrence is causal, one sample per step.

ARA: `src/ara/` contains document-controller and renderer classes built
against stub types (`src/ara/ara_types.h:21` checks for a header the ARA SDK
doesn't ship, so the real-SDK path never turns on). The CLAP entry point only
exports the plugin factory, so no host can reach them. They are exercised by
`tests/test_ara.cpp` only. In a DAW, EmT is a plain CLAP plugin.

## Known problems

From a code read and test runs on 2026-10-31; details and the fix plan are in
the repo audit, not fixed here:

- Weight reload races the audio thread. `load_weights` runs on the main
  thread (`clap_state.cpp:184`, `preset_codec.cpp:414,645`) while `process`
  may be inside the core; `SSMBlock::weights_loaded_` is a plain `bool`.
- `SSMWeights w{}` (~4 MB) is declared on the stack in `clap_plugin.cpp:624`.
  That can overflow a host thread (1 MB on Windows, 512 KB on macOS secondary
  threads).
- FTZ/DAZ is set per block and never restored, so it leaks into the host
  thread.
- Synth conditioning is built from the newest voice and shared by all 8.
- The D-skip term mixes head-strided inputs into unrelated channels
  (`ssm_block.cpp:314-316`), and the Python reference does the same, so the
  parity test can't catch it.
- clap-validator 0.4.1: 63 passed, 6 failed, 12 skipped. The failures are
  `state-reproducibility-{basic,binary,buffered}` on both plugins: parameter
  values differ after a save/reload round trip.

## Performance

Measured 2026-10-31 with `tests/benchmark_ssm_block`. Machine: shared 8-vCPU
Intel Xeon VM (other jobs running). Compiler: zig clang 22.1.8, Release
(`-O3 -mavx2 -mfma`). Core only, one voice, 48 kHz. The budget at 48 kHz is
20.8 µs per sample per core.

| Shape | Layers | µs/sample | % of one core |
|---|---|---|---|
| `small` (d_state 16, d_model 16, d_inner 32) | 1 | 2.22–2.26 | 10.7–10.9 % |
| `small` | 2 | 4.56–4.63 | 21.9–22.2 % |
| `train4070` (`training/configs/laptop_rtx4070.yaml`) | 1 | 1.92–2.02 | 9.2–9.7 % |
| `train4070` | 3 | 5.84–5.87 | 28.0–28.2 % |
| `train4070` | 6 | 11.4–11.5 | 54.8–55.2 % |
| `trainl4` (`training/configs/runpod_l4.yaml`) | 1 | 14.7–16.0 | 70–77 % |
| `trainl4` | 6 | 87–91 | 419–435 % |

Ranges are across block sizes 64–512. The synth runs one core per active voice,
so multiply by the voice count. The RunPod L4 training config doesn't fit a
48 kHz budget at one layer with any headroom, and the plugin uses the
compile-time maximum shape until a weight file is loaded. Re-measure on your
own machine before quoting these.

## Build and test

Needs CMake 3.20+, Ninja, a C++20 compiler, Python 3.10+ with `numpy`,
`pyyaml` and `torch` (CPU wheel is fine), and `libx11-dev` on Linux. GTest is
fetched by CMake.

```bash
scripts/ci.sh        # configure, build, parity vectors, test weights, ctest,
                     # loss self-test, trainer smoke test, benchmark gate,
                     # clap-validator if installed
```

Tests: ctest registers 125 tests across 10 binaries (126 with clap-validator
installed). Two need generated inputs: `CppParity` (parity vectors) and
`RealtimeSafety.FxAcceptsMultiLayerWeightsWithinLimits` (an exported model).
Plain `ctest` runs 123 and skips those 2; `scripts/ci.sh` generates the inputs
and sets `MAMBA_REQUIRE_PARITY=1` so a missing file fails instead of skipping.

On 2026-10-31 (machine and compiler as above) `scripts/ci.sh` gave: ctest
125/125 passed, loss self-test passed, trainer smoke test wrote
`plugin_weights.bin`, benchmark gate passed (8/8 at 40 %), clap-validator as
listed under Known problems. Windows and macOS builds have not been checked
recently.

GitHub Actions (`.github/workflows/ci.yml`) runs the same steps, but no run has
completed: the account's Actions billing is blocked.

[`docs/BUILD.md`](docs/BUILD.md) covers offline builds, MSVC and
troubleshooting. Install by copying `build/src/clap/EmTSynth.clap` to `~/.clap/`
(Linux), `C:\Program Files\Common Files\CLAP\` (Windows) or
`/Library/Audio/Plug-Ins/CLAP/` (macOS).

## Training and export

```bash
python training/train.py --smoke-test --steps 2 --loss hybrid          # no dataset needed
python training/train.py --config training/configs/laptop_rtx4070.yaml \
    --dataset-dir data/wav --output-dir runs/run1
python training/train.py --config training/configs/laptop_rtx4070.yaml \
    --dataset-dir data/drywet --output-dir runs/fx --loss hybrid        # paired dry/ and wet/
python training/losses.py                                              # objective self-test
```

Losses: `mse`, `l1`, `sisdr`, `mrstft`, `hybrid` (default in the configs). The
trainer won't export weights if no optimisation step ran. Pick the shape with
the table above in mind; see [`docs/TRAINING_PLAN.md`](docs/TRAINING_PLAN.md).

Export and check a weight file:

```bash
python scripts/export_mamba_weights.py --load-checkpoint runs/run1/final.pt --output models/model.bin
python scripts/validate_export.py models/model.bin
```

The format is in [`docs/WEIGHT_FORMAT_SPEC.md`](docs/WEIGHT_FORMAT_SPEC.md).

## Layout

```
src/core     SSM block, inference core, weight loader
src/clap     CLAP entry, synth and FX plugins, params, state, presets, GUI
src/ara      ARA classes against stub types (not host-reachable)
src/common   ring buffer, seqlock, aligned storage
src/dsp      smoothing, envelope, bypass fader
tests/       GTest suites and the benchmark
scripts/     exporter, parity reference, deviation checker, ci.sh
training/    PyTorch trainer, losses, configs
docs/        build guide, architecture, fidelity ledger, weight format, ADRs (docs/adr/)
extern/clap  CLAP 1.2.7 headers (MIT)
assets/gui   GUI mockups (unused by the code)
```

## License

Apache 2.0, see [`LICENSE`](LICENSE). Vendored CLAP headers are MIT
([`extern/clap/LICENSE`](extern/clap/LICENSE)).
