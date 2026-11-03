# EmT

[![CI](https://github.com/sgm-audio/EmT/actions/workflows/ci.yml/badge.svg)](https://github.com/sgm-audio/EmT/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/sgm-audio/EmT/blob/main/LICENSE)

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
- Plain Win32 and Xlib editors (`src/clap/gui_*.cpp`). The mockup images
  (`assets/gui/`, plus four not yet moved there; see `assets/gui/README.md`)
  aren't used by the code.
- 0 samples of reported latency; the recurrence is causal, one sample per step.

ARA: `src/ara/` contains document-controller and renderer classes built
against stub types (`src/ara/ara_types.h:21` checks for a header the ARA SDK
doesn't ship, so the real-SDK path never turns on). The CLAP entry point only
exports the plugin factory, so no host can reach them. They are exercised by
`tests/test_ara.cpp` only. In a DAW, EmT is a plain CLAP plugin.

## Known problems

(Updated 2026-11-03 after the fix chain; earlier problems fixed on the fix
branches are listed in CHANGELOG.md.)

- ARA is a stub: `src/ara/` compiles but the ARA SDK is never fetched, so no
  ARA classes are host-reachable (`README.md` in `src/ara/`).
- No preset discovery: the CLAP preset-discovery factory is not implemented
  (hosts can't browse the factory presets; the preset files/JSON themselves
  work).
- Windows/MinGW build is broken (`std::aligned_alloc` and the `clap_entry`
  `dllexport`); Linux (clang/GCC) and the MSVC branch are the tested paths.
- clap-validator 0.4.1: 68 passed, 0 failed, 12 skipped (1 intermittent
  denormals warning). The 12 skips are the
  preset-discovery tests (no factory, see above).

## Performance

Measured 2026-11-02 with `tests/benchmark_ssm_block`. Machine: WSL2 Ubuntu
26.04, AMD Ryzen AI 7 350 (8C/16T, idle), gcc 15.2.0, Release
(`-O3 -mavx2 -mfma`). Core only, one voice, 48 kHz. The budget at 48 kHz is
20.8 µs per sample per core.

| Shape | Layers | µs/sample | % of one core |
|---|---|---|---|
| `small` (d_state 16, d_model 16, d_inner 32, rank 1) | 1 | 0.91–0.95 | 4.4–4.6 % |
| `small` | 2 | 1.86 | 8.9 % |
| `small` | 6 | 5.71–5.76 | 27.4–27.7 % |
| `train4070` (`training/configs/laptop_rtx4070.yaml`) | 1 | 1.30–1.32 | 6.3 % |
| `train4070` | 3 | 3.91–3.95 | 18.8–19.0 % |
| `train4070` | 6 | 8.0–8.3 | 38.4–40.0 % |
| `trainl4` (`training/configs/runpod_l4.yaml`) | 1 | 10.7–11.4 | 51.6–55.0 % |
| `trainl4` | 6 | 64.6–66.8 | 311–321 % |

Ranges are across block sizes 64–512. The synth runs one core per active voice,
so multiply by the voice count. The RunPod L4 training config doesn't fit a
48 kHz budget at one layer with any headroom, and the plugin uses the
compile-time maximum shape until a weight file is loaded. The `small` shape
uses rank 1 (the loader refuses `mimo_rank > 1`), so it describes a loadable
model. Re-measure on your own machine before quoting these.

## Build and test

Needs CMake 3.20+, Ninja, a C++20 compiler, Python 3.10+ with `numpy`,
`pyyaml` and `torch` (CPU wheel is fine), and `libx11-dev` on Linux. GTest is
fetched by CMake.

```bash
bash scripts/ci.sh   # configure, build, parity vectors, test weights, ctest,
                     # loss self-test, trainer smoke test, benchmark gate,
                     # clap-validator if installed
```

Tests: ctest registers 136 tests across 14 binaries (137 with clap-validator
installed). Two need generated inputs: `CppParity` (parity vectors) and
`RealtimeSafety.FxAcceptsMultiLayerWeightsWithinLimits` (an exported model).
Plain `ctest` runs 134 and skips those 2; `scripts/ci.sh` generates the inputs
and sets `MAMBA_REQUIRE_PARITY=1` so a missing file fails instead of skipping.

On 2026-11-03 (machine and compiler as above) `scripts/ci.sh` gave: ctest
135/135 passed, loss self-test passed, trainer smoke test wrote
`plugin_weights.bin`, benchmark gate passed, clap-validator 68 passed / 0
failed / 12 skipped (the 12 skips are preset-discovery, see Known problems).
Windows and macOS builds have not been checked recently.

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
assets/gui   GUI mockups (unused by the code; 4 more at the root and in packages/)
```

## License

Apache 2.0, see [`LICENSE`](LICENSE). Vendored CLAP headers are MIT
([`extern/clap/LICENSE`](extern/clap/LICENSE)).
