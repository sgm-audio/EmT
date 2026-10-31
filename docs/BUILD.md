# Building and testing EmT

Quick version (see below for details, offline builds, and troubleshooting):

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel $(nproc)
python3 scripts/reference_mamba3.py --generate --output-dir build
export MAMBA_PARITY_DIR=$PWD/build MAMBA_REQUIRE_PARITY=1
ctest --test-dir build --output-on-failure
```

## Prerequisites

| Tool | Version | Notes |
|---|---|---|
| CMake | 3.20+ | `cmake --version` |
| C++20 compiler | MSVC 2022 (v143+) / GCC 13+ / Clang 17+ | AVX2 host for the `mavx2`/`arch:AVX2` flags |
| Python | 3.10+ | Weight export, parity vectors, training |
| Python packages | `numpy`, `pyyaml`, `torch` | `pip install numpy pyyaml torch` (torch: CPU wheel is enough) |
| Linux only | X11 headers | `sudo apt-get install -y libx11-dev` (for the X11 GUI backend) |
| Optional | `clap-validator` | Enables the `clap-validator` ctest step and build target |

GTest is fetched automatically via CMake FetchContent (v1.14.0, pinned hash).
See [Offline builds](#offline-builds) if the build machine has no network.

## Configure and build

```bash
# Linux / macOS, Release (what CI and the benchmark gate use)
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel $(nproc)

# Windows (MSVC, multi-config generator)
cmake -B build
cmake --build build --config Release --parallel
```

Useful configure flags:

| Flag | Effect |
|---|---|
| `-DEMT_USE_SYSTEM_GTEST=ON` | Use a system-installed GTest instead of FetchContent |
| `-DFETCHCONTENT_SOURCE_DIR_GOOGLETEST=<path>` | Use a vendored GTest checkout (combine with `-DFETCHCONTENT_FULLY_DISCONNECTED=ON` for fully offline config) |
| `-DX11_X11_INCLUDE_PATH=... -DX11_X11_LIB=...` | Point FindX11 at non-standard X11 headers/libs |

Per-target flags (set in the root `CMakeLists.txt`, not globally):

- The real-time libraries (`core`, `ara_plugin`, `mamba_synth`) build with
  `-fno-exceptions -fno-rtti -fvisibility=hidden` (`/GR-` on MSVC) plus
  `-Wall -Wextra -Wpedantic` (`/W4` on MSVC).
- Tests and GTest build with warnings but **with** exceptions/RTTI, which
  GTest and `std::filesystem` fixtures require.
- MSVC test executables (not the plugin) link with `/STACK:16777216`:
  `SSMWeights` is ~4 MB at max config and the 1 MB default stack is too small.
  A plugin can't set the host thread's stack, so `clap_plugin.cpp:624`
  (`SSMWeights w{}` on the stack) is a known overflow risk in hosts.

## Tests

Generate the parity vectors **before** running ctest — otherwise the
C++/Python parity test silently skips and a green run proves nothing:

```bash
python3 scripts/reference_mamba3.py --generate --output-dir build
```

Then run the suite (125 tests across 10 binaries, plus `clap-validator` when
the tool is installed):

```bash
export MAMBA_PARITY_DIR=$PWD/build   # where the parity vectors live
export MAMBA_REQUIRE_PARITY=1        # missing vectors fail instead of skip
export MAMBA_TEST_WEIGHTS=$PWD/models/ci_test_model.bin  # see below
ctest --test-dir build --output-on-failure --timeout 120
```

`MAMBA_TEST_WEIGHTS` points at a real multi-layer export so the FX model-load
path is exercised for real (CI generates it; see the checklist below). Without
it, that test skips.

A single suite: `./build/tests/test_weight_loader` (then
`--gtest_filter=*MimoRank*` etc. for subsets).

## Full CI-parity checklist

This mirrors `.github/workflows/ci.yml` step for step:

```bash
python3 scripts/reference_mamba3.py --generate --output-dir build
python3 scripts/export_mamba_weights.py \
    --state-dim 16 --model-dim 16 --inner-dim 32 --nheads 2 \
    --layers 2 --random-init \
    --output models/ci_test_model.bin
python3 scripts/validate_export.py models/ci_test_model.bin
export MAMBA_PARITY_DIR=$PWD/build MAMBA_REQUIRE_PARITY=1 \
    MAMBA_TEST_WEIGHTS=$PWD/models/ci_test_model.bin
ctest --test-dir build --output-on-failure --timeout 120
python3 training/losses.py
python3 training/train.py --smoke-test --steps 1 --loss hybrid \
    --output-dir /tmp/emt_smoke
./build/tests/benchmark_ssm_block \
    --config small --layers 1,2 --sample-rate 48000 \
    --budget-pct 40 --repeats 5 --seconds 0.1
python3 scripts/verify_deviations.py --build-dir build
```

`verify_deviations.py` is the deviation tripwire: D1/D2 are expected OPEN,
D3–D6 must be CLOSED. `--no-build-checks` runs the source-level checks only
(no test binaries needed).

## Offline builds

1. Vendor GTest once (any machine with network):
   `git clone --branch v1.14.0 --depth 1 https://github.com/google/googletest <path>`.
2. Configure with
   `-DFETCHCONTENT_SOURCE_DIR_GOOGLETEST=<path> -DFETCHCONTENT_FULLY_DISCONNECTED=ON`,
   or install GTest system-wide and pass `-DEMT_USE_SYSTEM_GTEST=ON`.
3. Python deps (`numpy`, `pyyaml`, `torch`) must already be installed.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `find_package(X11)` fails on Linux | Install `libx11-dev`, or point `X11_X11_INCLUDE_PATH` / `X11_X11_LIB` at a vendored copy |
| `CppParity` skips | Parity vectors missing: run `reference_mamba3.py --generate` first; set `MAMBA_REQUIRE_PARITY=1` to make skips fail |
| `FxAcceptsMultiLayerWeightsWithinLimits` skips | `MAMBA_TEST_WEIGHTS` unset: export a model as in the checklist |
| FetchContent download fails | No network: use the offline-build flags above |
| MSVC stack overflow in tests | The 16 MB `/STACK` flag is set per-target in `tests/CMakeLists.txt`; a custom target needs it too for `SSMWeights`-sized objects |
