#!/usr/bin/env bash
# Local mirror of .github/workflows/ci.yml. Uses $PYTHON (needs numpy, pyyaml,
# torch) and whatever C/C++ compiler CMake picks (override with CC/CXX).
# BUILD_DIR defaults to build/.
#
# Runs, in order and failing fast:
#   versions -> configure/build (-Werror) -> parity vectors -> ctest ->
#   python gates (ruff + losses + trainer smoke) -> clap-validator -> benchmark.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python3}"
B="${BUILD_DIR:-build}"

echo "== tool versions =="
cmake --version | head -1
"$PY" --version
if command -v clap-validator >/dev/null; then clap-validator --version; else echo "clap-validator: not on PATH"; fi

echo "== configure + build (warnings as errors) =="
cmake -B "$B" -G Ninja -DCMAKE_BUILD_TYPE=Release -DEMT_WERROR=ON ${CMAKE_ARGS:-}
cmake --build "$B" --parallel "$(nproc)"

echo "== parity vectors + weight export =="
"$PY" scripts/reference_mamba3.py --generate --output-dir "$B"
mkdir -p models
"$PY" scripts/export_mamba_weights.py --state-dim 16 --model-dim 16 --inner-dim 32 --nheads 2 \
  --layers 2 --random-init --output models/ci_test_model.bin
"$PY" scripts/validate_export.py models/ci_test_model.bin

echo "== ctest (parity required) =="
MAMBA_PARITY_DIR="$PWD/$B" MAMBA_REQUIRE_PARITY=1 MAMBA_TEST_WEIGHTS="$PWD/models/ci_test_model.bin" \
  ctest --test-dir "$B" --output-on-failure --timeout 120 -E '^clap-validator$'

echo "== python gates =="
"$PY" -m ruff check .
"$PY" -m ruff format --check .
"$PY" training/losses.py
T="$(mktemp -d)"; trap 'rm -rf "$T"' EXIT
"$PY" training/train.py --smoke-test --steps 1 --loss hybrid --output-dir "$T/smoke"
test -f "$T/smoke/plugin_weights.bin"

echo "== benchmark gate =="
BENCH="$(find "$B" -name benchmark_ssm_block -type f -perm -u+x | head -1)"
"$BENCH" --config small --layers 1,2 --sample-rate 48000 --budget-pct 40 --repeats 5 --seconds 0.1

echo "== clap-validator (any failure fails the script) =="
if command -v clap-validator >/dev/null; then
  clap-validator validate "$(find "$B" -name 'EmTSynth.clap' | head -1)"
else
  echo "SKIPPED: clap-validator not on PATH"
fi

echo "ci.sh: done"
