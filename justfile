# justfile - build/test/package pipeline for EmT
# Usage: just [build|test|validate|bench|package|clean]
# All commands use `ctest` directly (no ctest --config on Windows)

# Platform detection (just os() returns lists, so we use shell)
IS_WINDOWS := (shell uname -s | grep -qiE "MINGW|MSYS|CYGWIN" && echo 1 || echo 0)
IS_DARWIN := (shell uname -s | grep -qi "darwin" && echo 1 || echo 0)
IS_LINUX  := (shell uname -s | grep -qi "linux" && echo 1 || echo 0)

# Generator selection
ifeq ($(IS_WINDOWS), 1)
    GENERATOR := "Visual Studio 18 2026"
    BUILD_CONFIG := Release
    CTEST_CONFIG_FLAG := --build-config
else ifeq ($(IS_DARWIN), 1)
    GENERATOR := "Xcode"
    BUILD_CONFIG := Release
    CTEST_CONFIG_FLAG := --build-config
else
    GENERATOR := "Ninja"
    BUILD_CONFIG := Release
    CTEST_CONFIG_FLAG := --config
endif

# Build directory
BUILD_DIR := build/$(BUILD_CONFIG)

# Core targets
build:
    cmake -S . -B $(BUILD_DIR) -G $(GENERATOR) -DCMAKE_BUILD_TYPE=$(BUILD_CONFIG)
    cmake --build $(BUILD_DIR) --config $(BUILD_CONFIG) --parallel

# Run all tests (ctest uses --build-config on Windows, --config on Unix)
test: build
    ctest --test-dir $(BUILD_DIR) --output-on-failure $(CTEST_CONFIG_FLAG) $(BUILD_CONFIG)

# Run full test suite with generated parity vectors and multi-layer weights (fails on skip)
test-full: build
    python3 scripts/reference_mamba3.py --generate --output-dir $(BUILD_DIR)
    mkdir -p models
    python3 scripts/export_mamba_weights.py --state-dim 16 --model-dim 16 --inner-dim 32 --nheads 2 --layers 2 --random-init --output models/ci_test_model.bin
    MAMBA_PARITY_DIR="$(PWD)/$(BUILD_DIR)" MAMBA_REQUIRE_PARITY=1 MAMBA_TEST_WEIGHTS="$(PWD)/models/ci_test_model.bin" ctest --test-dir $(BUILD_DIR) --output-on-failure $(CTEST_CONFIG_FLAG) $(BUILD_CONFIG)

# Run clap-validator specifically
validate: build
    ctest --test-dir $(BUILD_DIR) -R "clap-validator" --output-on-failure $(CTEST_CONFIG_FLAG) $(BUILD_CONFIG)

# Benchmark SSM block (not a ctest target - run directly)
bench: build
    $(BUILD_DIR)/tests/benchmark_ssm_block$(if $(IS_WINDOWS), .exe)

# Package CLAP bundle (Linux/macOS only - uses package.sh)
package: build
    ./scripts/package.sh $(BUILD_DIR)

# Clean build artifacts
clean:
    rm -rf build dist

# Default
default: build