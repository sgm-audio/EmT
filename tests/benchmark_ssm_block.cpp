#include <chrono>
#include <cstdio>
#include <cmath>
#include <cstring>
#include "core/ssm_block.h"
#include "core/ssm_inference_core.h"
#include "config.h"

using namespace mamba;
using clock_type = std::chrono::steady_clock;

static double elapsed_seconds(clock_type::time_point start,
                              clock_type::time_point end) {
    return std::chrono::duration<double>(end - start).count();
}

static void setup_weights(SSMBlock& block) {
    SSMWeights w{};
    for (size_t i = 0; i < MAX_D_STATE; ++i) {
        w.A_log[i] = -1.0f;
        w.D_proj[i] = 0.5f;
        w.dt_bias[i] = 0.5f;
        for (size_t j = 0; j < MAX_D_INNER; ++j) {
            w.dt_proj_w[i][j] = 0.1f;
            w.B_proj_w[j][i] = 0.1f;
            w.C_proj_w[j][i] = 0.1f;
        }
    }
    for (size_t j = 0; j < MAX_D_INNER; ++j) {
        w.conv1d_b[j] = 0.1f;
        w.b_in[j] = 0.01f;
        for (size_t k = 0; k < MAX_D_CONV; ++k) {
            w.conv1d_w[k][j] = 0.05f;
        }
        for (size_t k = 0; k < MAX_D_MODEL; ++k) {
            w.W_in[j][k] = 0.05f;
        }
    }
    for (size_t j = 0; j < MAX_D_MODEL; ++j) {
        for (size_t k = 0; k < MAX_D_STATE; ++k) {
            w.W_ssm[j][k] = (j == k) ? 1.0f : 0.0f;
        }
        for (size_t k = 0; k < MAX_D_INNER; ++k) {
            w.W_out[j][k] = 0.01f;
        }
    }
    block.set_weights(w);
}

static double benchmark_per_sample(SSMBlock& block, size_t num_samples) {
    float input = 1.0f;
    auto start = clock_type::now();
    for (size_t n = 0; n < num_samples; ++n) {
        block.process_sample(input);
        input *= 0.999f;
    }
    auto end = clock_type::now();
    return elapsed_seconds(start, end) / static_cast<double>(num_samples);
}

static double benchmark_block(SSMBlock& block, size_t block_size,
                               size_t num_blocks) {
    alignas(64) float input[1024];
    alignas(64) float output[1024];
    for (size_t i = 0; i < block_size; ++i) input[i] = 1.0f;

    auto start = clock_type::now();
    for (size_t n = 0; n < num_blocks; ++n) {
        block.process_block(input, output, block_size);
    }
    auto end = clock_type::now();
    return elapsed_seconds(start, end) /
           static_cast<double>(num_blocks * block_size);
}

static double benchmark_multi_layer(SSMInferenceCore& core, size_t block_size,
                                     size_t num_blocks) {
    alignas(64) float input[1024];
    alignas(64) float output[1024];
    for (size_t i = 0; i < block_size; ++i) input[i] = 1.0f;

    auto start = clock_type::now();
    for (size_t n = 0; n < num_blocks; ++n) {
        core.process_block(input, output, block_size);
    }
    auto end = clock_type::now();
    return elapsed_seconds(start, end) /
           static_cast<double>(num_blocks * block_size);
}

int main() {
    SSMBlock::apply_ftz_daz();

    printf("=== Mamba-SSM Benchmark (config.h dimensions) ===\n");
    printf("MAX_D_STATE=%d, MAX_D_MODEL=%d, MAX_D_INNER=%d, "
           "MAX_D_CONV=%d, MAX_LAYERS=%d\n\n",
           MAX_D_STATE, MAX_D_MODEL, MAX_D_INNER, MAX_D_CONV, MAX_LAYERS);

    // --- Single-layer benchmarks ---
    SSMBlock block(MAX_D_STATE, MAX_D_MODEL, MAX_D_INNER, MAX_D_CONV);
    setup_weights(block);

    constexpr size_t WARMUP = 10000;
    constexpr size_t ITERS = 100000;

    // Warmup
    {
        float x = 1.0f;
        for (size_t n = 0; n < WARMUP; ++n) {
            block.process_sample(x);
            x *= 0.999f;
        }
    }
    block.reset();

    double ps = benchmark_per_sample(block, ITERS);
    printf("Single layer — per-sample:   %.3f us/sample  [%.0f samples/sec]\n",
           ps * 1e6, 1.0 / ps);

    block.reset();
    double pb64 = benchmark_block(block, 64, ITERS / 64);
    printf("Single layer — block(64):    %.3f us/sample  [%.0f samples/sec]\n",
           pb64 * 1e6, 1.0 / pb64);

    block.reset();
    double pb256 = benchmark_block(block, 256, ITERS / 256);
    printf("Single layer — block(256):   %.3f us/sample  [%.0f samples/sec]\n",
           pb256 * 1e6, 1.0 / pb256);

    block.reset();
    double pb512 = benchmark_block(block, 512, ITERS / 512);
    printf("Single layer — block(512):   %.3f us/sample  [%.0f samples/sec]\n",
           pb512 * 1e6, 1.0 / pb512);

    // --- Multi-layer benchmark ---
    {
        MambaConfig cfg{MAX_D_STATE, MAX_D_MODEL, MAX_D_INNER,
                        MAX_D_CONV, MAX_LAYERS, COND_DIM};
        SSMInferenceCore core(cfg);

        SSMWeights w{};
        for (size_t i = 0; i < MAX_D_STATE; ++i) {
            w.A_log[i] = -1.0f;
            w.D_proj[i] = 0.5f;
            w.dt_bias[i] = 0.5f;
            for (size_t j = 0; j < MAX_D_INNER; ++j) {
                w.dt_proj_w[i][j] = 0.1f;
                w.B_proj_w[j][i] = 0.1f;
                w.C_proj_w[j][i] = 0.1f;
            }
        }
        for (size_t j = 0; j < MAX_D_INNER; ++j) {
            w.conv1d_b[j] = 0.1f;
            w.b_in[j] = 0.01f;
            for (size_t k = 0; k < MAX_D_CONV; ++k) w.conv1d_w[k][j] = 0.05f;
            for (size_t k = 0; k < MAX_D_MODEL; ++k) w.W_in[j][k] = 0.05f;
        }
        for (size_t j = 0; j < MAX_D_MODEL; ++j) {
            for (size_t k = 0; k < MAX_D_STATE; ++k)
                w.W_ssm[j][k] = (j == k) ? 1.0f : 0.0f;
            for (size_t k = 0; k < MAX_D_INNER; ++k) w.W_out[j][k] = 0.01f;
        }
        for (uint32_t layer = 0; layer < MAX_LAYERS; ++layer) {
            core.set_layer_weights(layer, w);
        }

        // Warmup
        float x = 1.0f;
        for (size_t n = 0; n < WARMUP; ++n) {
            core.process_sample(x);
            x *= 0.999f;
        }
        core.reset();

        double psm = 0.0;
        {
            float x = 1.0f;
            auto start = clock_type::now();
            for (size_t n = 0; n < ITERS; ++n) {
                core.process_sample(x);
                x *= 0.999f;
            }
            auto end = clock_type::now();
            psm = elapsed_seconds(start, end) / ITERS;
        }
        printf("\n%d layers — per-sample:     %.3f us/sample  "
               "[%.0f samples/sec]\n",
               MAX_LAYERS, psm * 1e6, 1.0 / psm);

        core.reset();
        double pmb = benchmark_multi_layer(core, 512, ITERS / 512);
        printf("%d layers — block(512):    %.3f us/sample  "
               "[%.0f samples/sec]\n",
               MAX_LAYERS, pmb * 1e6, 1.0 / pmb);
    }

    // --- Check vs target ---
    double best_ps = ps;
    printf("\n---\n");
    if (best_ps < 10e-6) {
        printf("PASS: %.3f us/sample < 10 us/sample target\n",
               best_ps * 1e6);
    } else {
        printf("WARN: %.3f us/sample > 10 us/sample target\n",
               best_ps * 1e6);
    }

    return 0;
}
