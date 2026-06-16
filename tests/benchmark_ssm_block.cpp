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

static SSMWeights make_test_weights() {
    SSMWeights w{};
    std::memset(&w, 0, sizeof(w));
    for (size_t out = 0; out < SSMWeights::IN_PROJ_OUT; ++out) {
        for (size_t j = 0; j < MAX_D_MODEL; ++j)
            w.in_proj_w[out][j] = 0.1f;
    }
    for (size_t h = 0; h < MAX_NHEADS; ++h)
        w.dt_bias[h] = 0.5f;
    for (size_t s = 0; s < MAX_D_STATE; ++s) {
        w.B_norm_weight[s] = 1.0f;
        w.C_norm_weight[s] = 1.0f;
    }
    for (size_t j = 0; j < MAX_D_MODEL && j < MAX_D_INNER; ++j)
        w.out_proj_w[j][j] = 1.0f;
    return w;
}

static SSMBlock make_test_block() {
    return SSMBlock(MAX_D_STATE, MAX_D_MODEL, MAX_D_INNER,
                    MAX_NHEADS, MAX_HEADDIM, MAX_MIMO_RANK,
                    MAX_NUM_BC_HEADS, COND_DIM);
}

static double benchmark_per_sample(SSMBlock& block, size_t num_samples) {
    alignas(64) float output[MAX_D_MODEL];
    float input = 1.0f;
    auto start = clock_type::now();
    for (size_t n = 0; n < num_samples; ++n) {
        block.process_sample(input, output);
        input *= 0.999f;
    }
    auto end = clock_type::now();
    return elapsed_seconds(start, end) / static_cast<double>(num_samples);
}

static double benchmark_block(SSMBlock& block, size_t block_size,
                                size_t num_blocks) {
    alignas(64) float input[1024];
    alignas(64) float output[1024 * MAX_D_MODEL];
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
    alignas(64) float output[1024 * MAX_D_MODEL];
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

    printf("=== Mamba-3 Benchmark (config.h dimensions) ===\n");
    printf("MAX_D_STATE=%d, MAX_D_MODEL=%d, MAX_D_INNER=%d, "
           "MAX_NHEADS=%d, MAX_HEADDIM=%d, MAX_LAYERS=%d\n\n",
           MAX_D_STATE, MAX_D_MODEL, MAX_D_INNER,
           MAX_NHEADS, MAX_HEADDIM, MAX_LAYERS);

    auto block = make_test_block();
    block.set_weights(make_test_weights());

    constexpr size_t WARMUP = 10000;
    constexpr size_t ITERS = 100000;

    // Warmup
    {
        alignas(64) float output[MAX_D_MODEL];
        float x = 1.0f;
        for (size_t n = 0; n < WARMUP; ++n) {
            block.process_sample(x, output);
            x *= 0.999f;
        }
    }
    block.reset();

    double ps = benchmark_per_sample(block, ITERS);
    printf("Single layer - per-sample:   %.3f us/sample  [%.0f samples/sec]\n",
           ps * 1e6, 1.0 / ps);

    block.reset();
    double pb64 = benchmark_block(block, 64, ITERS / 64);
    printf("Single layer - block(64):    %.3f us/sample  [%.0f samples/sec]\n",
           pb64 * 1e6, 1.0 / pb64);

    block.reset();
    double pb512 = benchmark_block(block, 512, ITERS / 512);
    printf("Single layer - block(512):   %.3f us/sample  [%.0f samples/sec]\n",
           pb512 * 1e6, 1.0 / pb512);

    // Multi-layer
    {
        MambaConfig cfg = MambaConfig::max_config();
        cfg.num_layers = MAX_LAYERS;
        auto core_ptr = std::make_unique<SSMInferenceCore>(); core_ptr->init(cfg);

        for (uint32_t layer = 0; layer < MAX_LAYERS; ++layer) {
            core_ptr->set_layer_weights(layer, make_test_weights());
        }

        // Warmup
        {
            alignas(64) float output[MAX_D_MODEL];
            float x = 1.0f;
            for (size_t n = 0; n < WARMUP; ++n) {
                core_ptr->process_sample(x, output);
                x *= 0.999f;
            }
        }
        core_ptr->reset();

        double psm = 0.0;
        {
            alignas(64) float output[MAX_D_MODEL];
            float y = 1.0f;
            auto start = clock_type::now();
            for (size_t n = 0; n < ITERS; ++n) {
                core_ptr->process_sample(y, output);
                y *= 0.999f;
            }
            auto end = clock_type::now();
            psm = elapsed_seconds(start, end) / ITERS;
        }
        printf("\n%d layers - per-sample:     %.3f us/sample  "
               "[%.0f samples/sec]\n",
               MAX_LAYERS, psm * 1e6, 1.0 / psm);

        core_ptr->reset();
        double pmb = benchmark_multi_layer(*core_ptr, 512, ITERS / 512);
        printf("%d layers - block(512):    %.3f us/sample  "
               "[%.0f samples/sec]\n",
               MAX_LAYERS, pmb * 1e6, 1.0 / pmb);
    }

    double best_ps = ps;
    printf("\n---\n");
    if (best_ps < 10e-6) {
        printf("PASS: %.3f us/sample < 10 us/sample target\n", best_ps * 1e6);
    } else {
        printf("WARN: %.3f us/sample > 10 us/sample target\n", best_ps * 1e6);
    }

    return 0;
}
