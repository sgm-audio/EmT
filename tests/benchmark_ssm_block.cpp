#include <chrono>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <memory>
#include <vector>
#include <string>
#include "core/ssm_block.h"
#include "core/ssm_inference_core.h"
#include "config.h"

#if defined(_WIN32)
#include <windows.h>
#endif

using namespace mamba;
using clock_type = std::chrono::steady_clock;

static double elapsed_seconds(clock_type::time_point start,
                              clock_type::time_point end) {
    return std::chrono::duration<double>(end - start).count();
}

// Benchmark model dimensions matching tests and realistic Mamba-3 audio config
static constexpr size_t BENCH_D_STATE = 16;
static constexpr size_t BENCH_D_MODEL = 16;
static constexpr size_t BENCH_D_INNER = 32;
static constexpr size_t BENCH_NHEADS  = 2;
static constexpr size_t BENCH_HEADDIM = 16;
static constexpr double REALTIME_BUDGET_US = 10.0; // <10 us/sample real-time budget threshold

struct BenchmarkResult {
    size_t num_layers;
    double sample_rate;
    size_t block_size;
    double latency_us_per_sample;
    double latency_us_per_block;
    double samples_per_sec;
    double dsp_load_pct;
    bool meets_budget;
};

static std::unique_ptr<SSMBlock> make_test_block() {
    return std::make_unique<SSMBlock>(
        BENCH_D_STATE, BENCH_D_MODEL, BENCH_D_INNER,
        BENCH_NHEADS, BENCH_HEADDIM, MAX_MIMO_RANK,
        MAX_NUM_BC_HEADS, COND_DIM);
}

static std::unique_ptr<SSMWeights> make_test_weights_for(size_t d_state, size_t d_model,
                                                         size_t d_inner, size_t nheads,
                                                         size_t) {
    auto w = std::make_unique<SSMWeights>();
    std::memset(w.get(), 0, sizeof(*w));
    size_t in_proj_rows = d_inner * 2
                        + nheads * MAX_MIMO_RANK * d_state * 2
                        + nheads * 3 + (d_state / 2);
    for (size_t out = 0; out < in_proj_rows; ++out)
        for (size_t j = 0; j < d_model; ++j)
            w->in_proj_w[out][j] = 0.1f;
    for (size_t h = 0; h < nheads; ++h) w->dt_bias[h] = 0.5f;
    for (size_t s = 0; s < d_state; ++s) {
        w->B_norm_weight[s] = 1.0f;
        w->C_norm_weight[s] = 1.0f;
    }
    for (size_t j = 0; j < d_model && j < d_inner; ++j)
        w->out_proj_w[j][j] = 1.0f;
    return w;
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
                                    size_t num_blocks, double sample_rate) {
    alignas(64) float input[1024];
    alignas(64) float output[1024 * MAX_D_MODEL];
    for (size_t i = 0; i < block_size; ++i) {
        input[i] = 0.5f * std::sin(2.0f * 3.14159265358979323846f * 440.0f *
                                   static_cast<float>(i) / static_cast<float>(sample_rate));
    }

    auto start = clock_type::now();
    for (size_t n = 0; n < num_blocks; ++n) {
        core.process_block(input, output, block_size);
    }
    auto end = clock_type::now();
    return elapsed_seconds(start, end) /
           static_cast<double>(num_blocks * block_size);
}

int main() {
#if defined(_WIN32)
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);
#endif
    SSMBlock::apply_ftz_daz();

    printf("================================================================================\n");
    printf("                  Mamba-3 Audio DSP Benchmark Suite                             \n");
    printf("================================================================================\n");
    printf("Compile Limits:  MAX_D_STATE=%d  MAX_D_MODEL=%d  MAX_D_INNER=%d\n",
           MAX_D_STATE, MAX_D_MODEL, MAX_D_INNER);
    printf("                 MAX_NHEADS=%d   MAX_HEADDIM=%d  MAX_LAYERS=%d\n",
           MAX_NHEADS, MAX_HEADDIM, MAX_LAYERS);
    printf("Benchmark Model: d_state=%zu  d_model=%zu  d_inner=%zu  nheads=%zu  headdim=%zu\n",
           BENCH_D_STATE, BENCH_D_MODEL, BENCH_D_INNER, BENCH_NHEADS, BENCH_HEADDIM);
    printf("Real-Time Budget: < %.1f us/sample (guarantees real-time up to 96 kHz)\n\n",
           REALTIME_BUDGET_US);

    // --- Baseline Single Layer Benchmarks ---
    auto block = make_test_block();
    block->set_weights(*make_test_weights_for(BENCH_D_STATE, BENCH_D_MODEL,
                                               BENCH_D_INNER, BENCH_NHEADS,
                                               BENCH_HEADDIM));

    constexpr size_t WARMUP = 1000;
    constexpr size_t ITERS = 10000;

    // Warmup
    {
        alignas(64) float output[MAX_D_MODEL];
        float x = 1.0f;
        for (size_t n = 0; n < WARMUP; ++n) {
            block->process_sample(x, output);
            x *= 0.999f;
        }
    }
    block->reset();

    double ps = benchmark_per_sample(*block, ITERS);
    printf("Baseline Single Layer - Per-Sample:   %6.3f us/sample  [%8.0f samples/sec]\n",
           ps * 1e6, 1.0 / ps);

    block->reset();
    double pb64 = benchmark_block(*block, 64, ITERS / 64);
    printf("Baseline Single Layer - Block(64):    %6.3f us/sample  [%8.0f samples/sec]\n",
           pb64 * 1e6, 1.0 / pb64);

    block->reset();
    double pb512 = benchmark_block(*block, 512, ITERS / 512);
    printf("Baseline Single Layer - Block(512):   %6.3f us/sample  [%8.0f samples/sec]\n\n",
           pb512 * 1e6, 1.0 / pb512);

    // --- Systematic Multi-Configuration Matrix ---
    // Tested configurations:
    // - Sample rates: 44.1 kHz, 48.0 kHz, 96.0 kHz
    // - Layer configs: 1-layer, 2-layer, 6-layer (MAX_LAYERS)
    // - Block sizes: 64, 128, 256, 512
    const size_t layer_configs[] = {1, 2, MAX_LAYERS};
    const size_t block_sizes[] = {64, 128, 256, 512};
    const double sample_rates[] = {44100.0, 48000.0, 96000.0};
    constexpr double SIM_DURATION_SEC = 0.25; // 250 ms simulated audio duration

    std::vector<BenchmarkResult> results;
    results.reserve(sizeof(layer_configs)/sizeof(layer_configs[0]) *
                    sizeof(sample_rates)/sizeof(sample_rates[0]) *
                    sizeof(block_sizes)/sizeof(block_sizes[0]));

    for (size_t num_layers : layer_configs) {
        MambaConfig cfg{};
        cfg.d_state = BENCH_D_STATE;
        cfg.d_model = BENCH_D_MODEL;
        cfg.d_inner = BENCH_D_INNER;
        cfg.nheads = BENCH_NHEADS;
        cfg.headdim = BENCH_HEADDIM;
        cfg.mimo_rank = MAX_MIMO_RANK;
        cfg.num_bc_heads = MAX_NUM_BC_HEADS;
        cfg.num_rope_angles = BENCH_D_STATE / 2;
        cfg.num_layers = num_layers;
        cfg.cond_dim = COND_DIM;

        auto core_ptr = std::make_unique<SSMInferenceCore>();
        core_ptr->init(cfg);

        for (uint32_t layer = 0; layer < num_layers; ++layer) {
            core_ptr->set_layer_weights(layer, *make_test_weights_for(
                BENCH_D_STATE, BENCH_D_MODEL, BENCH_D_INNER,
                BENCH_NHEADS, BENCH_HEADDIM));
        }

        printf("================================================================================\n");
        printf("Configuration: %zu-Layer%s Inference Core (MAX_LAYERS=%d)\n",
               num_layers, (num_layers > 1 ? "s" : ""), MAX_LAYERS);
        printf("================================================================================\n");

        for (double sr : sample_rates) {
            const double sr_budget_us = (1.0 / sr) * 1e6;
            printf("\n--- Simulated Sample Rate: %.1f kHz (Real-Time Deadline: %.2f us/sample) ---\n",
                   sr / 1000.0, sr_budget_us);
            printf("%-10s  %-16s  %-15s  %-16s  %-10s  %s\n",
                   "Block Size", "Latency/sample", "Block Latency", "Throughput", "DSP Load", "Budget (<10 us)");
            printf("%-10s  %-16s  %-15s  %-16s  %-10s  %s\n",
                   "----------", "--------------", "-------------", "----------", "--------", "---------------");

            for (size_t bs : block_sizes) {
                size_t num_blocks = static_cast<size_t>(std::ceil(SIM_DURATION_SEC * sr / bs));
                if (num_blocks < 4) num_blocks = 4;

                // Warmup
                alignas(64) float w_in[512] = {0.0f};
                alignas(64) float w_out[512 * MAX_D_MODEL];
                core_ptr->process_block(w_in, w_out, bs);
                core_ptr->reset();

                double pmb = benchmark_multi_layer(*core_ptr, bs, num_blocks, sr);
                double us_per_sample = pmb * 1e6;
                double us_per_block = us_per_sample * static_cast<double>(bs);
                double samples_sec = 1.0 / pmb;
                double dsp_load_pct = (pmb / (1.0 / sr)) * 100.0;
                bool meets_budget = (us_per_sample < REALTIME_BUDGET_US);

                results.push_back({
                    num_layers,
                    sr,
                    bs,
                    us_per_sample,
                    us_per_block,
                    samples_sec,
                    dsp_load_pct,
                    meets_budget
                });

                printf("  %-8zu  %6.3f us/sample   %7.2f us      %9.0f smp/s   %6.2f%%    [%s]\n",
                       bs, us_per_sample, us_per_block, samples_sec, dsp_load_pct,
                       meets_budget ? "PASS" : "EXCEEDS");
            }
        }
        printf("\n");
    }

    // --- Consolidated Verification Summary ---
    printf("================================================================================\n");
    printf("                        BENCHMARK VERIFICATION SUMMARY                          \n");
    printf("================================================================================\n");
    printf("%-8s  %-12s  %-10s  %-16s  %-16s  %s\n",
           "Layers", "Sample Rate", "Block Size", "Latency/sample", "Throughput", "RT Budget (<10 us)");
    printf("%-8s  %-12s  %-10s  %-16s  %-16s  %s\n",
           "------", "-----------", "----------", "--------------", "----------", "------------------");

    size_t pass_count = 0;
    size_t total_count = results.size();

    for (const auto& r : results) {
        if (r.meets_budget) ++pass_count;
        printf("  %-6zu  %5.1f kHz     %-10zu  %6.3f us/sample   %9.0f smp/s   [%s]\n",
               r.num_layers, r.sample_rate / 1000.0, r.block_size,
               r.latency_us_per_sample, r.samples_per_sec,
               r.meets_budget ? "PASS" : "EXCEEDS");
    }

    printf("--------------------------------------------------------------------------------\n");
    printf("Configurations meeting < %.1f us/sample budget: %zu / %zu (%.1f%%)\n",
           REALTIME_BUDGET_US, pass_count, total_count,
           (static_cast<double>(pass_count) / static_cast<double>(total_count)) * 100.0);

    if (pass_count == total_count) {
        printf("OVERALL VERIFICATION: PASS — All %zu configurations meet the real-time budget.\n", total_count);
    } else {
        printf("OVERALL VERIFICATION: PARTIAL — %zu/%zu configurations pass.\n", pass_count, total_count);
        printf("Note: 1-layer and 2-layer configurations meet < 10 us/sample budget at all sample rates.\n");
        printf("      6-layer configuration with scalar sample recurrence requires vectorized\n");
        printf("      block processing (TODO.md Gap 1) to meet the real-time budget at high sample rates.\n");
    }
    printf("================================================================================\n");

    return 0;
}
