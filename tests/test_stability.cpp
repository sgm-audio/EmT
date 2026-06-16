#include <gtest/gtest.h>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <memory>
#include <random>
#include "core/ssm_block.h"
#include "core/ssm_inference_core.h"
#include "config.h"

using namespace mamba;

namespace {

void fill_random_weights(SSMWeights& w, std::mt19937& rng) {
    std::uniform_real_distribution<float> small_dist(-0.1f, 0.1f);
    for (size_t out = 0; out < SSMWeights::IN_PROJ_OUT; ++out)
        for (size_t j = 0; j < MAX_D_MODEL; ++j)
            w.in_proj_w[out][j] = small_dist(rng);
    for (size_t h = 0; h < MAX_NHEADS; ++h) {
        w.dt_bias[h] = small_dist(rng);
        for (size_t r = 0; r < MAX_MIMO_RANK; ++r)
            for (size_t s = 0; s < MAX_D_STATE; ++s) {
                w.B_bias[h][r][s] = small_dist(rng);
                w.C_bias[h][r][s] = small_dist(rng);
            }
    }
    for (size_t s = 0; s < MAX_D_STATE; ++s) {
        w.B_norm_weight[s] = 0.5f + small_dist(rng);
        w.C_norm_weight[s] = 0.5f + small_dist(rng);
    }
    for (size_t j = 0; j < MAX_D_MODEL; ++j)
        for (size_t i = 0; i < MAX_D_INNER; ++i)
            w.out_proj_w[j][i] = small_dist(rng);
}

constexpr int64_t kTotalSamples = 44100 * 5;  // 5 seconds for faster test

}  // namespace

TEST(Stability, LongRunNoNaN) {
    std::mt19937 rng(42);
    auto core_ptr = std::make_unique<SSMInferenceCore>();
    MambaConfig cfg{};
    cfg.d_state = 16; cfg.d_model = 16; cfg.d_inner = 32;
    cfg.nheads = 2; cfg.headdim = 16;
    cfg.mimo_rank = MAX_MIMO_RANK; cfg.num_bc_heads = MAX_NUM_BC_HEADS;
    cfg.num_rope_angles = 8; cfg.num_layers = 1; cfg.cond_dim = COND_DIM;
    core_ptr->init(cfg);
    auto& core = *core_ptr;

    SSMWeights w{};
    fill_random_weights(w, rng);
    core.set_layer_weights(0, w);

    std::uniform_real_distribution<float> input_dist(-1.0f, 1.0f);
    float max_abs_output = 0.0f;
    uint64_t nan_count = 0, inf_count = 0;
    constexpr int64_t kCheckInterval = 50000;

    for (int64_t n = 0; n < kTotalSamples; ++n) {
        float input = input_dist(rng);
        alignas(64) float outbuf[MAX_D_MODEL];
        core.process_sample(input, outbuf);
        float out_val = outbuf[0];
        if (std::isnan(out_val)) nan_count++;
        if (std::isinf(out_val)) inf_count++;
        float abs_out = std::fabs(out_val);
        if (abs_out > max_abs_output) max_abs_output = abs_out;
        if (n % kCheckInterval == 0 && n > 0) {
            ASSERT_EQ(nan_count, 0ull) << "NaN at sample " << n;
            ASSERT_EQ(inf_count, 0ull) << "Inf at sample " << n;
            ASSERT_LT(max_abs_output, 1e6f) << "Explosion at sample " << n;
        }
    }
    EXPECT_EQ(nan_count, 0ull);
    EXPECT_EQ(inf_count, 0ull);
    EXPECT_GT(max_abs_output, 0.0f);
}

TEST(Stability, ExtremeParameters) {
    auto core_ptr = std::make_unique<SSMInferenceCore>();
    MambaConfig cfg{};
    cfg.d_state = 16; cfg.d_model = 16; cfg.d_inner = 32;
    cfg.nheads = 2; cfg.headdim = 16;
    cfg.mimo_rank = MAX_MIMO_RANK; cfg.num_bc_heads = MAX_NUM_BC_HEADS;
    cfg.num_rope_angles = 8; cfg.num_layers = 1; cfg.cond_dim = COND_DIM;
    core_ptr->init(cfg);
    auto& core = *core_ptr;

    SSMWeights w{};
    std::memset(&w, 0, sizeof(w));
    for (size_t out = 0; out < SSMWeights::IN_PROJ_OUT; ++out)
        for (size_t j = 0; j < MAX_D_MODEL; ++j)
            w.in_proj_w[out][j] = 10.0f;
    for (size_t h = 0; h < MAX_NHEADS; ++h) w.dt_bias[h] = 10.0f;
    for (size_t s = 0; s < MAX_D_STATE; ++s) {
        w.B_norm_weight[s] = 1.0f;
        w.C_norm_weight[s] = 1.0f;
    }
    for (size_t j = 0; j < MAX_D_MODEL && j < MAX_D_INNER; ++j)
        w.out_proj_w[j][j] = 1.0f;
    core.set_layer_weights(0, w);

    float extremes[] = {0.0f, 1.0f, -1.0f, 100.0f, -100.0f};
    for (float input : extremes) {
        core.reset();
        for (int i = 0; i < 100; ++i) {
            alignas(64) float outbuf[MAX_D_MODEL];
            core.process_sample(input, outbuf);
            EXPECT_FALSE(std::isnan(outbuf[0])) << "NaN with input=" << input;
            EXPECT_FALSE(std::isinf(outbuf[0])) << "Inf with input=" << input;
        }
    }
}

TEST(Stability, ZeroInputStability) {
    auto core_ptr = std::make_unique<SSMInferenceCore>();
    MambaConfig cfg{};
    cfg.d_state = 16; cfg.d_model = 16; cfg.d_inner = 32;
    cfg.nheads = 2; cfg.headdim = 16;
    cfg.mimo_rank = MAX_MIMO_RANK; cfg.num_bc_heads = MAX_NUM_BC_HEADS;
    cfg.num_rope_angles = 8; cfg.num_layers = 1; cfg.cond_dim = COND_DIM;
    core_ptr->init(cfg);
    auto& core = *core_ptr;

    SSMWeights w{};
    std::mt19937 rng(99);
    fill_random_weights(w, rng);
    core.set_layer_weights(0, w);

    for (int64_t n = 0; n < 50000; ++n) {
        alignas(64) float outbuf[MAX_D_MODEL];
        core.process_sample(0.0f, outbuf);
        EXPECT_FALSE(std::isnan(outbuf[0]));
        EXPECT_FALSE(std::isinf(outbuf[0]));
    }
}
