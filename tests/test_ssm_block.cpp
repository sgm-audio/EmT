#include <gtest/gtest.h>
#include "core/ssm_block.h"
#include "core/ssm_inference_core.h"
#include <cstring>
#include <cmath>
#include <memory>

using namespace mamba;

// SSMBlock is ~6 MB — too large for stack. Use heap.
static std::unique_ptr<SSMBlock> make_test_block() {
    return std::make_unique<SSMBlock>(
        MAX_D_STATE, MAX_D_MODEL, MAX_D_INNER,
        MAX_NHEADS, MAX_HEADDIM, MAX_MIMO_RANK,
        MAX_NUM_BC_HEADS, COND_DIM);
}

// SSMWeights is ~4 MB — heap-allocate to avoid Debug stack overflow.
static std::unique_ptr<SSMWeights> make_test_weights() {
    auto w = std::make_unique<SSMWeights>();
    std::memset(w.get(), 0, sizeof(*w));
    for (size_t j = 0; j < MAX_D_MODEL && j < MAX_D_INNER; ++j) {
        w->in_proj_w[j][j] = 0.1f;
    }
    for (size_t j = 0; j < MAX_D_MODEL; ++j) {
        w->in_proj_w[MAX_D_INNER + j % MAX_D_INNER][j] = 0.1f;
    }
    for (size_t h = 0; h < MAX_NHEADS; ++h) {
        w->dt_bias[h] = 0.5f;
    }
    for (size_t s = 0; s < MAX_D_STATE; ++s) {
        w->B_norm_weight[s] = 1.0f;
        w->C_norm_weight[s] = 1.0f;
    }
    for (size_t j = 0; j < MAX_D_MODEL && j < MAX_D_INNER; ++j) {
        w->out_proj_w[j][j] = 1.0f;
    }
    return w;
}

TEST(SSMBlock, ResetClearsState) {
    auto block = make_test_block();
    block->set_weights(*make_test_weights());
    alignas(64) float buf[MAX_D_MODEL];
    for (int i = 0; i < 10; ++i) block->process_sample(1.0f, buf);
    block->reset();
    block->process_sample(0.0f, buf);
    EXPECT_NEAR(buf[0], 0.0f, 1e-3f);
}

TEST(SSMBlock, ProcessBlockProducesOutput) {
    auto block = make_test_block();
    block->set_weights(*make_test_weights());
    constexpr size_t N = 16;
    float input[N] = {1.0f, 0.5f};
    float output[N * MAX_D_MODEL] = {0.0f};
    block->process_block(input, output, N);
    for (size_t i = 0; i < N * MAX_D_MODEL; ++i) {
        EXPECT_FALSE(std::isnan(output[i]));
        EXPECT_FALSE(std::isinf(output[i]));
    }
}

TEST(SSMBlock, StateDimensions) {
    auto block = make_test_block();
    EXPECT_GE(block->get_d_state(), 1u);
    EXPECT_GE(block->get_nheads(), 1u);
}

TEST(SSMBlock, NoLeakAfterMultipleRuns) {
    auto block = make_test_block();
    block->set_weights(*make_test_weights());
    alignas(64) float buf[MAX_D_MODEL];
    for (int iter = 0; iter < 100; ++iter) block->process_sample(0.1f, buf);
    block->reset();
    block->process_sample(0.0f, buf);
    EXPECT_NEAR(buf[0], 0.0f, 1e-3f);
}

TEST(SSMBlock, DifferentWeightsProduceDifferentOutputs) {
    auto block1 = make_test_block();
    auto block2 = make_test_block();
    auto w1 = make_test_weights();
    auto w2 = make_test_weights();
    // Make the recurrence live (non-zero B/C) so dt_bias actually affects output.
    w1->B_bias[0][0][0] = 0.5f;
    w1->C_bias[0][0][0] = 0.5f;
    w2->B_bias[0][0][0] = 0.5f;
    w2->C_bias[0][0][0] = 0.5f;
    w2->dt_bias[0] = 10.0f;  // dramatically different dt
    block1->set_weights(*w1);
    block2->set_weights(*w2);
    alignas(64) float buf1[MAX_D_MODEL];
    alignas(64) float buf2[MAX_D_MODEL];
    block1->process_sample(1.0f, buf1);
    block2->process_sample(1.0f, buf2);
    EXPECT_FALSE(std::isnan(buf1[0]));
    EXPECT_FALSE(std::isnan(buf2[0]));
    EXPECT_GT(std::fabs(buf1[0] - buf2[0]), 1e-5f);
}

TEST(SSMBlock, SetConditioningDoesNotCrash) {
    auto block = make_test_block();
    block->set_weights(*make_test_weights());
    float cond[COND_DIM] = {0.0f};
    cond[0] = 120.0f; cond[1] = 0.5f;
    block->set_conditioning(cond);
    alignas(64) float outbuf[MAX_D_MODEL];
    block->process_sample(1.0f, outbuf);
    EXPECT_FALSE(std::isnan(outbuf[0]));
    EXPECT_FALSE(std::isinf(outbuf[0]));
}

// SSMInferenceCore is ~36 MB — too large for stack.
TEST(SSMInferenceCore, MultiLayerProcessing) {
    auto core = std::make_unique<SSMInferenceCore>();
    MambaConfig cfg = MambaConfig::max_config();
    cfg.num_layers = MAX_LAYERS;
    core->init(cfg);
    auto w = make_test_weights();
    w->B_bias[0][0][0] = 0.5f;
    w->C_bias[0][0][0] = 0.5f;
    SSMBlockParams params{};
    params.gain = 1.0f / static_cast<float>(MAX_D_MODEL);
    for (uint32_t layer = 0; layer < MAX_LAYERS; ++layer) {
        core->set_layer_weights(layer, *w);
        core->set_layer_params(layer, params);
    }
    alignas(64) float outbuf[MAX_D_MODEL];
    for (int i = 0; i < 64; ++i) {
        core->process_sample((i == 0) ? 1.0f : 0.25f, outbuf);
        ASSERT_FALSE(std::isnan(outbuf[0]));
        ASSERT_FALSE(std::isinf(outbuf[0]));
    }
    EXPECT_NE(outbuf[0], 0.0f);
}

TEST(SSMInferenceCore, ResetClearsState) {
    auto core = std::make_unique<SSMInferenceCore>();
    MambaConfig cfg = MambaConfig::max_config();
    cfg.num_layers = 2;
    core->init(cfg);
    for (uint32_t layer = 0; layer < 2; ++layer)
        core->set_layer_weights(layer, *make_test_weights());
    alignas(64) float rbuf[MAX_D_MODEL];
    for (int i = 0; i < 100; ++i) core->process_sample(0.5f, rbuf);
    core->reset();
    core->process_sample(0.0f, rbuf);
    EXPECT_NEAR(rbuf[0], 0.0f, 1e-3f);
}

TEST(SSMInferenceCore, GetSetStateRoundTrip) {
    auto core1 = std::make_unique<SSMInferenceCore>();
    auto core2 = std::make_unique<SSMInferenceCore>();
    MambaConfig cfg = MambaConfig::max_config();
    cfg.num_layers = 2;
    core1->init(cfg);
    core2->init(cfg);
    for (uint32_t layer = 0; layer < 2; ++layer) {
        core1->set_layer_weights(layer, *make_test_weights());
        core2->set_layer_weights(layer, *make_test_weights());
    }
    alignas(64) float rbuf[MAX_D_MODEL];
    for (int i = 0; i < 50; ++i) core1->process_sample(0.5f, rbuf);
    MambaState state;
    core1->get_state(state);
    core2->set_state(state);
    alignas(64) float buf1[MAX_D_MODEL], buf2[MAX_D_MODEL];
    core1->process_sample(0.3f, buf1);
    core2->process_sample(0.3f, buf2);
    EXPECT_FLOAT_EQ(buf1[0], buf2[0]);
}

TEST(SSMInferenceCore, ProcessSampleExplicitStateParity) {
    auto core1 = std::make_unique<SSMInferenceCore>();
    auto core2 = std::make_unique<SSMInferenceCore>();
    MambaConfig cfg = MambaConfig::max_config();
    cfg.num_layers = 2;
    core1->init(cfg);
    core2->init(cfg);
    for (uint32_t layer = 0; layer < 2; ++layer) {
        core1->set_layer_weights(layer, *make_test_weights());
        core2->set_layer_weights(layer, *make_test_weights());
    }
    auto explicit_state = std::make_unique<MambaState>();
    std::memset(explicit_state.get(), 0, sizeof(MambaState));
    alignas(64) float buf1[MAX_D_MODEL], buf2[MAX_D_MODEL];
    for (int i = 0; i < 50; ++i) {
        float in = 0.1f * static_cast<float>(i);
        core1->process_sample(in, buf1);
        core2->process_sample(in, buf2, *explicit_state);
        EXPECT_FLOAT_EQ(buf1[0], buf2[0]);
    }
}

TEST(SSMInferenceCore, LargeBlockChunkingMatchesPerSample) {
    auto core1 = std::make_unique<SSMInferenceCore>();
    auto core2 = std::make_unique<SSMInferenceCore>();
    MambaConfig cfg{};
    cfg.d_state = 16;
    cfg.d_model = 2;
    cfg.d_inner = 16;
    cfg.nheads = 2;
    cfg.headdim = 8;
    cfg.mimo_rank = 1;
    cfg.num_bc_heads = 1;
    cfg.num_rope_angles = 8;
    cfg.num_layers = 2;
    cfg.cond_dim = 4;
    core1->init(cfg);
    core2->init(cfg);
    auto w = make_test_weights();
    w->dt_film_proj_w[0][0] = 0.25f;
    w->dt_film_proj_w[2][1] = -0.5f;
    for (uint32_t layer = 0; layer < cfg.num_layers; ++layer) {
        core1->set_layer_weights(layer, *w);
        core2->set_layer_weights(layer, *w);
    }

    constexpr uint32_t N = MAX_BLOCK_SIZE + 317;
    std::vector<float> in(N);
    for (uint32_t i = 0; i < N; ++i) {
        in[i] = 0.05f * std::sin(0.01f * static_cast<float>(i));
    }
    std::vector<float> out_block(N * cfg.d_model, -99.0f);
    std::vector<float> out_sample(N * cfg.d_model, -99.0f);

    core1->process_block(in.data(), out_block.data(), N);
    for (uint32_t i = 0; i < N; ++i) {
        core2->process_sample(in[i], out_sample.data() + i * cfg.d_model);
    }
    for (uint32_t i = 0; i < N; ++i) {
        EXPECT_FLOAT_EQ(out_block[i * cfg.d_model], out_sample[i * cfg.d_model]);
        EXPECT_FLOAT_EQ(out_block[i * cfg.d_model + 1], out_sample[i * cfg.d_model + 1]);
    }
}

TEST(SSMInferenceCore, ZeroLayersZeroesExtraChannels) {
    auto core = std::make_unique<SSMInferenceCore>();
    MambaConfig cfg{};
    cfg.d_state = 16;
    cfg.d_model = 4;
    cfg.d_inner = 16;
    cfg.nheads = 2;
    cfg.headdim = 8;
    cfg.mimo_rank = 1;
    cfg.num_bc_heads = 1;
    cfg.num_rope_angles = 8;
    cfg.num_layers = 0;
    cfg.cond_dim = 4;
    core->init(cfg);
    float out[MAX_D_MODEL] = {9.0f, 9.0f, 9.0f, 9.0f};
    core->process_sample(0.75f, out);
    EXPECT_FLOAT_EQ(out[0], 0.75f);
    EXPECT_FLOAT_EQ(out[1], 0.0f);
    EXPECT_FLOAT_EQ(out[2], 0.0f);
    EXPECT_FLOAT_EQ(out[3], 0.0f);
}

TEST(SSMBlockTest, SetWeightsDtFilmCompactStrideCondDim) {
    // Regression test: SSMBlock::set_weights copies 2 * nheads_ rows of
    // dt_film_proj_w strided by COND_DIM into compact weights_.dt_film_proj_w
    // (row h*2+p at (h*2+p) * d_cond_), matching WeightLoader compact layout.
    SSMBlock block_small_cond(16, 1, 16, 2, 8, 1, 1, 4);
    SSMBlock block_ref_cond(16, 1, 16, 2, 8, 1, 1, 4);

    auto w_small = make_test_weights();
    w_small->dt_film_proj_w[0][0] = 1.25f;
    w_small->dt_film_proj_w[1][3] = -0.5f;
    w_small->dt_film_proj_w[2][1] = 0.25f;
    w_small->dt_film_proj_w[3][2] = 0.75f;
    // Deliberately write non-zero garbage past d_cond=4; it must NOT leak into row 1+!
    w_small->dt_film_proj_w[0][10] = 99.0f;
    w_small->dt_film_proj_w[1][10] = -99.0f;

    auto w_clean = make_test_weights();
    w_clean->dt_film_proj_w[0][0] = 1.25f;
    w_clean->dt_film_proj_w[1][3] = -0.5f;
    w_clean->dt_film_proj_w[2][1] = 0.25f;
    w_clean->dt_film_proj_w[3][2] = 0.75f;

    block_small_cond.set_weights(*w_small);
    block_ref_cond.set_weights(*w_clean);

    alignas(64) float cond[4] = {1.0f, 0.25f, 0.5f, -1.0f};
    block_small_cond.set_conditioning(cond);
    block_ref_cond.set_conditioning(cond);

    alignas(64) float out1[MAX_D_MODEL]{}, out2[MAX_D_MODEL]{};
    auto s1 = std::make_unique<SSMState>();
    auto s2 = std::make_unique<SSMState>();
    std::memset(s1.get(), 0, sizeof(SSMState));
    std::memset(s2.get(), 0, sizeof(SSMState));
    block_small_cond.process_sample(0.4f, out1, *s1);
    block_ref_cond.process_sample(0.4f, out2, *s2);
    EXPECT_FLOAT_EQ(out1[0], out2[0]);
}

TEST(SSMBlock, StrictCausalityPrefixInvariance) {
    MambaConfig cfg{};
    cfg.d_state = 16;
    cfg.d_model = 4;
    cfg.d_inner = 16;
    cfg.nheads = 2;
    cfg.headdim = 8;
    cfg.mimo_rank = 1;
    cfg.num_bc_heads = 1;
    cfg.num_rope_angles = 8;
    cfg.num_layers = 2;
    cfg.cond_dim = 4;

    auto core_a = std::make_unique<SSMInferenceCore>(cfg);
    auto core_b = std::make_unique<SSMInferenceCore>(cfg);
    auto w = make_test_weights();
    w->B_bias[0][0][0] = 0.5f;
    w->C_bias[0][0][0] = 0.5f;
    for (uint32_t l = 0; l < cfg.num_layers; ++l) {
        core_a->set_layer_weights(l, *w);
        core_b->set_layer_weights(l, *w);
    }

    constexpr size_t N = 64;
    constexpr size_t kSplit = 23;
    std::vector<float> in_a(N), in_b(N);
    for (size_t i = 0; i < N; ++i) {
        in_a[i] = std::sin(0.15f * static_cast<float>(i));
        in_b[i] = (i < kSplit) ? in_a[i] : std::numeric_limits<float>::quiet_NaN();
    }

    std::vector<float> out_a(N * cfg.d_model, 0.0f), out_b(N * cfg.d_model, 0.0f);
    core_a->process_block(in_a.data(), out_a.data(), N);
    core_b->process_block(in_b.data(), out_b.data(), N);

    // Prefix [0..kSplit-1] must be bit-identical regardless of future NaNs.
    ASSERT_EQ(std::memcmp(out_a.data(), out_b.data(), kSplit * cfg.d_model * sizeof(float)), 0);
}

TEST(SSMBlock, ZeroDeltaFreezesStateAndLargeDeltaBounded) {
    // 1. dt_bias = -50 -> softplus(dt_pre) == 0 -> A_bar == 1, B_bar == 0 (state freezes).
    {
        SSMBlock block(16, 4, 16, 2, 8, 1, 1, 4);
        auto w = make_test_weights();
        for (size_t h = 0; h < 2; ++h) w->dt_bias[h] = -50.0f;
        w->B_bias[0][0][0] = 1.0f;
        w->C_bias[0][0][0] = 1.0f;
        block.set_weights(*w);
        block.state().ssm_real[0][0][0] = 2.5f;

        alignas(64) float out[MAX_D_MODEL]{};
        for (int i = 0; i < 32; ++i) {
            block.process_sample(1.0f, out);
        }
        EXPECT_FLOAT_EQ(block.state().ssm_real[0][0][0], 2.5f);
    }

    // 2. dt_bias = +50 -> linear softplus regime (DT_t ~ 50), A_t <= -1 -> |a_bar| < 1, bounded.
    {
        SSMBlock block(16, 4, 16, 2, 8, 1, 1, 4);
        auto w = make_test_weights();
        for (size_t h = 0; h < 2; ++h) w->dt_bias[h] = 50.0f;
        w->B_bias[0][0][0] = 0.5f;
        w->C_bias[0][0][0] = 0.5f;
        block.set_weights(*w);

        alignas(64) float out[MAX_D_MODEL]{};
        for (int i = 0; i < 10000; ++i) {
            block.process_sample((i % 2 == 0) ? 1.0f : -1.0f, out);
            ASSERT_TRUE(std::isfinite(out[0])) << "sample " << i;
            ASSERT_LT(std::fabs(out[0]), 1e3f) << "sample " << i;
        }
    }
}

TEST(SSMBlock, RmsNormZeroInputAndRopeLongHorizon) {
    SSMBlock block(16, 4, 16, 2, 8, 1, 1, 4);
    auto w = make_test_weights();
    // Leave B/C in_proj rows at 0 so B_ptr == 0 and C_ptr == 0; set non-zero biases and RoPE angle.
    w->B_bias[0][0][0] = 0.25f;
    w->C_bias[0][0][0] = 0.5f;
    const size_t angle_row = 16 * 2 + 2 * MAX_MIMO_RANK * 16 * 2 + 2 * 3;
    w->in_proj_w[angle_row][0] = 1.5f;
    block.set_weights(*w);

    alignas(64) float out[MAX_D_MODEL]{};
    for (int i = 0; i < 100000; ++i) {
        block.process_sample(0.5f, out);
        if ((i & 4095) == 0) {
            ASSERT_TRUE(std::isfinite(out[0])) << "sample " << i;
        }
    }
    constexpr float kTwoPi = 2.0f * 3.14159265358979323846f;
    for (size_t h = 0; h < 2; ++h) {
        for (size_t a = 0; a < 8; ++a) {
            const float ang = block.state().angle_state[h][a];
            ASSERT_TRUE(std::isfinite(ang));
            EXPECT_LT(std::fabs(ang), kTwoPi + 1e-5f);
        }
    }
}

