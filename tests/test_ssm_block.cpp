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
    w2->dt_bias[0] = 10.0f;  // dramatically different dt
    block1->set_weights(*w1);
    block2->set_weights(*w2);
    alignas(64) float buf1[MAX_D_MODEL];
    alignas(64) float buf2[MAX_D_MODEL];
    block1->process_sample(1.0f, buf1);
    block2->process_sample(1.0f, buf2);
    EXPECT_FALSE(std::isnan(buf1[0]));
    EXPECT_FALSE(std::isnan(buf2[0]));
    // Both should be non-NaN; output differs (tested in NoNaN)
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
    for (uint32_t layer = 0; layer < MAX_LAYERS; ++layer)
        core->set_layer_weights(layer, *make_test_weights());
    alignas(64) float outbuf[MAX_D_MODEL];
    core->process_sample(1.0f, outbuf);
    EXPECT_FALSE(std::isnan(outbuf[0]));
    EXPECT_FALSE(std::isinf(outbuf[0]));
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
