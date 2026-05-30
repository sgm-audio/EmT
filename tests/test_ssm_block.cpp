#include <gtest/gtest.h>
#include "core/ssm_block.h"
#include "core/ssm_inference_core.h"
#include <cstring>
#include <cmath>

using namespace mamba;

TEST(SSMBlock, ResetClearsState) {
SSMBlock block(MAX_D_STATE, MAX_D_MODEL, MAX_D_INNER, MAX_D_CONV);

for (int i = 0; i < 10; ++i) {
block.process_sample(1.0f);
}
block.reset();

float output = block.process_sample(0.0f);
EXPECT_NEAR(output, 0.0f, 1e-4f);
}

TEST(SSMBlock, ProcessBlockProducesOutput) {
SSMBlock block(MAX_D_STATE, MAX_D_MODEL, MAX_D_INNER, MAX_D_CONV);

float input[16] = {1.0f, 0.5f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
float output[16] = {0.0f};

block.process_block(input, output, 16);

for (size_t i = 0; i < 16; ++i) {
EXPECT_FALSE(std::isnan(output[i]));
EXPECT_FALSE(std::isinf(output[i]));
}
}

TEST(SSMBlock, StateDimensions) {
SSMBlock block(16, 16, 32, 4);
EXPECT_EQ(block.get_d_state(), 16);

SSMBlock block2(MAX_D_STATE, MAX_D_MODEL, MAX_D_INNER, MAX_D_CONV);
EXPECT_EQ(block2.get_d_state(), MAX_D_STATE);
}

TEST(SSMBlock, NoLeakAfterMultipleRuns) {
SSMBlock block(MAX_D_STATE, MAX_D_MODEL, MAX_D_INNER, MAX_D_CONV);

for (int iter = 0; iter < 1000; ++iter) {
block.process_sample(0.1f);
}

for (int iter = 0; iter < 1000; ++iter) {
float input[64] = {0.0f};
float output[64] = {0.0f};
block.process_block(input, output, 64);
}

block.reset();
float out = block.process_sample(0.0f);
EXPECT_NEAR(out, 0.0f, 1e-4f);
}

TEST(SSMBlock, DifferentWeightsProduceDifferentOutputs) {
    SSMBlock block1(MAX_D_STATE, MAX_D_MODEL, MAX_D_INNER, MAX_D_CONV);
    SSMBlock block2(MAX_D_STATE, MAX_D_MODEL, MAX_D_INNER, MAX_D_CONV);

    SSMWeights w1{}, w2{};
    std::memset(&w1, 0, sizeof(w1));
    std::memset(&w2, 0, sizeof(w2));

    for (size_t i = 0; i < MAX_D_STATE; ++i) {
        w1.A_log[i] = -1.0f;
        w2.A_log[i] = -1.0f;
        w1.D_proj[i] = 0.5f;
        w2.D_proj[i] = 1.0f;
        w1.dt_bias[i] = 0.5f;
        w2.dt_bias[i] = 0.5f;
        for (size_t j = 0; j < MAX_D_INNER; ++j) {
            w1.dt_proj_w[i][j] = 0.1f;
            w2.dt_proj_w[i][j] = 0.1f;
            w1.B_proj_w[j][i] = 0.1f;
            w2.B_proj_w[j][i] = 0.2f;
            w1.C_proj_w[j][i] = 0.1f;
            w2.C_proj_w[j][i] = 0.1f;
        }
    }
    for (size_t j = 0; j < MAX_D_INNER; ++j) {
        w1.conv1d_b[j] = 0.1f;
        w2.conv1d_b[j] = 0.1f;
        w1.b_in[j] = 0.01f;
        w2.b_in[j] = 0.01f;
        for (size_t k = 0; k < MAX_D_MODEL; ++k) {
            w1.W_in[j][k] = 0.05f;
            w2.W_in[j][k] = 0.05f;
        }
    }
    for (size_t j = 0; j < MAX_D_MODEL; ++j) {
        for (size_t k = 0; k < MAX_D_STATE; ++k) {
            w1.W_ssm[j][k] = (j == k) ? 1.0f : 0.0f;
            w2.W_ssm[j][k] = (j == k) ? 0.5f : 0.0f;
        }
        for (size_t k = 0; k < MAX_D_INNER; ++k) {
            w1.W_out[j][k] = 0.01f;
            w2.W_out[j][k] = 0.02f;
        }
    }

    block1.set_weights(w1);
    block2.set_weights(w2);

    float input = 1.0f;
    float out1 = block1.process_sample(input);
    block2.reset();
    float out2 = block2.process_sample(input);

    EXPECT_NE(out1, out2);
    EXPECT_FALSE(std::isnan(out1));
    EXPECT_FALSE(std::isnan(out2));
}

TEST(SSMBlock, SetConditioningDoesNotCrash) {
SSMBlock block(MAX_D_STATE, MAX_D_MODEL, MAX_D_INNER, MAX_D_CONV);

float cond[COND_DIM] = {0.0f};
cond[0] = 120.0f;
cond[1] = 0.5f;

block.set_conditioning(cond);

float out = block.process_sample(1.0f);
EXPECT_FALSE(std::isnan(out));
EXPECT_FALSE(std::isinf(out));
}

TEST(SSMInferenceCore, MultiLayerProcessing) {
MambaConfig cfg{MAX_D_STATE, MAX_D_MODEL, MAX_D_INNER, MAX_D_CONV, MAX_LAYERS, COND_DIM};
SSMInferenceCore core(cfg);

float out = core.process_sample(1.0f);
EXPECT_FALSE(std::isnan(out));
EXPECT_FALSE(std::isinf(out));
}

TEST(SSMInferenceCore, ResetClearsState) {
MambaConfig cfg{MAX_D_STATE, MAX_D_MODEL, MAX_D_INNER, MAX_D_CONV, 2, COND_DIM};
SSMInferenceCore core(cfg);

for (int i = 0; i < 100; ++i) {
core.process_sample(0.5f);
}

core.reset();
float out = core.process_sample(0.0f);
EXPECT_NEAR(out, 0.0f, 1e-4f);
}

TEST(SSMInferenceCore, GetSetStateRoundTrip) {
MambaConfig cfg{MAX_D_STATE, MAX_D_MODEL, MAX_D_INNER, MAX_D_CONV, 2, COND_DIM};
SSMInferenceCore core1(cfg);
SSMInferenceCore core2(cfg);

for (int i = 0; i < 50; ++i) {
core1.process_sample(0.5f);
}

MambaState state;
core1.get_state(state);
core2.set_state(state);

float out1 = core1.process_sample(0.3f);
float out2 = core2.process_sample(0.3f);
EXPECT_FLOAT_EQ(out1, out2);
}
