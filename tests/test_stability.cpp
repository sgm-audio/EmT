#include <gtest/gtest.h>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <random>
#include "core/ssm_block.h"
#include "core/ssm_inference_core.h"
#include "common/ring_buffer.h"
#include "common/conditioning_frame.h"
#include "common/seqlock.h"
#include "config.h"

using namespace mamba;

namespace {

void fill_random_weights(SSMWeights& w, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::uniform_real_distribution<float> small_dist(0.0f, 0.1f);
    std::uniform_real_distribution<float> log_dist(-3.0f, 0.0f);

    for (size_t i = 0; i < MAX_D_STATE; ++i) {
        w.A_log[i] = log_dist(rng);
        w.D_proj[i] = small_dist(rng);
        w.dt_bias[i] = small_dist(rng);
        for (size_t j = 0; j < MAX_D_INNER; ++j) {
            w.dt_proj_w[i][j] = small_dist(rng);
            w.B_proj_w[j][i] = small_dist(rng);
            w.C_proj_w[j][i] = small_dist(rng);
        }
    }
    for (size_t j = 0; j < MAX_D_INNER; ++j) {
        w.conv1d_b[j] = small_dist(rng);
        w.b_in[j] = small_dist(rng);
        for (size_t k = 0; k < MAX_D_CONV; ++k) {
            w.conv1d_w[k][j] = small_dist(rng);
        }
        for (size_t k = 0; k < MAX_D_MODEL; ++k) {
            w.W_in[j][k] = small_dist(rng);
        }
    }
    for (size_t j = 0; j < MAX_D_MODEL; ++j) {
        for (size_t k = 0; k < MAX_D_INNER; ++k) {
            w.W_out[j][k] = small_dist(rng);
        }
        for (size_t k = 0; k < MAX_D_STATE; ++k) {
            w.W_ssm[j][k] = (j == k) ? 1.0f : small_dist(rng);
        }
    }
    for (size_t i = 0; i < MAX_D_STATE; ++i) {
        for (size_t c = 0; c < COND_DIM; ++c) {
            w.cond_proj_dt[i][c] = small_dist(rng);
            w.cond_proj_B[i][c] = small_dist(rng);
        }
        w.cond_dt_bias[i] = small_dist(rng);
        w.cond_B_bias[i] = small_dist(rng);
    }
}

constexpr int64_t kTotalSamples = 44100 * 300;

}

TEST(Stability, LongRunNoNaN) {
    std::mt19937 rng(42);

    MambaConfig cfg{MAX_D_STATE, MAX_D_MODEL, MAX_D_INNER, MAX_D_CONV, MAX_LAYERS, COND_DIM};
    SSMInferenceCore core(cfg);

    for (uint32_t layer = 0; layer < static_cast<uint32_t>(MAX_LAYERS); ++layer) {
        SSMWeights w{};
        fill_random_weights(w, rng);
        core.set_layer_weights(layer, w);
    }

    std::uniform_real_distribution<float> input_dist(-1.0f, 1.0f);
    std::uniform_int_distribution<int> bypass_dist(0, 1);
    std::uniform_real_distribution<float> param_dist(0.0f, 1.0f);

    float max_abs_output = 0.0f;
    uint64_t nan_count = 0;
    uint64_t inf_count = 0;
    uint64_t bypass_toggle_count = 0;

    float cond[COND_DIM] = {0.0f};
    for (size_t c = 0; c < COND_DIM; ++c) {
        cond[c] = param_dist(rng);
    }
    core.set_conditioning(cond);

    constexpr int64_t kCheckInterval = 100000;
    constexpr int64_t kBypassInterval = 10000;
    constexpr int64_t kCondInterval = 50000;

    for (int64_t n = 0; n < kTotalSamples; ++n) {
        float input = input_dist(rng);

        if (n % kBypassInterval == 0 && n > 0) {
            bypass_toggle_count++;
        }

        if (n % kCondInterval == 0 && n > 0) {
            for (size_t c = 0; c < COND_DIM; ++c) {
                cond[c] = param_dist(rng);
            }
            core.set_conditioning(cond);
        }

        float out = core.process_sample(input);

        if (std::isnan(out)) {
            nan_count++;
        }
        if (std::isinf(out)) {
            inf_count++;
        }
        float abs_out = std::fabs(out);
        if (abs_out > max_abs_output) {
            max_abs_output = abs_out;
        }

        if (n % kCheckInterval == 0 && n > 0) {
            ASSERT_EQ(nan_count, 0ull)
                << "NaN encountered at sample " << n;
            ASSERT_EQ(inf_count, 0ull)
                << "Inf encountered at sample " << n;
            ASSERT_LT(max_abs_output, 1e6f)
                << "State explosion detected at sample " << n;
        }
    }

    EXPECT_EQ(nan_count, 0ull);
    EXPECT_EQ(inf_count, 0ull);
    EXPECT_GT(max_abs_output, 0.0f);
    EXPECT_LT(max_abs_output, 100.0f)
        << "Output magnitude exceeded 100.0 after long run";

    MambaState final_state;
    core.get_state(final_state);
    for (uint32_t layer = 0; layer < static_cast<uint32_t>(MAX_LAYERS); ++layer) {
        for (size_t i = 0; i < MAX_D_STATE; ++i) {
            float v = final_state.layers[layer].h[i];
            EXPECT_FALSE(std::isnan(v)) << "h_state NaN in layer " << layer << "[" << i << "]";
            EXPECT_FALSE(std::isinf(v)) << "h_state Inf in layer " << layer << "[" << i << "]";
            EXPECT_GT(v, -100.0f) << "h_state too negative in layer " << layer << "[" << i << "]";
            EXPECT_LT(v, 100.0f)  << "h_state too positive in layer " << layer << "[" << i << "]";
        }
    }
}

TEST(Stability, ExtremeParameters) {
    std::mt19937 rng(123);

    MambaConfig cfg{MAX_D_STATE, MAX_D_MODEL, MAX_D_INNER, MAX_D_CONV, 1, COND_DIM};
    SSMInferenceCore core(cfg);

    SSMWeights w{};
    w.A_log[0] = 0.0f;
    w.D_proj[0] = 0.0f;
    w.dt_bias[0] = 10.0f;
    for (size_t j = 0; j < MAX_D_INNER; ++j) {
        w.dt_proj_w[0][j] = 10.0f;
        w.B_proj_w[j][0] = 10.0f;
        w.C_proj_w[j][0] = 10.0f;
        w.conv1d_b[j] = 10.0f;
        w.conv1d_w[0][j] = 10.0f;
        w.b_in[j] = 10.0f;
        for (size_t k = 0; k < MAX_D_MODEL; ++k) {
            w.W_in[j][k] = 10.0f;
            w.W_out[k][j] = 10.0f;
        }
    }
    for (size_t j = 0; j < MAX_D_MODEL; ++j) {
        for (size_t k = 0; k < MAX_D_STATE; ++k) {
            w.W_ssm[j][k] = 10.0f;
        }
    }
    core.set_layer_weights(0, w);

    float extremes[] = {0.0f, 1.0f, -1.0f, 100.0f, -100.0f, 1e6f, -1e6f};
    for (float input : extremes) {
        core.reset();
        for (int i = 0; i < 100; ++i) {
            float out = core.process_sample(input);
            EXPECT_FALSE(std::isnan(out)) << "NaN with input=" << input;
            EXPECT_FALSE(std::isinf(out)) << "Inf with input=" << input;
        }
    }

    float dc_out = 0.0f;
    core.reset();
    for (int i = 0; i < 1000; ++i) {
        dc_out = core.process_sample(1.0f);
        EXPECT_FALSE(std::isnan(dc_out));
        EXPECT_FALSE(std::isinf(dc_out));
    }
}

TEST(Stability, ZeroInputStability) {
    MambaConfig cfg{MAX_D_STATE, MAX_D_MODEL, MAX_D_INNER, MAX_D_CONV, MAX_LAYERS, COND_DIM};
    SSMInferenceCore core(cfg);
    SSMWeights w{};
    std::mt19937 rng(99);
    fill_random_weights(w, rng);
    for (uint32_t layer = 0; layer < static_cast<uint32_t>(MAX_LAYERS); ++layer) {
        core.set_layer_weights(layer, w);
    }

    for (int64_t n = 0; n < 50000; ++n) {
        float out = core.process_sample(0.0f);
        EXPECT_FALSE(std::isnan(out));
        EXPECT_FALSE(std::isinf(out));
    }
}