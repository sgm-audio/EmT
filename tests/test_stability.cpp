#include <gtest/gtest.h>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <memory>
#include <random>
#include <vector>
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

// ---------------------------------------------------------------------------
// Deviation D5 regression fixture
// ---------------------------------------------------------------------------
//
// Builds the constructed case from ARCHITECTURE_FIDELITY.md D5: the weights are
// set so the recurrence reaches a steady state of |h| = 4 (A = -1, dt = 1,
// B = 4, C = 1), which is exactly the regime in which the old periodic
// renormalization divided the state by max|h| every 4096 samples and stepped
// the output gain by 14.8 %.
//
// Derivation (constant input x = 1):
//   dt = softplus(0.5413) = 1.0
//   A  = -softplus(0.5413) = -1.0  (clamped at -1)
//   half = A*dt/2 = -0.5
//   A_bar = (1 + half)/(1 - half)   = 1/3
//   B_bar = dt/(1 - half)           = 2/3
//   h_ss  = B_bar * B * x / (1 - A_bar) = (2/3 * 4)/(2/3) = 4
constexpr float kLogE1m1 = 0.54132485f;  // softplus^-1(1.0) = log(e - 1)

void fill_steady_state_h4_weights(SSMWeights& w, size_t d_model, size_t d_inner,
                                  size_t nheads, size_t headdim, size_t d_state,
                                  size_t mimo_rank) {
    std::memset(&w, 0, sizeof(w));

    // Every u_film[j] equals the (scalar) input, so a row whose entries sum to
    // `target` produces in_proj output `target * input`.
    auto set_row_value = [&](size_t r, float target) {
        for (size_t j = 0; j < d_model; ++j) {
            w.in_proj_w[r][j] = target / static_cast<float>(d_model);
        }
    };

    // in_proj rows: [z | x | B | C | dd_dt | dd_A | trap | angles]
    for (size_t d = 0; d < d_inner; ++d) {
        set_row_value(d, 1.0f);  // z: constant 1.0 (silu scale)
    }
    size_t row = d_inner;
    for (size_t d = 0; d < d_inner; ++d, ++row) {
        set_row_value(row, 1.0f);  // x: 1.0
    }
    const size_t bc_size = nheads * mimo_rank * d_state;
    for (size_t i = 0; i < bc_size; ++i, ++row) {
        set_row_value(row, 1.0f);  // B: 1.0 (RMSNorm -> 1.0, + B_bias)
    }
    for (size_t i = 0; i < bc_size; ++i, ++row) {
        set_row_value(row, 1.0f);  // C: 1.0
    }
    for (size_t h = 0; h < nheads; ++h, ++row) {
        set_row_value(row, kLogE1m1);  // dd_dt: dt = 1.0
    }
    for (size_t h = 0; h < nheads; ++h, ++row) {
        set_row_value(row, kLogE1m1);  // dd_A: A = -1.0
    }
    // trap and angle rows stay zero.

    for (size_t h = 0; h < nheads; ++h) {
        for (size_t r = 0; r < mimo_rank; ++r) {
            for (size_t s = 0; s < d_state; ++s) {
                w.B_bias[h][r][s] = 3.0f;  // B_norm = 1.0 + 3.0 = 4.0
                w.C_bias[h][r][s] = 0.0f;  // C_norm = 1.0
            }
        }
    }
    for (size_t s = 0; s < d_state; ++s) {
        w.B_norm_weight[s] = 1.0f;
        w.C_norm_weight[s] = 1.0f;
    }
    for (size_t j = 0; j < d_model && j < d_inner; ++j) {
        w.out_proj_w[j][j] = 1.0f;
    }
}

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

    auto w = std::make_unique<SSMWeights>();
    fill_random_weights(*w, rng);
    core.set_layer_weights(0, *w);

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

    auto w = std::make_unique<SSMWeights>();
    std::memset(w.get(), 0, sizeof(*w));
    for (size_t out = 0; out < SSMWeights::IN_PROJ_OUT; ++out)
        for (size_t j = 0; j < MAX_D_MODEL; ++j)
            w->in_proj_w[out][j] = 10.0f;
    for (size_t h = 0; h < MAX_NHEADS; ++h) w->dt_bias[h] = 10.0f;
    for (size_t s = 0; s < MAX_D_STATE; ++s) {
        w->B_norm_weight[s] = 1.0f;
        w->C_norm_weight[s] = 1.0f;
    }
    for (size_t j = 0; j < MAX_D_MODEL && j < MAX_D_INNER; ++j)
        w->out_proj_w[j][j] = 1.0f;
    core.set_layer_weights(0, *w);

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

    auto w = std::make_unique<SSMWeights>();
    std::mt19937 rng(99);
    fill_random_weights(*w, rng);
    core.set_layer_weights(0, *w);

    for (int64_t n = 0; n < 50000; ++n) {
        alignas(64) float outbuf[MAX_D_MODEL];
        core.process_sample(0.0f, outbuf);
        EXPECT_FALSE(std::isnan(outbuf[0]));
        EXPECT_FALSE(std::isinf(outbuf[0]));
    }
}

// ---------------------------------------------------------------------------
// Deviation D5 regression
// ---------------------------------------------------------------------------

TEST(Stability, NoRenormalizationGainStepAtFormerIntervalBoundary) {
    constexpr size_t DS = 16, DM = 16, DI = 32, NH = 2, HD = 16, MR = 1;
    auto block = std::make_unique<SSMBlock>(DS, DM, DI, NH, HD, MR,
                                            MAX_NUM_BC_HEADS, COND_DIM);
    SSMWeights w;
    fill_steady_state_h4_weights(w, DM, DI, NH, HD, DS, MR);
    block->set_weights(w);

    // Two full old intervals plus headroom.
    constexpr int64_t kSamples = 9000;
    std::vector<float> y(kSamples);
    for (int64_t n = 0; n < kSamples; ++n) {
        alignas(64) float outbuf[MAX_D_MODEL];
        block->process_sample(1.0f, outbuf);  // constant drive -> steady state
        ASSERT_FALSE(std::isnan(outbuf[0])) << "NaN at sample " << n;
        y[static_cast<size_t>(n)] = outbuf[0];
    }

    // The constructed model must actually exceed unity in the state, otherwise
    // this test would pass vacuously.
    float max_h = 0.0f;
    const auto& st = block->state();
    for (size_t h = 0; h < NH; ++h)
        for (size_t d = 0; d < HD; ++d)
            for (size_t s = 0; s < DS; ++s)
                max_h = std::max(max_h, std::fabs(st.ssm_real[h][d][s]));
    ASSERT_GT(max_h, 1.0f) << "constructed model did not reach |h| > 1";
    ASSERT_LT(max_h, 100.0f) << "state is not bounded without the guard";

    // Largest sample-to-sample output change inside the old boundary windows.
    const float y_settled = y[kSamples - 1];
    ASSERT_GT(std::fabs(y_settled), 1e-3f);
    float boundary_step = 0.0f;
    for (int64_t b : {4096, 8192}) {
        for (int64_t n = b - 2; n <= b + 4; ++n) {
            boundary_step = std::max(boundary_step,
                                     std::fabs(y[static_cast<size_t>(n)] -
                                               y[static_cast<size_t>(n - 1)]));
        }
    }
    // Old behaviour: ~14.8 % of the settled level.  Requirement: no step beyond
    // 1 % (the trajectory is smooth and converged there).
    EXPECT_LT(boundary_step, 0.01f * std::fabs(y_settled))
        << "output steps at the former renormalization boundary: "
        << boundary_step << " vs settled " << y_settled;
}

TEST(Stability, HighStateLongRunStaysBounded) {
    constexpr size_t DS = 16, DM = 16, DI = 32, NH = 2, HD = 16, MR = 1;
    auto block = std::make_unique<SSMBlock>(DS, DM, DI, NH, HD, MR,
                                            MAX_NUM_BC_HEADS, COND_DIM);
    SSMWeights w;
    fill_steady_state_h4_weights(w, DM, DI, NH, HD, DS, MR);
    block->set_weights(w);

    // 16x the old renormalization interval with a state that stays above unity:
    // boundedness must come from the A-stable discretization, not from a guard.
    constexpr int64_t kSamples = 65536;
    for (int64_t n = 0; n < kSamples; ++n) {
        alignas(64) float outbuf[MAX_D_MODEL];
        block->process_sample((n % 2 == 0) ? 1.0f : 0.5f, outbuf);
        ASSERT_FALSE(std::isnan(outbuf[0])) << "NaN at sample " << n;
        ASSERT_FALSE(std::isinf(outbuf[0])) << "Inf at sample " << n;
        ASSERT_LT(std::fabs(outbuf[0]), 1e3f) << "divergence at sample " << n;
    }
}
