// tests/test_recurrence_kernel.cpp
//
// Pins for the D1 + D2 recurrence (src/core/recurrence_kernel.h).
//
// §a lifted the scalar update out of step_recurrence; §b replaced the two-term
// bilinear form with the paper's exponential-trapezoidal three-term recurrence
// (D1) and made the complex state live as per-element 2×2 rotation blocks (D2).
// These tests pin:
//   1. the coefficients and the element arithmetic are the same expressions the
//      NumPy reference implements (bit-exact for the same inputs),
//   2. the history term actually changes the impulse response by the analytic
//      amount (D1 is live, not dead code),
//   3. the rotation is a unit-norm transform that oscillates at the expected
//      frequency and never blows up (D2 stability), and
//   4. a reset clears the history so a reset block is bit-identical to a fresh
//      one.

#include "core/recurrence_kernel.h"
#include "core/ssm_block.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <cmath>
#include <memory>
#include <vector>

namespace {

uint32_t as_u32(float v) {
    uint32_t u = 0;
    std::memcpy(&u, &v, sizeof(u));
    return u;
}

// SSMBlock is ~6 MB and SSMWeights ~4 MB — heap-allocate (mirrors test_ssm_block).
std::unique_ptr<mamba::SSMBlock> make_test_block() {
    return std::make_unique<mamba::SSMBlock>(
        MAX_D_STATE, MAX_D_MODEL, MAX_D_INNER,
        MAX_NHEADS, MAX_HEADDIM, MAX_MIMO_RANK,
        MAX_NUM_BC_HEADS, COND_DIM);
}

std::unique_ptr<mamba::SSMWeights> make_test_weights() {
    auto w = std::make_unique<mamba::SSMWeights>();
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

} // namespace

// --- D1: coefficients -----------------------------------------------------

TEST(RecurrenceKernel, TrapezoidalCoeffsMatchFormula) {
    const float poles[] = {-8.0f, -3.5f, -1.0f, -0.25f, -0.001f};
    const float steps[] = {1.0e-4f, 0.001f, 0.01f, 0.1f, 0.5f, 1.0f, 2.0f};

    for (float A : poles) {
        for (float dt : steps) {
            const float half = A * dt * 0.5f;
            const float inv = 1.0f / (1.0f - half);
            const float expect_alpha = (1.0f + half) * inv;
            const float expect_beta = dt * 0.5f * inv;
            const float expect_gamma = dt * 0.5f * inv;

            const mamba::recurrence::RecurrenceCoeffs k =
                mamba::recurrence::trapezoidal_coeffs(A, dt);

            EXPECT_EQ(as_u32(k.alpha), as_u32(expect_alpha)) << "A=" << A << " dt=" << dt;
            EXPECT_EQ(as_u32(k.beta), as_u32(expect_beta)) << "A=" << A << " dt=" << dt;
            EXPECT_EQ(as_u32(k.gamma), as_u32(expect_gamma)) << "A=" << A << " dt=" << dt;

            // beta == gamma in the Tustin/trapezoidal rule.
            EXPECT_EQ(as_u32(k.beta), as_u32(k.gamma)) << "A=" << A << " dt=" << dt;

            // |alpha| < 1 for a decaying pole (D5 invariant).
            EXPECT_LT(std::fabs(k.alpha), 1.0f) << "A=" << A << " dt=" << dt;
        }
    }
}

// --- D2: rotation ---------------------------------------------------------

TEST(RecurrenceKernel, RotationPreservesMagnitude) {
    const float thetas[] = {0.0f, 0.1f, 0.5f, 1.0f, 3.0f, 6.0f};
    const float cases[][2] = {{1.0f, 0.0f}, {0.0f, 1.0f}, {0.6f, 0.8f}, {-1.0f, -1.0f}};

    for (float theta : thetas) {
        const auto rot = mamba::recurrence::rotation_scalars(theta);
        for (const auto& c : cases) {
            const float re = c[0], im = c[1];
            const float rot_re = rot.cos_t * re - rot.sin_t * im;
            const float rot_im = rot.sin_t * re + rot.cos_t * im;
            const float mag_before = std::sqrt(re * re + im * im);
            const float mag_after = std::sqrt(rot_re * rot_re + rot_im * rot_im);
            EXPECT_NEAR(mag_after, mag_before, 1e-5f) << "theta=" << theta;
        }
    }
}

// --- element arithmetic matches the reference formula ---------------------

TEST(RecurrenceKernel, ThreeTermElementMatchesInlineFormula) {
    const mamba::recurrence::RecurrenceCoeffs k{0.9f, 0.25f, 0.25f};
    const auto rot = mamba::recurrence::rotation_scalars(0.3f);
    const float C = 0.75f;

    const float cases[][6] = {
        // re_prev, im_prev, B_prev, x_prev, B, x
        {0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.25f},
        {1.0f, 0.5f, 0.0f, 0.0f, -1.0f, 0.25f},
        {0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f},   // history only
        {0.5f, -0.5f, 0.25f, 0.5f, 0.5f, -0.5f},
        {3.0f, -2.0f, 0.0f, 0.0f, 0.0f, 0.0f},   // free decay + rotation
    };

    for (const auto& c : cases) {
        const auto r = mamba::recurrence::three_term_element(
            c[0], c[1], c[2], c[3], c[4], C, c[5], k, rot);

        const float rot_re = rot.cos_t * c[0] - rot.sin_t * c[1];
        const float rot_im = rot.sin_t * c[0] + rot.cos_t * c[1];
        const float expect_re = k.alpha * rot_re + k.beta * (c[2] * c[3]) + k.gamma * (c[4] * c[5]);
        const float expect_im = k.alpha * rot_im;
        const float expect_y = C * expect_re;

        EXPECT_EQ(as_u32(r.re_new), as_u32(expect_re));
        EXPECT_EQ(as_u32(r.im_new), as_u32(expect_im));
        EXPECT_EQ(as_u32(r.y_contrib), as_u32(expect_y));
    }
}

// --- D1: history term is live --------------------------------------------

// A two-step impulse: after one sample of (B=1, x=1), the next sample's
// history term must add exactly beta * B_prev * x_prev to the response.
TEST(Recurrence, ThreeTermInputHistoryChangesImpulseResponse) {
    const mamba::recurrence::RecurrenceCoeffs k{0.5f, 0.25f, 0.25f};
    const auto rot = mamba::recurrence::rotation_scalars(0.0f);  // theta=0 isolates D1

    // Step 1: input x=1, B=1 -> re = gamma*(B*x) = 0.25; history saved (B_prev=1, x_prev=1).
    const auto r1 = mamba::recurrence::three_term_element(0.0f, 0.0f, 0.0f, 0.0f,
                                                          1.0f, 1.0f, 1.0f, k, rot);
    ASSERT_EQ(as_u32(r1.re_new), as_u32(0.25f));

    // Step 2 with the saved history: x=0, B=1.
    const auto r_hist = mamba::recurrence::three_term_element(r1.re_new, r1.im_new,
                                                              1.0f, 1.0f,   // B_prev, x_prev
                                                              1.0f, 1.0f, 0.0f, k, rot);
    // Step 2 with no history (the old two-term behaviour).
    const auto r_nohist = mamba::recurrence::three_term_element(r1.re_new, r1.im_new,
                                                                0.0f, 0.0f,
                                                                1.0f, 1.0f, 0.0f, k, rot);

    // Difference is exactly beta * B_prev * x_prev = 0.25.
    const float diff = r_hist.re_new - r_nohist.re_new;
    EXPECT_EQ(as_u32(diff), as_u32(k.beta * 1.0f * 1.0f));
}

// --- D2: resonance / stability -------------------------------------------

TEST(Recurrence, ResonancePreservation) {
    // Near-unit decay + a fixed rotation: the state must oscillate at theta per
    // sample and stay bounded (|alpha * e^{i theta}| = alpha < 1), with no NaN/Inf.
    const float alpha = 0.99999f;
    const float theta = 0.05f;
    const mamba::recurrence::RecurrenceCoeffs k{alpha, 0.0f, 0.0f};
    const auto rot = mamba::recurrence::rotation_scalars(theta);

    float re = 1.0f, im = 0.0f;
    float y = 0.0f;
    float prev_im = im;
    int zero_crossings = 0;
    const int N = 100000;

    for (int i = 0; i < N; ++i) {
        const auto r = mamba::recurrence::three_term_element(re, im, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                                                             k, rot);
        re = r.re_new;
        im = r.im_new;
        y += r.y_contrib;
        ASSERT_FALSE(std::isnan(re)) << "sample " << i;
        ASSERT_FALSE(std::isnan(im)) << "sample " << i;
        ASSERT_FALSE(std::isinf(re)) << "sample " << i;
        ASSERT_FALSE(std::isinf(im)) << "sample " << i;
        if (prev_im >= 0.0f && im < 0.0f) zero_crossings++;
        prev_im = im;
    }

    // Bounded: magnitude never exceeds the initial 1.0 (rotation is unit-norm,
    // alpha < 1).
    const float mag = std::sqrt(re * re + im * im);
    EXPECT_LE(mag, 1.0f);

    // Oscillates at theta per sample (period 2*pi/theta).  A negative-going zero
    // crossing of im happens once per period, so count = N * theta / (2*pi).
    const double kPi = 3.14159265358979323846;
    const double expected = N * theta / (2.0 * kPi);
    EXPECT_NEAR(zero_crossings, expected, expected * 0.1);
}

// --- reset clears the D1 history -----------------------------------------

TEST(Recurrence, ResetClearsHistory) {
    auto block = make_test_block();
    block->set_weights(*make_test_weights());
    alignas(64) float buf_a[MAX_D_MODEL];

    // Accumulate state and (crucially) B/x history.
    for (int i = 0; i < 50; ++i) block->process_sample(0.7f, buf_a);
    block->reset();

    auto fresh = make_test_block();
    fresh->set_weights(*make_test_weights());
    alignas(64) float buf_b[MAX_D_MODEL];

    const float inputs[8] = {0.1f, -0.2f, 0.3f, 0.4f, -0.5f, 0.6f, 0.7f, 0.8f};
    for (int i = 0; i < 8; ++i) {
        block->process_sample(inputs[i], buf_a);
        fresh->process_sample(inputs[i], buf_b);
        for (size_t j = 0; j < MAX_D_MODEL; ++j) {
            ASSERT_EQ(as_u32(buf_a[j]), as_u32(buf_b[j]))
                << "sample " << i << " channel " << j;
        }
    }
}
