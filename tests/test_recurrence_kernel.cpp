// tests/test_recurrence_kernel.cpp
//
// Pins for the extracted recurrence kernel (src/core/recurrence_kernel.h).
//
// F1 §a lifted the scalar two-term update out of SSMBlock::step_recurrence
// without changing it.  These tests pin the two things that refactor could
// silently break:
//   1. the coefficients and the element/head arithmetic are the same
//      expressions as the pre-extraction inline loop (bit-exact), and
//   2. the kernel writes only the slices it is given (no neighbour damage, no
//      hidden state), which is what makes the later D1/D2 kernels testable.
//
// They are deliberately not "behaviour" tests: when D1/D2 land, the two-term
// pins get replaced by three-term/rotation pins in the same commit.

#include "core/recurrence_kernel.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cmath>
#include <cstring>
#include <vector>

namespace {

// Deterministic pseudo-random floats, no <random> (stable across toolchains).
struct Lcg {
    uint32_t state = 0x12345678u;
    float next() {
        state = state * 1664525u + 1013904223u;
        return static_cast<float>(static_cast<int32_t>(state)) * (1.0f / 2147483648.0f);
    }
};

uint32_t as_u32(float v) {
    uint32_t u = 0;
    std::memcpy(&u, &v, sizeof(u));
    return u;
}

} // namespace

// The identity below is the pre-extraction code, copied verbatim:
//   float half_a = A_decay * dt * 0.5f;
//   float inv_denom = 1.0f / (1.0f - half_a);
//   float A_bar = (1.0f + half_a) * inv_denom;
//   float B_bar_scale = dt * inv_denom;
TEST(RecurrenceKernel, BilinearCoeffsMatchInlineFormula) {
    const float poles[] = {-8.0f, -3.5f, -1.0f, -0.25f, -0.001f};
    const float steps[] = {1.0e-4f, 0.001f, 0.01f, 0.1f, 0.5f, 1.0f, 2.0f};

    for (float A : poles) {
        for (float dt : steps) {
            const float half_a = A * dt * 0.5f;
            const float inv_denom = 1.0f / (1.0f - half_a);
            const float expect_a = (1.0f + half_a) * inv_denom;
            const float expect_b = dt * inv_denom;

            const mamba::recurrence::BilinearCoeffs k = mamba::recurrence::bilinear_coeffs(A, dt);

            // Bit-exact, not near-equal: same expression, same order.
            EXPECT_EQ(as_u32(k.a_bar), as_u32(expect_a)) << "A=" << A << " dt=" << dt;
            EXPECT_EQ(as_u32(k.b_bar_scale), as_u32(expect_b)) << "A=" << A << " dt=" << dt;

            // The stability invariant the register quotes (D5 note in
            // ssm_block.h): |a_bar| < 1 for a decaying pole.
            EXPECT_LT(std::fabs(k.a_bar), 1.0f) << "A=" << A << " dt=" << dt;
        }
    }
}

// One element: h <- a_bar*h + b_bar_scale*B*x, y += C*h.
TEST(RecurrenceKernel, TwoTermElementMatchesInlineFormula) {
    const float cases[][5] = {
        // h, a_bar, b_bar_scale, B, x   (C is fixed at 0.75 below)
        {0.0f, 0.9f, 0.5f, 1.0f, 0.25f},
        {1.0f, 0.9f, 0.5f, -1.0f, 0.25f},
        {-2.5f, 0.99f, 0.01f, 0.5f, -0.5f},
        {3.0f, 0.5f, 2.0f, 0.0f, 0.0f},
        {1e-3f, 0.999f, 1e-3f, 1e3f, 1e-3f},
    };
    const float C = 0.75f;
    const float y0 = 0.125f;

    for (const auto& c : cases) {
        float h = c[0];
        float y = y0;
        mamba::recurrence::two_term_element(h, y, c[1], c[2], c[3], C, c[4]);

        const float expect_h = c[1] * c[0] + c[2] * c[3] * c[4];
        const float expect_y = y0 + C * expect_h;

        EXPECT_EQ(as_u32(h), as_u32(expect_h));
        EXPECT_EQ(as_u32(y), as_u32(expect_y));
    }
}

// Whole head against a literal copy of the pre-extraction nested loop.
TEST(RecurrenceKernel, TwoTermHeadMatchesReferenceLoop) {
    constexpr size_t headdim = 64;
    constexpr size_t d_state = 32;

    Lcg rng;
    std::vector<float> h_kernel(headdim * d_state);
    std::vector<float> h_ref(headdim * d_state);
    std::vector<float> B(d_state);
    std::vector<float> C(d_state);
    std::vector<float> x(headdim);
    std::vector<float> y_kernel(headdim, 0.0f);
    std::vector<float> y_ref(headdim, 0.0f);

    for (size_t i = 0; i < h_kernel.size(); ++i) h_kernel[i] = h_ref[i] = rng.next();
    for (size_t s = 0; s < d_state; ++s) {
        B[s] = rng.next();
        C[s] = rng.next();
    }
    for (size_t d = 0; d < headdim; ++d) x[d] = rng.next();

    // Two different poles / step sizes through the same code path.
    for (float dt : {0.001f, 0.25f}) {
        const float A_decay = -1.0f;
        const float half_a = A_decay * dt * 0.5f;
        const float inv_denom = 1.0f / (1.0f - half_a);
        const float A_bar = (1.0f + half_a) * inv_denom;
        const float B_bar_scale = dt * inv_denom;

        // Reference: the exact loop that used to live in step_recurrence.
        for (size_t d = 0; d < headdim; ++d) {
            const float x_hd = x[d];
            for (size_t s = 0; s < d_state; ++s) {
                const float B_val = B[s];
                const float C_val = C[s];
                h_ref[d * d_state + s] =
                    A_bar * h_ref[d * d_state + s] + B_bar_scale * B_val * x_hd;
                y_ref[d] += C_val * h_ref[d * d_state + s];
            }
        }

        const auto k = mamba::recurrence::bilinear_coeffs(A_decay, dt);
        mamba::recurrence::two_term_head(h_kernel.data(), y_kernel.data(), B.data(), C.data(),
                                         x.data(), headdim, d_state, k);

        for (size_t i = 0; i < h_kernel.size(); ++i) {
            ASSERT_EQ(as_u32(h_kernel[i]), as_u32(h_ref[i])) << "state element " << i << " dt=" << dt;
        }
        for (size_t d = 0; d < headdim; ++d) {
            ASSERT_EQ(as_u32(y_kernel[d]), as_u32(y_ref[d])) << "output element " << d << " dt=" << dt;
        }
    }
}

// The kernel is given slices; it must not touch anything outside them. This is
// the property that lets D1/D2 kernel variants share the same state struct.
TEST(RecurrenceKernel, TwoTermHeadWritesOnlyItsSlices) {
    constexpr size_t headdim = 8;
    constexpr size_t d_state = 4;
    constexpr size_t heads = 3;

    // Three heads laid out contiguously, like SSMState::ssm_real.
    std::vector<float> h(heads * headdim * d_state, 7.0f);
    std::vector<float> y(heads * headdim, 0.0f);
    std::vector<float> B(d_state, 0.5f), C(d_state, 0.25f), x(headdim, 1.0f);

    const size_t target = 1;
    float* h_slice = h.data() + target * headdim * d_state;
    float* y_slice = y.data() + target * headdim;

    mamba::recurrence::two_term_head(h_slice, y_slice, B.data(), C.data(), x.data(), headdim,
                                     d_state, mamba::recurrence::bilinear_coeffs(-1.0f, 0.1f));

    for (size_t i = 0; i < h.size(); ++i) {
        const bool in_slice = i >= target * headdim * d_state && i < (target + 1) * headdim * d_state;
        if (!in_slice) {
            ASSERT_EQ(as_u32(h[i]), as_u32(7.0f)) << "state element " << i << " outside the slice changed";
        } else {
            ASSERT_NE(as_u32(h[i]), as_u32(7.0f)) << "state element " << i << " inside the slice not updated";
        }
    }
    for (size_t i = 0; i < y.size(); ++i) {
        const bool in_slice = i >= target * headdim && i < (target + 1) * headdim;
        if (!in_slice) {
            ASSERT_EQ(as_u32(y[i]), as_u32(0.0f)) << "output element " << i << " outside the slice changed";
        }
    }
}

// A zero input must decay, never grow, for every pole the clamp can produce.
TEST(RecurrenceKernel, ZeroInputDecaysForEveryClampedPole) {
    for (float A : {-16.0f, -8.0f, -2.0f, -1.0f, -0.5f}) {
        for (float dt : {1e-4f, 0.01f, 0.5f, 2.0f}) {
            const auto k = mamba::recurrence::bilinear_coeffs(A, dt);
            float h = 1.0f;
            float y = 0.0f;
            float prev = std::fabs(h);
            for (int i = 0; i < 1000; ++i) {
                mamba::recurrence::two_term_element(h, y, k.a_bar, k.b_bar_scale, 0.0f, 0.0f, 0.0f);
                ASSERT_LE(std::fabs(h), prev) << "A=" << A << " dt=" << dt << " sample " << i;
                prev = std::fabs(h);
            }
            ASSERT_LT(prev, 1.0f) << "A=" << A << " dt=" << dt;
        }
    }
}

// Closed-form LTI verification: for constant A, dt, B, C:
// 1. Unit impulse x = [1, 0, 0, ...] produces y[n] = C * b_bar * a_bar^n.
// 2. Unit step x = 1.0 converges monotonically to DC gain 1 / |A| (when B=C=1).
TEST(RecurrenceKernel, AnalyticalImpulseAndStepSteadyState) {
    const float A = -2.0f;
    const float dt = 0.1f;
    const auto k = mamba::recurrence::bilinear_coeffs(A, dt);
    const float B = 1.0f;
    const float C = 1.0f;

    // 1. Impulse response
    float h = 0.0f;
    float a_pow = 1.0f;
    for (int n = 0; n < 32; ++n) {
        float y = 0.0f;
        const float x = (n == 0) ? 1.0f : 0.0f;
        mamba::recurrence::two_term_element(h, y, k.a_bar, k.b_bar_scale, B, C, x);
        const float expected = C * k.b_bar_scale * B * a_pow;
        EXPECT_NEAR(y, expected, 1e-6f) << "impulse sample " << n;
        a_pow *= k.a_bar;
    }

    // 2. Step response steady state = B * C / |A| = 0.5
    h = 0.0f;
    float prev_y = -1.0f;
    for (int n = 0; n < 200; ++n) {
        float y = 0.0f;
        mamba::recurrence::two_term_element(h, y, k.a_bar, k.b_bar_scale, B, C, 1.0f);
        EXPECT_GE(y, prev_y) << "step response must be monotonic at sample " << n;
        prev_y = y;
    }
    EXPECT_NEAR(prev_y, 1.0f / std::fabs(A), 1e-5f);
}

// Adversarial numerical conditioning: extreme pole decay A and step sizes dt
// driven by Nyquist alternation x[n] = (-1)^n.
TEST(RecurrenceKernel, ExtremePoleAndDeltaNyquistResonance) {
    const float extreme_poles[] = {-1.0e4f, -100.0f, -1.0f, -1.0e-3f, -1.0e-5f};
    const float extreme_dts[] = {1.0e-5f, 1.0e-3f, 0.1f, 10.0f, 1.0e3f};
    const float B = 0.5f;
    const float C = 0.5f;

    for (float A : extreme_poles) {
        for (float dt : extreme_dts) {
            const auto k = mamba::recurrence::bilinear_coeffs(A, dt);

            // 1. Bilinear A-stability: continuous left half-plane Re(A) < 0 must
            // map inside or on the unit circle |a_bar| <= 1.0 in float32. In infinite
            // precision |a_bar| = |(1 + A*dt/2)/(1 - A*dt/2)| < 1 strictly for all A < 0, dt > 0.
            EXPECT_LE(std::fabs(k.a_bar), 1.0f) << "A=" << A << " dt=" << dt;
            const double half_d = static_cast<double>(A) * static_cast<double>(dt) * 0.5;
            const double a_bar_exact = (1.0 + half_d) / (1.0 - half_d);
            EXPECT_LT(std::fabs(a_bar_exact), 1.0) << "A=" << A << " dt=" << dt;

            EXPECT_GT(k.b_bar_scale, 0.0f) << "A=" << A << " dt=" << dt;
            EXPECT_FALSE(std::isnan(k.a_bar));
            EXPECT_FALSE(std::isinf(k.a_bar));
            EXPECT_FALSE(std::isnan(k.b_bar_scale));
            EXPECT_FALSE(std::isinf(k.b_bar_scale));

            // 2. Alternating Nyquist drive x[n] = (-1)^n
            float h = 0.0f;
            float max_h = 0.0f;
            for (int n = 0; n < 1000; ++n) {
                float y = 0.0f;
                const float x = (n % 2 == 0) ? 1.0f : -1.0f;
                mamba::recurrence::two_term_element(h, y, k.a_bar, k.b_bar_scale, B, C, x);

                ASSERT_TRUE(std::isfinite(h)) << "A=" << A << " dt=" << dt << " sample " << n;
                ASSERT_TRUE(std::isfinite(y)) << "A=" << A << " dt=" << dt << " sample " << n;
                max_h = std::max(max_h, std::fabs(h));
            }

            // Analytical upper bound on state magnitude for bounded |x| <= 1:
            // |h_n| <= b_bar * |B| * (1 / (1 - |a_bar|))
            const float bound = (k.b_bar_scale * std::fabs(B) / (1.0f - std::fabs(k.a_bar))) * 1.01f + 1e-4f;
            EXPECT_LE(max_h, bound) << "A=" << A << " dt=" << dt;
        }
    }
}

