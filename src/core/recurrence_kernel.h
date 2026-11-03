#ifndef RECURRENCE_KERNEL_H
#define RECURRENCE_KERNEL_H

// Scalar SSM recurrence kernels, split out of SSMBlock::step_recurrence.
//
// Why this file exists: the recurrence is the one place where this build can
// diverge from the Mamba-3 paper (deviations D1 and D2 in
// ARCHITECTURE_FIDELITY.md §2).  Keeping the arithmetic here means the
// discretization can change in one place, the NumPy reference can be diffed
// against it function by function, and a unit test can pin it without
// constructing a whole SSMBlock or loading weights.
//
// Contract for every function in this header — it runs on the audio thread:
//   * no allocation, no locks, no file/network I/O, no logging
//   * no exceptions (`noexcept` on everything), no RTTI, no virtual calls
//   * no hidden state: the caller passes explicit slices, the kernel reads and
//     writes only those slices, and nothing is stored between calls
//   * fixed-size loops over caller-provided lengths; no recursion
//
// The rule implemented today is the two-term bilinear (Tustin) update on a
// real pole, with `A <= 0` clamped upstream so |a_bar| < 1:
//
//   half        = A * dt / 2
//   a_bar       = (1 + half) / (1 - half)          state decay
//   b_bar_scale = dt / (1 - half)                  gain on B . x
//   h          <- a_bar * h + b_bar_scale * (B . x)
//   y          += C . h
//
// D1 replaces this with the three-term exponential-trapezoidal form
// (alpha/beta/gamma over a one-sample B.x history); D2 adds the 2x2 rotation
// on (re, im).  Both land here, and the signatures stay explicit so the
// Python reference can be checked against them term by term.

#include <cstddef>

namespace mamba {
namespace recurrence {

// Bilinear discretization of the continuous pole A for one step dt.
struct BilinearCoeffs {
    float a_bar;        // multiplier on the carried state
    float b_bar_scale;  // multiplier on the input term B . x
};

// Coeffs for one head.  `A` is negative (SSMBlock clamps it) and `dt > 0`;
// the caller keeps |a_bar| < 1 by construction — no clamping here, so a broken
// caller is visible in the tests instead of being silently corrected.
inline BilinearCoeffs bilinear_coeffs(float A, float dt) noexcept {
    const float half = A * dt * 0.5f;
    const float inv = 1.0f / (1.0f - half);
    return {(1.0f + half) * inv, dt * inv};
}

// One state element for one (head, head-dim, state-index) triple:
//   h <- a_bar * h + b_bar_scale * B * x
//   y += C * h
// The multiplication order is fixed (a_bar*h first, then b_bar*B*x, then C*h)
// so the output is reproducible across compilers for the same inputs.
inline void two_term_element(float& h, float& y, float a_bar, float b_bar_scale, float B, float C,
                             float x) noexcept {
    h = a_bar * h + b_bar_scale * B * x;
    y += C * h;
}

// One head, all head-dim x state-index elements.
//
// Slices (all owned by the caller):
//   h    h[headdim][d_state], row-major and tight (stride d_state)
//   y    y[headdim], accumulated in place (the caller zeroes it once per step)
//   B, C B[d_state], C[d_state] for this head and the active MIMO rank
//   x    x[headdim] for this head
inline void two_term_head(float* h, float* y, const float* B, const float* C, const float* x,
                          size_t headdim, size_t d_state, BilinearCoeffs k) noexcept {
    for (size_t d = 0; d < headdim; ++d) {
        float* h_row = h + d * d_state;
        const float x_d = x[d];
        for (size_t s = 0; s < d_state; ++s) {
            two_term_element(h_row[s], y[d], k.a_bar, k.b_bar_scale, B[s], C[s], x_d);
        }
    }
}

} // namespace recurrence
} // namespace mamba

#endif // RECURRENCE_KERNEL_H
