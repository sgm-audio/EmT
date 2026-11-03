#ifndef RECURRENCE_KERNEL_H
#define RECURRENCE_KERNEL_H

// Scalar SSM recurrence kernels, split out of SSMBlock::step_recurrence.
//
// Why this file exists: the recurrence is the one place where this build can
// diverge from the Mamba-3 paper (deviations D1 and D2 in
// docs/ARCHITECTURE_FIDELITY.md §2).  Keeping the arithmetic here means the
// discretization can change in one place, the NumPy reference can be diffed
// against it function by function, and a unit test can pin it without
// constructing a whole SSMBlock or loading weights.
//
// Contract for every function in this header — it runs on the audio thread:
//   * no allocation, no locks, no file/network I/O, no logging
//   * no exceptions (`noexcept` on everything), no RTTI, no virtual calls
//   * no hidden state: every function is pure — it reads its arguments and
//     returns its result; the caller owns and passes every slice
//   * no recursion, fixed argument count
//
// The rule implemented here is the paper's exponential-trapezoidal three-term
// recurrence on a *complex* state realized as per-element 2×2 rotation blocks
// (D1 + D2, both closed 2026-11-03).  With inv = 1 / (1 - A·dt/2):
//
//   alpha = (1 + A·dt/2) · inv      state decay
//   beta  = (dt/2) · inv            gain on the input-history term B_{t-1}⊙x_{t-1}
//   gamma = (dt/2) · inv            gain on the current term B_t⊙x_t
//
//   (h_re, h_im)_t  = alpha · R(θ) · (h_re, h_im)_{t-1}
//                     + beta · (B_prev · x_prev)     [added along the real axis]
//                     + gamma · (B · x)              [added along the real axis]
//   y += C · h_re_t
//
//   R(θ) = [[cos θ, -sin θ], [sin θ, cos θ]]
//
// θ is per head, taken from the first RoPE angle of SSMState::angle_state
// (see step_recurrence).  The input terms are added along the real axis, not
// rotated with the state; all four forward passes agree on this.
//
// Stability: |alpha · e^{iθ}| = |alpha| < 1 for the clamped A <= 0, so the
// homogeneous part decays and the state stays bounded without the periodic
// renormalization that was removed under D5.

#include <cstddef>
#include <cmath>

namespace mamba {
namespace recurrence {

// D1: three-term exponential-trapezoidal coefficients for one head/step.
struct RecurrenceCoeffs {
    float alpha;  // multiplier on the carried (rotated) state
    float beta;   // multiplier on B_{t-1} ⊙ x_{t-1}
    float gamma;  // multiplier on B_t ⊙ x_t
};

// Coeffs for one head.  `A` is negative (SSMBlock clamps it, D4) and `dt > 0`;
// the caller keeps |alpha| < 1 by construction — no clamping here, so a broken
// caller is visible in the tests instead of being silently corrected.
inline RecurrenceCoeffs trapezoidal_coeffs(float A, float dt) noexcept {
    const float half = A * dt * 0.5f;
    const float inv = 1.0f / (1.0f - half);
    return {(1.0f + half) * inv, dt * 0.5f * inv, dt * 0.5f * inv};
}

// D2: rotation scalars for one 2×2 block angle θ.
struct RotationScalars {
    float cos_t;
    float sin_t;
};

inline RotationScalars rotation_scalars(float theta) noexcept {
    return {std::cos(theta), std::sin(theta)};
}

// One complex state element, D1 + D2 combined (see the header banner for the
// derivation).  Pure: reads its arguments, returns the new state and the
// output contribution.  The multiplication order is fixed so the output is
// reproducible across compilers for the same inputs.
struct ThreeTermResult {
    float re_new;    // new real part of the state
    float im_new;    // new imaginary part of the state
    float y_contrib; // C · re_new, added into the head output accumulator
};

inline ThreeTermResult three_term_element(float re_prev, float im_prev,
                                          float B_prev, float x_prev,
                                          float B, float C, float x,
                                          RecurrenceCoeffs k, RotationScalars rot) noexcept {
    // Rotate the previous state by R(θ): (rot_re, rot_im) = R(θ)·(re_prev, im_prev).
    const float rot_re = rot.cos_t * re_prev - rot.sin_t * im_prev;
    const float rot_im = rot.sin_t * re_prev + rot.cos_t * im_prev;
    // Three-term update; the history and current input enter along the real axis.
    const float re_new = k.alpha * rot_re + k.beta * (B_prev * x_prev) + k.gamma * (B * x);
    const float im_new = k.alpha * rot_im;
    return {re_new, im_new, C * re_new};
}

} // namespace recurrence
} // namespace mamba

#endif // RECURRENCE_KERNEL_H
