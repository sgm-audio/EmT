#ifndef MAMBA_DSP_PARAM_SMOOTHER_H
#define MAMBA_DSP_PARAM_SMOOTHER_H

// One-pole parameter smoother shared by the synth and FX plugins (and any
// future offline renderer, e.g. ARA): both call sites resolve here via
// clap/clap_common_utils.h, so smoothing behavior is structurally shared,
// not duplicated.
//
// Extracted verbatim from clap/clap_common_utils.h (Phase 1 #8). The math
// is unchanged; only the home moved. This header is intentionally free of
// CLAP includes so non-plugin users can reuse it.
//
// No-alloc contract: these functions operate on caller-provided scalars
// only. They perform no heap allocation (no new/malloc, no containers, no
// exceptions, no RTTI) and are safe to call on the real-time audio thread.
// std::exp appears only in compute_one_pole_coeff, which runs on the
// message/UI thread when time constants change -- never in the per-sample
// step. Heap stability is verified by DspHelpersNoAlloc (tests/
// test_dsp_helpers.cpp) on glibc; elsewhere the contract holds by
// construction (see the banned-construct audit below: this header includes
// only <cmath>).

#include <cmath>

namespace mamba {

// Compute one-pole lowpass filter smoothing coefficient
inline double compute_one_pole_coeff(float smooth_time_s, double sample_rate) {
    if (smooth_time_s <= 0.0f) return 1.0;
    const double sr = (sample_rate > 1.0) ? sample_rate : 48000.0;
    const double t = static_cast<double>(smooth_time_s) * sr;
    return 1.0 - std::exp(-1.0 / t);
}

// Advance one-pole smoother step
inline void advance_one_pole(double& cur, double target, double coeff) {
    cur += coeff * (target - cur);
}

inline void advance_one_pole(float& cur, float target, float coeff) {
    cur += coeff * (target - cur);
}

} // namespace mamba

#endif // MAMBA_DSP_PARAM_SMOOTHER_H
