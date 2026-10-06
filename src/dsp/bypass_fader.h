#ifndef MAMBA_DSP_BYPASS_FADER_H
#define MAMBA_DSP_BYPASS_FADER_H

// Bypass crossfade primitives shared by the synth and FX plugins (and any
// future offline renderer, e.g. ARA): both call sites resolve here via
// clap/clap_common_utils.h, so bypass/fade math is structurally shared,
// not duplicated.
//
// Extracted verbatim from clap/clap_plugin.h (BypassState) and
// clap/clap_common_utils.h (advance_bypass_machine, compute_mixed_sample)
// (Phase 1 #8). The math is unchanged; only the home moved. This header is
// intentionally free of CLAP includes so non-plugin users can reuse it.
//
// NOTE on scope: the synth drives its bypass through the 4-state machine
// below (stepped once per process() block), while the FX plugin applies an
// inline per-sample fade with no state machine (see clap_fx_plugin.cpp).
// Those two behaviors observably differ and are deliberately NOT unified
// here -- that is #1 territory. This header only gives the machine itself
// (plus the dry/wet+bypass mix primitive) a single tested home.
//
// No-alloc contract: these functions operate on caller-provided scalars
// only. They perform no heap allocation (no new/malloc, no containers, no
// exceptions, no RTTI) and are safe to call on the real-time audio thread.
// Heap stability is verified by DspHelpersNoAlloc (tests/
// test_dsp_helpers.cpp) on glibc; elsewhere the contract holds by
// construction (this header includes only <algorithm> and <cstdint>).

#include <algorithm>
#include <cstdint>

namespace mamba {

enum BypassState : uint8_t {
    BYPASS_PROCESSING    = 0,
    BYPASS_FADING_OUT    = 1,
    BYPASS_BYPASSED      = 2,
    BYPASS_FADING_IN     = 3,
};

// 4-state crossfade bypass state machine transition
template <typename StateType>
inline void advance_bypass_machine(bool target, float& fade, StateType& state,
                                   uint32_t frames, float fade_rate) {
    const float step = static_cast<float>(frames) * fade_rate;
    switch (state) {
    case BYPASS_PROCESSING:
        if (target) {
            fade = std::min(1.0f, fade + step);
            state = (fade >= 1.0f) ? BYPASS_BYPASSED : BYPASS_FADING_OUT;
        }
        break;
    case BYPASS_FADING_OUT:
        if (!target) {
            fade = std::max(0.0f, fade - step);
            if (fade <= 0.0f) state = BYPASS_PROCESSING;
        } else {
            fade = std::min(1.0f, fade + step);
            if (fade >= 1.0f) state = BYPASS_BYPASSED;
        }
        break;
    case BYPASS_BYPASSED:
        if (!target) {
            fade = std::max(0.0f, fade - step);
            state = (fade <= 0.0f) ? BYPASS_PROCESSING : BYPASS_FADING_IN;
        }
        break;
    case BYPASS_FADING_IN:
        if (target) {
            fade = std::min(1.0f, fade + step);
            if (fade >= 1.0f) state = BYPASS_BYPASSED;
        } else {
            fade = std::max(0.0f, fade - step);
            if (fade <= 0.0f) state = BYPASS_PROCESSING;
        }
        break;
    default:
        state = BYPASS_PROCESSING;
        fade = 0.0f;
        break;
    }
}

// Dry/Wet mixing with bypass crossfade
inline float compute_mixed_sample(float dry, float wet, float mix, float bypass_fade) {
    const float mixed = dry * (1.0f - mix) + wet * mix;
    return mixed * (1.0f - bypass_fade) + dry * bypass_fade;
}

} // namespace mamba

#endif // MAMBA_DSP_BYPASS_FADER_H
