#ifndef MAMBA_DSP_VOICE_ENVELOPE_H
#define MAMBA_DSP_VOICE_ENVELOPE_H

// Per-voice amplitude envelope step shared by the synth's process() loop
// (and any future offline renderer, e.g. ARA): the single per-sample call
// site resolves here via clap/clap_plugin.h.
//
// Extracted verbatim from the voice loop in clap_plugin.cpp (Phase 1 #8).
// The math is unchanged; only the home moved. This header is intentionally
// free of CLAP includes so non-plugin users can reuse it.
//
// Two call-site facts are preserved, not "fixed":
//  - VOICE_IDLE: the synth's loop `continue`s past idle voices before
//    reaching the envelope step (skipping the atomic mirrors and voice
//    rendering); that guard stays at the call site. Inside this helper,
//    IDLE is defined as a hold (env/stage untouched) so standalone reuse
//    gets benign behavior for idle slots.
//  - default: an unknown stage zeroes env but leaves the stage value
//    untouched (it does NOT reset to IDLE). Pinned by
//    VoiceEnvelope.UnknownStageZeroesEnvKeepsStage.
// The envelope rates are plain parameters: the synth currently passes its
// fixed VOICE_ATTACK_RATE/VOICE_RELEASE_RATE constants (the Attack/Release
// knobs are accepted and stored but not wired to these rates -- see O10 in
// docs/PLAN_NEXT_STEPS.md).
//
// No-alloc contract: this function operates on caller-provided scalars
// only. It performs no heap allocation (no new/malloc, no containers, no
// exceptions, no RTTI) and is safe to call on the real-time audio thread.
// Heap stability is verified by DspHelpersNoAlloc (tests/
// test_dsp_helpers.cpp) on glibc; elsewhere the contract holds by
// construction (this header includes only <cstdint>).

#include <cstdint>

namespace mamba {

enum VoiceState : uint8_t {
    VOICE_IDLE     = 0,
    VOICE_ATTACK   = 1,
    VOICE_SUSTAIN  = 2,
    VOICE_RELEASE  = 3,
    VOICE_SILENT   = 4,
};

// Advance one voice envelope by a single sample.
template <typename StageT>
inline void voice_envelope_step(float& env, StageT& stage,
                                float attack_rate, float release_rate) {
    switch (stage) {
    case VOICE_IDLE:
        break;
    case VOICE_ATTACK:
        env += attack_rate;
        if (env >= 1.0f) { env = 1.0f; stage = VOICE_SUSTAIN; }
        break;
    case VOICE_SUSTAIN:
        env = 1.0f;
        break;
    case VOICE_RELEASE:
        env -= release_rate;
        if (env <= 0.0f) { env = 0.0f; stage = VOICE_SILENT; }
        break;
    case VOICE_SILENT:
        env = 0.0f;
        break;
    default:
        env = 0.0f;
        break;
    }
}

} // namespace mamba

#endif // MAMBA_DSP_VOICE_ENVELOPE_H
