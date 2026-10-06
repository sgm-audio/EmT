// Regression vectors for the header-only RT DSP helpers in src/dsp/ that the
// synth and FX plugins share (Phase 1 #8): one-pole parameter smoothing,
// the 4-state bypass crossfade machine, the per-voice amplitude envelope,
// and the dry/wet+bypass mix primitive.
//
// The vectors use binary-exact float/double values (0.25/0.5 steps, unity
// coefficients) so every pin is bit-exact, not tolerance-based. The last
// test enforces the headers' no-alloc contract by asserting zero heap
// growth across 100k helper calls (glibc mallinfo2; skipped elsewhere,
// where the contract holds by construction -- the headers include only
// <cmath>, <algorithm>, and <cstdint>).

#include "dsp/param_smoother.h"
#include "dsp/bypass_fader.h"
#include "dsp/voice_envelope.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#if defined(__GLIBC__)
#include <malloc.h>
#endif

namespace mamba {
namespace {

// --- One-pole smoother --------------------------------------------------------

TEST(DspSmoother, CoeffZeroOrNegativeTimeIsUnity) {
    EXPECT_DOUBLE_EQ(compute_one_pole_coeff(0.0f, 48000.0), 1.0);
    EXPECT_DOUBLE_EQ(compute_one_pole_coeff(-0.5f, 48000.0), 1.0);
}

TEST(DspSmoother, CoeffFallsBackTo48kBelow1Hz) {
    // sample_rate <= 1.0 selects the 48000 Hz fallback; 44100 Hz proves the
    // fallback value is actually engaged rather than coincidental.
    EXPECT_DOUBLE_EQ(compute_one_pole_coeff(0.01f, 0.0),
                     compute_one_pole_coeff(0.01f, 48000.0));
    EXPECT_DOUBLE_EQ(compute_one_pole_coeff(0.01f, 1.0),
                     compute_one_pole_coeff(0.01f, 48000.0));
    EXPECT_NE(compute_one_pole_coeff(0.01f, 44100.0),
              compute_one_pole_coeff(0.01f, 48000.0));
}

TEST(DspSmoother, CoeffMatchesOnePoleFormula) {
    // NOTE: the expected value must convert 0.01f to double exactly as the
    // helper does -- float 0.01 is not double 0.01.
    const double t = static_cast<double>(0.01f) * 48000.0;
    EXPECT_DOUBLE_EQ(compute_one_pole_coeff(0.01f, 48000.0), 1.0 - std::exp(-1.0 / t));
    const double c = compute_one_pole_coeff(0.01f, 48000.0);
    EXPECT_GT(c, 0.0);
    EXPECT_LT(c, 1.0);
}

TEST(DspSmoother, AdvanceDoubleExactSteps) {
    double cur = 0.0;
    advance_one_pole(cur, 1.0, 0.5);
    EXPECT_DOUBLE_EQ(cur, 0.5);
    advance_one_pole(cur, 1.0, 0.5);
    EXPECT_DOUBLE_EQ(cur, 0.75);
    advance_one_pole(cur, 1.0, 0.5);
    EXPECT_DOUBLE_EQ(cur, 0.875);
}

TEST(DspSmoother, AdvanceDoubleSnapsAtUnityCoeff) {
    double cur = 0.125;
    advance_one_pole(cur, 0.75, 1.0);
    EXPECT_DOUBLE_EQ(cur, 0.75);
}

TEST(DspSmoother, AdvanceDoubleHoldsAtTarget) {
    double cur = 0.3;
    advance_one_pole(cur, 0.3, 0.3);
    EXPECT_DOUBLE_EQ(cur, 0.3);
}

TEST(DspSmoother, AdvanceFloatExactSteps) {
    float cur = 0.0f;
    advance_one_pole(cur, 1.0f, 0.5f);
    EXPECT_FLOAT_EQ(cur, 0.5f);
    advance_one_pole(cur, 1.0f, 0.5f);
    EXPECT_FLOAT_EQ(cur, 0.75f);
    advance_one_pole(cur, 1.0f, 0.5f);
    EXPECT_FLOAT_EQ(cur, 0.875f);
}

// --- Bypass state machine -----------------------------------------------------
// The synth steps this machine with a uint32_t state (loaded from its atomic),
// so the vectors use uint32_t too.

TEST(DspBypass, FullFadeOutVector) {
    uint32_t state = BYPASS_PROCESSING;
    float fade = 0.0f;
    advance_bypass_machine(true, fade, state, 1u, 0.25f);
    EXPECT_FLOAT_EQ(fade, 0.25f);
    EXPECT_EQ(state, BYPASS_FADING_OUT);
    advance_bypass_machine(true, fade, state, 1u, 0.25f);
    EXPECT_FLOAT_EQ(fade, 0.5f);
    EXPECT_EQ(state, BYPASS_FADING_OUT);
    advance_bypass_machine(true, fade, state, 1u, 0.25f);
    EXPECT_FLOAT_EQ(fade, 0.75f);
    EXPECT_EQ(state, BYPASS_FADING_OUT);
    advance_bypass_machine(true, fade, state, 1u, 0.25f);
    EXPECT_FLOAT_EQ(fade, 1.0f);
    EXPECT_EQ(state, BYPASS_BYPASSED);
    // Latched: further bypassed blocks are a no-op.
    advance_bypass_machine(true, fade, state, 1u, 0.25f);
    EXPECT_FLOAT_EQ(fade, 1.0f);
    EXPECT_EQ(state, BYPASS_BYPASSED);
}

TEST(DspBypass, ReversalReturnsToProcessing) {
    uint32_t state = BYPASS_FADING_OUT;
    float fade = 0.5f;
    advance_bypass_machine(false, fade, state, 1u, 0.25f);
    EXPECT_FLOAT_EQ(fade, 0.25f);
    EXPECT_EQ(state, BYPASS_FADING_OUT);
    advance_bypass_machine(false, fade, state, 1u, 0.25f);
    EXPECT_FLOAT_EQ(fade, 0.0f);
    EXPECT_EQ(state, BYPASS_PROCESSING);
}

TEST(DspBypass, FadeInFromBypassed) {
    uint32_t state = BYPASS_BYPASSED;
    float fade = 1.0f;
    advance_bypass_machine(false, fade, state, 1u, 0.25f);
    EXPECT_FLOAT_EQ(fade, 0.75f);
    EXPECT_EQ(state, BYPASS_FADING_IN);
    advance_bypass_machine(false, fade, state, 1u, 0.25f);
    advance_bypass_machine(false, fade, state, 1u, 0.25f);
    advance_bypass_machine(false, fade, state, 1u, 0.25f);
    EXPECT_FLOAT_EQ(fade, 0.0f);
    EXPECT_EQ(state, BYPASS_PROCESSING);
}

TEST(DspBypass, FadeInReversal) {
    uint32_t state = BYPASS_FADING_IN;
    float fade = 0.25f;
    advance_bypass_machine(true, fade, state, 1u, 0.25f);
    EXPECT_FLOAT_EQ(fade, 0.5f);
    EXPECT_EQ(state, BYPASS_FADING_IN);
    advance_bypass_machine(true, fade, state, 1u, 0.25f);
    advance_bypass_machine(true, fade, state, 1u, 0.25f);
    EXPECT_FLOAT_EQ(fade, 1.0f);
    EXPECT_EQ(state, BYPASS_BYPASSED);
}

TEST(DspBypass, LatchedStatesIgnoreUnchangedTarget) {
    uint32_t state = BYPASS_PROCESSING;
    float fade = 0.0f;
    advance_bypass_machine(false, fade, state, 64u, 0.25f);
    EXPECT_FLOAT_EQ(fade, 0.0f);
    EXPECT_EQ(state, BYPASS_PROCESSING);
    state = BYPASS_BYPASSED;
    fade = 1.0f;
    advance_bypass_machine(true, fade, state, 64u, 0.25f);
    EXPECT_FLOAT_EQ(fade, 1.0f);
    EXPECT_EQ(state, BYPASS_BYPASSED);
}

TEST(DspBypass, BlockRateStepCompletesInOneCall) {
    // The synth steps with frames=block size and fade_rate=1/64, so a full
    // 64-sample block fades 64*(1/64)=1.0 exactly.
    uint32_t state = BYPASS_PROCESSING;
    float fade = 0.0f;
    advance_bypass_machine(true, fade, state, 64u, 1.0f / 64.0f);
    EXPECT_FLOAT_EQ(fade, 1.0f);
    EXPECT_EQ(state, BYPASS_BYPASSED);
}

TEST(DspBypass, UnknownStateResets) {
    uint32_t state = 99u;
    float fade = 0.5f;
    advance_bypass_machine(true, fade, state, 1u, 0.25f);
    EXPECT_FLOAT_EQ(fade, 0.0f);
    EXPECT_EQ(state, BYPASS_PROCESSING);
}

// --- Voice envelope -----------------------------------------------------------
// Voice::stage is uint8_t, so the vectors use uint8_t too.

TEST(DspEnvelope, AttackRisesToSustain) {
    uint8_t stage = VOICE_ATTACK;
    float env = 0.0f;
    voice_envelope_step(env, stage, 0.25f, 0.25f);
    EXPECT_FLOAT_EQ(env, 0.25f);
    EXPECT_EQ(stage, VOICE_ATTACK);
    voice_envelope_step(env, stage, 0.25f, 0.25f);
    EXPECT_FLOAT_EQ(env, 0.5f);
    voice_envelope_step(env, stage, 0.25f, 0.25f);
    EXPECT_FLOAT_EQ(env, 0.75f);
    voice_envelope_step(env, stage, 0.25f, 0.25f);
    EXPECT_FLOAT_EQ(env, 1.0f);
    EXPECT_EQ(stage, VOICE_SUSTAIN);
}

TEST(DspEnvelope, AttackClampsOvershoot) {
    uint8_t stage = VOICE_ATTACK;
    float env = 0.9f;
    voice_envelope_step(env, stage, 0.25f, 0.25f);
    EXPECT_FLOAT_EQ(env, 1.0f);
    EXPECT_EQ(stage, VOICE_SUSTAIN);
}

TEST(DspEnvelope, SustainForcesUnity) {
    uint8_t stage = VOICE_SUSTAIN;
    float env = 0.3f;
    voice_envelope_step(env, stage, 0.25f, 0.25f);
    EXPECT_FLOAT_EQ(env, 1.0f);
    EXPECT_EQ(stage, VOICE_SUSTAIN);
}

TEST(DspEnvelope, ReleaseFallsToSilent) {
    uint8_t stage = VOICE_RELEASE;
    float env = 1.0f;
    voice_envelope_step(env, stage, 0.25f, 0.25f);
    EXPECT_FLOAT_EQ(env, 0.75f);
    EXPECT_EQ(stage, VOICE_RELEASE);
    voice_envelope_step(env, stage, 0.25f, 0.25f);
    EXPECT_FLOAT_EQ(env, 0.5f);
    voice_envelope_step(env, stage, 0.25f, 0.25f);
    EXPECT_FLOAT_EQ(env, 0.25f);
    voice_envelope_step(env, stage, 0.25f, 0.25f);
    EXPECT_FLOAT_EQ(env, 0.0f);
    EXPECT_EQ(stage, VOICE_SILENT);
}

TEST(DspEnvelope, ReleaseClampsUndershoot) {
    uint8_t stage = VOICE_RELEASE;
    float env = 0.1f;
    voice_envelope_step(env, stage, 0.25f, 0.25f);
    EXPECT_FLOAT_EQ(env, 0.0f);
    EXPECT_EQ(stage, VOICE_SILENT);
}

TEST(DspEnvelope, SilentForcesZero) {
    uint8_t stage = VOICE_SILENT;
    float env = 0.5f;
    voice_envelope_step(env, stage, 0.25f, 0.25f);
    EXPECT_FLOAT_EQ(env, 0.0f);
    EXPECT_EQ(stage, VOICE_SILENT);
}

TEST(DspEnvelope, IdleHolds) {
    uint8_t stage = VOICE_IDLE;
    float env = 0.42f;
    voice_envelope_step(env, stage, 0.25f, 0.25f);
    EXPECT_FLOAT_EQ(env, 0.42f);
    EXPECT_EQ(stage, VOICE_IDLE);
}

TEST(DspEnvelope, UnknownStageZeroesEnvKeepsStage) {
    // Preserved quirk from the inline switch: default zeroes env but does
    // NOT reset the stage.
    uint8_t stage = 99u;
    float env = 0.42f;
    voice_envelope_step(env, stage, 0.25f, 0.25f);
    EXPECT_FLOAT_EQ(env, 0.0f);
    EXPECT_EQ(stage, 99u);
}

// --- Dry/wet + bypass mix -----------------------------------------------------

TEST(DspMix, MixVectors) {
    EXPECT_FLOAT_EQ(compute_mixed_sample(1.0f, 3.0f, 0.0f, 0.0f), 1.0f);
    EXPECT_FLOAT_EQ(compute_mixed_sample(1.0f, 3.0f, 1.0f, 0.0f), 3.0f);
    EXPECT_FLOAT_EQ(compute_mixed_sample(1.0f, 3.0f, 0.25f, 0.0f), 1.5f);
    EXPECT_FLOAT_EQ(compute_mixed_sample(1.0f, 3.0f, 0.25f, 1.0f), 1.0f);
    EXPECT_FLOAT_EQ(compute_mixed_sample(2.0f, 6.0f, 0.5f, 0.5f), 3.0f);
}

// --- No-alloc contract ----------------------------------------------------------

TEST(DspHelpersNoAlloc, HelpersDoNotGrowHeap) {
#if defined(__GLIBC__)
    // Warm up (covers any first-touch lazy init), then assert zero heap
    // growth across 100k calls of every helper.
    double cur = 0.0;
    float fcur = 0.0f, fade = 0.0f, env = 0.0f;
    uint32_t bstate = BYPASS_PROCESSING;
    uint8_t vstage = VOICE_ATTACK;
    for (int i = 0; i < 1000; ++i) {
        const double c = compute_one_pole_coeff(0.01f, 48000.0);
        advance_one_pole(cur, 1.0, c);
        advance_one_pole(fcur, 1.0f, 0.5f);
        advance_bypass_machine(true, fade, bstate, 64u, 1.0f / 64.0f);
        voice_envelope_step(env, vstage, 0.25f, 0.25f);
        fcur = compute_mixed_sample(1.0f, 3.0f, 0.25f, fade);
    }
    const size_t before = mallinfo2().uordblks;
    for (int i = 0; i < 100000; ++i) {
        const double c = compute_one_pole_coeff(0.01f, 48000.0);
        advance_one_pole(cur, 1.0, c);
        advance_one_pole(fcur, 1.0f, 0.5f);
        advance_bypass_machine(true, fade, bstate, 64u, 1.0f / 64.0f);
        voice_envelope_step(env, vstage, 0.25f, 0.25f);
        fcur = compute_mixed_sample(1.0f, 3.0f, 0.25f, fade);
    }
    const size_t after = mallinfo2().uordblks;
    EXPECT_EQ(before, after);
#else
    GTEST_SKIP() << "heap-growth check needs glibc mallinfo2; the no-alloc "
                    "contract holds by construction elsewhere";
#endif
}

} // namespace
} // namespace mamba
