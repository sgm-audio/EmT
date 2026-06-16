#ifndef MAMBA_CONDITIONING_FRAME_H
#define MAMBA_CONDITIONING_FRAME_H

#include <cstdint>
#include "config.h"

namespace mamba {

// ---------------------------------------------------------------------------
// ConditioningFrame (Mamba-3 FiLM)
// ---------------------------------------------------------------------------
// Plain-data frame holding the 128-float conditioning vector the audio
// thread reads at the start of each block.
//
// Layout (128 floats total):
//   [0..15]    tempo, key, chord (legacy 16-float ARA fields)
//   [16..47]   f0 probability vector (32 bins from CREPE 360→32 projection)
//   [48..127]  semantic embedding (80 floats from cross-track aggregator)
//
// Written through Seqlocked<T> by the main/ARA thread.  The audio thread
// reads a snapshot and passes it to SSMBlock::set_conditioning(), which
// uses it for FiLM modulation (gamma, beta per d_model channel).
//
// alignas(64): cache-line alignment prevents false sharing between the
// `data` field and the adjacent seqlock counter.

struct alignas(64) ConditioningFrame {
    float tempo_bpm;
    float key_sig;
    float chord_one_hot[12];
    float reserved[2];              // bytes 0..15: legacy ARA fields
    float f0_probs[32];             // bytes 16..47: pitch probability
    float semantic[80];             // bytes 48..127: semantic embedding
};

static_assert(sizeof(ConditioningFrame) == sizeof(float) * 128,
              "ConditioningFrame must be exactly 128 floats (COND_DIM)");
static_assert(alignof(ConditioningFrame) >= 64,
              "ConditioningFrame must be cache-line aligned");

// ARAConditioningFrame is structurally identical
using ARAConditioningFrame = ConditioningFrame;

} // namespace mamba

#endif
