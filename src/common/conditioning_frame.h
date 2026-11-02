#ifndef MAMBA_CONDITIONING_FRAME_H
#define MAMBA_CONDITIONING_FRAME_H

#include <cstdint>

namespace mamba {

// ---------------------------------------------------------------------------
// ConditioningFrame
// ---------------------------------------------------------------------------
// Plain-data frame holding the conditioning vector the audio thread reads
// at the start of each block.  Used in two places:
//
//   1. From the CLAP main thread (ConditioningFrame set by the host UI /
//      note-on handler).
//   2. From the ARA thread (ARAConditioningFrame set by the ARA document
//      controller; structurally identical so it can reuse the same wire
//      format).
//
// Both are written through a Seqlocked<T> in the audio path, so a write is
// always a single seqlock transaction (no per-field locks, no per-field
// ordering).  See src/common/seqlock.h.
//
// alignas(64): the frame lives in MambaSynth / ARAHostDocument as part of
// the seqlock; cache-line alignment prevents false sharing between the
// `data` field and the adjacent `seqlock` counter.

struct alignas(64) ConditioningFrame {
    float tempo_bpm;
    float key_sig;
    float chord_one_hot[12];
    float reserved[2];
};

// ARAConditioningFrame is structurally identical to ConditioningFrame so
// the ARA document controller can publish through the same seqlock
// protocol that the CLAP layer reads from.  Typed separately so the type
// system documents which thread writes which frame.
using ARAConditioningFrame = ConditioningFrame;

// Compile-time guard: the conditioning vector the SSM core sees (COND_DIM
// floats, packed) must fit in a ConditioningFrame exactly.  The CLAP
// plugin re-packs ARAConditioningFrame into a COND_DIM vector at read
// time, and the SSMBlock::set_conditioning() API copies COND_DIM floats
// from the caller's pointer.
//
// Layout (16 floats total):
//   [0]      tempo_bpm
//   [1]      key_sig
//   [2..13]  chord_one_hot[12]
//   [14..15] reserved
static_assert(sizeof(ConditioningFrame) == sizeof(float) * 16,
              "ConditioningFrame must remain a tight 16-float POD");
static_assert(alignof(ConditioningFrame) >= 64,
              "ConditioningFrame must be cache-line aligned");

} // namespace mamba

#endif
