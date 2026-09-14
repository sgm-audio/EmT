#ifndef MAMBA_ARA_TYPES_H
#define MAMBA_ARA_TYPES_H

// ---------------------------------------------------------------------------
// Compile-time selection: real ARA SDK vs. stub types.
//
// If ARA_INCLUDE_DIR is set, we attempt to use the real ARA::PlugIn
// namespace from the Celemony ARA 2.0 SDK.  Otherwise we use
// hand-written stubs in namespace mamba that mirror the same API
// surface, so the plugin compiles and runs without the SDK.
//
// The conditioning pipeline (ara_conditioning_bridge.h) is fully
// decoupled from this selection: it communicates through seqlocked
// ARAConditioningFrame structs and works identically either way.
// ---------------------------------------------------------------------------

#include <cstdint>
#include <cstddef>

// Forward-declare ARA SDK types so we can reference them when available.
#if __has_include(<ARA_API/ARAPlugInBase.h>)
#define MAMBA_ARA_REAL_SDK
#endif

#ifdef MAMBA_ARA_REAL_SDK
#include <ARA_API/ARAPlugInBase.h>
#include <ARA_API/ARADocument.h>
#include <ARA_API/ARADocumentController.h>
#include <ARA_API/ARAPlaybackRenderer.h>
#include <ARA_API/ARAEditorRenderer.h>
#else
// Stub forward declarations for compilation without SDK.
namespace ARA { namespace PlugIn {
    class PlugIn;
    class Document;
    class DocumentController;
    class PlaybackRenderer;
    class EditorRenderer;
}}
#endif

#include <cstdint>

namespace mamba {

// ---------------------------------------------------------------------------
// Document ID — opaque handle used by both stub and real modes.
// ---------------------------------------------------------------------------

using ARADocumentID = uint64_t;
constexpr ARADocumentID kInvalidDocumentID = 0;

// ---------------------------------------------------------------------------
// ARALifecycleState — state machine per ADR ara_renderer_lifecycle_state_machine
// ---------------------------------------------------------------------------

enum class ARALifecycleState : uint32_t {
    Uninitialized    = 0,
    Initializing     = 1,
    Ready            = 2,
    PlugInAdded      = 3,
    OfflineRendering = 4,
    Analysis         = 5,
    PlugInRemoved    = 6
};

// ---------------------------------------------------------------------------
// ARAAnalysisState — state of background / offline analysis tasks
// ---------------------------------------------------------------------------

enum class ARAAnalysisState : uint32_t {
    Idle       = 0,
    Requested  = 1,
    InProgress = 2,
    Completed  = 3,
    Failed     = 4
};

// ---------------------------------------------------------------------------
// ARAAnalysisBounds — time range in samples with sample rate for analysis
// ---------------------------------------------------------------------------

struct ARAAnalysisBounds {
    int64_t start_sample{0};
    int64_t length_samples{0};
    double sample_rate{48000.0};

    bool isValid() const {
        return start_sample >= 0 && length_samples > 0 && sample_rate > 0.0;
    }

    int64_t end_sample() const {
        return start_sample + length_samples;
    }
};

// ---------------------------------------------------------------------------
// ARAPlaybackRegion — minimal region description used by the stub
// playback renderer.  When the real SDK is present, the real region
// types are used instead.
// ---------------------------------------------------------------------------

struct ARAPlaybackRegion {
    int64_t start_sample{0};
    int64_t length_samples{0};
    float tempo_bpm{120.0f};
    float key_sig{0.0f};
    float chord_one_hot[12]{0.0f};
};

// ---------------------------------------------------------------------------
// ARAContextState — snapshot of musical context that the document
// controller pushes to the conditioning seqlock.
// ---------------------------------------------------------------------------

struct ARAContextState {
    float tempo_bpm{120.0f};
    float key_sig{0.0f};
    float chord_one_hot[12]{0.0f};
};

} // namespace mamba

#endif