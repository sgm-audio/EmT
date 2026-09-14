#ifndef MAMBA_ARA_DOCUMENT_CONTROLLER_H
#define MAMBA_ARA_DOCUMENT_CONTROLLER_H

#include "ara_types.h"
#include "common/seqlock.h"
#include "common/double_buffer_seqlock.h"
#include "common/conditioning_frame.h"
#include <cstdint>
#include <cstddef>

namespace mamba {

// ---------------------------------------------------------------------------
// ARADocumentControllerStub — ARA 2.0 DocumentController implementation.
//
// Lifecycle (mirrors ARA::PlugIn::DocumentController and ADR lifecycle state machine):
//   Uninitialized -> Initializing -> Ready -> PlugInAdded (Document active)
//   -> OfflineRendering / Analysis -> PlugInRemoved -> Ready -> Uninitialized
//
// Features:
//   - Document lifecycle state machine
//   - Analysis bounds negotiation
//   - Musical context conversion to cache-aligned ARAConditioningFrame
//   - Double-buffered seqlock exchange with audio thread (DoubleBufferSeqLock)
//   - Backward-compatible single-buffered Seqlocked frame
//   - Playback region timeline musical context lookups
// ---------------------------------------------------------------------------

class ARADocumentControllerStub {
public:
    static constexpr size_t kMaxRegions = 64;
    static constexpr int64_t kMinAnalysisSamples = 128;
    static constexpr int64_t kAnalysisChunkAlignment = 128;

    ARADocumentControllerStub();
    ~ARADocumentControllerStub();

    // Plugin & Document lifecycle
    void initialize();
    void uninitialize();
    ARALifecycleState lifecycleState() const { return lifecycle_state_; }
    bool isDocumentActive() const { return active_doc_ != kInvalidDocumentID; }

    ARADocumentID createDocument();
    void destroyDocument(ARADocumentID doc_id);
    ARADocumentID activeDocument() const { return active_doc_; }

    // Offline rendering transitions
    bool beginOfflineRendering();
    void endOfflineRendering();

    // Analysis bounds negotiation and execution
    bool negotiateAnalysisBounds(const ARAAnalysisBounds& requested,
                                 ARAAnalysisBounds& negotiated,
                                 int64_t max_source_samples = -1) const;
    bool requestAnalysis(const ARAAnalysisBounds& bounds, int64_t max_source_samples = -1);
    void updateAnalysisProgress(float progress_0_to_1);
    void completeAnalysis(float tempo_bpm, float key_sig, const float* chord_one_hot);
    void cancelAnalysis();
    ARAAnalysisState analysisState() const { return analysis_state_; }
    const ARAAnalysisBounds& currentAnalysisBounds() const { return analysis_bounds_; }
    float analysisProgress() const { return analysis_progress_; }

    // Musical context conversion: tempo, key signature, chord -> ARAConditioningFrame
    static ARAConditioningFrame convertMusicalContext(float tempo_bpm,
                                                     float key_sig,
                                                     const float* chord_one_hot);

    // Musical context update — pushes to both conditioning_sl_ and double buffer
    void updateMusicalContext(float tempo_bpm, float key_sig,
                              const float* chord_one_hot);

    // DoubleBufferSeqLock exchange (wait-free, zero allocation for audio thread)
    DoubleBufferSeqLock<ARAConditioningFrame>* conditioningDoubleBuffer() {
        return &conditioning_db_;
    }
    const DoubleBufferSeqLock<ARAConditioningFrame>* conditioningDoubleBuffer() const {
        return &conditioning_db_;
    }
    bool readConditioning(ARAConditioningFrame& out) const {
        return conditioning_db_.read(out);
    }

    // ARA seqlock — backward compatible accessor
    Seqlocked<ARAConditioningFrame>* conditioningSeqlock() {
        return &conditioning_sl_;
    }

    // Playback regions management (timeline context)
    void addPlaybackRegion(const ARAPlaybackRegion& region);
    void clearPlaybackRegions();
    size_t playbackRegionCount() const { return region_count_; }
    const ARAPlaybackRegion* playbackRegion(size_t index) const;
    bool getMusicalContextAtSample(int64_t sample_pos, float& tempo_bpm,
                                  float& key_sig, float* chord_one_hot) const;
    void updateMusicalContextForSample(int64_t sample_pos);

    // Renderer access (created by createPlaybackRenderer / createEditorRenderer)
    void* playbackRendererData() const { return pb_renderer_data_; }
    void setPlaybackRendererData(void* d) { pb_renderer_data_ = d; }
    void* editorRendererData() const { return ed_renderer_data_; }
    void setEditorRendererData(void* d) { ed_renderer_data_ = d; }

private:
    ARALifecycleState lifecycle_state_{ARALifecycleState::Uninitialized};
    ARADocumentID next_doc_id_{1};
    ARADocumentID active_doc_{kInvalidDocumentID};

    ARAAnalysisState analysis_state_{ARAAnalysisState::Idle};
    ARAAnalysisBounds analysis_bounds_{};
    float analysis_progress_{0.0f};

    ARAPlaybackRegion regions_[kMaxRegions]{};
    size_t region_count_{0};

    Seqlocked<ARAConditioningFrame> conditioning_sl_;
    alignas(64) DoubleBufferSeqLock<ARAConditioningFrame> conditioning_db_;

    void* pb_renderer_data_{nullptr};
    void* ed_renderer_data_{nullptr};
};

using ARADocumentController = ARADocumentControllerStub;

// ---------------------------------------------------------------------------
// Global-conditioning free functions (backward-compatible with existing
// CLAP plugin code that includes the old ara_document_controller.h API).
// These delegate to the global ARADocumentControllerStub instance.
// ---------------------------------------------------------------------------

void ARA_set_tempo(float bpm);
void ARA_set_key_signature(float key);
void ARA_set_chord(const float* chord_one_hot);
bool ARA_get_conditioning(float& tempo, float& key, float* chord);
Seqlocked<ARAConditioningFrame>* ARA_get_conditioning_seqlock();
void ARA_write_conditioning_to(Seqlocked<ARAConditioningFrame>* target,
                               float bpm, float key,
                               const float* chord_one_hot);

DoubleBufferSeqLock<ARAConditioningFrame>* ARA_get_conditioning_double_buffer();
bool ARA_read_conditioning_double_buffer(ARAConditioningFrame& out);
void ARA_write_conditioning_double_buffer(float bpm, float key, const float* chord_one_hot);

// Get/init the global document controller (used by CLAP bridge).
ARADocumentControllerStub* ARA_get_document_controller();

} // namespace mamba

#endif