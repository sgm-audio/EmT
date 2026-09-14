#ifndef MAMBA_ARA_DOCUMENT_CONTROLLER_H
#define MAMBA_ARA_DOCUMENT_CONTROLLER_H

#include "ara_types.h"
#include "common/seqlock.h"
#include "common/conditioning_frame.h"
#include <cstdint>
#include <cstddef>

namespace emt {

// ---------------------------------------------------------------------------
// ARADocumentControllerStub — stub document controller.
//
// Lifecycle (mirrors ARA::PlugIn::DocumentController):
//   createDocument()   -> allocates a per-document context
//   destroyDocument()  -> frees the context
//   updateMusicalContext() -> pushes ARAConditioningFrame to the seqlock
//
// In real-SDK mode, this class would inherit from
// ARA::PlugIn::DocumentController instead.
// ---------------------------------------------------------------------------

class ARADocumentControllerStub {
public:
    ARADocumentControllerStub();
    ~ARADocumentControllerStub();

    // Document lifecycle.
    ARADocumentID createDocument();
    void destroyDocument(ARADocumentID doc_id);
    ARADocumentID activeDocument() const { return active_doc_; }

    // Musical context update — pushes to the conditioning seqlock.
    void updateMusicalContext(float tempo_bpm, float key_sig,
                              const float* chord_one_hot);

    // ARA seqlock — the audio thread reads conditioning from here.
    // The CLAP plugin's process() calls this to get the seqlock pointer.
    Seqlocked<ARAConditioningFrame>* conditioningSeqlock() {
        return &conditioning_sl_;
    }

    // Renderer access (created by createPlaybackRenderer / createEditorRenderer).
    void* playbackRendererData() const { return pb_renderer_data_; }
    void setPlaybackRendererData(void* d) { pb_renderer_data_ = d; }
    void* editorRendererData() const { return ed_renderer_data_; }
    void setEditorRendererData(void* d) { ed_renderer_data_ = d; }

private:
    ARADocumentID next_doc_id_;
    ARADocumentID active_doc_;
    Seqlocked<ARAConditioningFrame> conditioning_sl_;
    void* pb_renderer_data_;
    void* ed_renderer_data_;
};

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

// Get/init the global document controller (used by CLAP bridge).
ARADocumentControllerStub* ARA_get_document_controller();

} // namespace emt

#endif