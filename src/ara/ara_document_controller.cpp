#include "ara/ara_document_controller.h"
#include "ara/ara_types.h"
#include "common/seqlock.h"
#include "common/conditioning_frame.h"
#include <cstring>

namespace emt {

// ---------------------------------------------------------------------------
// ARADocumentControllerStub
// ---------------------------------------------------------------------------

ARADocumentControllerStub::ARADocumentControllerStub()
    : next_doc_id_(1)
    , active_doc_(kInvalidDocumentID)
    , pb_renderer_data_(nullptr)
    , ed_renderer_data_(nullptr) {
    ARAConditioningFrame init{};
    init.tempo_bpm = 120.0f;
    init.key_sig = 0.0f;
    std::memset(init.chord_one_hot, 0, sizeof(init.chord_one_hot));
    std::memset(init.reserved, 0, sizeof(init.reserved));
    conditioning_sl_.write(init);
}

ARADocumentControllerStub::~ARADocumentControllerStub() = default;

ARADocumentID ARADocumentControllerStub::createDocument() {
    ARADocumentID id = next_doc_id_++;
    active_doc_ = id;
    return id;
}

void ARADocumentControllerStub::destroyDocument(ARADocumentID doc_id) {
    if (doc_id == active_doc_) {
        active_doc_ = kInvalidDocumentID;
    }
}

void ARADocumentControllerStub::updateMusicalContext(
    float tempo_bpm, float key_sig, const float* chord_one_hot) {
    ARAConditioningFrame frame{};
    frame.tempo_bpm = tempo_bpm;
    frame.key_sig = key_sig;
    if (chord_one_hot) {
        std::memcpy(frame.chord_one_hot, chord_one_hot,
                    sizeof(frame.chord_one_hot));
    } else {
        std::memset(frame.chord_one_hot, 0, sizeof(frame.chord_one_hot));
    }
    std::memset(frame.reserved, 0, sizeof(frame.reserved));
    conditioning_sl_.write(frame);
}

// ---------------------------------------------------------------------------
// Global singleton + backward-compatible free functions
// ---------------------------------------------------------------------------

static ARADocumentControllerStub& globalDocController() {
    static ARADocumentControllerStub instance;
    return instance;
}

void ARA_set_tempo(float bpm) {
    globalDocController().updateMusicalContext(
        bpm,
        []{
            ARAConditioningFrame f;
            globalDocController().conditioningSeqlock()->read(f);
            return f.key_sig;
        }(),
        []{
            static float chord[12] = {0};
            ARAConditioningFrame f;
            globalDocController().conditioningSeqlock()->read(f);
            std::memcpy(chord, f.chord_one_hot, sizeof(chord));
            return chord;
        }());
}

void ARA_set_key_signature(float key) {
    globalDocController().updateMusicalContext(
        []{
            ARAConditioningFrame f;
            globalDocController().conditioningSeqlock()->read(f);
            return f.tempo_bpm;
        }(),
        key,
        []{
            static float chord[12] = {0};
            ARAConditioningFrame f;
            globalDocController().conditioningSeqlock()->read(f);
            std::memcpy(chord, f.chord_one_hot, sizeof(chord));
            return chord;
        }());
}

void ARA_set_chord(const float* chord_one_hot) {
    ARAConditioningFrame f;
    globalDocController().conditioningSeqlock()->read(f);
    globalDocController().updateMusicalContext(f.tempo_bpm, f.key_sig,
                                                chord_one_hot);
}

bool ARA_get_conditioning(float& tempo, float& key, float* chord) {
    ARAConditioningFrame frame;
    if (!globalDocController().conditioningSeqlock()->read(frame)) {
        return false;
    }
    tempo = frame.tempo_bpm;
    key = frame.key_sig;
    if (chord) {
        std::memcpy(chord, frame.chord_one_hot, 12 * sizeof(float));
    }
    return true;
}

Seqlocked<ARAConditioningFrame>* ARA_get_conditioning_seqlock() {
    return globalDocController().conditioningSeqlock();
}

void ARA_write_conditioning_to(Seqlocked<ARAConditioningFrame>* target,
                                float bpm, float key,
                                const float* chord_one_hot) {
    if (!target) return;
    ARAConditioningFrame frame;
    target->read(frame);
    frame.tempo_bpm = bpm;
    frame.key_sig = key;
    if (chord_one_hot) {
        std::memcpy(frame.chord_one_hot, chord_one_hot,
                    sizeof(frame.chord_one_hot));
    }
    target->write(frame);
}

ARADocumentControllerStub* ARA_get_document_controller() {
    return &globalDocController();
}

} // namespace emt