#include "ara/ara_document_controller.h"
#include "common/seqlock.h"
#include <cstring>

namespace mamba {

static Seqlocked<ARAConditioningFrame>& get_global_conditioning() {
    static Seqlocked<ARAConditioningFrame> instance;
    return instance;
}

void ARA_set_tempo(float bpm) {
    ARAConditioningFrame frame;
    get_global_conditioning().read(frame);
    frame.tempo_bpm = bpm;
    get_global_conditioning().write(frame);
}

void ARA_set_key_signature(float key) {
    ARAConditioningFrame frame;
    get_global_conditioning().read(frame);
    frame.key_sig = key;
    get_global_conditioning().write(frame);
}

void ARA_set_chord(const float* chord_one_hot) {
    ARAConditioningFrame frame;
    get_global_conditioning().read(frame);
    std::memcpy(frame.chord_one_hot, chord_one_hot, 12 * sizeof(float));
    get_global_conditioning().write(frame);
}

bool ARA_get_conditioning(float& tempo, float& key, float* chord) {
    ARAConditioningFrame frame;
    if (!get_global_conditioning().read(frame)) {
        return false;
    }
    tempo = frame.tempo_bpm;
    key = frame.key_sig;
    if (chord) std::memcpy(chord, frame.chord_one_hot, 12 * sizeof(float));
    return true;
}

Seqlocked<ARAConditioningFrame>* ARA_get_conditioning_seqlock() {
    return &get_global_conditioning();
}

void ARA_write_conditioning_to(Seqlocked<ARAConditioningFrame>* target, float bpm, float key, const float* chord_one_hot) {
    if (!target) return;
    ARAConditioningFrame frame;
    target->read(frame);
    frame.tempo_bpm = bpm;
    frame.key_sig = key;
    if (chord_one_hot) {
        std::memcpy(frame.chord_one_hot, chord_one_hot, 12 * sizeof(float));
    }
    target->write(frame);
}

ARAHostDocument::ARAHostDocument() : tempo_bpm_(120.0f), key_sig_(0.0f) {
    std::memset(chord_, 0, sizeof(chord_));
}

ARAHostDocument::~ARAHostDocument() = default;

void ARAHostDocument::set_tempo(float bpm) {
    tempo_bpm_ = bpm;
    ARA_set_tempo(bpm);
}

void ARAHostDocument::set_key_signature(float key) {
    key_sig_ = key;
    ARA_set_key_signature(key);
}

void ARAHostDocument::set_chord(const float* chord_one_hot) {
    std::memcpy(chord_, chord_one_hot, 12 * sizeof(float));
    ARA_set_chord(chord_one_hot);
}

float ARAHostDocument::get_tempo() const { return tempo_bpm_; }
float ARAHostDocument::get_key_signature() const { return key_sig_; }
void ARAHostDocument::get_chord(float* out_chord) const {
    std::memcpy(out_chord, chord_, 12 * sizeof(float));
}

} // namespace mamba
