#include "ara/ara_document_controller.h"
#include "ara/ara_types.h"
#include "common/seqlock.h"
#include "common/double_buffer_seqlock.h"
#include "common/conditioning_frame.h"
#include <algorithm>
#include <cmath>
#include <cstring>

namespace mamba {

// ---------------------------------------------------------------------------
// ARADocumentControllerStub Implementation
// ---------------------------------------------------------------------------

ARADocumentControllerStub::ARADocumentControllerStub()
    : lifecycle_state_(ARALifecycleState::Uninitialized)
    , next_doc_id_(1)
    , active_doc_(kInvalidDocumentID)
    , analysis_state_(ARAAnalysisState::Idle)
    , analysis_bounds_{}
    , analysis_progress_(0.0f)
    , region_count_(0)
    , pb_renderer_data_(nullptr)
    , ed_renderer_data_(nullptr) {
    initialize();
}

ARADocumentControllerStub::~ARADocumentControllerStub() {
    uninitialize();
}

void ARADocumentControllerStub::initialize() {
    lifecycle_state_ = ARALifecycleState::Initializing;

    ARAConditioningFrame init = convertMusicalContext(120.0f, 0.0f, nullptr);
    conditioning_sl_.write(init);

    ARAConditioningFrame* slot = conditioning_db_.begin_write();
    if (slot) {
        *slot = init;
        conditioning_db_.end_write();
        conditioning_db_.swap();
    }

    analysis_state_ = ARAAnalysisState::Idle;
    analysis_progress_ = 0.0f;
    region_count_ = 0;

    lifecycle_state_ = ARALifecycleState::Ready;
}

void ARADocumentControllerStub::uninitialize() {
    active_doc_ = kInvalidDocumentID;
    analysis_state_ = ARAAnalysisState::Idle;
    analysis_progress_ = 0.0f;
    region_count_ = 0;
    lifecycle_state_ = ARALifecycleState::Uninitialized;
}

ARADocumentID ARADocumentControllerStub::createDocument() {
    ARADocumentID id = next_doc_id_++;
    active_doc_ = id;
    lifecycle_state_ = ARALifecycleState::PlugInAdded;
    return id;
}

void ARADocumentControllerStub::destroyDocument(ARADocumentID doc_id) {
    if (doc_id == active_doc_) {
        active_doc_ = kInvalidDocumentID;
        region_count_ = 0;
        lifecycle_state_ = ARALifecycleState::PlugInRemoved;
        lifecycle_state_ = ARALifecycleState::Ready;
    }
}

bool ARADocumentControllerStub::beginOfflineRendering() {
    if (lifecycle_state_ == ARALifecycleState::PlugInAdded) {
        lifecycle_state_ = ARALifecycleState::OfflineRendering;
        return true;
    }
    return false;
}

void ARADocumentControllerStub::endOfflineRendering() {
    if (lifecycle_state_ == ARALifecycleState::OfflineRendering) {
        lifecycle_state_ = ARALifecycleState::PlugInAdded;
    }
}

bool ARADocumentControllerStub::negotiateAnalysisBounds(
    const ARAAnalysisBounds& requested,
    ARAAnalysisBounds& negotiated,
    int64_t max_source_samples) const {

    double sr = requested.sample_rate;
    if (sr <= 0.0 || std::isnan(sr) || std::isinf(sr)) {
        sr = 48000.0;
    }
    negotiated.sample_rate = sr;

    int64_t start = requested.start_sample;
    if (start < 0) {
        start = 0;
    }
    if (max_source_samples >= 0 && start >= max_source_samples) {
        return false;
    }
    negotiated.start_sample = start;

    int64_t length = requested.length_samples;
    if (length <= 0) {
        length = kMinAnalysisSamples;
    }
    if (length < kMinAnalysisSamples) {
        length = kMinAnalysisSamples;
    }

    if (kAnalysisChunkAlignment > 0) {
        int64_t rem = length % kAnalysisChunkAlignment;
        if (rem != 0) {
            length += (kAnalysisChunkAlignment - rem);
        }
    }

    if (max_source_samples >= 0) {
        if (start + length > max_source_samples) {
            length = max_source_samples - start;
        }
    }

    if (length <= 0) {
        return false;
    }

    negotiated.length_samples = length;
    return true;
}

bool ARADocumentControllerStub::requestAnalysis(
    const ARAAnalysisBounds& bounds, int64_t max_source_samples) {

    ARAAnalysisBounds negotiated{};
    if (!negotiateAnalysisBounds(bounds, negotiated, max_source_samples)) {
        analysis_state_ = ARAAnalysisState::Failed;
        return false;
    }

    analysis_bounds_ = negotiated;
    analysis_state_ = ARAAnalysisState::InProgress;
    analysis_progress_ = 0.0f;

    if (lifecycle_state_ == ARALifecycleState::PlugInAdded) {
        lifecycle_state_ = ARALifecycleState::Analysis;
    }
    return true;
}

void ARADocumentControllerStub::updateAnalysisProgress(float progress_0_to_1) {
    if (std::isnan(progress_0_to_1) || std::isinf(progress_0_to_1)) {
        progress_0_to_1 = 0.0f;
    }
    analysis_progress_ = std::clamp(progress_0_to_1, 0.0f, 1.0f);
}

void ARADocumentControllerStub::completeAnalysis(
    float tempo_bpm, float key_sig, const float* chord_one_hot) {

    analysis_state_ = ARAAnalysisState::Completed;
    analysis_progress_ = 1.0f;
    updateMusicalContext(tempo_bpm, key_sig, chord_one_hot);

    if (lifecycle_state_ == ARALifecycleState::Analysis) {
        lifecycle_state_ = ARALifecycleState::PlugInAdded;
    }
}

void ARADocumentControllerStub::cancelAnalysis() {
    analysis_state_ = ARAAnalysisState::Idle;
    analysis_progress_ = 0.0f;

    if (lifecycle_state_ == ARALifecycleState::Analysis) {
        lifecycle_state_ = ARALifecycleState::PlugInAdded;
    }
}

ARAConditioningFrame ARADocumentControllerStub::convertMusicalContext(
    float tempo_bpm, float key_sig, const float* chord_one_hot) {

    ARAConditioningFrame frame{};
    std::memset(&frame, 0, sizeof(frame));

    // Sanitize & clamp tempo
    if (std::isnan(tempo_bpm) || std::isinf(tempo_bpm) || tempo_bpm <= 0.0f) {
        tempo_bpm = 120.0f;
    }
    tempo_bpm = std::clamp(tempo_bpm, 20.0f, 360.0f);
    frame.tempo_bpm = tempo_bpm;

    // Sanitize & normalize key signature (pitch class 0..11)
    if (std::isnan(key_sig) || std::isinf(key_sig)) {
        key_sig = 0.0f;
    }
    float normalized_key = std::fmod(std::fmod(key_sig, 12.0f) + 12.0f, 12.0f);
    frame.key_sig = normalized_key;

    // Sanitize chord vector
    if (chord_one_hot) {
        for (int i = 0; i < 12; ++i) {
            float val = chord_one_hot[i];
            if (std::isnan(val) || std::isinf(val)) {
                val = 0.0f;
            }
            frame.chord_one_hot[i] = std::clamp(val, 0.0f, 1.0f);
        }
    }

    // Legacy ARA normalized fields
    frame.reserved[0] = (tempo_bpm - 20.0f) / 340.0f;
    frame.reserved[1] = normalized_key / 12.0f;

    // Distribute chord energy across f0 probability bins (32 bins from 12 pitch classes)
    if (chord_one_hot) {
        for (int p = 0; p < 12; ++p) {
            float energy = frame.chord_one_hot[p];
            if (energy > 0.0f) {
                size_t bin0 = (p * 32) / 12;
                if (bin0 < 32) {
                    frame.f0_probs[bin0] = std::max(frame.f0_probs[bin0], energy);
                }
            }
        }
    }

    // Semantic vector: encode harmonic center
    int root = static_cast<int>(normalized_key) % 12;
    if (root >= 0 && root < 80) {
        frame.semantic[root] = 1.0f;
    }

    return frame;
}

void ARADocumentControllerStub::updateMusicalContext(
    float tempo_bpm, float key_sig, const float* chord_one_hot) {

    ARAConditioningFrame frame = convertMusicalContext(tempo_bpm, key_sig, chord_one_hot);

    // Update single-buffered seqlock
    conditioning_sl_.write(frame);

    // Update double-buffered seqlock
    ARAConditioningFrame* slot = conditioning_db_.begin_write();
    if (slot) {
        *slot = frame;
        conditioning_db_.end_write();
        conditioning_db_.swap();
    }
}

void ARADocumentControllerStub::addPlaybackRegion(const ARAPlaybackRegion& region) {
    if (region_count_ < kMaxRegions) {
        regions_[region_count_++] = region;
    }
}

void ARADocumentControllerStub::clearPlaybackRegions() {
    region_count_ = 0;
}

const ARAPlaybackRegion* ARADocumentControllerStub::playbackRegion(size_t index) const {
    if (index < region_count_) {
        return &regions_[index];
    }
    return nullptr;
}

bool ARADocumentControllerStub::getMusicalContextAtSample(
    int64_t sample_pos, float& tempo_bpm, float& key_sig, float* chord_one_hot) const {

    for (size_t i = 0; i < region_count_; ++i) {
        const auto& r = regions_[i];
        if (sample_pos >= r.start_sample && sample_pos < (r.start_sample + r.length_samples)) {
            tempo_bpm = r.tempo_bpm;
            key_sig = r.key_sig;
            if (chord_one_hot) {
                std::memcpy(chord_one_hot, r.chord_one_hot, sizeof(r.chord_one_hot));
            }
            return true;
        }
    }
    return false;
}

void ARADocumentControllerStub::updateMusicalContextForSample(int64_t sample_pos) {
    float bpm = 120.0f;
    float key = 0.0f;
    float chord[12] = {0.0f};
    if (getMusicalContextAtSample(sample_pos, bpm, key, chord)) {
        updateMusicalContext(bpm, key, chord);
    }
}

// ---------------------------------------------------------------------------
// Global singleton + backward-compatible free functions
// ---------------------------------------------------------------------------

static ARADocumentControllerStub& globalDocController() {
    static ARADocumentControllerStub instance;
    return instance;
}

void ARA_set_tempo(float bpm) {
    ARAConditioningFrame f{};
    globalDocController().conditioningDoubleBuffer()->read(f);
    globalDocController().updateMusicalContext(bpm, f.key_sig, f.chord_one_hot);
}

void ARA_set_key_signature(float key) {
    ARAConditioningFrame f{};
    globalDocController().conditioningDoubleBuffer()->read(f);
    globalDocController().updateMusicalContext(f.tempo_bpm, key, f.chord_one_hot);
}

void ARA_set_chord(const float* chord_one_hot) {
    ARAConditioningFrame f{};
    globalDocController().conditioningDoubleBuffer()->read(f);
    globalDocController().updateMusicalContext(f.tempo_bpm, f.key_sig, chord_one_hot);
}

bool ARA_get_conditioning(float& tempo, float& key, float* chord) {
    ARAConditioningFrame frame{};
    if (!globalDocController().conditioningDoubleBuffer()->read(frame)) {
        if (!globalDocController().conditioningSeqlock()->read(frame)) {
            return false;
        }
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
    ARAConditioningFrame frame = ARADocumentControllerStub::convertMusicalContext(
        bpm, key, chord_one_hot);
    target->write(frame);
}

DoubleBufferSeqLock<ARAConditioningFrame>* ARA_get_conditioning_double_buffer() {
    return globalDocController().conditioningDoubleBuffer();
}

bool ARA_read_conditioning_double_buffer(ARAConditioningFrame& out) {
    return globalDocController().readConditioning(out);
}

void ARA_write_conditioning_double_buffer(float bpm, float key, const float* chord_one_hot) {
    globalDocController().updateMusicalContext(bpm, key, chord_one_hot);
}

ARADocumentControllerStub* ARA_get_document_controller() {
    return &globalDocController();
}

} // namespace mamba