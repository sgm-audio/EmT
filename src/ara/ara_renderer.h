#ifndef ARA_RENDERER_H
#define ARA_RENDERER_H

#include <cstdint>
#include <cstddef>
#include "ara_types.h"
#include "config.h"

namespace mamba {

class ARAHostDocument;
class ARADocumentControllerStub;
class SSMInferenceCore;
struct SSMWeights;
template<typename T> class DoubleBufferSeqLock;
struct ConditioningFrame;
using ARAConditioningFrame = ConditioningFrame;

struct PlaybackRenderer {
    void (*render)(PlaybackRenderer* renderer, float* output, size_t num_samples);
    void* user_data;
};

struct EditorRenderer {
    void (*update)(EditorRenderer* renderer, const float* audio_data, size_t num_samples);
    void* user_data;
};

// Lifecycle
void PlaybackRenderer_create(PlaybackRenderer* renderer);
void PlaybackRenderer_destroy(PlaybackRenderer* renderer);

// Primary render callback (backward-compatible)
void PlaybackRenderer_render(PlaybackRenderer* renderer, float* output, size_t num_samples);

// Offline chunk rendering API
void PlaybackRenderer_render_chunked(PlaybackRenderer* renderer,
                                     const float* input,
                                     float* output,
                                     size_t num_samples,
                                     int64_t start_sample_pos = 0,
                                     size_t chunk_size = MAX_CHUNK_SIZE);

// Context & DocumentController binding
void PlaybackRenderer_set_context(PlaybackRenderer* renderer,
                                  const ARAContextState* context);

void PlaybackRenderer_set_document_controller(PlaybackRenderer* renderer,
                                              ARADocumentControllerStub* doc);

void PlaybackRenderer_set_conditioning_double_buffer(PlaybackRenderer* renderer,
                                                     DoubleBufferSeqLock<ARAConditioningFrame>* db);

void PlaybackRenderer_set_weights(PlaybackRenderer* renderer,
                                  const SSMWeights& weights);

SSMInferenceCore* PlaybackRenderer_get_core(PlaybackRenderer* renderer);

void PlaybackRenderer_reset(PlaybackRenderer* renderer);

// Editor renderer
void EditorRenderer_create(EditorRenderer* renderer, ARAHostDocument* doc);
void EditorRenderer_destroy(EditorRenderer* renderer);
void EditorRenderer_update(EditorRenderer* renderer, const float* audio_data, size_t num_samples);

} // namespace mamba

#endif
