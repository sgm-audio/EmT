#ifndef ARA_RENDERER_H
#define ARA_RENDERER_H

#include <cstdint>
#include <cstddef>

namespace mamba {

class ARAHostDocument;

struct PlaybackRenderer {
    void (*render)(PlaybackRenderer* renderer, float* output, size_t num_samples);
    void* user_data;
};

struct EditorRenderer {
    void (*update)(EditorRenderer* renderer, const float* audio_data, size_t num_samples);
    void* user_data;
};

void PlaybackRenderer_create(PlaybackRenderer* renderer);
void PlaybackRenderer_destroy(PlaybackRenderer* renderer);
void PlaybackRenderer_render(PlaybackRenderer* renderer, float* output, size_t num_samples);

void EditorRenderer_create(EditorRenderer* renderer, ARAHostDocument* doc);
void EditorRenderer_destroy(EditorRenderer* renderer);
void EditorRenderer_update(EditorRenderer* renderer, const float* audio_data, size_t num_samples);

} // namespace mamba

#endif
