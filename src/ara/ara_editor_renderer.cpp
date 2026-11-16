#include "ara/ara_renderer.h"
#include "ara/ara_document_controller.h"
#include <cstdlib>
#include <cstring>

namespace mamba {

struct EditorRendererContext {
    float* ring_buffer;
    size_t ring_capacity;
    size_t ring_write_pos;
    ARAHostDocument* document;
};

void EditorRenderer_create(EditorRenderer* renderer, ARAHostDocument* doc) {
    if (!renderer) return;
    renderer->user_data = nullptr;
    renderer->update = nullptr;

    auto* ctx = static_cast<EditorRendererContext*>(std::malloc(sizeof(EditorRendererContext)));
    if (!ctx) return;
    std::memset(ctx, 0, sizeof(EditorRendererContext));

    ctx->ring_capacity = 8192;
#if defined(_MSC_VER)
    ctx->ring_buffer = static_cast<float*>(::_aligned_malloc(ctx->ring_capacity * sizeof(float), 64));
#else
    ctx->ring_buffer = static_cast<float*>(std::aligned_alloc(64, ctx->ring_capacity * sizeof(float)));
#endif
    if (!ctx->ring_buffer) {
        std::free(ctx);
        return;
    }
    std::memset(ctx->ring_buffer, 0, ctx->ring_capacity * sizeof(float));
    ctx->ring_write_pos = 0;
    ctx->document = doc;

    renderer->user_data = ctx;
    renderer->update = [](EditorRenderer* r, const float* audio_data, size_t num_samples) {
        if (!r || !r->user_data || !audio_data || num_samples == 0) return;
        auto* c = static_cast<EditorRendererContext*>(r->user_data);
        if (!c->ring_buffer || c->ring_capacity == 0) return;
        for (size_t i = 0; i < num_samples; ++i) {
            c->ring_buffer[c->ring_write_pos] = audio_data[i];
            c->ring_write_pos = (c->ring_write_pos + 1) % c->ring_capacity;
        }
    };
}

void EditorRenderer_destroy(EditorRenderer* renderer) {
    if (!renderer || !renderer->user_data) return;
    auto* ctx = static_cast<EditorRendererContext*>(renderer->user_data);
#if defined(_MSC_VER)
    ::_aligned_free(ctx->ring_buffer);
#else
    std::free(ctx->ring_buffer);
#endif
    std::free(ctx);
    renderer->user_data = nullptr;
    renderer->update = nullptr;
}

void EditorRenderer_update(EditorRenderer* renderer, const float* audio_data, size_t num_samples) {
    if (!renderer || !renderer->update) return;
    renderer->update(renderer, audio_data, num_samples);
}

} // namespace mamba