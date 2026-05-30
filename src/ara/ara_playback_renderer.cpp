#include "ara/ara_renderer.h"
#include "core/ssm_block.h"
#include <cstdlib>
#include <cstring>

namespace mamba {

struct alignas(64) PlaybackRendererContext {
    alignas(64) std::byte ssm_storage[sizeof(SSMBlock)];
    SSMBlock* ssm_block;
    float buffer[4096];
    size_t buffer_fill;
};

void PlaybackRenderer_create(PlaybackRenderer* renderer) {
    auto* ctx = new PlaybackRendererContext();
    std::memset(ctx, 0, sizeof(PlaybackRendererContext));

    ctx->ssm_block = new (&ctx->ssm_storage) SSMBlock(MAX_D_STATE, MAX_D_MODEL, MAX_D_INNER, MAX_D_CONV);
    ctx->buffer_fill = 0;

    renderer->user_data = ctx;
    renderer->render = [](PlaybackRenderer* r, float* output, size_t num_samples) {
        auto* c = static_cast<PlaybackRendererContext*>(r->user_data);
        for (size_t i = 0; i < num_samples; ++i) {
            output[i] = c->ssm_block->process_sample(0.0f);
        }
    };
}

void PlaybackRenderer_destroy(PlaybackRenderer* renderer) {
    auto* ctx = static_cast<PlaybackRendererContext*>(renderer->user_data);
    ctx->ssm_block->~SSMBlock();
    delete ctx;
}

void PlaybackRenderer_render(PlaybackRenderer* renderer, float* output, size_t num_samples) {
    renderer->render(renderer, output, num_samples);
}

}