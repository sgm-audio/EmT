#include "ara/ara_renderer.h"
#include "ara/ara_document_controller.h"
#include "core/ssm_inference_core.h"
#include "core/ssm_block.h"
#include "common/double_buffer_seqlock.h"
#include "common/conditioning_frame.h"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <memory>

namespace mamba {

struct alignas(64) PlaybackRendererContext {
    std::unique_ptr<SSMInferenceCore> core;
    MambaConfig cfg;
    ARADocumentControllerStub* doc_controller{nullptr};
    ARAContextState current_context{120.0f, 0.0f, {0.0f}};
    bool has_explicit_context{false};
    int64_t current_sample_position{0};
    DoubleBufferSeqLock<ARAConditioningFrame>* conditioning_db{nullptr};
};

void PlaybackRenderer_create(PlaybackRenderer* renderer) {
    if (!renderer) return;

    auto* ctx = new PlaybackRendererContext();
    ctx->cfg = MambaConfig::max_config();
    ctx->cfg.num_layers = 1;
    ctx->core = std::make_unique<SSMInferenceCore>(ctx->cfg);
    ctx->core->reset();
    ctx->current_sample_position = 0;
    ctx->doc_controller = nullptr;
    ctx->has_explicit_context = false;
    ctx->conditioning_db = nullptr;

    renderer->user_data = ctx;
    renderer->render = [](PlaybackRenderer* r, float* output, size_t num_samples) {
        PlaybackRenderer_render_chunked(r, nullptr, output, num_samples, 0, MAX_CHUNK_SIZE);
    };
}

void PlaybackRenderer_destroy(PlaybackRenderer* renderer) {
    if (!renderer || !renderer->user_data) return;
    auto* ctx = static_cast<PlaybackRendererContext*>(renderer->user_data);
    delete ctx;
    renderer->user_data = nullptr;
    renderer->render = nullptr;
}

void PlaybackRenderer_set_context(PlaybackRenderer* renderer, const ARAContextState* context) {
    if (!renderer || !renderer->user_data) return;
    auto* ctx = static_cast<PlaybackRendererContext*>(renderer->user_data);
    if (context) {
        ctx->current_context = *context;
        ctx->has_explicit_context = true;
    } else {
        ctx->has_explicit_context = false;
    }
}

void PlaybackRenderer_set_document_controller(PlaybackRenderer* renderer, ARADocumentControllerStub* doc) {
    if (!renderer || !renderer->user_data) return;
    auto* ctx = static_cast<PlaybackRendererContext*>(renderer->user_data);
    ctx->doc_controller = doc;
}

void PlaybackRenderer_set_conditioning_double_buffer(PlaybackRenderer* renderer,
                                                     DoubleBufferSeqLock<ARAConditioningFrame>* db) {
    if (!renderer || !renderer->user_data) return;
    auto* ctx = static_cast<PlaybackRendererContext*>(renderer->user_data);
    ctx->conditioning_db = db;
}

void PlaybackRenderer_set_weights(PlaybackRenderer* renderer, const SSMWeights& weights) {
    if (!renderer || !renderer->user_data) return;
    auto* ctx = static_cast<PlaybackRendererContext*>(renderer->user_data);
    if (ctx->core) {
        ctx->core->set_layer_weights(0, weights);
    }
}

SSMInferenceCore* PlaybackRenderer_get_core(PlaybackRenderer* renderer) {
    if (!renderer || !renderer->user_data) return nullptr;
    auto* ctx = static_cast<PlaybackRendererContext*>(renderer->user_data);
    return ctx->core.get();
}

void PlaybackRenderer_reset(PlaybackRenderer* renderer) {
    if (!renderer || !renderer->user_data) return;
    auto* ctx = static_cast<PlaybackRendererContext*>(renderer->user_data);
    if (ctx->core) {
        ctx->core->reset();
    }
    ctx->current_sample_position = 0;
}

void PlaybackRenderer_render_chunked(PlaybackRenderer* renderer,
                                     const float* input,
                                     float* output,
                                     size_t num_samples,
                                     int64_t start_sample_pos,
                                     size_t chunk_size) {
    if (!renderer || !renderer->user_data || !output || num_samples == 0) return;
    auto* ctx = static_cast<PlaybackRendererContext*>(renderer->user_data);
    if (!ctx->core) return;

    // Apply real-time denormal flags
    SSMBlock::apply_ftz_daz();

    if (chunk_size == 0 || chunk_size > MAX_CHUNK_SIZE) {
        chunk_size = MAX_CHUNK_SIZE;
    }

    size_t processed = 0;
    int64_t current_pos = start_sample_pos;
    alignas(64) float sample_buf[MAX_D_MODEL];

    while (processed < num_samples) {
        size_t this_chunk = std::min(chunk_size, num_samples - processed);

        // 1. Extract musical context for this chunk
        float bpm = 120.0f;
        float key = 0.0f;
        float chord[12] = {0.0f};
        bool context_extracted = false;

        // Try document controller timeline region lookup first
        if (ctx->doc_controller) {
            context_extracted = ctx->doc_controller->getMusicalContextAtSample(
                current_pos, bpm, key, chord);
            if (!context_extracted) {
                // Fall back to document controller's latest conditioning frame
                ARAConditioningFrame frame{};
                if (ctx->doc_controller->readConditioning(frame)) {
                    bpm = frame.tempo_bpm;
                    key = frame.key_sig;
                    std::memcpy(chord, frame.chord_one_hot, sizeof(chord));
                    context_extracted = true;
                }
            }
        }

        // Next, try explicit context
        if (!context_extracted && ctx->has_explicit_context) {
            bpm = ctx->current_context.tempo_bpm;
            key = ctx->current_context.key_sig;
            std::memcpy(chord, ctx->current_context.chord_one_hot, sizeof(chord));
            context_extracted = true;
        }

        // Next, try double-buffered seqlock
        if (!context_extracted && ctx->conditioning_db) {
            ARAConditioningFrame frame{};
            if (ctx->conditioning_db->read(frame)) {
                bpm = frame.tempo_bpm;
                key = frame.key_sig;
                std::memcpy(chord, frame.chord_one_hot, sizeof(chord));
                context_extracted = true;
            }
        }

        // 2. Convert to ARAConditioningFrame
        ARAConditioningFrame cond_frame = ARADocumentControllerStub::convertMusicalContext(
            bpm, key, chord);

        // 3. Set conditioning on SSMInferenceCore
        ctx->core->set_conditioning(&cond_frame.tempo_bpm);

        // 4. Render chunk through SSMInferenceCore
        for (size_t i = 0; i < this_chunk; ++i) {
            float in_sample = input ? input[processed + i] : 0.0f;
            ctx->core->process_sample(in_sample, sample_buf);
            output[processed + i] = sample_buf[0];
        }

        processed += this_chunk;
        current_pos += this_chunk;
    }

    ctx->current_sample_position = current_pos;
}

void PlaybackRenderer_render(PlaybackRenderer* renderer, float* output, size_t num_samples) {
    if (!renderer || !renderer->user_data) return;
    auto* ctx = static_cast<PlaybackRendererContext*>(renderer->user_data);
    PlaybackRenderer_render_chunked(renderer, nullptr, output, num_samples,
                                    ctx->current_sample_position, MAX_CHUNK_SIZE);
}

} // namespace mamba