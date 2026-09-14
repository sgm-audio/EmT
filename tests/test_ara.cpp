// tests/test_ara.cpp
// Comprehensive tests for ARA 2.0 DocumentController and PlaybackRenderer

#include "ara/ara_document_controller.h"
#include "ara/ara_renderer.h"
#include "ara/ara_types.h"
#include "common/double_buffer_seqlock.h"
#include "common/conditioning_frame.h"
#include "core/ssm_inference_core.h"
#include "config.h"

#include <gtest/gtest.h>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <memory>
#include <thread>
#include <vector>

using namespace mamba;

// ===========================================================================
// 1. Document Controller Lifecycle Tests
// ===========================================================================

TEST(ARADocumentControllerTest, InitialStateAndLifecycle) {
    ARADocumentController doc;
    EXPECT_EQ(doc.lifecycleState(), ARALifecycleState::Ready);
    EXPECT_FALSE(doc.isDocumentActive());
    EXPECT_EQ(doc.activeDocument(), kInvalidDocumentID);

    ARADocumentID id = doc.createDocument();
    EXPECT_NE(id, kInvalidDocumentID);
    EXPECT_EQ(doc.activeDocument(), id);
    EXPECT_TRUE(doc.isDocumentActive());
    EXPECT_EQ(doc.lifecycleState(), ARALifecycleState::PlugInAdded);

    doc.destroyDocument(id);
    EXPECT_FALSE(doc.isDocumentActive());
    EXPECT_EQ(doc.activeDocument(), kInvalidDocumentID);
    EXPECT_EQ(doc.lifecycleState(), ARALifecycleState::Ready);
}

TEST(ARADocumentControllerTest, OfflineRenderingTransitions) {
    ARADocumentController doc;
    EXPECT_FALSE(doc.beginOfflineRendering()); // Cannot begin without active document

    doc.createDocument();
    EXPECT_TRUE(doc.beginOfflineRendering());
    EXPECT_EQ(doc.lifecycleState(), ARALifecycleState::OfflineRendering);

    doc.endOfflineRendering();
    EXPECT_EQ(doc.lifecycleState(), ARALifecycleState::PlugInAdded);
}

// ===========================================================================
// 2. Analysis Bounds Negotiation Tests
// ===========================================================================

TEST(ARADocumentControllerTest, NegotiateAnalysisBoundsStandard) {
    ARADocumentController doc;
    ARAAnalysisBounds requested{256, 512, 48000.0};
    ARAAnalysisBounds negotiated{};

    EXPECT_TRUE(doc.negotiateAnalysisBounds(requested, negotiated, 48000));
    EXPECT_EQ(negotiated.start_sample, 256);
    EXPECT_EQ(negotiated.length_samples, 512);
    EXPECT_DOUBLE_EQ(negotiated.sample_rate, 48000.0);
}

TEST(ARADocumentControllerTest, NegotiateAnalysisBoundsNegativeStartClamped) {
    ARADocumentController doc;
    ARAAnalysisBounds requested{-500, 256, 44100.0};
    ARAAnalysisBounds negotiated{};

    EXPECT_TRUE(doc.negotiateAnalysisBounds(requested, negotiated, 48000));
    EXPECT_EQ(negotiated.start_sample, 0); // clamped to 0
    EXPECT_GE(negotiated.length_samples, 256);
}

TEST(ARADocumentControllerTest, NegotiateAnalysisBoundsMinimumLengthAndAlignment) {
    ARADocumentController doc;
    ARAAnalysisBounds requested{100, 50, 48000.0};
    ARAAnalysisBounds negotiated{};

    EXPECT_TRUE(doc.negotiateAnalysisBounds(requested, negotiated, 10000));
    EXPECT_EQ(negotiated.start_sample, 100);
    // Should be at least kMinAnalysisSamples (128) and aligned to 128
    EXPECT_GE(negotiated.length_samples, ARADocumentController::kMinAnalysisSamples);
    EXPECT_EQ(negotiated.length_samples % ARADocumentController::kAnalysisChunkAlignment, 0);
}

TEST(ARADocumentControllerTest, NegotiateAnalysisBoundsOutOfBoundsRejected) {
    ARADocumentController doc;
    ARAAnalysisBounds requested{50000, 1024, 48000.0};
    ARAAnalysisBounds negotiated{};

    // start_sample (50000) >= max_source_samples (48000)
    EXPECT_FALSE(doc.negotiateAnalysisBounds(requested, negotiated, 48000));
}

TEST(ARADocumentControllerTest, NegotiateAnalysisBoundsClampToSourceDuration) {
    ARADocumentController doc;
    ARAAnalysisBounds requested{47000, 2000, 48000.0};
    ARAAnalysisBounds negotiated{};

    EXPECT_TRUE(doc.negotiateAnalysisBounds(requested, negotiated, 48000));
    EXPECT_EQ(negotiated.start_sample, 47000);
    EXPECT_LE(negotiated.start_sample + negotiated.length_samples, 48000);
}

TEST(ARADocumentControllerTest, AnalysisStateExecutionFlow) {
    ARADocumentController doc;
    doc.createDocument();

    ARAAnalysisBounds req{0, 1024, 48000.0};
    EXPECT_TRUE(doc.requestAnalysis(req, 48000));
    EXPECT_EQ(doc.analysisState(), ARAAnalysisState::InProgress);
    EXPECT_EQ(doc.lifecycleState(), ARALifecycleState::Analysis);

    doc.updateAnalysisProgress(0.5f);
    EXPECT_FLOAT_EQ(doc.analysisProgress(), 0.5f);

    float chord[12] = {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f}; // C Major
    doc.completeAnalysis(135.0f, 0.0f, chord);

    EXPECT_EQ(doc.analysisState(), ARAAnalysisState::Completed);
    EXPECT_FLOAT_EQ(doc.analysisProgress(), 1.0f);
    EXPECT_EQ(doc.lifecycleState(), ARALifecycleState::PlugInAdded);

    // Verify context was updated
    ARAConditioningFrame frame{};
    EXPECT_TRUE(doc.readConditioning(frame));
    EXPECT_FLOAT_EQ(frame.tempo_bpm, 135.0f);
    EXPECT_FLOAT_EQ(frame.key_sig, 0.0f);
    EXPECT_FLOAT_EQ(frame.chord_one_hot[0], 1.0f);
    EXPECT_FLOAT_EQ(frame.chord_one_hot[4], 1.0f);
    EXPECT_FLOAT_EQ(frame.chord_one_hot[7], 1.0f);
}

// ===========================================================================
// 3. Musical Context Conversion Tests
// ===========================================================================

TEST(ARADocumentControllerTest, ConvertMusicalContextSanitization) {
    // Normal context
    float chord[12] = {0.0f};
    chord[0] = 1.0f;
    ARAConditioningFrame f1 = ARADocumentController::convertMusicalContext(140.0f, 2.0f, chord);
    EXPECT_FLOAT_EQ(f1.tempo_bpm, 140.0f);
    EXPECT_FLOAT_EQ(f1.key_sig, 2.0f);
    EXPECT_FLOAT_EQ(f1.chord_one_hot[0], 1.0f);

    // Extreme tempo clamping (under 20 -> clamped to 20, over 360 -> clamped to 360)
    ARAConditioningFrame f_low = ARADocumentController::convertMusicalContext(10.0f, 0.0f, nullptr);
    EXPECT_FLOAT_EQ(f_low.tempo_bpm, 20.0f);

    ARAConditioningFrame f_high = ARADocumentController::convertMusicalContext(400.0f, 0.0f, nullptr);
    EXPECT_FLOAT_EQ(f_high.tempo_bpm, 360.0f);

    // Invalid tempo (negative, NaN, Inf) -> defaults to 120
    ARAConditioningFrame f_neg = ARADocumentController::convertMusicalContext(-50.0f, 0.0f, nullptr);
    EXPECT_FLOAT_EQ(f_neg.tempo_bpm, 120.0f);

    float nan_val = std::numeric_limits<float>::quiet_NaN();
    ARAConditioningFrame f_nan = ARADocumentController::convertMusicalContext(nan_val, nan_val, nullptr);
    EXPECT_FLOAT_EQ(f_nan.tempo_bpm, 120.0f);
    EXPECT_FLOAT_EQ(f_nan.key_sig, 0.0f);

    // Key signature normalization
    ARAConditioningFrame f_key = ARADocumentController::convertMusicalContext(120.0f, 14.0f, nullptr);
    EXPECT_FLOAT_EQ(f_key.key_sig, 2.0f); // 14 % 12 == 2

    // Chord vector with NaN/Inf values
    float bad_chord[12] = {nan_val, 2.5f, -0.5f, 0.0f};
    ARAConditioningFrame f_bad = ARADocumentController::convertMusicalContext(120.0f, 0.0f, bad_chord);
    EXPECT_FLOAT_EQ(f_bad.chord_one_hot[0], 0.0f); // NaN replaced with 0.0
    EXPECT_FLOAT_EQ(f_bad.chord_one_hot[1], 1.0f); // clamped to 1.0
    EXPECT_FLOAT_EQ(f_bad.chord_one_hot[2], 0.0f); // clamped to 0.0
}

// ===========================================================================
// 4. Playback Regions Timeline Context Tests
// ===========================================================================

TEST(ARADocumentControllerTest, PlaybackRegionTimelineLookup) {
    ARADocumentController doc;
    doc.createDocument();

    ARAPlaybackRegion r1{};
    r1.start_sample = 0;
    r1.length_samples = 48000;
    r1.tempo_bpm = 120.0f;
    r1.key_sig = 0.0f;
    r1.chord_one_hot[0] = 1.0f;
    doc.addPlaybackRegion(r1);

    ARAPlaybackRegion r2{};
    r2.start_sample = 48000;
    r2.length_samples = 48000;
    r2.tempo_bpm = 140.0f;
    r2.key_sig = 5.0f;
    r2.chord_one_hot[5] = 1.0f;
    doc.addPlaybackRegion(r2);

    EXPECT_EQ(doc.playbackRegionCount(), 2);

    float bpm = 0.0f, key = 0.0f;
    float chord[12] = {0.0f};

    // Sample inside region 1
    EXPECT_TRUE(doc.getMusicalContextAtSample(24000, bpm, key, chord));
    EXPECT_FLOAT_EQ(bpm, 120.0f);
    EXPECT_FLOAT_EQ(key, 0.0f);
    EXPECT_FLOAT_EQ(chord[0], 1.0f);

    // Sample inside region 2
    EXPECT_TRUE(doc.getMusicalContextAtSample(60000, bpm, key, chord));
    EXPECT_FLOAT_EQ(bpm, 140.0f);
    EXPECT_FLOAT_EQ(key, 5.0f);
    EXPECT_FLOAT_EQ(chord[5], 1.0f);

    // Sample outside any region
    EXPECT_FALSE(doc.getMusicalContextAtSample(100000, bpm, key, chord));
}

// ===========================================================================
// 5. Double-Buffered SeqLock Multi-threaded Exchange Test
// ===========================================================================

TEST(ARAConditioningExchangeTest, ConcurrentARAWriterAndAudioReader) {
    ARADocumentController doc;
    doc.createDocument();

    constexpr int kNumUpdates = 10000;
    std::atomic<bool> start_flag{false};
    std::atomic<bool> writer_done{false};
    std::atomic<int64_t> valid_reads{0};
    std::atomic<bool> saw_corruption{false};

    // ARA Thread: writes rapid musical context updates
    std::thread ara_thread([&]() {
        while (!start_flag.load(std::memory_order_acquire)) {}

        for (int i = 0; i < kNumUpdates; ++i) {
            float bpm = 60.0f + static_cast<float>(i % 180);
            float key = static_cast<float>(i % 12);
            float chord[12] = {0.0f};
            chord[i % 12] = 1.0f;

            doc.updateMusicalContext(bpm, key, chord);
        }
        writer_done.store(true, std::memory_order_release);
    });

    // Audio Thread: reads continuously without locking
    std::thread audio_thread([&]() {
        while (!start_flag.load(std::memory_order_acquire)) {}

        ARAConditioningFrame frame{};
        while (!writer_done.load(std::memory_order_acquire)) {
            if (doc.readConditioning(frame)) {
                // Verify data consistency (no tear, no NaNs, valid ranges)
                if (std::isnan(frame.tempo_bpm) || frame.tempo_bpm < 20.0f || frame.tempo_bpm > 360.0f) {
                    saw_corruption.store(true, std::memory_order_release);
                }
                if (std::isnan(frame.key_sig) || frame.key_sig < 0.0f || frame.key_sig >= 12.0f) {
                    saw_corruption.store(true, std::memory_order_release);
                }
                valid_reads.fetch_add(1, std::memory_order_relaxed);
            }
        }
    });

    start_flag.store(true, std::memory_order_release);
    ara_thread.join();
    audio_thread.join();

    EXPECT_FALSE(saw_corruption.load(std::memory_order_acquire));
    EXPECT_GT(valid_reads.load(std::memory_order_relaxed), 50);
}

// ===========================================================================
// 6. PlaybackRenderer Offline Chunk Rendering Tests
// ===========================================================================

TEST(ARAPlaybackRendererTest, CreateDestroyLifecycle) {
    PlaybackRenderer renderer{};
    PlaybackRenderer_create(&renderer);
    EXPECT_NE(renderer.user_data, nullptr);
    EXPECT_NE(renderer.render, nullptr);
    EXPECT_NE(PlaybackRenderer_get_core(&renderer), nullptr);

    PlaybackRenderer_destroy(&renderer);
    EXPECT_EQ(renderer.user_data, nullptr);
}

TEST(ARAPlaybackRendererTest, OfflineChunkRenderingGenerativeSynth) {
    PlaybackRenderer renderer{};
    PlaybackRenderer_create(&renderer);

    constexpr size_t kRenderSamples = 512;
    std::vector<float> output(kRenderSamples, 0.0f);

    // Offline chunk rendering with default context (chunk_size = 128)
    PlaybackRenderer_render_chunked(&renderer, nullptr, output.data(), kRenderSamples, 0, 128);

    for (size_t i = 0; i < kRenderSamples; ++i) {
        EXPECT_FALSE(std::isnan(output[i]));
        EXPECT_FALSE(std::isinf(output[i]));
    }

    PlaybackRenderer_destroy(&renderer);
}

TEST(ARAPlaybackRendererTest, OfflineChunkRenderingWithExtractedContext) {
    ARADocumentController doc;
    doc.createDocument();

    // Add region with distinct musical context
    ARAPlaybackRegion region{};
    region.start_sample = 0;
    region.length_samples = 1024;
    region.tempo_bpm = 145.0f;
    region.key_sig = 7.0f; // G
    region.chord_one_hot[7] = 1.0f;
    doc.addPlaybackRegion(region);

    PlaybackRenderer renderer{};
    PlaybackRenderer_create(&renderer);
    PlaybackRenderer_set_document_controller(&renderer, &doc);

    constexpr size_t kRenderSamples = 512;
    std::vector<float> input(kRenderSamples, 0.5f);
    std::vector<float> output(kRenderSamples, 0.0f);

    PlaybackRenderer_render_chunked(&renderer, input.data(), output.data(), kRenderSamples, 0, 128);

    for (size_t i = 0; i < kRenderSamples; ++i) {
        EXPECT_FALSE(std::isnan(output[i]));
        EXPECT_FALSE(std::isinf(output[i]));
    }

    PlaybackRenderer_destroy(&renderer);
}

TEST(ARAPlaybackRendererTest, OfflineChunkRenderingWithExplicitContext) {
    PlaybackRenderer renderer{};
    PlaybackRenderer_create(&renderer);

    ARAContextState ctx{130.0f, 4.0f, {0.0f}};
    ctx.chord_one_hot[4] = 1.0f;
    PlaybackRenderer_set_context(&renderer, &ctx);

    constexpr size_t kRenderSamples = 256;
    std::vector<float> output(kRenderSamples, 0.0f);

    PlaybackRenderer_render_chunked(&renderer, nullptr, output.data(), kRenderSamples, 0, 128);

    for (size_t i = 0; i < kRenderSamples; ++i) {
        EXPECT_FALSE(std::isnan(output[i]));
        EXPECT_FALSE(std::isinf(output[i]));
    }

    PlaybackRenderer_destroy(&renderer);
}
