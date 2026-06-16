// tests/test_conditioning.cpp
// Mamba-3 FiLM conditioning tests

#include "core/ssm_block.h"
#include "core/ssm_inference_core.h"
#include "common/seqlock.h"
#include "common/conditioning_frame.h"
#include "config.h"

#include <gtest/gtest.h>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <thread>
#include <vector>

using namespace mamba;

// SSMBlock is ~6 MB — too large for stack.
static std::unique_ptr<SSMBlock> make_test_block() {
    return std::make_unique<SSMBlock>(
        MAX_D_STATE, MAX_D_MODEL, MAX_D_INNER,
        MAX_NHEADS, MAX_HEADDIM, MAX_MIMO_RANK,
        MAX_NUM_BC_HEADS, COND_DIM);
}

// Helper: minimal test weights
static SSMWeights make_test_weights() {
    SSMWeights w{};
    std::memset(&w, 0, sizeof(w));
    for (size_t j = 0; j < MAX_D_MODEL && j < MAX_D_INNER; ++j) {
        w.in_proj_w[j][j] = 0.1f;
    }
    for (size_t h = 0; h < MAX_NHEADS; ++h) {
        w.dt_bias[h] = 0.5f;
    }
    for (size_t s = 0; s < MAX_D_STATE; ++s) {
        w.B_norm_weight[s] = 1.0f;
        w.C_norm_weight[s] = 1.0f;
    }
    for (size_t j = 0; j < MAX_D_MODEL && j < MAX_D_INNER; ++j) {
        w.out_proj_w[j][j] = 1.0f;
    }
    return w;
}

TEST(ConditioningModulation, SetConditioningDoesNotCrash) {
    auto block = make_test_block();
    block->set_weights(make_test_weights());

    float cond[COND_DIM] = {0.0f};
    cond[0] = 120.0f;
    cond[1] = 0.5f;
    block->set_conditioning(cond);

    alignas(64) float outbuf[MAX_D_MODEL];
    block->process_sample(1.0f, outbuf);
    EXPECT_FALSE(std::isnan(outbuf[0]));
    EXPECT_FALSE(std::isinf(outbuf[0]));
}

TEST(ConditioningModulation, ZeroConditioningWorks) {
    auto block = make_test_block();
    block->set_weights(make_test_weights());

    float cond[COND_DIM] = {0.0f};
    block->set_conditioning(cond);  // all zero → inactive

    alignas(64) float outbuf[MAX_D_MODEL];
    block->process_sample(0.5f, outbuf);
    EXPECT_FALSE(std::isnan(outbuf[0]));
    EXPECT_FALSE(std::isinf(outbuf[0]));
}

TEST(ConditioningModulation, NonZeroConditioningWorks) {
    auto block = make_test_block();
    block->set_weights(make_test_weights());

    float cond[COND_DIM] = {0.0f};
    cond[0] = 1.0f;
    block->set_conditioning(cond);

    alignas(64) float outbuf[MAX_D_MODEL];
    block->process_sample(1.0f, outbuf);
    EXPECT_FALSE(std::isnan(outbuf[0]));
    EXPECT_FALSE(std::isinf(outbuf[0]));
}

// Seqlock atomicity test (same as before, unchanged)
namespace {
constexpr int kAtomicityIters = 50000;
std::atomic<bool> g_reader_saw_nan{false};
std::atomic<int64_t> g_reader_iters{0};

void seqlock_writer(Seqlocked<ConditioningFrame>* sl) {
    ConditioningFrame f;
    for (int i = 0; i < kAtomicityIters; ++i) {
        f.tempo_bpm = static_cast<float>(i % 240) + 0.5f;
        f.key_sig = static_cast<float>(i % 12);
        for (int c = 0; c < 12; ++c)
            f.chord_one_hot[c] = static_cast<float>((i + c) & 0xff) * 0.01f;
        f.reserved[0] = static_cast<float>(i) * 0.001f;
        f.reserved[1] = static_cast<float>(-i) * 0.001f;
        sl->write(f);
    }
}

void seqlock_reader(Seqlocked<ConditioningFrame>* sl,
                    std::atomic<bool>* stop) {
    while (!stop->load(std::memory_order_acquire)) {
        ConditioningFrame f;
        if (sl->read(f)) {
            if (std::isnan(f.tempo_bpm)) g_reader_saw_nan.store(true, std::memory_order_release);
            for (int c = 0; c < 12; ++c)
                if (std::isnan(f.chord_one_hot[c])) g_reader_saw_nan.store(true, std::memory_order_release);
            g_reader_iters.fetch_add(1, std::memory_order_relaxed);
        }
    }
}
} // namespace

TEST(ConditioningSeqlock, ReaderNeverObservesTornWrite) {
    Seqlocked<ConditioningFrame> sl;
    ConditioningFrame init{};
    init.tempo_bpm = 120.0f;
    sl.write(init);

    std::atomic<bool> stop{false};
    std::thread reader(seqlock_reader, &sl, &stop);
    std::thread writer(seqlock_writer, &sl);
    writer.join();
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    stop.store(true, std::memory_order_release);
    reader.join();

    EXPECT_FALSE(g_reader_saw_nan.load(std::memory_order_acquire));
    EXPECT_GT(g_reader_iters.load(std::memory_order_relaxed), 1000);
}
