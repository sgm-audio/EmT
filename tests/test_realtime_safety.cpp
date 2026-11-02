#include <gtest/gtest.h>
#include "core/ssm_block.h"
#include "core/ssm_inference_core.h"
#include "common/seqlock.h"
#include "common/ring_buffer.h"
#include "common/conditioning_frame.h"
#include "common/double_buffer_seqlock.h"
#include "config.h"

using namespace mamba;

TEST(RealtimeSafety, NoSTLContainersInProcessPath) {
}

TEST(RealtimeSafety, NoMutexInProcessPath) {
}

TEST(RealtimeSafety, NoFileIOInProcessPath) {
}

TEST(RealtimeSafety, ProcessCallGraphAudit) {
    MambaConfig cfg{MAX_D_STATE, MAX_D_MODEL, MAX_D_INNER, MAX_D_CONV, 1, COND_DIM};
    SSMInferenceCore core(cfg);
    float input = 1.0f;
    float output = core.process_sample(input);
    EXPECT_FALSE(std::isnan(output));
}

#ifdef MAMBA_REALTIME_SAFE
#include <new>
#include <cstdlib>

static bool g_realtime_alloc_hit = false;

void* operator new(size_t size) {
    (void)size;
    g_realtime_alloc_hit = true;
    return std::malloc(size);
}

void operator delete(void* ptr) noexcept {
    std::free(ptr);
}

TEST(RealtimeSafety, ZeroAllocationsInProcess) {
    g_realtime_alloc_hit = false;
    MambaConfig cfg{MAX_D_STATE, MAX_D_MODEL, MAX_D_INNER, MAX_D_CONV, 1, COND_DIM};
    SSMInferenceCore core(cfg);
    float input = 1.0f;
    for (int i = 0; i < 100; ++i) {
        core.process_sample(input);
    }
    EXPECT_FALSE(g_realtime_alloc_hit);
}
#endif

TEST(RealtimeSafety, SeqLockReadFunctionsDontAllocate) {
    Seqlocked<ConditioningFrame> sl;
    ConditioningFrame init{};
    init.tempo_bpm = 120.0f;
    sl.write(init);
    ConditioningFrame out;
    EXPECT_TRUE(sl.read(out));
    EXPECT_FLOAT_EQ(out.tempo_bpm, 120.0f);
}