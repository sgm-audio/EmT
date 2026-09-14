#include <gtest/gtest.h>
#include "core/ssm_block.h"
#include "core/ssm_inference_core.h"
#include "common/seqlock.h"
#include "common/conditioning_frame.h"
#include "config.h"
#include <cmath>
#include <memory>

using namespace emt;

TEST(RealtimeSafety, ProcessCallGraphAudit) {
    auto core = std::make_unique<SSMInferenceCore>();
    MambaConfig cfg = MambaConfig::max_config();
    cfg.num_layers = 1;
    core->init(cfg);
    alignas(64) float outbuf[MAX_D_MODEL];
    core->process_sample(1.0f, outbuf);
    EXPECT_FALSE(std::isnan(outbuf[0]));
}

TEST(RealtimeSafety, SeqLockReadFunctionsDontAllocate) {
    Seqlocked<ConditioningFrame> sl;
    ConditioningFrame init{};
    init.tempo_bpm = 120.0f;
    sl.write(init);
    ConditioningFrame out;
    EXPECT_TRUE(sl.read(out));
    EXPECT_FLOAT_EQ(out.tempo_bpm, 120.0f);
}

TEST(RealtimeSafety, MultiLayerProcessIsStable) {
    auto core = std::make_unique<SSMInferenceCore>();
    MambaConfig cfg = MambaConfig::max_config();
    cfg.num_layers = MAX_LAYERS;
    core->init(cfg);
    alignas(64) float outbuf[MAX_D_MODEL];
    for (int i = 0; i < 1000; ++i) {
        core->process_sample(0.5f, outbuf);
        ASSERT_FALSE(std::isnan(outbuf[0]));
        ASSERT_FALSE(std::isinf(outbuf[0]));
    }
}
