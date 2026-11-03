// Stage 03 acceptance tests: MXCSR restore (E8/E19) and per-instance
// conditioning independence (E15).
#include <gtest/gtest.h>
#include "core/ssm_block.h"
#include "core/ssm_inference_core.h"
#include "config.h"

#include <memory>

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64)
#include <xmmintrin.h>
#endif

using namespace mamba;

namespace {
MambaConfig small_config() {
    MambaConfig cfg{};
    cfg.d_state = 16;
    cfg.d_model = 16;
    cfg.d_inner = 32;
    cfg.nheads = 2;
    cfg.headdim = 8;
    cfg.mimo_rank = 1;
    cfg.num_bc_heads = 1;
    cfg.num_rope_angles = 8;
    cfg.num_layers = 2;
    cfg.cond_dim = COND_DIM;
    return cfg;
}
} // namespace

// The FPU guard must set FTZ/DAZ for the block and restore the previous MXCSR
// on scope exit so the host's FPU environment is untouched (E8, E19).
TEST(RtSafety, FtzDazRestoresMxcsr) {
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64)
    const unsigned int before = _mm_getcsr();
    {
        ScopedFtzDaz guard;
        const unsigned int during = _mm_getcsr();
        EXPECT_EQ(during & 0x8040u, 0x8040u);  // FTZ + DAZ both set
    }
    EXPECT_EQ(_mm_getcsr(), before);            // restored exactly
#else
    GTEST_SKIP() << "MXCSR test is x86-only";
#endif
}

// Two cores with different conditioning must produce different output: the
// conditioning state lives in each core (and therefore in each plugin
// instance), never in a shared global (E15).
TEST(RtSafety, TwoInstancesIndependentConditioning) {
    auto a = std::make_unique<SSMInferenceCore>(small_config());
    auto b = std::make_unique<SSMInferenceCore>(small_config());

    SSMWeights w{};
    w.D_proj[0] = 1.0f;          // skip passthrough so output tracks input
    w.film_proj_w[0][0] = 1.0f;  // gamma for output 0 = cond[0]
    for (size_t l = 0; l < small_config().num_layers; ++l) {
        a->set_layer_weights(l, w);
        b->set_layer_weights(l, w);
    }

    float cond_a[COND_DIM] = {0};
    float cond_b[COND_DIM] = {0};
    cond_a[0] = 1.0f;   // instance A: FiLM gamma = 1
    a->set_conditioning(cond_a);
    b->set_conditioning(cond_b);  // instance B: no conditioning

    alignas(64) float oa[MAX_D_MODEL];
    alignas(64) float ob[MAX_D_MODEL];
    a->process_sample(0.5f, oa);
    b->process_sample(0.5f, ob);

    EXPECT_NE(oa[0], ob[0]);  // conditioning is per-instance
}
