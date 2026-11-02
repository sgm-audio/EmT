// RT-safety stress: audio-thread process() vs main-thread weight/state load.
//
// Reproduces the data races from ISSUES.md E6 (plain bool weights_loaded_ /
// weights_ memcpy read by the audio thread), E12 (load_weights re-inits the
// shared core in place while process() runs) and the state-mirror aspect of
// the state save path (get_state on "main" vs process on the audio thread).
//
// Chain/03 task 1: written BEFORE the fix. Built with
//   -DCMAKE_CXX_FLAGS="-fsanitize=thread -g -O1"
// it must REPORT data races on the base commit; after the fix the same
// scenarios run through the publish/retire API with zero reports (the
// base-API in-place-init scenario is then forbidden by contract and was
// replaced - see the stage-03 PR body).
//
// Duration: EMT_RT_STRESS_SECONDS (default 10).
#include <gtest/gtest.h>
#include "core/ssm_block.h"
#include "core/ssm_inference_core.h"
#include "config.h"

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <memory>
#include <thread>
#include <vector>

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

double stress_seconds() {
    // 10 s for the dedicated TSan runs (EMT_RT_STRESS_SECONDS=10); short by
    // default so the regular ctest suite stays fast.
    const char* env = std::getenv("EMT_RT_STRESS_SECONDS");
    return env ? std::atof(env) : 2.0;
}

} // namespace

// Thread A: process() in a loop with real buffers (per-voice states).
// Thread B: loops what load_weights / state load do today:
//   - core.init(cfg)  (in-place layer reinit - the clap load_weights path)
//   - layer.set_weights(w) on a live layer (set_layer_weights path)
//   - get_state / set_state (state save / reload path)
TEST(RtStress, BaseApiProcessVsWeightAndStateLoad) {
    auto core = std::make_unique<SSMInferenceCore>(small_config());  // heap: the core is ~6 MB
    std::atomic<bool> stop{false};
    std::atomic<long> audio_blocks{0};
    std::atomic<long> load_cycles{0};

    std::thread audio([&] {
        std::vector<MambaState> states(4);
        std::vector<float> in(64), out(64 * MAX_D_MODEL);
        for (float& v : in) v = 0.25f;
        while (!stop.load(std::memory_order_relaxed)) {
            for (size_t i = 0; i < in.size(); ++i) {
                core->process_sample(in[i], out.data() + i * MAX_D_MODEL,
                                     states[i % states.size()]);
            }
            audio_blocks.fetch_add(1, std::memory_order_relaxed);
        }
    });

    std::thread loader([&] {
        const auto deadline = std::chrono::steady_clock::now()
                            + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                                  std::chrono::duration<double>(stress_seconds()));
        auto w = std::make_unique<SSMWeights>();  // ~4 MB: heap, never the stack (E7)
        w->D_proj[0] = 0.5f;  // non-zero so the D-skip path reads real memory
        auto snap = std::make_unique<MambaState>();  // too big for the stack (E7 disease)
        while (std::chrono::steady_clock::now() < deadline) {
            core->init(small_config());                   // in-place reinit (E12)
            core->layer(0).set_weights(*w);               // live-layer weight copy (E6)
            core->get_state(*snap);                       // state save path
            core->set_state(*snap);                       // state load path
            load_cycles.fetch_add(1, std::memory_order_relaxed);
        }
        stop.store(true, std::memory_order_relaxed);
    });

    loader.join();
    audio.join();

    EXPECT_GT(audio_blocks.load(), 0);
    EXPECT_GT(load_cycles.load(), 0);
    std::printf("stress: %ld audio blocks vs %ld load cycles\n",
                audio_blocks.load(), load_cycles.load());
}
