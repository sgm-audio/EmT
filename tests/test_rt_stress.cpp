// RT-safety stress: audio-thread process() vs main-thread weight load.
//
// The fixed contract (stage 03): the inference core is swapped RCU-style via
// RcuSlot. The main thread builds a fresh core (new SSMInferenceCore + weight
// copies) and publish()es it; the audio thread adopt()s the latest at the
// start of each process block and never mutates it. The previous core is
// reclaimed on the main thread once the audio thread reports it retired.
//
// Built with -fsanitize=thread this must run clean (0 reports). The base-API
// version that reproduced the E6/E12 races (in-place core->init() and
// set_weights() on a live core while process() ran) is in the stage-03 commit
// history; those patterns are now forbidden by contract.
//
// Duration: EMT_RT_STRESS_SECONDS (default 2).
#include <gtest/gtest.h>
#include "common/rcu_slot.h"
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
    const char* env = std::getenv("EMT_RT_STRESS_SECONDS");
    return env ? std::atof(env) : 2.0;
}

// Build a fresh core with non-zero weights so the audio thread reads real
// memory on every step (mirrors load_weights).
SSMInferenceCore* build_core() {
    auto* core = new SSMInferenceCore(small_config());
    SSMWeights w{};
    w.D_proj[0] = 0.5f;
    for (size_t layer = 0; layer < small_config().num_layers; ++layer) {
        core->set_layer_weights(layer, w);
    }
    return core;
}

} // namespace

TEST(RtStress, PublishCoresWhileProcessing) {
    RcuSlot<SSMInferenceCore> slot;
    slot.publish(build_core());

    std::atomic<bool> stop{false};
    std::atomic<long> audio_blocks{0};
    std::atomic<long> load_cycles{0};

    std::thread audio([&] {
        std::vector<MambaState> states(4);
        std::vector<float> in(64), out(64 * MAX_D_MODEL);
        for (float& v : in) v = 0.25f;
        while (!stop.load(std::memory_order_relaxed)) {
            SSMInferenceCore* core = slot.adopt();
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
        while (std::chrono::steady_clock::now() < deadline) {
            slot.reclaim();                       // free previously retired cores
            slot.publish(build_core());           // main thread builds + publishes
            load_cycles.fetch_add(1, std::memory_order_relaxed);
        }
        stop.store(true, std::memory_order_relaxed);
    });

    loader.join();
    audio.join();
    slot.destroy();

    EXPECT_GT(audio_blocks.load(), 0);
    EXPECT_GT(load_cycles.load(), 0);
    std::printf("stress: %ld audio blocks vs %ld publishes\n",
                audio_blocks.load(), load_cycles.load());
}
