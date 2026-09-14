#include <gtest/gtest.h>
#include "core/ssm_block.h"
#include "core/ssm_inference_core.h"
#include "common/seqlock.h"
#include "common/conditioning_frame.h"
#include "config.h"
#include "clap/clap_fx_plugin.h"
#include <cmath>
#include <memory>
#include <vector>
#include <cstring>
#include <algorithm>

using namespace mamba;

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

TEST(RealtimeSafety, ZeroCopyVoiceStateIsolation) {
    auto core = std::make_unique<SSMInferenceCore>();
    MambaConfig cfg = MambaConfig::max_config();
    cfg.num_layers = 2;
    core->init(cfg);

    auto voice1 = std::make_unique<MambaState>();
    auto voice2 = std::make_unique<MambaState>();
    std::memset(voice1.get(), 0, sizeof(MambaState));
    std::memset(voice2.get(), 0, sizeof(MambaState));

    alignas(64) float out1[MAX_D_MODEL];
    alignas(64) float out2[MAX_D_MODEL];

    for (int i = 0; i < 100; ++i) {
        core->process_sample(1.0f, out1, *voice1);
        core->process_sample(0.0f, out2, *voice2);
        ASSERT_FALSE(std::isnan(out1[0]));
        ASSERT_FALSE(std::isnan(out2[0]));
        EXPECT_NEAR(out2[0], 0.0f, 1e-6f);
    }
}

TEST(RealtimeSafety, FxStateSaveLoadRoundtrip) {
    const clap_plugin_t* plugin1 = mamba_fx_create(nullptr);
    ASSERT_NE(plugin1, nullptr);
    ASSERT_TRUE(plugin1->init(plugin1));

    auto* state_ext1 = static_cast<const clap_plugin_state_t*>(
        plugin1->get_extension(plugin1, CLAP_EXT_STATE));
    ASSERT_NE(state_ext1, nullptr);
    ASSERT_NE(state_ext1->save, nullptr);
    ASSERT_NE(state_ext1->load, nullptr);

    auto* fx1 = static_cast<FxState*>(plugin1->plugin_data);
    ASSERT_NE(fx1, nullptr);

    // Modify FX state
    const double test_gain = 1.75;
    const double test_mix = 0.40;
    const double test_bypass = 1.0;
    const char* test_path = "models/test_model.bin";

    fx1->param_values[FX_PARAM_GAIN].store(test_gain, std::memory_order_relaxed);
    fx1->params[FX_PARAM_GAIN].target.store(test_gain, std::memory_order_relaxed);
    fx1->params[FX_PARAM_GAIN].current.store(test_gain, std::memory_order_relaxed);

    fx1->param_values[FX_PARAM_MIX].store(test_mix, std::memory_order_relaxed);
    fx1->params[FX_PARAM_MIX].target.store(test_mix, std::memory_order_relaxed);
    fx1->params[FX_PARAM_MIX].current.store(test_mix, std::memory_order_relaxed);

    fx1->param_values[FX_PARAM_BYPASS].store(test_bypass, std::memory_order_relaxed);
    fx1->params[FX_PARAM_BYPASS].target.store(test_bypass, std::memory_order_relaxed);
    fx1->params[FX_PARAM_BYPASS].current.store(test_bypass, std::memory_order_relaxed);
    fx1->bypass_target.store(true, std::memory_order_relaxed);

    std::strncpy(fx1->weight_path, test_path, sizeof(fx1->weight_path) - 1);
    fx1->weight_path[sizeof(fx1->weight_path) - 1] = '\0';

    // In-memory stream implementation
    struct MemoryStream {
        std::vector<uint8_t> buffer;
        size_t read_pos = 0;

        static int64_t CLAP_ABI write_fn(const clap_ostream* stream, const void* buffer, uint64_t size) {
            auto* self = static_cast<MemoryStream*>(stream->ctx);
            const auto* src = static_cast<const uint8_t*>(buffer);
            self->buffer.insert(self->buffer.end(), src, src + size);
            return static_cast<int64_t>(size);
        }

        static int64_t CLAP_ABI read_fn(const clap_istream* stream, void* buffer, uint64_t size) {
            auto* self = static_cast<MemoryStream*>(stream->ctx);
            if (self->read_pos >= self->buffer.size()) return 0;
            size_t to_read = std::min<size_t>(size, self->buffer.size() - self->read_pos);
            std::memcpy(buffer, self->buffer.data() + self->read_pos, to_read);
            self->read_pos += to_read;
            return static_cast<int64_t>(to_read);
        }
    } mem;

    clap_ostream ostream{&mem, MemoryStream::write_fn};
    EXPECT_TRUE(state_ext1->save(plugin1, &ostream));
    EXPECT_GT(mem.buffer.size(), 0u);

    // Verify magic 'MFXS' = 0x4D465853 in stream header
    ASSERT_GE(mem.buffer.size(), sizeof(uint32_t) * 2);
    uint32_t magic = 0;
    std::memcpy(&magic, mem.buffer.data(), sizeof(uint32_t));
    EXPECT_EQ(magic, 0x4D465853u);

    uint32_t version = 0;
    std::memcpy(&version, mem.buffer.data() + sizeof(uint32_t), sizeof(uint32_t));
    EXPECT_EQ(version, 1u);

    // Create a second plugin instance to verify load
    const clap_plugin_t* plugin2 = mamba_fx_create(nullptr);
    ASSERT_NE(plugin2, nullptr);
    ASSERT_TRUE(plugin2->init(plugin2));

    auto* state_ext2 = static_cast<const clap_plugin_state_t*>(
        plugin2->get_extension(plugin2, CLAP_EXT_STATE));
    ASSERT_NE(state_ext2, nullptr);

    clap_istream istream{&mem, MemoryStream::read_fn};
    EXPECT_TRUE(state_ext2->load(plugin2, &istream));

    auto* fx2 = static_cast<FxState*>(plugin2->plugin_data);
    ASSERT_NE(fx2, nullptr);

    EXPECT_DOUBLE_EQ(fx2->param_values[FX_PARAM_GAIN].load(std::memory_order_acquire), test_gain);
    EXPECT_DOUBLE_EQ(fx2->param_values[FX_PARAM_MIX].load(std::memory_order_acquire), test_mix);
    EXPECT_DOUBLE_EQ(fx2->param_values[FX_PARAM_BYPASS].load(std::memory_order_acquire), test_bypass);
    EXPECT_TRUE(fx2->bypass_target.load(std::memory_order_acquire));
    EXPECT_STREQ(fx2->weight_path, test_path);

    // Also verify via CLAP params extension get_value
    auto* params_ext2 = static_cast<const clap_plugin_params_t*>(
        plugin2->get_extension(plugin2, CLAP_EXT_PARAMS));
    ASSERT_NE(params_ext2, nullptr);
    double val_gain = 0.0, val_mix = 0.0, val_bypass = 0.0;
    EXPECT_TRUE(params_ext2->get_value(plugin2, FX_PARAM_GAIN, &val_gain));
    EXPECT_TRUE(params_ext2->get_value(plugin2, FX_PARAM_MIX, &val_mix));
    EXPECT_TRUE(params_ext2->get_value(plugin2, FX_PARAM_BYPASS, &val_bypass));
    EXPECT_DOUBLE_EQ(val_gain, test_gain);
    EXPECT_DOUBLE_EQ(val_mix, test_mix);
    EXPECT_DOUBLE_EQ(val_bypass, test_bypass);

    // Verify corruption rejection: invalid magic
    {
        MemoryStream bad_magic_mem;
        bad_magic_mem.buffer = mem.buffer;
        uint32_t bad_magic = 0x12345678;
        std::memcpy(bad_magic_mem.buffer.data(), &bad_magic, sizeof(uint32_t));
        clap_istream bad_stream{&bad_magic_mem, MemoryStream::read_fn};
        EXPECT_FALSE(state_ext2->load(plugin2, &bad_stream));
    }

    // Verify corruption rejection: truncated stream
    {
        MemoryStream trunc_mem;
        trunc_mem.buffer.assign(mem.buffer.begin(), mem.buffer.begin() + 4);
        clap_istream trunc_stream{&trunc_mem, MemoryStream::read_fn};
        EXPECT_FALSE(state_ext2->load(plugin2, &trunc_stream));
    }

    plugin1->destroy(plugin1);
    plugin2->destroy(plugin2);
}


