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
static std::unique_ptr<SSMWeights> make_test_weights() {
    auto w = std::make_unique<SSMWeights>();
    std::memset(w.get(), 0, sizeof(*w));
    for (size_t j = 0; j < MAX_D_MODEL && j < MAX_D_INNER; ++j) {
        w->in_proj_w[j][j] = 0.1f;
    }
    for (size_t h = 0; h < MAX_NHEADS; ++h) {
        w->dt_bias[h] = 0.5f;
    }
    for (size_t s = 0; s < MAX_D_STATE; ++s) {
        w->B_norm_weight[s] = 1.0f;
        w->C_norm_weight[s] = 1.0f;
    }
    for (size_t j = 0; j < MAX_D_MODEL && j < MAX_D_INNER; ++j) {
        w->out_proj_w[j][j] = 1.0f;
    }
    return w;
}

TEST(ConditioningModulation, SetConditioningDoesNotCrash) {
    auto block = make_test_block();
    block->set_weights(*make_test_weights());

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
    block->set_weights(*make_test_weights());

    float cond[COND_DIM] = {0.0f};
    block->set_conditioning(cond);  // all zero → inactive

    alignas(64) float outbuf[MAX_D_MODEL];
    block->process_sample(0.5f, outbuf);
    EXPECT_FALSE(std::isnan(outbuf[0]));
    EXPECT_FALSE(std::isinf(outbuf[0]));
}

// Deviation D6 regression: dt_film_proj_w must actually modulate Δ.
//
// make_test_weights() populates a small-model layout (rows/cols < MAX_D_MODEL),
// so it leaves the x projection and the B/C projections at zero.  This fixture
// first makes the recurrence live: x rows driven by the input, B/C driven by
// their biases.  film_proj_w stays zero, so with a zero dt_film_proj_w the
// conditioning vector can have no effect at all, and any output difference must
// come from the Δ-FiLM path (per-head gamma/beta, rows 2h / 2h+1).
//
// NOTE: the DC gain of the trapezoidal update is 1/|A| and is independent of Δ,
// so a converged constant-input response hides Δ.  Drive an alternating input
// and compare the whole trajectory instead.
static void make_dt_film_test_weights(SSMWeights& w) {
    std::memset(&w, 0, sizeof(w));

    // z rows (d_inner): small constant -> SiLU gate != 0
    for (size_t i = 0; i < MAX_D_INNER; ++i) w.in_proj_w[i][0] = 0.1f;
    // x rows (d_inner, offset d_inner): gain 1.0 -> x = input
    for (size_t i = 0; i < MAX_D_INNER; ++i) w.in_proj_w[MAX_D_INNER + i][0] = 1.0f;

    // B/C come from the biases because their projections are zero.
    for (size_t h = 0; h < MAX_NHEADS; ++h) {
        for (size_t r = 0; r < MAX_MIMO_RANK; ++r) {
            for (size_t s = 0; s < MAX_D_STATE; ++s) {
                w.B_bias[h][r][s] = 0.5f;
                w.C_bias[h][r][s] = 1.0f;
            }
        }
    }
    for (size_t s = 0; s < MAX_D_STATE; ++s) {
        w.B_norm_weight[s] = 1.0f;
        w.C_norm_weight[s] = 1.0f;
    }
    for (size_t h = 0; h < MAX_NHEADS; ++h) w.dt_bias[h] = 0.5f;
    for (size_t j = 0; j < MAX_D_MODEL && j < MAX_D_INNER; ++j) {
        w.out_proj_w[j][j] = 1.0f;
    }
}

TEST(ConditioningModulation, DtFilmModulatesDelta) {
    float cond[COND_DIM] = {0.0f};
    cond[0] = 0.5f;
    cond[1] = -0.25f;

    auto run = [](const SSMWeights& w, const float* c, int samples) {
        auto block = make_test_block();
        block->set_weights(w);
        if (c) block->set_conditioning(c);
        std::vector<float> ys(static_cast<size_t>(samples));
        for (int i = 0; i < samples; ++i) {
            alignas(64) float outbuf[MAX_D_MODEL];
            block->process_sample((i % 2) ? -0.4f : 0.7f, outbuf);
            ys[static_cast<size_t>(i)] = outbuf[0];
        }
        return ys;
    };

    // (a) Zero Δ-FiLM weights: conditioning is a no-op -> identical trajectory.
    SSMWeights w_zero;
    make_dt_film_test_weights(w_zero);
    EXPECT_EQ(run(w_zero, nullptr, 200), run(w_zero, cond, 200))
        << "zero dt_film_proj_w must leave unconditioned and conditioned "
           "inference identical";

    // (b) Non-zero Δ-FiLM weights: conditioning must change the trajectory.
    SSMWeights w_dt;
    make_dt_film_test_weights(w_dt);
    for (size_t h = 0; h < MAX_NHEADS; ++h) {
        for (size_t k = 0; k < COND_DIM; ++k) {
            w_dt.dt_film_proj_w[2 * h][k] = 0.5f;      // gamma
            w_dt.dt_film_proj_w[2 * h + 1][k] = 0.2f;  // beta
        }
    }
    const std::vector<float> base = run(w_dt, nullptr, 200);
    const std::vector<float> with_cond = run(w_dt, cond, 200);
    float max_diff = 0.0f;
    for (size_t i = 0; i < base.size(); ++i) {
        max_diff = std::max(max_diff, std::fabs(base[i] - with_cond[i]));
    }
    EXPECT_GT(max_diff, 1e-4f)
        << "Δ-FiLM weights are loaded but have no effect (deviation D6)";
}

TEST(ConditioningModulation, NonZeroConditioningWorks) {
    auto block = make_test_block();
    block->set_weights(*make_test_weights());

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

// ---------------------------------------------------------------------------
// Preset Management & Serialization Unit Tests
// ---------------------------------------------------------------------------

#include "clap/clap_plugin.h"
#include "clap/clap_presets.h"
#include "clap/param_table.h"
#include <filesystem>

namespace mamba {
// Provide load_weights implementation for test_conditioning target
bool MambaSynth::load_weights(const char* path) {
    if (!path || path[0] == '\0') return false;
    return true;
}
} // namespace mamba

namespace {

struct TestSynthDeleter {
    void operator()(MambaSynth* m) const {
        if (m) {
            if (m->core) {
                m->core->~SSMInferenceCore();
                m->core = nullptr;
            }
            delete m;
        }
    }
};
using TestSynthPtr = std::unique_ptr<MambaSynth, TestSynthDeleter>;

static TestSynthPtr create_test_synth() {
    auto m = std::unique_ptr<MambaSynth>(new (std::nothrow) MambaSynth());
    if (!m) return nullptr;
    m->host = nullptr;
    m->core_cfg = MambaConfig::max_config();
    m->core_cfg.num_layers = 1;
    m->core = new (m->core_storage.data) SSMInferenceCore(m->core_cfg);
    m->sample_rate = 48000.0;
    m->block_size = 512;
    m->is_active = false;
    m->is_processing = false;
    m->latency = 0;
    m->weights_loaded.store(false, std::memory_order_relaxed);
    m->next_note_id = 1;
    m->gui_data = nullptr;
    m->weight_path[0] = '\0';

    for (uint32_t v = 0; v < MAX_VOICES; ++v) {
        auto& voice = m->voices[v];
        std::memset(&voice, 0, sizeof(voice));
        voice.stage = VOICE_IDLE;
        voice.key = -1;
        voice.channel = -1;
        voice.attack_rate = MambaSynth::VOICE_ATTACK_RATE;
        voice.release_rate = MambaSynth::VOICE_RELEASE_RATE;
        auto& va = m->voice_atomics[v];
        va.stage.store(VOICE_IDLE, std::memory_order_relaxed);
        va.env.store(0.0f, std::memory_order_relaxed);
        va.key.store(-1, std::memory_order_relaxed);
        va.channel.store(-1, std::memory_order_relaxed);
        va.velocity.store(0.0f, std::memory_order_relaxed);
    }

    // Same metadata the plugin installs at init (range, default, curve,
    // smoothing time) -- the shared param table is the single source.
    init_param_smooth(m->params, kSynthParamTable);

    for (uint32_t i = 0; i < PARAM_COUNT; ++i) {
        m->param_values[i].store(m->params[i].default_value, std::memory_order_relaxed);
        m->params[i].target.store(m->params[i].default_value, std::memory_order_relaxed);
        m->params[i].current.store(m->params[i].default_value, std::memory_order_relaxed);
    }
    m->bypass_target.store(false, std::memory_order_relaxed);
    m->bypass_state.store(BYPASS_PROCESSING, std::memory_order_relaxed);
    m->bypass_fade.store(0.0f, std::memory_order_relaxed);
    m->next_note_id = 1;

    StateMirror* slot = m->state_mirror.begin_write();
    if (slot) {
        std::memset(slot, 0, sizeof(StateMirror));
        for (uint32_t i = 0; i < PARAM_COUNT; ++i) {
            slot->param_values[i] = m->params[i].default_value;
        }
        slot->bypass_target = false;
        slot->bypass_fade = 0.0f;
        slot->weight_path[0] = '\0';
        m->state_mirror.end_write();
    }

    return TestSynthPtr(m.release(), TestSynthDeleter());
}

} // namespace

TEST(PresetManagement, FactoryPresetsBank) {
    EXPECT_EQ(get_factory_preset_count(), 6u);

    const char* expected_names[6] = {
        "01 - Mamba Default",
        "02 - Dark Lead",
        "03 - Lush Pad",
        "04 - Resonant Sub",
        "05 - Fast Perc",
        "06 - Neural Delay"
    };

    const char* expected_categories[6] = {
        "Default",
        "Lead",
        "Pad",
        "Bass",
        "Percussion",
        "FX"
    };

    for (size_t i = 0; i < 6; ++i) {
        const FactoryPreset* fp = get_factory_preset(i);
        ASSERT_NE(fp, nullptr);
        EXPECT_STREQ(fp->name, expected_names[i]);
        EXPECT_STREQ(fp->category, expected_categories[i]);

        // Find by name
        const FactoryPreset* found = find_factory_preset(expected_names[i]);
        ASSERT_NE(found, nullptr);
        EXPECT_EQ(found, fp);

        // Every entry inside the table range for its ParamIds slot.
        ASSERT_EQ(fp->param_values.size(), static_cast<size_t>(PARAM_COUNT));
        for (uint32_t pi = 0; pi < PARAM_COUNT; ++pi) {
            EXPECT_GE(fp->param_values[pi], kSynthParamTable[pi].min_value)
                << "preset " << i << " param " << pi;
            EXPECT_LE(fp->param_values[pi], kSynthParamTable[pi].max_value)
                << "preset " << i << " param " << pi;
        }
    }

    EXPECT_EQ(get_factory_preset(6), nullptr);
    EXPECT_EQ(find_factory_preset("Unknown Preset"), nullptr);
}

TEST(PresetManagement, ApplyFactoryPresets) {
    auto synth = create_test_synth();
    ASSERT_NE(synth, nullptr);

    // Apply by index: "02 - Dark Lead"
    EXPECT_TRUE(apply_factory_preset(synth.get(), 1));
    EXPECT_NEAR(synth->param_values[PARAM_GAIN].load(), 1.2, 1e-6);
    EXPECT_NEAR(synth->param_values[PARAM_ATTACK].load(), 0.015, 1e-6);
    EXPECT_NEAR(synth->param_values[PARAM_RELEASE].load(), 0.4, 1e-6);
    EXPECT_NEAR(synth->param_values[PARAM_MIX].load(), 0.85, 1e-6);
    EXPECT_NEAR(synth->param_values[PARAM_BYPASS].load(), 0.0, 1e-6);

    // Apply by name: "03 - Lush Pad"
    EXPECT_TRUE(apply_factory_preset(synth.get(), "03 - Lush Pad"));
    EXPECT_NEAR(synth->param_values[PARAM_GAIN].load(), 0.9, 1e-6);
    EXPECT_NEAR(synth->param_values[PARAM_ATTACK].load(), 0.6, 1e-6);
    EXPECT_NEAR(synth->param_values[PARAM_RELEASE].load(), 1.5, 1e-6);
    EXPECT_NEAR(synth->param_values[PARAM_MIX].load(), 1.0, 1e-6);

    // Apply by name: "04 - Resonant Sub"
    EXPECT_TRUE(apply_factory_preset(synth.get(), "04 - Resonant Sub"));
    EXPECT_NEAR(synth->param_values[PARAM_GAIN].load(), 1.4, 1e-6);
    EXPECT_NEAR(synth->param_values[PARAM_ATTACK].load(), 0.005, 1e-6);
    EXPECT_NEAR(synth->param_values[PARAM_RELEASE].load(), 0.25, 1e-6);

    // Apply by name: "05 - Fast Perc"
    EXPECT_TRUE(apply_factory_preset(synth.get(), "05 - Fast Perc"));
    EXPECT_NEAR(synth->param_values[PARAM_ATTACK].load(), 0.001, 1e-6);
    EXPECT_NEAR(synth->param_values[PARAM_RELEASE].load(), 0.05, 1e-6);

    // Apply by name: "06 - Neural Delay"
    EXPECT_TRUE(apply_factory_preset(synth.get(), "06 - Neural Delay"));
    EXPECT_NEAR(synth->param_values[PARAM_MIX].load(), 0.5, 1e-6);

    // Invalid applications
    EXPECT_FALSE(apply_factory_preset(synth.get(), 99));
    EXPECT_FALSE(apply_factory_preset(synth.get(), "NonExistent"));
    EXPECT_FALSE(apply_factory_preset(nullptr, static_cast<size_t>(0)));
}

TEST(PresetManagement, BinaryPresetSaveLoadRoundTrip) {
    auto synth1 = create_test_synth();
    ASSERT_NE(synth1, nullptr);

    synth1->param_values[PARAM_GAIN].store(1.35);
    synth1->param_values[PARAM_ATTACK].store(0.045);
    synth1->param_values[PARAM_RELEASE].store(0.750);
    synth1->param_values[PARAM_MIX].store(0.60);
    synth1->param_values[PARAM_BYPASS].store(1.0);
    synth1->bypass_target.store(true);

    auto tmp_file = std::filesystem::temp_directory_path() / "test_mamba_preset.mambapreset";
    std::string path_str = tmp_file.string();

    EXPECT_TRUE(save_preset(synth1.get(), path_str.c_str(), "Test Binary Preset"));

    auto synth2 = create_test_synth();
    ASSERT_NE(synth2, nullptr);

    EXPECT_TRUE(load_preset(synth2.get(), path_str.c_str()));

    EXPECT_NEAR(synth2->param_values[PARAM_GAIN].load(), 1.35, 1e-6);
    EXPECT_NEAR(synth2->param_values[PARAM_ATTACK].load(), 0.045, 1e-6);
    EXPECT_NEAR(synth2->param_values[PARAM_RELEASE].load(), 0.750, 1e-6);
    EXPECT_NEAR(synth2->param_values[PARAM_MIX].load(), 0.60, 1e-6);
    EXPECT_NEAR(synth2->param_values[PARAM_BYPASS].load(), 1.0, 1e-6);
    EXPECT_TRUE(synth2->bypass_target.load());

    std::error_code ec;
    std::filesystem::remove(tmp_file, ec);
}

TEST(PresetManagement, JsonPresetExportImportFileRoundTrip) {
    auto synth1 = create_test_synth();
    ASSERT_NE(synth1, nullptr);

    synth1->param_values[PARAM_GAIN].store(0.82);
    synth1->param_values[PARAM_ATTACK].store(0.125);
    synth1->param_values[PARAM_RELEASE].store(1.400);
    synth1->param_values[PARAM_MIX].store(0.90);
    synth1->param_values[PARAM_BYPASS].store(0.0);
    synth1->bypass_target.store(false);

    auto tmp_file = std::filesystem::temp_directory_path() / "test_mamba_preset.json";
    std::string path_str = tmp_file.string();

    EXPECT_TRUE(export_preset_json(synth1.get(), path_str.c_str(), "My JSON Preset", "Custom Category"));

    auto synth2 = create_test_synth();
    ASSERT_NE(synth2, nullptr);

    EXPECT_TRUE(import_preset_json(synth2.get(), path_str.c_str()));

    EXPECT_NEAR(synth2->param_values[PARAM_GAIN].load(), 0.82, 1e-6);
    EXPECT_NEAR(synth2->param_values[PARAM_ATTACK].load(), 0.125, 1e-6);
    EXPECT_NEAR(synth2->param_values[PARAM_RELEASE].load(), 1.400, 1e-6);
    EXPECT_NEAR(synth2->param_values[PARAM_MIX].load(), 0.90, 1e-6);
    EXPECT_NEAR(synth2->param_values[PARAM_BYPASS].load(), 0.0, 1e-6);
    EXPECT_FALSE(synth2->bypass_target.load());

    std::error_code ec;
    std::filesystem::remove(tmp_file, ec);
}

TEST(PresetManagement, JsonStringExportImportAndValidation) {
    auto synth1 = create_test_synth();
    ASSERT_NE(synth1, nullptr);

    std::string json_str;
    EXPECT_TRUE(export_preset_json_string(synth1.get(), "Custom String Preset", "Experimental", json_str));
    EXPECT_NE(json_str.find("\"format\": \"mamba-preset\""), std::string::npos);
    EXPECT_NE(json_str.find("\"name\": \"Custom String Preset\""), std::string::npos);
    EXPECT_NE(json_str.find("\"category\": \"Experimental\""), std::string::npos);

    auto synth2 = create_test_synth();
    ASSERT_NE(synth2, nullptr);

    EXPECT_TRUE(import_preset_json_string(synth2.get(), json_str));

    // Reject broken JSON syntax
    EXPECT_FALSE(import_preset_json_string(synth2.get(), "{ broken json"));
    EXPECT_FALSE(import_preset_json_string(synth2.get(), ""));

    // Reject out-of-bounds parameter values
    EXPECT_FALSE(import_preset_json_string(synth2.get(), R"({"parameters": {"gain": 99.0}})"));
    EXPECT_FALSE(import_preset_json_string(synth2.get(), R"({"parameters": {"attack": 0.00001}})"));
}

TEST(PresetManagement, ExportFactoryPresetJson) {
    std::string json;
    EXPECT_TRUE(export_factory_preset_json(kFactoryPresets[1], json));
    EXPECT_NE(json.find("\"02 - Dark Lead\""), std::string::npos);
    EXPECT_NE(json.find("\"Lead\""), std::string::npos);

    auto synth = create_test_synth();
    ASSERT_NE(synth, nullptr);

    EXPECT_TRUE(import_preset_json_string(synth.get(), json));
    EXPECT_NEAR(synth->param_values[PARAM_GAIN].load(), 1.2, 1e-6);
    EXPECT_NEAR(synth->param_values[PARAM_ATTACK].load(), 0.015, 1e-6);
    EXPECT_NEAR(synth->param_values[PARAM_RELEASE].load(), 0.4, 1e-6);
    EXPECT_NEAR(synth->param_values[PARAM_MIX].load(), 0.85, 1e-6);
}
