// Stage 05: state save -> load -> save byte-equality (E3).
#include <gtest/gtest.h>
#include "clap/clap_plugin.h"
#include "clap/clap_common_utils.h"
#include <clap/clap.h>
#include <cstring>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <vector>

using namespace mamba;

namespace mamba {
// clap_state.cpp calls this; the round-trip test never loads weights
// (weights_loaded stays false), so a stub satisfies the linker.
bool MambaSynth::load_weights(const char* path) {
    (void)path;
    return false;
}
} // namespace mamba

namespace {

// Minimal in-memory CLAP streams.
struct MemStream {
    std::vector<uint8_t> data;
    size_t read_pos = 0;
};

int64_t ostream_write(const clap_ostream_t* s, const void* buffer, uint64_t size) {
    auto* m = static_cast<MemStream*>(s->ctx);
    const uint8_t* p = static_cast<const uint8_t*>(buffer);
    m->data.insert(m->data.end(), p, p + size);
    return static_cast<int64_t>(size);
}

int64_t istream_read(const clap_istream_t* s, void* buffer, uint64_t size) {
    auto* m = static_cast<MemStream*>(s->ctx);
    if (m->read_pos >= m->data.size()) return 0;
    uint64_t n = std::min<uint64_t>(size, m->data.size() - m->read_pos);
    std::memcpy(buffer, m->data.data() + m->read_pos, n);
    m->read_pos += n;
    return static_cast<int64_t>(n);
}

clap_ostream_t make_ostream(MemStream& m) {
    clap_ostream_t s{};
    s.ctx = &m;
    s.write = ostream_write;
    return s;
}
clap_istream_t make_istream(MemStream& m) {
    clap_istream_t s{};
    s.ctx = &m;
    s.read = istream_read;
    return s;
}

struct TestSynthDeleter {
    void operator()(MambaSynth* m) const {
        if (m) { m->core.destroy(); delete m; }
    }
};
using TestSynthPtr = std::unique_ptr<MambaSynth, TestSynthDeleter>;

TestSynthPtr make_synth() {
    auto m = TestSynthPtr(new (std::nothrow) MambaSynth());
    if (!m) return nullptr;
    m->host = nullptr;
    m->core_cfg = MambaConfig::max_config();
    m->core_cfg.num_layers = 1;
    m->core.publish(new SSMInferenceCore(m->core_cfg));
    m->sample_rate = 48000.0;
    m->block_size = 512;
    m->is_active = false;
    m->is_processing = false;
    m->latency = 0;
    m->weights_loaded.store(false, std::memory_order_relaxed);
    m->next_note_id = 1;
    m->gui_data = nullptr;
    m->weight_path[0] = '\0';
    return m;
}

} // namespace

// The validator's state-reproducibility-basic: set params, save, reload into a
// fresh instance, save again, and compare byte-for-byte (and the parameter
// values must round-trip).
TEST(StateRoundtrip, SaveLoadSaveIsByteIdentical) {
    // s_state is already declared extern in clap_plugin.h; use it directly.
    auto a = make_synth();
    ASSERT_NE(a, nullptr);
    // Simulate the end-of-process publish_state_mirror: fill the back slot of
    // the double-buffer seqlock with non-default params (what the validator's
    // randomized params + process() would produce), then make it readable.
    a->param_values[PARAM_GAIN].store(0.1112, std::memory_order_release);
    a->param_values[PARAM_MIX].store(0.1821, std::memory_order_release);
    a->param_values[PARAM_BYPASS].store(1.0, std::memory_order_release);
    a->bypass_target.store(true, std::memory_order_release);
    {
        StateMirror* slot = a->state_mirror.begin_write();
        for (uint32_t i = 0; i < PARAM_COUNT; ++i) {
            slot->param_values[i] = a->param_values[i].load(std::memory_order_acquire);
        }
        slot->bypass_target = a->bypass_target.load(std::memory_order_acquire);
        slot->bypass_fade = 0.0f;
        std::memset(slot->weight_path, 0, sizeof(slot->weight_path));
        for (uint32_t v = 0; v < MAX_VOICES; ++v) {
            std::memset(&slot->voice_snapshots[v], 0, sizeof(slot->voice_snapshots[v]));
            slot->voice_snapshots[v].key = -1;
            slot->voice_snapshots[v].channel = -1;
        }
        a->state_mirror.end_write();
    }
    a->state_mirror.swap();

    clap_plugin_t plugin_a{};
    plugin_a.plugin_data = a.get();

    MemStream stream_a;
    clap_ostream_t oa = make_ostream(stream_a);
    ASSERT_TRUE(s_state.save(&plugin_a, &oa)) << "first save failed";

    // Fresh instance, load, save.
    auto b = make_synth();
    ASSERT_NE(b, nullptr);
    clap_plugin_t plugin_b{};
    plugin_b.plugin_data = b.get();

    MemStream load_stream = stream_a;  // copy bytes
    clap_istream_t ib = make_istream(load_stream);
    ASSERT_TRUE(s_state.load(&plugin_b, &ib)) << "load failed";

    MemStream stream_b;
    clap_ostream_t ob = make_ostream(stream_b);
    ASSERT_TRUE(s_state.save(&plugin_b, &ob)) << "second save failed";

    ASSERT_EQ(stream_a.data.size(), stream_b.data.size())
        << "saved size differs";
    {
        // debug: dump the first param double in each save
        double da = 0, db = 0;
        if (stream_a.data.size() >= 32) std::memcpy(&da, stream_a.data.data() + 24, 8);
        if (stream_b.data.size() >= 32) std::memcpy(&db, stream_b.data.data() + 24, 8);
        printf("debug: saveA param0=%g saveB param0=%g (size A=%zu B=%zu)\n",
               da, db, stream_a.data.size(), stream_b.data.size());
    }
    bool identical = stream_a.data == stream_b.data;
    if (!identical) {
        size_t first = 0;
        while (first < stream_a.data.size() && first < stream_b.data.size() &&
               stream_a.data[first] == stream_b.data[first]) ++first;
        printf("first differing byte at offset %zu (A=%02x B=%02x)\n",
               first, first < stream_a.data.size() ? stream_a.data[first] : 0,
               first < stream_b.data.size() ? stream_b.data[first] : 0);
    }
    EXPECT_TRUE(identical) << "save->load->save is not byte-identical";

    // Parameter values must round-trip.
    EXPECT_NEAR(b->param_values[PARAM_GAIN].load(), 0.1112, 1e-9);
    EXPECT_NEAR(b->param_values[PARAM_MIX].load(), 0.1821, 1e-9);
    EXPECT_NEAR(b->param_values[PARAM_BYPASS].load(), 1.0, 1e-9);
}
