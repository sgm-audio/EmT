#include "clap/clap_plugin.h"
#include "core/weight_loader.h"
#include <clap/clap.h>
#include <cstring>
#include <cstdio>

namespace mamba {

static MambaSynth* get_synth(const clap_plugin_t* plugin) {
    return static_cast<MambaSynth*>(plugin->plugin_data);
}

// State stream format (binary, little-endian):
//   uint32 magic   = 0x4D424D53   'MBM S'  -- "Mamba Synth State"
//   uint32 version = (MAJ<<16)|(MIN<<8)|PATCH
//   uint32 reserved = 0           (alignment / future flags)
//   uint32 schema  = SCHEMA_V1
//   uint32 param_count
//   double param_values[param_count]
//   uint32 bypass_target (0/1)
//   float  bypass_fade
//   uint32 h_state_present (0/1)
//   MambaState h_state (if present)
//   uint32 weight_path_len
//   char   weight_path[weight_path_len]  (no terminator in stream)
//
// The first 4 fields are 16 bytes (4 * uint32). All offsets are
// stable so partial reads/writes from a buffered stream still work.

static constexpr uint32_t kStateMagic   = 0x4D424D53u; // 'MBM S'
static constexpr uint32_t kStateVersion =
    (VERSION_MAJOR << 16) | (VERSION_MINOR << 8) | VERSION_PATCH;
static constexpr uint32_t kStateSchemaV1 = 0x00010000u;

static bool stream_write_all(const clap_ostream_t* stream, const void* data, uint64_t size) {
    const uint8_t* ptr = static_cast<const uint8_t*>(data);
    uint64_t remaining = size;
    while (remaining > 0) {
        int64_t written = stream->write(stream, ptr, remaining);
        if (written < 0) return false;
        if (written == 0) return false;
        ptr += written;
        remaining -= static_cast<uint64_t>(written);
    }
    return true;
}

static bool stream_read_all(const clap_istream_t* stream, void* data, uint64_t size) {
    uint8_t* ptr = static_cast<uint8_t*>(data);
    uint64_t remaining = size;
    while (remaining > 0) {
        int64_t rd = stream->read(stream, ptr, remaining);
        if (rd < 0) return false;
        if (rd == 0) return false;
        ptr += rd;
        remaining -= static_cast<uint64_t>(rd);
    }
    return true;
}

static bool stream_write_u32(const clap_ostream_t* s, uint32_t v) {
    return stream_write_all(s, &v, sizeof(v));
}
static bool stream_read_u32(const clap_istream_t* s, uint32_t& v) {
    return stream_read_all(s, &v, sizeof(v));
}
static bool stream_write_f32(const clap_ostream_t* s, float v) {
    return stream_write_all(s, &v, sizeof(v));
}
static bool stream_read_f32(const clap_istream_t* s, float& v) {
    return stream_read_all(s, &v, sizeof(v));
}
static bool stream_write_f64(const clap_ostream_t* s, double v) {
    return stream_write_all(s, &v, sizeof(v));
}
static bool stream_read_f64(const clap_istream_t* s, double& v) {
    return stream_read_all(s, &v, sizeof(v));
}

static bool state_save(const clap_plugin_t* plugin, const clap_ostream_t* stream) {
    if (!plugin || !stream) return false;
    auto* m = get_synth(plugin);

    if (!stream_write_u32(stream, kStateMagic)) return false;
    if (!stream_write_u32(stream, kStateVersion)) return false;
    if (!stream_write_u32(stream, 0)) return false;             // reserved
    if (!stream_write_u32(stream, kStateSchemaV1)) return false;

    // Pull a fresh state-mirror snapshot from the audio thread.
    StateMirror mirror;
    {
        constexpr int kMaxRetries = 8;
        bool ok = false;
        for (int retry = 0; retry < kMaxRetries; ++retry) {
            std::memset(&mirror, 0, sizeof(mirror));
            if (m->state_mirror.read(mirror)) { ok = true; break; }
        }
        if (!ok) return false;
    }

    if (!stream_write_u32(stream, PARAM_COUNT)) return false;
    for (uint32_t i = 0; i < PARAM_COUNT; ++i) {
        if (!stream_write_f64(stream, mirror.param_values[i])) return false;
    }
    if (!stream_write_u32(stream, mirror.bypass_target ? 1u : 0u)) return false;
    if (!stream_write_f32(stream, mirror.bypass_fade)) return false;

    // h_state is only present if weights have been loaded; otherwise
    // we save a sentinel and zero state, so the load() doesn't try to
    // overwrite the core's uninitialized state.
    const bool h_present = m->weights_loaded.load(std::memory_order_acquire);
    if (!stream_write_u32(stream, h_present ? 1u : 0u)) return false;
    if (h_present) {
        if (!stream_write_all(stream, &mirror.h_state, sizeof(mirror.h_state))) return false;
    }

    uint32_t path_len = 0;
    if (mirror.weight_path[0] != '\0') {
        path_len = static_cast<uint32_t>(std::strlen(mirror.weight_path));
        if (path_len >= MambaSynth::MAX_WEIGHT_PATH) {
            path_len = MambaSynth::MAX_WEIGHT_PATH - 1;
        }
    }
    if (!stream_write_u32(stream, path_len)) return false;
    if (path_len > 0) {
        if (!stream_write_all(stream, mirror.weight_path, path_len)) return false;
    }

    return true;
}

static bool state_load(const clap_plugin_t* plugin, const clap_istream_t* stream) {
    if (!plugin || !stream) return false;
    auto* m = get_synth(plugin);

    uint32_t magic = 0;
    if (!stream_read_u32(stream, magic)) return false;
    if (magic != kStateMagic) return false;

    uint32_t version = 0;
    if (!stream_read_u32(stream, version)) return false;
    (void)version; // we accept any of our own versions

    uint32_t reserved = 0;
    if (!stream_read_u32(stream, reserved)) return false;
    (void)reserved;

    uint32_t schema = 0;
    if (!stream_read_u32(stream, schema)) return false;
    if (schema != kStateSchemaV1) return false;

    uint32_t param_count = 0;
    if (!stream_read_u32(stream, param_count)) return false;
    if (param_count == 0 || param_count > PARAM_COUNT) return false;

    double pv[PARAM_COUNT] = {0};
    for (uint32_t i = 0; i < param_count; ++i) {
        if (!stream_read_f64(stream, pv[i])) return false;
    }

    uint32_t bypass_int = 0;
    if (!stream_read_u32(stream, bypass_int)) return false;
    float fade = 0.0f;
    if (!stream_read_f32(stream, fade)) return false;

    uint32_t h_present = 0;
    if (!stream_read_u32(stream, h_present)) return false;
    MambaState ms{};
    if (h_present) {
        if (!stream_read_all(stream, &ms, sizeof(ms))) return false;
    }

    uint32_t path_len = 0;
    if (!stream_read_u32(stream, path_len)) return false;
    if (path_len >= MambaSynth::MAX_WEIGHT_PATH) return false;
    char path_buf[MambaSynth::MAX_WEIGHT_PATH] = {0};
    if (path_len > 0) {
        if (!stream_read_all(stream, path_buf, path_len)) return false;
    }
    path_buf[path_len] = '\0';

    // Apply.
    for (uint32_t i = 0; i < param_count; ++i) {
        m->param_values[i].store(pv[i], std::memory_order_release);
        m->params[i].target.store(pv[i], std::memory_order_release);
        // Snap current to target too; otherwise the smoother would
        // glide to a stale value after load.
        m->params[i].current.store(pv[i], std::memory_order_release);
    }
    m->bypass_target.store(bypass_int != 0, std::memory_order_release);
    m->bypass_fade.store(fade, std::memory_order_relaxed);
    m->bypass_state.store(
        bypass_int != 0 ? BYPASS_BYPASSED : BYPASS_PROCESSING,
        std::memory_order_relaxed);

    if (h_present && m->weights_loaded.load(std::memory_order_acquire)) {
        // Only restore h_state if weights are actually loaded; the
        // validator would otherwise crash on a fresh plugin instance
        // with no weights (SSM state h is uninitialized garbage).
        m->core->set_state(ms);
    }

    std::memset(m->weight_path, 0, sizeof(m->weight_path));
    if (path_len > 0) {
        std::memcpy(m->weight_path, path_buf, path_len);
        m->weight_path[path_len] = '\0';
    }

    return true;
}

const clap_plugin_state_t s_state = {
    state_save,
    state_load
};

}
