#include "clap/clap_plugin.h"
#include "core/weight_loader.h"
#include <clap/clap.h>
#include <cstring>

namespace mamba {

static MambaSynth* get_synth(const clap_plugin_t* plugin) {
    return static_cast<MambaSynth*>(plugin->plugin_data);
}

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

static bool state_save(const clap_plugin_t* plugin, const clap_ostream_t* stream) {
    auto* m = get_synth(plugin);

    uint32_t magic = 0x4D424D53;
    if (!stream_write_all(stream, &magic, sizeof(magic))) return false;

    uint32_t version = (VERSION_MAJOR << 16) | (VERSION_MINOR << 8) | VERSION_PATCH;
    if (!stream_write_all(stream, &version, sizeof(version))) return false;

    for (uint32_t i = 0; i < PARAM_COUNT; ++i) {
        double val = m->param_values[i].load(std::memory_order_acquire);
        if (!stream_write_all(stream, &val, sizeof(val))) return false;
    }

    MambaState ms;
    m->core->get_state(ms);
    if (!stream_write_all(stream, &ms, sizeof(ms))) return false;

    uint32_t path_len = 0;
    while (path_len < MambaSynth::MAX_WEIGHT_PATH && m->weight_path[path_len] != '\0') {
        ++path_len;
    }
    if (!stream_write_all(stream, &path_len, sizeof(path_len))) return false;
    if (path_len > 0) {
        if (!stream_write_all(stream, m->weight_path, path_len)) return false;
    }

    return true;
}

static bool state_load(const clap_plugin_t* plugin, const clap_istream_t* stream) {
    auto* m = get_synth(plugin);

    uint32_t magic = 0;
    if (!stream_read_all(stream, &magic, sizeof(magic))) return false;
    if (magic != 0x4D424D53) return false;

    uint32_t version = 0;
    if (!stream_read_all(stream, &version, sizeof(version))) return false;

    for (uint32_t i = 0; i < PARAM_COUNT; ++i) {
        double val = 0.0;
        if (!stream_read_all(stream, &val, sizeof(val))) return false;
        m->param_values[i].store(val, std::memory_order_release);
    }

    MambaState ms;
    if (!stream_read_all(stream, &ms, sizeof(ms))) return false;
    m->core->set_state(ms);

    bool bypass_val = m->param_values[PARAM_BYPASS].load(std::memory_order_acquire) > 0.5;
    m->bypass_target.store(bypass_val, std::memory_order_release);

    uint32_t path_len = 0;
    if (!stream_read_all(stream, &path_len, sizeof(path_len))) return false;
    if (path_len > 0 && path_len < MambaSynth::MAX_WEIGHT_PATH) {
        if (!stream_read_all(stream, m->weight_path, path_len)) return false;
        m->weight_path[path_len] = '\0';
    } else {
        m->weight_path[0] = '\0';
    }

    return true;
}

const clap_plugin_state_t s_state = {
    state_save,
    state_load
};

}