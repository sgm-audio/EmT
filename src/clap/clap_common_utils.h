#ifndef MAMBA_CLAP_COMMON_UTILS_H
#define MAMBA_CLAP_COMMON_UTILS_H

#include <algorithm>
#include <cstdint>
#include "clap/clap_plugin.h"
#include "dsp/param_smoother.h"
#include "dsp/bypass_fader.h"

namespace mamba {

// One-pole smoother, bypass state machine, and dry/wet+bypass mix primitives
// live in dsp/param_smoother.h and dsp/bypass_fader.h (included above) --
// single definitions shared by the synth, the FX plugin, and any future
// offline renderer. They remain visible through this header so existing
// call sites are unchanged.

// Extract stereo/mono audio input
inline float extract_audio_input(const clap_audio_buffer_t* in_bus, uint32_t sample_idx, float dry[2]) {
    dry[0] = 0.0f;
    dry[1] = 0.0f;
    if (!in_bus || in_bus->channel_count == 0) return 0.0f;

    const uint32_t in_channels = std::min<uint32_t>(in_bus->channel_count, 2);
    for (uint32_t ch = 0; ch < in_channels; ++ch) {
        if (in_bus->data32 && in_bus->data32[ch]) {
            dry[ch] = in_bus->data32[ch][sample_idx];
        } else if (in_bus->data64 && in_bus->data64[ch]) {
            dry[ch] = static_cast<float>(in_bus->data64[ch][sample_idx]);
        }
    }
    return (in_channels > 1) ? 0.5f * (dry[0] + dry[1]) : dry[0];
}

// Write stereo/mono audio output
inline void write_audio_output(clap_audio_buffer_t& out_bus, uint32_t sample_idx,
                               float sample_ch0, float sample_ch1) {
    for (uint32_t ch = 0; ch < out_bus.channel_count; ++ch) {
        float s = (ch == 0) ? sample_ch0 : ((ch == 1) ? sample_ch1 : sample_ch0);
        if (out_bus.data32 && out_bus.data32[ch]) {
            out_bus.data32[ch][sample_idx] = s;
        }
        if (out_bus.data64 && out_bus.data64[ch]) {
            out_bus.data64[ch][sample_idx] = static_cast<double>(s);
        }
    }
}

// ---------------------------------------------------------------------------
// Robust CLAP Stream I/O helpers
// ---------------------------------------------------------------------------

inline bool stream_write_all(const clap_ostream_t* stream, const void* data, uint64_t size) {
    if (!stream || !data) return false;
    const uint8_t* ptr = static_cast<const uint8_t*>(data);
    uint64_t remaining = size;
    while (remaining > 0) {
        int64_t written = stream->write(stream, ptr, remaining);
        if (written <= 0) return false;
        ptr += written;
        remaining -= static_cast<uint64_t>(written);
    }
    return true;
}

inline bool stream_read_all(const clap_istream_t* stream, void* data, uint64_t size) {
    if (!stream || !data) return false;
    uint8_t* ptr = static_cast<uint8_t*>(data);
    uint64_t remaining = size;
    while (remaining > 0) {
        int64_t rd = stream->read(stream, ptr, remaining);
        if (rd <= 0) return false;
        ptr += rd;
        remaining -= static_cast<uint64_t>(rd);
    }
    return true;
}

inline bool stream_write_u32(const clap_ostream_t* s, uint32_t v) {
    return stream_write_all(s, &v, sizeof(v));
}

inline bool stream_read_u32(const clap_istream_t* s, uint32_t& v) {
    return stream_read_all(s, &v, sizeof(v));
}

inline bool stream_write_f32(const clap_ostream_t* s, float v) {
    return stream_write_all(s, &v, sizeof(v));
}

inline bool stream_read_f32(const clap_istream_t* s, float& v) {
    return stream_read_all(s, &v, sizeof(v));
}

inline bool stream_write_f64(const clap_ostream_t* s, double v) {
    return stream_write_all(s, &v, sizeof(v));
}

inline bool stream_read_f64(const clap_istream_t* s, double& v) {
    return stream_read_all(s, &v, sizeof(v));
}

} // namespace mamba

#endif // MAMBA_CLAP_COMMON_UTILS_H
