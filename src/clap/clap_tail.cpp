#include "clap/clap_plugin.h"
#include <clap/clap.h>

namespace mamba {

// Tail is the time the plugin keeps producing audio after the host has
// stopped sending new events (e.g. a held note still ringing out).
// clap.tail::get() returns samples. We return a small constant
// corresponding to a conservative portion of the envelope/SSM decay.
//
// INT32_MAX is reserved for "infinite tail" per the CLAP spec.
static uint32_t tail_get(const clap_plugin_t* plugin) {
    auto* m = static_cast<MambaSynth*>(plugin->plugin_data);
    if (!m) return 0;
    // 100ms at the current sample rate, with a small lower bound so
    // hosts don't think we are tail-less.
    const double sr = m->sample_rate > 1.0 ? m->sample_rate : 48000.0;
    double tail = 0.100 * sr;
    if (tail < 256.0) tail = 256.0;
    if (tail > 0xFFFFFFFEu) tail = 0xFFFFFFFEu;
    return static_cast<uint32_t>(tail);
}

const clap_plugin_tail_t s_tail = {
    tail_get
};

} // namespace mamba
