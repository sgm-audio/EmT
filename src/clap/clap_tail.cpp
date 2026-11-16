#include "clap/clap_plugin.h"
#include <clap/clap.h>
#include <algorithm>
#include <cmath>

namespace mamba {

// Tail is the time the plugin keeps producing audio after the host has
// stopped sending new events (e.g. a held note still ringing out).
// clap.tail::get() returns samples. We return a small constant
// corresponding to the conv1d ringdown plus a few hundred ms of
// implicit envelope decay.
//
// INT32_MAX is reserved for "infinite tail" per the CLAP spec.
static uint32_t tail_get(const clap_plugin_t* plugin) {
    auto* m = static_cast<MambaSynth*>(plugin->plugin_data);
    if (!m) return 0;
    const double sr = m->sample_rate > 1.0 ? m->sample_rate : 48000.0;
    const double release_s = m->param_values[PARAM_RELEASE].load(std::memory_order_acquire);
    const double tail_s = std::max(0.100, (std::isfinite(release_s) && release_s > 0.0) ? release_s : 0.100);
    double tail = tail_s * sr;
    if (tail < 256.0) tail = 256.0;
    if (tail > 4294967294.0) tail = 4294967294.0;
    return static_cast<uint32_t>(tail);
}

const clap_plugin_tail_t s_tail = {
    tail_get
};

} // namespace mamba
