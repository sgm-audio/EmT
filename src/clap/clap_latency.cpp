#include "clap/clap_plugin.h"
#include <clap/clap.h>

namespace mamba {

// Returns the algorithmic latency in samples introduced by the conv1d
// pre-processing. The conv1d kernel is MAX_D_CONV samples wide, of
// which the first MAX_D_CONV-1 are part of the delay line that the
// causal convolution needs to "fill" before the output is correct.
// We return that as our reported latency so the host can compensate.
static uint32_t latency_get(const clap_plugin_t* plugin) {
    auto* m = static_cast<MambaSynth*>(plugin->plugin_data);
    if (!m) return 0;
    return m->latency;
}

const clap_plugin_latency_t s_latency = {
    latency_get
};

}
