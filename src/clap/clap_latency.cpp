#include "clap/clap_plugin.h"
#include <clap/clap.h>

namespace mamba {

static uint32_t latency_get(const clap_plugin_t* plugin) {
    auto* m = static_cast<MambaSynth*>(plugin->plugin_data);
    return m->latency;
}

const clap_plugin_latency_t s_latency = {
    latency_get
};

}