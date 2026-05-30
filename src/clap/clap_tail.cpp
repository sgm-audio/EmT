#include "clap/clap_plugin.h"
#include <clap/clap.h>

namespace mamba {

static uint32_t tail_get(const clap_plugin_t* plugin) {
    auto* m = static_cast<MambaSynth*>(plugin->plugin_data);
    return m->latency;
}

const clap_plugin_tail_t s_tail = {
tail_get
};

} // namespace mamba
