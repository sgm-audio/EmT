#include "clap/clap_plugin.h"
#include <clap/clap.h>

namespace emt {

// Mamba-3: no conv1d → zero algorithmic latency.
// Returns 0 always.
static uint32_t latency_get(const clap_plugin_t* plugin) {
    (void)plugin;
    return 0;
}

const clap_plugin_latency_t s_latency = {
    latency_get
};

}
