#ifndef MAMBA_FX_PLUGIN_H
#define MAMBA_FX_PLUGIN_H

#include <clap/clap.h>

namespace emt {

// Factory function for the Mamba FX plugin.
const clap_plugin_t* mamba_fx_create(const clap_host_t* host);

// Descriptor for the FX plugin.
extern const clap_plugin_descriptor_t mamba_fx_descriptor;

} // namespace emt

#endif
