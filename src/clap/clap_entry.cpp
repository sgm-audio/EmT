#include <clap/clap.h>
#include "clap/clap_plugin.h"
#include <cstring>

namespace mamba {

extern const clap_plugin_descriptor_t mamba_synth_descriptor;
const clap_plugin_t* mamba_synth_create(const clap_host_t* host);

} // namespace mamba

static uint32_t factory_get_plugin_count(const clap_plugin_factory_t* factory) {
    (void)factory;
    return 1;
}

static const clap_plugin_descriptor_t* factory_get_plugin_descriptor(
    const clap_plugin_factory_t* factory, uint32_t index) {
    (void)factory;
    if (index == 0) return &mamba::mamba_synth_descriptor;
    return nullptr;
}

static const clap_plugin_t* factory_create_plugin(
    const clap_plugin_factory_t* factory, const clap_host_t* host, const char* plugin_id) {
    (void)factory;
    if (!strcmp(plugin_id, mamba::mamba_synth_descriptor.id)) {
        return mamba::mamba_synth_create(host);
    }
    return nullptr;
}

static const clap_plugin_factory_t s_factory = {
    factory_get_plugin_count,
    factory_get_plugin_descriptor,
    factory_create_plugin
};

static bool entry_init(const char* plugin_path) {
    (void)plugin_path;
    return true;
}

static void entry_deinit(void) {}

static const void* entry_get_factory(const char* factory_id) {
    if (strcmp(factory_id, CLAP_PLUGIN_FACTORY_ID) == 0) {
        return &s_factory;
    }
    return nullptr;
}

extern "C" {

CLAP_EXPORT const clap_plugin_entry_t clap_entry = {
    CLAP_VERSION_INIT,
    entry_init,
    entry_deinit,
    entry_get_factory
};

}
