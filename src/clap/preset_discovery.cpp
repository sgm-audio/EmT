// CLAP preset discovery: exposes the bundled factory presets to hosts through
// the clap.preset-discovery-factory/2 API, and loads them through the
// clap.preset-load/2 extension (E13).
#include <clap/clap.h>
#include <clap/factory/preset-discovery.h>
#include "clap/clap_presets.h"
#include <cstring>
#include <new>

namespace mamba {

namespace {

const clap_preset_discovery_provider_descriptor_t s_provider_descriptor = {
    CLAP_VERSION_INIT,
    "com.emt.synth.factory-presets",
    "EmT Factory Presets",
    "EmT Audio"
};

// The provider object. provider_data points at this, so the static callbacks
// can recover the indexer we stashed at create() time.
struct EmtPresetProvider {
    clap_preset_discovery_provider_t api;
    const clap_preset_discovery_indexer_t* indexer;
};

bool provider_init(const clap_preset_discovery_provider_t* provider) {
    auto* p = static_cast<EmtPresetProvider*>(provider->provider_data);
    if (p && p->indexer && p->indexer->declare_location) {
        // Factory presets are bundled in the plugin DSO (a PLUGIN location),
        // so the crawl does not touch the filesystem.
        const clap_preset_discovery_location_t loc = {
            CLAP_PRESET_DISCOVERY_IS_FACTORY_CONTENT,
            "Factory Presets",
            CLAP_PRESET_DISCOVERY_LOCATION_PLUGIN,
            nullptr
        };
        p->indexer->declare_location(p->indexer, &loc);
    }
    return true;
}

void provider_destroy(const clap_preset_discovery_provider_t* provider) {
    delete static_cast<EmtPresetProvider*>(provider->provider_data);
}

bool provider_get_metadata(const clap_preset_discovery_provider_t* provider,
                           uint32_t location_kind,
                           const char* location,
                           const clap_preset_discovery_metadata_receiver_t* receiver) {
    (void)provider;
    (void)location_kind;
    (void)location;
    (void)receiver;
    return true;  // EXPERIMENT: no presets yet
}

const void* provider_get_extension(const clap_preset_discovery_provider_t* provider,
                                   const char* extension_id) {
    (void)provider;
    (void)extension_id;
    return nullptr;
}

} // namespace

uint32_t preset_discovery_count(const clap_preset_discovery_factory_t* factory) {
    (void)factory;
    return 1;
}

const clap_preset_discovery_provider_descriptor_t* preset_discovery_get_descriptor(
    const clap_preset_discovery_factory_t* factory, uint32_t index) {
    (void)factory;
    if (index == 0) return &s_provider_descriptor;
    return nullptr;
}

const clap_preset_discovery_provider_t* preset_discovery_create(
    const clap_preset_discovery_factory_t* factory,
    const clap_preset_discovery_indexer_t* indexer,
    const char* provider_id) {
    (void)factory;
    if (!provider_id || std::strcmp(provider_id, s_provider_descriptor.id) != 0) return nullptr;

    auto* p = new (std::nothrow) EmtPresetProvider();
    if (!p) return nullptr;
    p->indexer = indexer;
    p->api.desc = &s_provider_descriptor;
    p->api.provider_data = p;
    p->api.init = provider_init;
    p->api.destroy = provider_destroy;
    p->api.get_metadata = provider_get_metadata;
    p->api.get_extension = provider_get_extension;
    return &p->api;
}

// extern so the symbol has external linkage (a namespace-scope `const` would
// otherwise be internal and dlopen would fail to resolve it from the entry).
extern const clap_preset_discovery_factory_t s_preset_discovery_factory = {
    preset_discovery_count,
    preset_discovery_get_descriptor,
    preset_discovery_create
};

} // namespace mamba
