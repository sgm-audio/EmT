#include "clap/clap_plugin.h"
#include <clap/clap.h>
#include <cstring>

namespace mamba {

static uint32_t audio_ports_count(const clap_plugin_t* plugin, bool is_input) {
    (void)plugin;
    (void)is_input;
    return 1;
}

static bool audio_ports_get(const clap_plugin_t* plugin, uint32_t index, bool is_input, clap_audio_port_info_t* info) {
(void)plugin;
if (index != 0) return false;

info->id = is_input ? 0 : 1;
info->channel_count = 2;
info->flags = CLAP_AUDIO_PORT_IS_MAIN;
info->in_place_pair = is_input ? 1 : 0;
info->port_type = CLAP_PORT_STEREO;
std::strncpy(info->name, is_input ? "Stereo In" : "Stereo Out", CLAP_NAME_SIZE);
return true;
}

const clap_plugin_audio_ports_t s_audio_ports = {
audio_ports_count,
audio_ports_get
};

} // namespace mamba
