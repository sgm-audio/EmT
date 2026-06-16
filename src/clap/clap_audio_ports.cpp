#include "clap/clap_plugin.h"
#include <clap/clap.h>
#include <cstring>

namespace mamba {

// 4.1: Define mono + stereo input/output port pairs. Each side gets one
// mono and one stereo port. The stereo is the main port (must be at
// index 0 for both inputs and outputs per CLAP_AUDIO_PORT_IS_MAIN).
static const uint32_t kNumInputPorts  = 2;
static const uint32_t kNumOutputPorts = 2;

static uint32_t audio_ports_count(const clap_plugin_t* plugin, bool is_input) {
    (void)plugin;
    return is_input ? kNumInputPorts : kNumOutputPorts;
}

static bool audio_ports_get(const clap_plugin_t* plugin, uint32_t index, bool is_input, clap_audio_port_info_t* info) {
    (void)plugin;
    std::memset(info, 0, sizeof(*info));

    if (is_input) {
        if (index >= kNumInputPorts) return false;
        if (index == 0) {
            // Main stereo input.
            info->id = 0;
            info->channel_count = 2;
            info->flags = CLAP_AUDIO_PORT_IS_MAIN | CLAP_AUDIO_PORT_SUPPORTS_64BITS;
            info->in_place_pair = 1; // paired with stereo output
            info->port_type = CLAP_PORT_STEREO;
            std::strncpy(info->name, "Stereo In", CLAP_NAME_SIZE);
        } else {
            // Mono input (aux).
            info->id = 2;
            info->channel_count = 1;
            info->flags = CLAP_AUDIO_PORT_SUPPORTS_64BITS;
            info->in_place_pair = 3; // paired with mono output
            info->port_type = CLAP_PORT_MONO;
            std::strncpy(info->name, "Mono In", CLAP_NAME_SIZE);
        }
    } else {
        if (index >= kNumOutputPorts) return false;
        if (index == 0) {
            // Main stereo output.
            info->id = 1;
            info->channel_count = 2;
            info->flags = CLAP_AUDIO_PORT_IS_MAIN | CLAP_AUDIO_PORT_SUPPORTS_64BITS;
            info->in_place_pair = 0; // paired with stereo input
            info->port_type = CLAP_PORT_STEREO;
            std::strncpy(info->name, "Stereo Out", CLAP_NAME_SIZE);
        } else {
            // Mono output.
            info->id = 3;
            info->channel_count = 1;
            info->flags = CLAP_AUDIO_PORT_SUPPORTS_64BITS;
            info->in_place_pair = 2; // paired with mono input
            info->port_type = CLAP_PORT_MONO;
            std::strncpy(info->name, "Mono Out", CLAP_NAME_SIZE);
        }
    }
    return true;
}

const clap_plugin_audio_ports_t s_audio_ports = {
    audio_ports_count,
    audio_ports_get
};

} // namespace mamba
