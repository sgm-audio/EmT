#include "clap/clap_plugin.h"
#include <clap/clap.h>
#include <cstring>

namespace mamba {

static uint32_t note_ports_count(const clap_plugin_t* plugin, bool is_input) {
(void)plugin;
return is_input ? 1 : 0;
}

static bool note_ports_get(const clap_plugin_t* plugin, uint32_t index, bool is_input, clap_note_port_info_t* info) {
(void)plugin;
if (!is_input || index != 0) return false;

info->id = 0;
info->supported_dialects = CLAP_NOTE_DIALECT_MIDI | CLAP_NOTE_DIALECT_CLAP;
info->preferred_dialect = CLAP_NOTE_DIALECT_CLAP;
std::strncpy(info->name, "Note Input", CLAP_NAME_SIZE);
return true;
}

const clap_plugin_note_ports_t s_note_ports = {
note_ports_count,
note_ports_get
};

} // namespace mamba
