#include "clap/clap_plugin.h"
#include <clap/clap.h>
#include <cstring>
#include <cstdlib>
#include <cstdio>

namespace mamba {

static MambaSynth* get_synth(const clap_plugin_t* plugin) {
return static_cast<MambaSynth*>(plugin->plugin_data);
}

static uint32_t params_count(const clap_plugin_t* plugin) {
(void)plugin;
return PARAM_COUNT;
}

static bool params_get_info(const clap_plugin_t* plugin, uint32_t param_index, clap_param_info_t* info) {
(void)plugin;
std::memset(info, 0, sizeof(*info));

switch (static_cast<ParamIds>(param_index)) {
case PARAM_GAIN:
info->id = PARAM_GAIN;
info->flags = CLAP_PARAM_IS_AUTOMATABLE;
std::strncpy(info->name, "Gain", CLAP_NAME_SIZE);
std::strncpy(info->module, "SSM", CLAP_PATH_SIZE);
info->min_value = 0.0;
info->max_value = 2.0;
info->default_value = 1.0;
return true;
case PARAM_ATTACK:
info->id = PARAM_ATTACK;
info->flags = CLAP_PARAM_IS_AUTOMATABLE;
std::strncpy(info->name, "Attack", CLAP_NAME_SIZE);
std::strncpy(info->module, "SSM", CLAP_PATH_SIZE);
info->min_value = 0.001;
info->max_value = 1.0;
info->default_value = 0.1;
return true;
case PARAM_RELEASE:
info->id = PARAM_RELEASE;
info->flags = CLAP_PARAM_IS_AUTOMATABLE;
std::strncpy(info->name, "Release", CLAP_NAME_SIZE);
std::strncpy(info->module, "SSM", CLAP_PATH_SIZE);
info->min_value = 0.001;
info->max_value = 2.0;
info->default_value = 0.3;
return true;
case PARAM_MIX:
info->id = PARAM_MIX;
info->flags = CLAP_PARAM_IS_AUTOMATABLE;
std::strncpy(info->name, "Mix", CLAP_NAME_SIZE);
std::strncpy(info->module, "SSM", CLAP_PATH_SIZE);
info->min_value = 0.0;
info->max_value = 1.0;
info->default_value = 1.0;
return true;
case PARAM_BYPASS:
info->id = PARAM_BYPASS;
info->flags = CLAP_PARAM_IS_STEPPED | CLAP_PARAM_IS_BYPASS | CLAP_PARAM_IS_AUTOMATABLE;
std::strncpy(info->name, "Bypass", CLAP_NAME_SIZE);
std::strncpy(info->module, "", CLAP_PATH_SIZE);
info->min_value = 0.0;
info->max_value = 1.0;
info->default_value = 0.0;
return true;
default:
return false;
}
}

static bool params_get_value(const clap_plugin_t* plugin, clap_id param_id, double* out_value) {
auto* m = get_synth(plugin);
if (param_id >= PARAM_COUNT) return false;
*out_value = m->param_values[param_id].load(std::memory_order_acquire);
return true;
}

static bool params_value_to_text(const clap_plugin_t* plugin, clap_id param_id, double value, char* out_buffer, uint32_t out_buffer_capacity) {
(void)plugin;
if (param_id >= PARAM_COUNT) return false;

if (param_id == PARAM_BYPASS) {
const char* s = value > 0.5 ? "On" : "Off";
std::strncpy(out_buffer, s, out_buffer_capacity - 1);
out_buffer[out_buffer_capacity - 1] = '\0';
} else if (param_id == PARAM_MIX) {
std::snprintf(out_buffer, out_buffer_capacity, "%.0f%%", value * 100.0);
} else {
std::snprintf(out_buffer, out_buffer_capacity, "%.3f", value);
}
return true;
}

static bool params_text_to_value(const clap_plugin_t* plugin, clap_id param_id, const char* param_value_text, double* out_value) {
(void)plugin;
if (param_id >= PARAM_COUNT) return false;

if (param_id == PARAM_BYPASS) {
*out_value = (param_value_text[0] == 'O' && param_value_text[1] == 'n') ? 1.0 : 0.0;
} else {
char* end = nullptr;
*out_value = std::strtod(param_value_text, &end);
if (end == param_value_text) return false;
}
return true;
}

static void params_flush(const clap_plugin_t* plugin, const clap_input_events_t* in, const clap_output_events_t* out) {
    (void)out;
    auto* m = get_synth(plugin);

    uint32_t count = in->size(in);
    for (uint32_t i = 0; i < count; ++i) {
        const clap_event_header_t* header = in->get(in, i);
        if (header->type == CLAP_EVENT_PARAM_VALUE) {
            auto* ev = reinterpret_cast<const clap_event_param_value_t*>(header);
            if (ev->param_id < PARAM_COUNT) {
                ParamAtom atom;
                atom.id = ev->param_id;
                atom.value = ev->value;
                atom.frame_offset = header->time;
                m->param_queue.push(atom);
            }
        } else if (header->type == CLAP_EVENT_NOTE_ON) {
            ParamAtom atom;
            atom.id = PARAM_GAIN;
            atom.value = m->param_values[PARAM_GAIN].load(std::memory_order_acquire);
            atom.frame_offset = header->time;
            m->param_queue.push(atom);
        }
    }
}

const clap_plugin_params_t s_params = {
params_count,
params_get_info,
params_get_value,
params_value_to_text,
params_text_to_value,
params_flush
};

} // namespace mamba
