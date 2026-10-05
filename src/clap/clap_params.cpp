#include "clap/clap_plugin.h"
#include "clap/param_table.h"
#include <clap/clap.h>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <cmath>

namespace mamba {

static MambaSynth* get_synth(const clap_plugin_t* plugin) {
    return static_cast<MambaSynth*>(plugin->plugin_data);
}

static uint32_t params_count(const clap_plugin_t* plugin) {
    (void)plugin;
    return PARAM_COUNT;
}

static bool params_get_info(const clap_plugin_t* plugin, uint32_t param_index, clap_param_info_t* info) {
    auto* m = get_synth(plugin);
    if (param_index >= PARAM_COUNT) return false;
    std::memset(info, 0, sizeof(*info));

    // Range, default, flags, and strings all come from the shared table.
    // (cookie still points at the live smoothing state, as before.)
    const ParamDesc& desc = kSynthParamTable[param_index];
    info->id = desc.id;
    info->cookie = &m->params[param_index];
    info->min_value = desc.min_value;
    info->max_value = desc.max_value;
    info->default_value = desc.default_value;
    info->flags = desc.clap_flags;
    std::strncpy(info->name, desc.name, CLAP_NAME_SIZE);
    std::strncpy(info->module, desc.module, CLAP_PATH_SIZE);
    return true;
}

static bool params_get_value(const clap_plugin_t* plugin, clap_id param_id, double* out_value) {
    auto* m = get_synth(plugin);
    if (param_id >= PARAM_COUNT || !out_value) return false;
    *out_value = m->param_values[param_id].load(std::memory_order_acquire);
    return true;
}

static bool params_value_to_text(const clap_plugin_t* plugin, clap_id param_id, double value, char* out_buffer, uint32_t out_buffer_capacity) {
    auto* m = get_synth(plugin);
    (void)m;  // Metadata now comes from the table; the deref stays as a null-plugin guard.
    if (param_id >= PARAM_COUNT || !out_buffer || out_buffer_capacity == 0) return false;
    if (value != value) value = 0.0; // NaN guard

    const ParamDesc& desc = kSynthParamTable[param_id];

    // Round to plain value first (i.e. the value the audio thread sees).
    if (value < desc.min_value) value = desc.min_value;
    if (value > desc.max_value) value = desc.max_value;

    return param_value_to_text(desc.format, value, out_buffer, out_buffer_capacity);
}

static bool params_text_to_value(const clap_plugin_t* plugin, clap_id param_id, const char* param_value_text, double* out_value) {
    auto* m = get_synth(plugin);
    (void)m;  // Metadata now comes from the table; the deref stays as a null-plugin guard.
    if (param_id >= PARAM_COUNT || !param_value_text || !out_value) return false;

    const ParamDesc& desc = kSynthParamTable[param_id];
    return param_text_to_value(desc.format, param_value_text, desc.min_value,
                               desc.max_value, out_value);
}

// Apply a single param atom to the plugin's atomic state. The SPSC
// queue is consumed in process(); when flush() is called outside of
// audio (e.g. when the plugin is not active), we also write directly
// to the atomics so the host's get_value() reads see the new value.
static void apply_atom_to_state(MambaSynth* m, const ParamAtom& atom) {
    if (atom.id >= PARAM_COUNT) return;
    m->param_values[atom.id].store(atom.value, std::memory_order_release);
    m->params[atom.id].target.store(atom.value, std::memory_order_release);
    if (atom.id == PARAM_BYPASS) {
        m->bypass_target.store(atom.value > 0.5, std::memory_order_release);
    }
}

static void params_flush(const clap_plugin_t* plugin, const clap_input_events_t* in, const clap_output_events_t* out) {
    (void)out;
    auto* m = get_synth(plugin);
    if (!in) return;

    const uint32_t count = in->size(in);
    for (uint32_t i = 0; i < count; ++i) {
        const clap_event_header_t* header = in->get(in, i);
        if (!header) continue;

        // 4.10/param-set-wrong-namespace fix: only handle events from
        // the clap.core event space; the host sends wrong-namespace
        // events to verify the plugin ignores them.
        if (header->space_id != CLAP_CORE_EVENT_SPACE_ID) continue;

        if (header->type == CLAP_EVENT_PARAM_VALUE) {
            const auto* ev = reinterpret_cast<const clap_event_param_value_t*>(header);
            if (ev->param_id >= PARAM_COUNT) continue;

            ParamAtom atom{ev->param_id, ev->value, 0};
            apply_atom_to_state(m, atom);
        } else if (header->type == CLAP_EVENT_PARAM_MOD) {
            // Modulation is not supported in Phase 4; we still must
            // not crash on it. Ignore.
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
