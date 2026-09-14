#include "clap/clap_plugin.h"
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

// Compute the smoothed dB value for log-curve params. Used by value_to_text
// for attack/release/gain. Plain value is the actual control value
// passed to the DSP.
static double log_curve_to_knob(double v, double min_v, double max_v) {
    if (max_v <= min_v) return v;
    if (v <= min_v) return 0.0;
    if (v >= max_v) return 1.0;
    const double lo = std::log(min_v);
    const double hi = std::log(max_v);
    return (std::log(v) - lo) / (hi - lo);
}

static double log_knob_to_value(double k, double min_v, double max_v) {
    if (k <= 0.0) return min_v;
    if (k >= 1.0) return max_v;
    const double lo = std::log(min_v);
    const double hi = std::log(max_v);
    return std::exp(lo + k * (hi - lo));
}

static bool params_get_info(const clap_plugin_t* plugin, uint32_t param_index, clap_param_info_t* info) {
    auto* m = get_synth(plugin);
    if (param_index >= PARAM_COUNT) return false;
    std::memset(info, 0, sizeof(*info));

    const ParamSmooth& p = m->params[param_index];
    info->id = static_cast<clap_id>(param_index);
    info->cookie = const_cast<ParamSmooth*>(&p);
    info->min_value = p.min_value;
    info->max_value = p.max_value;
    info->default_value = p.default_value;

    // Bypass is always stepped.
    // Log-curve params are automatable + (technically) not periodic.
    // Stepped params have IS_STEPPED.
    switch (static_cast<ParamIds>(param_index)) {
    case PARAM_GAIN:
    case PARAM_ATTACK:
    case PARAM_RELEASE:
    case PARAM_MIX:
        info->flags = CLAP_PARAM_IS_AUTOMATABLE;
        break;
    case PARAM_BYPASS:
        info->flags = CLAP_PARAM_IS_AUTOMATABLE
                    | CLAP_PARAM_IS_STEPPED
                    | CLAP_PARAM_IS_BYPASS;
        break;
    default:
        return false;
    }

    switch (static_cast<ParamIds>(param_index)) {
    case PARAM_GAIN:
        std::strncpy(info->name, "Gain", CLAP_NAME_SIZE);
        std::strncpy(info->module, "SSM", CLAP_PATH_SIZE);
        break;
    case PARAM_ATTACK:
        std::strncpy(info->name, "Attack", CLAP_NAME_SIZE);
        std::strncpy(info->module, "Envelope", CLAP_PATH_SIZE);
        break;
    case PARAM_RELEASE:
        std::strncpy(info->name, "Release", CLAP_NAME_SIZE);
        std::strncpy(info->module, "Envelope", CLAP_PATH_SIZE);
        break;
    case PARAM_MIX:
        std::strncpy(info->name, "Mix", CLAP_NAME_SIZE);
        std::strncpy(info->module, "SSM", CLAP_PATH_SIZE);
        break;
    case PARAM_BYPASS:
        std::strncpy(info->name, "Bypass", CLAP_NAME_SIZE);
        std::strncpy(info->module, "", CLAP_PATH_SIZE);
        break;
    default:
        return false;
    }
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
    if (param_id >= PARAM_COUNT || !out_buffer || out_buffer_capacity == 0) return false;
    if (value != value) value = 0.0; // NaN guard

    const ParamSmooth& p = m->params[param_id];

    // Round to plain value first (i.e. the value the audio thread sees).
    if (value < p.min_value) value = p.min_value;
    if (value > p.max_value) value = p.max_value;

    switch (static_cast<ParamIds>(param_id)) {
    case PARAM_BYPASS: {
        const char* s = value > 0.5 ? "On" : "Off";
        std::strncpy(out_buffer, s, out_buffer_capacity - 1);
        out_buffer[out_buffer_capacity - 1] = '\0';
        return true;
    }
    case PARAM_MIX: {
        // Display as percent of full wet.
        std::snprintf(out_buffer, out_buffer_capacity, "%.0f%%", value * 100.0);
        return true;
    }
    case PARAM_GAIN: {
        // Log curve in [0, 2] mapped to dB.
        const double db = 20.0 * std::log10(value > 0.0 ? value : 1e-9);
        std::snprintf(out_buffer, out_buffer_capacity, "%.1f dB", db);
        return true;
    }
    case PARAM_ATTACK:
    case PARAM_RELEASE: {
        std::snprintf(out_buffer, out_buffer_capacity, "%.0f ms", value * 1000.0);
        return true;
    }
    default:
        return false;
    }
}

static bool params_text_to_value(const clap_plugin_t* plugin, clap_id param_id, const char* param_value_text, double* out_value) {
    auto* m = get_synth(plugin);
    if (param_id >= PARAM_COUNT || !param_value_text || !out_value) return false;

    const ParamSmooth& p = m->params[param_id];

    switch (static_cast<ParamIds>(param_id)) {
    case PARAM_BYPASS: {
        // Accept "On"/"Off" (case-insensitive) and numeric 0/1.
        if (param_value_text[0] == 'O' || param_value_text[0] == 'o') {
            if ((param_value_text[1] == 'N' || param_value_text[1] == 'n')
                && param_value_text[2] == '\0') {
                *out_value = 1.0;
                return true;
            }
            if (param_value_text[1] == 'f' || param_value_text[1] == 'F') {
                // Allow "Off" / "of" prefix; reject anything else.
                if (param_value_text[2] == 'f' || param_value_text[2] == 'F') {
                    if (param_value_text[3] == '\0') {
                        *out_value = 0.0;
                        return true;
                    }
                }
            }
        }
        // Fall through to numeric parsing.
        char* end = nullptr;
        double v = std::strtod(param_value_text, &end);
        if (end == param_value_text) return false;
        *out_value = v > 0.5 ? 1.0 : 0.0;
        return true;
    }
    case PARAM_MIX: {
        // Accept "75%" or "75" or "0.75".
        char* end = nullptr;
        double v = std::strtod(param_value_text, &end);
        if (end == param_value_text) return false;
        // If "%" suffix is present, treat as percent; otherwise if the
        // number is > 1.0, also treat as percent (host convention).
        bool percent = false;
        if (*end == '%') { percent = true; ++end; }
        while (*end == ' ' || *end == '\t') ++end;
        if (*end != '\0' && !(end == param_value_text + std::strlen(param_value_text) - 1
                              || (*end >= '0' && *end <= '9'))) {
            // trailing junk
        }
        if (percent || v > 1.0) v = v / 100.0;
        if (v < 0.0) v = 0.0;
        if (v > 1.0) v = 1.0;
        *out_value = v;
        return true;
    }
    case PARAM_GAIN: {
        // Accept "0.5", "-6.0 dB" (dB suffix), or "0.5x".
        char* end = nullptr;
        double v = std::strtod(param_value_text, &end);
        if (end == param_value_text) return false;
        while (*end == ' ') ++end;
        if (*end == 'd' || *end == 'D') {
            // dB
            v = std::pow(10.0, v / 20.0);
        }
        if (v < 0.0) v = 0.0;
        if (v > 2.0) v = 2.0;
        *out_value = v;
        return true;
    }
    case PARAM_ATTACK:
    case PARAM_RELEASE: {
        // Accept "100", "100 ms", "0.1 s".
        char* end = nullptr;
        double v = std::strtod(param_value_text, &end);
        if (end == param_value_text) return false;
        while (*end == ' ') ++end;
        if (*end == 'm' || *end == 'M') {
            // already in ms
        } else if (*end == 's' || *end == 'S') {
            v = v * 1000.0; // seconds -> ms; we're stored in seconds.
        }
        v = v / 1000.0;
        if (v < p.min_value) v = p.min_value;
        if (v > p.max_value) v = p.max_value;
        *out_value = v;
        return true;
    }
    default:
        return false;
    }
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
