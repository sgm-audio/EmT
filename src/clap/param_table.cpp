// Value<->text conversion behind the ParamFormat tags in param_table.h.
//
// The conversion rules moved here verbatim from params_value_to_text /
// params_text_to_value in clap_params.cpp (synth) and fx_params_value_to_text /
// fx_params_text_to_value in clap_fx_plugin.cpp. Per-plugin differences are
// preserved as distinct tags (see the ParamFormat docs); the only deliberate
// cleanup is dropping an empty trailing-junk if-body in the synth Mix parser
// that provably did nothing.

#include "clap/param_table.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace mamba {
namespace {

// Accept "On"/"Off" (case-insensitive) and numeric 0/1. Synth bypass
// semantics: strict '>' threshold, no isfinite check, fails only when no
// number could be parsed at all.
bool parse_bypass_strict(const char* text, double* out_value) {
    if (text[0] == 'O' || text[0] == 'o') {
        if ((text[1] == 'N' || text[1] == 'n') && text[2] == '\0') {
            *out_value = 1.0;
            return true;
        }
        if (text[1] == 'f' || text[1] == 'F') {
            // Allow "Off" / "of" prefix; reject anything else.
            if (text[2] == 'f' || text[2] == 'F') {
                if (text[3] == '\0') {
                    *out_value = 0.0;
                    return true;
                }
            }
        }
    }
    // Fall through to numeric parsing.
    char* end = nullptr;
    double v = std::strtod(text, &end);
    if (end == text) return false;
    *out_value = v > 0.5 ? 1.0 : 0.0;
    return true;
}

// FX bypass semantics (quirk preserved, see O7): "On"/"1" -> 1, literally
// anything else -> 0; never fails.
bool parse_bypass_fx(const char* text, double* out_value) {
    *out_value = (!std::strcmp(text, "On") || !std::strcmp(text, "1")) ? 1.0 : 0.0;
    return true;
}

// Accept "0.5", "-6.0 dB" (dB suffix), or "0.5x". A single 'd'/'D' triggers
// the dB conversion; anything else is a plain linear value.
bool parse_gain_db(const char* text, double min_value, double max_value,
                   double* out_value) {
    char* end = nullptr;
    double v = std::strtod(text, &end);
    if (end == text) return false;
    while (*end == ' ') ++end;
    if (*end == 'd' || *end == 'D') {
        // dB
        v = std::pow(10.0, v / 20.0);
    }
    if (v < min_value) v = min_value;
    if (v > max_value) v = max_value;
    *out_value = v;
    return true;
}

// Accept "75%" or "75" or "0.75". If "%" suffix is present, treat as percent;
// otherwise if the number is > 1.0, also treat as percent (host convention).
bool parse_percent(const char* text, double min_value, double max_value,
                   double* out_value) {
    char* end = nullptr;
    double v = std::strtod(text, &end);
    if (end == text) return false;
    bool percent = false;
    if (*end == '%') {
        percent = true;
        ++end;
    }
    while (*end == ' ' || *end == '\t') ++end;
    if (percent || v > 1.0) v = v / 100.0;
    if (v < min_value) v = min_value;
    if (v > max_value) v = max_value;
    *out_value = v;
    return true;
}

// Accept "100", "100 ms", "0.1 s". Bare numbers are milliseconds.
bool parse_milliseconds(const char* text, double min_value, double max_value,
                        double* out_value) {
    char* end = nullptr;
    double v = std::strtod(text, &end);
    if (end == text) return false;
    while (*end == ' ') ++end;
    if (*end == 'm' || *end == 'M') {
        // already in ms
    } else if (*end == 's' || *end == 'S') {
        v = v * 1000.0;  // seconds -> ms; we're stored in seconds.
    }
    v = v / 1000.0;
    if (v < min_value) v = min_value;
    if (v > max_value) v = max_value;
    *out_value = v;
    return true;
}

}  // namespace

bool param_value_to_text(ParamFormat format, double value, char* out,
                         uint32_t out_capacity) {
    if (!out || out_capacity == 0) return false;
    // NOTE: snprintf goes straight into the caller's buffer (no intermediate
    // fixed buffer), so unclamped Plain values format exactly as before.
    switch (format) {
        case ParamFormat::Plain:
            std::snprintf(out, out_capacity, "%.3f", value);
            break;
        case ParamFormat::GainDb: {
            // Log curve in [0, 2] mapped to dB.
            double db = 20.0 * std::log10(value > 0.0 ? value : 1e-9);
            std::snprintf(out, out_capacity, "%.1f dB", db);
            break;
        }
        case ParamFormat::Percent:
            // Display as percent of full wet.
            std::snprintf(out, out_capacity, "%.0f%%", value * 100.0);
            break;
        case ParamFormat::Milliseconds:
            std::snprintf(out, out_capacity, "%.0f ms", value * 1000.0);
            break;
        case ParamFormat::Bypass:
            // NOTE: strict '>' here, unlike BypassFx's '>=' below.
            std::snprintf(out, out_capacity, "%s", value > 0.5 ? "On" : "Off");
            break;
        case ParamFormat::BypassFx:
            std::snprintf(out, out_capacity, "%s", value >= 0.5 ? "On" : "Off");
            break;
        default:
            return false;
    }
    return true;
}

bool param_text_to_value(ParamFormat format, const char* text, double min_value,
                         double max_value, double* out_value) {
    if (!text || !out_value) return false;
    switch (format) {
        case ParamFormat::Plain: {
            char* end = nullptr;
            double v = std::strtod(text, &end);
            if (end == text || !std::isfinite(v)) return false;
            *out_value = v;  // FX plain params never clamped, as-shipped
            return true;
        }
        case ParamFormat::GainDb:
            return parse_gain_db(text, min_value, max_value, out_value);
        case ParamFormat::Percent:
            return parse_percent(text, min_value, max_value, out_value);
        case ParamFormat::Milliseconds:
            return parse_milliseconds(text, min_value, max_value, out_value);
        case ParamFormat::Bypass:
            return parse_bypass_strict(text, out_value);
        case ParamFormat::BypassFx:
            return parse_bypass_fx(text, out_value);
        default:
            return false;
    }
}

}  // namespace mamba
