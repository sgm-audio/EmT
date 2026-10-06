#ifndef EMT_CLAP_PARAM_TABLE_H
#define EMT_CLAP_PARAM_TABLE_H

// Data-driven parameter metadata for both plugins.
//
// Before this table, every parameter fact (range, default, curve, smoothing
// time, display name, module, CLAP flags, text format) was restated
// imperatively in up to five places: init_param_metadata() in
// clap_plugin.cpp, clap_fx_plugin.cpp and tests/test_conditioning.cpp, the
// get_info switches in clap_params.cpp / clap_fx_plugin.cpp, and the GUI
// slider labels. Adding a parameter meant editing all of them in lockstep.
//
// Now: add one ParamDesc row (and one enum id). The static_asserts below pin
// the table shape; tests/test_param_table.cpp pins the contents against
// PARAMETER_REFERENCE.md.
//
// Display-value formatting travels with the row as a ParamFormat tag.
// Per-plugin quirks are preserved as data, not branches: FX gain/mix use
// Plain ("%.3f", no clamping on parse) while the synth uses GainDb/Percent,
// and FX bypass parsing keeps its as-shipped never-fails behavior
// (BypassFx) instead of the synth's strict Bypass. See observations O7/O8
// in REFACTORING_PLAN.md.

#include "clap/clap_plugin.h"

#include <clap/clap.h>
#include <cstddef>
#include <cstdint>

namespace mamba {

// Display/parse behavior for params_value_to_text / params_text_to_value.
// Implemented by param_value_to_text() / param_text_to_value().
enum class ParamFormat : uint8_t {
    Plain,       // "%.3f" both ways via strtod; no clamping (FX gain/mix)
    GainDb,      // "%.1f dB" / decimal-or-dB, clamped to [min, max]
    Percent,     // "%.0f%%" / percent-or-fraction, clamped to [min, max]
    Milliseconds,// "%.0f ms" / ms-or-seconds, clamped to [min, max]
    Bypass,      // "On"/"Off" with '>' 0.5; case-insensitive + numeric (synth)
    BypassFx,    // "On"/"Off" with '>=' 0.5; "On"/"1" -> 1 else 0, never fails (FX)
};

struct ParamDesc {
    uint32_t id;          // CLAP param id; must equal the row index
    const char* name;     // info->name (CLAP_NAME_SIZE-bound, see test)
    const char* module;   // info->module (CLAP_PATH_SIZE-bound, see test)
    double min_value;
    double max_value;
    double default_value;
    ParamCurve curve;
    float smooth_time_s;  // one-pole time constant; 0 = instant
    uint32_t clap_flags;  // CLAP_PARAM_IS_* combination
    ParamFormat format;   // value<->text behavior tag
};

// --- EmT Synth (com.emt.synth): ids match ParamIds in clap_plugin.h ---

inline constexpr ParamDesc kSynthParamTable[PARAM_COUNT] = {
    {PARAM_GAIN,    "Gain",    "SSM",      0.0,   2.0, 1.0,
     PARAM_CURVE_LOG,     0.010f, CLAP_PARAM_IS_AUTOMATABLE,
     ParamFormat::GainDb},
    {PARAM_ATTACK,  "Attack",  "Envelope", 0.001, 1.0, 0.1,
     PARAM_CURVE_LOG,     0.020f, CLAP_PARAM_IS_AUTOMATABLE,
     ParamFormat::Milliseconds},
    {PARAM_RELEASE, "Release", "Envelope", 0.001, 2.0, 0.3,
     PARAM_CURVE_LOG,     0.050f, CLAP_PARAM_IS_AUTOMATABLE,
     ParamFormat::Milliseconds},
    {PARAM_MIX,     "Mix",     "SSM",      0.0,   1.0, 1.0,
     PARAM_CURVE_LINEAR,  0.010f, CLAP_PARAM_IS_AUTOMATABLE,
     ParamFormat::Percent},
    {PARAM_BYPASS,  "Bypass",  "",         0.0,   1.0, 0.0,
     PARAM_CURVE_STEPPED, 0.0f,
     CLAP_PARAM_IS_AUTOMATABLE | CLAP_PARAM_IS_STEPPED | CLAP_PARAM_IS_BYPASS,
     ParamFormat::Bypass},
};

// --- EmT FX (com.emt.fx): ids match FxParamId in clap_fx_plugin.h ---
//
// NOTE: FX modules are empty as-shipped (fx_params_get_info never set them),
// even though PARAMETER_REFERENCE.md §1.2 claims "SSM". Preserved verbatim;
// see O8. The static_assert in clap_fx_plugin.cpp pins the row count to
// FX_PARAM_COUNT.
inline constexpr ParamDesc kFxParamTable[] = {
    {0, "Gain",   "", 0.0, 2.0, 1.0,
     PARAM_CURVE_LOG,     0.010f, CLAP_PARAM_IS_AUTOMATABLE,
     ParamFormat::Plain},
    {1, "Mix",    "", 0.0, 1.0, 1.0,
     PARAM_CURVE_LINEAR,  0.010f, CLAP_PARAM_IS_AUTOMATABLE,
     ParamFormat::Plain},
    {2, "Bypass", "", 0.0, 1.0, 0.0,
     PARAM_CURVE_STEPPED, 0.0f,
     CLAP_PARAM_IS_AUTOMATABLE | CLAP_PARAM_IS_STEPPED | CLAP_PARAM_IS_BYPASS,
     ParamFormat::BypassFx},
};

inline constexpr size_t kFxParamTableCount =
    sizeof(kFxParamTable) / sizeof(kFxParamTable[0]);

// Every row's id must equal its index, defaults must sit inside the range,
// and name/module must be non-null (lengths are pinned by the unit test,
// which can use strlen; constexpr-strlen here would only add noise).
constexpr bool param_table_valid(const ParamDesc* table, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        if (table[i].id != i) return false;
        if (!(table[i].min_value <= table[i].default_value &&
              table[i].default_value <= table[i].max_value))
            return false;
        if (table[i].name == nullptr || table[i].module == nullptr) return false;
    }
    return true;
}

static_assert(param_table_valid(kSynthParamTable, PARAM_COUNT),
              "synth param table: ids must match row indices, defaults in range");
static_assert(param_table_valid(kFxParamTable, kFxParamTableCount),
              "FX param table: ids must match row indices, defaults in range");

// Initialize ParamSmooth metadata (range, default, curve, smoothing time)
// from a table. Array-size deduction fails to compile if the smoothing array
// and the table ever disagree on the parameter count.
template <size_t N>
void init_param_smooth(ParamSmooth (&params)[N], const ParamDesc (&table)[N]) {
    for (size_t i = 0; i < N; ++i) {
        params[i].min_value = table[i].min_value;
        params[i].max_value = table[i].max_value;
        params[i].default_value = table[i].default_value;
        params[i].curve = table[i].curve;
        params[i].smooth_time_s = table[i].smooth_time_s;
    }
}

// Format/parse a display value per a row's ParamFormat tag. Both assume a
// finite value; the callers keep their as-shipped pre-checks (the synth maps
// NaN to 0 and clamps to range, the FX rejects non-finite input).
// Returns false only when the buffer is null/empty (format) or the text is
// unparseable (parse; never for BypassFx, which preserves the FX quirk).
bool param_value_to_text(ParamFormat format, double value, char* out,
                         uint32_t out_capacity);
bool param_text_to_value(ParamFormat format, const char* text, double min_value,
                         double max_value, double* out_value);

}  // namespace mamba

#endif  // EMT_CLAP_PARAM_TABLE_H
