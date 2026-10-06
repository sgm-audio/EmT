#ifndef CLAP_PRESETS_H
#define CLAP_PRESETS_H

#include "clap_plugin.h"
#include <array>
#include <string>

namespace mamba {

// .mambapreset file format (binary, little-endian):
//   uint32 magic   = 0x4D425052  'MBPR'
//   uint32 version = 0x00010000  (v1)
//   uint32 reserved
//   uint32 param_count
//   double param_values[param_count]
//   uint32  bypass_target (0/1)
//   float   bypass_fade
//   uint32  h_state_present (0/1)
//   MambaState h_state[MAX_VOICES]  (if present)
//   float   voice_envs[MAX_VOICES]  (if present)
//   uint32  weight_path_len
//   char    weight_path[weight_path_len]
//   uint32  preset_name_len
//   char    preset_name[preset_name_len]

static constexpr uint32_t kPresetMagic   = 0x4D425052u;
static constexpr uint32_t kPresetVersion = 0x00020000u;
static constexpr size_t   kMaxPresetName = 128;

// Factory preset definition. param_values is indexed by ParamIds and must
// always carry exactly PARAM_COUNT entries: build it with
// make_synth_preset_values(), whose arity the compiler checks. (A bare braced
// list would silently zero-fill a short list; std::array also keeps the
// .size() tripwire available to validation code.)
struct FactoryPreset {
    const char* name;
    const char* category;
    std::array<double, PARAM_COUNT> param_values;
    bool bypass_target;
    float bypass_fade;
    const char* weight_path;
};

// Named-argument constructor for FactoryPreset::param_values, in ParamIds
// order. Adding a parameter breaks the static_assert until the arity here
// (and every call site) is updated deliberately.
constexpr std::array<double, PARAM_COUNT> make_synth_preset_values(
    double gain, double attack, double release, double mix, double bypass) {
    static_assert(PARAM_COUNT == 5, "update make_synth_preset_values arity");
    return {gain, attack, release, mix, bypass};
}

// Factory preset bank
static constexpr size_t kFactoryPresetCount = 6;
extern const std::array<FactoryPreset, kFactoryPresetCount> kFactoryPresets;

// Query factory presets
size_t get_factory_preset_count();
const FactoryPreset* get_factory_preset(size_t index);
const FactoryPreset* find_factory_preset(const char* name);

// Apply a factory preset to plugin state
bool apply_factory_preset(MambaSynth* m, size_t index);
bool apply_factory_preset(MambaSynth* m, const char* name);
bool apply_factory_preset(MambaSynth* m, const FactoryPreset* preset);

// Resolve preset directory path, returns true if dir exists/created.
// Uses XDG_DATA_HOME/clap/presets/com.mamba-ssm.synth/ on Linux,
// or equivalent on other platforms.
bool get_preset_directory(char* out, size_t out_size);

// Scan preset directory for .mambapreset files. Returns count.
// If presets_out is non-null, fills up to max_presets names.
int scan_presets(const char* dir, char names[][kMaxPresetName], int max_presets);

// Save current plugin state as a binary preset file.
bool save_preset(MambaSynth* m, const char* path, const char* name);

// Load a binary preset file into plugin state.
bool load_preset(MambaSynth* m, const char* path);

// Export current plugin state as a JSON preset file.
bool export_preset_json(MambaSynth* m, const char* path, const char* name, const char* category = nullptr);

// Import a JSON preset file into plugin state.
bool import_preset_json(MambaSynth* m, const char* path);

// In-memory string versions of JSON preset export/import.
bool export_preset_json_string(MambaSynth* m, const char* name, const char* category, std::string& out_json);
bool import_preset_json_string(MambaSynth* m, const std::string& json_str);

// Export a factory preset to JSON string.
bool export_factory_preset_json(const FactoryPreset& preset, std::string& out_json);

// Resolve a weight file path with fallback search:
//   1. Try path as-is
//   2. Try relative to preset directory
//   3. Try relative to $MAMBA_WEIGHTS_DIR or default models/
bool resolve_weight_path(const char* path, char* out, size_t out_size);

} // namespace mamba

#endif
