#ifndef CLAP_PRESETS_H
#define CLAP_PRESETS_H

#include "clap_plugin.h"

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
static constexpr uint32_t kPresetVersion = 0x00010000u;
static constexpr size_t   kMaxPresetName = 128;

// Resolve preset directory path, returns true if dir exists/created.
// Uses XDG_DATA_HOME/clap/presets/com.mamba-ssm.synth/ on Linux,
// or equivalent on other platforms.
bool get_preset_directory(char* out, size_t out_size);

// Scan preset directory for .mambapreset files. Returns count.
// If presets_out is non-null, fills up to max_presets names.
int scan_presets(const char* dir, char names[][kMaxPresetName], int max_presets);

// Save current plugin state as a preset file.
bool save_preset(MambaSynth* m, const char* path, const char* name);

// Load a preset file into plugin state.
bool load_preset(MambaSynth* m, const char* path);

// Resolve a weight file path with fallback search:
//   1. Try path as-is
//   2. Try relative to preset directory
//   3. Try relative to $MAMBA_WEIGHTS_DIR or default models/
bool resolve_weight_path(const char* path, char* out, size_t out_size);

} // namespace mamba

#endif
