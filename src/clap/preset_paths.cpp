// Preset/weight path handling: preset directory creation, directory
// scanning, and weight-file search. Split out of clap_presets.cpp
// (Phase 1 #3); see preset_bank.cpp (factory data) and preset_codec.cpp
// (file/JSON codecs).
#include "clap/clap_presets.h"
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>

namespace mamba {

bool get_preset_directory(char* out, size_t out_size) {
    // Linux: $XDG_DATA_HOME/clap/presets/com.mamba-ssm.synth/
    const char* xdg = std::getenv("XDG_DATA_HOME");
    std::string base;
    if (xdg && xdg[0]) {
        base = xdg;
    } else {
        const char* home = std::getenv("HOME");
#if defined(_WIN32)
        if (!home) home = std::getenv("USERPROFILE");
        if (!home) home = std::getenv("APPDATA");
#endif
        if (!home) return false;
        base = std::string(home) + "/.local/share";
    }
    std::string dir = base + "/clap/presets/com.mamba-ssm.synth";
    std::filesystem::create_directories(dir);
    std::strncpy(out, dir.c_str(), out_size - 1);
    out[out_size - 1] = '\0';
    return true;
}

int scan_presets(const char* dir, char names[][kMaxPresetName], int max_presets) {
    int count = 0;
    if (!dir) return 0;
    std::error_code ec;
    if (!std::filesystem::is_directory(dir, ec)) return 0;
    for (auto& entry : std::filesystem::directory_iterator(dir, ec)) {
        if (!entry.is_regular_file()) continue;
        auto path = entry.path();
        if (path.extension() != ".mambapreset" && path.extension() != ".json") continue;
        if (count >= max_presets) break;
        std::string stem = path.stem().string();
        std::strncpy(names[count], stem.c_str(), kMaxPresetName - 1);
        names[count][kMaxPresetName - 1] = '\0';
        count++;
    }
    return count;
}

bool resolve_weight_path(const char* path, char* out, size_t out_size) {
    if (!path || !path[0]) return false;

    // 1. Try path as-is
    if (std::filesystem::exists(path)) {
        std::strncpy(out, path, out_size - 1);
        out[out_size - 1] = '\0';
        return true;
    }

    // 2. Try relative to preset directory
    char preset_dir[512] = {0};
    if (get_preset_directory(preset_dir, sizeof(preset_dir))) {
        std::string candidate = std::string(preset_dir) + "/" + path;
        if (std::filesystem::exists(candidate)) {
            std::strncpy(out, candidate.c_str(), out_size - 1);
            out[out_size - 1] = '\0';
            return true;
        }
        candidate = std::string(preset_dir) + "/../weights/" + path;
        if (std::filesystem::exists(candidate)) {
            std::strncpy(out, candidate.c_str(), out_size - 1);
            out[out_size - 1] = '\0';
            return true;
        }
    }

    // 3. Try $EMT_WEIGHTS_DIR / $MAMBA_WEIGHTS_DIR
    const char* mwd = std::getenv("EMT_WEIGHTS_DIR");
    if (!mwd || !mwd[0]) mwd = std::getenv("MAMBA_WEIGHTS_DIR");
    if (mwd && mwd[0]) {
        std::string candidate = std::string(mwd) + "/" + path;
        if (std::filesystem::exists(candidate)) {
            std::strncpy(out, candidate.c_str(), out_size - 1);
            out[out_size - 1] = '\0';
            return true;
        }
    }

    // 4. Try models/ relative to cwd
    {
        std::string candidate = std::string("models/") + path;
        if (std::filesystem::exists(candidate)) {
            std::strncpy(out, candidate.c_str(), out_size - 1);
            out[out_size - 1] = '\0';
            return true;
        }
    }

    // Not found — return input path as-is (caller will handle failure)
    std::strncpy(out, path, out_size - 1);
    out[out_size - 1] = '\0';
    return false;
}

} // namespace mamba
