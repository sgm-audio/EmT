#include "clap/clap_presets.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <climits>
#include <filesystem>
#include <string>
#include <vector>

namespace mamba {
namespace {

// ---------------------------------------------------------------------------
// Stream helpers (identical to clap_state.cpp but file-based)
// ---------------------------------------------------------------------------

static bool file_write_all(std::FILE* f, const void* data, size_t size) {
    return std::fwrite(data, 1, size, f) == size;
}

static bool file_read_all(std::FILE* f, void* data, size_t size) {
    return std::fread(data, 1, size, f) == size;
}

static bool file_write_u32(std::FILE* f, uint32_t v) { return file_write_all(f, &v, sizeof(v)); }
static bool file_read_u32(std::FILE* f, uint32_t& v) { return file_read_all(f, &v, sizeof(v)); }
static bool file_write_f32(std::FILE* f, float v) { return file_write_all(f, &v, sizeof(v)); }
static bool file_read_f32(std::FILE* f, float& v) { return file_read_all(f, &v, sizeof(v)); }
static bool file_write_f64(std::FILE* f, double v) { return file_write_all(f, &v, sizeof(v)); }
static bool file_read_f64(std::FILE* f, double& v) { return file_read_all(f, &v, sizeof(v)); }

} // namespace

bool get_preset_directory(char* out, size_t out_size) {
    // Linux: $XDG_DATA_HOME/clap/presets/com.mamba-ssm.synth/
    const char* xdg = std::getenv("XDG_DATA_HOME");
    std::string base;
    if (xdg && xdg[0]) {
        base = xdg;
    } else {
        const char* home = std::getenv("HOME");
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
    // Check directory exists first (avoid throwing)
    std::error_code ec;
    if (!std::filesystem::is_directory(dir, ec)) return 0;
    for (auto& entry : std::filesystem::directory_iterator(dir, ec)) {
        if (!entry.is_regular_file()) continue;
        auto path = entry.path();
        if (path.extension() != ".mambapreset") continue;
        if (count >= max_presets) break;
        std::string stem = path.stem().string();
        std::strncpy(names[count], stem.c_str(), kMaxPresetName - 1);
        names[count][kMaxPresetName - 1] = '\0';
        count++;
    }
    return count;
}

bool save_preset(MambaSynth* m, const char* path, const char* name) {
    if (!m || !path) return false;

    // Pull a fresh state mirror from the audio thread.
    StateMirror mirror{};
    {
        constexpr int kMaxRetries = 8;
        bool ok = false;
        for (int retry = 0; retry < kMaxRetries; ++retry) {
            mirror = StateMirror{};
            if (m->state_mirror.read(mirror)) { ok = true; break; }
        }
        if (!ok) return false;
    }

    std::FILE* f = std::fopen(path, "wb");
    if (!f) return false;

    bool ok = true;
    ok = ok && file_write_u32(f, kPresetMagic);
    ok = ok && file_write_u32(f, kPresetVersion);
    ok = ok && file_write_u32(f, 0); // reserved
    ok = ok && file_write_u32(f, PARAM_COUNT);
    for (uint32_t i = 0; i < PARAM_COUNT && ok; ++i)
        ok = file_write_f64(f, mirror.param_values[i]);
    ok = ok && file_write_u32(f, mirror.bypass_target ? 1u : 0u);
    ok = ok && file_write_f32(f, mirror.bypass_fade);

    const bool h_present = m->weights_loaded.load(std::memory_order_acquire);
    ok = ok && file_write_u32(f, h_present ? 1u : 0u);
    if (h_present && ok) {
        for (uint32_t v = 0; v < MAX_VOICES; ++v) {
            ok = file_write_all(f, &mirror.voice_snapshots[v].ssm_state,
                                sizeof(MambaState));
            ok = ok && file_write_f32(f, mirror.voice_snapshots[v].env);
        }
    }

    uint32_t path_len = 0;
    if (mirror.weight_path[0]) {
        path_len = static_cast<uint32_t>(std::strlen(mirror.weight_path));
        if (path_len >= MambaSynth::MAX_WEIGHT_PATH)
            path_len = MambaSynth::MAX_WEIGHT_PATH - 1;
    }
    ok = ok && file_write_u32(f, path_len);
    if (path_len > 0 && ok)
        ok = file_write_all(f, mirror.weight_path, path_len);

    uint32_t name_len = name ? static_cast<uint32_t>(std::strlen(name)) : 0;
    if (name_len > kMaxPresetName - 1) name_len = kMaxPresetName - 1;
    ok = ok && file_write_u32(f, name_len);
    if (name_len > 0 && ok)
        ok = file_write_all(f, name, name_len);

    std::fclose(f);
    if (!ok) { std::remove(path); return false; }
    return true;
}

bool load_preset(MambaSynth* m, const char* path) {
    if (!m || !path) return false;

    std::FILE* f = std::fopen(path, "rb");
    if (!f) return false;

    auto fail = [f]() { std::fclose(f); return false; };

    uint32_t magic = 0;
    if (!file_read_u32(f, magic) || magic != kPresetMagic) return fail();

    uint32_t version = 0;
    if (!file_read_u32(f, version)) return fail();

    uint32_t reserved = 0;
    if (!file_read_u32(f, reserved)) return fail();

    uint32_t param_count = 0;
    if (!file_read_u32(f, param_count)) return fail();
    if (param_count > PARAM_COUNT) return fail();

    double pv[PARAM_COUNT] = {0};
    for (uint32_t i = 0; i < param_count; ++i)
        if (!file_read_f64(f, pv[i])) return fail();

    uint32_t bypass_int = 0;
    if (!file_read_u32(f, bypass_int)) return fail();
    float fade = 0.0f;
    if (!file_read_f32(f, fade)) return fail();

    uint32_t h_present = 0;
    if (!file_read_u32(f, h_present)) return fail();

    struct { MambaState state; float env; } vstates[MAX_VOICES]{};
    if (h_present) {
        for (uint32_t v = 0; v < MAX_VOICES; ++v) {
            if (!file_read_all(f, &vstates[v].state, sizeof(MambaState))) return fail();
            if (!file_read_f32(f, vstates[v].env)) return fail();
        }
    }

    uint32_t path_len = 0;
    if (!file_read_u32(f, path_len)) return fail();
    if (path_len >= MambaSynth::MAX_WEIGHT_PATH) return fail();
    char path_buf[MambaSynth::MAX_WEIGHT_PATH] = {0};
    if (path_len > 0)
        if (!file_read_all(f, path_buf, path_len)) return fail();
    path_buf[path_len] = '\0';

    // Read optional preset name (ignore it, just consume the bytes)
    uint32_t name_len = 0;
    if (!file_read_u32(f, name_len)) return fail();
    if (name_len > 0) {
        // Skip past the name bytes
        std::fseek(f, name_len, SEEK_CUR);
    }

    std::fclose(f);
    f = nullptr;

    // Apply: parameters
    for (uint32_t i = 0; i < param_count; ++i) {
        m->param_values[i].store(pv[i], std::memory_order_release);
        m->params[i].target.store(pv[i], std::memory_order_release);
        m->params[i].current.store(pv[i], std::memory_order_release);
    }
    m->bypass_target.store(bypass_int != 0, std::memory_order_release);
    m->bypass_fade.store(fade, std::memory_order_relaxed);
    m->bypass_state.store(
        bypass_int != 0 ? BYPASS_BYPASSED : BYPASS_PROCESSING,
        std::memory_order_relaxed);

    // Apply: voice states
    if (h_present && m->weights_loaded.load(std::memory_order_acquire)) {
        for (uint32_t v = 0; v < MAX_VOICES; ++v) {
            m->voices[v].ssm_state = vstates[v].state;
            m->voices[v].env = vstates[v].env;
            m->voices[v].stage = vstates[v].env > 0.0f ? VOICE_SUSTAIN : VOICE_IDLE;
        }
        // Set voice 0 into core as starting state
        m->core->set_state(vstates[0].state);
    }

    // Resolve weight path and reload if changed
    if (path_len > 0 && path_buf[0]) {
        char resolved[MambaSynth::MAX_WEIGHT_PATH] = {0};
        if (resolve_weight_path(path_buf, resolved, sizeof(resolved))) {
            if (std::strcmp(resolved, m->weight_path) != 0) {
                m->load_weights(resolved);
            }
        }
    }

    return true;
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

    // 3. Try $MAMBA_WEIGHTS_DIR
    const char* mwd = std::getenv("MAMBA_WEIGHTS_DIR");
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
