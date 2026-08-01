#include "clap/clap_presets.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <climits>
#include <filesystem>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>

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
            ok = file_write_all(f, &mirror.voice_snapshots[v], sizeof(VoiceSnapshot));
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
    if (!file_read_u32(f, version) || (version != 0x00010000u && version != kPresetVersion)) return fail();

    uint32_t reserved = 0;
    if (!file_read_u32(f, reserved) || reserved != 0) return fail();

    uint32_t param_count = 0;
    if (!file_read_u32(f, param_count)) return fail();
    if (param_count > PARAM_COUNT) return fail();

    double pv[PARAM_COUNT] = {0};
    for (uint32_t i = 0; i < param_count; ++i)
        if (!file_read_f64(f, pv[i]) || !std::isfinite(pv[i]) ||
            pv[i] < m->params[i].min_value || pv[i] > m->params[i].max_value) return fail();

    uint32_t bypass_int = 0;
    if (!file_read_u32(f, bypass_int)) return fail();
    float fade = 0.0f;
    if (!file_read_f32(f, fade) || !std::isfinite(fade) || fade < 0.0f || fade > 1.0f) return fail();

    uint32_t h_present = 0;
    if (!file_read_u32(f, h_present) || h_present > 1 || bypass_int > 1) return fail();

    VoiceSnapshot vstates[MAX_VOICES]{};
    for (uint32_t v = 0; v < MAX_VOICES; ++v) {
        vstates[v].key = -1;
        vstates[v].channel = -1;
        vstates[v].attack_rate = MambaSynth::VOICE_ATTACK_RATE;
        vstates[v].release_rate = MambaSynth::VOICE_RELEASE_RATE;
    }
    if (h_present) {
        for (uint32_t v = 0; v < MAX_VOICES; ++v) {
            if (version >= kPresetVersion) {
                if (!file_read_all(f, &vstates[v], sizeof(VoiceSnapshot))) return fail();
            } else {
                if (!file_read_all(f, &vstates[v].ssm_state, sizeof(MambaState))) return fail();
                if (!file_read_f32(f, vstates[v].env)) return fail();
                vstates[v].stage = vstates[v].env > 0.0f ? VOICE_SUSTAIN : VOICE_IDLE;
            }
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
    if (name_len >= kMaxPresetName) return fail();
    char preset_name[kMaxPresetName]{};
    if (name_len > 0 && !file_read_all(f, preset_name, name_len)) return fail();

    std::fclose(f);
    f = nullptr;

    if (path_len > 0 && path_buf[0]) {
        char resolved[MambaSynth::MAX_WEIGHT_PATH]{};
        if (!resolve_weight_path(path_buf, resolved, sizeof(resolved)) ||
            (std::strcmp(resolved, m->weight_path) != 0 && !m->load_weights(resolved))) return false;
    }

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
            const VoiceSnapshot& saved = vstates[v];
            m->voices[v].ssm_state = saved.ssm_state;
            m->voices[v].env = saved.env;
            m->voices[v].attack_rate = saved.attack_rate;
            m->voices[v].release_rate = saved.release_rate;
            m->voices[v].velocity = saved.velocity;
            m->voices[v].key = saved.key;
            m->voices[v].channel = saved.channel;
            m->voices[v].note_id = saved.note_id;
            m->voices[v].stage = saved.stage;
            m->voice_atomics[v].stage.store(saved.stage, std::memory_order_relaxed);
            m->voice_atomics[v].env.store(saved.env, std::memory_order_relaxed);
            m->voice_atomics[v].key.store(saved.key, std::memory_order_relaxed);
            m->voice_atomics[v].channel.store(saved.channel, std::memory_order_relaxed);
            m->voice_atomics[v].velocity.store(saved.velocity, std::memory_order_relaxed);
            m->next_note_id = std::max(m->next_note_id, saved.note_id + 1);
        }
        // Set voice 0 into core as starting state
        m->core->set_state(vstates[0].ssm_state);
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
