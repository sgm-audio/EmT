// Preset file/JSON codecs: binary .mambapreset save/load and JSON
// export/import over plain structs. Split out of clap_presets.cpp
// (Phase 1 #3); see preset_bank.cpp (factory data) and preset_paths.cpp
// (directories).
#include "clap/clap_presets.h"
#include <charconv>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>
#include <algorithm>
#include <cmath>
#include <utility>
#include <memory>
#include <new>

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

// ---------------------------------------------------------------------------
// JSON serialization and parsing helpers
// ---------------------------------------------------------------------------

static std::string json_escape(std::string_view s) {
    std::string res;
    res.reserve(s.size() + 16);
    for (char c : s) {
        switch (c) {
            case '"':  res += "\\\""; break;
            case '\\': res += "\\\\"; break;
            case '\b': res += "\\b";  break;
            case '\f': res += "\\f";  break;
            case '\n': res += "\\n";  break;
            case '\r': res += "\\r";  break;
            case '\t': res += "\\t";  break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned int>(static_cast<unsigned char>(c)));
                    res += buf;
                } else {
                    res += c;
                }
                break;
        }
    }
    return res;
}

struct JsonVal {
    enum Type { TypeNull, TypeBool, TypeNumber, TypeString, TypeArray, TypeObject };
    Type type = TypeNull;
    bool bool_val = false;
    double num_val = 0.0;
    std::string str_val;
    std::vector<JsonVal> arr_val;
    std::vector<std::pair<std::string, JsonVal>> obj_val;

    const JsonVal* get(const std::string& key) const {
        if (type != TypeObject) return nullptr;
        for (const auto& kv : obj_val) {
            if (kv.first == key) return &kv.second;
        }
        return nullptr;
    }
};

class JsonParser {
public:
    explicit JsonParser(std::string_view src) : src_(src), pos_(0) {}

    bool parse(JsonVal& out) {
        skip_ws();
        if (pos_ >= src_.size()) return false;
        if (!parse_value(out, 0)) return false;
        skip_ws();
        return pos_ == src_.size();
    }

private:
    static constexpr int kMaxDepth = 64;
    std::string_view src_;
    size_t pos_;

    void skip_ws() {
        while (pos_ < src_.size()) {
            char c = src_[pos_];
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
                ++pos_;
            } else {
                break;
            }
        }
    }

    static bool parse_hex4(std::string_view sv, size_t pos, uint32_t& out) {
        if (pos + 4 > sv.size()) return false;
        uint32_t v = 0;
        for (size_t i = 0; i < 4; ++i) {
            char c = sv[pos + i];
            uint32_t d = 0;
            if (c >= '0' && c <= '9') d = static_cast<uint32_t>(c - '0');
            else if (c >= 'a' && c <= 'f') d = static_cast<uint32_t>(10 + c - 'a');
            else if (c >= 'A' && c <= 'F') d = static_cast<uint32_t>(10 + c - 'A');
            else return false;
            v = (v << 4) | d;
        }
        out = v;
        return true;
    }

    static void append_utf8(std::string& s, uint32_t cp) {
        if (cp < 0x80) {
            s += static_cast<char>(cp);
        } else if (cp < 0x800) {
            s += static_cast<char>(0xC0u | (cp >> 6));
            s += static_cast<char>(0x80u | (cp & 0x3Fu));
        } else if (cp < 0x10000) {
            s += static_cast<char>(0xE0u | (cp >> 12));
            s += static_cast<char>(0x80u | ((cp >> 6) & 0x3Fu));
            s += static_cast<char>(0x80u | (cp & 0x3Fu));
        } else {
            s += static_cast<char>(0xF0u | (cp >> 18));
            s += static_cast<char>(0x80u | ((cp >> 12) & 0x3Fu));
            s += static_cast<char>(0x80u | ((cp >> 6) & 0x3Fu));
            s += static_cast<char>(0x80u | (cp & 0x3Fu));
        }
    }

    bool parse_value(JsonVal& v, int depth) {
        if (depth > kMaxDepth) return false;
        skip_ws();
        if (pos_ >= src_.size()) return false;
        char c = src_[pos_];
        if (c == '{') return parse_object(v, depth);
        if (c == '[') return parse_array(v, depth);
        if (c == '"') return parse_string(v.str_val) && (v.type = JsonVal::TypeString, true);
        if (c == 't' || c == 'f') return parse_bool(v);
        if (c == 'n') return parse_null(v);
        if (c == '-' || (c >= '0' && c <= '9')) return parse_number(v);
        return false;
    }

    bool parse_object(JsonVal& v, int depth) {
        if (pos_ >= src_.size() || src_[pos_] != '{') return false;
        ++pos_;
        v.type = JsonVal::TypeObject;
        v.obj_val.clear();
        skip_ws();
        if (pos_ < src_.size() && src_[pos_] == '}') {
            ++pos_;
            return true;
        }
        while (pos_ < src_.size()) {
            skip_ws();
            if (pos_ >= src_.size() || src_[pos_] != '"') return false;
            std::string key;
            if (!parse_string(key)) return false;
            skip_ws();
            if (pos_ >= src_.size() || src_[pos_] != ':') return false;
            ++pos_;
            JsonVal val;
            if (!parse_value(val, depth + 1)) return false;
            v.obj_val.emplace_back(std::move(key), std::move(val));
            skip_ws();
            if (pos_ < src_.size() && src_[pos_] == '}') {
                ++pos_;
                return true;
            }
            if (pos_ >= src_.size() || src_[pos_] != ',') return false;
            ++pos_;
        }
        return false;
    }

    bool parse_array(JsonVal& v, int depth) {
        if (pos_ >= src_.size() || src_[pos_] != '[') return false;
        ++pos_;
        v.type = JsonVal::TypeArray;
        v.arr_val.clear();
        skip_ws();
        if (pos_ < src_.size() && src_[pos_] == ']') {
            ++pos_;
            return true;
        }
        while (pos_ < src_.size()) {
            JsonVal val;
            if (!parse_value(val, depth + 1)) return false;
            v.arr_val.push_back(std::move(val));
            skip_ws();
            if (pos_ < src_.size() && src_[pos_] == ']') {
                ++pos_;
                return true;
            }
            if (pos_ >= src_.size() || src_[pos_] != ',') return false;
            ++pos_;
        }
        return false;
    }

    bool parse_string(std::string& s) {
        if (pos_ >= src_.size() || src_[pos_] != '"') return false;
        ++pos_;
        s.clear();
        while (pos_ < src_.size()) {
            unsigned char uc = static_cast<unsigned char>(src_[pos_++]);
            if (uc == '"') return true;
            if (uc < 0x20u) return false;  // RFC 8259 forbids unescaped control chars
            if (uc == '\\') {
                if (pos_ >= src_.size()) return false;
                char esc = src_[pos_++];
                switch (esc) {
                    case '"':  s += '"'; break;
                    case '\\': s += '\\'; break;
                    case '/':  s += '/'; break;
                    case 'b':  s += '\b'; break;
                    case 'f':  s += '\f'; break;
                    case 'n':  s += '\n'; break;
                    case 'r':  s += '\r'; break;
                    case 't':  s += '\t'; break;
                    case 'u': {
                        uint32_t cp = 0;
                        if (!parse_hex4(src_, pos_, cp)) return false;
                        pos_ += 4;
                        if (cp >= 0xD800u && cp <= 0xDBFFu) {
                            if (pos_ + 6 > src_.size() || src_[pos_] != '\\' || src_[pos_ + 1] != 'u') {
                                return false;
                            }
                            uint32_t low = 0;
                            if (!parse_hex4(src_, pos_ + 2, low) || low < 0xDC00u || low > 0xDFFFu) {
                                return false;
                            }
                            pos_ += 6;
                            cp = 0x10000u + (((cp - 0xD800u) << 10) | (low - 0xDC00u));
                        } else if (cp >= 0xDC00u && cp <= 0xDFFFu) {
                            return false;  // Lone low surrogate
                        }
                        append_utf8(s, cp);
                        break;
                    }
                    default: return false;  // Invalid RFC 8259 escape
                }
            } else {
                s += static_cast<char>(uc);
            }
        }
        return false;
    }

    bool parse_number(JsonVal& v) {
        size_t p = pos_;
        if (p < src_.size() && src_[p] == '-') ++p;
        if (p >= src_.size()) return false;
        if (src_[p] == '0') {
            ++p;
        } else if (src_[p] >= '1' && src_[p] <= '9') {
            while (p < src_.size() && src_[p] >= '0' && src_[p] <= '9') ++p;
        } else {
            return false;
        }
        if (p < src_.size() && src_[p] == '.') {
            ++p;
            if (p >= src_.size() || src_[p] < '0' || src_[p] > '9') return false;
            while (p < src_.size() && src_[p] >= '0' && src_[p] <= '9') ++p;
        }
        if (p < src_.size() && (src_[p] == 'e' || src_[p] == 'E')) {
            ++p;
            if (p < src_.size() && (src_[p] == '+' || src_[p] == '-')) ++p;
            if (p >= src_.size() || src_[p] < '0' || src_[p] > '9') return false;
            while (p < src_.size() && src_[p] >= '0' && src_[p] <= '9') ++p;
        }
        double val = 0.0;
        auto res = std::from_chars(src_.data() + pos_, src_.data() + p, val);
        if (res.ec != std::errc{} || res.ptr != src_.data() + p || !std::isfinite(val)) {
            return false;
        }
        pos_ = p;
        v.type = JsonVal::TypeNumber;
        v.num_val = val;
        return true;
    }

    bool parse_bool(JsonVal& v) {
        if (src_.substr(pos_, 4) == "true") {
            v.type = JsonVal::TypeBool;
            v.bool_val = true;
            pos_ += 4;
            return true;
        }
        if (src_.substr(pos_, 5) == "false") {
            v.type = JsonVal::TypeBool;
            v.bool_val = false;
            pos_ += 5;
            return true;
        }
        return false;
    }

    bool parse_null(JsonVal& v) {
        if (src_.substr(pos_, 4) == "null") {
            v.type = JsonVal::TypeNull;
            pos_ += 4;
            return true;
        }
        return false;
    }
};

} // namespace

static bool all_finite_floats(const float* p, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        if (!std::isfinite(p[i])) return false;
    }
    return true;
}

static bool ssm_state_all_finite(const MambaState& s) {
    for (size_t l = 0; l < MAX_LAYERS; ++l) {
        const SSMState& st = s.layers[l];
        if (!all_finite_floats(&st.ssm_real[0][0][0], sizeof(st.ssm_real) / sizeof(float)) ||
            !all_finite_floats(&st.ssm_imag[0][0][0], sizeof(st.ssm_imag) / sizeof(float)) ||
            !all_finite_floats(&st.k_state[0][0][0], sizeof(st.k_state) / sizeof(float)) ||
            !all_finite_floats(&st.angle_state[0][0], sizeof(st.angle_state) / sizeof(float))) {
            return false;
        }
    }
    return true;
}

bool save_preset(MambaSynth* m, const char* path, const char* name) {
    if (!m || !path) return false;

    double pv[PARAM_COUNT]{};
    for (uint32_t i = 0; i < PARAM_COUNT; ++i) {
        pv[i] = m->param_values[i].load(std::memory_order_acquire);
    }
    const bool bypass_target = m->bypass_target.load(std::memory_order_acquire);
    const float bypass_fade = m->bypass_fade.load(std::memory_order_relaxed);
    const bool h_present = m->weights_loaded.load(std::memory_order_acquire);

    std::unique_ptr<StateMirror> mirror_ptr;
    if (h_present && m->is_processing.load(std::memory_order_acquire)) {
        mirror_ptr.reset(new (std::nothrow) StateMirror());
        if (!mirror_ptr) return false;
        constexpr int kMaxRetries = 8;
        bool ok_read = false;
        for (int retry = 0; retry < kMaxRetries; ++retry) {
            std::memset(mirror_ptr.get(), 0, sizeof(StateMirror));
            if (m->state_mirror.read(*mirror_ptr)) { ok_read = true; break; }
        }
        if (!ok_read) return false;
    }

    std::FILE* f = std::fopen(path, "wb");
    if (!f) return false;

    bool ok = true;
    ok = ok && file_write_u32(f, kPresetMagic);
    ok = ok && file_write_u32(f, kPresetVersion);
    ok = ok && file_write_u32(f, 0); // reserved
    ok = ok && file_write_u32(f, PARAM_COUNT);
    for (uint32_t i = 0; i < PARAM_COUNT && ok; ++i)
        ok = file_write_f64(f, pv[i]);
    ok = ok && file_write_u32(f, bypass_target ? 1u : 0u);
    ok = ok && file_write_f32(f, bypass_fade);

    ok = ok && file_write_u32(f, h_present ? 1u : 0u);
    if (h_present && ok) {
        if (mirror_ptr) {
            for (uint32_t v = 0; v < MAX_VOICES && ok; ++v) {
                ok = file_write_all(f, &mirror_ptr->voice_snapshots[v], sizeof(VoiceSnapshot));
            }
        } else {
            for (uint32_t v = 0; v < MAX_VOICES && ok; ++v) {
                VoiceSnapshot snap{};
                snap.ssm_state = m->voices[v].ssm_state;
                snap.env = m->voices[v].env;
                snap.attack_rate = m->voices[v].attack_rate;
                snap.release_rate = m->voices[v].release_rate;
                snap.velocity = m->voices[v].velocity;
                snap.key = m->voices[v].key;
                snap.channel = m->voices[v].channel;
                snap.note_id = m->voices[v].note_id;
                snap.stage = m->voices[v].stage;
                ok = file_write_all(f, &snap, sizeof(VoiceSnapshot));
            }
        }
    }

    uint32_t path_len = 0;
    if (m->weight_path[0]) {
        path_len = static_cast<uint32_t>(std::strlen(m->weight_path));
        if (path_len >= MambaSynth::MAX_WEIGHT_PATH)
            path_len = MambaSynth::MAX_WEIGHT_PATH - 1;
    }
    ok = ok && file_write_u32(f, path_len);
    if (path_len > 0 && ok)
        ok = file_write_all(f, m->weight_path, path_len);

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
    if (param_count == 0 || param_count > PARAM_COUNT) return fail();

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

    auto vs_ptr = std::unique_ptr<VoiceSnapshot[]>(new (std::nothrow) VoiceSnapshot[MAX_VOICES]());
    if (!vs_ptr) return fail();
    VoiceSnapshot* vstates = vs_ptr.get();
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
                if (vstates[v].stage > VOICE_SILENT ||
                    !std::isfinite(vstates[v].env) || vstates[v].env < 0.0f || vstates[v].env > 1.0f ||
                    !std::isfinite(vstates[v].velocity) || vstates[v].velocity < 0.0f || vstates[v].velocity > 1.0f ||
                    vstates[v].key < -1 || vstates[v].key > 127 ||
                    !std::isfinite(vstates[v].attack_rate) || vstates[v].attack_rate < 0.0f ||
                    !std::isfinite(vstates[v].release_rate) || vstates[v].release_rate < 0.0f ||
                    !ssm_state_all_finite(vstates[v].ssm_state)) return fail();
            } else {
                if (!file_read_all(f, &vstates[v].ssm_state, sizeof(MambaState))) return fail();
                if (!ssm_state_all_finite(vstates[v].ssm_state)) return fail();
                if (!file_read_f32(f, vstates[v].env) || !std::isfinite(vstates[v].env) ||
                    vstates[v].env < 0.0f || vstates[v].env > 1.0f) return fail();
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

    std::memset(m->weight_path, 0, sizeof(m->weight_path));
    if (path_len > 0 && path_buf[0]) {
        char resolved[MambaSynth::MAX_WEIGHT_PATH]{};
        if (resolve_weight_path(path_buf, resolved, sizeof(resolved))) {
            std::snprintf(m->weight_path, sizeof(m->weight_path), "%s", resolved);
            std::FILE* probe = std::fopen(resolved, "rb");
            if (probe) {
                std::fclose(probe);
                if (!m->load_weights(resolved)) return false;
            } else {
                m->weights_loaded.store(false, std::memory_order_release);
            }
        } else {
            m->weights_loaded.store(false, std::memory_order_release);
        }
    } else {
        m->weights_loaded.store(false, std::memory_order_release);
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
            m->voices[v].clap_note_id = -1;
            m->voices[v].note_id = saved.note_id;
            m->voices[v].stage = saved.stage;
            m->voice_atomics[v].stage.store(saved.stage, std::memory_order_relaxed);
            m->voice_atomics[v].env.store(saved.env, std::memory_order_relaxed);
            m->voice_atomics[v].key.store(saved.key, std::memory_order_relaxed);
            m->voice_atomics[v].channel.store(saved.channel, std::memory_order_relaxed);
            m->voice_atomics[v].velocity.store(saved.velocity, std::memory_order_relaxed);
            if (saved.note_id < UINT64_MAX) {
                m->next_note_id = std::max(m->next_note_id, saved.note_id + 1);
            }
        }
        // Set voice 0 into core as starting state (main thread; audio is
        // stopped while a preset/state load runs - CLAP contract).
        if (auto* c = m->core.published()) {
            c->set_state(vstates[0].ssm_state);
        }
    }

    if (!m->is_processing.load(std::memory_order_acquire)) {
        StateMirror* slot = m->state_mirror.begin_write();
        if (slot) {
            for (uint32_t i = 0; i < param_count; ++i) {
                slot->param_values[i] = pv[i];
            }
            for (uint32_t i = param_count; i < PARAM_COUNT; ++i) {
                slot->param_values[i] = m->param_values[i].load(std::memory_order_acquire);
            }
            slot->bypass_target = (bypass_int != 0);
            slot->bypass_fade = fade;
            for (uint32_t v = 0; v < MAX_VOICES; ++v) {
                if (h_present) {
                    slot->voice_snapshots[v] = vstates[v];
                } else {
                    slot->voice_snapshots[v].ssm_state = m->voices[v].ssm_state;
                    slot->voice_snapshots[v].env = m->voices[v].env;
                    slot->voice_snapshots[v].attack_rate = m->voices[v].attack_rate;
                    slot->voice_snapshots[v].release_rate = m->voices[v].release_rate;
                    slot->voice_snapshots[v].velocity = m->voices[v].velocity;
                    slot->voice_snapshots[v].key = m->voices[v].key;
                    slot->voice_snapshots[v].channel = m->voices[v].channel;
                    slot->voice_snapshots[v].note_id = m->voices[v].note_id;
                    slot->voice_snapshots[v].stage = m->voices[v].stage;
                }
            }
            m->state_mirror.end_write();
            m->state_mirror.swap();
        }
    }

    return true;
}

namespace {
std::string format_json_fixed(double v, int precision) {
    if (!std::isfinite(v)) v = 0.0;
    char buf[64];
    auto res = std::to_chars(buf, buf + sizeof(buf), v, std::chars_format::fixed, precision);
    if (res.ec == std::errc{}) {
        return std::string(buf, static_cast<size_t>(res.ptr - buf));
    }
    return "0.0";
}

std::string build_preset_json(const std::string& name, const std::string& category,
                              const std::string& weight_path, bool bypass_target,
                              float bypass_fade, const double* param_values) {
    std::string out;
    out.reserve(1024);
    out += "{\n"
           "  \"format\": \"mamba-preset\",\n"
           "  \"version\": 2,\n"
           "  \"name\": \"" + json_escape(name) + "\",\n"
           "  \"category\": \"" + json_escape(category) + "\",\n"
           "  \"weight_path\": \"" + json_escape(weight_path) + "\",\n"
           "  \"bypass\": ";
    out += (bypass_target ? "true" : "false");
    out += ",\n";
    out += "  \"bypass_fade\": " + format_json_fixed(static_cast<double>(bypass_fade), 4) + ",\n"
           "  \"parameters\": {\n"
           "    \"gain\": " + format_json_fixed(param_values[PARAM_GAIN], 6) + ",\n"
           "    \"attack\": " + format_json_fixed(param_values[PARAM_ATTACK], 6) + ",\n"
           "    \"release\": " + format_json_fixed(param_values[PARAM_RELEASE], 6) + ",\n"
           "    \"mix\": " + format_json_fixed(param_values[PARAM_MIX], 6) + ",\n"
           "    \"bypass\": " + format_json_fixed(param_values[PARAM_BYPASS], 6) + "\n"
           "  },\n"
           "  \"param_values\": [\n"
           "    " + format_json_fixed(param_values[0], 6) + ",\n"
           "    " + format_json_fixed(param_values[1], 6) + ",\n"
           "    " + format_json_fixed(param_values[2], 6) + ",\n"
           "    " + format_json_fixed(param_values[3], 6) + ",\n"
           "    " + format_json_fixed(param_values[4], 6) + "\n"
           "  ]\n"
           "}\n";
    return out;
}
} // namespace

// ---------------------------------------------------------------------------
// JSON Preset Export / Import
// ---------------------------------------------------------------------------

bool export_preset_json_string(MambaSynth* m, const char* name, const char* category, std::string& out_json) {
    if (!m) return false;

    double pv[PARAM_COUNT]{};
    for (uint32_t i = 0; i < PARAM_COUNT; ++i) {
        pv[i] = m->param_values[i].load(std::memory_order_acquire);
    }
    const bool bypass_target = m->bypass_target.load(std::memory_order_acquire);
    const float bypass_fade = m->bypass_fade.load(std::memory_order_relaxed);

    out_json = build_preset_json(
        name ? name : "",
        category ? category : "",
        m->weight_path,
        bypass_target,
        bypass_fade,
        pv);
    return true;
}

bool import_preset_json_string(MambaSynth* m, const std::string& json_str) {
    if (!m || json_str.empty()) return false;

    JsonParser parser(json_str);
    JsonVal root;
    if (!parser.parse(root) || root.type != JsonVal::TypeObject) {
        return false;
    }

    double pv[PARAM_COUNT];
    for (uint32_t i = 0; i < PARAM_COUNT; ++i) {
        pv[i] = m->params[i].default_value;
    }
    bool bypass_target = false;
    float bypass_fade = 0.0f;
    std::string weight_path;

    // Check "param_values" array first
    const JsonVal* pv_arr = root.get("param_values");
    if (pv_arr) {
        if (pv_arr->type != JsonVal::TypeArray) return false;
        size_t count = std::min(pv_arr->arr_val.size(), static_cast<size_t>(PARAM_COUNT));
        for (size_t i = 0; i < count; ++i) {
            if (pv_arr->arr_val[i].type != JsonVal::TypeNumber) return false;
            pv[i] = pv_arr->arr_val[i].num_val;
        }
    }

    // Check "parameters" object (overrides array values if present)
    const JsonVal* params_obj = root.get("parameters");
    if (params_obj && params_obj->type == JsonVal::TypeObject) {
        const JsonVal* g = params_obj->get("gain");
        if (g && g->type == JsonVal::TypeNumber) pv[PARAM_GAIN] = g->num_val;

        const JsonVal* a = params_obj->get("attack");
        if (a && a->type == JsonVal::TypeNumber) pv[PARAM_ATTACK] = a->num_val;

        const JsonVal* r = params_obj->get("release");
        if (r && r->type == JsonVal::TypeNumber) pv[PARAM_RELEASE] = r->num_val;

        const JsonVal* mx = params_obj->get("mix");
        if (mx && mx->type == JsonVal::TypeNumber) pv[PARAM_MIX] = mx->num_val;

        const JsonVal* byp = params_obj->get("bypass");
        if (byp) {
            if (byp->type == JsonVal::TypeNumber) pv[PARAM_BYPASS] = byp->num_val;
            else if (byp->type == JsonVal::TypeBool) pv[PARAM_BYPASS] = byp->bool_val ? 1.0 : 0.0;
        }
    }

    // Validate parameters
    for (uint32_t i = 0; i < PARAM_COUNT; ++i) {
        if (!std::isfinite(pv[i])) return false;
        if (pv[i] < m->params[i].min_value || pv[i] > m->params[i].max_value) {
            return false;
        }
    }

    // Check "bypass"
    const JsonVal* bp = root.get("bypass");
    if (bp) {
        if (bp->type == JsonVal::TypeBool) bypass_target = bp->bool_val;
        else if (bp->type == JsonVal::TypeNumber) bypass_target = (bp->num_val > 0.5);
    } else {
        bypass_target = (pv[PARAM_BYPASS] > 0.5);
    }

    const JsonVal* bf = root.get("bypass_fade");
    if (bf && bf->type == JsonVal::TypeNumber) {
        bypass_fade = static_cast<float>(bf->num_val);
        if (!std::isfinite(bypass_fade) || bypass_fade < 0.0f || bypass_fade > 1.0f) {
            bypass_fade = 0.0f;
        }
    }

    const JsonVal* wp = root.get("weight_path");
    if (wp && wp->type == JsonVal::TypeString) {
        weight_path = wp->str_val;
    }

    if (!weight_path.empty()) {
        char resolved[MambaSynth::MAX_WEIGHT_PATH]{};
        if (!resolve_weight_path(weight_path.c_str(), resolved, sizeof(resolved)) ||
            (std::strcmp(resolved, m->weight_path) != 0 && !m->load_weights(resolved))) {
            return false;
        }
    }

    // Apply parameters
    for (uint32_t i = 0; i < PARAM_COUNT; ++i) {
        m->param_values[i].store(pv[i], std::memory_order_release);
        m->params[i].target.store(pv[i], std::memory_order_release);
        m->params[i].current.store(pv[i], std::memory_order_release);
    }
    m->bypass_target.store(bypass_target, std::memory_order_release);
    m->bypass_fade.store(bypass_fade, std::memory_order_relaxed);
    m->bypass_state.store(
        bypass_target ? BYPASS_BYPASSED : BYPASS_PROCESSING,
        std::memory_order_relaxed);

    if (!m->is_processing.load(std::memory_order_acquire)) {
        StateMirror* slot = m->state_mirror.begin_write();
        if (slot) {
            for (uint32_t v = 0; v < MAX_VOICES; ++v) {
                slot->voice_snapshots[v].ssm_state = m->voices[v].ssm_state;
                slot->voice_snapshots[v].env = m->voices[v].env;
                slot->voice_snapshots[v].attack_rate = m->voices[v].attack_rate;
                slot->voice_snapshots[v].release_rate = m->voices[v].release_rate;
                slot->voice_snapshots[v].velocity = m->voices[v].velocity;
                slot->voice_snapshots[v].key = m->voices[v].key;
                slot->voice_snapshots[v].channel = m->voices[v].channel;
                slot->voice_snapshots[v].note_id = m->voices[v].note_id;
                slot->voice_snapshots[v].stage = m->voices[v].stage;
            }
            for (uint32_t i = 0; i < PARAM_COUNT; ++i) {
                slot->param_values[i] = pv[i];
            }
            slot->bypass_target = bypass_target;
            slot->bypass_fade = bypass_fade;
            m->state_mirror.end_write();
            m->state_mirror.swap();
        }
    }

    return true;
}

bool export_preset_json(MambaSynth* m, const char* path, const char* name, const char* category) {
    if (!m || !path) return false;
    std::string json;
    if (!export_preset_json_string(m, name, category, json)) return false;

    std::FILE* f = std::fopen(path, "wb");
    if (!f) return false;
    size_t written = std::fwrite(json.data(), 1, json.size(), f);
    std::fclose(f);
    return written == json.size();
}

bool import_preset_json(MambaSynth* m, const char* path) {
    if (!m || !path) return false;

    std::FILE* f = std::fopen(path, "rb");
    if (!f) return false;
    std::fseek(f, 0, SEEK_END);
    long sz = std::ftell(f);
    if (sz < 0 || sz > 10 * 1024 * 1024) { // reject > 10MB
        std::fclose(f);
        return false;
    }
    std::fseek(f, 0, SEEK_SET);

    std::string content;
    content.resize(static_cast<size_t>(sz));
    size_t read_bytes = std::fread(content.data(), 1, content.size(), f);
    std::fclose(f);
    if (read_bytes != content.size()) return false;

    return import_preset_json_string(m, content);
}

bool export_factory_preset_json(const FactoryPreset& fp, std::string& out_json) {
    out_json = build_preset_json(
        fp.name ? fp.name : "",
        fp.category ? fp.category : "",
        fp.weight_path ? fp.weight_path : "",
        fp.bypass_target,
        fp.bypass_fade,
        fp.param_values.data());
    return true;
}

} // namespace mamba
