// Preset file/JSON codecs: binary .mambapreset save/load and JSON
// export/import over plain structs. Split out of clap_presets.cpp
// (Phase 1 #3); see preset_bank.cpp (factory data) and preset_paths.cpp
// (directories).
#include "clap/clap_presets.h"
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
        if (!parse_value(out)) return false;
        skip_ws();
        return pos_ == src_.size();
    }

private:
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

    bool parse_value(JsonVal& v) {
        skip_ws();
        if (pos_ >= src_.size()) return false;
        char c = src_[pos_];
        if (c == '{') return parse_object(v);
        if (c == '[') return parse_array(v);
        if (c == '"') return parse_string(v.str_val) && (v.type = JsonVal::TypeString, true);
        if (c == 't' || c == 'f') return parse_bool(v);
        if (c == 'n') return parse_null(v);
        if (c == '-' || (c >= '0' && c <= '9')) return parse_number(v);
        return false;
    }

    bool parse_object(JsonVal& v) {
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
            if (!parse_value(val)) return false;
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

    bool parse_array(JsonVal& v) {
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
            if (!parse_value(val)) return false;
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
            char c = src_[pos_++];
            if (c == '"') return true;
            if (c == '\\') {
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
                        if (pos_ + 4 > src_.size()) return false;
                        char hex[5] = { src_[pos_], src_[pos_ + 1], src_[pos_ + 2], src_[pos_ + 3], '\0' };
                        pos_ += 4;
                        char* end = nullptr;
                        long cp = std::strtol(hex, &end, 16);
                        if (end != hex + 4) return false;
                        if (cp >= 0 && cp < 0x80) {
                            s += static_cast<char>(cp);
                        } else {
                            s += '?';
                        }
                        break;
                    }
                    default: s += esc; break;
                }
            } else {
                s += c;
            }
        }
        return false;
    }

    bool parse_number(JsonVal& v) {
        const char* start = src_.data() + pos_;
        char* end = nullptr;
        double val = std::strtod(start, &end);
        if (end == start) return false;
        pos_ += static_cast<size_t>(end - start);
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

bool save_preset(MambaSynth* m, const char* path, const char* name) {
    if (!m || !path) return false;

    // If not processing, ensure state mirror is synchronized with current atomic values
    if (!m->is_processing) {
        StateMirror* slot = m->state_mirror.begin_write();
        if (slot) {
            for (uint32_t i = 0; i < PARAM_COUNT; ++i) {
                slot->param_values[i] = m->param_values[i].load(std::memory_order_acquire);
            }
            slot->bypass_target = m->bypass_target.load(std::memory_order_acquire);
            slot->bypass_fade = m->bypass_fade.load(std::memory_order_relaxed);
            std::strncpy(slot->weight_path, m->weight_path, kMaxWeightPath - 1);
            slot->weight_path[kMaxWeightPath - 1] = '\0';
            m->state_mirror.end_write();
        }
    }

    // Pull a fresh state mirror from the audio thread.
    auto mirror_ptr = std::make_unique<StateMirror>();
    if (!mirror_ptr) return false;
    StateMirror& mirror = *mirror_ptr;
    {
        constexpr int kMaxRetries = 8;
        bool ok = false;
        for (int retry = 0; retry < kMaxRetries; ++retry) {
            std::memset(&mirror, 0, sizeof(StateMirror));
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
        if (m->core) {
            m->core->set_state(vstates[0].ssm_state);
        }
    }

    if (!m->is_processing) {
        StateMirror* slot = m->state_mirror.begin_write();
        if (slot) {
            for (uint32_t i = 0; i < param_count; ++i) {
                slot->param_values[i] = pv[i];
            }
            slot->bypass_target = (bypass_int != 0);
            slot->bypass_fade = fade;
            if (path_len > 0) {
                std::strncpy(slot->weight_path, path_buf, kMaxWeightPath - 1);
                slot->weight_path[kMaxWeightPath - 1] = '\0';
            }
            if (h_present) {
                for (uint32_t v = 0; v < MAX_VOICES; ++v) {
                    slot->voice_snapshots[v] = vstates[v];
                }
            }
            m->state_mirror.end_write();
        }
    }

    return true;
}

// ---------------------------------------------------------------------------
// JSON Preset Export / Import
// ---------------------------------------------------------------------------

bool export_preset_json_string(MambaSynth* m, const char* name, const char* category, std::string& out_json) {
    if (!m) return false;

    if (!m->is_processing) {
        StateMirror* slot = m->state_mirror.begin_write();
        if (slot) {
            for (uint32_t i = 0; i < PARAM_COUNT; ++i) {
                slot->param_values[i] = m->param_values[i].load(std::memory_order_acquire);
            }
            slot->bypass_target = m->bypass_target.load(std::memory_order_acquire);
            slot->bypass_fade = m->bypass_fade.load(std::memory_order_relaxed);
            std::strncpy(slot->weight_path, m->weight_path, kMaxWeightPath - 1);
            slot->weight_path[kMaxWeightPath - 1] = '\0';
            m->state_mirror.end_write();
        }
    }

    auto mirror_ptr = std::make_unique<StateMirror>();
    if (!mirror_ptr) return false;
    StateMirror& mirror = *mirror_ptr;
    {
        constexpr int kMaxRetries = 8;
        bool ok = false;
        for (int retry = 0; retry < kMaxRetries; ++retry) {
            std::memset(&mirror, 0, sizeof(StateMirror));
            if (m->state_mirror.read(mirror)) { ok = true; break; }
        }
        if (!ok) return false;
    }

    const std::string preset_name = name ? name : "";
    const std::string preset_cat = category ? category : "";

    char buf[2048];
    std::snprintf(buf, sizeof(buf),
        "{\n"
        "  \"format\": \"mamba-preset\",\n"
        "  \"version\": 2,\n"
        "  \"name\": \"%s\",\n"
        "  \"category\": \"%s\",\n"
        "  \"weight_path\": \"%s\",\n"
        "  \"bypass\": %s,\n"
        "  \"bypass_fade\": %.4f,\n"
        "  \"parameters\": {\n"
        "    \"gain\": %.6f,\n"
        "    \"attack\": %.6f,\n"
        "    \"release\": %.6f,\n"
        "    \"mix\": %.6f,\n"
        "    \"bypass\": %.6f\n"
        "  },\n"
        "  \"param_values\": [\n"
        "    %.6f,\n"
        "    %.6f,\n"
        "    %.6f,\n"
        "    %.6f,\n"
        "    %.6f\n"
        "  ]\n"
        "}\n",
        json_escape(preset_name).c_str(),
        json_escape(preset_cat).c_str(),
        json_escape(mirror.weight_path).c_str(),
        mirror.bypass_target ? "true" : "false",
        static_cast<double>(mirror.bypass_fade),
        mirror.param_values[PARAM_GAIN],
        mirror.param_values[PARAM_ATTACK],
        mirror.param_values[PARAM_RELEASE],
        mirror.param_values[PARAM_MIX],
        mirror.param_values[PARAM_BYPASS],
        mirror.param_values[0],
        mirror.param_values[1],
        mirror.param_values[2],
        mirror.param_values[3],
        mirror.param_values[4]
    );

    out_json = buf;
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
    if (pv_arr && pv_arr->type == JsonVal::TypeArray) {
        size_t count = std::min(pv_arr->arr_val.size(), static_cast<size_t>(PARAM_COUNT));
        for (size_t i = 0; i < count; ++i) {
            if (pv_arr->arr_val[i].type == JsonVal::TypeNumber) {
                pv[i] = pv_arr->arr_val[i].num_val;
            }
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

    if (!m->is_processing) {
        StateMirror* slot = m->state_mirror.begin_write();
        if (slot) {
            for (uint32_t i = 0; i < PARAM_COUNT; ++i) {
                slot->param_values[i] = pv[i];
            }
            slot->bypass_target = bypass_target;
            slot->bypass_fade = bypass_fade;
            if (!weight_path.empty()) {
                std::strncpy(slot->weight_path, weight_path.c_str(), kMaxWeightPath - 1);
                slot->weight_path[kMaxWeightPath - 1] = '\0';
            }
            m->state_mirror.end_write();
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
    char buf[2048];
    std::snprintf(buf, sizeof(buf),
        "{\n"
        "  \"format\": \"mamba-preset\",\n"
        "  \"version\": 2,\n"
        "  \"name\": \"%s\",\n"
        "  \"category\": \"%s\",\n"
        "  \"weight_path\": \"%s\",\n"
        "  \"bypass\": %s,\n"
        "  \"bypass_fade\": %.4f,\n"
        "  \"parameters\": {\n"
        "    \"gain\": %.6f,\n"
        "    \"attack\": %.6f,\n"
        "    \"release\": %.6f,\n"
        "    \"mix\": %.6f,\n"
        "    \"bypass\": %.6f\n"
        "  },\n"
        "  \"param_values\": [\n"
        "    %.6f,\n"
        "    %.6f,\n"
        "    %.6f,\n"
        "    %.6f,\n"
        "    %.6f\n"
        "  ]\n"
        "}\n",
        json_escape(fp.name ? fp.name : "").c_str(),
        json_escape(fp.category ? fp.category : "").c_str(),
        json_escape(fp.weight_path ? fp.weight_path : "").c_str(),
        fp.bypass_target ? "true" : "false",
        static_cast<double>(fp.bypass_fade),
        fp.param_values[PARAM_GAIN],
        fp.param_values[PARAM_ATTACK],
        fp.param_values[PARAM_RELEASE],
        fp.param_values[PARAM_MIX],
        fp.param_values[PARAM_BYPASS],
        fp.param_values[0],
        fp.param_values[1],
        fp.param_values[2],
        fp.param_values[3],
        fp.param_values[4]
    );
    out_json = buf;
    return true;
}

} // namespace mamba
