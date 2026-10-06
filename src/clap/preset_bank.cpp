// Factory preset bank: the kFactoryPresets data table, query helpers, and
// apply_factory_preset(). Split out of clap_presets.cpp (Phase 1 #3); see
// preset_codec.cpp (file/JSON codecs) and preset_paths.cpp (directories).
#include "clap/clap_presets.h"
#include <array>
#include <cstring>

namespace mamba {

// ---------------------------------------------------------------------------
// Factory Preset Bank
// ---------------------------------------------------------------------------

const std::array<FactoryPreset, kFactoryPresetCount> kFactoryPresets = {{
    {
        "01 - Mamba Default",
        "Default",
        make_synth_preset_values(1.0, 0.100, 0.300, 1.0, 0.0),
        false,
        0.0f,
        ""
    },
    {
        "02 - Dark Lead",
        "Lead",
        make_synth_preset_values(1.2, 0.015, 0.400, 0.85, 0.0),
        false,
        0.0f,
        ""
    },
    {
        "03 - Lush Pad",
        "Pad",
        make_synth_preset_values(0.9, 0.600, 1.500, 1.0, 0.0),
        false,
        0.0f,
        ""
    },
    {
        "04 - Resonant Sub",
        "Bass",
        make_synth_preset_values(1.4, 0.005, 0.250, 1.0, 0.0),
        false,
        0.0f,
        ""
    },
    {
        "05 - Fast Perc",
        "Percussion",
        make_synth_preset_values(1.1, 0.001, 0.050, 1.0, 0.0),
        false,
        0.0f,
        ""
    },
    {
        "06 - Neural Delay",
        "FX",
        make_synth_preset_values(0.85, 0.050, 0.800, 0.5, 0.0),
        false,
        0.0f,
        ""
    }
}};

size_t get_factory_preset_count() {
    return kFactoryPresetCount;
}

const FactoryPreset* get_factory_preset(size_t index) {
    if (index >= kFactoryPresetCount) return nullptr;
    return &kFactoryPresets[index];
}

const FactoryPreset* find_factory_preset(const char* name) {
    if (!name || !name[0]) return nullptr;
    for (size_t i = 0; i < kFactoryPresetCount; ++i) {
        if (std::strcmp(kFactoryPresets[i].name, name) == 0) {
            return &kFactoryPresets[i];
        }
    }
    return nullptr;
}

bool apply_factory_preset(MambaSynth* m, const FactoryPreset* fp) {
    if (!m || !fp) return false;

    for (uint32_t i = 0; i < PARAM_COUNT; ++i) {
        double v = fp->param_values[i];
        if (v < m->params[i].min_value) v = m->params[i].min_value;
        if (v > m->params[i].max_value) v = m->params[i].max_value;
        m->param_values[i].store(v, std::memory_order_release);
        m->params[i].target.store(v, std::memory_order_release);
        m->params[i].current.store(v, std::memory_order_release);
    }
    m->bypass_target.store(fp->bypass_target, std::memory_order_release);
    m->bypass_fade.store(fp->bypass_fade, std::memory_order_relaxed);
    m->bypass_state.store(
        fp->bypass_target ? BYPASS_BYPASSED : BYPASS_PROCESSING,
        std::memory_order_relaxed);

    if (fp->weight_path && fp->weight_path[0]) {
        char resolved[MambaSynth::MAX_WEIGHT_PATH]{};
        if (resolve_weight_path(fp->weight_path, resolved, sizeof(resolved))) {
            if (std::strcmp(resolved, m->weight_path) != 0) {
                m->load_weights(resolved);
            }
        }
    }

    if (!m->is_processing) {
        StateMirror* slot = m->state_mirror.begin_write();
        if (slot) {
            for (uint32_t i = 0; i < PARAM_COUNT; ++i) {
                slot->param_values[i] = m->param_values[i].load(std::memory_order_acquire);
            }
            slot->bypass_target = fp->bypass_target;
            slot->bypass_fade = fp->bypass_fade;
            if (fp->weight_path && fp->weight_path[0]) {
                std::strncpy(slot->weight_path, fp->weight_path, kMaxWeightPath - 1);
                slot->weight_path[kMaxWeightPath - 1] = '\0';
            }
            m->state_mirror.end_write();
        }
    }

    return true;
}

bool apply_factory_preset(MambaSynth* m, size_t index) {
    const FactoryPreset* fp = get_factory_preset(index);
    return apply_factory_preset(m, fp);
}

bool apply_factory_preset(MambaSynth* m, const char* name) {
    const FactoryPreset* fp = find_factory_preset(name);
    return apply_factory_preset(m, fp);
}

} // namespace mamba
