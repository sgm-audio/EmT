// Host-facing GUI dispatch (CLAP_EXT_GUI + timer support) and the helpers
// shared by every backend. Split out of clap_gui.cpp (Phase 1 #4); the
// platform code lives behind the gui_backend_* seam, see gui_backend.h.
#include "clap/ext/gui.h"
#include "clap/ext/params.h"
#include "clap/ext/timer-support.h"
#include "clap/gui_backend.h"
#include "clap/clap_presets.h"
#include <clap/clap.h>
#include <algorithm>
#include <cstdio>
#include <new>

namespace mamba {

static GuiInstance* gui(const clap_plugin_t* plugin) {
    auto* synth = static_cast<MambaSynth*>(plugin->plugin_data);
    return synth ? static_cast<GuiInstance*>(synth->gui_data) : nullptr;
}

void set_param(GuiInstance* g, clap_id id, double value) {
    if (!g || id >= PARAM_COUNT) return;
    ParamSmooth& p = g->synth->params[id];
    value = std::clamp(value, p.min_value, p.max_value);
    if (p.curve == PARAM_CURVE_STEPPED) value = value >= 0.5 ? 1.0 : 0.0;
    g->synth->param_values[id].store(value, std::memory_order_release);
    g->synth->params[id].target.store(value, std::memory_order_release);
    if (id == PARAM_BYPASS) g->synth->bypass_target.store(value > 0.5, std::memory_order_release);
    if (g->synth->host && g->synth->host->get_extension) {
        const auto* hp = static_cast<const clap_host_params_t*>(
            g->synth->host->get_extension(g->synth->host, CLAP_EXT_PARAMS));
        if (hp && hp->request_flush) hp->request_flush(g->synth->host);
    }
}

void default_preset_path(char* path, size_t size) {
    char dir[512]{};
    if (!get_preset_directory(dir, sizeof(dir))) { path[0] = '\0'; return; }
    std::snprintf(path, size, "%s/Default.mambapreset", dir);
}

static bool clap_gui_is_api_supported(const clap_plugin_t* plugin, const char* api, bool floating) {
    (void)plugin;
    if (!api || floating) return false;
    return gui_backend_is_api_supported(api);
}

static bool gui_get_preferred_api(const clap_plugin_t* plugin, const char** api, bool* floating) {
    (void)plugin;
    if (!api || !floating) return false;
    const char* preferred = gui_backend_preferred_api();
    if (!preferred) return false;
    *api = preferred;
    *floating = false;
    return true;
}

static bool gui_create(const clap_plugin_t* plugin, const char* api, bool floating) {
    if (!clap_gui_is_api_supported(plugin, api, floating)) return false;
    auto* synth = static_cast<MambaSynth*>(plugin->plugin_data);
    if (!synth || synth->gui_data) return false;

    auto* g = new (std::nothrow) GuiInstance();
    if (!g) return false;
    g->synth = synth;

    if (!gui_backend_create(g)) {
        delete g;
        return false;
    }

    if (synth->host && synth->host->get_extension) {
        g->host_timer = static_cast<const clap_host_timer_support_t*>(
            synth->host->get_extension(synth->host, CLAP_EXT_TIMER_SUPPORT));
        if (g->host_timer && g->host_timer->register_timer) {
            g->host_timer->register_timer(synth->host, 33, &g->timer_id);
        }
    }

    synth->gui_data = g;
    return true;
}

static void gui_destroy(const clap_plugin_t* plugin) {
    GuiInstance* g = gui(plugin);
    if (!g) return;

    if (g->host_timer && g->timer_id != CLAP_INVALID_ID && g->host_timer->unregister_timer) {
        g->host_timer->unregister_timer(g->synth->host, g->timer_id);
        g->timer_id = CLAP_INVALID_ID;
    }

    gui_backend_destroy(g);

    g->synth->gui_data = nullptr;
    delete g;
}

static bool gui_set_size(const clap_plugin_t* p, uint32_t w, uint32_t h) {
    GuiInstance* g = gui(p);
    if (!g) return false;
    g->width = static_cast<int>(w);
    g->height = static_cast<int>(h);
    gui_backend_set_size(g, w, h);
    return true;
}

static bool gui_set_parent(const clap_plugin_t* p, const clap_window_t* window) {
    GuiInstance* g = gui(p);
    if (!g || !window || !window->api) return false;
    return gui_backend_set_parent(g, window);
}

static bool gui_show(const clap_plugin_t* p) {
    GuiInstance* g = gui(p);
    if (!g) return false;
    return gui_backend_show(g);
}

static bool gui_hide(const clap_plugin_t* p) {
    GuiInstance* g = gui(p);
    if (!g) return false;
    return gui_backend_hide(g);
}

static void gui_on_timer(const clap_plugin_t* p, clap_id id) {
    GuiInstance* g = gui(p);
    if (!g) return;
    gui_backend_on_timer(g, id);
}

static bool gui_set_scale(const clap_plugin_t*, double) { return true; }

static bool gui_get_size(const clap_plugin_t* p, uint32_t* w, uint32_t* h) {
    if (!w || !h) return false;
    GuiInstance* g = gui(p);
    *w = g ? static_cast<uint32_t>(g->width) : 460;
    *h = g ? static_cast<uint32_t>(g->height) : 260;
    return true;
}

static bool gui_can_resize(const clap_plugin_t*) { return true; }

static bool gui_get_resize_hints(const clap_plugin_t*, clap_gui_resize_hints_t* h) {
    if (!h) return false;
    h->can_resize_horizontally = true;
    h->can_resize_vertically = true;
    h->preserve_aspect_ratio = false;
    h->aspect_ratio_width = 0;
    h->aspect_ratio_height = 0;
    return true;
}

static bool gui_adjust_size(const clap_plugin_t*, uint32_t* w, uint32_t* h) {
    if (!w || !h) return false;
    *w = std::max(*w, 320u);
    *h = std::max(*h, 220u);
    return true;
}

static bool gui_set_transient(const clap_plugin_t*, const clap_window_t*) { return false; }

static void gui_suggest_title(const clap_plugin_t*, const char*) {}

const clap_plugin_gui_t s_gui = {
    clap_gui_is_api_supported,
    gui_get_preferred_api,
    gui_create,
    gui_destroy,
    gui_set_scale,
    gui_get_size,
    gui_can_resize,
    gui_get_resize_hints,
    gui_adjust_size,
    gui_set_size,
    gui_set_parent,
    gui_set_transient,
    gui_suggest_title,
    gui_show,
    gui_hide
};

const clap_plugin_timer_support_t s_timer_support = { gui_on_timer };

} // namespace mamba
