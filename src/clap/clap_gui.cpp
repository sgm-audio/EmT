#include "clap/ext/gui.h"
#include "clap/ext/params.h"
#include "clap/ext/timer-support.h"
#include "clap/clap_plugin.h"
#include "clap/clap_presets.h"
#include <clap/clap.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <new>

#if defined(__linux__)
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#endif

namespace mamba {

#if defined(__linux__)
struct GuiInstance {
    MambaSynth* synth{};
    Display* display{};
    Window window{};
    GC gc{};
    int width{400};
    int height{250};
    int drag_param{-1};
    int drag_start_y{};
    double drag_start_value{};
    const clap_host_timer_support_t* host_timer{};
    clap_id timer_id{CLAP_INVALID_ID};
};

static GuiInstance* gui(const clap_plugin_t* plugin) {
    auto* synth = static_cast<MambaSynth*>(plugin->plugin_data);
    return synth ? static_cast<GuiInstance*>(synth->gui_data) : nullptr;
}

static void set_param(GuiInstance* g, clap_id id, double value) {
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

static void draw(GuiInstance* g) {
    if (!g || !g->display || !g->window) return;
    XSetForeground(g->display, g->gc, 0x2D2D2D);
    XFillRectangle(g->display, g->window, g->gc, 0, 0, g->width, g->height);
    XSetForeground(g->display, g->gc, 0xFFFFFF);
    const char* title = "Mamba Synth";
    XDrawString(g->display, g->window, g->gc, 10, 20, title, 11);
    const int spacing = g->width / (PARAM_COUNT + 1);
    for (uint32_t i = 0; i < PARAM_COUNT; ++i) {
        const ParamSmooth& p = g->synth->params[i];
        const double value = g->synth->param_values[i].load(std::memory_order_relaxed);
        const double norm = std::clamp((value - p.min_value) / (p.max_value - p.min_value), 0.0, 1.0);
        const int x = spacing * (static_cast<int>(i) + 1);
        XDrawArc(g->display, g->window, g->gc, x - 25, 65, 50, 50, 0, 360 * 64);
        XSetForeground(g->display, g->gc, 0x4A90D9);
        XFillRectangle(g->display, g->window, g->gc, x - 20, 120 - static_cast<int>(norm * 45), 40, static_cast<unsigned>(norm * 45));
        XSetForeground(g->display, g->gc, 0xFFFFFF);
        char label[24];
        std::snprintf(label, sizeof(label), "%u: %.2f", i + 1, value);
        XDrawString(g->display, g->window, g->gc, x - 25, 145, label, static_cast<int>(std::strlen(label)));
    }
    XDrawRectangle(g->display, g->window, g->gc, 20, g->height - 45, 100, 28);
    XDrawString(g->display, g->window, g->gc, 42, g->height - 26, "Save Preset", 11);
    XDrawRectangle(g->display, g->window, g->gc, 135, g->height - 45, 100, 28);
    XDrawString(g->display, g->window, g->gc, 157, g->height - 26, "Load Preset", 11);
    XFlush(g->display);
}

static void default_preset_path(char* path, size_t size) {
    char dir[512]{};
    if (!get_preset_directory(dir, sizeof(dir))) { path[0] = '\0'; return; }
    std::snprintf(path, size, "%s/Default.mambapreset", dir);
}

static void pump_events(GuiInstance* g) {
    if (!g || !g->display || !g->window) return;
    XEvent event;
    while (XCheckWindowEvent(g->display, g->window,
           ExposureMask | ButtonPressMask | ButtonReleaseMask | PointerMotionMask, &event)) {
        if (event.type == Expose) draw(g);
        else if (event.type == ButtonPress) {
            if (event.xbutton.y >= g->height - 45) {
                char path[768]{}; default_preset_path(path, sizeof(path));
                if (path[0] && event.xbutton.x >= 20 && event.xbutton.x <= 120)
                    save_preset(g->synth, path, "Default");
                else if (path[0] && event.xbutton.x >= 135 && event.xbutton.x <= 235)
                    load_preset(g->synth, path);
            } else {
                const int spacing = g->width / (PARAM_COUNT + 1);
                const int id = std::clamp(event.xbutton.x / std::max(spacing, 1) - 1, 0, static_cast<int>(PARAM_COUNT) - 1);
                g->drag_param = id;
                g->drag_start_y = event.xbutton.y;
                g->drag_start_value = g->synth->param_values[id].load(std::memory_order_relaxed);
            }
        } else if (event.type == MotionNotify && g->drag_param >= 0) {
            const ParamSmooth& p = g->synth->params[g->drag_param];
            const double delta = (g->drag_start_y - event.xmotion.y) / 150.0;
            set_param(g, static_cast<clap_id>(g->drag_param),
                      g->drag_start_value + delta * (p.max_value - p.min_value));
            draw(g);
        } else if (event.type == ButtonRelease) g->drag_param = -1;
    }
}
#endif

static bool gui_is_api_supported(const clap_plugin_t*, const char* api, bool floating) {
#if defined(__linux__)
    return api && !floating && std::strcmp(api, CLAP_WINDOW_API_X11) == 0;
#else
    (void)api; (void)floating; return false;
#endif
}
static bool gui_get_preferred_api(const clap_plugin_t*, const char** api, bool* floating) {
#if defined(__linux__)
    if (!api || !floating) return false; *api = CLAP_WINDOW_API_X11; *floating = false; return true;
#else
    (void)api; (void)floating; return false;
#endif
}
static bool gui_create(const clap_plugin_t* plugin, const char* api, bool floating) {
#if defined(__linux__)
    if (!gui_is_api_supported(plugin, api, floating)) return false;
    auto* synth = static_cast<MambaSynth*>(plugin->plugin_data);
    if (synth->gui_data) return false;
    auto* g = new (std::nothrow) GuiInstance();
    if (!g) return false;
    g->synth = synth; g->display = XOpenDisplay(nullptr);
    if (!g->display) { delete g; return false; }
    if (synth->host && synth->host->get_extension) {
        g->host_timer = static_cast<const clap_host_timer_support_t*>(synth->host->get_extension(synth->host, CLAP_EXT_TIMER_SUPPORT));
        if (g->host_timer && g->host_timer->register_timer)
            g->host_timer->register_timer(synth->host, 33, &g->timer_id);
    }
    synth->gui_data = g; return true;
#else
    (void)plugin; (void)api; (void)floating; return false;
#endif
}
static void gui_destroy(const clap_plugin_t* plugin) {
#if defined(__linux__)
    GuiInstance* g = gui(plugin); if (!g) return;
    if (g->host_timer && g->timer_id != CLAP_INVALID_ID && g->host_timer->unregister_timer)
        g->host_timer->unregister_timer(g->synth->host, g->timer_id);
    if (g->gc && g->display) XFreeGC(g->display, g->gc);
    if (g->window && g->display) XDestroyWindow(g->display, g->window);
    if (g->display) XCloseDisplay(g->display);
    g->synth->gui_data = nullptr; delete g;
#else
    (void)plugin;
#endif
}
static bool gui_set_scale(const clap_plugin_t*, double) { return true; }
static bool gui_get_size(const clap_plugin_t* p, uint32_t* w, uint32_t* h) {
    if (!w || !h) return false;
#if defined(__linux__)
    GuiInstance* g=gui(p); *w=g?g->width:400; *h=g?g->height:250;
#else
    (void)p; *w=400; *h=250;
#endif
    return true;
}
static bool gui_can_resize(const clap_plugin_t*) { return true; }
static bool gui_get_resize_hints(const clap_plugin_t*, clap_gui_resize_hints_t* h) {
    if (!h) return false; h->can_resize_horizontally=true; h->can_resize_vertically=true;
    h->preserve_aspect_ratio=false; h->aspect_ratio_width=0; h->aspect_ratio_height=0; return true;
}
static bool gui_adjust_size(const clap_plugin_t*, uint32_t* w, uint32_t* h) {
    if (!w || !h) return false; *w=std::max(*w,300u); *h=std::max(*h,200u); return true;
}
static bool gui_set_size(const clap_plugin_t* p, uint32_t w, uint32_t h) {
#if defined(__linux__)
    GuiInstance* g=gui(p); if (!g) return false; g->width=static_cast<int>(w); g->height=static_cast<int>(h);
    if (g->window) XResizeWindow(g->display, g->window, w, h); draw(g);
#else
    (void)p; (void)w; (void)h;
#endif
    return true;
}
static bool gui_set_parent(const clap_plugin_t* p, const clap_window_t* window) {
#if defined(__linux__)
    GuiInstance* g=gui(p); if (!g || !window || !window->api || std::strcmp(window->api,CLAP_WINDOW_API_X11)) return false;
    g->window = XCreateSimpleWindow(g->display, window->x11, 0, 0,
                                    static_cast<unsigned>(g->width), static_cast<unsigned>(g->height),
                                    0, 0, 0x2D2D2D);
    if (!g->window) return false;
    g->gc=XCreateGC(g->display,g->window,0,nullptr);
    XSelectInput(g->display,g->window,ExposureMask|ButtonPressMask|ButtonReleaseMask|PointerMotionMask);
    XMapWindow(g->display, g->window); draw(g); return true;
#else
    (void)p; (void)window; return false;
#endif
}
static bool gui_set_transient(const clap_plugin_t*, const clap_window_t*) { return false; }
static void gui_suggest_title(const clap_plugin_t*, const char*) {}
static bool gui_show(const clap_plugin_t* p) {
#if defined(__linux__)
    GuiInstance* g=gui(p); if(!g||!g->window)return false; XMapRaised(g->display,g->window); draw(g); return true;
#else
    (void)p; return false;
#endif
}
static bool gui_hide(const clap_plugin_t* p) {
#if defined(__linux__)
    GuiInstance* g=gui(p); if(!g||!g->window)return false; XUnmapWindow(g->display,g->window); return true;
#else
    (void)p; return false;
#endif
}
static void gui_on_timer(const clap_plugin_t* p, clap_id id) {
#if defined(__linux__)
    GuiInstance* g=gui(p); if(g && g->timer_id==id) { pump_events(g); draw(g); }
#else
    (void)p; (void)id;
#endif
}

const clap_plugin_gui_t s_gui = {gui_is_api_supported,gui_get_preferred_api,gui_create,gui_destroy,
    gui_set_scale,gui_get_size,gui_can_resize,gui_get_resize_hints,gui_adjust_size,gui_set_size,
    gui_set_parent,gui_set_transient,gui_suggest_title,gui_show,gui_hide};
const clap_plugin_timer_support_t s_timer_support = {gui_on_timer};
} // namespace mamba