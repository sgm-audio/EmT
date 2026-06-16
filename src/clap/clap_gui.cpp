#include "clap/ext/gui.h"
#include "clap/clap_plugin.h"
#include <clap/clap.h>
#include <cstring>
#include <cmath>
#include <cstdio>

// ponytail: X11-only GUI for Linux. macOS/Windows need platform-specific
// drawing; add when cross-platform GUI is required.

#if defined(__linux__)
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/Xatom.h>
#endif

namespace mamba {

#if defined(__linux__)
static Display* s_display = nullptr;
static bool s_gui_initialized = false;

struct GuiInstance {
    MambaSynth* synth;
    Window window;
    GC gc;
    int width;
    int height;
    int drag_param;     // -1 = none, else param id being dragged
    int drag_start_y;
    double drag_start_value;

    unsigned long bg_color;
    unsigned long fg_color;
    unsigned long knob_color;
    unsigned long text_color;
    unsigned long active_color;

    void init_colors() {
        bg_color = 0x2D2D2D;
        fg_color = 0xCCCCCC;
        knob_color = 0x4A90D9;
        text_color = 0xFFFFFF;
        active_color = 0xF5A623;
    }
};

static GuiInstance s_gui_state;

static const char* param_label(int id) {
    switch (id) {
        case PARAM_GAIN:    return "Gain";
        case PARAM_ATTACK:  return "Attack";
        case PARAM_RELEASE: return "Release";
        case PARAM_MIX:     return "Mix";
        case PARAM_BYPASS:  return "Bypass";
        default:            return "";
    }
}

static void draw_knob(Display* dpy, Window win, GC gc, int cx, int cy, int r,
                       float value, const char* label, bool active) {
    XSetForeground(dpy, gc, s_gui_state.fg_color);
    XDrawArc(dpy, win, gc, cx - r, cy - r, r * 2, r * 2, 0, 360 * 64);

    int start_angle = 225 * 64;
    int arc_span = static_cast<int>(value * 270.0f * 64);
    if (arc_span > 0) {
        XSetForeground(dpy, gc, active ? s_gui_state.active_color : s_gui_state.knob_color);
        XDrawArc(dpy, win, gc, cx - r, cy - r, r * 2, r * 2, start_angle, arc_span);
    }

    XSetForeground(dpy, gc, s_gui_state.fg_color);
    XFillArc(dpy, win, gc, cx - 3, cy - 3, 6, 6, 0, 360 * 64);

    XSetForeground(dpy, gc, s_gui_state.text_color);
    XDrawString(dpy, win, gc, cx - 4 * static_cast<int>(std::strlen(label)) / 2,
                cy + r + 20, label, static_cast<int>(std::strlen(label)));
}

static void gui_redraw() {
    if (!s_display || !s_gui_state.window) return;
    Display* dpy = s_display;
    Window win = s_gui_state.window;

    XSetForeground(dpy, s_gui_state.gc, s_gui_state.bg_color);
    XFillRectangle(dpy, win, s_gui_state.gc, 0, 0, s_gui_state.width, s_gui_state.height);

    int knob_count = PARAM_COUNT;
    int spacing = s_gui_state.width / (knob_count + 1);
    int cy = s_gui_state.height / 2 - 20;
    int knob_r = 30;

    for (int i = 0; i < knob_count; ++i) {
        int cx = spacing * (i + 1);
        double val = s_gui_state.synth->param_values[i].load(std::memory_order_relaxed);
        const auto& p = s_gui_state.synth->params[i];
        float norm = static_cast<float>((val - p.min_value) / (p.max_value - p.min_value));
        if (norm < 0.0f) norm = 0.0f;
        if (norm > 1.0f) norm = 1.0f;
        bool is_active = s_gui_state.drag_param == i;
        draw_knob(dpy, win, s_gui_state.gc, cx, cy, knob_r, norm, param_label(i), is_active);

        char val_str[32];
        std::snprintf(val_str, sizeof(val_str), "%.2f", val);
        XSetForeground(dpy, s_gui_state.gc, s_gui_state.text_color);
        XDrawString(dpy, win, s_gui_state.gc,
                    cx - 3 * static_cast<int>(std::strlen(val_str)) / 2,
                    cy + knob_r + 40,
                    val_str, static_cast<int>(std::strlen(val_str)));
    }

    const char* title = "Mamba Synth";
    XSetForeground(dpy, s_gui_state.gc, s_gui_state.text_color);
    XDrawString(dpy, win, s_gui_state.gc, 10, 20, title, static_cast<int>(std::strlen(title)));
    XFlush(dpy);
}
#endif // __linux__

// ---------------------------------------------------------------------------
// CLAP GUI extension callbacks
// ---------------------------------------------------------------------------

static bool gui_is_api_supported(const clap_plugin_t* plugin, const char* api,
                                  bool is_floating) {
    (void)plugin;
    if (is_floating) return false;
    return std::strcmp(api, CLAP_WINDOW_API_X11) == 0;
}

static bool gui_get_preferred_api(const clap_plugin_t* plugin,
                                   const char** api, bool* is_floating) {
    (void)plugin;
    *api = CLAP_WINDOW_API_X11;
    *is_floating = false;
    return true;
}

static bool gui_create(const clap_plugin_t* plugin, const char* api,
                        bool is_floating) {
    (void)plugin;
    (void)api;
    (void)is_floating;
#if defined(__linux__)
    if (!s_gui_initialized) {
        s_display = XOpenDisplay(nullptr);
        if (!s_display) return false;
        s_gui_initialized = true;
    }
    return true;
#else
    return false;
#endif
}

static void gui_destroy(const clap_plugin_t* plugin) {
    (void)plugin;
#if defined(__linux__)
    s_gui_state.window = 0;
    s_gui_state.synth = nullptr;
#endif
}

static bool gui_set_scale(const clap_plugin_t* plugin, double scale) {
    (void)plugin;
    (void)scale;
    return true;
}

static bool gui_get_size(const clap_plugin_t* plugin, uint32_t* w, uint32_t* h) {
    (void)plugin;
    *w = 400;
    *h = 250;
    return true;
}

static bool gui_can_resize(const clap_plugin_t* plugin) {
    (void)plugin;
    return true;
}

static bool gui_get_resize_hints(const clap_plugin_t* plugin,
                                  clap_gui_resize_hints_t* hints) {
    (void)plugin;
    hints->can_resize_horizontally = true;
    hints->can_resize_vertically = false;
    hints->preserve_aspect_ratio = false;
    hints->aspect_ratio_width = 0;
    hints->aspect_ratio_height = 0;
    return true;
}

static bool gui_adjust_size(const clap_plugin_t* plugin, uint32_t* w, uint32_t* h) {
    (void)plugin;
    if (*w < 300) *w = 300;
    if (*h < 200) *h = 200;
    return true;
}

static bool gui_set_size(const clap_plugin_t* plugin, uint32_t w, uint32_t h) {
    (void)plugin;
#if defined(__linux__)
    s_gui_state.width = static_cast<int>(w);
    s_gui_state.height = static_cast<int>(h);
    gui_redraw();
#endif
    return true;
}

static bool gui_set_parent(const clap_plugin_t* plugin,
                            const clap_window_t* window) {
    auto* m = static_cast<MambaSynth*>(plugin->plugin_data);
    if (!window) { gui_destroy(plugin); return false; }
#if defined(__linux__)
    if (std::strcmp(window->api, CLAP_WINDOW_API_X11) != 0) return false;
    s_gui_state.synth = m;
    s_gui_state.window = window->x11;
    s_gui_state.width = 400;
    s_gui_state.height = 250;
    s_gui_state.drag_param = -1;
    s_gui_state.init_colors();

    XWindowAttributes attr;
    XGetWindowAttributes(s_display, s_gui_state.window, &attr);
    s_gui_state.gc = XCreateGC(s_display, s_gui_state.window, 0, nullptr);

    XSelectInput(s_display, s_gui_state.window,
                 ExposureMask | ButtonPressMask | ButtonReleaseMask |
                 PointerMotionMask);

    XEvent ev;
    while (XCheckWindowEvent(s_display, s_gui_state.window, ExposureMask, &ev))
        gui_redraw();
    gui_redraw();
    return true;
#else
    (void)m;
    return false;
#endif
}

static bool gui_set_transient(const clap_plugin_t* plugin,
                               const clap_window_t* window) {
    (void)plugin;
    (void)window;
    return false; // non-floating only
}

static void gui_suggest_title(const clap_plugin_t* plugin, const char* title) {
    (void)plugin;
    (void)title;
}

static bool gui_show(const clap_plugin_t* plugin) {
    (void)plugin;
#if defined(__linux__)
    if (s_gui_state.window) {
        XMapRaised(s_display, s_gui_state.window);
        gui_redraw();
    }
    return true;
#else
    return false;
#endif
}

static bool gui_hide(const clap_plugin_t* plugin) {
    (void)plugin;
#if defined(__linux__)
    if (s_gui_state.window)
        XUnmapWindow(s_display, s_gui_state.window);
    return true;
#else
    return false;
#endif
}

const clap_plugin_gui_t s_gui = {
    gui_is_api_supported,
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
    gui_hide,
};

} // namespace mamba
