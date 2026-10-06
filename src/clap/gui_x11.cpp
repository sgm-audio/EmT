// Linux/X11 GUI backend: immediate-mode drawing plus event pumping behind
// the gui_backend_* seam. Split out of clap_gui.cpp (Phase 1 #4).
#include "clap/gui_backend.h"
#include "clap/clap_presets.h"
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <algorithm>
#include <cstdio>
#include <cstring>

namespace mamba {

static void draw(GuiInstance* g) {
    if (!g || !g->display || !g->window) return;
    XSetForeground(g->display, g->gc, 0x2D2D2D);
    XFillRectangle(g->display, g->window, g->gc, 0, 0, g->width, g->height);
    XSetForeground(g->display, g->gc, 0xFFFFFF);
    const char* title = "EmT Synth";
    XDrawString(g->display, g->window, g->gc, 10, 20, title, static_cast<int>(std::strlen(title)));
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

bool gui_backend_is_api_supported(const char* api) {
    return std::strcmp(api, CLAP_WINDOW_API_X11) == 0;
}

const char* gui_backend_preferred_api() {
    return CLAP_WINDOW_API_X11;
}

bool gui_backend_create(GuiInstance* g) {
    g->display = XOpenDisplay(nullptr);
    if (!g->display) return false;
    return true;
}

void gui_backend_destroy(GuiInstance* g) {
    if (g->gc && g->display) XFreeGC(g->display, g->gc);
    if (g->window && g->display) XDestroyWindow(g->display, g->window);
    if (g->display) XCloseDisplay(g->display);
}

void gui_backend_set_size(GuiInstance* g, uint32_t w, uint32_t h) {
    if (g->window) XResizeWindow(g->display, g->window, w, h);
    draw(g);
}

bool gui_backend_set_parent(GuiInstance* g, const clap_window_t* window) {
    if (std::strcmp(window->api, CLAP_WINDOW_API_X11) != 0) return false;
    g->window = XCreateSimpleWindow(g->display, window->x11, 0, 0,
                                    static_cast<unsigned>(g->width), static_cast<unsigned>(g->height),
                                    0, 0, 0x2D2D2D);
    if (!g->window) return false;
    g->gc = XCreateGC(g->display, g->window, 0, nullptr);
    XSelectInput(g->display, g->window, ExposureMask | ButtonPressMask | ButtonReleaseMask | PointerMotionMask);
    XMapWindow(g->display, g->window);
    draw(g);
    return true;
}

bool gui_backend_show(GuiInstance* g) {
    if (!g->window) return false;
    XMapRaised(g->display, g->window);
    draw(g);
    return true;
}

bool gui_backend_hide(GuiInstance* g) {
    if (!g->window) return false;
    XUnmapWindow(g->display, g->window);
    return true;
}

void gui_backend_on_timer(GuiInstance* g, clap_id id) {
    if (g->timer_id == id) {
        pump_events(g);
        draw(g);
    }}

} // namespace mamba
