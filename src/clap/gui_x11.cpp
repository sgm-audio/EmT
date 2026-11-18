// Linux/X11 GUI backend: immediate-mode drawing plus event pumping behind
// the gui_backend_* seam. Split out of clap_gui.cpp (Phase 1 #4).
#include "clap/gui_backend.h"
#include "clap/clap_presets.h"
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace mamba {

static unsigned long alloc_x11_color(Display* display, int screen, uint8_t r, uint8_t g, uint8_t b,
                                     unsigned long fallback) {
    Colormap cmap = DefaultColormap(display, screen);
    XColor xc{};
    xc.red = static_cast<unsigned short>(r) * 257u;
    xc.green = static_cast<unsigned short>(g) * 257u;
    xc.blue = static_cast<unsigned short>(b) * 257u;
    xc.flags = DoRed | DoGreen | DoBlue;
    if (XAllocColor(display, cmap, &xc)) {
        return xc.pixel;
    }
    return fallback;
}

static void ensure_x11_colors(GuiInstance* g) {
    if (!g || !g->display || g->colors_init) return;
    const int screen = DefaultScreen(g->display);
    g->color_bg = alloc_x11_color(g->display, screen, 0x2D, 0x2D, 0x2D, BlackPixel(g->display, screen));
    g->color_fg = alloc_x11_color(g->display, screen, 0xFF, 0xFF, 0xFF, WhitePixel(g->display, screen));
    g->color_bar = alloc_x11_color(g->display, screen, 0x4A, 0x90, 0xD9, WhitePixel(g->display, screen));
    g->colors_init = true;
}

static void draw(GuiInstance* g) {
    if (!g || !g->synth || !g->display || !g->window || !g->gc) return;
    ensure_x11_colors(g);
    XSetForeground(g->display, g->gc, g->color_bg);
    XFillRectangle(g->display, g->window, g->gc, 0, 0,
                   static_cast<unsigned>(std::max(g->width, 1)),
                   static_cast<unsigned>(std::max(g->height, 1)));
    XSetForeground(g->display, g->gc, g->color_fg);
    const char* title = "EmT Synth";
    XDrawString(g->display, g->window, g->gc, 10, 20, title, static_cast<int>(std::strlen(title)));
    const int spacing = g->width / (PARAM_COUNT + 1);
    for (uint32_t i = 0; i < PARAM_COUNT; ++i) {
        const ParamSmooth& p = g->synth->params[i];
        const double value = g->synth->param_values[i].load(std::memory_order_relaxed);
        g->last_params[i] = value;
        const double span = p.max_value - p.min_value;
        const double norm = span > 0.0 ? std::clamp((value - p.min_value) / span, 0.0, 1.0) : 0.0;
        const int x = spacing * (static_cast<int>(i) + 1);
        XDrawArc(g->display, g->window, g->gc, x - 25, 65, 50, 50, 0, 360 * 64);
        XSetForeground(g->display, g->gc, g->color_bar);
        XFillRectangle(g->display, g->window, g->gc, x - 20, 120 - static_cast<int>(norm * 45),
                       40, static_cast<unsigned>(norm * 45));
        XSetForeground(g->display, g->gc, g->color_fg);
        char label[24];
        std::snprintf(label, sizeof(label), "%u: %.2f", i + 1, value);
        XDrawString(g->display, g->window, g->gc, x - 25, 145, label, static_cast<int>(std::strlen(label)));
    }
    for (uint32_t v = 0; v < MAX_VOICES; ++v) {
        g->last_voice_stages[v] = g->synth->voice_atomics[v].stage.load(std::memory_order_relaxed);
    }
    XDrawRectangle(g->display, g->window, g->gc, 20, g->height - 45, 100, 28);
    XDrawString(g->display, g->window, g->gc, 42, g->height - 26, "Save Preset", 11);
    XDrawRectangle(g->display, g->window, g->gc, 135, g->height - 45, 100, 28);
    XDrawString(g->display, g->window, g->gc, 157, g->height - 26, "Load Preset", 11);
    XFlush(g->display);
    g->dirty = false;
}

static void pump_events(GuiInstance* g) {
    if (!g || !g->synth || !g->display || !g->window) return;
    XEvent event;
    while (XCheckWindowEvent(g->display, g->window,
           ExposureMask | ButtonPressMask | ButtonReleaseMask | PointerMotionMask, &event)) {
        if (event.type == Expose) {
            g->dirty = true;
        } else if (event.type == ButtonPress) {
            if (event.xbutton.y >= g->height - 45) {
                char path[768]{}; default_preset_path(path, sizeof(path));
                if (path[0] && event.xbutton.x >= 20 && event.xbutton.x <= 120) {
                    save_preset(g->synth, path, "Default");
                    g->dirty = true;
                } else if (path[0] && event.xbutton.x >= 135 && event.xbutton.x <= 235) {
                    load_preset(g->synth, path);
                    g->dirty = true;
                }
            } else {
                const int spacing = g->width / (PARAM_COUNT + 1);
                const int id = std::clamp(event.xbutton.x / std::max(spacing, 1) - 1, 0, static_cast<int>(PARAM_COUNT) - 1);
                g->drag_param = id;
                g->drag_start_y = event.xbutton.y;
                g->drag_start_value = g->synth->param_values[id].load(std::memory_order_relaxed);
            }
        } else if (event.type == MotionNotify && g->drag_param >= 0 && g->drag_param < static_cast<int>(PARAM_COUNT)) {
            const ParamSmooth& p = g->synth->params[g->drag_param];
            const double delta = (g->drag_start_y - event.xmotion.y) / 150.0;
            set_param(g, static_cast<clap_id>(g->drag_param),
                      g->drag_start_value + delta * (p.max_value - p.min_value));
            g->dirty = true;
        } else if (event.type == ButtonRelease) {
            g->drag_param = -1;
        }
    }
}

bool gui_backend_is_api_supported(const char* api) {
    return api && std::strcmp(api, CLAP_WINDOW_API_X11) == 0;
}

const char* gui_backend_preferred_api() {
    return CLAP_WINDOW_API_X11;
}

bool gui_backend_create(GuiInstance* g) {
    if (!g) return false;
    g->display = XOpenDisplay(nullptr);
    if (!g->display) return false;
    ensure_x11_colors(g);
    g->dirty = true;
    return true;
}

void gui_backend_destroy(GuiInstance* g) {
    if (!g) return;
    if (g->gc && g->display) XFreeGC(g->display, g->gc);
    if (g->window && g->display) XDestroyWindow(g->display, g->window);
    if (g->display) XCloseDisplay(g->display);
    g->gc = nullptr;
    g->window = 0;
    g->display = nullptr;
    g->colors_init = false;
}

void gui_backend_set_size(GuiInstance* g, uint32_t w, uint32_t h) {
    if (!g || !g->display) return;
    if (g->window) XResizeWindow(g->display, g->window, std::max(w, 1u), std::max(h, 1u));
    draw(g);
}

bool gui_backend_set_parent(GuiInstance* g, const clap_window_t* window) {
    if (!g || !g->display || !window || !window->api) return false;
    if (std::strcmp(window->api, CLAP_WINDOW_API_X11) != 0) return false;
    ensure_x11_colors(g);
    g->window = XCreateSimpleWindow(g->display, window->x11, 0, 0,
                                    static_cast<unsigned>(std::max(g->width, 1)),
                                    static_cast<unsigned>(std::max(g->height, 1)),
                                    0, g->color_fg, g->color_bg);
    if (!g->window) return false;
    g->gc = XCreateGC(g->display, g->window, 0, nullptr);
    if (!g->gc) {
        XDestroyWindow(g->display, g->window);
        g->window = 0;
        return false;
    }
    XSelectInput(g->display, g->window, ExposureMask | ButtonPressMask | ButtonReleaseMask | PointerMotionMask);
    XMapWindow(g->display, g->window);
    draw(g);
    return true;
}

bool gui_backend_show(GuiInstance* g) {
    if (!g || !g->display || !g->window) return false;
    XMapRaised(g->display, g->window);
    draw(g);
    return true;
}

bool gui_backend_hide(GuiInstance* g) {
    if (!g || !g->display || !g->window) return false;
    XUnmapWindow(g->display, g->window);
    return true;
}

void gui_backend_on_timer(GuiInstance* g, clap_id id) {
    if (!g || !g->synth || g->timer_id != id) return;
    pump_events(g);
    if (!g->dirty) {
        for (uint32_t i = 0; i < PARAM_COUNT; ++i) {
            const double val = g->synth->param_values[i].load(std::memory_order_relaxed);
            if (std::fabs(val - g->last_params[i]) > 1e-12) {
                g->dirty = true;
                break;
            }
        }
    }
    if (!g->dirty) {
        for (uint32_t v = 0; v < MAX_VOICES; ++v) {
            if (g->synth->voice_atomics[v].stage.load(std::memory_order_relaxed) != g->last_voice_stages[v]) {
                g->dirty = true;
                break;
            }
        }
    }
    if (g->dirty) {
        draw(g);
    }
}

} // namespace mamba
