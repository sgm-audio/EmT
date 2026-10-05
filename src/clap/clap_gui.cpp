#include "clap/ext/gui.h"
#include "clap/ext/params.h"
#include "clap/ext/timer-support.h"
#include "clap/clap_plugin.h"
#include "clap/param_table.h"
#include "clap/clap_presets.h"
#include <clap/clap.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <new>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

#if defined(__linux__)
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#endif

namespace mamba {

struct GuiInstance {
    MambaSynth* synth{};
    int width{460};
    int height{260};
    int drag_param{-1};
    int drag_start_y{};
    double drag_start_value{};
    const clap_host_timer_support_t* host_timer{};
    clap_id timer_id{CLAP_INVALID_ID};

#if defined(__linux__)
    Display* display{};
    Window window{};
    GC gc{};
#elif defined(_WIN32)
    HWND hwnd{};
    UINT_PTR win32_timer_id{0};
    bool is_dragging{false};
    HFONT title_font{nullptr};
    HFONT ui_font{nullptr};
    HFONT small_font{nullptr};
#endif
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

static void default_preset_path(char* path, size_t size) {
    char dir[512]{};
    if (!get_preset_directory(dir, sizeof(dir))) { path[0] = '\0'; return; }
    std::snprintf(path, size, "%s/Default.mambapreset", dir);
}

#if defined(__linux__)
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
#endif

#if defined(_WIN32)
static const char* kWindowClassName = "EmTSynthGui";

struct SliderItem {
    clap_id id;
    const char* label;
};

// Slider labels come from the shared param table (same strings as get_info).
static const SliderItem kSliders[] = {
    { PARAM_GAIN,    kSynthParamTable[PARAM_GAIN].name },
    { PARAM_MIX,     kSynthParamTable[PARAM_MIX].name },
    { PARAM_BYPASS,  kSynthParamTable[PARAM_BYPASS].name },
    { PARAM_ATTACK,  kSynthParamTable[PARAM_ATTACK].name },
    { PARAM_RELEASE, kSynthParamTable[PARAM_RELEASE].name },
};
static constexpr int kNumSliders = static_cast<int>(sizeof(kSliders) / sizeof(kSliders[0]));

static void init_win32_fonts(GuiInstance* g) {
    if (!g->title_font) {
        g->title_font = CreateFontA(
            18, 0, 0, 0, FW_BOLD, FALSE, FALSE, FALSE,
            DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS,
            CLEARTYPE_QUALITY, DEFAULT_PITCH | FF_DONTCARE, "Segoe UI");
    }
    if (!g->ui_font) {
        g->ui_font = CreateFontA(
            14, 0, 0, 0, FW_NORMAL, FALSE, FALSE, FALSE,
            DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS,
            CLEARTYPE_QUALITY, DEFAULT_PITCH | FF_DONTCARE, "Segoe UI");
    }
    if (!g->small_font) {
        g->small_font = CreateFontA(
            12, 0, 0, 0, FW_SEMIBOLD, FALSE, FALSE, FALSE,
            DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS,
            CLEARTYPE_QUALITY, DEFAULT_PITCH | FF_DONTCARE, "Segoe UI");
    }
}

static void cleanup_win32_fonts(GuiInstance* g) {
    if (g->title_font) { DeleteObject(g->title_font); g->title_font = nullptr; }
    if (g->ui_font) { DeleteObject(g->ui_font); g->ui_font = nullptr; }
    if (g->small_font) { DeleteObject(g->small_font); g->small_font = nullptr; }
}

static int get_track_w(int client_w) {
    return std::max(client_w - 205, 40);
}

static int hit_test_slider(int client_w, int mx, int my, clap_id* out_id) {
    const int start_y = 52;
    const int row_h = 32;
    const int track_x = 85;
    const int track_w = get_track_w(client_w);

    for (int i = 0; i < kNumSliders; ++i) {
        int y_row = start_y + i * row_h;
        if (my >= y_row && my < y_row + row_h) {
            if (mx >= 16 && mx <= track_x + track_w + 12) {
                if (out_id) *out_id = kSliders[i].id;
                return i;
            }
        }
    }
    return -1;
}

static void win32_paint(HWND hwnd, GuiInstance* g) {
    PAINTSTRUCT ps;
    HDC hdc = BeginPaint(hwnd, &ps);
    if (!hdc) return;

    RECT cr;
    GetClientRect(hwnd, &cr);
    int w = cr.right - cr.left;
    int h = cr.bottom - cr.top;
    if (w <= 0 || h <= 0) {
        EndPaint(hwnd, &ps);
        return;
    }

    HDC memDC = CreateCompatibleDC(hdc);
    HBITMAP memBitmap = CreateCompatibleBitmap(hdc, w, h);
    HGDIOBJ oldBitmap = SelectObject(memDC, memBitmap);

    SetBkMode(memDC, TRANSPARENT);

    // 1. Dark background
    HBRUSH bg_brush = CreateSolidBrush(RGB(28, 30, 36));
    FillRect(memDC, &cr, bg_brush);
    DeleteObject(bg_brush);

    // Header bar
    RECT hdr_rect{ 0, 0, w, 44 };
    HBRUSH hdr_brush = CreateSolidBrush(RGB(20, 22, 26));
    FillRect(memDC, &hdr_rect, hdr_brush);
    DeleteObject(hdr_brush);

    HPEN div_pen = CreatePen(PS_SOLID, 1, RGB(44, 48, 56));
    HGDIOBJ old_pen = SelectObject(memDC, div_pen);
    MoveToEx(memDC, 0, 44, nullptr);
    LineTo(memDC, w, 44);

    // Title: 'Mamba-3 SSM Synth'
    SelectObject(memDC, g->title_font ? g->title_font : GetStockObject(DEFAULT_GUI_FONT));
    SetTextColor(memDC, RGB(245, 245, 250));
    RECT title_rect{ 16, 10, 240, 36 };
    DrawTextA(memDC, "EmT Synth", -1, &title_rect, DT_LEFT | DT_VCENTER | DT_SINGLELINE);

    // Active voice count and voice indicators
    int active_voices = 0;
    if (g->synth) {
        for (uint32_t v = 0; v < MAX_VOICES; ++v) {
            uint8_t stage = g->synth->voice_atomics[v].stage.load(std::memory_order_relaxed);
            if (stage == VOICE_ATTACK || stage == VOICE_SUSTAIN || stage == VOICE_RELEASE) {
                ++active_voices;
            }
        }
    }

    SelectObject(memDC, g->small_font ? g->small_font : GetStockObject(DEFAULT_GUI_FONT));
    SetTextColor(memDC, RGB(180, 190, 205));
    char voice_str[48];
    std::snprintf(voice_str, sizeof(voice_str), "Active Voices: %d / %u", active_voices, MAX_VOICES);
    RECT voice_rect{ w - 240, 12, w - 105, 32 };
    DrawTextA(memDC, voice_str, -1, &voice_rect, DT_RIGHT | DT_VCENTER | DT_SINGLELINE);

    int leds_x = w - 96;
    for (uint32_t v = 0; v < MAX_VOICES; ++v) {
        bool v_active = false;
        if (g->synth) {
            uint8_t st = g->synth->voice_atomics[v].stage.load(std::memory_order_relaxed);
            v_active = (st == VOICE_ATTACK || st == VOICE_SUSTAIN || st == VOICE_RELEASE);
        }
        RECT led_rect{ leds_x + static_cast<int>(v) * 11, 17, leds_x + static_cast<int>(v) * 11 + 8, 25 };
        HBRUSH led_brush = CreateSolidBrush(v_active ? RGB(40, 220, 140) : RGB(50, 54, 64));
        FillRect(memDC, &led_rect, led_brush);
        DeleteObject(led_brush);
    }

    // 2. Visual sliders for Gain, Mix, Bypass state (and Attack, Release)
    const int start_y = 52;
    const int row_h = 32;
    const int track_x = 85;
    const int track_w = get_track_w(w);

    for (int i = 0; i < kNumSliders; ++i) {
        clap_id id = kSliders[i].id;
        int y_row = start_y + i * row_h;
        int track_y = y_row + 9;
        int track_h = 10;

        const ParamSmooth& p = g->synth->params[id];
        double val = g->synth->param_values[id].load(std::memory_order_relaxed);
        double norm = (p.max_value > p.min_value)
            ? std::clamp((val - p.min_value) / (p.max_value - p.min_value), 0.0, 1.0)
            : 0.0;
        if (id == PARAM_BYPASS) {
            norm = (val > 0.5) ? 1.0 : 0.0;
        }

        // Label
        SelectObject(memDC, g->ui_font ? g->ui_font : GetStockObject(DEFAULT_GUI_FONT));
        SetTextColor(memDC, RGB(210, 215, 225));
        RECT lbl_rect{ 18, y_row + 4, 78, y_row + 24 };
        DrawTextA(memDC, kSliders[i].label, -1, &lbl_rect, DT_LEFT | DT_VCENTER | DT_SINGLELINE);

        // Track background
        RECT tr_rect{ track_x, track_y, track_x + track_w, track_y + track_h };
        HBRUSH tr_brush = CreateSolidBrush(RGB(40, 44, 52));
        FillRect(memDC, &tr_rect, tr_brush);
        DeleteObject(tr_brush);

        // Track border
        HPEN tr_pen = CreatePen(PS_SOLID, 1, RGB(60, 66, 78));
        SelectObject(memDC, tr_pen);
        SelectObject(memDC, GetStockObject(NULL_BRUSH));
        Rectangle(memDC, track_x, track_y, track_x + track_w, track_y + track_h);
        SelectObject(memDC, old_pen);
        DeleteObject(tr_pen);

        // Filled track
        if (norm > 0.0) {
            int fill_w = static_cast<int>(norm * track_w);
            RECT fill_rect{ track_x, track_y, track_x + fill_w, track_y + track_h };
            COLORREF fill_color = RGB(65, 150, 240);
            if (id == PARAM_BYPASS) {
                fill_color = (norm > 0.5) ? RGB(235, 75, 65) : RGB(50, 54, 64);
            } else if (id == PARAM_MIX) {
                fill_color = RGB(80, 175, 235);
            }
            HBRUSH fill_brush = CreateSolidBrush(fill_color);
            FillRect(memDC, &fill_rect, fill_brush);
            DeleteObject(fill_brush);
        }

        // Thumb handle
        int thumb_cx = track_x + static_cast<int>(norm * track_w);
        RECT thumb_rect{ thumb_cx - 4, track_y - 3, thumb_cx + 5, track_y + track_h + 3 };
        HBRUSH th_brush = CreateSolidBrush(RGB(230, 235, 245));
        FillRect(memDC, &thumb_rect, th_brush);
        DeleteObject(th_brush);

        HPEN th_pen = CreatePen(PS_SOLID, 1, RGB(140, 150, 165));
        SelectObject(memDC, th_pen);
        SelectObject(memDC, GetStockObject(NULL_BRUSH));
        Rectangle(memDC, thumb_rect.left, thumb_rect.top, thumb_rect.right, thumb_rect.bottom);
        SelectObject(memDC, old_pen);
        DeleteObject(th_pen);

        // Value text
        char val_buf[32]{};
        COLORREF val_color = RGB(150, 200, 255);
        if (id == PARAM_BYPASS) {
            std::snprintf(val_buf, sizeof(val_buf), "%s", (norm > 0.5) ? "BYPASS" : "ACTIVE");
            val_color = (norm > 0.5) ? RGB(255, 100, 90) : RGB(90, 220, 140);
        } else if (id == PARAM_GAIN) {
            double db = 20.0 * std::log10(val > 0.0 ? val : 1e-9);
            std::snprintf(val_buf, sizeof(val_buf), "%.1f dB", db);
        } else if (id == PARAM_MIX) {
            std::snprintf(val_buf, sizeof(val_buf), "%.0f%%", val * 100.0);
        } else if (id == PARAM_ATTACK || id == PARAM_RELEASE) {
            std::snprintf(val_buf, sizeof(val_buf), "%.0f ms", val * 1000.0);
        } else {
            std::snprintf(val_buf, sizeof(val_buf), "%.2f", val);
        }

        SelectObject(memDC, g->small_font ? g->small_font : GetStockObject(DEFAULT_GUI_FONT));
        SetTextColor(memDC, val_color);
        RECT val_rect{ track_x + track_w + 12, y_row + 4, w - 8, y_row + 24 };
        DrawTextA(memDC, val_buf, -1, &val_rect, DT_LEFT | DT_VCENTER | DT_SINGLELINE);
    }

    // 3. Footer area (Presets)
    int footer_y = h - 42;
    if (footer_y > start_y + kNumSliders * row_h) {
        SelectObject(memDC, div_pen);
        MoveToEx(memDC, 0, footer_y - 6, nullptr);
        LineTo(memDC, w, footer_y - 6);

        RECT btn_save{ 18, footer_y, 118, footer_y + 24 };
        HBRUSH btn_brush = CreateSolidBrush(RGB(42, 45, 54));
        FillRect(memDC, &btn_save, btn_brush);

        HPEN btn_pen = CreatePen(PS_SOLID, 1, RGB(70, 75, 88));
        SelectObject(memDC, btn_pen);
        SelectObject(memDC, GetStockObject(NULL_BRUSH));
        Rectangle(memDC, btn_save.left, btn_save.top, btn_save.right, btn_save.bottom);

        SelectObject(memDC, g->small_font ? g->small_font : GetStockObject(DEFAULT_GUI_FONT));
        SetTextColor(memDC, RGB(210, 215, 225));
        DrawTextA(memDC, "Save Preset", -1, &btn_save, DT_CENTER | DT_VCENTER | DT_SINGLELINE);

        RECT btn_load{ 128, footer_y, 228, footer_y + 24 };
        FillRect(memDC, &btn_load, btn_brush);
        Rectangle(memDC, btn_load.left, btn_load.top, btn_load.right, btn_load.bottom);
        DrawTextA(memDC, "Load Preset", -1, &btn_load, DT_CENTER | DT_VCENTER | DT_SINGLELINE);

        DeleteObject(btn_brush);
        SelectObject(memDC, old_pen);
        DeleteObject(btn_pen);

        RECT info_rect{ 240, footer_y, w - 16, footer_y + 24 };
        SetTextColor(memDC, RGB(100, 105, 120));
        DrawTextA(memDC, "Mamba-3 SSM | 0-sample latency", -1, &info_rect, DT_RIGHT | DT_VCENTER | DT_SINGLELINE);
    }

    SelectObject(memDC, old_pen);
    DeleteObject(div_pen);

    BitBlt(hdc, 0, 0, w, h, memDC, 0, 0, SRCCOPY);

    SelectObject(memDC, oldBitmap);
    DeleteObject(memBitmap);
    DeleteDC(memDC);

    EndPaint(hwnd, &ps);
}

static LRESULT CALLBACK WndProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {
    GuiInstance* g = reinterpret_cast<GuiInstance*>(GetWindowLongPtrA(hwnd, GWLP_USERDATA));

    if (uMsg == WM_NCCREATE) {
        auto* cs = reinterpret_cast<CREATESTRUCTA*>(lParam);
        g = reinterpret_cast<GuiInstance*>(cs->lpCreateParams);
        SetWindowLongPtrA(hwnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(g));
        if (g) {
            g->hwnd = hwnd;
            init_win32_fonts(g);
        }
        return DefWindowProcA(hwnd, uMsg, wParam, lParam);
    }

    if (!g) {
        return DefWindowProcA(hwnd, uMsg, wParam, lParam);
    }

    switch (uMsg) {
    case WM_PAINT: {
        win32_paint(hwnd, g);
        return 0;
    }

    case WM_ERASEBKGND:
        return 1;

    case WM_TIMER: {
        InvalidateRect(hwnd, nullptr, FALSE);
        return 0;
    }

    case WM_LBUTTONDOWN: {
        int mx = static_cast<int>(static_cast<short>(LOWORD(lParam)));
        int my = static_cast<int>(static_cast<short>(HIWORD(lParam)));

        RECT cr;
        GetClientRect(hwnd, &cr);
        int w = cr.right - cr.left;
        int h = cr.bottom - cr.top;

        int footer_y = h - 42;
        if (my >= footer_y && my <= footer_y + 24) {
            char path[768]{};
            default_preset_path(path, sizeof(path));
            if (path[0]) {
                if (mx >= 18 && mx <= 118) {
                    save_preset(g->synth, path, "Default");
                    InvalidateRect(hwnd, nullptr, FALSE);
                    return 0;
                } else if (mx >= 128 && mx <= 228) {
                    load_preset(g->synth, path);
                    InvalidateRect(hwnd, nullptr, FALSE);
                    return 0;
                }
            }
        }

        clap_id param_id = CLAP_INVALID_ID;
        int slider_idx = hit_test_slider(w, mx, my, &param_id);
        if (slider_idx >= 0 && param_id < PARAM_COUNT) {
            g->drag_param = static_cast<int>(param_id);
            g->is_dragging = true;
            SetCapture(hwnd);

            const int track_x = 85;
            const int track_w = get_track_w(w);
            const ParamSmooth& p = g->synth->params[param_id];

            if (param_id == PARAM_BYPASS) {
                double cur = g->synth->param_values[PARAM_BYPASS].load(std::memory_order_relaxed);
                set_param(g, PARAM_BYPASS, (cur > 0.5) ? 0.0 : 1.0);
            } else {
                double norm = std::clamp(static_cast<double>(mx - track_x) / track_w, 0.0, 1.0);
                double val = p.min_value + norm * (p.max_value - p.min_value);
                set_param(g, param_id, val);
            }
            InvalidateRect(hwnd, nullptr, FALSE);
            return 0;
        }
        return 0;
    }

    case WM_MOUSEMOVE: {
        if (g->is_dragging && g->drag_param >= 0 && g->drag_param < static_cast<int>(PARAM_COUNT)) {
            int mx = static_cast<int>(static_cast<short>(LOWORD(lParam)));
            RECT cr;
            GetClientRect(hwnd, &cr);
            int w = cr.right - cr.left;
            const int track_x = 85;
            const int track_w = get_track_w(w);
            const clap_id param_id = static_cast<clap_id>(g->drag_param);
            const ParamSmooth& p = g->synth->params[param_id];

            if (param_id == PARAM_BYPASS) {
                double norm = static_cast<double>(mx - track_x) / track_w;
                set_param(g, param_id, (norm >= 0.5) ? 1.0 : 0.0);
            } else {
                double norm = std::clamp(static_cast<double>(mx - track_x) / track_w, 0.0, 1.0);
                double val = p.min_value + norm * (p.max_value - p.min_value);
                set_param(g, param_id, val);
            }
            InvalidateRect(hwnd, nullptr, FALSE);
            return 0;
        }
        return 0;
    }

    case WM_LBUTTONUP: {
        if (g->is_dragging) {
            g->is_dragging = false;
            g->drag_param = -1;
            ReleaseCapture();
            InvalidateRect(hwnd, nullptr, FALSE);
            return 0;
        }
        return 0;
    }

    case WM_DESTROY: {
        if (g->win32_timer_id) {
            KillTimer(hwnd, g->win32_timer_id);
            g->win32_timer_id = 0;
        }
        cleanup_win32_fonts(g);
        g->hwnd = nullptr;
        return 0;
    }

    default:
        break;
    }

    return DefWindowProcA(hwnd, uMsg, wParam, lParam);
}

static bool register_window_class() {
    static bool s_registered = false;
    if (s_registered) return true;

    WNDCLASSEXA wc{};
    wc.cbSize = sizeof(WNDCLASSEXA);
    wc.style = CS_HREDRAW | CS_VREDRAW | CS_DBLCLKS;
    wc.lpfnWndProc = WndProc;
    wc.cbClsExtra = 0;
    wc.cbWndExtra = 0;
    wc.hInstance = GetModuleHandleA(nullptr);
    wc.hIcon = nullptr;
    wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
    wc.hbrBackground = nullptr;
    wc.lpszMenuName = nullptr;
    wc.lpszClassName = kWindowClassName;
    wc.hIconSm = nullptr;

    if (!RegisterClassExA(&wc)) {
        DWORD err = GetLastError();
        if (err != ERROR_CLASS_ALREADY_EXISTS) {
            return false;
        }
    }
    s_registered = true;
    return true;
}
#endif

static bool clap_gui_is_api_supported(const clap_plugin_t* plugin, const char* api, bool floating) {
    (void)plugin;
    if (!api || floating) return false;
#if defined(__linux__)
    return std::strcmp(api, CLAP_WINDOW_API_X11) == 0;
#elif defined(_WIN32)
    return std::strcmp(api, CLAP_WINDOW_API_WIN32) == 0;
#else
    return false;
#endif
}

static bool gui_get_preferred_api(const clap_plugin_t* plugin, const char** api, bool* floating) {
    (void)plugin;
    if (!api || !floating) return false;
#if defined(__linux__)
    *api = CLAP_WINDOW_API_X11;
    *floating = false;
    return true;
#elif defined(_WIN32)
    *api = CLAP_WINDOW_API_WIN32;
    *floating = false;
    return true;
#else
    return false;
#endif
}

static bool gui_create(const clap_plugin_t* plugin, const char* api, bool floating) {
    if (!clap_gui_is_api_supported(plugin, api, floating)) return false;
    auto* synth = static_cast<MambaSynth*>(plugin->plugin_data);
    if (!synth || synth->gui_data) return false;

    auto* g = new (std::nothrow) GuiInstance();
    if (!g) return false;
    g->synth = synth;

#if defined(__linux__)
    g->display = XOpenDisplay(nullptr);
    if (!g->display) { delete g; return false; }
#elif defined(_WIN32)
    if (!register_window_class()) {
        delete g;
        return false;
    }
#endif

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

#if defined(__linux__)
    if (g->gc && g->display) XFreeGC(g->display, g->gc);
    if (g->window && g->display) XDestroyWindow(g->display, g->window);
    if (g->display) XCloseDisplay(g->display);
#elif defined(_WIN32)
    if (g->hwnd) {
        if (g->win32_timer_id) {
            KillTimer(g->hwnd, g->win32_timer_id);
            g->win32_timer_id = 0;
        }
        DestroyWindow(g->hwnd);
        g->hwnd = nullptr;
    }
    cleanup_win32_fonts(g);
#endif

    g->synth->gui_data = nullptr;
    delete g;
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

static bool gui_set_size(const clap_plugin_t* p, uint32_t w, uint32_t h) {
    GuiInstance* g = gui(p);
    if (!g) return false;
    g->width = static_cast<int>(w);
    g->height = static_cast<int>(h);
#if defined(__linux__)
    if (g->window) XResizeWindow(g->display, g->window, w, h);
    draw(g);
#elif defined(_WIN32)
    if (g->hwnd) {
        SetWindowPos(g->hwnd, nullptr, 0, 0, g->width, g->height,
                     SWP_NOMOVE | SWP_NOZORDER | SWP_NOACTIVATE);
        InvalidateRect(g->hwnd, nullptr, FALSE);
    }
#endif
    return true;
}

static bool gui_set_parent(const clap_plugin_t* p, const clap_window_t* window) {
    GuiInstance* g = gui(p);
    if (!g || !window || !window->api) return false;

#if defined(__linux__)
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
#elif defined(_WIN32)
    if (std::strcmp(window->api, CLAP_WINDOW_API_WIN32) != 0) return false;
    HWND parent_hwnd = static_cast<HWND>(window->win32);
    if (!parent_hwnd || !IsWindow(parent_hwnd)) return false;

    if (g->hwnd) {
        SetParent(g->hwnd, parent_hwnd);
        SetWindowPos(g->hwnd, nullptr, 0, 0, g->width, g->height,
                     SWP_NOZORDER | SWP_NOACTIVATE | SWP_SHOWWINDOW);
        return true;
    }

    HMODULE hInst = nullptr;
    GetModuleHandleExA(
        GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
        reinterpret_cast<LPCSTR>(&clap_gui_is_api_supported),
        &hInst);
    if (!hInst) hInst = GetModuleHandleA(nullptr);

    g->hwnd = CreateWindowExA(
        0,
        kWindowClassName,
        "Mamba-3 SSM Synth",
        WS_CHILD | WS_VISIBLE | WS_CLIPSIBLINGS,
        0, 0,
        g->width, g->height,
        parent_hwnd,
        nullptr,
        hInst,
        g
    );

    if (!g->hwnd) return false;

    if (g->timer_id == CLAP_INVALID_ID && !g->win32_timer_id) {
        g->win32_timer_id = SetTimer(g->hwnd, 1, 33, nullptr);
    }

    InvalidateRect(g->hwnd, nullptr, FALSE);
    UpdateWindow(g->hwnd);
    return true;
#else
    return false;
#endif
}

static bool gui_set_transient(const clap_plugin_t*, const clap_window_t*) { return false; }

static void gui_suggest_title(const clap_plugin_t*, const char*) {}

static bool gui_show(const clap_plugin_t* p) {
    GuiInstance* g = gui(p);
    if (!g) return false;
#if defined(__linux__)
    if (!g->window) return false;
    XMapRaised(g->display, g->window);
    draw(g);
    return true;
#elif defined(_WIN32)
    if (!g->hwnd) return false;
    ShowWindow(g->hwnd, SW_SHOW);
    if (g->timer_id == CLAP_INVALID_ID && !g->win32_timer_id) {
        g->win32_timer_id = SetTimer(g->hwnd, 1, 33, nullptr);
    }
    InvalidateRect(g->hwnd, nullptr, FALSE);
    UpdateWindow(g->hwnd);
    return true;
#else
    return false;
#endif
}

static bool gui_hide(const clap_plugin_t* p) {
    GuiInstance* g = gui(p);
    if (!g) return false;
#if defined(__linux__)
    if (!g->window) return false;
    XUnmapWindow(g->display, g->window);
    return true;
#elif defined(_WIN32)
    if (!g->hwnd) return false;
    ShowWindow(g->hwnd, SW_HIDE);
    if (g->win32_timer_id) {
        KillTimer(g->hwnd, g->win32_timer_id);
        g->win32_timer_id = 0;
    }
    return true;
#else
    return false;
#endif
}

static void gui_on_timer(const clap_plugin_t* p, clap_id id) {
    GuiInstance* g = gui(p);
    if (!g) return;
#if defined(__linux__)
    if (g->timer_id == id) {
        pump_events(g);
        draw(g);
    }
#elif defined(_WIN32)
    if (g->timer_id == id && g->hwnd) {
        InvalidateRect(g->hwnd, nullptr, FALSE);
    }
#else
    (void)id;
#endif
}

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