#ifndef EMT_GUI_BACKEND_H
#define EMT_GUI_BACKEND_H

// Internal GUI seam (Phase 1 #4). gui_common.cpp owns the CLAP dispatch
// (s_gui / s_timer_support) plus the helpers shared by every backend;
// exactly one backend TU implements gui_backend_* (selected by CMake):
//   gui_x11.cpp    Linux/X11   (CLAP_WINDOW_API_X11)
//   gui_win32.cpp  Windows     (CLAP_WINDOW_API_WIN32)
//   gui_stub.cpp   anything else (macOS/Cocoa goes here; reports unsupported)
// All seam entry points assume a non-null instance; api/window arguments are
// non-null where taken (gui_common.cpp checks first).

#include "clap/clap_plugin.h"

#include <clap/clap.h>

#include <cstddef>
#include <cstdint>

// Forward declarations for the platform handle types below, so including
// this header never drags in <windows.h> / <X11/Xlib.h> (the backend TUs
// include the real headers; these match their stable ABI typedefs).
#if defined(_WIN32)
struct HWND__;
using HWND = HWND__*;
struct HFONT__;
using HFONT = HFONT__*;
#elif defined(__linux__)
struct _XDisplay;
using Display = _XDisplay;  // NB: Display is the struct; APIs take Display*
using XID = unsigned long;
using Window = XID;
struct _XGC;
using GC = _XGC*;
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
    uintptr_t win32_timer_id{0};  // UINT_PTR without needing <windows.h>
    bool is_dragging{false};
    HFONT title_font{nullptr};
    HFONT ui_font{nullptr};
    HFONT small_font{nullptr};
#endif
};

// Shared helpers (gui_common.cpp), used by the backends to apply drag input
// and to locate the Default preset file.
void set_param(GuiInstance* g, clap_id id, double value);
void default_preset_path(char* path, size_t size);

// Backend seam (one implementation compiled per platform).
bool gui_backend_is_api_supported(const char* api);
const char* gui_backend_preferred_api();  // nullptr when the platform has no GUI
bool gui_backend_create(GuiInstance* g);
void gui_backend_destroy(GuiInstance* g);
void gui_backend_set_size(GuiInstance* g, uint32_t w, uint32_t h);
bool gui_backend_set_parent(GuiInstance* g, const clap_window_t* window);
bool gui_backend_show(GuiInstance* g);
bool gui_backend_hide(GuiInstance* g);
void gui_backend_on_timer(GuiInstance* g, clap_id id);

}  // namespace mamba

#endif  // EMT_GUI_BACKEND_H
