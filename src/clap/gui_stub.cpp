// Fallback GUI backend for platforms without a native implementation
// (today: macOS, where a Cocoa backend goes). Reports "unsupported" exactly
// like the old #else branches did; each function is the insertion point for
// the future backend. Split out of clap_gui.cpp (Phase 1 #4).
#include "clap/gui_backend.h"

namespace mamba {

bool gui_backend_is_api_supported(const char* api) {
    (void)api;
    return false;  // Cocoa: accept CLAP_WINDOW_API_COCOA here
}

const char* gui_backend_preferred_api() {
    return nullptr;  // Cocoa: return CLAP_WINDOW_API_COCOA here
}

bool gui_backend_create(GuiInstance* g) {
    (void)g;
    return false;  // Cocoa: allocate the NSView here
}

void gui_backend_destroy(GuiInstance* g) {
    (void)g;  // Cocoa: tear down here
}

void gui_backend_set_size(GuiInstance* g, uint32_t w, uint32_t h) {
    (void)g;
    (void)w;
    (void)h;  // Cocoa: resize here
}

bool gui_backend_set_parent(GuiInstance* g, const clap_window_t* window) {
    (void)g;
    (void)window;
    return false;  // Cocoa: attach to parent NSView here
}

bool gui_backend_show(GuiInstance* g) {
    (void)g;
    return false;  // Cocoa: orderFront here
}

bool gui_backend_hide(GuiInstance* g) {
    (void)g;
    return false;  // Cocoa: orderOut here
}

void gui_backend_on_timer(GuiInstance* g, clap_id id) {
    (void)g;
    (void)id;  // Cocoa: pump/redraw here
}

}  // namespace mamba
