// Pins the host-facing GUI dispatch in src/clap/gui_common.cpp, linked
// against the fallback gui_stub.cpp backend so the tests are deterministic
// on every platform (no display needed) and need no X11 libraries.
// What this covers: the null checks, default sizes, resize policy, and
// "unsupported backend" behavior that the #4 split moved out of the old
// #if-ladder in clap_gui.cpp. The X11/Win32 backends themselves moved
// verbatim and are covered by construction (see the split script's
// line-presence checks); they cannot run headless.

#include "clap/clap_gui.h"
#include "clap/clap_plugin.h"
#include "clap/gui_backend.h"

#include <gtest/gtest.h>

#include <clap/clap.h>

namespace mamba {
namespace {

clap_plugin_t MakeNullPlugin() {
    clap_plugin_t plugin{};
    plugin.plugin_data = nullptr;  // gui() resolves this to a null instance
    return plugin;
}

TEST(GuiDispatch, ApiSupportedRejectsBadInput) {
    clap_plugin_t plugin = MakeNullPlugin();
    EXPECT_FALSE(s_gui.is_api_supported(&plugin, nullptr, false));
    EXPECT_FALSE(s_gui.is_api_supported(&plugin, CLAP_WINDOW_API_X11, true));
    EXPECT_FALSE(s_gui.is_api_supported(&plugin, "bogus.api", false));
    // The stub backend supports nothing.
    EXPECT_FALSE(s_gui.is_api_supported(&plugin, CLAP_WINDOW_API_X11, false));
}

TEST(GuiDispatch, PreferredApi) {
    clap_plugin_t plugin = MakeNullPlugin();
    const char* api = nullptr;
    bool floating = true;
    EXPECT_FALSE(s_gui.get_preferred_api(&plugin, nullptr, &floating));
    EXPECT_FALSE(s_gui.get_preferred_api(&plugin, &api, nullptr));
    // Stub platform: no preferred API, outputs untouched.
    EXPECT_FALSE(s_gui.get_preferred_api(&plugin, &api, &floating));
    EXPECT_EQ(api, nullptr);
    EXPECT_TRUE(floating);
}

TEST(GuiDispatch, CreateFailsWithoutBackend) {
    clap_plugin_t plugin = MakeNullPlugin();
    EXPECT_FALSE(s_gui.create(&plugin, nullptr, false));
    EXPECT_FALSE(s_gui.create(&plugin, CLAP_WINDOW_API_X11, false));
    EXPECT_FALSE(s_gui.create(&plugin, CLAP_WINDOW_API_X11, true));
}

TEST(GuiDispatch, GetSizeDefaults) {
    clap_plugin_t plugin = MakeNullPlugin();
    uint32_t w = 0, h = 0;
    EXPECT_FALSE(s_gui.get_size(&plugin, nullptr, &h));
    EXPECT_FALSE(s_gui.get_size(&plugin, &w, nullptr));
    EXPECT_TRUE(s_gui.get_size(&plugin, &w, &h));
    EXPECT_EQ(w, 460u);
    EXPECT_EQ(h, 260u);
}

TEST(GuiDispatch, ResizePolicy) {
    clap_plugin_t plugin = MakeNullPlugin();
    EXPECT_TRUE(s_gui.can_resize(&plugin));

    EXPECT_FALSE(s_gui.get_resize_hints(&plugin, nullptr));
    clap_gui_resize_hints_t hints{};
    EXPECT_TRUE(s_gui.get_resize_hints(&plugin, &hints));
    EXPECT_TRUE(hints.can_resize_horizontally);
    EXPECT_TRUE(hints.can_resize_vertically);
    EXPECT_FALSE(hints.preserve_aspect_ratio);

    uint32_t w = 0, h = 0;
    EXPECT_FALSE(s_gui.adjust_size(&plugin, nullptr, &h));
    EXPECT_FALSE(s_gui.adjust_size(&plugin, &w, nullptr));
    w = 100;
    h = 100;
    EXPECT_TRUE(s_gui.adjust_size(&plugin, &w, &h));
    EXPECT_EQ(w, 320u);
    EXPECT_EQ(h, 220u);
    w = 800;
    h = 600;
    EXPECT_TRUE(s_gui.adjust_size(&plugin, &w, &h));
    EXPECT_EQ(w, 800u);
    EXPECT_EQ(h, 600u);
}

TEST(GuiDispatch, SetSizeNeedsInstance) {
    clap_plugin_t plugin = MakeNullPlugin();
    EXPECT_FALSE(s_gui.set_size(&plugin, 800, 600));
}

TEST(GuiDispatch, ParentShowHideNeedInstance) {
    clap_plugin_t plugin = MakeNullPlugin();
    clap_window_t window{};
    window.api = CLAP_WINDOW_API_X11;
    EXPECT_FALSE(s_gui.set_parent(&plugin, nullptr));
    EXPECT_FALSE(s_gui.set_parent(&plugin, &window));
    EXPECT_FALSE(s_gui.show(&plugin));
    EXPECT_FALSE(s_gui.hide(&plugin));
}

TEST(GuiDispatch, TransientAndTitle) {
    clap_plugin_t plugin = MakeNullPlugin();
    clap_window_t window{};
    EXPECT_FALSE(s_gui.set_transient(&plugin, &window));
    s_gui.suggest_title(&plugin, "EmT");  // no-op, must not crash
}

TEST(GuiDispatch, DestroyAndTimerAreNullSafe) {
    clap_plugin_t plugin = MakeNullPlugin();
    s_gui.destroy(&plugin);  // no-op, must not crash
    s_timer_support.on_timer(&plugin, 0);
    s_timer_support.on_timer(&plugin, CLAP_INVALID_ID);
}

TEST(GuiInstance, Defaults) {
    GuiInstance g;
    EXPECT_EQ(g.width, 460);
    EXPECT_EQ(g.height, 260);
    EXPECT_EQ(g.drag_param, -1);
    EXPECT_EQ(g.timer_id, CLAP_INVALID_ID);
    EXPECT_EQ(g.synth, nullptr);
}

TEST(GuiHelpers, SetParamNullSafe) {
    set_param(nullptr, 0, 1.0);  // early return, must not crash
}

}  // namespace
}  // namespace mamba
