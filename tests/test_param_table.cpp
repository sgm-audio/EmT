// Pins the data-driven parameter tables in src/clap/param_table.h/.cpp
// against PARAMETER_REFERENCE.md, plus the value<->text conversions that
// moved there verbatim from clap_params.cpp / clap_fx_plugin.cpp.
//
// Several pins document as-shipped quirks (marked QUIRK with the
// REFACTORING_PLAN.md observation id): they lock in current behavior so any
// future cleanup updates the test deliberately.

#include "clap/param_table.h"

#include <gtest/gtest.h>

#include <clap/clap.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>

namespace mamba {
namespace {

TEST(ParamTable, SynthRowCount) {
    EXPECT_EQ(PARAM_COUNT, 5u);
    EXPECT_EQ(sizeof(kSynthParamTable) / sizeof(kSynthParamTable[0]),
              static_cast<size_t>(PARAM_COUNT));
    EXPECT_EQ(kFxParamTableCount, 3u);
}

TEST(ParamTable, SynthRowsMatchReference) {
    struct ExpectRow {
        const char* name;
        const char* module;
        double min_v;
        double max_v;
        double def;
        ParamCurve curve;
        float smooth_s;
        uint32_t flags;
        ParamFormat format;
    };
    const ExpectRow kExpect[PARAM_COUNT] = {
        {"Gain", "SSM", 0.0, 2.0, 1.0, PARAM_CURVE_LOG, 0.010f,
         CLAP_PARAM_IS_AUTOMATABLE, ParamFormat::GainDb},
        {"Attack", "Envelope", 0.001, 1.0, 0.1, PARAM_CURVE_LOG, 0.020f,
         CLAP_PARAM_IS_AUTOMATABLE, ParamFormat::Milliseconds},
        {"Release", "Envelope", 0.001, 2.0, 0.3, PARAM_CURVE_LOG, 0.050f,
         CLAP_PARAM_IS_AUTOMATABLE, ParamFormat::Milliseconds},
        {"Mix", "SSM", 0.0, 1.0, 1.0, PARAM_CURVE_LINEAR, 0.010f,
         CLAP_PARAM_IS_AUTOMATABLE, ParamFormat::Percent},
        {"Bypass", "", 0.0, 1.0, 0.0, PARAM_CURVE_STEPPED, 0.0f,
         CLAP_PARAM_IS_AUTOMATABLE | CLAP_PARAM_IS_STEPPED | CLAP_PARAM_IS_BYPASS,
         ParamFormat::Bypass},
    };
    for (uint32_t i = 0; i < PARAM_COUNT; ++i) {
        const ParamDesc& d = kSynthParamTable[i];
        EXPECT_EQ(d.id, i) << "row " << i;
        EXPECT_STREQ(d.name, kExpect[i].name) << "row " << i;
        EXPECT_STREQ(d.module, kExpect[i].module) << "row " << i;
        EXPECT_DOUBLE_EQ(d.min_value, kExpect[i].min_v) << "row " << i;
        EXPECT_DOUBLE_EQ(d.max_value, kExpect[i].max_v) << "row " << i;
        EXPECT_DOUBLE_EQ(d.default_value, kExpect[i].def) << "row " << i;
        EXPECT_EQ(d.curve, kExpect[i].curve) << "row " << i;
        EXPECT_FLOAT_EQ(d.smooth_time_s, kExpect[i].smooth_s) << "row " << i;
        EXPECT_EQ(d.clap_flags, kExpect[i].flags) << "row " << i;
        EXPECT_EQ(d.format, kExpect[i].format) << "row " << i;
    }
}

TEST(ParamTable, FxRowsMatchReference) {
    // QUIRK (O8): FX modules are empty as-shipped even though
    // PARAMETER_REFERENCE.md §1.2 claims "SSM". Pinned, not fixed.
    EXPECT_STREQ(kFxParamTable[0].name, "Gain");
    EXPECT_STREQ(kFxParamTable[0].module, "");
    EXPECT_DOUBLE_EQ(kFxParamTable[0].min_value, 0.0);
    EXPECT_DOUBLE_EQ(kFxParamTable[0].max_value, 2.0);
    EXPECT_DOUBLE_EQ(kFxParamTable[0].default_value, 1.0);
    EXPECT_EQ(kFxParamTable[0].curve, PARAM_CURVE_LOG);
    EXPECT_FLOAT_EQ(kFxParamTable[0].smooth_time_s, 0.010f);
    EXPECT_EQ(kFxParamTable[0].clap_flags, CLAP_PARAM_IS_AUTOMATABLE);
    EXPECT_EQ(kFxParamTable[0].format, ParamFormat::Plain);

    EXPECT_STREQ(kFxParamTable[1].name, "Mix");
    EXPECT_STREQ(kFxParamTable[1].module, "");
    EXPECT_DOUBLE_EQ(kFxParamTable[1].min_value, 0.0);
    EXPECT_DOUBLE_EQ(kFxParamTable[1].max_value, 1.0);
    EXPECT_DOUBLE_EQ(kFxParamTable[1].default_value, 1.0);
    EXPECT_EQ(kFxParamTable[1].curve, PARAM_CURVE_LINEAR);
    EXPECT_FLOAT_EQ(kFxParamTable[1].smooth_time_s, 0.010f);
    EXPECT_EQ(kFxParamTable[1].clap_flags, CLAP_PARAM_IS_AUTOMATABLE);
    EXPECT_EQ(kFxParamTable[1].format, ParamFormat::Plain);

    EXPECT_STREQ(kFxParamTable[2].name, "Bypass");
    EXPECT_STREQ(kFxParamTable[2].module, "");
    EXPECT_DOUBLE_EQ(kFxParamTable[2].min_value, 0.0);
    EXPECT_DOUBLE_EQ(kFxParamTable[2].max_value, 1.0);
    EXPECT_DOUBLE_EQ(kFxParamTable[2].default_value, 0.0);
    EXPECT_EQ(kFxParamTable[2].curve, PARAM_CURVE_STEPPED);
    EXPECT_FLOAT_EQ(kFxParamTable[2].smooth_time_s, 0.0f);
    EXPECT_EQ(kFxParamTable[2].clap_flags,
              CLAP_PARAM_IS_AUTOMATABLE | CLAP_PARAM_IS_STEPPED | CLAP_PARAM_IS_BYPASS);
    EXPECT_EQ(kFxParamTable[2].format, ParamFormat::BypassFx);

    for (size_t i = 0; i < kFxParamTableCount; ++i) {
        EXPECT_EQ(kFxParamTable[i].id, i) << "FX row " << i;
    }
}

TEST(ParamTable, StringsFitClapBuffers) {
    // get_info strncpy's these into fixed CLAP buffers; silently truncated
    // names would be a host-visible bug, so pin the headroom.
    for (uint32_t i = 0; i < PARAM_COUNT; ++i) {
        EXPECT_LT(std::strlen(kSynthParamTable[i].name), CLAP_NAME_SIZE);
        EXPECT_LT(std::strlen(kSynthParamTable[i].module), CLAP_PATH_SIZE);
    }
    for (size_t i = 0; i < kFxParamTableCount; ++i) {
        EXPECT_LT(std::strlen(kFxParamTable[i].name), CLAP_NAME_SIZE);
        EXPECT_LT(std::strlen(kFxParamTable[i].module), CLAP_PATH_SIZE);
    }
}

TEST(ParamTable, InitParamSmoothCopiesTable) {
    ParamSmooth params[PARAM_COUNT] = {};
    init_param_smooth(params, kSynthParamTable);
    for (uint32_t i = 0; i < PARAM_COUNT; ++i) {
        EXPECT_DOUBLE_EQ(params[i].min_value, kSynthParamTable[i].min_value);
        EXPECT_DOUBLE_EQ(params[i].max_value, kSynthParamTable[i].max_value);
        EXPECT_DOUBLE_EQ(params[i].default_value, kSynthParamTable[i].default_value);
        EXPECT_EQ(params[i].curve, kSynthParamTable[i].curve);
        EXPECT_FLOAT_EQ(params[i].smooth_time_s, kSynthParamTable[i].smooth_time_s);
    }

    ParamSmooth fx_params[kFxParamTableCount] = {};
    init_param_smooth(fx_params, kFxParamTable);
    for (size_t i = 0; i < kFxParamTableCount; ++i) {
        EXPECT_DOUBLE_EQ(fx_params[i].min_value, kFxParamTable[i].min_value);
        EXPECT_DOUBLE_EQ(fx_params[i].max_value, kFxParamTable[i].max_value);
        EXPECT_DOUBLE_EQ(fx_params[i].default_value, kFxParamTable[i].default_value);
        EXPECT_EQ(fx_params[i].curve, kFxParamTable[i].curve);
        EXPECT_FLOAT_EQ(fx_params[i].smooth_time_s, kFxParamTable[i].smooth_time_s);
    }
}

// ---------- value -> text ----------

std::string Format(ParamFormat fmt, double v) {
    char buf[64];
    EXPECT_TRUE(param_value_to_text(fmt, v, buf, sizeof(buf)));
    return buf;
}

TEST(ParamTable, FormatGainDb) {
    EXPECT_EQ(Format(ParamFormat::GainDb, 1.0), "0.0 dB");
    EXPECT_EQ(Format(ParamFormat::GainDb, 2.0), "6.0 dB");
    EXPECT_EQ(Format(ParamFormat::GainDb, 0.5), "-6.0 dB");
    EXPECT_EQ(Format(ParamFormat::GainDb, 0.0), "-180.0 dB");
}

TEST(ParamTable, FormatPercent) {
    EXPECT_EQ(Format(ParamFormat::Percent, 0.75), "75%");
    EXPECT_EQ(Format(ParamFormat::Percent, 1.0), "100%");
    EXPECT_EQ(Format(ParamFormat::Percent, 0.0), "0%");
}

TEST(ParamTable, FormatMilliseconds) {
    EXPECT_EQ(Format(ParamFormat::Milliseconds, 0.1), "100 ms");
    EXPECT_EQ(Format(ParamFormat::Milliseconds, 0.001), "1 ms");
    EXPECT_EQ(Format(ParamFormat::Milliseconds, 1.5), "1500 ms");
}

TEST(ParamTable, FormatBypassThresholds) {
    EXPECT_EQ(Format(ParamFormat::Bypass, 1.0), "On");
    EXPECT_EQ(Format(ParamFormat::Bypass, 0.0), "Off");
    // QUIRK (O7): synth uses strict '>' while FX uses '>='.
    EXPECT_EQ(Format(ParamFormat::Bypass, 0.5), "Off");
    EXPECT_EQ(Format(ParamFormat::BypassFx, 0.5), "On");
    EXPECT_EQ(Format(ParamFormat::BypassFx, 1.0), "On");
    EXPECT_EQ(Format(ParamFormat::BypassFx, 0.0), "Off");
}

TEST(ParamTable, FormatPlain) {
    EXPECT_EQ(Format(ParamFormat::Plain, 1.0), "1.000");
    EXPECT_EQ(Format(ParamFormat::Plain, 0.85), "0.850");
}

TEST(ParamTable, FormatRejectsBadBuffer) {
    char buf[8];
    EXPECT_FALSE(param_value_to_text(ParamFormat::Plain, 1.0, nullptr, sizeof(buf)));
    EXPECT_FALSE(param_value_to_text(ParamFormat::Plain, 1.0, buf, 0));
}

// ---------- text -> value ----------

struct ParseResult {
    bool ok;
    double value;
};

ParseResult Parse(ParamFormat fmt, const char* text, double min_v, double max_v) {
    ParseResult r{false, 0.0};
    r.ok = param_text_to_value(fmt, text, min_v, max_v, &r.value);
    return r;
}

TEST(ParamTable, ParseGainDb) {
    EXPECT_DOUBLE_EQ(Parse(ParamFormat::GainDb, "1.0", 0.0, 2.0).value, 1.0);
    EXPECT_TRUE(Parse(ParamFormat::GainDb, "1.0", 0.0, 2.0).ok);
    // A single 'd'/'D' (any case) triggers the dB conversion.
    EXPECT_NEAR(Parse(ParamFormat::GainDb, "-6 dB", 0.0, 2.0).value, 0.501187, 1e-6);
    EXPECT_NEAR(Parse(ParamFormat::GainDb, "-6 db", 0.0, 2.0).value, 0.501187, 1e-6);
    EXPECT_NEAR(Parse(ParamFormat::GainDb, "-6 D", 0.0, 2.0).value, 0.501187, 1e-6);
    EXPECT_TRUE(Parse(ParamFormat::GainDb, "-6 dB", 0.0, 2.0).ok);
    EXPECT_DOUBLE_EQ(Parse(ParamFormat::GainDb, "5", 0.0, 2.0).value, 2.0);
    EXPECT_DOUBLE_EQ(Parse(ParamFormat::GainDb, "-3", 0.0, 2.0).value, 0.0);
    EXPECT_FALSE(Parse(ParamFormat::GainDb, "garbage", 0.0, 2.0).ok);
    EXPECT_FALSE(Parse(ParamFormat::GainDb, "", 0.0, 2.0).ok);
    // QUIRK (O7): no isfinite validation -- "nan" parses successfully.
    ParseResult nan = Parse(ParamFormat::GainDb, "nan", 0.0, 2.0);
    EXPECT_TRUE(nan.ok);
    EXPECT_TRUE(std::isnan(nan.value));
}

TEST(ParamTable, ParsePercent) {
    EXPECT_DOUBLE_EQ(Parse(ParamFormat::Percent, "75%", 0.0, 1.0).value, 0.75);
    EXPECT_DOUBLE_EQ(Parse(ParamFormat::Percent, "0.75", 0.0, 1.0).value, 0.75);
    EXPECT_DOUBLE_EQ(Parse(ParamFormat::Percent, "75", 0.0, 1.0).value, 0.75);
    EXPECT_DOUBLE_EQ(Parse(ParamFormat::Percent, "50 %", 0.0, 1.0).value, 0.5);
    EXPECT_DOUBLE_EQ(Parse(ParamFormat::Percent, "150%", 0.0, 1.0).value, 1.0);
    EXPECT_FALSE(Parse(ParamFormat::Percent, "mix", 0.0, 1.0).ok);
}

TEST(ParamTable, ParseMilliseconds) {
    // Bare numbers are milliseconds.
    EXPECT_DOUBLE_EQ(Parse(ParamFormat::Milliseconds, "100", 0.001, 1.0).value, 0.1);
    EXPECT_DOUBLE_EQ(Parse(ParamFormat::Milliseconds, "100 ms", 0.001, 1.0).value, 0.1);
    EXPECT_DOUBLE_EQ(Parse(ParamFormat::Milliseconds, "100 MS", 0.001, 1.0).value, 0.1);
    EXPECT_NEAR(Parse(ParamFormat::Milliseconds, "0.1 s", 0.001, 1.0).value, 0.1, 1e-12);
    // QUIRK (O7): bare "0.1" means 0.1 ms, clamped up to the 1 ms minimum.
    EXPECT_DOUBLE_EQ(Parse(ParamFormat::Milliseconds, "0.1", 0.001, 1.0).value, 0.001);
    EXPECT_FALSE(Parse(ParamFormat::Milliseconds, "fast", 0.001, 1.0).ok);
}

TEST(ParamTable, ParseBypassSynth) {
    EXPECT_DOUBLE_EQ(Parse(ParamFormat::Bypass, "On", 0.0, 1.0).value, 1.0);
    EXPECT_DOUBLE_EQ(Parse(ParamFormat::Bypass, "on", 0.0, 1.0).value, 1.0);
    EXPECT_DOUBLE_EQ(Parse(ParamFormat::Bypass, "ON", 0.0, 1.0).value, 1.0);
    EXPECT_DOUBLE_EQ(Parse(ParamFormat::Bypass, "Off", 0.0, 1.0).value, 0.0);
    EXPECT_DOUBLE_EQ(Parse(ParamFormat::Bypass, "off", 0.0, 1.0).value, 0.0);
    EXPECT_DOUBLE_EQ(Parse(ParamFormat::Bypass, "OFF", 0.0, 1.0).value, 0.0);
    EXPECT_DOUBLE_EQ(Parse(ParamFormat::Bypass, "1", 0.0, 1.0).value, 1.0);
    EXPECT_DOUBLE_EQ(Parse(ParamFormat::Bypass, "0", 0.0, 1.0).value, 0.0);
    EXPECT_DOUBLE_EQ(Parse(ParamFormat::Bypass, "0.7", 0.0, 1.0).value, 1.0);
    EXPECT_DOUBLE_EQ(Parse(ParamFormat::Bypass, "0.3", 0.0, 1.0).value, 0.0);
    // Strict '>' numeric threshold, matching the display side.
    EXPECT_DOUBLE_EQ(Parse(ParamFormat::Bypass, "0.5", 0.0, 1.0).value, 0.0);
    // Trailing junk after "On" fails the exact match and is not numeric.
    EXPECT_FALSE(Parse(ParamFormat::Bypass, "On ", 0.0, 1.0).ok);
    EXPECT_FALSE(Parse(ParamFormat::Bypass, "garbage", 0.0, 1.0).ok);
}

TEST(ParamTable, ParseBypassFx) {
    // QUIRK (O7): FX bypass parsing is case-sensitive and never fails --
    // anything but "On"/"1" silently becomes 0.
    ParseResult on = Parse(ParamFormat::BypassFx, "On", 0.0, 1.0);
    EXPECT_TRUE(on.ok);
    EXPECT_DOUBLE_EQ(on.value, 1.0);
    EXPECT_DOUBLE_EQ(Parse(ParamFormat::BypassFx, "1", 0.0, 1.0).value, 1.0);
    ParseResult off = Parse(ParamFormat::BypassFx, "on", 0.0, 1.0);
    EXPECT_TRUE(off.ok);
    EXPECT_DOUBLE_EQ(off.value, 0.0);
    EXPECT_DOUBLE_EQ(Parse(ParamFormat::BypassFx, "Off", 0.0, 1.0).value, 0.0);
    EXPECT_DOUBLE_EQ(Parse(ParamFormat::BypassFx, "0.7", 0.0, 1.0).value, 0.0);
    ParseResult junk = Parse(ParamFormat::BypassFx, "garbage", 0.0, 1.0);
    EXPECT_TRUE(junk.ok);
    EXPECT_DOUBLE_EQ(junk.value, 0.0);
}

TEST(ParamTable, ParsePlainFx) {
    // QUIRK (O7): FX gain/mix parsing handles no suffixes and never clamps.
    EXPECT_DOUBLE_EQ(Parse(ParamFormat::Plain, "1.5", 0.0, 2.0).value, 1.5);
    EXPECT_DOUBLE_EQ(Parse(ParamFormat::Plain, "75%", 0.0, 1.0).value, 75.0);
    EXPECT_DOUBLE_EQ(Parse(ParamFormat::Plain, "5", 0.0, 2.0).value, 5.0);
    EXPECT_FALSE(Parse(ParamFormat::Plain, "nan", 0.0, 2.0).ok);
    EXPECT_FALSE(Parse(ParamFormat::Plain, "inf", 0.0, 2.0).ok);
    EXPECT_FALSE(Parse(ParamFormat::Plain, "gain", 0.0, 2.0).ok);
    EXPECT_FALSE(param_text_to_value(ParamFormat::Plain, nullptr, 0.0, 2.0, nullptr));
}

TEST(ParamTable, FormatParseRoundTrip) {
    // Spot-check that formatted defaults parse back to (near-)identical values.
    char buf[32];
    ASSERT_TRUE(param_value_to_text(ParamFormat::GainDb, 1.0, buf, sizeof(buf)));
    EXPECT_DOUBLE_EQ(Parse(ParamFormat::GainDb, buf, 0.0, 2.0).value, 1.0);
    ASSERT_TRUE(param_value_to_text(ParamFormat::Percent, 1.0, buf, sizeof(buf)));
    EXPECT_DOUBLE_EQ(Parse(ParamFormat::Percent, buf, 0.0, 1.0).value, 1.0);
    ASSERT_TRUE(param_value_to_text(ParamFormat::Milliseconds, 0.3, buf, sizeof(buf)));
    EXPECT_DOUBLE_EQ(Parse(ParamFormat::Milliseconds, buf, 0.001, 2.0).value, 0.3);
}

}  // namespace
}  // namespace mamba
