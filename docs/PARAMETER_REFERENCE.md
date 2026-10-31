# Parameter Reference

**EmT CLAP/ARA Plugin Suite** — Architecture Version: Mamba-3 (`v0.3.0`)  
**Source Files:** `src/clap/param_table.h`, `src/clap/param_table.cpp`, `src/clap/clap_params.cpp`, `src/clap/clap_plugin.h`, `src/clap/clap_fx_plugin.h`, `src/clap/clap_fx_plugin.cpp`, `src/clap/clap_note_ports.cpp`

Parameter metadata (ranges, defaults, curves, smoothing times, names, modules,
flags, text formats) lives in the `ParamDesc` tables in `src/clap/param_table.h`
and is pinned by `tests/test_param_table.cpp`. The tables are the single source
of truth; the sections below restate them for readability.

This document provides complete, normative documentation for all parameter definitions, data ranges, default values, smoothing filters, MIDI CC bindings, and control behavior for both plugins exported by the binary:
1. **EmT Synth** (`com.emt.synth`): Generative neural synthesizer with envelope modulation, multi-voice polyphony, and FiLM conditioning.
2. **EmT FX** (`com.emt.fx`): Neural audio effect processor utilizing the Mamba-3 SSM inference core for dynamic non-linear transformation.

---

## 1. Quick Reference Summaries

### 1.1 EmT Synth Parameters (`com.emt.synth`)

Exposed via `CLAP_EXT_PARAMS`. Total count: `PARAM_COUNT = 5`.

| ID | Name | Module | Min | Max | Default | Curve | Smooth Time | CLAP Flags |
|---|---|---|---|---|---|---|---|---|
| **0** | **Gain** | `SSM` | `0.0` (-inf dB) | `2.0` (+6.0 dB) | `1.0` (0.0 dB) | Logarithmic | 10 ms | `CLAP_PARAM_IS_AUTOMATABLE` |
| **1** | **Attack** | `Envelope` | `0.001` s (1 ms) | `1.0` s (1000 ms) | `0.1` s (100 ms) | Logarithmic | 20 ms | `CLAP_PARAM_IS_AUTOMATABLE` |
| **2** | **Release** | `Envelope` | `0.001` s (1 ms) | `2.0` s (2000 ms) | `0.3` s (300 ms) | Logarithmic | 50 ms | `CLAP_PARAM_IS_AUTOMATABLE` |
| **3** | **Mix** | `SSM` | `0.0` (0% wet) | `1.0` (100% wet) | `1.0` (100% wet) | Linear | 10 ms | `CLAP_PARAM_IS_AUTOMATABLE` |
| **4** | **Bypass** | `""` (root) | `0.0` (Off) | `1.0` (On) | `0.0` (Off) | Stepped | 0 ms (Instant) | `IS_AUTOMATABLE \| IS_STEPPED \| IS_BYPASS` |

### 1.2 EmT FX Parameters (`com.emt.fx`)

Exposed via `CLAP_EXT_PARAMS`. Total count: `FX_PARAM_COUNT = 3`.

| ID | Name | Module | Min | Max | Default | Curve | Smooth Time | CLAP Flags |
|---|---|---|---|---|---|---|---|---|
| **0** | **Gain** | `""` (empty as-shipped; see O8) | `0.0` (-inf dB) | `2.0` (+6.0 dB) | `1.0` (0.0 dB) | Logarithmic | 10 ms | `CLAP_PARAM_IS_AUTOMATABLE` |
| **1** | **Mix** | `""` (empty as-shipped; see O8) | `0.0` (0% wet) | `1.0` (100% wet) | `1.0` (100% wet) | Linear | 10 ms | `CLAP_PARAM_IS_AUTOMATABLE` |
| **2** | **Bypass** | `""` (root) | `0.0` (Off) | `1.0` (On) | `0.0` (Off) | Stepped | 0 ms (Instant) | `IS_AUTOMATABLE \| IS_STEPPED \| IS_BYPASS` |

---

## 2. Detailed Parameter Specifications

### 2.1 PARAM_GAIN / FX_PARAM_GAIN (ID: 0)

| Property | Value |
|---|---|
| **CLAP ID** | `0` (`PARAM_GAIN` / `FX_PARAM_GAIN`) |
| **Display Name** | `"Gain"` |
| **Module Hierarchy** | Synth: `"SSM"`; FX: `""` (empty as-shipped, see O8) |
| **Internal Unit** | Linear amplitude scalar (\(A\)) |
| **Minimum Value** | `0.0` (\(-\infty\text{ dB}\)) |
| **Maximum Value** | `2.0` (\(+6.02\text{ dB}\)) |
| **Default Value** | `1.0` (\(0.00\text{ dB}\), unity gain) |
| **Mapping Curve** | `PARAM_CURVE_LOG` (Logarithmic taper) |
| **Smoothing Constant** | `smooth_time_s = 0.010f` (10 ms) |
| **Flags** | `CLAP_PARAM_IS_AUTOMATABLE` |

#### Value to Text Formatting
- **Synth:** Formatted in decibels:
  $$\text{dB} = 20 \cdot \log_{10}(\max(v, 10^{-9}))$$
  Output string format: `snprintf(out_buffer, cap, "%.1f dB", db)`.
- **FX:** Formatted as decimal amplitude factor:
  Output string format: `snprintf(out, size, "%.3f", value)`.

#### Text to Value Parsing
- **Synth:** Accepts raw decimal strings (e.g., `"0.5"`, `"1.0"`, `"1.414"`).
- **Synth:** A single `'d'`/`'D'` after optional spaces triggers dB conversion (e.g., `"-6.0 dB"`, `"+3 D"`, `"0dB"`):
  $$v = 10^{\frac{\text{dB}}{20}}$$
- **Synth:** Clamped strictly to \([0.0, 2.0]\).
- **FX:** Plain `strtod` decimal only: no dB handling and no clamping (`"-6 dB"` parses as `-6.0`). Non-finite input is rejected.

#### DSP Signal Flow
The smoothed gain value scales the wet neural output prior to channel bus mixing:
$$y_{\text{wet}}[n] = y_{\text{ssm}}[n] \cdot \text{gain}[n]$$

---

### 2.2 PARAM_ATTACK (Synth ID: 1)

| Property | Value |
|---|---|
| **CLAP ID** | `1` (`PARAM_ATTACK`) |
| **Display Name** | `"Attack"` |
| **Module Hierarchy** | `"Envelope"` |
| **Internal Unit** | Seconds (\(s\)) |
| **Minimum Value** | `0.001` s (1.0 ms) |
| **Maximum Value** | `1.0` s (1000.0 ms) |
| **Default Value** | `0.1` s (100.0 ms) |
| **Mapping Curve** | `PARAM_CURVE_LOG` (Logarithmic taper) |
| **Smoothing Constant** | `smooth_time_s = 0.020f` (20 ms) |
| **Flags** | `CLAP_PARAM_IS_AUTOMATABLE` |

#### Value to Text Formatting
- Displayed in milliseconds:
  $$\text{ms} = v \cdot 1000.0$$
  Output string format: `snprintf(out_buffer, cap, "%.0f ms", v * 1000.0)`.

#### Text to Value Parsing
- Accepts plain integers/decimals representing milliseconds (e.g., `"50"`, `"250 ms"`).
- Accepts second notations (e.g., `"0.05s"`, `"0.5 s"`). Values with an `'s'` or `'S'` suffix are multiplied by 1000 before conversion to internal seconds.
- Clamped strictly to \([0.001, 1.0]\).

#### DSP Signal Flow
Controls the rise rate of the monophonic envelope for each allocated voice. At note-on (`CLAP_EVENT_NOTE_ON`), the voice transitions to `VOICE_ATTACK`:
$$\text{attack\_rate} = \frac{1.0}{\max(1.0, \text{attack\_time} \cdot f_s)}$$
On each sample \(n\):
$$\text{env}[n] = \text{env}[n-1] + \text{attack\_rate}$$
When \(\text{env}[n] \ge 1.0\), the envelope saturates to \(1.0\) and transitions to `VOICE_SUSTAIN`.

> **Shipped behavior (O10, found in #8):** the `attack_time` → `attack_rate` formula above is the *intended* mapping, not the wired one. `start_voice()` assigns the fixed constant `VOICE_ATTACK_RATE` (1/1024 per sample); the `PARAM_ATTACK` value is accepted, range-checked, stored, and has smoothing state allocated, but its smoothed value is never stepped in `process()` and nothing reads it. The per-sample step itself is `voice_envelope_step()` in `src/dsp/voice_envelope.h`, which takes the rates as parameters -- wiring the knob is a rate computation at the call site (Phase 2).

---

### 2.3 PARAM_RELEASE (Synth ID: 2)

| Property | Value |
|---|---|
| **CLAP ID** | `2` (`PARAM_RELEASE`) |
| **Display Name** | `"Release"` |
| **Module Hierarchy** | `"Envelope"` |
| **Internal Unit** | Seconds (\(s\)) |
| **Minimum Value** | `0.001` s (1.0 ms) |
| **Maximum Value** | `2.0` s (2000.0 ms) |
| **Default Value** | `0.3` s (300.0 ms) |
| **Mapping Curve** | `PARAM_CURVE_LOG` (Logarithmic taper) |
| **Smoothing Constant** | `smooth_time_s = 0.050f` (50 ms) |
| **Flags** | `CLAP_PARAM_IS_AUTOMATABLE` |

#### Value to Text Formatting
- Displayed in milliseconds:
  $$\text{ms} = v \cdot 1000.0$$
  Output string format: `snprintf(out_buffer, cap, "%.0f ms", v * 1000.0)`.

#### Text to Value Parsing
- Accepts millisecond strings (e.g., `"300"`, `"300 ms"`).
- Accepts second strings (e.g., `"1.2 s"`, `"0.3s"`).
- Clamped strictly to \([0.001, 2.0]\).

#### DSP Signal Flow
Controls the decay rate of the voice envelope when a note-off event occurs (`CLAP_EVENT_NOTE_OFF` or velocity \(\le 0\)). The voice enters `VOICE_RELEASE`:
$$\text{release\_rate} = \frac{1.0}{\max(1.0, \text{release\_time} \cdot f_s)}$$
On each sample \(n\):
$$\text{env}[n] = \text{env}[n-1] - \text{release\_rate}$$
When \(\text{env}[n] \le 0.0\), the voice enters `VOICE_SILENT` and becomes available for reallocation.

> **Shipped behavior (O10, found in #8):** the `release_time` → `release_rate` formula above is the *intended* mapping, not the wired one. `start_voice()` assigns the fixed constant `VOICE_RELEASE_RATE` (1/4096 per sample); the `PARAM_RELEASE` value is accepted, range-checked, stored, and has smoothing state allocated, but its smoothed value is never stepped in `process()` and nothing reads it. The per-sample step itself is `voice_envelope_step()` in `src/dsp/voice_envelope.h`, which takes the rates as parameters -- wiring the knob is a rate computation at the call site (Phase 2).

---

### 2.4 PARAM_MIX / FX_PARAM_MIX (Synth ID: 3, FX ID: 1)

| Property | Value |
|---|---|
| **CLAP ID** | Synth: `3`, FX: `1` |
| **Display Name** | `"Mix"` |
| **Module Hierarchy** | Synth: `"SSM"`; FX: `""` (empty as-shipped, see O8) |
| **Internal Unit** | Normalized wet ratio (\([0.0, 1.0]\)) |
| **Minimum Value** | `0.0` (100% dry passthrough) |
| **Maximum Value** | `1.0` (100% wet neural output) |
| **Default Value** | `1.0` (100% wet) |
| **Mapping Curve** | `PARAM_CURVE_LINEAR` |
| **Smoothing Constant** | `smooth_time_s = 0.010f` (10 ms) |
| **Flags** | `CLAP_PARAM_IS_AUTOMATABLE` |

#### Value to Text Formatting
- **Synth:** Displayed as an integer percentage:
  $$\text{Percent} = v \cdot 100.0$$
  Output string format: `snprintf(out_buffer, cap, "%.0f%%", value * 100.0)`.
- **FX:** Displayed as a decimal fraction:
  Output string format: `snprintf(out, size, "%.3f", value)`.

#### Text to Value Parsing
- **Synth:** Accepts percentage values (e.g., `"75%"`, `"50 %"`).
- **Synth:** Accepts raw numeric representations (e.g., `"0.75"`, `"1.0"`).
- **Synth:** Numbers entered \(> 1.0\) without a `%` sign are treated as percentage values (\(v = v / 100.0\)).
- **Synth:** Clamped strictly to \([0.0, 1.0]\).
- **FX:** Plain `strtod` decimal only: `"75%"` parses as `75.0`, no clamping. Non-finite input is rejected.

#### DSP Signal Flow
Blends the unprocessed incoming audio (dry) with the SSM inference output (wet):
$$y_{\text{mixed}}[n] = x_{\text{dry}}[n] \cdot (1.0 - \text{mix}[n]) + y_{\text{wet}}[n] \cdot \text{mix}[n]$$

---

### 2.5 PARAM_BYPASS / FX_PARAM_BYPASS (Synth ID: 4, FX ID: 2)

| Property | Value |
|---|---|
| **CLAP ID** | Synth: `4`, FX: `2` |
| **Display Name** | `"Bypass"` |
| **Module Hierarchy** | `""` (root) |
| **Internal Unit** | Boolean toggle (`0.0` = Inactive, `1.0` = Active) |
| **Minimum Value** | `0.0` |
| **Maximum Value** | `1.0` |
| **Default Value** | `0.0` (Bypass disabled) |
| **Mapping Curve** | `PARAM_CURVE_STEPPED` |
| **Smoothing Constant** | `0.0f` (Bypass smoothing disabled; managed by crossfade engine) |
| **Flags** | `CLAP_PARAM_IS_AUTOMATABLE \| CLAP_PARAM_IS_STEPPED \| CLAP_PARAM_IS_BYPASS` |

#### Value to Text Formatting
- **Synth:** `value > 0.5` \(\to\) `"On"`, otherwise `"Off"` (strict threshold).
- **FX:** `value >= 0.5` \(\to\) `"On"`, otherwise `"Off"`.

#### Text to Value Parsing
- **Synth:** `"On"`/`"on"`/`"ON"` \(\to 1.0\); `"Off"`/`"off"`/`"OFF"` \(\to 0.0\); anything else falls back to numeric parsing with `v > 0.5` \(\to 1.0\) (so `"1"` \(\to 1.0\), `"0"`/`"0.5"` \(\to 0.0\)). Fails only when no number is present.
- **FX:** Only `"On"` and `"1"` \(\to 1.0\); literally anything else \(\to 0.0\), and parsing never fails.

#### DSP Signal Flow
Drives the 4-state crossfade state machine (`BYPASS_PROCESSING`, `BYPASS_FADING_OUT`, `BYPASS_BYPASSED`, `BYPASS_FADING_IN`). Crossfade rate is fixed at:
$$\text{BYPASS\_FADE\_RATE} = \frac{1.0}{64.0} \quad (\approx 1.33\text{ ms at } 48\text{ kHz})$$
Guarantees click-free transitions between processed audio and clean dry input.

---

## 3. Parameter Smoothing Filter Curves

### 3.1 One-Pole Lowpass IIR Filter Architecture

Every continuous automatable parameter is smoothed on the real-time audio thread using an exponential single-pole lowpass filter (leaky integrator). This prevents Zipper noise and discontinuities caused by quantized automation steps or MIDI CC streams.

```
 Target Value x[n] ─────────(+)───────[ × coeff ]──────(+)───────┬────── Smoothed Value y[n]
                             ▲                          ▲        │
                             │ -1                       │        │
                             └──────────[ z⁻¹ ]─────────┴────────┘
```

### 3.2 Mathematical Formulation

The sample-rate dependent filter coefficient is calculated when the plugin activates or when the sample rate changes:

$$\tau = \text{smooth\_time\_s}$$
$$t_{\text{samples}} = \tau \cdot f_s$$
$$c = \begin{cases} 1.0 - \exp\left(-\frac{1.0}{t_{\text{samples}}}\right) & \text{if } \tau > 0 \\ 1.0 & \text{if } \tau \le 0 \end{cases}$$

During the sample loop, the filter is updated on each frame:

$$y[n] = y[n-1] + c \cdot (x[n] - y[n-1])$$

Where:
- \(x[n]\) is the automation target stored in atomic memory from the SPSC ring buffer or host parameter flush.
- \(y[n]\) is the current smoothed value used in DSP arithmetic.
- \(c\) is the precomputed filter coefficient.

### 3.3 Frequency and Step Response

The step response of the filter reaching within \(\epsilon = 1\%\) (\(-40\text{ dB}\)) of the target occurs at:
$$t_{99\%} \approx 4.6 \cdot \tau$$

| Parameter | \(\tau\) (Smooth Time) | \(c\) @ 44.1 kHz | \(c\) @ 48 kHz | \(c\) @ 96 kHz | \(t_{99\%}\) (Settling Time) |
|---|---|---|---|---|---|
| **Gain** | 10 ms (`0.010 s`) | `0.002265` | `0.002081` | `0.001041` | ~46.0 ms |
| **Attack** | 20 ms (`0.020 s`) | `0.001133` | `0.001041` | `0.000521` | ~92.1 ms |
| **Release** | 50 ms (`0.050 s`) | `0.000453` | `0.000417` | `0.000208` | ~230.3 ms |
| **Mix** | 10 ms (`0.010 s`) | `0.002265` | `0.002081` | `0.001041` | ~46.0 ms |
| **Bypass** | 0 ms (`0.000 s`) | `1.000000` | `1.000000` | `1.000000` | Instant (managed by fade ramp) |

> **Note (O10):** the Attack/Release coefficients above are computed at activation but never stepped in `process()` -- only Gain and Mix advance in the sample loops. Their settling times are therefore inert until the knobs are wired to the envelope rates (Phase 2).

### 3.4 Knob Curves (Host-Side)

The `curve` field in the parameter table (`PARAM_CURVE_LOG` / `PARAM_CURVE_LINEAR` /
`PARAM_CURVE_STEPPED`) is descriptive metadata: the DSP reads raw engineering values and
hosts map their own knobs linearly over `[min_value, max_value]`. The plugin consumes the
curve only for stepped detection (`PARAM_CURVE_STEPPED` drives toggle behavior in the
built-in GUI and the `CLAP_PARAM_IS_STEPPED` flag). The former log knob-mapping helpers
were dead code and were removed; there is currently no plugin-side logarithmic taper.

---

## 4. MIDI CC Mappings & Note Event Handling

### 4.1 CLAP Note Ports & Dialects

The plugin advertises a single note input port via `CLAP_EXT_NOTE_PORTS`:
- **Dialects Supported:** `CLAP_NOTE_DIALECT_CLAP | CLAP_NOTE_DIALECT_MIDI`
- **Preferred Dialect:** `CLAP_NOTE_DIALECT_CLAP`

```
DAW MIDI / Automation Input
        │
        ├── CLAP_EVENT_NOTE_ON  ──────► Voice Allocation & Envelope Trigger
        ├── CLAP_EVENT_NOTE_OFF ─────► Voice Release Phase Trigger
        └── CLAP_EVENT_PARAM_VALUE ──► SPSC Ring Buffer ──► 1-Pole Smoothing Filter
```

### 4.2 Standard Recommended MIDI CC Mapping

In compliant CLAP hosts (such as Bitwig Studio, Reaper, Ableton Live with CLAP wrappers), MIDI CC messages assigned to plugin controls are translated directly by the host into `CLAP_EVENT_PARAM_VALUE` events.

For hardware MIDI controllers sending standard CC messages directly, the recommended default mapping profile is:

| MIDI CC | Controller Function | Target Parameter | Range / Mapping |
|---|---|---|---|
| **CC 7** | Channel Volume | `PARAM_GAIN` | `0 – 127` \(\to 0.0 - 2.0\) linear (unity at CC 100 \(\approx 1.0\)) |
| **CC 1** | Modulation Wheel | `PARAM_MIX` | `0 – 127` \(\to 0.0 - 1.0\) (0% to 100% wet) |
| **CC 73** | Attack Time | `PARAM_ATTACK` | `0 – 127` \(\to 0.001 - 1.0\) s (logarithmic taper) |
| **CC 72** | Release Time | `PARAM_RELEASE` | `0 – 127` \(\to 0.001 - 2.0\) s (logarithmic taper) |
| **CC 64** | Sustain Pedal | Envelope Hold | Holds voices in `VOICE_SUSTAIN` until CC 64 \(< 64\) |
| **CC 11** | Expression | Secondary Dynamics | Multiplied with voice velocity |
| **CC 120** | All Sound Off | Panic / Reset | Mutes all voices immediately (`VOICE_SILENT`) |
| **CC 123** | All Notes Off | Release All | Transitions all active voices to `VOICE_RELEASE` |

### 4.3 CLAP Automation vs MIDI Conflict Prevention

Per the CLAP specification:
- When a parameter changes due to a MIDI CC, the host or plugin marks the event with `CLAP_EVENT_DONT_RECORD` to prevent duplicate automation feedback loops.
- `params_flush()` verifies `space_id == CLAP_CORE_EVENT_SPACE_ID`. Any event outside the core CLAP namespace is silently ignored.
- Sample-accurate timing: Each parameter atom carries a `frame_offset` within the processing block, allowing sub-sample slicing during audio rendering.

---

## 5. State Serialization & Preset Format

### 5.1 Parameter State Block in Presets (`.mambapreset`)

When saved via `CLAP_EXT_STATE` or the standalone preset manager (`preset_codec.cpp`), parameters are serialized in little-endian binary format:

```
Offset   Type     Field             Description
0x00     uint32   magic             0x4D425052 ('MBPR')
0x04     uint32   version           0x00020000 (v2.0)
0x08     uint32   reserved          0x00000000
0x0C     uint32   param_count       PARAM_COUNT (5)
0x10     double   param_values[5]   IEEE 754 64-bit float per parameter
0x38     uint32   bypass_target     0 (False) or 1 (True)
0x3C     float    bypass_fade       Current fade crossfade value [0.0, 1.0]
0x40     uint32   h_state_present   1 if neural states follow, 0 otherwise
```

When presets are loaded, all values are validated against `min_value` and `max_value`. Values outside the valid range cause the load operation to reject gracefully without corrupting memory or destabilizing the DSP core.
