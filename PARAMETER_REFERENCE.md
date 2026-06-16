# Parameter Reference

All parameters are exposed via `CLAP_EXT_PARAMS`. Source: `src/clap/clap_params.cpp`,
`src/clap/clap_plugin.h`.

---

## Parameter Summary

| ID | Name | Range | Default | Curve | Module | CLAP Flags |
|----|------|-------|---------|-------|--------|------------|
| 0 | Gain | 0.0 – 2.0 | 1.0 | Log | SSM | `IS_AUTOMATABLE` |
| 1 | Attack | 0.001 – 5.0 s | 0.01 s | Log | Envelope | `IS_AUTOMATABLE` |
| 2 | Release | 0.001 – 5.0 s | 0.1 s | Log | Envelope | `IS_AUTOMATABLE` |
| 3 | Mix | 0.0 – 1.0 | 0.5 | Linear | SSM | `IS_AUTOMATABLE` |
| 4 | Bypass | 0.0 / 1.0 | 0 (Off) | Stepped | — | `IS_AUTOMATABLE \| IS_STEPPED \| IS_BYPASS` |

---

## Detailed Parameter Descriptions

### PARAM_GAIN (ID: 0)

| Property | Value |
|----------|-------|
| **CLAP ID** | `0` |
| **Name** | `"Gain"` |
| **Module** | `"SSM"` |
| **Min / Max** | `0.0` / `2.0` |
| **Default** | `1.0` |
| **Curve** | Logarithmic |
| **Flags** | `CLAP_PARAM_IS_AUTOMATABLE` |

**Text display:** `"%.1f dB"` — computed as `20 * log10(value)`.

**Text input:** Accepts plain linear (`"0.5"`, `"1.2"`) or dB suffix (`"-6.0 dB"`, `"3.0dB"`).
dB values are converted: `value = 10^(dB / 20)`.

**Smoothing:** One-pole lowpass, coefficient computed from `smooth_time_s = 0.010f` at the
current sample rate.

---

### PARAM_ATTACK (ID: 1)

| Property | Value |
|----------|-------|
| **CLAP ID** | `1` |
| **Name** | `"Attack"` |
| **Module** | `"Envelope"` |
| **Min / Max** | `0.001` / `5.0` seconds |
| **Default** | `0.01` seconds |
| **Curve** | Logarithmic |
| **Flags** | `CLAP_PARAM_IS_AUTOMATABLE` |

**Text display:** `"%.0f ms"` — converted: `value * 1000`.

**Text input:** Accepts raw milliseconds (`"100"`, `"100 ms"`), or seconds (`"0.1 s"`, `"1.0s"`).
Seconds values are multiplied by 1000 before converting back to the stored seconds unit.

**DSP usage:** Controls the attack rate of the monophonic voice envelope. During the attack
phase, `voice_env` ramps from 0.0 to `velocity` over `attack_time * sample_rate` samples.

---

### PARAM_RELEASE (ID: 2)

| Property | Value |
|----------|-------|
| **CLAP ID** | `2` |
| **Name** | `"Release"` |
| **Module** | `"Envelope"` |
| **Min / Max** | `0.001` / `5.0` seconds |
| **Default** | `0.1` seconds |
| **Curve** | Logarithmic |
| **Flags** | `CLAP_PARAM_IS_AUTOMATABLE` |

**Text display:** `"%.0f ms"` — converted: `value * 1000`.

**Text input:** Same format as Attack (ms or seconds).

**DSP usage:** Controls the release rate of the monophonic voice envelope. During the release
phase, `voice_env` ramps from current level to 0.0 over `release_time * sample_rate` samples,
after which the voice enters `VOICE_SILENT` state.

---

### PARAM_MIX (ID: 3)

| Property | Value |
|----------|-------|
| **CLAP ID** | `3` |
| **Name** | `"Mix"` |
| **Module** | `"SSM"` |
| **Min / Max** | `0.0` / `1.0` |
| **Default** | `0.5` |
| **Curve** | Linear |
| **Flags** | `CLAP_PARAM_IS_AUTOMATABLE` |

**Text display:** `"%.0f%%"` — displayed as percentage of full wet (`value * 100`).

**Text input:** Accepts percentage (`"75%"`, `"50%"`), raw decimal (`"0.75"`), or number > 1
treated as percent. Clamped to `[0.0, 1.0]`.

**DSP usage:** Blends between dry input and wet SSM output.
`output = (1 - mix) * dry + mix * wet`.

---

### PARAM_BYPASS (ID: 4)

| Property | Value |
|----------|-------|
| **CLAP ID** | `4` |
| **Name** | `"Bypass"` |
| **Module** | `""` (root) |
| **Min / Max** | `0.0` / `1.0` |
| **Default** | `0.0` |
| **Curve** | Stepped |
| **Flags** | `CLAP_PARAM_IS_AUTOMATABLE \| CLAP_PARAM_IS_STEPPED \| CLAP_PARAM_IS_BYPASS` |

**Text display:** `"On"` (value > 0.5) or `"Off"` (value ≤ 0.5).

**Text input:** Accepts `"On"`/`"Off"` (case-insensitive first letter), or numeric parsing.
Numeric values > 0.5 → On, ≤ 0.5 → Off.

**DSP usage:** 4-state bypass crossfade state machine:
1. `BYPASS_PROCESSING` — full wet signal
2. `BYPASS_FADING_OUT` — crossfade from wet to dry over ~64 samples (rate `1.0/64.0`)
3. `BYPASS_BYPASSED` — full dry passthrough
4. `BYPASS_FADING_IN` — crossfade from dry back to wet

---

## Parameter Smoothing

All automatable parameters use a one-pole smoothing filter applied on the audio thread
during `process()`. The coefficient is precomputed at activation time:

```
coeff = exp(-1.0 / (smooth_time_s * sample_rate))
```

The smoothed value per-sample:
```
current = current + coeff * (target - current)
```

Where `target` is the latest automation value (from the SPSC ring buffer) and `current`
is the smoothed value used by the DSP. Both are `std::atomic<double>` aligned to 64 bytes.

| Parameter | Smooth Time | Typical Coefficient at 48kHz |
|-----------|-------------|------------------------------|
| Gain | 10 ms | 0.9979 |
| Attack | 10 ms | 0.9979 |
| Release | 10 ms | 0.9979 |
| Mix | 10 ms | 0.9979 |
| Bypass | 10 ms | 0.9979 |

---

## Host Integration Notes

- **Parameter count:** `PARAM_COUNT = 5`
- **`params_flush()`** handles `CLAP_EVENT_PARAM_VALUE` events. Modulation events
  (`CLAP_EVENT_PARAM_MOD`) are accepted but ignored (not supported in current phase).
- **Wrong-namespace filtering:** Events with `space_id != CLAP_CORE_EVENT_SPACE_ID` are
  silently ignored (per CLAP 1.0 §4.10 compliance).
- **Direct atomic write:** During `flush()`, parameter values are written both to the
  SPSC ring buffer (for the audio thread) and directly to atomics (for `get_value()` reads
  when the plugin is not active).
- **State persistence:** All parameter values are saved/restored via
  `CLAP_EXT_STATE` (binary serialization).
