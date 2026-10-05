# EmT refactoring plan

Mechanical, behavior-preserving restructuring to reduce duplication and make
future work (deviations D1/D2, ARA SDK, new parameters) land in one place.
Each item lists its approach, its impact, and the guardrails that prove it
changed nothing observable.

Status legend: ✅ done · ⬜ planned · 🔵 in progress.

## Phase 0 — zero behavior change (✅ complete)

| # | Item | Commit | Change | Guardrail |
|---|---|---|---|---|
| 5 | Single-source weight-format module | `92ab711` | New `scripts/emt_format.py` (header consts, `MAX_*` strides, `floats_per_layer`, `make_header`, `parse_header`); exporter, validator and both NumPy references import it | Parity vectors regenerate **byte-identical**; `validate_export` passes; `reference_forward` smoke OK |
| 7 | Named `LoadError` codes | `692c28d` | `enum class LoadError` replaces 12+ magic ints; `load()` values locked by `static_assert`; accessor failures get their own `-20`… range; `dequantize_layer_to` propagates the underlying error | 72/72 ctest; tripwire D3 pins the name **and** the stable `-16` |
| 11 | Shared test fixtures | `cd5bbd0` | New `tests/fixtures/weight_fixture.h` (`TempWeightFile`, `WeightFileSpec`/`write_weight_file`, layout math); `test_weight_loader.cpp` migrated; `test_fixtures` INTERFACE lib | `test_weight_loader` 23/23; full ctest 72/72 |
| 12 | Per-target CMake hygiene | `c8cd020` | RT flags PRIVATE on `core`/`ara_plugin`/`mamba_synth`; tests/GTest keep exceptions; no global includes; `emt_add_test()` helper; `EMT_USE_SYSTEM_GTEST`; real `clap-validator` test; LF + `.editorconfig` + `.gitattributes` | From-scratch configure+build; flag scoping verified in `build.ninja`; 72/72 ctest; benchmark gate PASS |
| 13 | Docs consolidation | `94a7cb5` | Superseded reports → `docs/archive/`; new `docs/README.md` index + `docs/BUILD.md` guide; README pointer + test-count fixes (64/67 → 72) | Link check on moved files; no content changes to ledgers |

Phase 0 verification (all green at `94a7cb5`):

```bash
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build build
python3 scripts/reference_mamba3.py --generate --output-dir build
export MAMBA_PARITY_DIR=$PWD/build MAMBA_REQUIRE_PARITY=1 \
    MAMBA_TEST_WEIGHTS=$PWD/models/ci_test_model.bin
ctest --test-dir build --output-on-failure        # 72/72
python3 scripts/verify_deviations.py --build-dir build  # 4 closed, 2 open, 0 failing
```

## Phase 1 — structural, still behavior-preserving (🔵 #2 done, rest planned)

Do in this order: #2 before #3 (presets index by `ParamIds`); #8 before #1
(shared DSP helpers shrink the plugin-core merge).

| # | Item | Approach | Impact |
|---|---|---|---|
| 2 | ✅ Data-driven parameter table | Shipped (`21a4cd1`: `src/clap/param_table.h/.cpp`): `constexpr ParamDesc` tables (id, name, module, min/max/default, curve, smooth time, flags, `ParamFormat` tag) consumed by both params extensions, all three `init_param_metadata` copies, GUI labels, and preset validation; `static_assert` id alignment + defaults-in-range + FX row count; arity-checked `make_synth_preset_values()` | Adding a parameter is one table row; per-plugin text quirks preserved as format tags (O7); 90/90 ctest (18 new pins) |
| 3 | ✅ Split the presets monolith | Shipped (`HASH3`): `clap_presets.cpp` (984 L) → `preset_bank.cpp` (factory data, 139 L) + `preset_codec.cpp` (JSON/binary, 761 L) + `preset_paths.cpp` (platform dirs, 108 L); pure code motion, `clap_presets.h` umbrella unchanged | Direct codec round-trip tests already existed and cover the split; 90/90 ctest, 0 skips |
| 4 | Split GUI by platform backend | `clap_gui.cpp` (861 L, ~15 Win32/X11 `#if`s) → `gui_common.cpp` + `gui_win32.cpp` + `gui_x11.cpp` behind an `IGuiBackend` seam; graceful Cocoa stub for macOS (currently silent `false`) | Backends build/reason independently; macOS support becomes additive |
| 8 | Extract RT DSP helpers | Header-only `src/dsp/`: `param_smoother.h`, `bypass_fader.h`, `voice_envelope.h`, each with a no-alloc contract + regression vectors | Synth and FX provably share smoothing/bypass behavior; reusable for the ARA offline renderer |
| 1 | Shared Synth/FX plugin core | Extract `PluginCommon` (engine handle, smoothing array, bypass FSM, weight loader, thread-check); `MambaSynth`/`FxState` embed it and add only voices vs stereo-through | Halves the CLAP maintenance surface; stops the already-visible synth/FX drift (`FxState` lacks `param_queue`, conditioning seqlocks, state mirror) |
| 9 | ARA seam: interface + injected backend | `IARABackend` implemented by `NullARABackend` (today's stub); plugin holds a pointer instead of reaching the `globalDocController()` singleton | Real-SDK integration becomes purely additive; removes global mutable state from the plugin path |

Per-item guardrails: full ctest green, `verify_deviations.py` green, and
`benchmark_ssm_block --budget-pct` before/after for anything touching
`process_sample`/`process()`.

## Phase 2 — behavior-adjacent, benchmark-gated (⬜ planned)

| # | Item | Approach | Impact |
|---|---|---|---|
| 10 | Recurrence kernel extraction (D1/D2 prep) | Lift the scalar recurrence out of `SSMBlock::step_recurrence` into `recurrence.h` as a free function over explicit state slices; D1/D2 land as new kernels behind a flag with parity tests + benchmark comparison | De-risks the two remaining OPEN deviations; creates the seam for SIMD/block recurrence |
| 6 | Merge the two NumPy references | `emt_model/` package: `format` (done in #5) + one `reference.py` forward + `torch_model.py` imported by exporter and trainer; old entry points become thin CLI shims, then deleted | One executable spec of the forward pass instead of three; training/inference drift becomes structurally harder |

Plus the event-scheduler extraction (sorted events → sub-block boundaries as
a host-independent, unit-testable `event_scheduler.h`): the actual fix for
automation-heavy sessions, testable without a CLAP host.

## Observations (found during Phase 0, not yet scheduled)

- **O1 — `clap_fx_plugin.cpp` param one-liners** ✅ resolved in #2: the
  `value_to_text`/`text_to_value` chains became table-dispatched calls and the
  sibling `fx_params_flush` one-liner was split; `-Wmisleading-indentation`
  is gone.
- **O2 — dead helpers in `clap_params.cpp`** ✅ resolved in #2:
  `log_knob_to_value` / `log_curve_to_knob` deleted (were `-Wunused-function`
  dead code); `PARAMETER_REFERENCE.md` §3.4 rewritten to describe the actual
  host-side knob mapping instead of the dead helpers' math.
- **O3 — `mamba_getpid` still duplicated** in `tests/test_conditioning.cpp`.
  Migrate it (and later suites) to `tests/fixtures/weight_fixture.h`.
- **O4 — CRLF sources**: 6 files under `src/` still use CRLF
  (`ara_editor_renderer.cpp`, `clap_audio_ports.cpp`, `clap_params.cpp`,
  `clap_plugin.cpp`, `clap_state.cpp`, `clap_tail.cpp`). `.gitattributes`
  (`text=auto`, added in #12) normalizes them on next touch; do not
  mass-convert (blame pollution).
- **O5 — doc test-count drift**: README quoted 64/67 while ctest ran 72.
  Fixed in #13; the claim policy (`ARCHITECTURE_FIDELITY.md` §1) now covers
  funnels like this via `docs/BUILD.md`.
- **O6 — accessor error propagation** (shipped in #7): `dequantize_layer_to`
  now returns the underlying accessor code instead of collapsing to `-3`/`-4`.
  Undocumented values, tests only asserted `0`/non-`0` — safe, but noted here
  for the record.
- **O7 — `text_to_value` quirks (found in #2, pinned, not fixed)**: FX bypass
  parsing is case-sensitive and never fails (`BypassFx` tag); FX gain/mix
  parse plain decimals with no suffix handling and no clamping; the synth
  bypass display uses strict `>` while FX uses `>=`; synth parsers have no
  `isfinite` validation (`"nan"` parses successfully); bare attack/release
  numbers mean milliseconds. All pinned by `tests/test_param_table.cpp`;
  unify deliberately in a behavior-change release.
- **O8 — FX `module` strings empty as-shipped** (found in #2): the FX
  `get_info` never sets `info->module`, so FX Gain/Mix report `""` where the
  synth reports `"SSM"`. Preserved verbatim; `PARAMETER_REFERENCE.md` now
  documents the shipped behavior. If a host-visible regrouping is acceptable,
  set the two FX table modules to `"SSM"` and revert the doc note.
