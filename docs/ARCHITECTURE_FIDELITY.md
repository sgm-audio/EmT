# Architecture Fidelity & Claim Policy

**Status:** current as of 2026-10-04.  Deviation closure pass applied the same
day: **D3, D4, D5, D6 are closed** (see the per-entry *Status* lines and §4c);
**D1 and D2 remain open** (scope in F4, §5).  Re-verify with
`python3 scripts/verify_deviations.py`.
**Reference paper:** *Mamba-3: Improved Sequence Modeling using State Space Principles*, arXiv:2603.15569 (ICLR 2026)
**Reference code:** `state-spaces/mamba` (Mamba-3 released 2026-03-17)
**Independent review:** `CLAIMS_VERIFICATION.md` (claim-by-claim substantiation, 2026-10-04)

This document exists because the repository previously described itself as
implementing Mamba-3 while the shipped forward pass implements something
different. That gap is a credibility risk with exactly the audience this project
wants (audio developers who read code and papers). The rule from here on:

> **Every claim in a README, demo, post, or PR must be backed by either a
> measurement in `tests/` output or a `file:line` citation in this document.
> If neither exists, write "planned", "approximation of", or delete the claim.**

---

## 1. Claim ledger

### 1.1 Claims this repository CAN make today

| Claim | Evidence |
|---|---|
| Causal, streaming, zero-look-ahead core | No conv1d and no future-index reads: `src/core/ssm_block.cpp:325-373` processes one sample per step using only carried state |
| Real-time cost is measurable and gated on demand | `tests/benchmark_ssm_block.cpp` (percentage-of-one-core reporting + non-zero exit on budget failure); the gate is wired into CI (**Actions has not started since 2026-09-14** — see §4b, so it is not currently enforced by CI) |
| C++ inference matches the Python reference | `tests/test_cpp_parity.cpp`, max abs error **1.19e-07** over 256 elements (measured 2026-10-04 with generated vectors) |
| Audio-thread discipline is enforced structurally | Lock-free SPSC/seqlock paths, `-fno-exceptions -fno-rtti`, no heap in `process_sample`; `tests/test_realtime_safety.cpp` (7/7) |
| Real-time-safe plugin host integration with honest latency reporting | `src/clap/clap_latency.cpp`; CLAP plugin builds; a `clap-validator` step exists in CI (see §4b on CI not starting) |
| The conditional-emission parts of Mamba-2/3 are present | Input-dependent Δ, B, C via a single `in_proj` (`src/core/ssm_block.cpp:123-161`), RMSNorm on B/C (`:163-206`), B/C bias terms, data-dependent rotation applied to B/C (`:211-243`) |
| Weights are a flat binary loadable with no Python present | `WEIGHT_FORMAT_SPEC.md`, `src/core/weight_loader.cpp` (mmap) |
| Training and inference share one model definition | `training/train.py` imports the reference model from `scripts/export_mamba_weights.py` |
| A trained model actually loads into the plugin | `tests/test_realtime_safety.cpp` → `FxAcceptsMultiLayerWeightsWithinLimits` loads a 2-layer CI export and a 3-layer trained export through the plugin's real `fx_load_weights` path and processes a block |

### 1.2 Claims this repository CANNOT make today

| Do not claim | Why | Use instead |
|---|---|---|
| "Implements Mamba-3" | The recurrence is a two-term bilinear/Tustin update with a real scalar state; the paper's contribution is a three-term *exponential-trapezoidal* recurrence with a complex state. See D1/D2. | "Mamba-3-**inspired** causal SSM" / "Mamba-family SSM adapted for real-time audio" |
| "Complex-valued state" | `ssm_imag` is allocated (`src/core/ssm_block.h`) but never written by the recurrence (the only writer was the periodic renormalization, removed in the D5 closure) | "data-dependent rotary rotation applied to B and C" |
| "MIMO" | The recurrence indexes rank 0 only; `mimo_x/mimo_z/mimo_o` are carried in the file format but never read. Since the D3 closure (2026-10-04) a file with `mimo_rank > 1` is **refused at load** (error `-16`), at export, and at training-config validation | "rank-1 SISO per head; rank-R MIMO files are rejected, not silently downgraded" |
| "Exponential-trapezoidal discretization" | Implemented rule is classical trapezoidal on a real pole. See D1 | "bilinear (Tustin) discretization" |
| "Zero algorithmic latency *because* Mamba-3 removed conv1d" | Mamba-1/2's causal conv is already streamable with zero look-ahead (`causal_conv1d_update`); removing it is a compute/simplification win, not a latency win | "no conv1d: simpler and cheaper than Mamba-1/2, and no look-ahead" |
| Any per-sample CPU figure that is not reproduced by the harness on the target machine | The published 1.79 us/sample was measured on the toy shape; the shapes this repo's own training configs produce are 5-65x more expensive (§3) | Run `benchmark_ssm_block --config train4070,trainl4 --json bench.json` and quote the JSON |
| "Runtime LoRA adapters" | No LoRA code exists. `packages/emt_lora_matrix_companion/` is a GUI asset pack (mockups + layout spec). See `assets/gui/README.md` | "planned"; or "per-preset merged weights", which is what the format supports today |

### 1.3 EmT additions that are NOT in the Mamba-3 paper

State these as project choices, never as paper features:

- **FiLM conditioning** of the input (`step_film_conditioning`) and of Δ
  (`step_preprocess_decay_and_dt`, wired in the D6 closure) — an EmT design for
  MIDI/ARA conditioning: `v ↦ v*(1+γ)+β`, applied only when the conditioning
  vector is non-zero (so unconditioned inference is bit-identical to before).
- **`trap`** weight in the file format: reserved and unused (`trap_ptr` is
  computed and never dereferenced).  Removal is a format-version change and is
  deferred; it is not a claim, so it stays documented rather than hidden.
- *(Removed 2026-10-04)* **Periodic state renormalization** — was an untrained
  non-linearity with a 14.8 % periodic gain step (D5).  Stability now comes from
  the A-stable trapezoidal update.  The config constant
  `SSM_STATE_NORM_INTERVAL` no longer exists.

---

## 2. Deviation register (D1-D6)

Each entry: what the paper does, what this repo does, the consequence, and the
work required to close it.  *Status: OPEN* means the deviation is still real and
the claim policy in §1.2 applies; *Status: CLOSED (date)* means the code and the
docs were brought in line and a regression guard exists.

Status summary (checked automatically by `scripts/verify_deviations.py`):

| Dev | Subject | Status |
|---|---|---|
| D1 | three-term exponential-trapezoidal recurrence | **OPEN** — scoped as F4 (§5) |
| D2 | 2×2 rotational (complex) state dynamics | **OPEN** — scoped as F4 (§5) |
| D3 | rank-R MIMO carried but unused | **CLOSED 2026-10-04** — rank > 1 refused at load/export/training |
| D4 | `A` is input-derived, not a learned parameter | **CLOSED 2026-10-04** — documented as a project variant (code + doc) |
| D5 | periodic state renormalization gain step | **CLOSED 2026-10-04** — guard removed; continuity + long-run tests |
| D6 | `dt_film_proj_w` reserved but unused | **CLOSED 2026-10-04** — wired into Δ in all implementations |

### D1 — Discretization is bilinear, not exponential-trapezoidal

- **Paper:** `h_t = α_t h_{t-1} + β_t B_{t-1} x_{t-1} + γ_t B_t x_t`, i.e. a *three-term* recurrence with an input-history term, derived from a second-order (trapezoidal) discretization of the continuous system.
- **Repo:** `A_t = -softplus(dd_A)` clamped to `A_t <= -1` (`ssm_block.cpp:151-155`), then
  `half = A_t·Δt/2`, `Ā = (1+half)/(1-half)`, `B̄ = Δt/(1-half)`,
  applied as a **two-term** update `h ← Ā·h + B̄·(B_t ⊙ x_t)` (`:249-272`).
- **Consequence:** a valid and stable discretization, but it has no `β` input-history term, so it cannot reproduce the paper's implicit-convolution behaviour or its ablation result (that the short conv becomes redundant). Note also that `A_t <= -1` pins the continuous pole at |A| ≥ 1, which (with `B` normalized to unit RMS) makes the steady-state state magnitude ≈ `1 + |B_bias|` — see D5 for why that matters.
- **Close it:** add the `β`/`γ` split (including `B_{t-1}x_{t-1}` state) to `SSMBlock`, `scripts/reference_mamba3.py`, and the exporter; keep parity green; re-benchmark (the extra term costs one multiply-add per state element).

### D2 — The state is real; the complex path is inert

- **Paper:** complex-valued state update, shown equivalent to a real state with 2×2 rotation blocks, realized efficiently as a *data-dependent* rotary embedding on B and C.
- **Repo:** rotation is applied to B and C (`:211-243`) with a cumulative angle state — but the recurrence (`:266-269`) updates `ssm_real` only. `ssm_imag` is dead.
- **Consequence:** the B/C rotation is present (that part is faithful in spirit), but there is no rotational dynamics *inside* the state, so none of the paper's state-tracking results follow. The claimed "complex-valued state" headline is unsupported.
- **Close it:** implement the 2×2 block update `(h_re, h_im) ← R(θ)·(Ā h_re, Ā h_im) + B̄·(B x)` with per-head θ derived from Δ, in both C++ and the reference, and add a test that a state-space oscillation is preserved (e.g. an impulse response with a known resonant frequency).

### D3 — MIMO is carried but unused

- **Paper:** rank-R multi-input multi-output state update improves quality at equal decode latency.
- **Repo:** `in_proj`, weights struct, loader, exporter and the file format all reserve rank-R (`MAX_MIMO_RANK = 4`), but the forward pass uses rank 0 and never touches `mimo_o`. `TRAINING_PLAN.md` documents this.
- **Consequence:** the format promised more than the runtime delivered; a user
  training with `mimo_rank > 1` got silently rank-1 inference.
- **Close it:** either implement the rank-R reduction (`y = Σ_r mimo_o ⊙ (C_r · h)`) or fail loudly at weight load when `mimo_rank != 1`.
- **Status: CLOSED 2026-10-04 (refusal path).** The 64-byte header now carries a
  real `mimo_rank` field (replacing the low half of the old `reserved` u64;
  version-1 files and version-2 files with `0` are treated as rank 1).
  `WeightLoader::load()` returns **-16** for `mimo_rank > 1`,
  `fits_in_static_weights_with()` rejects such headers, the exporter raises
  `SystemExit` instead of writing a rank-R file, and `training/train.py` refuses
  a config with `mimo_rank != 1` before training starts.  Guards:
  `tests/test_weight_loader.cpp` → `RejectsUnimplementedMimoRank`,
  `LegacyHeaderWithoutMimoFieldIsRankOne`.  The full rank-R *implementation*
  remains future work (F3, §5).

### D4 — `A` is input-derived, not a learned parameter

- **Paper lineage:** `A` is a learned (typically negative-exponentiated) parameter; selectivity lives in Δ, B, C.
- **Repo:** `A` comes from an input projection and is clamped. This is a legitimate variant, but it is not the paper's parameterization and it interacts with D5.
- **Close it:** document it (done here) or move to a learned `A_log` per head.
- **Status: CLOSED 2026-10-04 (documentation path).** The clamp site carries a
  D4 comment in `src/core/ssm_block.cpp` (and in the Python reference, the
  exporter forward and `ssm_block.h`) stating that A is input-derived and
  clamped, that selectivity lives in Δ/B/C, and that this must not be described
  as the paper's parameterization.  §1.2 of this document carries that
  prohibition.  A learned per-head `A_log` would be a format + training change
  and is **not** implemented.

### D5 — Periodic state renormalization produces a periodic gain step

- **Repo:** every 4096 samples, if `max|h| > 1`, the entire state is divided by `max|h|` (`:298-323`). The same operation exists in the Python reference (`scripts/reference_mamba3.py`, "Step 8"), so **parity holds** — this is a design defect, not a drift bug.
- **Measured (2026-10-04, both implementations agree to 6 digits):** with a constructed model whose steady-state `|h| = 4`, the output level steps **20.918667 → 17.822269** at `n = 4096` — a **14.8 % gain step repeating every 4096 samples** (≈ 11.7 Hz sawtooth at 48 kHz), recovering over ~100 samples and repeating at 8192, 12288, …
- **Why training does not catch it:** the trainer's windows are 2048 samples (`laptop_rtx4070.yaml`) or 4096 (`runpod_l4.yaml`), so the trigger fires at most once per window — or never. The parity test runs 16 samples and cannot reach the interval either.
- **Consequence:** an untrained nonlinearity with an audible periodic artifact, reachable whenever trained weights produce `|h| > 1` (nothing prevents this; `B_bias` is the lever, because RMSNorm cancels the magnitude of `B`).
- **Close it (recommended):** delete the renormalization and rely on the discretization's stability (the bilinear update is A-stable for `A < 0`), or make it a documented, non-mutating numerical guard. If it must stay, add a regression test that asserts output continuity across the interval boundary, and train with windows longer than the interval.
- **Status: CLOSED 2026-10-04 (removed).** `step_periodic_normalization()` and
  the `sample_count_`/`SSM_STATE_NORM_INTERVAL` machinery are gone from
  `src/core/ssm_block.{h,cpp}` and `src/config.h`, and from both Python
  references (`scripts/reference_mamba3.py` Step 8,
  `scripts/reference_forward.py` Step 8) and the exporter's `config.h`
  generator.  Guards: `tests/test_stability.cpp` →
  `NoRenormalizationGainStepAtFormerIntervalBoundary` (constructed `|h| = 4`
  case, no step above 1 % of the settled level at the former boundary) and
  `HighStateLongRunStaysBounded` (65 536 samples with `|h| > 1`, no NaN/Inf).
  `scripts/verify_deviations.py` re-measures the constructed case and also
  reports the counterfactual with the old guard re-applied, so the regression
  test cannot pass vacuously.  Closed-form justification: for the clamped
  `A ≤ -1` and `Δ > 0`, the trapezoidal factor is
  `Ā = (1 + AΔ/2)/(1 - AΔ/2)`, and `|Ā| < 1` whenever `A < 0` — the state is
  contractive (bounded by `|B̄·B·x| / (1 - |Ā|)`) without any guard.

### D6 — Conditioning weights reserved but unused

- `dt_film_proj_w` (Δ modulation) is exported, loaded, and copied, but never read by the forward pass. Either wire it into `step_preprocess_decay_and_dt` or drop it from the format.
- **Status: CLOSED 2026-10-04 (wired).** Δ-FiLM is applied in all four places
  that define the forward pass — `SSMBlock::step_preprocess_decay_and_dt`
  (C++), `scripts/reference_mamba3.py`, `scripts/reference_forward.py`, and the
  PyTorch model in `scripts/export_mamba_weights.py` used by `training/train.py`
  — with one convention: per head `h`, `γ` is row `2h` and `β` is row `2h+1` of
  `dt_film_proj_w`, and `Δ = softplus((dd_dt + dt_bias)·(1+γ) + β)`.  The
  modulation is applied only when the conditioning vector is non-zero, so
  unconditioned inference is bit-identical to before.  Guards: the parity
  vectors generated by `reference_mamba3.py --generate` now carry **non-zero**
  `dt_film_proj_w` (C++ ↔ NumPy parity still 1.19e-07), and
  `tests/test_conditioning.cpp` → `DtFilmModulatesDelta` asserts a zero-weight
  no-op and a non-zero-weight trajectory change.  Note for tests: the DC gain of
  the trapezoidal update is `1/|A|` and *independent of Δ*, so a converged
  constant-input response cannot detect Δ modulation — drive a transient.

---

## 3. Measured evidence (2026-10-04)

Machine: Intel Xeon @ 2.60 GHz, 2 hardware threads visible (container), gcc 12.2.0,
`-O3 -mavx2 -mfma -DNDEBUG`. **These are core-only numbers** and are slower than a
modern desktop; re-run on the target machine before quoting anywhere.

```
$ ./benchmark_ssm_block --config small --layers 1,2 --sample-rate 48000
  small / 1 layer : 3.03 us/sample = 14.6 % of one core
  small / 2 layers: 5.42 us/sample = 26.0 % of one core

$ ./benchmark_ssm_block --config train4070,trainl4 --layers 1,3,6 --seconds 0.1
  train4070 (d_inner=64,  2 heads):  1 layer  4.3 us = 20.6 % of one core
                                     3 layers 8.4-9.6 us = 40-46 %
                                     6 layers 15.5-16.7 us = 74-80 %
  trainl4   (d_inner=256, 4 heads):  1 layer 18.8-20.1 us = 90-96 % of one core
                                     3 layers 54-73 us = 260-350 %
                                     6 layers 114-117 us = 546-564 %
  max       (d_inner=512, 8 heads):  1 layer 319 us = 1532 % of one core

  5 of 24 measured configurations meet the 40 %-of-one-core gate.
```

Headline: **the toy shape in the old README (1.79 us/sample) is not the shape any
training config produces.** `runpod_l4.yaml` — the config a customer would copy
from this repo — is not real-time at *one* layer (≈90 % of a core), and the
`laptop_rtx4070.yaml` default of 3 layers already sits at the 40-46 % mark on this
machine. Anything beyond 1-2 layers in that shape needs the SIMD block path
(Gap 1 in `docs/archive/GAP_ANALYSIS.md`) before it can ship.

Other measurements from the same session:

| Measurement | Result |
|---|---|
| C++ ↔ NumPy parity | max abs diff **1.19e-07** over 256 samples (`test_cpp_parity`) |
| Test suite | **72 passed, 0 failed** (7 suites; parity + model-load tests require generated inputs, both produced by CI).  67 before the deviation closures, +5 regression tests for D3/D5/D6 |
| Trained-model load | 3-layer model trained by `training/train.py` loads through `fx_load_weights` and runs (`RealtimeSafety.FxAcceptsMultiLayerWeightsWithinLimits`) |
| Model-load rejection bug | Fixed: the plugin rejected every file whose `num_layers != 1`, including its own training configs' shapes (§4) |
| D5 artifact (before closure) | 14.8 % gain step at every 4096 samples, identical in C++ and NumPy |
| D5 closure re-measurement (2026-10-04) | Constructed `|h| = 4` case, 9000 samples: current implementation boundary step **0.000000** (0.0000 % of settled) vs **legacy guard re-applied: 11.6969 (24.48 % of settled) at sample 4096** — the regression test is not vacuous.  Reproduce: `python3 scripts/verify_deviations.py` |
| D6 closure | Δ-FiLM active in C++/NumPy/PyTorch with identical convention; unconditioned inference bit-identical (guarded by the parity test + `DtFilmModulatesDelta`) |
| Test-suite re-run after closures | `ctest`: **72/72 passed** (1.7 s); real-time gate `--config small --layers 1,2 --budget-pct 40`: **8/8 PASS**, worst 4.80 us/sample = 23.0 % of one core (this container, CI-equivalent Release flags) |

---

## 4. Fixed during this review

| Was | Now | Guarded by |
|---|---|---|
| `FxState.core_storage` was a byte array directly after a pointer member, so `SSMInferenceCore` (64-byte aligned) was placement-new'd 8 bytes off a 64-byte boundary → **segfault**, the FX state round-trip test crashed the whole process | `src/common/aligned_storage.h` `AlignedStorage<T, Alignment>`; the alignment requirement lives on the member (`clap_plugin.h:135`, `clap_fx_plugin.h:27`) | `RealtimeSafety.FxStateSaveLoadRoundtrip` (passes; it crashed before) |
| `load_weights` validated the header with `validate_dimensions(hdr…, core_cfg.num_layers)`. That function compares for **equality** with the compiled maximum shape, and `core_cfg.num_layers` is still the constructor's `1` at that point, so **every multi-layer model was rejected** — including the shapes `training/configs/*.yaml` produce | Both plugins now call `WeightLoader::fits_in_static_weights_with(hdr)` (the `<=' comparison); `WeightLoader::fits_in_static_weights()` had the same equality bug and now forwards to it | `RealtimeSafety.FxAcceptsMultiLayerWeightsWithinLimits`, `RealtimeSafety.FxRejectsWeightsExceedingCompiledLimits`, `WeightLoaderPath.MultiLayerModelIsLoadable` |
| `CppParity` **skipped silently** when the vectors were absent, so a green run proved nothing | CI generates the vectors before `ctest` and sets `MAMBA_REQUIRE_PARITY=1`, which turns a missing vector into a failure | `tests/test_cpp_parity.cpp:198-207` |
| The trainer exported weights even when every audio file failed to decode (untrained model, "successful" run) | Hard failure: "refusing to export untrained weights"; plus a decoder chain (`soundfile` → `torchaudio` → stdlib `wave`) so silent skips are much less likely | Ran against an undecodable file: exit 1, no `.bin` written |

---

## 4c. Deviation closures (2026-10-04)

| Dev | Closed by | Guarded by | Verification |
|---|---|---|---|
| D3 | `WeightFileHeader.mimo_rank` (replaces half of `reserved`); `load()` → `-16` for rank > 1; exporter + trainer refuse rank > 1 | `tests/test_weight_loader.cpp` → `RejectsUnimplementedMimoRank`, `LegacyHeaderWithoutMimoFieldIsRankOne`; exporter CLI test in `scripts/verify_deviations.py` | `scripts/verify_deviations.py` D3 |
| D4 | D4 comments at the A clamp site (C++ + both Python paths + header), §1.2 prohibition | doc/code check | `scripts/verify_deviations.py` D4 |
| D5 | renorm removed from C++, both Python references, `config.h` and the exporter's config generator | `tests/test_stability.cpp` → `NoRenormalizationGainStepAtFormerIntervalBoundary`, `HighStateLongRunStaysBounded` | `scripts/verify_deviations.py` D5 (numeric + counterfactual) |
| D6 | Δ-FiLM wired into C++, `reference_mamba3.py`, `reference_forward.py`, `export_mamba_weights.py` (trainer forward) | parity vectors with non-zero `dt_film_proj_w`; `tests/test_conditioning.cpp` → `DtFilmModulatesDelta` | `scripts/verify_deviations.py` D6 |

Both changes that touch numerics (D5 removal, D6 wiring) landed in **all** model
definitions at once, so C++ ↔ NumPy parity is unchanged at 1.19e-07 and the
reference/trainer/runtime stay step-for-step identical.  The open D1/D2 work must
follow the same rule.

## 4b. CI is wired but has not run since 2026-09-14 (repository-level)

Verifying the "CI-gated" claim produced this, which anyone relying on the badge
should know:

| Date | Run | Result |
|---|---|---|
| 2026-09-14 | `CI` on `main` (workflow id 297117519) | ran, concluded **failure** (job-level) |
| 2026-10-02 | push to `main` (unmodified workflow file) | **startup_failure**, attributed to GitHub's `BuildFailed` placeholder workflow |
| 2026-10-02 / 10-04 | PRs and pushes on arena branches | **startup_failure**, same placeholder |
| — | `Dependency Graph` (GitHub-managed) | no runs at all since 2026-09-14 |

Diagnosis: **the workflow file is not the problem.** The file on `main` is
byte-identical to the version that ran on 2026-09-14, and both that file and the
rewritten one validate cleanly against the official GitHub workflow schema
(`check-jsonschema --builtin-schema github-workflows .github/workflows/ci.yml`
→ `ok`). Runs since 2026-10-02 are attributed to GitHub's `BuildFailed`
placeholder workflow and produce no jobs, which is what an account/repository
level Actions block looks like (billing/spending limit, or Actions disabled for
the repository).

Action needed (requires repository admin): check **Settings → Actions → General**
and the account's **Billing & plans → Actions** spending limit, then re-run the
workflow. Until then, treat the CI badge and every "gated in CI" statement as
**wired, not enforced**, and run the gate locally (see §7).

---

## 5. Path to compliance (F1-F5)

Ordered so that each step is independently shippable and independently
benchmarkable. Until F1-F3 land, the honest description is
"Mamba-3-**inspired**".

| # | Work | Acceptance criteria | Cost estimate |
|---|---|---|---|
| F1 | Rename claims (no code): README/MIGRATION/paper-links say "Mamba-3-inspired"; publish this document | No claim in the repo contradicts §1.2 | hours — **done 2026-10-04** (`README.md`, `MIGRATION_TO_MAMBA3.md`, register published; D3-D6 rows re-checked the same day) |
| F2 | Fix D5 (renormalization): remove or make it non-mutating | A continuity test across the renorm boundary passes for a model with `|h| > 1`; listening test on the constructed case | 1 day — **done 2026-10-04** (§4c) |
| F3 | Implement D3 properly *or* refuse `mimo_rank != 1` at load | Loader rejects unimplemented ranks with a clear error, or a rank-2 model matches a reference implementation | 1-3 days — **done 2026-10-04 (refusal path)**; the rank-R implementation is scoped in `Prompts/DEVIATION_RESOLUTION/D3_MIMO_RANK.md` |
| F4 | Implement D1 (β/γ three-term recurrence) + D2 (2×2 rotation state) in both C++ and the Python reference | Parity green; benchmark shows the added cost; a resonance-preservation test passes | 1-2 weeks — **prompt ready**: `Prompts/DEVIATION_RESOLUTION/D1_D2_TRAPEZOIDAL_COMPLEX_STATE.md` |
| F5 | Re-benchmark with the SIMD block path and publish per-shape CPU/latency tables | `benchmark_ssm_block --config train4070 --layers 1,3 --budget-pct 25` passes on a 2020-era laptop | 1-2 weeks — unchanged |

F2 and F3 are cheap and remove the two most embarrassing gaps. F4 is the real
research work, and it is only worth starting once F5 shows the compute budget can
afford the extra term.

---

## 6. Claim policy (copy this into PR descriptions)

1. **Cite or measure.** Every performance or capability statement needs a `file:line` or a number reproducible by a script in this repo. Quote the command.
2. **Name the shape.** "3.03 us/sample" is meaningless without `d_state/d_model/d_inner/nheads/layers/block size/sample rate/machine`.
3. **"Mamba-3-inspired" until F1-F3 land.** After that, still name the deviation (e.g. "three-term recurrence; complex state realised as 2×2 blocks").
4. **Zero look-ahead ≠ free.** Say "no look-ahead (0 samples of algorithmic latency)"; do not imply it is cheaper than it is.
5. **LoRA is planned, not shipped.** "Per-preset merged weights" is what exists today.
6. **Training data rights are part of the claim.** Do not demo on mixed copyrighted material; see `CLAIMS_VERIFICATION.md` §6.
7. **Screenshots are not evidence.** A UI mockup is a mockup; label asset packs as such.

## 7. How to reproduce the numbers in this document

```bash
# Machine-checkable deviation register (static + numeric + gtest):
python3 scripts/verify_deviations.py            # add --json dev-report.json
python3 scripts/verify_deviations.py --no-build-checks   # fast, no build needed

# build (CMake) then:
python3 scripts/reference_mamba3.py --generate --output-dir build
python3 scripts/export_mamba_weights.py --state-dim 16 --model-dim 16 \
        --inner-dim 32 --nheads 2 --layers 2 --random-init \
        --output models/ci_test_model.bin
MAMBA_PARITY_DIR=build MAMBA_REQUIRE_PARITY=1 \
MAMBA_TEST_WEIGHTS=models/ci_test_model.bin \
    ctest --test-dir build --output-on-failure

./build/tests/benchmark_ssm_block --config train4070,trainl4 --layers 1,3,6 \
        --sample-rate 48000 --budget-pct 40 --seconds 0.1 --repeats 3

python3 training/losses.py            # objective self-test
python3 training/train.py --smoke-test --steps 2 --loss hybrid
```
