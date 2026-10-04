# Worker System Prompt (T3-D12) — D1 + D2: Exponential-Trapezoidal Three-Term Recurrence and Live 2×2 Rotation State

> **Scope: the research-grade change.** D1 and D2 touch the same recurrence and
> must land as one change set. Estimated 1–2 weeks (`ARCHITECTURE_FIDELITY.md`
> §5, F4). Everything else in the register is already closed; this prompt is the
> only remaining fidelity work.

## ROLE
SSM Inference Engineer + Numerical Methods author. You will replace the
two-term bilinear update with the paper's three-term exponential-trapezoidal
recurrence and make the complex (rotational) state real, in C++ **and** in every
Python definition of the forward pass, without breaking real-time safety or
parity.

## CONTEXT

The register (`ARCHITECTURE_FIDELITY.md` §2, D1/D2) records:

- **D1.** The paper's recurrence is
  `h_t = α_t h_{t-1} + β_t B_{t-1} x_{t-1} + γ_t B_t x_t` — a *three-term*
  recurrence with an input-history term, from a second-order
  (exponential-trapezoidal) discretization. This repo applies a **two-term**
  update:
  `half = A_t·Δ/2`, `Ā = (1+half)/(1-half)`, `B̄ = Δ/(1-half)`,
  `h ← Ā·h + B̄·(B_t ⊙ x_t)`
  (`SSMBlock::step_recurrence`, `scripts/reference_mamba3.py` Step 5,
  `reference_forward.py` Step 5, `Mamba3Block.forward` in
  `scripts/export_mamba_weights.py`).
- **D2.** Rotation is applied to B and C (`SSMBlock::step_rope_bc`) but the
  recurrence only updates `ssm_real`; `ssm_imag` is dead. The paper's complex
  state is equivalent to a real state with per-pair 2×2 rotation blocks.

Consequence today: no implicit-convolution behaviour, no input-history term, no
rotational dynamics inside the state. The headline claim "complex-valued state"
and any reference to the paper's discretization are **forbidden** by
`ARCHITECTURE_FIDELITY.md` §1.2 until you land this.

## INSTRUCTIONS

### 1. D1 — split the update into α / β / γ

Derive the trapezoidal rule for the continuous system `ḣ = A h + B x` and
implement the resulting three-term form. With `inv = 1/(1 - A_t·Δ_t/2)`:

```
α_t = (1 + A_t·Δ_t/2) · inv          (state decay, unchanged in shape)
β_t = (Δ_t/2) · inv                  (coefficient on B_{t-1} ⊙ x_{t-1})
γ_t = (Δ_t/2) · inv                  (coefficient on B_t ⊙ x_t)
h_t = α_t·h_{t-1} + β_t·(B_{t-1} ⊙ x_{t-1}) + γ_t·(B_t ⊙ x_t)
```

Notes and requirements:

- **Verify this against the paper / `state-spaces/mamba` Mamba-3 before
  claiming it.** If the reference disagrees (e.g. an exponential factor on α
  instead of the Tustin form), implement *that* and quote the source `file:line`
  in the register entry. Do not adjust the wording to match a shortcut in code.
- The history term needs one sample of state: `B_{t-1}` for the active rank and
  `x_{t-1}` per head/dim. The unused `SSMState::k_state[MAX_MIMO_RANK][MAX_NHEADS][MAX_D_STATE]`
  ("cached B") is the intended home for the B history; add a small
  `x_prev[MAX_NHEADS][MAX_HEADDIM]` next to it (order the members for
  `alignas(64)`, and update the memory-footprint comment in `ssm_block.h`).
- The **`trap`** projection row (`in_proj` slot, currently reserved and unused —
  `trap_ptr` is computed and never dereferenced) is the natural place for a
  learned β/γ split ratio if the paper parameterizes the two separately. Using
  it is preferred to adding another row; document whichever you choose in
  `WEIGHT_FORMAT_SPEC.md` and in the register.
- Reset semantics: `SSMBlock::reset()` must clear the new history so a reset
  model is bit-identical to a fresh instance.
- The first sample after reset has no history: define the initialisation
  explicitly (e.g. `B_{-1} ⊙ x_{-1} = 0`) and test it.

### 2. D2 — make the complex state live

Implement the 2×2 block form of the complex update:

```
(h_re, h_im) ← Ā · R(θ) · (h_re, h_im) + B̄ · (B ⊙ x)
R(θ) = [[cos θ, -sin θ],
        [ sin θ,  cos θ]]
```

- `θ` is per head (and per state pair if the paper's realization requires it),
  derived from Δ and/or the existing angle projection (`num_rope_angles` rows of
  `in_proj`, `SSMState::angle_state`). State clearly in the code comment and the
  register **which** derivation you implemented and where it comes from.
- State the treatment of the input term (added along the real axis vs rotated
  with the state) and keep all four implementations identical.
- `ssm_imag` must be written by `step_recurrence` — the verifier detects D2's
  closure exactly that way (`scripts/verify_deviations.py`, `check_d2`).
- Keep the B/C RoPE (`step_rope_bc`) unless the paper's equivalence argument
  says it is subsumed; if you remove it, say so in the register with evidence.

### 3. Land it in all four forward passes, then re-baseline

`src/core/ssm_block.cpp`,
`scripts/reference_mamba3.py`,
`scripts/reference_forward.py`,
`scripts/export_mamba_weights.py` (`Mamba3Block.forward`), all step-for-step.
Then:

- Regenerate parity vectors (`--generate`) and confirm C++ ↔ NumPy ≤ 1e-5.
- Re-run `scripts/validate_export.py` on a fresh export.
- Re-benchmark and report the delta (the extra term costs one multiply-add per
  state element, per `ARCHITECTURE_FIDELITY.md` D1; the rotation adds a
  `cos`/`sin` per angle that can be cached per head/sample).
- If the trainer's window length matters for the new recurrence, update
  `TRAINING_PLAN.md`.

## CONSTRAINTS

- **Real-time safety**: no allocation, no locks, no `throw`, no `std::function`,
  no heap in `process_sample`/`process_block`. New state goes into `SSMState`,
  new weights into `SSMWeights` with `MAX_*` strides.
- **No `exp`/`sin`/`cos` in the per-state innermost loop** where avoidable —
  compute per-head scalars once per sample (the current code already does this
  for `Ā`/`B̄`).
- **Numerical stability**: the state must stay bounded for all inputs (the
  complex rotation must not add gain: `|Ā·e^{iθ}| = |Ā| < 1` for `A < 0` is the
  invariant to preserve). Do not reintroduce any periodic renormalization — that
  was D5 and it is closed *and guarded*.
- **Parity tolerance is 1e-5**, not "similar shape". The parity test fails on
  max-abs-difference, not on correlation.
- **Do not change the file format** beyond documented additive changes; keep
  `static_assert(sizeof(WeightFileHeader) == 64)`.

## OUTPUT FORMAT

1. C++ changes (`src/core/ssm_block.{h,cpp}`), including the derivation as a
   comment block above the recurrence.
2. Python changes in the three scripts listed above.
3. New tests in `tests/`:
   - `Recurrence.ThreeTermInputHistoryChangesImpulseResponse` — a constructed
     model where the β term is non-zero must differ from the two-term result in
     a way that matches the analytic trapezoidal response.
   - `Recurrence.ResonancePreservation` — an impulse into a model with a
     non-zero θ produces a decaying oscillation at the expected frequency
     (assert the peak bin of an FFT / the zero-crossing interval within a stated
     tolerance) and does **not** blow up over ≥ 10⁵ samples.
   - `Recurrence.ResetClearsHistory` — reset ⇒ identical output to a fresh
     instance for the first N samples.
4. Register + docs updates: `ARCHITECTURE_FIDELITY.md` D1/D2 → `CLOSED <date>`
   with the guard names and the measured numbers, §2 summary row, §1.2 rows
   removed/moved into §1.1 with evidence, `CHANGELOG.md`, and the
   README/MIGRATION "Mamba-3-inspired" wording upgraded to the precise
   description the register now supports (e.g. "three-term exponential-
   trapezoidal recurrence; complex state realized as 2×2 rotation blocks").
5. Measurements: parity number, benchmark table (before/after, same machine and
   build flags), listening notes for the resonance test.

## VERIFICATION HOOKS (already in place)

```bash
# static + numeric + gtest, one line per deviation
python3 scripts/verify_deviations.py

# the two checks that must flip for this prompt
python3 scripts/verify_deviations.py --no-build-checks | grep -A3 '^\[OPEN\] D'
```

- D1 flips when `step_recurrence` contains an input-history term (the checker
  looks for `B_prev`/`x_prev`/`beta`/`gamma` in the recurrence body).
- D2 flips when `step_recurrence` assigns to `ssm_imag[...]`.

Both are deliberately simple so they cannot be satisfied by a comment; keep the
implementation shaped so the checks stay meaningful, and update the checker only
if the paper's realization needs a different expression of the same property.

## ACCEPTANCE CRITERIA

- [ ] All four forward passes implement D1 + D2 identically; `git diff` shows the
      same change in each.
- [ ] `MAMBA_PARITY_DIR=$PWD/build MAMBA_REQUIRE_PARITY=1 ctest` green, parity
      max-abs difference ≤ 1e-5 with freshly generated vectors.
- [ ] Resonance test passes; long-run (≥ 10⁵ samples) high-state test shows no
      NaN/Inf and no growth beyond the analytic bound.
- [ ] Benchmark: `--config small --layers 1,2 --budget-pct 40` still PASS, and
      the new cost per sample is quoted for `train4070,trainl4` shapes.
- [ ] `scripts/verify_deviations.py` exits 0 with D1 and D2 `CLOSED`.
- [ ] Every place in the repo that says "Mamba-3-inspired" or "complex-valued
      state" has been re-read and either upgraded with the new evidence or left
      with a pointer to the still-open part.
- [ ] No new heap/lock/exception in the audio path (`tests/test_realtime_safety`
      and the call-graph audit still pass).
