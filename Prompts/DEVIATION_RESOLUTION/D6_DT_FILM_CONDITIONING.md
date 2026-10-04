# Worker System Prompt (T3-D6) — D6: `dt_film_proj_w` (Δ-FiLM) wired

**Status: CLOSED 2026-10-04 (wired).** This file records the exact convention so
that any future change to the conditioning path — or a re-implementation after
the D1/D2 work — keeps all four forward passes identical.

## ROLE
Conditioning owner. The invariant: **the conditioning vector modulates the input
(FiLM) and Δ (Δ-FiLM) with one convention, applied only when conditioning is
active.**

## CONTEXT — what the register says

`ARCHITECTURE_FIDELITY.md` D6: `dt_film_proj_w` — the per-head Δ modulation
projection — was exported, loaded, copied into `SSMWeights`, and **never read** by
the forward pass.  The trainer's `Mamba3Block.forward` ignored it too, so its
parameters received no gradient.

## THE CONVENTION (identical in all four implementations)

For head `h`, with `cond[k]` the conditioning vector (`COND_DIM` floats):

```
γ_h = Σ_k dt_film_proj_w[2h   ][k] · cond[k]     (row 2h   → γ)
β_h = Σ_k dt_film_proj_w[2h+1 ][k] · cond[k]     (row 2h+1 → β)
dt_pre = (dd_dt[h] + dt_bias[h]) · (1 + γ_h) + β_h
Δ_h    = softplus(dt_pre)
```

Applied **only when `conditioning_active`** (i.e. the conditioning vector has a
non-zero element).  With `cond ≡ 0` the code path is skipped, so unconditioned
inference is bit-identical to the pre-D6 build — this is what keeps the parity
vectors and every existing measurement comparable.

The input FiLM uses the same `v ↦ v·(1+γ) + β` shape
(`SBlock::step_film_conditioning`, `film_proj_w` rows `2j`/`2j+1`).  Keep the two
paths symmetric; asymmetric conventions are the easiest way to make training and
inference disagree.

## WHERE IT LIVES

| Implementation | Site |
|---|---|
| C++ runtime | `SSMBlock::step_preprocess_decay_and_dt` (`src/core/ssm_block.cpp`) |
| Parity reference | `scripts/reference_mamba3.py` `mamba3_process_sample`, Step 2 (takes `dt_film_proj_w` after `film_proj_w`) |
| Second reference | `scripts/reference_forward.py`, Step 2 |
| Trainer model | `scripts/export_mamba_weights.py`, `Mamba3Block.forward` (`cond_active` flag, `dt_gb` split) — this is the model `training/train.py` trains |

Weight layout: `dt_film_proj_w[2*MAX_NHEADS][COND_DIM]`, row-major, exported
`[2*nheads, cond_dim]` per layer (array 13 of 13 in `WEIGHT_FORMAT_SPEC.md`).

## GUARDS

```
tests/test_conditioning.cpp → ConditioningModulation.DtFilmModulatesDelta
    (a) zero dt_film_proj_w + non-zero cond ⇒ byte-identical trajectory
    (b) non-zero dt_film_proj_w + non-zero cond ⇒ trajectory differs
scripts/reference_mamba3.py --generate
    writes NON-ZERO dt_film_proj_w (randn·0.01), so the C++ ↔ NumPy parity test
    covers the Δ-FiLM path instead of multiplying by zeros
scripts/verify_deviations.py → D6 (four implementation checks + the gtest)
```

**Testing trap (documented because it bit the first version of the test):** the DC
gain of the trapezoidal update is `1/|A|` and is *independent of Δ*, so a
converged constant-input response cannot detect Δ modulation.  Drive a transient
(or an alternating input) and compare trajectories.  Also note that
`tests/test_conditioning.cpp`'s shared `make_test_weights()` leaves the x and B/C
projections at zero for the MAX-dimension block — a fixture that does not drive
x and does not bias B/C has a dead state and will "pass" vacuously.

## RULES FOR THE FUTURE

1. If D1/D2 change the recurrence, `dt_pre` semantics may change (e.g. if the
   paper's coefficients come from a different discretization).  Re-derive the
   Δ-FiLM insertion point rather than porting the expression blindly, and keep
   all four implementations in one commit.
2. Never drop `dt_film_proj_w` from the format without a header-version change;
   it is array 13 of 13 and the loader/`bytes_per_layer()` depend on its size.
3. If Δ ever stops being the conditioning target, update the register entry's
   wording instead of deleting it, and state what the conditioning does instead.

## ACCEPTANCE CRITERIA

- [ ] `python3 scripts/verify_deviations.py` reports D6 `CLOSED`.
- [ ] Parity green with non-zero `dt_film_proj_w` in the generated vectors.
- [ ] Unconditioned output is bit-identical to the pre-D6 build for the same
      weights (checkable by running the parity test with `cond ≡ 0`).
- [ ] A trained model that actually learns `dt_film_proj` (non-zero after
      training) produces audibly different output for different conditioning
      vectors — quote the loss/weights evidence if claiming this.
