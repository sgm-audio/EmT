# Worker System Prompt (T3-D5) — D5: Periodic State Renormalization

**Status: CLOSED 2026-10-04 (guard removed).** This file records the defect, the
fix, its guards, and the strict conditions under which any future numerical
guard may be reintroduced.

## ROLE
Numerical-stability owner. The invariant you protect: **the state stays bounded
because the discretization is contractive — not because something periodically
mutates it.**

## CONTEXT — what the register says

`ARCHITECTURE_FIDELITY.md` D5:

- The old code ran `step_periodic_normalization()` every
  `SSM_STATE_NORM_INTERVAL = 4096` samples: if `max|h| > 1`, the entire state was
  divided by `max|h|`.  The same operation existed in both Python references, so
  parity held — it was a design defect, not a drift bug.
- **Measured artifact:** with a constructed model whose steady state is `|h| = 4`,
  the output stepped at `n = 4096` (register's original measurement
  `20.918667 → 17.822269`, a 14.8 % gain step repeating every 4096 samples
  ≈ 11.7 Hz sawtooth at 48 kHz).  The trainer's windows (2048/4096 samples) and
  the 16-sample parity test could not catch it.
- Cause of the achievable `|h| > 1`: with `A_t <= -1` and B normalized to unit
  RMS plus `B_bias`, the steady state magnitude is ≈ `|B_bias| + 1`.

## CLOSURE RECORD (what exists now — do not regress)

Removed from every definition of the forward pass:

| Removed | From |
|---|---|
| `SSMBlock::step_periodic_normalization()` + its call | `src/core/ssm_block.cpp` |
| `STATE_NORM_INTERVAL`, `sample_count_`, the method declaration | `src/core/ssm_block.h` |
| `#define SSM_STATE_NORM_INTERVAL 4096` | `src/config.h` **and** the `generate_config_h()` output in `scripts/export_mamba_weights.py` |
| "Step 8" renormalization + `sample_count` | `scripts/reference_mamba3.py`, `scripts/reference_forward.py` |

Why it is safe (this is the argument to preserve in code comments):

```
Ā = (1 + A·Δ/2) / (1 - A·Δ/2)      B̄ = Δ / (1 - A·Δ/2)
A < 0  ⇒  |Ā| < 1  for every Δ > 0
state bound:  |h| ≤ |B̄·B·x| / (1 - |Ā|)     (geometric series)
DC gain:      B̄/(1 - Ā) = 1/|A|             (independent of Δ)
```

Guards:

```
tests/test_stability.cpp → Stability.NoRenormalizationGainStepAtFormerIntervalBoundary
                           (constructed |h| = 4 case; no step > 1 % of the settled
                            level inside the former 4096/8192 boundary windows;
                            asserts max|h| > 1 so the test cannot pass vacuously)
                         → Stability.HighStateLongRunStaysBounded
                           (65 536 samples with |h| > 1: no NaN/Inf, no divergence)
scripts/verify_deviations.py → D5: static scan for renorm *code* (comments are
                           allowed and expected), the numeric continuity check,
                           and the counterfactual with the old guard re-applied
                           (currently ≈ 24 % step for the same model, so the
                            regression test is demonstrably non-vacuous)
```

## RULES FOR THE FUTURE

1. **Never reintroduce a mutating state guard.** If a numerical bound is ever
   needed again (e.g. an extreme quantization path), it must be either
   - provably non-mutating (a bound check that clamps *local* numerics, not the
     carried state), or
   - a documented, opt-in guard with all of:
     (a) a continuity regression test across the interval boundary,
     (b) training windows longer than the interval, and
     (c) a listening test on the constructed `|h| > 1` case,
     plus an entry in `ARCHITECTURE_FIDELITY.md` §1.3 marked as an EmT addition.
2. **Do not add a guard to "fix" a parity failure.** Parity failures are drift
   bugs; the guard masks both sides identically and hides them.
3. When you touch the recurrence (D1/D2 work), re-run the two stability tests —
   the bound argument above holds for the trapezoidal factor; a new
   discretization needs its own argument.

## VERIFICATION

```bash
./build/tests/test_stability --gtest_filter='*Renormalization*:*HighState*'
python3 scripts/verify_deviations.py --no-build-checks   # numeric continuity + counterfactual
```

## ACCEPTANCE CRITERIA

- [ ] `python3 scripts/verify_deviations.py` reports D5 `CLOSED` with a current
      boundary step < 1 % of the settled level and a legacy counterfactual well
      above it.
- [ ] No `STATE_NORM_INTERVAL` / `sample_count` *code* exists anywhere (comments
      explaining the removal are fine and are ignored by the checker).
- [ ] Both stability tests pass and would fail against the old guard (verified
      through the counterfactual measurement, not by inspection).
