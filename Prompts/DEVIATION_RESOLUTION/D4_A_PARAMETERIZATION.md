# Worker System Prompt (T3-D4) — D4: `A` is input-derived, not a learned parameter

**Status: CLOSED 2026-10-04 (documentation path).** This file records the
decision and scopes the alternative (learned `A`), which is *not* implemented.

## ROLE
Maintainer of the claim ledger. Your job is to make sure nobody — including a
future agent — upgrades this variant into "the paper's A parameterization" in a
README, PR description, demo or comment.

## CONTEXT — what the register says

`ARCHITECTURE_FIDELITY.md` D4:

- **Paper lineage:** `A` is a learned (typically negative-exponentiated)
  parameter; selectivity lives in Δ, B, C.
- **Repo:** `A` comes from the input projection and is clamped:
  `A_t = -softplus(dd_A)`, clamped to `A_t <= -1`
  (`SSMBlock::step_preprocess_decay_and_dt`; the same in
  `reference_mamba3.py` Step 2, `reference_forward.py` Step 2 and the PyTorch
  forward).  This is a legitimate variant, but it is not the paper's
  parameterization.

## CLOSURE RECORD (what exists now)

- D4 comments at the clamp site in `src/core/ssm_block.cpp` and the header,
  and in the Python definitions, stating: input-derived, clamped, selectivity in
  Δ/B/C, "do not describe this as the paper's A parameterization".
- `ARCHITECTURE_FIDELITY.md` §1.2 forbids the "learned A" claim; §2 D4 is marked
  closed; `scripts/verify_deviations.py` → `check_d4` fails if either the code
  comment or the register status disappears.
- The clamp is load-bearing for stability: with `A_t <= -1` the trapezoidal
  factor `Ā = (1 + A_tΔ/2)/(1 - A_tΔ/2)` satisfies `|Ā| < 1` for all `Δ > 0`,
  which is the argument that made the D5 guard removable.  A learned `A` must
  preserve a bound of the same kind.

## ALTERNATIVE (not implemented): learned per-head `A_log`

Only pursue this if training shows the input-derived `A` limits quality. It is a
format + training change, not a refactor:

1. Format: add a per-head `A_log[nheads]` array (or reuse the reserved `trap`
   slot budget), bump the header version, keep version-2 files loadable, and
   extend `bytes_per_layer()` / `floats_per_layer()` / `get_layer_pointers()` /
   the exporter.
2. Behaviour: `A = -softplus(A_log)` (or `-exp(A_log)` per the paper — cite the
   source) and **keep a clamp or a stability proof**; remove the clamp only with
   a long-run test that shows the state cannot diverge.
3. Training: make it a parameter with a sensible init (e.g. so that the
   effective pole matches today's distribution), and re-train before making any
   quality claim.
4. Docs: if the parameterization changes, D4's register entry changes meaning —
   update the wording rather than deleting the entry, and quote the training
   evidence (loss curves / A-B comparison) as the substantiation, per the §6
   claim policy.

## ACCEPTANCE CRITERIA

Documentation (current state — keep green):
- [ ] `python3 scripts/verify_deviations.py` reports D4 `CLOSED`.
- [ ] No file in the repo describes `A` as learned/paper-faithful (grep
      `learned A`, `A_log`, `A parameter`).

If the learned-`A` path is ever implemented:
- [ ] Parity green with freshly generated vectors; long-run stability test
      passes with the new parameterization.
- [ ] A trained model with learned `A` is compared against the current variant
      on the same dataset, with numbers quoted from the harness.
- [ ] `ARCHITECTURE_FIDELITY.md` §1.2/§2 and `WEIGHT_FORMAT_SPEC.md` updated.
