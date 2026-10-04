# Worker System Prompt (T3-D3) — D3: Rank-R MIMO

**Status: CLOSED 2026-10-04 (refusal path).** This file both records the closure
and scopes the unimplemented half (rank-R execution) should it ever be wanted.

## ROLE
Two roles depending on which half you are doing:
- **Maintenance (default):** keep the refusal correct and the guards green.
- **Feature (only if the product needs rank-R):** implement the rank-R reduction
  end-to-end and delete the refusal.

## CONTEXT — what the register says

`ARCHITECTURE_FIDELITY.md` D3:

- **Paper:** rank-R multi-input multi-output state update.
- **Repo (before closure):** `in_proj`, `SSMWeights`, the loader, the exporter
  and the file format all reserve rank-R (`MAX_MIMO_RANK = 4`), but the forward
  pass indexed rank 0 only and never touched `mimo_o`.  A user training with
  `mimo_rank > 1` silently got rank-1 inference — the format promised more than
  the runtime delivered.

## CLOSURE RECORD (what exists now — do not regress)

| Piece | Where | Behaviour |
|---|---|---|
| Header field | `WeightFileHeader::mimo_rank` (`src/core/weight_loader.h`) | replaces the low half of the old `reserved` u64; header stays 64 bytes (`static_assert`) |
| Loader | `WeightLoader::load()` (`src/core/weight_loader.cpp`) | returns **-16** when `version >= 2 && mimo_rank > 1`; `0` = legacy pre-field file ⇒ treated as rank 1 |
| Predicate | `WeightLoader::fits_in_static_weights_with()` | rejects `mimo_rank > 1`, so a rank-R file cannot even look loadable |
| Exporter | `scripts/export_mamba_weights.py` `export_model()` | raises `SystemExit` with an actionable message instead of writing the file |
| Trainer | `training/train.py` `main()` | refuses a config with `mimo_rank != 1` before training starts |
| Writers/readers | `reference_mamba3.py`, `reference_forward.py`, `validate_export.py` | header format is `'<12IQII'`; `mimo_rank` is written as `1` by the parity generator |
| Format string | `'<12IQII'` = 12×u32 + u64 `weight_data_size` + u32 `mimo_rank` + u32 `reserved` | 64 bytes |

Guards:

```
tests/test_weight_loader.cpp  → WeightLoaderTest.RejectsUnimplementedMimoRank
                              → WeightLoaderTest.LegacyHeaderWithoutMimoFieldIsRankOne
scripts/verify_deviations.py  → D3 (static + gtests + a rank-2 export subprocess)
```

Maintenance checklist: if you change the header layout, `bytes_per_layer()`,
`floats_per_layer()`, `get_layer_pointers()` or the exporter, re-run the D3
checks and keep a legacy (`mimo_rank == 0`) file loadable.

## FEATURE PROMPT — implement rank-R (only if required)

The rank-R execution is **not** implemented. If the product needs it:

1. Reduction: `y = Σ_r mimo_o[h][r][d] ⊙ (C_r · h_r)`, with the rank-r state
   update `h_r ← Ā·h_r + B̄·(B_r ⊙ x)` — per the paper. **Confirm the exact form
   against the paper/`state-spaces/mamba` Mamba-3 and cite `file:line`.** The
   repo's own `MIGRATION_TO_MAMBA3.md` §"MIMO combine (PAPER FORM — not executed
   by this repo)" records the intended einsum sequence; treat it as a pointer,
   not a source of truth.
2. Decide and document where `mimo_x` / `mimo_z` enter (the migration doc's
   "PAPER FORM" suggests they gate the input and output projections); a rank-2
   model must match a reference implementation, not merely "run".
3. State/cost: rank-R multiplies the state memory and the per-sample state
   update by R. `SSMState` is per-voice; measure and report the cost at
   `MAX_MIMO_RANK = 4` before enabling it. Keep `MAX_MIMO_RANK` as the compile
   limit and make rank a runtime property read from the header.
4. Training: `Mamba3Block.forward` must implement the same reduction so trained
   weights mean something; today `mimo_x/mimo_z/mimo_o` are zero-initialised and
   never used in the loss, so they receive no gradient.
5. Remove the refusal **last**: loader (`-16`), `fits_in_static_weights_with`,
   exporter `SystemExit`, trainer guard, and the D3 checks in
   `scripts/verify_deviations.py`.
6. Update `ARCHITECTURE_FIDELITY.md` D3 with the new status and evidence, the §2
   summary row, and §1.2's "MIMO" row (it currently forbids the claim).

## CONSTRAINTS

- Do not relax the refusal to "warn and continue" as an intermediate step — the
  whole point of the closure is that silent rank-1 execution cannot happen.
- Keep error code `-16` reserved for this condition; document new codes in
  `weight_loader.h` where the existing -10…-16 list lives.
- Real-time rules apply to the rank-R path as well (no allocation; state in
  `SSMState`).

## ACCEPTANCE CRITERIA

Refusal (maintenance):
- [ ] `python3 scripts/verify_deviations.py` reports D3 `CLOSED`, including the
      rank-2 export subprocess test.
- [ ] A rank-2 file with a valid header/payload is rejected by `load()` with
      `-16`, and `fits_in_static_weights()` is false.

Feature (only if implemented):
- [ ] Rank-2 C++ output matches a NumPy/PyTorch reference within 1e-5, and the
      trainer's forward produces the same numbers for the same weights.
- [ ] Per-sample benchmark for rank 1/2/4 at one training-config shape is
      published; the 40 %-of-one-core gate still passes at the shipped rank.
- [ ] The refusal paths are removed together with their tests, and the register
      records the implementation (with citations).
