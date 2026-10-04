# Deviation-Resolution Prompts (T1-D … T3-D)

Prompts for resolving the deviation register in `ARCHITECTURE_FIDELITY.md` §2 —
the list of places where this repository's implementation differs from the
Mamba-3 paper.  They follow the same format as the rest of `Prompts/`
(ROLE / CONTEXT / INSTRUCTIONS / CONSTRAINTS / OUTPUT FORMAT / ACCEPTANCE
CRITERIA), with one addition: every prompt names the automated check that must
flip when it is done.

## Status (2026-10-04, verified by `scripts/verify_deviations.py`)

| Deviation | Status | Prompt |
|---|---|---|
| D1 — bilinear two-term update vs the paper's exponential-trapezoidal three-term recurrence | **OPEN** (research work, F4) | `D1_D2_TRAPEZOIDAL_COMPLEX_STATE.md` |
| D2 — real state, `ssm_imag` inert | **OPEN** (same change set as D1) | `D1_D2_TRAPEZOIDAL_COMPLEX_STATE.md` |
| D3 — rank-R MIMO carried but not executed | **CLOSED** — rank > 1 refused at load (`-16`), export, and training-config validation | `D3_MIMO_RANK.md` (record + rank-R feature scope) |
| D4 — `A` input-derived, not learned | **CLOSED** — documented as a project variant, claim forbidden | `D4_A_PARAMETERIZATION.md` |
| D5 — periodic state renormalization → 14.8 % gain step | **CLOSED** — guard removed; stability from the A-stable update; continuity + long-run tests | `D5_STATE_RENORMALIZATION.md` |
| D6 — `dt_film_proj_w` loaded but never read | **CLOSED** — Δ-FiLM wired in all four forward passes | `D6_DT_FILM_CONDITIONING.md` |

Closed entries are **invariants with guards**, not history: the verifier fails if
any of them regresses.  The open entries keep the claim policy in
`ARCHITECTURE_FIDELITY.md` §1.2 binding ("Mamba-3-inspired" until D1/D2 land).

## Order of work

1. `D1_D2_TRAPEZOIDAL_COMPLEX_STATE.md` — the only remaining fidelity work.  It
   touches the recurrence in C++ and in all three Python definitions of the
   forward pass, and must land as one change set, with the register, README and
   CHANGELOG updated in the same PR.
2. The D3/D4/D5/D6 files are maintenance contracts: read the one that covers the
   file you are about to change, and keep its checks green.
3. `00_MASTER_DEVIATION_RESOLUTION.md` defines the frame: ordering, the
   all-implementations rule, the measurement protocol and the definition of done.

## How to verify

```bash
# one status line per deviation: static checks + numeric measurements + gtests
python3 scripts/verify_deviations.py

# fast iteration (no build needed): static + numeric only
python3 scripts/verify_deviations.py --no-build-checks

# machine-readable, for a PR description
python3 scripts/verify_deviations.py --json dev-report.json
```

Exit status is 0 when every deviation the register marks `CLOSED` passes its
checks and the open ones are still honestly declared open.  A `CLOSED` deviation
that regresses fails the run — that is the point of the register.

## Related documents

- `ARCHITECTURE_FIDELITY.md` — the ledger itself (claim policy §1, register §2,
  measurements §3, closure records §4c, F1–F5 plan §5, claim policy §6,
  reproduction §7).
- `CLAIMS_VERIFICATION.md` — the independent, claim-by-claim review that led to
  the register.
- `Prompts/` — the T1–T3 hierarchy for the rest of the plugin suite.
