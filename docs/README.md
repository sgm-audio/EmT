# EmT documentation index

Start at the repository root [`README.md`](../README.md), then come here for
the full guides. The claim policy applies to every document: any architecture
or performance statement must be backed by a measurement in `tests/` output
or a `file:line` citation in
[`ARCHITECTURE_FIDELITY.md`](../ARCHITECTURE_FIDELITY.md) — otherwise write
"planned", "approximation of", or delete the claim.

## Guides (this folder)

| Document | Contents |
|---|---|
| [`BUILD.md`](BUILD.md) | Prerequisites, configure/build/test commands, offline builds, CI-parity checklist, troubleshooting |
| [`archive/`](archive/) | Superseded reports, kept for history. Do not quote numbers from these without checking the current ledgers first |

## Normative references (repository root)

| Document | Contents |
|---|---|
| [`ARCHITECTURE_FIDELITY.md`](../ARCHITECTURE_FIDELITY.md) | **Authoritative.** Implemented-vs-paper ledger, deviation register (D1–D6), claim policy |
| [`CLAIMS_VERIFICATION.md`](../CLAIMS_VERIFICATION.md) | Independent claim-by-claim substantiation |
| [`WEIGHT_FORMAT_SPEC.md`](../WEIGHT_FORMAT_SPEC.md) | Byte-level weight file format + `LoadError` table |
| [`PARAMETER_REFERENCE.md`](../PARAMETER_REFERENCE.md) | Plugin parameter definitions |
| [`ARCHITECTURE.md`](../ARCHITECTURE.md) | Concurrency model, memory architecture, DSP pipeline |
| [`REFACTORING_PLAN.md`](../REFACTORING_PLAN.md) | Phased refactoring plan and status |
| [`AGENTS.md`](../AGENTS.md) | Contributor quick reference (build, types, gotchas) |
| [`TODO.md`](../TODO.md) | Gap log and verification evidence |
| [`TRAINING_PLAN.md`](../TRAINING_PLAN.md) + [`training/`](../training/) | Training configs and objectives |
| [`ADR's/`](../ADR's/) | Architecture decision records (threading, memory, ARA bridge) |
| [`Prompts/`](../Prompts/) | Implementation prompts per subsystem, incl. deviation resolution |

## Archive policy

A document moves to `docs/archive/` when a newer document supersedes it.
Archived files keep their original content (link targets are repaired);
inbound references are updated to the `docs/archive/` path at move time.
