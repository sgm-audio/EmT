# EmT docs

Start with the root [`README.md`](../README.md). Any architecture or
performance statement should be backed by test or benchmark output, or by a
`file:line` reference in [`ARCHITECTURE_FIDELITY.md`](ARCHITECTURE_FIDELITY.md).
Otherwise say "planned" or leave it out.

| Document | Contents |
|---|---|
| [`BUILD.md`](BUILD.md) | Prerequisites, build and test commands, offline builds, troubleshooting |
| [`ARCHITECTURE_FIDELITY.md`](ARCHITECTURE_FIDELITY.md) | What's implemented vs the Mamba-3 paper, deviation register D1–D6, claim policy |
| [`CLAIMS_VERIFICATION.md`](CLAIMS_VERIFICATION.md) | Claim-by-claim check of the earlier strategy and architecture docs |
| [`WEIGHT_FORMAT_SPEC.md`](WEIGHT_FORMAT_SPEC.md) | Byte-level weight file format and `LoadError` codes |
| [`PARAMETER_REFERENCE.md`](PARAMETER_REFERENCE.md) | Plugin parameters |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | Threads, memory layout, DSP path (design; see the root README's Known problems for where the code falls short) |
| [`MIGRATION_TO_MAMBA3.md`](MIGRATION_TO_MAMBA3.md) | Migration notes from the earlier Mamba-2 style core |
| [`REFACTORING_PLAN.md`](REFACTORING_PLAN.md) | Refactoring phases and status |
| [`TODO.md`](TODO.md) | Gap log |
| [`TRAINING_PLAN.md`](TRAINING_PLAN.md) | Training configs and objectives |
| [`adr/`](adr/) | Architecture decision records (threading, memory, ARA bridge) |
