# Sanitize manifest: EmT (2026-10-31)

Branch `chore/sanitize`, cut from `main` @ `855e84e`. 5 local commits, including this file. Nothing pushed.

## Removed
| Path | Files | Why |
|---|---|---|
| `Prompts/`, `AGENTS.md`, `agent.md`, `PROMPT_CLAUDE.md`, `omo-config.json` | 22 | Agent prompts and config, not project docs. The deviation register is in `docs/ARCHITECTURE_FIDELITY.md`. |
| `test_ssm_output.txt`, `test_ssm_error.txt` | 2 | Captured test output, containing a local Windows path and agent config. |
| `selective_scan_interface.py` | 1 | Stray upstream file. Nothing imports it. |
| `emt_lora_matrix_companion_assets.zip`, `emt_monolith_instrument_assets.zip` | 2 | Zips of the `packages/` PNGs. |
| `runs/` | 21 | Smoke-run configs, `final.pt`, `plugin_weights.bin`. Regenerate with `training/train.py --smoke-test`. |
| `docs/archive/AUDIT_REPORT.md`, `GAP_ANALYSIS.md` | 2 | Agent-generated audit ("96/100") and gap analysis, both superseded. |
| `extern/clap/include/{ext,factory,private}` | 44 | Byte-identical copies of `include/clap/*`. Nothing includes them. `extern/clap/LICENSE` (MIT) and a README naming CLAP 1.2.7 were added. |
| `ARA_SDK` gitlink | 1 | No `.gitmodules` entry, so a clone can't fetch it. |
| `packages/` | 10 | Duplicate PNGs of `assets/gui`. The two unique tiles and the previews moved to `assets/gui/`. |

## Moved
- Root docs → `docs/`: ARCHITECTURE, ARCHITECTURE_FIDELITY, CLAIMS_VERIFICATION, MIGRATION_TO_MAMBA3, PARAMETER_REFERENCE, REFACTORING_PLAN, TODO, TRAINING_PLAN, WEIGHT_FORMAT_SPEC.
- `ADR's/` → `docs/adr/` (the apostrophe in the folder name broke shell scripts).
- GUI images → `assets/gui/`, with a README saying they're mockups the plugin doesn't load.
- `verify_deviations.py` now reads `docs/ARCHITECTURE_FIDELITY.md`. 26 absolute `file:///x:/...` links in TODO.md are now relative.

## Added
`scripts/ci.sh` (local mirror of ci.yml), `assets/gui/README.md`, `extern/clap/LICENSE` and `README.md`, and `.gitignore` entries for runs/, models/, checkpoints, zips and test output. LICENSE is Apache-2.0 and unchanged. Scott's MIT rule is only for repos with no licence.

## Code fixes (tested)
- `CMakeLists.txt` and `ci.yml`: clap-validator is now called as `clap-validator validate <bundle>` (0.4.x CLI). Before, the ctest failed with "unrecognized subcommand" whenever clap-validator was installed.
- `benchmark_ssm_block`: the headline printed a rate in Hz under a kHz label.

## Doc claims changed
- Removed the static ctest/ARA badges and the other shields.
- FX state magic corrected to `MFXS` 0x4D465853 (the README said MBFX).
- Preset path corrected to `$XDG_DATA_HOME/clap/presets/com.mamba-ssm.synth`.
- Removed "16 MB plugin stack" (only the test executables get /STACK), "8,192 memcpy eliminated" (never measured), "Transparent Fallback", "zero malloc" stated as a guarantee, and the real-time-safety table.
- Added a Known problems section: weight-reload race, 4 MB stack object, FTZ leak, shared conditioning, D-skip wiring, clap-validator state failures.
- ARCHITECTURE.md §1 rewritten:
  - no conv1d, 0 latency;
  - bilinear (Tustin) discretisation, not exp-trapezoidal;
  - real state, not complex;
  - scalar recurrence;
  - 8 voices.
- **Not done:** the later ARCHITECTURE.md sections (around the old lines 104, 164 and 419) weren't re-toned line by line. They still need a pass.

## Tests, before and after (Shared 8-vCPU Intel Xeon VM (1 thread per core), Linux, zig clang 22.1.8, Release)
| Check | Before (`main` @ 855e84e) | After |
|---|---|---|
| `scripts/ci.sh` equivalent (configure, build, ctest) | exit 8: 125/126, the clap-validator ctest failed ("unrecognized subcommand") | exit 0: 125/125 (123 + 2 skipped without generated inputs) |
| clap-validator 0.4.1 full run | could not run through ctest | 81 tests: 63 pass, 6 fail (state reproducibility), 12 skipped. Reported separately, not gating |
| `benchmark_ssm_block`, small config, 48 kHz mono | GATE PASS | GATE PASS. 1 layer 2.219–2.260 µs/sample, 2 layers 4.564–4.628 µs/sample |
Logs: `_logs/emt-before-ci.log`, `emt-before-rest.log`, `emt-after-ci.log`, `emt-after-bench-training-shapes.log`.

## Needs Scott
- The 6 clap-validator state-reproducibility failures.
- CI is blocked on Actions billing.
- History rewrite approval (below).

## History rewrite (prepared, not run on the real repo)

Nothing has been rewritten or pushed. The commands below were tried on a throwaway `--mirror` clone in `/tmp` on 2026-10-31. They exited 0, and afterwards `git log --all -p` and the commit messages had 0 matches for the local-path patterns. Pack size went from 29.16 MiB to 17.94 MiB. That scratch clone was deleted.

The secret scans (gitleaks `git` and `dir`, and trufflehog, over all history) found no secrets. So this rewrite is about local paths and large or unneeded files, not live keys. Scott says any exposed keys were rotated anyway.

`paths-EmT.txt` (paths removed from every commit):
```text
AGENTS.md
agent.md
PROMPT_CLAUDE.md
omo-config.json
Prompts/
packages/
runs/
emt_lora_matrix_companion_assets.zip
emt_monolith_instrument_assets.zip
test_ssm_output.txt
test_ssm_error.txt
selective_scan_interface.py
docs/archive/
```

`replace.txt` is kept outside the repo, so the paths it removes aren't republished here. Use `/workspace/handoff/_history/replace-paths.txt`. It has 8 literal lines covering the Linux, Fedora Atomic (`/var/home`), macOS and Windows home-directory prefixes and the `x:` workspace URI. The `/var/home` line comes before the plain `/home` line.

Commands:
```bash
# 1. Merge chore/sanitize into the default branch first, so the rewrite covers the cleaned tree.
git clone --mirror https://github.com/sgm-audio/EmT.git EmT-rewrite.git
cd EmT-rewrite.git
git filter-repo --invert-paths --paths-from-file ../paths-EmT.txt --replace-text ../replace.txt
# 2. Check before pushing
git log --all -p | grep -cEf ../path-patterns.txt   # expect 0 (patterns file is next to replace.txt)
gitleaks git . --no-banner --redact                                                       # expect no leaks
git count-objects -vH
# 3. Push. Needs Scott's OK: this rewrites every commit hash.
git push --force --all origin && git push --force --tags origin
```

Caveats:
- GitHub keeps `refs/pull/*` for every PR, and a force-push can't touch them. The old objects stay reachable through those PR refs and cached commit URLs until GitHub Support runs a GC; ask them to purge after the push.
- Forks and existing clones keep the old history. Collaborators need to re-clone, and open PRs have to be recreated.
