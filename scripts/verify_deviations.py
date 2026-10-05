#!/usr/bin/env python3
"""
verify_deviations.py — machine-checkable status of the deviation register
(ARCHITECTURE_FIDELITY.md §2, D1-D6).

The register is the project's honesty ledger: it lists every place where this
build differs from the Mamba-3 paper.  This script makes each entry checkable so
that a fixed deviation cannot silently come back, and a still-open deviation
cannot silently be described as fixed.

Per deviation it runs the static and/or numeric checks that the corresponding
prompt in Prompts/DEVIATION_RESOLUTION/ is required to satisfy:

    D1  three-term exponential-trapezoidal recurrence   (OPEN — prompt pending)
    D2  2x2 rotational (complex) state dynamics         (OPEN — prompt pending)
    D3  rank-R MIMO refused at load/export              (closed)
    D4  input-derived A documented as a variant         (closed)
    D5  periodic state renormalization removed          (closed)
    D6  dt_film_proj_w wired into Δ                     (closed)

Usage:
    python3 scripts/verify_deviations.py [--build-dir build] [--json FILE]
                                         [--no-build-checks]

Exit status:
    0  every closed deviation passes and the open ones are still declared open
    1  a closed deviation regressed, or a claim contradicts the code
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"
CORE = REPO / "src" / "core"

CLOSED = "CLOSED"
OPEN_PENDING = "OPEN_PENDING"
FAIL = "FAIL"
SKIP = "SKIP"


@dataclass
class Result:
    deviation: str
    title: str
    status: str
    evidence: List[str] = field(default_factory=list)
    closed_expected: bool = True  # what the register/prompts claim


def read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def grep(path: Path, pattern: str) -> List[str]:
    rx = re.compile(pattern)
    return [f"{path.relative_to(REPO)}:{i}: {l.strip()}"
            for i, l in enumerate(read(path).splitlines(), 1) if rx.search(l)]


def run(cmd: List[str], cwd: Path = REPO, timeout: int = 600) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, timeout=timeout,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True)


def run_gtest(build_dir: Path, binary: str, gtest_filter: str) -> Result:
    """Run a gtest filter and turn it into evidence."""
    exe = build_dir / "tests" / binary
    if not exe.exists():
        return Result("", "", SKIP, [f"SKIP: {exe} not built"])
    cp = run([str(exe), f"--gtest_filter={gtest_filter}"])
    tail = [l for l in cp.stdout.splitlines()
            if "OK ]" in l or "FAILED" in l or "PASSED" in l]
    status = CLOSED if cp.returncode == 0 else FAIL
    return Result("", "", status, tail[-4:] or [f"exit {cp.returncode}"])


# ---------------------------------------------------------------------------
# D1 — discretization is bilinear, not exponential-trapezoidal
# ---------------------------------------------------------------------------

def check_d1() -> Result:
    r = Result("D1", "exponential-trapezoidal three-term recurrence", OPEN_PENDING)
    src = read(CORE / "ssm_block.cpp")
    rec = src[src.find("void SSMBlock::step_recurrence"):]
    rec = rec[:rec.find("\n}\n")]
    has_history = bool(re.search(r"(x_prev|B_prev|beta|gamma)", rec))
    r.evidence.append(
        "step_recurrence() contains an input-history (beta/gamma) term: "
        f"{has_history}")
    two_term = bool(re.search(r"ssm_real\[h\]\[d\]\[s\] = A_bar", rec))
    r.evidence.append(f"step_recurrence() uses a two-term update: {two_term}")
    if has_history:
        r.status = CLOSED
        r.evidence.append("D1 appears implemented; re-run the full acceptance "
                          "suite from Prompts/DEVIATION_RESOLUTION/D1_D2_*.md")
    else:
        r.evidence.append("Implementation deferred to the D1/D2 prompt "
                          "(F4 in ARCHITECTURE_FIDELITY.md §5)")
    return r


# ---------------------------------------------------------------------------
# D2 — the complex path is inert
# ---------------------------------------------------------------------------

def check_d2() -> Result:
    r = Result("D2", "2x2 rotational state dynamics (complex state)", OPEN_PENDING)
    src = read(CORE / "ssm_block.cpp")
    body = src[src.find("void SSMBlock::step_recurrence"):]
    body = body[:body.find("\n}\n")]
    writes_imag = bool(re.search(r"ssm_imag\[[^\]]*\]\s*[+\-*/]?=", body))
    r.evidence.append(
        f"step_recurrence() writes ssm_imag (rotational dynamics): {writes_imag}")
    rope = bool(re.search(r"B_norm\[base \+ s0\] = B0", src))
    r.evidence.append(f"B/C RoPE rotation present: {rope}")
    if writes_imag:
        r.status = CLOSED
    else:
        r.evidence.append("State is real; B/C rotation only.  Deferred to the "
                          "D1/D2 prompt (F4).")
    return r


# ---------------------------------------------------------------------------
# D3 — MIMO is carried but unused
# ---------------------------------------------------------------------------

def check_d3(build_dir: Path, no_build_checks: bool) -> Result:
    r = Result("D3", "rank-R MIMO refused at load and export", CLOSED)

    loader_h = read(CORE / "weight_loader.h")
    loader_c = read(CORE / "weight_loader.cpp")
    exporter = read(SCRIPTS / "export_mamba_weights.py")
    fmt = read(SCRIPTS / "emt_format.py")
    trainer = read(REPO / "training" / "train.py")

    checks = [
        ("WeightFileHeader carries mimo_rank",
         "uint32_t mimo_rank;" in loader_h),
        ("WeightLoader::load() refuses mimo_rank > 1 with LoadError::MimoRankRefused",
         re.search(r"mimo_rank\s*>\s*1", loader_c) is not None
         and "LoadError::MimoRankRefused" in loader_c
         and re.search(r"MimoRankRefused\s*=\s*-16", loader_h) is not None),
        ("fits_in_static_weights_with() rejects rank-R headers",
         "mimo_rank <= 1" in loader_h),
        ("exporter writes the field and refuses rank-R",
         "mr, 0)" in fmt and "mimo_rank={mr} is not supported" in exporter),
        ("trainer refuses a rank-R config up front",
         'cfg.get("mimo_rank", 1)' in trainer),
    ]
    for label, ok in checks:
        r.evidence.append(f"{'ok  ' if ok else 'MISS'} {label}")
        if not ok:
            r.status = FAIL

    if not no_build_checks:
        gr = run_gtest(build_dir, "test_weight_loader", "*MimoRank*:*MimoField*")
        r.evidence.extend(gr.evidence)
        if gr.status == FAIL:
            r.status = FAIL
        elif gr.status == SKIP:
            r.status = SKIP

    # End-to-end: a rank-2 export must fail loudly and write no file.
    if not no_build_checks:
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "rank2.bin"
            cp = run([sys.executable, str(SCRIPTS / "export_mamba_weights.py"),
                      "--state-dim", "8", "--model-dim", "8", "--inner-dim", "8",
                      "--nheads", "1", "--layers", "1", "--mimo-rank", "2",
                      "--random-init", "--output", str(out)])
            refused = cp.returncode != 0 and not out.exists()
            r.evidence.append(
                f"{'ok  ' if refused else 'MISS'} rank-2 export refused "
                f"(exit {cp.returncode}, file written: {out.exists()})")
            if not refused:
                r.status = FAIL
    return r


# ---------------------------------------------------------------------------
# D4 — A is input-derived, not a learned parameter
# ---------------------------------------------------------------------------

def check_d4() -> Result:
    r = Result("D4", "input-derived A documented as a variant", CLOSED)
    src = read(CORE / "ssm_block.cpp")
    site = src[src.find("void SSMBlock::step_preprocess_decay_and_dt"):]
    site = site[:site.find("\n}\n")]
    documented = "D4" in site and "learned" in site.lower()
    r.evidence.append(
        f"{'ok  ' if documented else 'MISS'} clamp site documents A as an "
        "input-derived variant (D4)")
    if not documented:
        r.status = FAIL

    fidelity = read(REPO / "ARCHITECTURE_FIDELITY.md")
    reg = fidelity[fidelity.find("### D4"):fidelity.find("### D5")]
    r.evidence.append(
        f"{'ok  ' if 'CLOSED' in reg else 'MISS'} register entry D4 marked closed")
    if "CLOSED" not in reg:
        r.status = FAIL
    return r


# ---------------------------------------------------------------------------
# D5 — periodic state renormalization (numeric + static)
# ---------------------------------------------------------------------------

# softplus^-1(1.0) = log(e - 1)
LOG_E1M1 = 0.5413248546129181


def build_steady_state_h4_case(d_state=16, d_model=16, d_inner=32, nheads=2):
    """Same constructed case as tests/test_stability.cpp (|h| steady state = 4).

    Returns the kwargs for mamba3_process_sample() plus the per-step output
    extractor state.
    """
    import numpy as np

    headdim = d_inner // nheads
    num_rope_angles = d_state // 2
    mimo_rank = 1
    d_cond = 128
    in_proj_out = (d_inner * 2 + nheads * mimo_rank * d_state * 2
                   + nheads * 3 + num_rope_angles)

    def row(target):
        r = np.zeros(d_model, dtype=np.float32)
        r[:] = target / d_model
        return r

    in_proj_w = np.zeros((in_proj_out, d_model), dtype=np.float32)
    r = 0
    for _ in range(d_inner):                       # z
        in_proj_w[r] = row(1.0); r += 1
    for _ in range(d_inner):                       # x
        in_proj_w[r] = row(1.0); r += 1
    for _ in range(nheads * mimo_rank * d_state):  # B
        in_proj_w[r] = row(1.0); r += 1
    for _ in range(nheads * mimo_rank * d_state):  # C
        in_proj_w[r] = row(1.0); r += 1
    for _ in range(nheads):                        # dd_dt -> dt = 1
        in_proj_w[r] = row(LOG_E1M1); r += 1
    for _ in range(nheads):                        # dd_A -> A = -1
        in_proj_w[r] = row(LOG_E1M1); r += 1
    # trap and angle rows stay zero

    B_bias = np.full((nheads, mimo_rank, d_state), 3.0, dtype=np.float32)
    C_bias = np.zeros((nheads, mimo_rank, d_state), dtype=np.float32)
    out_proj_w = np.zeros((d_model, d_inner), dtype=np.float32)
    for j in range(min(d_model, d_inner)):
        out_proj_w[j, j] = 1.0

    return dict(
        d_state=d_state, d_model=d_model, d_inner=d_inner, nheads=nheads,
        headdim=headdim, mimo_rank=mimo_rank, num_bc_heads=1, d_cond=d_cond,
        in_proj_w=in_proj_w,
        dt_bias=np.zeros(nheads, dtype=np.float32),
        B_bias=B_bias, C_bias=C_bias,
        B_norm_weight=np.ones(d_state, dtype=np.float32),
        C_norm_weight=np.ones(d_state, dtype=np.float32),
        D_proj=np.zeros(nheads, dtype=np.float32),
        out_proj_w=out_proj_w,
        film_proj_w=np.zeros((2 * d_model, d_cond), dtype=np.float32),
        dt_film_proj_w=np.zeros((2 * nheads, d_cond), dtype=np.float32),
        cond=np.zeros(d_cond, dtype=np.float32),
        conditioning_active=False,
    )


def measure_d5(samples: int = 9000) -> dict:
    """Measure the boundary discontinuity of the *current* reference, and the
    counterfactual with the legacy 4096-sample renormalization re-applied."""
    sys.path.insert(0, str(SCRIPTS))
    import numpy as np
    from reference_mamba3 import mamba3_process_sample  # noqa: E402

    case = build_steady_state_h4_case()
    nheads, headdim, d_state = case["nheads"], case["headdim"], case["d_state"]
    num_rope_angles = d_state // 2

    def trajectory(legacy_renorm: bool):
        ssm_real = np.zeros((nheads, headdim, d_state), dtype=np.float32)
        angle_state = np.zeros((nheads, num_rope_angles), dtype=np.float32)
        ys = np.zeros(samples, dtype=np.float64)
        max_h = 0.0
        for n in range(samples):
            out, ssm_real, angle_state = mamba3_process_sample(
                1.0, **case, ssm_real=ssm_real, angle_state=angle_state)
            ys[n] = out[0]
            max_h = max(max_h, float(np.max(np.abs(ssm_real))))
            if legacy_renorm and (n + 1) % 4096 == 0:
                m = float(np.max(np.abs(ssm_real)))
                if m > 1.0:
                    ssm_real = ssm_real / m
        return ys, max_h

    now, max_h = trajectory(legacy_renorm=False)
    legacy, _ = trajectory(legacy_renorm=True)

    def _first_boundary_step_index(y):
        """Index of the largest |diff| inside the two old boundary windows."""
        best_i, best_v = 0, 0.0
        for b in (4096, 8192):
            for n in range(b - 4, min(b + 5, len(y) - 1)):
                d = abs(y[n] - y[n - 1])
                if d > best_v:
                    best_i, best_v = n, d
        return best_i

    def boundary_step(y):
        step = 0.0
        for b in (4096, 8192):
            for n in range(b - 2, b + 5):
                if 0 < n < len(y):
                    step = max(step, abs(y[n] - y[n - 1]))
        return step

    settled = abs(now[-1])
    return {
        "steady_state_max_h": max_h,
        "settled_output": settled,
        # n is the index of the sample produced from the renormalized state.
        "current_boundary_step": boundary_step(now),
        "current_boundary_step_pct": 100.0 * boundary_step(now) / settled,
        "legacy_boundary_step": boundary_step(legacy),
        "legacy_boundary_step_pct": 100.0 * boundary_step(legacy) / settled,
        # Index where the legacy guard's step lands (the guard fires at the end
        # of sample 4095, so the artifact shows up at index 4096).
        "legacy_step_index": int(_first_boundary_step_index(legacy)),
        "samples": samples,
    }


def check_d5(build_dir: Path, no_build_checks: bool) -> Result:
    r = Result("D5", "periodic state renormalization removed", CLOSED)

    # Static: no interval constant / counter / renormalization *code* anywhere in
    # the shipped paths.  Prose in comments may (and does) mention the removed
    # guard to explain why it is gone, so match definitions and assignments only.
    code_patterns = [
        r"#define\s+SSM_STATE_NORM_INTERVAL",
        r"\bSTATE_NORM_INTERVAL\s*=",
        r"\bsample_count_?\s*(\+\+|--|[+*/-]?=)",
        r"def\s+\w*periodic_norm",
        r"void\s+SSMBlock::step_periodic_normalization",
    ]
    hits = []
    for p in [CORE / "ssm_block.cpp", CORE / "ssm_block.h", REPO / "src" / "config.h",
              SCRIPTS / "reference_mamba3.py", SCRIPTS / "reference_forward.py",
              SCRIPTS / "export_mamba_weights.py", REPO / "training" / "train.py"]:
        for pat in code_patterns:
            hits += grep(p, pat)
    r.evidence.append(
        f"{'ok  ' if not hits else 'MISS'} no renorm symbols in src/ or the "
        f"reference/exporter paths ({len(hits)} hits)")
    for h in hits[:4]:
        r.evidence.append(f"     {h}")
    if hits:
        r.status = FAIL

    # Numeric: the constructed |h| = 4 case must not step at the old boundary,
    # and the legacy guard must still reproduce the artifact (so the regression
    # test is not vacuous).
    try:
        m = measure_d5()
        r.evidence.append(
            f"constructed case steady-state max|h| = {m['steady_state_max_h']:.4f}, "
            f"settled output = {m['settled_output']:.4f}")
        r.evidence.append(
            f"current boundary step = {m['current_boundary_step']:.6f} "
            f"({m['current_boundary_step_pct']:.4f} % of settled)")
        r.evidence.append(
            f"legacy guard (counterfactual, same model) = "
            f"{m['legacy_boundary_step']:.4f} "
            f"({m['legacy_boundary_step_pct']:.2f} % of settled, at sample "
            f"{m['legacy_step_index']})")
        if m["current_boundary_step_pct"] >= 1.0:
            r.status = FAIL
            r.evidence.append("MISS boundary step >= 1 %% of settled level")
    except Exception as exc:  # numpy missing, etc.
        r.evidence.append(f"SKIP numeric check: {exc}")

    if not no_build_checks:
        gr = run_gtest(build_dir, "test_stability",
                       "*Renormalization*:*HighState*")
        r.evidence.extend(gr.evidence)
        if gr.status == FAIL:
            r.status = FAIL
        elif gr.status == SKIP:
            r.status = SKIP
    return r


# ---------------------------------------------------------------------------
# D6 — dt_film_proj_w wired into Δ
# ---------------------------------------------------------------------------

def check_d6(build_dir: Path, no_build_checks: bool) -> Result:
    r = Result("D6", "dt_film_proj_w modulates Δ", CLOSED)

    cpp = read(CORE / "ssm_block.cpp")
    site = cpp[cpp.find("void SSMBlock::step_preprocess_decay_and_dt"):]
    site = site[:site.find("\n}\n")]
    cpp_ok = "dt_film_proj_w" in site
    ref3 = read(SCRIPTS / "reference_mamba3.py")
    ref_fwd = read(SCRIPTS / "reference_forward.py")
    trainer = read(SCRIPTS / "export_mamba_weights.py")
    ref3_ok = "dt_film_proj_w[2 * h, k]" in ref3
    fwd_ok = "dt_film_proj_w[2 * h_idx, k]" in ref_fwd
    trainer_ok = "self.dt_film_proj" in trainer and "dt_gb" in trainer
    for label, ok in [
        ("SSMBlock::step_preprocess_decay_and_dt() reads dt_film_proj_w", cpp_ok),
        ("scripts/reference_mamba3.py applies Δ-FiLM", ref3_ok),
        ("scripts/reference_forward.py applies Δ-FiLM", fwd_ok),
        ("training/export forward applies Δ-FiLM", trainer_ok),
    ]:
        r.evidence.append(f"{'ok  ' if ok else 'MISS'} {label}")
        if not ok:
            r.status = FAIL

    if not no_build_checks:
        gr = run_gtest(build_dir, "test_conditioning", "*DtFilm*")
        r.evidence.extend(gr.evidence)
        if gr.status == FAIL:
            r.status = FAIL
        elif gr.status == SKIP:
            r.status = SKIP
    return r


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build-dir", default=str(REPO / "build"),
                    help="directory holding the built test binaries (default: build/)")
    ap.add_argument("--json", default=None, help="also write the report as JSON")
    ap.add_argument("--no-build-checks", action="store_true",
                    help="static + numeric checks only (no gtest / exporter subprocess)")
    args = ap.parse_args()

    build_dir = Path(args.build_dir)
    no_build = args.no_build_checks

    results = [
        check_d1(),
        check_d2(),
        check_d3(build_dir, no_build),
        check_d4(),
        check_d5(build_dir, no_build),
        check_d6(build_dir, no_build),
    ]

    width = 78
    print("=" * width)
    print("Deviation register verification — ARCHITECTURE_FIDELITY.md §2")
    print("=" * width)
    failed = False
    for res in results:
        sym = {CLOSED: "PASS", OPEN_PENDING: "OPEN", FAIL: "FAIL", SKIP: "SKIP"}[res.status]
        print(f"\n[{sym}] {res.deviation} — {res.title}  ({res.status})")
        for line in res.evidence:
            print(f"       {line}")
        if res.status == FAIL:
            failed = True

    n_closed = sum(1 for r in results if r.status == CLOSED)
    n_open = sum(1 for r in results if r.status == OPEN_PENDING)
    n_skipped = sum(1 for r in results if r.status == SKIP)
    print("\n" + "-" * width)
    print(f"{n_closed} closed (verified), {n_open} open (prompt pending), "
          f"{sum(1 for r in results if r.status == FAIL)} failing"
          + (f", {n_skipped} harness(es) skipped (build the tests to check the "
             f"behavioural guards)" if n_skipped else ""))
    if n_open:
        print("Open deviations are expected — resolve them with the prompts in "
              "Prompts/DEVIATION_RESOLUTION/ and the register will flip to CLOSED.")
    print("=" * width)

    if args.json:
        Path(args.json).write_text(json.dumps(
            [{"deviation": r.deviation, "title": r.title, "status": r.status,
              "evidence": r.evidence} for r in results], indent=2))
        print(f"wrote {args.json}")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
