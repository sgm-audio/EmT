#!/usr/bin/env python3
"""
Reference Mamba conditioning projection (PyTorch).

Mirrors the C++ SSMBlock::process_sample() conditioning application so the
C++ output can be cross-checked bit-for-bit against a Python ground truth.

Per the WEIGHT_FORMAT_SPEC.md rows 13-16 and TODO.md Phase 3 (lines
184-211), the conditioning vector (COND_DIM floats) is projected into
per-state contributions for dt and B:

    dt_eff = softplus( dt_base + cond_proj_dt @ cond + cond_dt_bias )
    B_eff  = B_base            + cond_proj_B  @ cond + cond_B_bias

A zero cond vector short-circuits the projection (the C++ side also has
`conditioning_active_` for this; the C++ test enables the projection by
passing a non-zero vector).

Usage:
    python scripts/reference_conditioning.py
        --cond-dim 16 --d-state 16
        --input "0.1,0.0,1.0,0.5,0.2,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0"
        --proj-dt "0.01,0.02,..." --proj-B "0.03,0.04,..."
        --dt-bias  "0.0,..."     --B-bias   "0.0,..."

If no weights are provided, a deterministic toy set is used so the
self-test produces a known vector the C++ test_conditioning
TestConditioningProjectionMatchesPython can hardcode.
"""

import argparse
import math
import sys


def softplus(x: float) -> float:
    if x > 20.0:
        return x
    if x < -20.0:
        return 0.0
    return math.log1p(math.exp(x))


def projection(cond, proj_dt, proj_B, dt_bias, B_bias, d_state, cond_dim):
    """Return (dt_eff[d_state], B_eff[d_state]) for a single sample."""
    assert len(cond) == cond_dim
    assert len(proj_dt) == d_state * cond_dim
    assert len(proj_B) == d_state * cond_dim
    assert len(dt_bias) == d_state
    assert len(B_bias) == d_state

    dt_eff = [0.0] * d_state
    B_eff = [0.0] * d_state
    for i in range(d_state):
        s_dt = dt_bias[i]
        s_B = B_bias[i]
        for c in range(cond_dim):
            w_dt = proj_dt[i * cond_dim + c]
            w_B = proj_B[i * cond_dim + c]
            cv = cond[c]
            s_dt += w_dt * cv
            s_B += w_B * cv
        dt_eff[i] = softplus(s_dt)
        B_eff[i] = s_B
    return dt_eff, B_eff


def deterministic_toy(d_state: int = 4, cond_dim: int = 4):
    """A 4x4 toy set with a non-zero cond vector and easily-checkable
    projection values.  Used by the C++ test_conditioning
    TestConditioningProjectionMatchesPython case (no need to parse
    command-line flags)."""
    cond = [0.5, -0.25, 1.0, 0.0]
    # proj_dt[i*cd + c] = (i + 1) * 0.1 * (c + 1) * 0.01
    proj_dt = []
    proj_B = []
    for i in range(d_state):
        for c in range(cond_dim):
            proj_dt.append((i + 1) * 0.1 * (c + 1) * 0.01)
            proj_B.append((i + 1) * 0.05 * (c + 1) * 0.02)
    dt_bias = [0.1 * (i + 1) for i in range(d_state)]
    B_bias = [-0.2 * (i + 1) for i in range(d_state)]
    return cond, proj_dt, proj_B, dt_bias, B_bias


def fmt_list(xs):
    return ", ".join(f"{v:.6f}" for v in xs)


def main():
    p = argparse.ArgumentParser(description="Reference Mamba conditioning")
    p.add_argument("--cond-dim", type=int, default=16)
    p.add_argument("--d-state", type=int, default=16)
    p.add_argument("--cond", default=None,
                   help="comma-separated float list of length cond-dim")
    p.add_argument("--proj-dt", default=None)
    p.add_argument("--proj-B", default=None)
    p.add_argument("--dt-bias", default=None)
    p.add_argument("--B-bias", default=None)
    p.add_argument("--toy", action="store_true",
                   help="Use the deterministic 4x4 toy set")
    args = p.parse_args()

    if args.toy:
        cond, proj_dt, proj_B, dt_bias, B_bias = deterministic_toy(
            d_state=4, cond_dim=4)
        cd, ds = 4, 4
    else:
        cd = args.cond_dim
        ds = args.d_state
        if not (args.cond and args.proj_dt and args.proj_B
                and args.dt_bias and args.B_bias):
            print("ERROR: provide --cond, --proj-dt, --proj-B, "
                  "--dt-bias, --B-bias, or use --toy", file=sys.stderr)
            return 2
        cond = [float(x) for x in args.cond.split(",")]
        proj_dt = [float(x) for x in args.proj_dt.split(",")]
        proj_B = [float(x) for x in args.proj_B.split(",")]
        dt_bias = [float(x) for x in args.dt_bias.split(",")]
        B_bias = [float(x) for x in args.B_bias.split(",")]

    dt_eff, B_eff = projection(cond, proj_dt, proj_B, dt_bias, B_bias,
                                ds, cd)

    print(f"cond (len={len(cond)}): [{fmt_list(cond)}]")
    print(f"dt_eff (len={ds}): [{fmt_list(dt_eff)}]")
    print(f"B_eff  (len={ds}): [{fmt_list(B_eff)}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
