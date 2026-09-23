"""
retune.py
=========

Re-tune the vertical scaling factors ``d_vec`` against the corrected RecFIF
builder, and write a new ``d_vec_opt.npy`` -- but only if it is actually better.

What this is for
----------------
``d_vec_opt.npy`` was produced by ``simple_tuner`` running against the pre-fix
builder, which did not interpolate its own knots. The open question was whether
the tuned roughness was a genuine fit to the data or an artefact of that defect.

Measured answer on weekly NIFTY (380 knots): **it was genuine.** Re-tuning against
the fixed builder moves median d_k only from 0.2968 to 0.3162, and the cached
vector still scores better at full resolution (MAPE 0.798% against 0.853%). The
low values were not the tuner dodging a bug.

Note also that "median d_k ~ 0.3" does not mean "nearly piecewise linear". Box
dimension goes as

    D = 1 + log(sum|d_k|) / log(N)

which for sum|d_k| = 132.7 over N = 379 maps gives D ~ 1.82 -- a genuinely rough
graph. The dimension depends on the *sum*, not the median.

Why a re-tune can lose
----------------------
``simple_tuner`` is greedy coordinate descent. Each update is non-worsening, but
only relative to where it started, and starting from the heuristic (MAPE ~2.77%)
lands in a worse local optimum than the cached vector already occupies. The tuner
also optimises a cheap proxy resolution that is not the one used to report
results. So this script:

* scores the cached vector at full resolution first;
* warm-starts the tuner from it by default, which cannot end up worse on the
  tuner's own objective;
* re-scores the result at full resolution;
* **refuses to overwrite a better vector with a worse one** unless forced.

Usage
-----
    python retune.py                     # warm start, keep the better vector
    python retune.py --init heuristic    # cold start, for comparison
    python retune.py --passes 3
    python retune.py --dry-run           # report only
    python retune.py --force             # write even if worse (not advised)

Author: Sagnik Barman, 2026.
"""

from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import numpy as np

from utils import fetch_weekly_nifty, extract_extrema
from RecFIF import build_recurrent_fif_with_dvec, knot_interpolation_error
from coordinate_tuner import simple_tuner, heuristic_d_vec

HERE = Path(__file__).resolve().parent
DVEC_PATH = HERE / "d_vec_opt.npy"


def mape(f_fn, df) -> float:
    fit = np.asarray([float(f_fn(t)) for t in df["t"].values], dtype=float)
    actual = np.asarray(df["Open"].values, dtype=float)
    return float(100.0 * np.mean(np.abs(fit - actual) / actual))


def score(d_vec, knots_t, knots_y, df, m, iters, grid) -> tuple:
    """Full-resolution MAPE and knot error for a candidate vector."""
    _, _, f_fn, _ = build_recurrent_fif_with_dvec(
        knots_t, knots_y, d_vec, m=m, iters=iters, grid_len=grid)
    return mape(f_fn, df), knot_interpolation_error(f_fn, knots_t, knots_y)


def describe(d, label) -> float:
    """Print the distribution; return the implied box dimension."""
    d = np.asarray(d, dtype=float)
    print(f"  {label}")
    print(f"    n={d.size}  min {d.min():.4f}  median {np.median(d):.4f}  "
          f"mean {d.mean():.4f}  max {d.max():.4f}")
    row = "    "
    for lo, hi in [(0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.0)]:
        pct = 100.0 * float(((d >= lo) & (d < hi)).sum()) / d.size
        row += f"[{lo:.1f},{hi:.1f}) {pct:5.1f}%   "
    print(row)
    s = float(np.abs(d).sum())
    n = d.size
    dim = 1.0 + np.log(s) / np.log(n) if s > 1 and n > 1 else 1.0
    print(f"    sum|d_k| = {s:.1f}   implied box dimension = {dim:.4f}")
    return dim


def main():
    ap = argparse.ArgumentParser(description="Re-tune d_vec against the fixed builder")
    ap.add_argument("--m", type=int, default=2)
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--fast-iters", type=int, default=150)
    ap.add_argument("--fast-grid", type=int, default=1000)
    ap.add_argument("--eval-iters", type=int, default=400)
    ap.add_argument("--eval-grid", type=int, default=2500)
    ap.add_argument("--init", choices=["cached", "heuristic"], default="cached",
                    help="warm start from the cached vector (default) or cold start")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="write the tuned vector even if it scores worse")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    print("Loading weekly NIFTY data ...")
    df = fetch_weekly_nifty()
    opens = np.asarray(df["Open"].values, dtype=float)
    idx = extract_extrema(opens, order=1)
    knots_t = np.asarray(df["t"].values, dtype=float)[idx]
    knots_y = opens[idx]
    K = len(knots_t)
    print(f"Knots selected: {K}  ({K-1} scaling factors to tune)")

    ev = dict(m=args.m, iters=args.eval_iters, grid=args.eval_grid)

    # --- incumbent ----------------------------------------------------------
    cached = None
    if DVEC_PATH.exists():
        c = np.load(DVEC_PATH)
        if len(c) == K - 1:
            cached = c
        else:
            print(f"  cached d_vec has {len(c)} entries, need {K-1} -- ignoring")

    m_cached = None
    if cached is not None:
        print("\nIncumbent d_vec:")
        describe(cached, "distribution")
        m_cached, k_cached = score(cached, knots_t, knots_y, df, **ev)
        print(f"    MAPE (full resolution): {m_cached:.4f}%")
        print(f"    knot error: {k_cached['max']:.3e}")

    # --- starting point -----------------------------------------------------
    if args.init == "cached" and cached is not None:
        init_d = cached
        print("\nWarm-starting the tuner from the incumbent.")
    else:
        init_d = heuristic_d_vec(knots_t, knots_y, m=args.m)
        print("\nCold-starting the tuner from the heuristic.")

    print(f"Re-tuning ({args.passes} pass(es), "
          f"iters={args.fast_iters}, grid={args.fast_grid}) ...")
    t0 = time.time()
    d_new = simple_tuner(
        knots_t, knots_y, df, m=args.m, passes=args.passes,
        candidate_factors=(0.6, 0.75, 0.9, 1.0, 1.1, 1.25, 1.5),
        fast_iters=args.fast_iters, fast_grid=args.fast_grid, init_d=init_d,
    )
    d_new = np.clip(np.asarray(d_new, dtype=float), 0.01, 0.99)
    print(f"Tuning finished in {time.time() - t0:.1f}s")

    print("\nCandidate d_vec:")
    describe(d_new, "distribution")
    m_new, k_new = score(d_new, knots_t, knots_y, df, **ev)
    print(f"    MAPE (full resolution): {m_new:.4f}%")
    print(f"    knot error: {k_new['max']:.3e}")

    # --- decide -------------------------------------------------------------
    better = True
    if m_cached is not None:
        delta = m_new - m_cached
        print(f"\nMAPE: {m_cached:.4f}%  ->  {m_new:.4f}%   ({delta:+.4f} pp)")
        med_shift = float(np.median(d_new) - np.median(cached))
        print(f"Median d_k: {np.median(cached):.4f} -> {np.median(d_new):.4f} "
              f"({med_shift:+.4f})")
        better = m_new < m_cached
        if not better:
            print("\n  The candidate is WORSE than the incumbent at full resolution.")
            print("  Greedy coordinate descent settles in whichever local optimum it")
            print("  starts nearest, and the tuner optimises a cheaper proxy than the")
            print("  score above. Keeping the incumbent.")
        if abs(med_shift) < 0.05:
            print("\n  Roughness is essentially unchanged, so the tuned values were a")
            print("  genuine fit to the data rather than an artefact of the old")
            print("  interpolation defect.")

    if args.dry_run:
        print("\n--dry-run: nothing written.")
        return 0
    if not better and not args.force:
        print("\nNothing written. Re-run with --force to overwrite anyway, or")
        print("--passes 4 / --init heuristic to search differently.")
        return 0

    out = Path(args.out) if args.out else DVEC_PATH
    if out.exists():
        backup = out.with_suffix(".prev.npy")
        shutil.copy2(out, backup)
        print(f"\nPrevious vector backed up to {backup.name}")
    np.save(out, d_new)
    print(f"Wrote {out.name} ({d_new.size} entries)")
    print("\nNext: python evaluate_coverage.py --noise-fit-fraction 0.7 --no-show")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
