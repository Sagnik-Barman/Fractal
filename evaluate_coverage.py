"""
evaluate_coverage.py
====================

Diagnostics for the 95% per-week bands produced by ``main.main_propagation``.

A single coverage percentage hides the thing that matters most. If the bands are
honest, weeks falling outside them should be scattered. If the misses arrive in
bunches, the weighted-characteristic-function step's assumption that knot noise
is independent is being violated, and the nominal 95% does not mean what it says
even when the headline number lands on target.

This module reports:

* coverage overall, and split at the CTS fit / hold-out boundary
* coverage per calendar year
* every week outside the band, with how far outside it sits
* the longest run of consecutive misses
* a Wald-Wolfowitz runs test on the inside/outside sequence
* lag-1 autocorrelation of the miss indicator, and the effective sample size
  and corrected standard error that follow from it
* coverage split by realised-volatility tercile, which is where an aggregate
  number that sits on 95% can still be hiding two badly calibrated regimes
* an attribution of each miss to either the interpolant being in the wrong
  place or the noise distribution being too narrow

Usage::

    python evaluate_coverage.py --noise-fit-fraction 0.7
    python evaluate_coverage.py --sample-every 8 --ncos 512   # fast, coarse

Author: Sagnik Barman, 2026.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------

def runs_test(indicator) -> dict:
    """Wald-Wolfowitz runs test on a binary sequence.

    ``indicator`` is True where the observation is *inside* the band. A run is a
    maximal stretch of identical values. Fewer runs than expected under
    independence means the outcomes are clustered.
    """
    x = np.asarray(indicator).astype(bool)
    n = x.size
    n1 = int(x.sum())          # inside
    n2 = n - n1                # outside
    if n1 == 0 or n2 == 0:
        return {"runs": 1, "expected": float("nan"), "z": float("nan"),
                "p": float("nan"), "n_inside": n1, "n_outside": n2}

    runs = 1 + int(np.sum(x[1:] != x[:-1]))
    expected = 2.0 * n1 * n2 / n + 1.0
    var = (2.0 * n1 * n2 * (2.0 * n1 * n2 - n)) / (n * n * (n - 1.0))
    z = (runs - expected) / np.sqrt(var) if var > 0 else float("nan")
    # two-sided normal p-value
    from math import erfc, sqrt
    p = erfc(abs(z) / sqrt(2.0)) if np.isfinite(z) else float("nan")
    return {"runs": runs, "expected": float(expected), "z": float(z),
            "p": float(p), "n_inside": n1, "n_outside": n2}


def longest_run_of_misses(indicator) -> tuple:
    """Longest stretch of consecutive misses, returned as (length, start index)."""
    x = ~np.asarray(indicator).astype(bool)
    best = cur = 0
    best_start = cur_start = 0
    for i, v in enumerate(x):
        if v:
            if cur == 0:
                cur_start = i
            cur += 1
            if cur > best:
                best, best_start = cur, cur_start
        else:
            cur = 0
    return best, best_start


def volatility_terciles(returns, window: int = 26) -> np.ndarray:
    """Label each week low/mid/high by trailing realised volatility.

    Uses a backward-looking rolling standard deviation so the label for a week
    depends only on information available before it.
    """
    r = pd.Series(np.asarray(returns, dtype=float))
    vol = r.rolling(window, min_periods=max(4, window // 4)).std()
    vol = vol.bfill().values
    lo, hi = np.nanpercentile(vol, [33.333, 66.667])
    labels = np.where(vol <= lo, "low", np.where(vol <= hi, "mid", "high"))
    return labels


def attribute_misses(out: dict) -> dict:
    """Split misses into interpolation failures and genuine tail events.

    Each band is ``interp_open * (1 + q)``. A week can therefore fall outside it
    for two quite different reasons:

    * the fractal interpolant put ``interp_open`` in the wrong place, so the
      band is centred away from where the market actually opened; or
    * the interpolant was fine and the week's Open-to-Close move genuinely
      exceeded the fitted 2.5%/97.5% quantiles.

    Re-centring the same quantiles on the *actual* Open separates the two. A
    miss that disappears under re-centring was an interpolation failure; one
    that survives is a real tail event and indicts the noise model.
    """
    df = out["df"]
    closes = np.asarray(df["Close"].values, dtype=float)
    opens = np.asarray(df["Open"].values, dtype=float)
    interp = np.asarray(out["interp_open"], dtype=float)
    low = np.asarray(out["close_low"], dtype=float)
    high = np.asarray(out["close_high"], dtype=float)

    # recover the per-week return quantiles from the bands
    with np.errstate(divide="ignore", invalid="ignore"):
        q_low = low / interp - 1.0
        q_high = high / interp - 1.0

    inside_model = (closes >= low) & (closes <= high)
    low_recentred = opens * (1.0 + q_low)
    high_recentred = opens * (1.0 + q_high)
    inside_recentred = (closes >= low_recentred) & (closes <= high_recentred)

    miss = ~inside_model
    interp_fault = miss & inside_recentred        # fixed by re-centring
    tail_event = miss & ~inside_recentred         # survives re-centring

    interp_err = np.abs(interp - opens) / opens
    return {
        "n_miss": int(miss.sum()),
        "n_interp_fault": int(interp_fault.sum()),
        "n_tail_event": int(tail_event.sum()),
        "interp_fault": interp_fault,
        "tail_event": tail_event,
        "interp_mape_all": float(100.0 * interp_err.mean()),
        "interp_mape_miss": float(100.0 * interp_err[miss].mean()) if miss.any() else float("nan"),
        "interp_mape_hit": float(100.0 * interp_err[~miss].mean()) if (~miss).any() else float("nan"),
        "coverage_recentred": float(100.0 * inside_recentred.mean()),
    }


def miss_autocorrelation(indicator) -> dict:
    """Lag-1 autocorrelation of the miss indicator, with effective sample size.

    Clustered misses inflate the true standard error of a coverage estimate well
    beyond the naive binomial value. For an AR(1)-like dependence the effective
    sample size is roughly ``n * (1 - rho) / (1 + rho)``.
    """
    m = (~np.asarray(indicator).astype(bool)).astype(float)
    n = m.size
    if n < 3 or m.std() == 0:
        return {"rho1": 0.0, "n_eff": float(n), "se_naive": float("nan"),
                "se_corrected": float("nan")}
    rho = float(np.corrcoef(m[:-1], m[1:])[0, 1])
    rho_c = float(np.clip(rho, -0.99, 0.99))
    n_eff = n * (1.0 - rho_c) / (1.0 + rho_c)
    n_eff = float(max(n_eff, 1.0))
    p = 0.95
    se_naive = 100.0 * np.sqrt(p * (1 - p) / n)
    se_corr = 100.0 * np.sqrt(p * (1 - p) / n_eff)
    return {"rho1": rho, "n_eff": n_eff,
            "se_naive": float(se_naive), "se_corrected": float(se_corr)}


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def coverage_report(out: dict, nominal: float = 95.0, max_misses_listed: int = 40):
    """Print the full diagnostic report for a ``main_propagation`` result dict."""
    df = out["df"]
    closes = np.asarray(df["Close"].values, dtype=float)
    low = np.asarray(out["close_low"], dtype=float)
    high = np.asarray(out["close_high"], dtype=float)
    dates = pd.to_datetime(df["Date"].values)
    n_fit = int(out.get("n_fit", len(closes)))

    inside = (closes >= low) & (closes <= high)
    n = inside.size

    print("=" * 72)
    print("COVERAGE DIAGNOSTICS")
    print("=" * 72)
    print(f"weeks: {n}   nominal coverage: {nominal:.1f}%")
    print(f"CTS fitted on the first {n_fit} weeks "
          f"({dates[0].date()} to {dates[min(n_fit, n - 1)].date()})")

    # --- headline ----------------------------------------------------------
    cov = 100.0 * inside.mean()
    print(f"\noverall        {cov:6.2f}%   ({int(inside.sum())}/{n})")
    if n_fit < n:
        cov_tr = 100.0 * inside[:n_fit].mean()
        cov_te = 100.0 * inside[n_fit:].mean()
        print(f"  in-sample    {cov_tr:6.2f}%   ({int(inside[:n_fit].sum())}/{n_fit})")
        print(f"  held out     {cov_te:6.2f}%   "
              f"({int(inside[n_fit:].sum())}/{n - n_fit})")

    # --- dependence --------------------------------------------------------
    ac = miss_autocorrelation(inside)
    print(f"\nmiss indicator lag-1 autocorrelation: {ac['rho1']:+.4f}")
    print(f"  naive binomial SE      {ac['se_naive']:.2f} pp  (assumes independence)")
    print(f"  dependence-corrected   {ac['se_corrected']:.2f} pp  "
          f"(effective n = {ac['n_eff']:.0f} of {n})")
    z_naive = (cov - nominal) / ac["se_naive"] if ac["se_naive"] > 0 else float("nan")
    z_corr = (cov - nominal) / ac["se_corrected"] if ac["se_corrected"] > 0 else float("nan")
    print(f"  overall coverage is {z_naive:+.2f} SE from nominal (naive), "
          f"{z_corr:+.2f} SE (corrected)")

    rt = runs_test(inside)
    print(f"\nruns test: {rt['runs']} runs observed, {rt['expected']:.1f} expected "
          f"under independence")
    print(f"  z = {rt['z']:+.3f}   p = {rt['p']:.4f}", end="   ")
    if np.isfinite(rt["p"]) and rt["p"] < 0.05:
        print("-> misses are CLUSTERED, not scattered" if rt["z"] < 0
              else "-> misses alternate more than chance")
    else:
        print("-> consistent with scattered misses")

    run_len, run_start = longest_run_of_misses(inside)
    if run_len > 0:
        print(f"\nlongest run of consecutive misses: {run_len} weeks "
              f"starting {dates[run_start].date()}")

    # --- by year -----------------------------------------------------------
    years = pd.Series(dates).dt.year.values
    print(f"\n{'year':<8}{'weeks':>7}{'inside':>8}{'coverage':>11}")
    print("-" * 34)
    for y in sorted(set(years)):
        sel = years == y
        k = int(inside[sel].sum())
        tot = int(sel.sum())
        flag = "  <-- low" if tot >= 10 and 100.0 * k / tot < nominal - 10 else ""
        print(f"{y:<8}{tot:>7}{k:>8}{100.0 * k / tot:>10.1f}%{flag}")

    # --- volatility regimes ------------------------------------------------
    returns = (closes - np.asarray(df["Open"].values, dtype=float)) / np.asarray(
        df["Open"].values, dtype=float)
    labels = volatility_terciles(returns)
    print(f"\ncoverage by trailing realised-volatility tercile")
    print(f"{'regime':<9}{'weeks':>7}{'inside':>8}{'coverage':>11}")
    print("-" * 35)
    for lab in ("low", "mid", "high"):
        sel = labels == lab
        if not sel.any():
            continue
        k, tot = int(inside[sel].sum()), int(sel.sum())
        print(f"{lab:<9}{tot:>7}{k:>8}{100.0 * k / tot:>10.1f}%")
    spread = None
    if (labels == "low").any() and (labels == "high").any():
        c_lo = 100.0 * inside[labels == "low"].mean()
        c_hi = 100.0 * inside[labels == "high"].mean()
        spread = c_lo - c_hi
        print(f"\n  low-vol minus high-vol coverage: {spread:+.1f} pp")
        if abs(spread) > 5:
            print("  -> a single time-invariant noise law cannot serve both regimes:")
            print("     bands are too wide when calm and too narrow when stressed,")
            print("     and the aggregate number is an average of two wrong ones.")

    # --- what caused each miss ---------------------------------------------
    att = attribute_misses(out)
    if att["n_miss"]:
        print(f"\nmiss attribution ({att['n_miss']} misses)")
        print(f"  interpolant in the wrong place   {att['n_interp_fault']:>4}"
              f"   ({100.0 * att['n_interp_fault'] / att['n_miss']:.0f}%)")
        print(f"  genuine tail event               {att['n_tail_event']:>4}"
              f"   ({100.0 * att['n_tail_event'] / att['n_miss']:.0f}%)")
        print(f"\n  interpolant error |fit-Open|/Open:")
        print(f"    all weeks   {att['interp_mape_all']:.3f}%")
        print(f"    miss weeks  {att['interp_mape_miss']:.3f}%")
        print(f"    hit weeks   {att['interp_mape_hit']:.3f}%")
        print(f"  coverage if bands were re-centred on the actual Open: "
              f"{att['coverage_recentred']:.2f}%")

    # --- direction of misses -----------------------------------------------
    n_above = int(np.sum(closes > high))
    n_below = int(np.sum(closes < low))
    if n_above + n_below:
        tot_m = n_above + n_below
        z_dir = (n_above - 0.5 * tot_m) / np.sqrt(0.25 * tot_m)
        print(f"\ndirection: {n_above} above the band, {n_below} below "
              f"({100.0 * n_above / tot_m:.0f}% above, z = {z_dir:+.2f} vs 50/50)")
        if abs(z_dir) > 2:
            print("  -> misses are one-sided, which is a bias in the band centre,")
            print("     not symmetric tail thinness.")

    # --- individual misses -------------------------------------------------
    miss_idx = np.flatnonzero(~inside)
    print(f"\nweeks outside the band: {miss_idx.size}")
    if miss_idx.size:
        half = 0.5 * (high - low)
        # how far outside, as a fraction of the band half-width
        excess = np.where(closes < low, (low - closes) / half, (closes - high) / half)
        order = miss_idx[np.argsort(-excess[miss_idx])]
        shown = order[:max_misses_listed]
        print(f"(worst {len(shown)}, by distance outside as a multiple of the "
              f"band half-width)\n")
        print(f"  {'date':<12}{'side':>7}{'close':>11}{'band':>21}{'excess':>9}")
        for i in shown:
            side = "below" if closes[i] < low[i] else "above"
            seg = "train" if i < n_fit else "held-out"
            print(f"  {str(dates[i].date()):<12}{side:>7}{closes[i]:>11.1f}"
                  f"{low[i]:>10.0f}-{high[i]:<10.0f}{excess[i]:>8.2f}x  {seg}")
        if order.size > len(shown):
            print(f"  ... and {order.size - len(shown)} more")

    print("\n" + "=" * 72)
    return {
        "inside": inside,
        "coverage": cov,
        "runs_test": rt,
        "autocorr": ac,
        "longest_miss_run": run_len,
        "miss_index": miss_idx,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run the propagation and report coverage diagnostics")
    parser.add_argument("--sample-every", type=int, default=1)
    parser.add_argument("--ncos", type=int, default=1024)
    parser.add_argument("--basis-iters", type=int, default=250)
    parser.add_argument("--basis-grid", type=int, default=1400)
    parser.add_argument("--m", type=int, default=2)
    parser.add_argument("--noise-fit-fraction", type=float, default=0.7)
    parser.add_argument("--legacy-noise", action="store_true")
    parser.add_argument("--noise-regimes", type=int, default=0,
                        help="fit one CTS law per trailing-volatility regime (try 3)")
    parser.add_argument("--regime-window", type=int, default=26)
    parser.add_argument("--save-plot", default=None)
    args = parser.parse_args()

    from main import main_propagation

    out = main_propagation(
        sample_every=args.sample_every,
        Ncos_per_week=args.ncos,
        basis_iters=args.basis_iters,
        basis_grid=args.basis_grid,
        m=args.m,
        fit_noise=not args.legacy_noise,
        noise_fit_fraction=args.noise_fit_fraction,
        show_plot=False,
        save_plot=args.save_plot,
        noise_regimes=args.noise_regimes,
        regime_window=args.regime_window,
    )
    print()
    coverage_report(out)

    if args.sample_every > 1:
        computed = len(range(0, len(out["df"]), args.sample_every))
        print(f"\nNOTE: --sample-every {args.sample_every} means only {computed} of "
              f"{len(out['df'])} weeks had quantiles computed; the rest were\n"
              f"linearly interpolated. Re-run with --sample-every 1 before quoting "
              f"these numbers.")


if __name__ == "__main__":
    main()
