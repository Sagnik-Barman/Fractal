"""
regime_fit.py
=============

Regime-conditioned CTS estimation.

The problem this addresses
-------------------------
A single time-invariant tempered-stable law cannot calibrate across market
regimes. Measured on weekly NIFTY with one pooled fit, the 95% bands achieve

    low volatility    100.0%   (not one week in 250 outside the band)
    mid volatility     99.2%
    high volatility    84.4%

a 15.6 pp spread that averages to a respectable-looking 94.53% overall. The
aggregate is the mean of two opposite failures.

The fix tested here is to fit the noise separately per volatility regime and let
each week draw its band from the regime it is actually in.

Leakage
-------
Two things must not peek at the future, or the exercise is worthless:

* **Regime labels.** Volatility is a *trailing* rolling standard deviation, so
  week t is labelled using returns up to and including t and nothing after.
* **Thresholds.** The tercile cut points are computed on the training slice only
  and then applied unchanged to the held-out period. Computing them over the full
  sample would leak the future distribution of volatility into every label.

Both are enforced by ``assign_regimes(..., thresholds=...)``: fit thresholds on
the training slice, pass them back in for the full series.

Small regimes
-------------
Splitting 524 training weeks three ways leaves roughly 175 per regime, which is
enough for the characteristic-function estimator but not generous. Any regime
with fewer than ``min_obs`` observations, or whose fit fails its sanity gates,
falls back to the pooled fit rather than producing a degenerate one.

Author: Sagnik Barman, 2026.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from cts_fit import fit_cts

__all__ = [
    "REGIME_NAMES",
    "trailing_volatility",
    "regime_thresholds",
    "assign_regimes",
    "fit_cts_by_regime",
    "params_per_week",
]

REGIME_NAMES = ("low", "mid", "high")


def trailing_volatility(returns, window: int = 26) -> np.ndarray:
    """Backward-looking rolling standard deviation of returns.

    Week ``t`` uses returns up to and including ``t``. The leading weeks, before a
    full window exists, use whatever history is available (at least a quarter of
    the window) and are back-filled from the first valid value.
    """
    r = pd.Series(np.asarray(returns, dtype=float))
    vol = r.rolling(window, min_periods=max(4, window // 4)).std()
    return vol.bfill().values


def regime_thresholds(vol, n_regimes: int = 3) -> np.ndarray:
    """Quantile cut points of ``vol``. Compute these on the TRAINING slice only."""
    qs = np.linspace(0.0, 100.0, n_regimes + 1)[1:-1]
    return np.percentile(np.asarray(vol, dtype=float), qs)


def assign_regimes(returns, window: int = 26, thresholds=None, n_regimes: int = 3):
    """Label each week by trailing-volatility regime.

    Returns ``(labels, thresholds, vol)`` where ``labels`` are integers
    ``0 .. n_regimes-1`` in increasing volatility order.

    Pass ``thresholds`` from the training slice to score a later period without
    leaking its volatility distribution into the labels.
    """
    vol = trailing_volatility(returns, window=window)
    if thresholds is None:
        thresholds = regime_thresholds(vol, n_regimes=n_regimes)
    thresholds = np.asarray(thresholds, dtype=float)
    labels = np.searchsorted(thresholds, vol, side="right").astype(int)
    labels = np.clip(labels, 0, n_regimes - 1)
    return labels, thresholds, vol


def fit_cts_by_regime(returns, labels, n_regimes: int = 3, min_obs: int = 60,
                      pooled=None, verbose: bool = True, **fit_kwargs) -> dict:
    """Fit CTS parameters separately within each regime.

    Returns ``{regime_index: params_tuple}``. A regime with fewer than
    ``min_obs`` observations, or whose fit fails its sanity checks, takes the
    pooled parameters instead.

    ``pooled`` may be supplied to avoid refitting it; otherwise it is fitted here.
    """
    returns = np.asarray(returns, dtype=float)
    labels = np.asarray(labels, dtype=int)

    if pooled is None:
        if verbose:
            print("\n[regime] pooled fit (fallback and comparison):")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pooled_res = fit_cts(returns, verbose=verbose, **fit_kwargs)
        pooled = pooled_res.params

    out, diagnostics = {}, {}
    for g in range(n_regimes):
        sel = labels == g
        n = int(sel.sum())
        name = REGIME_NAMES[g] if g < len(REGIME_NAMES) else f"regime{g}"
        if n < min_obs:
            if verbose:
                print(f"\n[regime] {name}: only {n} observations "
                      f"(< {min_obs}) -- using the pooled fit")
            out[g] = tuple(pooled)
            diagnostics[g] = None
            continue
        if verbose:
            print(f"\n[regime] {name}: fitting on {n} weeks")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = fit_cts(returns[sel], verbose=verbose, **fit_kwargs)
        if not res.converged:
            if verbose:
                print(f"[regime] {name}: fit failed its sanity checks "
                      f"({res.message}) -- using the pooled fit")
            out[g] = tuple(pooled)
            diagnostics[g] = res
            continue
        out[g] = tuple(res.params)
        diagnostics[g] = res

    if verbose:
        print("\n[regime] fitted parameters by regime:")
        print(f"  {'regime':<8}{'n':>6}{'alpha':>9}{'lam+':>10}{'lam-':>10}"
              f"{'sd':>11}{'KS p':>8}")
        from cts_fit import cts_cumulants
        for g in range(n_regimes):
            name = REGIME_NAMES[g] if g < len(REGIME_NAMES) else f"regime{g}"
            p = out[g]
            sd = np.sqrt(cts_cumulants(p)[1])
            d = diagnostics.get(g)
            ksp = f"{d.ks_pvalue:.3f}" if d is not None and np.isfinite(d.ks_pvalue) else "-"
            print(f"  {name:<8}{int((labels == g).sum()):>6}{p[0]:>9.4f}"
                  f"{p[3]:>10.3f}{p[4]:>10.3f}{sd:>11.5f}{ksp:>8}")
        print(f"  {'pooled':<8}{len(returns):>6}{pooled[0]:>9.4f}"
              f"{pooled[3]:>10.3f}{pooled[4]:>10.3f}"
              f"{np.sqrt(cts_cumulants(pooled)[1]):>11.5f}")

    out["_pooled"] = tuple(pooled)
    out["_diagnostics"] = diagnostics
    return out


def params_per_week(labels, regime_params) -> list:
    """Expand ``{regime: params}`` into one parameter tuple per week."""
    labels = np.asarray(labels, dtype=int)
    pooled = regime_params.get("_pooled")
    return [tuple(regime_params.get(int(g), pooled)) for g in labels]
