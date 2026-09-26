"""
test_regime_fit.py
==================

Tests for regime-conditioned CTS estimation.

The properties that matter are not about accuracy, they are about not cheating:
labels must be computable in real time, and the tercile thresholds must not have
seen the held-out period. A regime model that leaks is worse than no regime model,
because it looks better while being useless.

Offline and deterministic.

    python test_regime_fit.py

Author: Sagnik Barman, 2026.
"""

import sys
import warnings

import numpy as np

from regime_fit import (
    assign_regimes,
    fit_cts_by_regime,
    params_per_week,
    regime_thresholds,
    trailing_volatility,
)


def _two_regime_returns(n=400, seed=3):
    """Returns whose volatility alternates between two known levels."""
    rng = np.random.default_rng(seed)
    scale = np.where((np.arange(n) % 100) < 40, 0.030, 0.012)
    return scale * rng.standard_t(5.0, n), scale


def test_volatility_is_backward_looking():
    """Week t's volatility must not change when the future changes."""
    r, _ = _two_regime_returns()
    v_full = trailing_volatility(r, window=26)
    cut = 250
    v_trunc = trailing_volatility(r[:cut], window=26)
    err = float(np.max(np.abs(v_full[:cut] - v_trunc)))
    ok = err < 1e-12
    print(f"  max |vol(full)[:t] - vol(truncated)| = {err:.3e}  {'OK' if ok else 'FAIL'}")
    return bool(ok)


def test_thresholds_do_not_leak():
    """Labels for held-out weeks must use training thresholds only."""
    r, _ = _two_regime_returns()
    n_fit = 280
    _, thr_train, _ = assign_regimes(r[:n_fit], window=26)
    labels_a, _, _ = assign_regimes(r, window=26, thresholds=thr_train)

    # Perturbing the held-out tail must not change any training-period label.
    r2 = r.copy()
    r2[n_fit:] *= 5.0
    _, thr_train2, _ = assign_regimes(r2[:n_fit], window=26)
    labels_b, _, _ = assign_regimes(r2, window=26, thresholds=thr_train2)

    same_thr = np.allclose(thr_train, thr_train2)
    same_lab = np.array_equal(labels_a[:n_fit], labels_b[:n_fit])
    ok = same_thr and same_lab
    print(f"  thresholds unchanged by future data: {same_thr}")
    print(f"  training labels unchanged: {same_lab}  {'OK' if ok else 'FAIL'}")
    return bool(ok)


def test_regimes_track_true_volatility():
    """High-vol weeks should mostly land in the high regime."""
    r, scale = _two_regime_returns()
    labels, _, _ = assign_regimes(r, window=26)
    hi_true = scale > 0.02
    # allow lag: the rolling window takes time to notice a regime change
    frac = float(np.mean(labels[hi_true] == 2))
    ok = frac > 0.5
    print(f"  {100*frac:.0f}% of genuinely high-vol weeks labelled 'high'  "
          f"{'OK' if ok else 'FAIL'}")
    return bool(ok)


def test_small_regime_falls_back_to_pooled():
    """A regime with too few observations must reuse the pooled fit, not fail."""
    r, _ = _two_regime_returns(n=200)
    labels = np.zeros(len(r), dtype=int)
    labels[:5] = 2          # a deliberately tiny 'high' regime
    pooled = (1.45, 1e-3, 1e-3, 12.0, 10.0, 0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = fit_cts_by_regime(r, labels, n_regimes=3, min_obs=60,
                                pooled=pooled, verbose=False)
    ok = out[2] == tuple(pooled) and out[1] == tuple(pooled)
    print(f"  tiny and empty regimes both fell back to pooled  {'OK' if ok else 'FAIL'}")
    return bool(ok)


def test_params_per_week_shape():
    """One parameter tuple per week, drawn from that week's regime."""
    r, _ = _two_regime_returns(n=150)
    labels, _, _ = assign_regimes(r, window=26)
    fake = {0: (1.4, 1e-3, 1e-3, 10.0, 10.0, 0.0),
            1: (1.5, 2e-3, 2e-3, 11.0, 11.0, 0.0),
            2: (1.6, 3e-3, 3e-3, 12.0, 12.0, 0.0),
            "_pooled": (1.45, 1e-3, 1e-3, 9.0, 9.0, 0.0)}
    pw = params_per_week(labels, fake)
    ok = len(pw) == len(r) and all(pw[i] == fake[labels[i]] for i in range(len(r)))
    print(f"  {len(pw)} tuples for {len(r)} weeks, each matching its regime  "
          f"{'OK' if ok else 'FAIL'}")
    return bool(ok)


TESTS = [
    ("volatility is backward-looking", test_volatility_is_backward_looking),
    ("thresholds do not leak", test_thresholds_do_not_leak),
    ("regimes track true volatility", test_regimes_track_true_volatility),
    ("small regime falls back to pooled", test_small_regime_falls_back_to_pooled),
    ("params_per_week mapping", test_params_per_week_shape),
]


def main():
    print("=" * 68)
    print("regime_fit test suite")
    print("=" * 68)
    passed = 0
    for i, (name, fn) in enumerate(TESTS, 1):
        print(f"\n[{i}/{len(TESTS)}] {name}")
        try:
            if fn():
                passed += 1
            else:
                print("  -> FAILED")
        except Exception as exc:
            print(f"  -> ERROR: {type(exc).__name__}: {exc}")
    print("\n" + "=" * 68)
    print(f"Passed: {passed}/{len(TESTS)}")
    print("=" * 68)
    return 0 if passed == len(TESTS) else 1


if __name__ == "__main__":
    sys.exit(main())
