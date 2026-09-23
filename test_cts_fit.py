"""
test_cts_fit.py
===============

Tests for the CTS parameter estimation added in ``cts_fit.py``, plus an
end-to-end smoke test of ``main_propagation``.

Unlike ``test_flow.py`` these tests do not touch the network: the price fetch is
replaced with a synthetic series, so the suite runs offline and deterministically.

    python test_cts_fit.py

Author: Sagnik Barman, 2026.
"""

import sys
import warnings

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # no display needed

import utils
from RecFIF import compute_dk_perp_heuristic
from cts_fit import (
    cts_cumulants,
    cts_rvs,
    fit_cts,
    moment_match_init,
    neg_loglik_params,
    sample_cumulants,
    self_test as cts_self_test,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def synthetic_weekly(n=140, seed=7, heavy=False):
    """A NIFTY-shaped weekly Open/Close frame with no network access."""
    rng = np.random.default_rng(seed)
    if heavy:
        # Student-t innovations give the leptokurtosis CTS is designed for.
        shocks = 0.012 * rng.standard_t(3.0, n)
        close_shocks = 0.010 * rng.standard_t(3.0, n)
    else:
        shocks = rng.normal(0.002, 0.025, n)
        close_shocks = rng.normal(0.001, 0.022, n)
    op = np.exp(np.cumsum(shocks) + np.log(5000.0))
    cl = op * (1.0 + close_shocks)
    dates = pd.date_range("2015-01-02", periods=n, freq="W-FRI")
    return pd.DataFrame({"Date": dates, "Open": op, "Close": cl,
                         "t": np.arange(n) / (n - 1)})


def heuristic_d_vec(df, m=2):
    idx = utils.extract_extrema(df["Open"].values, order=1)
    K = len(idx)
    kt = df["t"].values[idx]
    ky = df["Open"].values[idx]
    d_vec = np.array([
        compute_dk_perp_heuristic(kt, ky, max(0, (k - 1) - m), min(K - 1, (k - 1) + m + 1))
        for k in range(1, K)
    ])
    return idx, d_vec


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

def test_cumulants_and_recovery():
    """Closed-form cumulants match numerical CF derivatives; estimator recovers moments."""
    return cts_self_test(verbose=True)


def test_moment_match_is_finite_on_light_tails():
    """A platykurtic sample must not send the initialiser to the Gaussian limit."""
    rng = np.random.default_rng(11)
    x = rng.uniform(-0.05, 0.05, 500)          # excess kurtosis < 0
    c1, c2, c3, c4 = sample_cumulants(x)
    assert c4 < 0, "fixture should be platykurtic"

    params = moment_match_init(x, alpha=1.4)
    alpha, C_p, C_m, lam_p, lam_m, mu = params
    sd = float(np.std(x))
    ok = np.all(np.isfinite(params)) and lam_p < 100.0 / sd and C_p > 0 and C_m > 0
    print(f"  light tails: lam={lam_p:.3f} (cap {100.0 / sd:.1f})  "
          f"C+={C_p:.3e}  C-={C_m:.3e}  {'OK' if ok else 'FAIL'}")
    return bool(ok)


def test_fit_reproduces_sample_moments():
    """On heavy-tailed data the fit should match the sample scale and location."""
    true_params = (1.45, 3.0e-3, 2.2e-3, 10.0, 12.0, 2e-4)
    x = cts_rvs(true_params, size=3000, random_state=4242)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = fit_cts(x, refine=False, verbose=False)

    c1, c2, _, _ = sample_cumulants(x)
    fc = cts_cumulants(fit.params)
    sd_err = abs(np.sqrt(fc[1]) - np.sqrt(c2)) / np.sqrt(c2)
    mean_gap = abs(fc[0] - c1) / np.sqrt(c2)
    ok = sd_err < 0.15 and mean_gap < 1.0 and fit.converged
    print(f"  sd rel err={sd_err:.4f}  mean gap={mean_gap:.3f} sd  "
          f"converged={fit.converged}  {'OK' if ok else 'FAIL'}")
    return bool(ok)


def test_mle_never_worse_than_ecf():
    """The refinement must only be accepted when it improves the likelihood."""
    true_params = (1.5, 2.0e-3, 2.0e-3, 11.0, 11.0, 0.0)
    x = cts_rvs(true_params, size=800, random_state=99)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ecf_only = fit_cts(x, refine=False, verbose=False)
        refined = fit_cts(x, refine=True, verbose=False)

    nll_ecf = neg_loglik_params(ecf_only.params, x)
    nll_ref = neg_loglik_params(refined.params, x)
    ok = nll_ref <= nll_ecf + 1e-6
    print(f"  neg-loglik ECF={nll_ecf:.4f}  after refinement={nll_ref:.4f}  "
          f"{'OK' if ok else 'FAIL'}")
    return bool(ok)


def test_fetch_raises_on_empty(monkeypatched=True):
    """An empty download must raise a clear error, not a KeyError on 'Date'."""
    import types
    original = utils.yf.download
    try:
        utils.yf.download = lambda *a, **k: pd.DataFrame()
        try:
            utils.fetch_weekly_nifty()
        except RuntimeError as exc:
            ok = "No data returned" in str(exc)
            print(f"  raised RuntimeError: {str(exc)[:60]}...  {'OK' if ok else 'FAIL'}")
            return bool(ok)
        except Exception as exc:
            print(f"  raised {type(exc).__name__} instead of RuntimeError  FAIL")
            return False
        print("  no exception raised  FAIL")
        return False
    finally:
        utils.yf.download = original


def test_end_to_end():
    """main_propagation runs, fits noise, and reports held-out coverage."""
    df = synthetic_weekly(n=140, seed=7, heavy=True)
    idx, d_vec = heuristic_d_vec(df)

    utils.fetch_weekly_nifty = lambda *a, **k: df
    import main
    main.fetch_weekly_nifty = lambda *a, **k: df

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = main.main_propagation(
            sample_every=8, Ncos_per_week=512, basis_iters=40, basis_grid=250,
            m=2, d_vec=d_vec, noise_fit_fraction=0.7, show_plot=False,
        )

    ok = (
        out["cts_fit"] is not None
        and 0.0 <= out["coverage"] <= 100.0
        and np.isfinite(out["coverage_test"])
        and len(out["params_noise"]) == 6
    )
    print(f"  coverage={out['coverage']:.2f}%  "
          f"train={out['coverage_train']:.2f}%  test={out['coverage_test']:.2f}%  "
          f"{'OK' if ok else 'FAIL'}")
    return bool(ok)


TESTS = [
    ("cumulants and parameter recovery", test_cumulants_and_recovery),
    ("moment init on light tails", test_moment_match_is_finite_on_light_tails),
    ("fit reproduces sample moments", test_fit_reproduces_sample_moments),
    ("MLE refinement never worse", test_mle_never_worse_than_ecf),
    ("empty download raises clearly", test_fetch_raises_on_empty),
    ("end-to-end propagation", test_end_to_end),
]


def main():
    print("=" * 68)
    print("cts_fit test suite")
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
