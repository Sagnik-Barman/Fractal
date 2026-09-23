"""
test_recfif.py
==============

Correctness tests for the recurrent FIF construction.

The defining property of a fractal interpolation function is that it passes
through its data. The original implementation did not: knot error grew with
``d_k`` and did not shrink with more iterations or a finer grid. These tests pin
the fixed behaviour so the defect cannot return silently.

Offline and deterministic.

    python test_recfif.py

Author: Sagnik Barman, 2026.
"""

import sys

import numpy as np

from RecFIF import (
    build_recurrent_fif_with_dvec,
    build_recurrent_fif_legacy,
    compute_dk_perp_heuristic,
    knot_interpolation_error,
)

SEED = 1
K = 25


def _fixture():
    rng = np.random.default_rng(SEED)
    kt = np.linspace(0.0, 1.0, K)
    ky = np.cumsum(rng.normal(0.0, 1.0, K)) + 10.0
    return kt, ky


def _heuristic_dvec(kt, ky, m=2):
    return np.array([
        compute_dk_perp_heuristic(kt, ky, max(0, (k - 1) - m), min(K - 1, (k - 1) + m + 1))
        for k in range(1, K)
    ])


def test_interpolates_at_every_d():
    """f(t_j) = y_j to machine precision, for every admissible d_k."""
    kt, ky = _fixture()
    ok = True
    for dval in (0.0, 0.3, 0.6, 0.9, 0.99):
        _, _, fn, _ = build_recurrent_fif_with_dvec(
            kt, ky, np.full(K - 1, dval), m=2, iters=600, grid_len=3000)
        err = knot_interpolation_error(fn, kt, ky)
        good = err["max"] < 1e-10
        ok &= good
        print(f"  d={dval:<5} max knot error = {err['max']:.3e}  {'OK' if good else 'FAIL'}")
    return bool(ok)


def test_zero_d_is_linear_interpolation():
    """d_k = 0 must collapse exactly to piecewise linear interpolation."""
    kt, ky = _fixture()
    _, _, fn, _ = build_recurrent_fif_with_dvec(
        kt, ky, np.zeros(K - 1), m=2, iters=400, grid_len=2000)
    ts = np.linspace(0.0, 1.0, 377)
    err = float(np.max(np.abs(np.array([float(fn(t)) for t in ts]) - np.interp(ts, kt, ky))))
    ok = err < 1e-10
    print(f"  max |f - linear| = {err:.3e}  {'OK' if ok else 'FAIL'}")
    return bool(ok)


def test_basis_is_a_delta_basis():
    """omega_i(t_j) = delta_ij. The weighted-CF noise propagation depends on this."""
    kt, ky = _fixture()
    dv = _heuristic_dvec(kt, ky)
    rows = []
    for i in range(K):
        yb = np.zeros(K)
        yb[i] = 1.0
        _, _, fn, _ = build_recurrent_fif_with_dvec(kt, yb, dv, m=2, iters=400, grid_len=1200)
        rows.append([float(fn(t)) for t in kt])
    err = float(np.abs(np.array(rows) - np.eye(K)).max())
    ok = err < 1e-10
    print(f"  max |Omega - I| = {err:.3e}  {'OK' if ok else 'FAIL'}")
    return bool(ok)


def test_partition_of_unity_and_linearity():
    """sum_i omega_i = 1, and sum_i omega_i*y_i reproduces a direct build."""
    kt, ky = _fixture()
    dv = _heuristic_dvec(kt, ky)
    basis = []
    for i in range(K):
        yb = np.zeros(K)
        yb[i] = 1.0
        _, _, fn, _ = build_recurrent_fif_with_dvec(kt, yb, dv, m=2, iters=400, grid_len=1200)
        basis.append(fn)
    _, _, direct, _ = build_recurrent_fif_with_dvec(kt, ky, dv, m=2, iters=400, grid_len=1200)

    sums, lins = [], []
    for t in np.linspace(0.0, 1.0, 50):
        w = np.array([float(f(t)) for f in basis])
        sums.append(abs(w.sum() - 1.0))
        lins.append(abs(float((w * ky).sum()) - float(direct(t))))
    ok = max(sums) < 1e-8 and max(lins) < 1e-7
    print(f"  max |sum(omega) - 1| = {max(sums):.3e}   "
          f"max linearity error = {max(lins):.3e}  {'OK' if ok else 'FAIL'}")
    return bool(ok)


def test_roughness_increases_with_d():
    """Still a fractal: total variation must grow with the scaling factors."""
    kt, ky = _fixture()
    tvs = []
    for dval in (0.0, 0.3, 0.6, 0.9):
        _, fv, _, _ = build_recurrent_fif_with_dvec(
            kt, ky, np.full(K - 1, dval), m=2, iters=400, grid_len=4000)
        tvs.append(float(np.sum(np.abs(np.diff(fv)))))
    ok = all(b > a for a, b in zip(tvs, tvs[1:]))
    print(f"  total variation: " + " -> ".join(f"{v:.1f}" for v in tvs) +
          f"  {'OK' if ok else 'FAIL'}")
    return bool(ok)


def test_contraction_is_enforced():
    """|d_k| >= 1 is not a contraction and must be rejected, not silently run."""
    kt, ky = _fixture()
    try:
        build_recurrent_fif_with_dvec(kt, ky, np.full(K - 1, 1.0), m=2, iters=10, grid_len=200)
    except ValueError as exc:
        ok = "contraction" in str(exc)
        print(f"  raised ValueError: {str(exc)[:52]}...  {'OK' if ok else 'FAIL'}")
        return bool(ok)
    print("  no exception raised  FAIL")
    return False


def test_legacy_still_reproduces_the_defect():
    """The kept-for-reproduction path should still show the original behaviour."""
    kt, ky = _fixture()
    _, _, fn, _ = build_recurrent_fif_legacy(
        kt, ky, np.full(K - 1, 0.9), m=2, iters=600, grid_len=3000)
    err = knot_interpolation_error(fn, kt, ky)
    ok = err["max_pct_of_range"] > 1.0
    print(f"  legacy knot error at d=0.9: {err['max_pct_of_range']:.1f}% of range  "
          f"{'OK (defect preserved for comparison)' if ok else 'FAIL'}")
    return bool(ok)


TESTS = [
    ("interpolates at every d", test_interpolates_at_every_d),
    ("d=0 is linear interpolation", test_zero_d_is_linear_interpolation),
    ("basis is a delta basis", test_basis_is_a_delta_basis),
    ("partition of unity and linearity", test_partition_of_unity_and_linearity),
    ("roughness increases with d", test_roughness_increases_with_d),
    ("contraction condition enforced", test_contraction_is_enforced),
    ("legacy path reproduces the defect", test_legacy_still_reproduces_the_defect),
]


def main():
    print("=" * 68)
    print("RecFIF correctness suite")
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
