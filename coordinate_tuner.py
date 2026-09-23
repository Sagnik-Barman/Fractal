"""
coordinate_tuner.py
===================

Greedy coordinate descent over the vertical scaling factors ``d_vec``, minimising
MAPE of the interpolant against the Open series.

Because ``1.0`` is among the candidate factors, each coordinate update is
non-worsening, so the objective is monotone non-increasing along a run. That
guarantee is only relative to the *starting point*: the method is greedy and will
settle in whichever local optimum the start is nearest. Starting from the
heuristic reaches a materially worse optimum on NIFTY than starting from a
previously tuned vector, which is why ``init_d`` exists.

The objective is also evaluated at a cheap resolution (``fast_iters`` /
``fast_grid``) that is not the resolution used to report results, so the ranking
being optimised is a proxy. Verify a tuned vector at full resolution before
adopting it -- ``retune.py`` does this and refuses to write a vector that scores
worse than the one it replaces.
"""

import numpy as np

from RecFIF import compute_dk_perp_heuristic, build_recurrent_fif_with_dvec

__all__ = ["simple_tuner", "heuristic_d_vec", "mape_of"]


def heuristic_d_vec(knots_t, knots_y, m=2, sensitivity=0.5, dmin=0.05, dmax=0.99):
    """The per-window heuristic starting vector."""
    K = len(knots_t)
    return np.array([
        compute_dk_perp_heuristic(knots_t, knots_y, max(0, (k - 1) - m), min(K - 1, (k - 1) + m + 1),
                                  sensitivity=sensitivity, dmin=dmin, dmax=dmax)
        for k in range(1, K)
    ])


def mape_of(d_vec, knots_t, knots_y, df, m=2, iters=100, grid_len=800) -> float:
    """MAPE of the interpolant against the Open series, in percent."""
    _, _, f_fn, _ = build_recurrent_fif_with_dvec(
        knots_t, knots_y, d_vec, m=m, iters=iters, grid_len=grid_len)
    fit = f_fn(df['t'].values)
    actual = np.asarray(df['Open'].values, dtype=float)
    return float(np.mean(np.abs(fit - actual) / actual) * 100.0)


def simple_tuner(knots_t, knots_y, df, m=2, init_sensitivity=0.5, dmin=0.05, dmax=0.99,
                 passes=1, candidate_factors=(0.75, 0.9, 1.0, 1.1),
                 fast_iters=100, fast_grid=800, init_d=None, verbose=True):
    """Greedy coordinate descent on ``d_vec``.

    Parameters
    ----------
    init_d : array_like, optional
        Starting vector. Defaults to the per-window heuristic. Pass a previously
        tuned vector to warm-start -- the descent is greedy, so the starting point
        determines which local optimum is reached, and a warm start cannot end up
        worse than where it began (1.0 is always among the candidates).
    passes : int
        Number of sweeps over all coordinates.
    """
    knots_t = np.asarray(knots_t, dtype=float)
    knots_y = np.asarray(knots_y, dtype=float)

    if init_d is None:
        d_vec = heuristic_d_vec(knots_t, knots_y, m=m,
                                sensitivity=init_sensitivity, dmin=dmin, dmax=dmax)
        source = "heuristic"
    else:
        d_vec = np.clip(np.asarray(init_d, dtype=float).copy(), dmin, dmax)
        source = "supplied vector"
        if len(d_vec) != len(knots_t) - 1:
            raise ValueError(f"init_d has {len(d_vec)} entries; "
                             f"{len(knots_t)} knots need {len(knots_t) - 1}")

    base_mape = mape_of(d_vec, knots_t, knots_y, df, m=m,
                        iters=fast_iters, grid_len=fast_grid)
    if verbose:
        print(f"Tuner: starting from {source}, MAPE (fast) = {base_mape:.6f}%")

    n = len(d_vec)
    for p in range(int(passes)):
        improved = 0
        for k in range(n):
            best, best_mape = d_vec[k], base_mape
            for fac in candidate_factors:
                cand = float(np.clip(d_vec[k] * fac, dmin, dmax))
                if cand == d_vec[k]:
                    continue
                d_try = d_vec.copy()
                d_try[k] = cand
                mape_try = mape_of(d_try, knots_t, knots_y, df, m=m,
                                   iters=fast_iters, grid_len=fast_grid)
                if mape_try < best_mape:
                    best_mape, best = mape_try, cand
            if best != d_vec[k]:
                improved += 1
            d_vec[k] = best
            base_mape = best_mape
        if verbose:
            print(f"Tuner: pass {p+1}/{passes} -- MAPE (fast) = {base_mape:.6f}%  "
                  f"({improved}/{n} coordinates moved)")
        if improved == 0:
            if verbose:
                print("Tuner: no coordinate moved; converged.")
            break

    return d_vec
