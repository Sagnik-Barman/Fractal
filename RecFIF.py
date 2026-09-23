"""
RecFIF.py
=========

Recurrent fractal interpolation functions.

Each subinterval [t_{k-1}, t_k] receives an affine map from a local neighbourhood
window [t_l, t_r] (width controlled by ``m``) rather than from the whole domain as
in a classical Barnsley FIF. The vertical scaling factors ``d_vec`` are the free
parameters and control local roughness; |d_k| < 1 gives contraction, and the graph's
box dimension rises with sum|d_k|.

The fixed point is the unique continuous function satisfying

    f(phi_k(s)) = c_k*s + d_k*f(s) + e_k      for s in [t_l, t_r]

where phi_k(s) = a_k*s + b_k maps [t_l, t_r] onto [t_{k-1}, t_k], and (c_k, e_k)
are fixed by the two endpoint conditions.

Interpolation
-------------
Writing T for the operator above, evaluating at s = t_l gives

    (Tf)(t_{k-1}) = c_k*t_l + d_k*f(t_l) + e_k
                  = y_{k-1} + d_k * ( f(t_l) - y_l )

using the endpoint condition c_k*t_l + d_k*y_l + e_k = y_{k-1}. So T maps
interpolating functions to interpolating functions, T is a contraction with factor
max|d_k| < 1, and the set of interpolating functions is closed. Its unique fixed
point therefore interpolates the data exactly. That is a theorem about the exact
operator, and a numerical implementation only inherits it if f(t_l) is represented
*exactly* at the window endpoints.

The previous implementation evaluated on a uniform grid that did not contain the
knots, so f(t_l) carried an interpolation error, the residual d_k*(f(t_l) - y_l)
above was non-zero at every iteration, and the fixed point of the *implemented*
operator was not an interpolant. The error scaled with d_k and did not shrink with
more iterations or a finer grid:

    d_k      max |f(t_j) - y_j|, as % of data range
    0.00     0.09%       (grid resolution only)
    0.30     0.88%
    0.60     15.2%
    0.90     174.8%
    0.99     963.2%

``build_recurrent_fif_with_dvec`` below fixes this by (a) putting every knot on the
evaluation grid, (b) snapping pre-images that land on a knot to that knot exactly,
and (c) evaluating T in pull-back form so each grid point is written by exactly one
map, removing the deposit-and-average step and its zero-masking bug. Knot error is
then at machine precision for every d_k.

``build_recurrent_fif_legacy`` preserves the original behaviour so earlier results
can still be reproduced for comparison.
"""

import numpy as np
from scipy import interpolate

__all__ = [
    "compute_dk_perp_heuristic",
    "build_recurrent_fif_with_dvec",
    "build_recurrent_fif_legacy",
    "knot_interpolation_error",
]


def compute_dk_perp_heuristic(knots_t, knots_y, ell, r, sensitivity=0.5, dmin=0.05, dmax=0.99):
    """Heuristic starting value for a vertical scaling factor.

    Fits a line across the neighbourhood window and sets
    ``d_k = 1 - sensitivity * (max residual / y-range)``.

    Note on direction: larger |d_k| means a rougher graph (higher box dimension),
    so this assigns the *roughest* fractal to the *straightest* window. That is the
    upstream convention and is kept here for compatibility, but it is worth
    revisiting -- see the README.
    """
    t_dom = knots_t[ell:r+1]
    y_dom = knots_y[ell:r+1]
    if len(t_dom) < 2:
        return 0.5
    A = np.vstack([t_dom, np.ones_like(t_dom)]).T
    sol, *_ = np.linalg.lstsq(A, y_dom, rcond=None)
    alpha, beta = sol[0], sol[1]
    y_fit = alpha * t_dom + beta
    residuals = np.abs(y_dom - y_fit)
    max_res = np.max(residuals)
    y_range = np.max(y_dom) - np.min(y_dom)
    res_norm = max_res / y_range if y_range > 0 else 0.0
    d_k = 1.0 - sensitivity * res_norm
    return float(max(dmin, min(dmax, d_k)))


def _build_maps(knots_t, knots_y, d_vec, m):
    """Affine maps, one per subinterval, with endpoint conditions imposed."""
    K = len(knots_t)
    maps = []
    for k in range(1, K):
        ell = max(0, (k - 1) - m)
        r = min(K - 1, (k - 1) + m + 1)
        # The window must strictly contain the target interval for a_k < 1.
        if r <= ell:
            ell, r = max(0, k - 1), min(K - 1, k)
        t_l, t_r = float(knots_t[ell]), float(knots_t[r])
        span = t_r - t_l
        if span <= 0:
            raise ValueError(f"degenerate window for map {k}: [{t_l}, {t_r}]")
        a_k = (knots_t[k] - knots_t[k - 1]) / span
        b_k = knots_t[k - 1] - a_k * t_l
        d_k = float(d_vec[k - 1])
        y_l, y_r = float(knots_y[ell]), float(knots_y[r])
        A = np.array([[t_l, 1.0], [t_r, 1.0]])
        rhs = np.array([knots_y[k - 1] - d_k * y_l, knots_y[k] - d_k * y_r])
        try:
            sol = np.linalg.solve(A, rhs)
            c_k, e_k = float(sol[0]), float(sol[1])
        except np.linalg.LinAlgError:
            c_k, e_k = 0.0, float(knots_y[k - 1])
        maps.append({'k_idx': k, 'ell': ell, 'r': r,
                     'a': float(a_k), 'b': float(b_k),
                     'd': float(d_k), 'c': float(c_k), 'e': float(e_k),
                     't_l': t_l, 't_r': t_r})
    return maps


def build_recurrent_fif_with_dvec(knots_t, knots_y, d_vec, m=2, iters=300,
                                  grid_len=1500, verbose=False, tol=1e-12):
    """Build the recurrent FIF attractor. Interpolates the knots exactly.

    Returns ``(t_grid, f, f_fn, maps)``, matching the original signature.

    ``t_grid`` is a uniform grid of about ``grid_len`` points unioned with the knot
    locations, so every knot is represented exactly. Iteration stops when the
    sup-norm change falls below ``tol``.
    """
    knots_t = np.asarray(knots_t, dtype=float)
    knots_y = np.asarray(knots_y, dtype=float)
    d_vec = np.asarray(d_vec, dtype=float)

    K = len(knots_t)
    if len(d_vec) != K - 1:
        raise ValueError(f"d_vec has {len(d_vec)} entries; {K} knots need {K - 1}")
    if np.any(np.abs(d_vec) >= 1.0):
        raise ValueError("|d_k| must be < 1 for the operator to be a contraction")
    if np.any(np.diff(knots_t) <= 0):
        raise ValueError("knots_t must be strictly increasing")

    maps = _build_maps(knots_t, knots_y, d_vec, m)
    t0, tN = float(knots_t[0]), float(knots_t[-1])

    # --- grid containing every knot exactly --------------------------------
    t_grid = np.unique(np.concatenate([np.linspace(t0, tN, int(grid_len)), knots_t]))

    # --- assign each grid point to exactly one map -------------------------
    # Interval j covers [t_j, t_{j+1}); the final point belongs to the last map.
    owner = np.searchsorted(knots_t, t_grid, side='right') - 1
    owner = np.clip(owner, 0, K - 2)

    # --- pre-images, with knot snapping ------------------------------------
    a_arr = np.array([mp['a'] for mp in maps])[owner]
    b_arr = np.array([mp['b'] for mp in maps])[owner]
    c_arr = np.array([mp['c'] for mp in maps])[owner]
    d_arr = np.array([mp['d'] for mp in maps])[owner]
    e_arr = np.array([mp['e'] for mp in maps])[owner]

    pre_s = (t_grid - b_arr) / a_arr
    pre_s = np.clip(pre_s, t0, tN)

    # A pre-image that lands on a knot must read that knot's value exactly, or the
    # residual d_k*(f(t_l) - y_l) reappears at the 1e-16 level and is amplified by
    # the iteration. Snap anything within a tight tolerance onto the exact knot.
    scale = max(tN - t0, 1e-300)
    j_near = np.searchsorted(knots_t, pre_s)
    for cand in (j_near - 1, j_near):
        ok = (cand >= 0) & (cand < K)
        idx = np.where(ok, cand, 0)
        close = ok & (np.abs(pre_s - knots_t[idx]) < 1e-9 * scale)
        pre_s = np.where(close, knots_t[idx], pre_s)

    # --- iterate to the fixed point ----------------------------------------
    f = np.interp(t_grid, knots_t, knots_y)   # starts interpolating; stays that way
    maxdiff = np.inf
    for it in range(int(iters)):
        f_s = np.interp(pre_s, t_grid, f)
        f_new = c_arr * pre_s + d_arr * f_s + e_arr
        maxdiff = float(np.max(np.abs(f_new - f)))
        f = f_new
        if verbose and (it % 50 == 0 or it == iters - 1):
            print(f"Iter {it+1}/{iters}, maxdiff={maxdiff:.6g}")
        if maxdiff < tol:
            if verbose:
                print(f"Converged after {it+1} iterations (maxdiff={maxdiff:.3g})")
            break

    f_fn = interpolate.interp1d(t_grid, f, bounds_error=False, fill_value="extrapolate")
    return t_grid, f, f_fn, maps


def build_recurrent_fif_legacy(knots_t, knots_y, d_vec, m=2, iters=300,
                               grid_len=1500, verbose=False):
    """The original deposit-and-average implementation, kept for reproduction.

    Does not interpolate its own knots; see the module docstring. Use only to
    reproduce results produced before the fix.
    """
    K = len(knots_t)
    t0, tN = knots_t[0], knots_t[-1]
    maps = _build_maps(np.asarray(knots_t, float), np.asarray(knots_y, float),
                       np.asarray(d_vec, float), m)
    t_grid = np.linspace(t0, tN, grid_len)
    f = np.interp(t_grid, knots_t, knots_y)
    domain_t_arrays, mapped_x_arrays = [], []
    for mp in maps:
        mask = (t_grid >= mp['t_l']) & (t_grid <= mp['t_r'])
        domain_t = t_grid[mask]
        domain_t_arrays.append((mask, domain_t))
        mapped_x_arrays.append(mp['a'] * domain_t + mp['b'])
    for it in range(iters):
        f_new = np.zeros_like(f); count = np.zeros_like(f)
        for idx_map, mp in enumerate(maps):
            mask, domain_t = domain_t_arrays[idx_map]
            x = mapped_x_arrays[idx_map]
            d_k, c_k, e_k = mp['d'], mp['c'], mp['e']
            f_s = np.interp(domain_t, t_grid, f)
            v = c_k * domain_t + d_k * f_s + e_k
            deposited = np.interp(t_grid, x, v, left=0.0, right=0.0)
            contributed_mask = deposited != 0.0
            f_new += deposited
            count += contributed_mask.astype(float)
        nonzero = count > 0
        f_new[nonzero] = f_new[nonzero] / count[nonzero]
        f_new[~nonzero] = f[~nonzero]
        maxdiff = np.max(np.abs(f_new - f))
        f = f_new
        if verbose and (it % 50 == 0 or it == iters-1):
            print(f"Iter {it+1}/{iters}, maxdiff={maxdiff:.6g}")
        if maxdiff < 1e-9:
            break
    f_fn = interpolate.interp1d(t_grid, f, bounds_error=False, fill_value="extrapolate")
    return t_grid, f, f_fn, maps


def knot_interpolation_error(f_fn, knots_t, knots_y):
    """max, mean and relative max of |f(t_j) - y_j| over the knots.

    For a correct FIF the maximum should sit at machine precision.
    """
    knots_t = np.asarray(knots_t, dtype=float)
    knots_y = np.asarray(knots_y, dtype=float)
    err = np.abs(np.asarray([float(f_fn(t)) for t in knots_t]) - knots_y)
    rng = float(np.ptp(knots_y)) or 1.0
    return {"max": float(err.max()), "mean": float(err.mean()),
            "max_pct_of_range": float(100.0 * err.max() / rng)}
