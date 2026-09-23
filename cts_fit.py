"""
cts_fit.py
==========

Estimation of Classical Tempered Stable (CTS / CGMY) parameters from return data.

Motivation
----------
The original pipeline in ``main.py`` looked for a variable ``params_hat`` in the
module globals and, when it was not found, silently fell back to a hardcoded
parameter vector::

    (alpha, C_plus, C_minus, lam_p, lam_m, mu) = (1.4, 1e-3, 1e-3, 9.5, 9.5, mean)

Because ``params_hat`` was never actually defined anywhere in the repository,
every run used those fixed numbers. The confidence bands produced downstream
were therefore driven by constants rather than by anything estimated from the
NIFTY series. This module replaces that fallback with real estimation.

Three estimators are provided, in increasing order of cost:

1. ``moment_match_init``  -- closed-form method of moments from the first four
   sample cumulants. Fast, no optimisation, used as a starting point.
2. ``fit_cts_ecf``        -- weighted least squares between the empirical and
   model characteristic functions. Robust and fast; does not require evaluating
   a density.
3. ``fit_cts_mle``        -- maximum likelihood, where the density is
   reconstructed from the characteristic function with the COS method already
   present in ``COS_inversion_helper.py``.

``fit_cts`` runs 1 -> 2 -> 3 and returns the refined estimate along with a
Kolmogorov-Smirnov goodness-of-fit statistic.

Parameterisation
----------------
The characteristic exponent implemented in ``cts_char.py`` is

    psi(u) = i*mu*u + Gamma(-alpha) * [ C_p*((lam_p - i*u)^alpha - lam_p^alpha)
                                      + C_m*((lam_m + i*u)^alpha - lam_m^alpha) ]

with parameter vector ``(alpha, C_p, C_m, lam_p, lam_m, mu)``.

Note on ``alpha``: ``Gamma(-alpha)`` has poles at alpha = 0, 1, 2, so the
admissible range is (0, 2) excluding 1. Optimisation is confined to a single
branch, by default (1.05, 1.95), which is the branch the source paper works in
(it reports alpha near 1.4). Pass ``alpha_bounds=(0.05, 0.95)`` to fit on the
lower branch instead.

Author
------
Sagnik Barman, 2026. Added on top of the RFIF implementation by Devharish N,
which implements Kumar, Upadhye & Chand (EPJ Special Topics, 2025). The
underlying RFIF and CTS formulations are theirs; the estimation code in this
file is new.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
from scipy.special import gamma as _gamma
from scipy.optimize import minimize
from scipy import interpolate

from cts_char import cts_cf, cts_char_exponent
from COS_inversion_helper import cos_invert_cf_from_cffunc

__all__ = [
    "CTSFitResult",
    "cts_cumulants",
    "sample_cumulants",
    "moment_match_init",
    "fit_cts_ecf",
    "fit_cts_mle",
    "fit_cts",
    "ks_goodness_of_fit",
    "cts_rvs",
]

# Parameter order used throughout: (alpha, C_p, C_m, lam_p, lam_m, mu)
PARAM_NAMES = ("alpha", "C_plus", "C_minus", "lam_plus", "lam_minus", "mu")


@dataclass
class CTSFitResult:
    """Container for a fitted CTS parameter set and its diagnostics."""

    params: tuple
    method: str
    objective: float
    ks_stat: float = float("nan")
    ks_pvalue: float = float("nan")
    n_obs: int = 0
    converged: bool = True
    message: str = ""
    history: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return dict(zip(PARAM_NAMES, self.params))

    def __str__(self) -> str:  # pragma: no cover - presentation only
        d = self.as_dict()
        body = "\n".join(f"    {k:<10s} = {v: .6g}" for k, v in d.items())
        return (
            f"CTS fit [{self.method}]  n={self.n_obs}  "
            f"objective={self.objective:.6g}\n{body}\n"
            f"    KS stat    = {self.ks_stat:.4f}  (p = {self.ks_pvalue:.4f})"
        )


# ---------------------------------------------------------------------------
# Cumulants
# ---------------------------------------------------------------------------

def cts_cumulants(params) -> tuple:
    """Return the first four cumulants (c1, c2, c3, c4) of the CTS law.

    For n >= 2 the CTS cumulants have the closed form

        c_n = Gamma(n - alpha) * [ C_p*lam_p^(alpha-n) + (-1)^n * C_m*lam_m^(alpha-n) ]

    and the mean is

        c_1 = mu + Gamma(1 - alpha) * [ C_p*lam_p^(alpha-1) - C_m*lam_m^(alpha-1) ].

    These follow from differentiating ``cts_char_exponent`` at u = 0 and are
    verified numerically in ``self_test()``.
    """
    alpha, C_p, C_m, lam_p, lam_m, mu = params

    c1 = mu + _gamma(1.0 - alpha) * (
        C_p * lam_p ** (alpha - 1.0) - C_m * lam_m ** (alpha - 1.0)
    )
    c2 = _gamma(2.0 - alpha) * (
        C_p * lam_p ** (alpha - 2.0) + C_m * lam_m ** (alpha - 2.0)
    )
    c3 = _gamma(3.0 - alpha) * (
        C_p * lam_p ** (alpha - 3.0) - C_m * lam_m ** (alpha - 3.0)
    )
    c4 = _gamma(4.0 - alpha) * (
        C_p * lam_p ** (alpha - 4.0) + C_m * lam_m ** (alpha - 4.0)
    )
    return float(c1), float(c2), float(c3), float(c4)


def sample_cumulants(x) -> tuple:
    """First four sample cumulants of ``x`` (mean, variance, third, fourth)."""
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 4:
        raise ValueError("need at least 4 observations")
    m = x.mean()
    d = x - m
    m2 = np.mean(d ** 2)
    m3 = np.mean(d ** 3)
    m4 = np.mean(d ** 4)
    # central moments -> cumulants
    c1 = m
    c2 = m2 * n / (n - 1) if n > 1 else m2
    c3 = m3
    c4 = m4 - 3.0 * m2 ** 2
    return float(c1), float(c2), float(c3), float(c4)


# ---------------------------------------------------------------------------
# Method of moments initialiser
# ---------------------------------------------------------------------------

def moment_match_init(x, alpha: float = 1.4, symmetric_lambda: bool = True) -> tuple:
    """Closed-form starting values from the first four sample cumulants.

    ``alpha`` is held fixed (it is poorly identified by low-order moments alone)
    and a common tempering rate ``lam_p = lam_m = lam`` is assumed. Writing
    ``C = C_p + C_m``, the variance and fourth cumulant give

        c2 = Gamma(2-alpha) * C * lam^(alpha-2)
        c4 = Gamma(4-alpha) * C * lam^(alpha-4)

    whose ratio yields ``lam = sqrt( (c2/Gamma(2-alpha)) / (c4/Gamma(4-alpha)) )``
    -- note the exponent difference is exactly 2. ``C`` then follows from c2,
    the asymmetry ``C_p - C_m`` from c3, and ``mu`` from c1.
    """
    c1, c2, c3, c4 = sample_cumulants(x)

    if c2 <= 0:
        raise ValueError("sample variance must be positive")

    g2 = _gamma(2.0 - alpha)
    g3 = _gamma(3.0 - alpha)
    g4 = _gamma(4.0 - alpha)

    A = c2 / g2                      # = C * lam^(alpha-2)

    # The lambda solved from the c2/c4 ratio is only meaningful when the sample
    # is genuinely heavy-tailed. A non-positive fourth cumulant would send
    # lambda towards infinity, i.e. straight into the Gaussian limit of the
    # family, where the parameters stop being identified. In that case anchor
    # lambda at the natural scale 1/sd instead and take C from the variance.
    sd = float(np.sqrt(c2))
    if c4 > 0:
        B = c4 / g4                  # = C * lam^(alpha-4)
        lam = float(np.sqrt(A / B))  # A/B = lam^2
    else:
        lam = 1.0 / sd
    lam = float(np.clip(lam, 1e-3, 50.0 / sd))

    C = A * lam ** (2.0 - alpha)
    C = float(max(C, 1e-12))

    # Asymmetry from the third cumulant: c3 = Gamma(3-a) * (C_p - C_m) * lam^(a-3)
    D = c3 * lam ** (3.0 - alpha) / g3
    C_p = 0.5 * (C + D)
    C_m = 0.5 * (C - D)
    # Keep both jump intensities strictly positive.
    floor = 1e-4 * C
    C_p = float(max(C_p, floor))
    C_m = float(max(C_m, floor))

    mu = c1 - _gamma(1.0 - alpha) * (
        C_p * lam ** (alpha - 1.0) - C_m * lam ** (alpha - 1.0)
    )

    if not symmetric_lambda:
        # Allow the optimiser a nudge away from the symmetric start.
        return (alpha, C_p, C_m, lam * 1.01, lam * 0.99, float(mu))
    return (alpha, C_p, C_m, lam, lam, float(mu))


# ---------------------------------------------------------------------------
# Parameter transform (unconstrained <-> constrained)
# ---------------------------------------------------------------------------

def make_bounds(x, alpha_bounds=(1.05, 1.95)) -> dict:
    """Data-scaled box constraints for every parameter.

    Without an upper bound on the tempering rates, a sample that is not
    heavy-tailed drives the optimiser towards the Gaussian limit of the CTS law
    (lambda -> infinity, C -> infinity, with mu compensating). That limit is a
    perfectly good fit but the individual parameters diverge, the characteristic
    function suffers catastrophic cancellation, and the returned numbers are
    meaningless. Bounding lambda to a window around the natural scale ``1/sd``
    keeps the estimate in the region where the parameters mean something.
    """
    x = np.asarray(x, dtype=float)
    sd = float(np.std(x))
    mean = float(np.mean(x))
    if sd <= 0:
        raise ValueError("sample has zero dispersion")
    lam_ref = 1.0 / sd
    return {
        "alpha": (float(alpha_bounds[0]), float(alpha_bounds[1])),
        "C": (1e-10, 1e4),
        # Generous at the bottom (weak tempering is a legitimate fit for heavy
        # tails); the upper bound is the one that matters, since that is the
        # direction in which the family degenerates to a Gaussian.
        "lam": (0.01 * lam_ref, 50.0 * lam_ref),
        "mu": (mean - 30.0 * sd, mean + 30.0 * sd),
    }


def _squash(raw, lo, hi):
    raw = float(np.clip(raw, -30.0, 30.0))
    return lo + (hi - lo) / (1.0 + np.exp(-raw))


def _unsquash(val, lo, hi):
    frac = (val - lo) / (hi - lo)
    frac = float(np.clip(frac, 1e-9, 1 - 1e-9))
    return float(np.log(frac / (1.0 - frac)))


def _log_squash(raw, lo, hi):
    """Sigmoid in log space: keeps positive parameters inside [lo, hi]."""
    return float(np.exp(_squash(raw, np.log(lo), np.log(hi))))


def _log_unsquash(val, lo, hi):
    val = float(np.clip(val, lo * (1 + 1e-9), hi * (1 - 1e-9)))
    return _unsquash(np.log(val), np.log(lo), np.log(hi))


def _to_unconstrained(params, bounds):
    alpha, C_p, C_m, lam_p, lam_m, mu = params
    return np.array(
        [
            _unsquash(alpha, *bounds["alpha"]),
            _log_unsquash(C_p, *bounds["C"]),
            _log_unsquash(C_m, *bounds["C"]),
            _log_unsquash(lam_p, *bounds["lam"]),
            _log_unsquash(lam_m, *bounds["lam"]),
            _unsquash(mu, *bounds["mu"]),
        ],
        dtype=float,
    )


def _to_constrained(theta, bounds):
    return (
        _squash(theta[0], *bounds["alpha"]),
        _log_squash(theta[1], *bounds["C"]),
        _log_squash(theta[2], *bounds["C"]),
        _log_squash(theta[3], *bounds["lam"]),
        _log_squash(theta[4], *bounds["lam"]),
        _squash(theta[5], *bounds["mu"]),
    )


def _at_bound(params, bounds, tol=1e-3) -> tuple:
    """Parameters sitting on the edge of their box.

    Returns ``(all_hits, critical_hits)``. Only the *critical* ones indicate a
    degenerate fit:

    * ``lam_*`` at its **upper** bound -- the Gaussian limit of the family, where
      the parameters stop being identified;
    * ``C_*`` or ``mu`` at either bound -- the optimiser has walked out of the
      region where the parameters mean anything.

    ``lam_*`` at its lower bound (weak tempering) and ``alpha`` at a branch edge
    are reported for information but are legitimate places for a fit to land on
    genuinely heavy-tailed data.
    """
    alpha, C_p, C_m, lam_p, lam_m, mu = params
    all_hits, critical = [], []
    for name, val, (lo, hi), logscale, crit_lo, crit_hi in (
        ("alpha", alpha, bounds["alpha"], False, False, False),
        ("C_plus", C_p, bounds["C"], True, True, True),
        ("C_minus", C_m, bounds["C"], True, True, True),
        ("lam_plus", lam_p, bounds["lam"], True, False, True),
        ("lam_minus", lam_m, bounds["lam"], True, False, True),
        ("mu", mu, bounds["mu"], False, True, True),
    ):
        if logscale:
            span = np.log(hi) - np.log(lo)
            frac = (np.log(val) - np.log(lo)) / span
        else:
            frac = (val - lo) / (hi - lo)
        if frac < tol:
            all_hits.append(f"{name}(lower)")
            if crit_lo:
                critical.append(name)
        elif frac > 1 - tol:
            all_hits.append(f"{name}(upper)")
            if crit_hi:
                critical.append(name)
    return all_hits, critical


# ---------------------------------------------------------------------------
# Empirical characteristic function estimator
# ---------------------------------------------------------------------------

def _empirical_cf(x, u):
    """phi_hat(u) = (1/n) sum_j exp(i*u*x_j), evaluated on the grid ``u``."""
    x = np.asarray(x, dtype=float)
    # Chunk the outer product so large samples do not blow up memory.
    out = np.zeros(u.shape, dtype=complex)
    chunk = max(1, int(4_000_000 // max(u.size, 1)))
    for start in range(0, x.size, chunk):
        xb = x[start:start + chunk]
        out += np.exp(1j * np.outer(u, xb)).sum(axis=1)
    return out / x.size


def _ecf_objective(theta, x_cf, u, w, bounds):
    params = _to_constrained(theta, bounds)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            try:
                model = cts_cf(u, params)
            except (FloatingPointError, ValueError, OverflowError):
                return 1e12
    if not np.all(np.isfinite(model)):
        return 1e12
    resid = np.abs(x_cf - model) ** 2
    return float(np.sum(w * resid))


def fit_cts_ecf(
    x,
    init=None,
    alpha_bounds=(1.05, 1.95),
    n_grid: int = 256,
    u_max: float | None = None,
    weight: str = "gaussian",
    maxiter: int = 800,
) -> CTSFitResult:
    """Fit CTS parameters by weighted least squares on the characteristic function.

    The objective is

        sum_k w(u_k) * | phi_hat(u_k) - phi(u_k; theta) |^2

    over a symmetric grid of frequencies. ``u_max`` defaults to ``10 / sd(x)``,
    which covers the range over which the empirical CF carries signal; beyond
    that it is dominated by sampling noise. The default Gaussian weight
    down-weights the high-frequency tail for the same reason.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 30:
        raise ValueError("need at least 30 finite observations to fit")

    sd = float(np.std(x))
    if sd <= 0:
        raise ValueError("sample has zero dispersion")

    if u_max is None:
        u_max = 10.0 / sd
    u = np.linspace(-u_max, u_max, n_grid)

    if weight == "gaussian":
        w = np.exp(-0.5 * (u / (0.5 * u_max)) ** 2)
    elif weight == "uniform":
        w = np.ones_like(u)
    else:
        raise ValueError("weight must be 'gaussian' or 'uniform'")
    w = w / w.sum()

    phi_hat = _empirical_cf(x, u)

    bounds = make_bounds(x, alpha_bounds)

    if init is None:
        alpha0 = 0.5 * (alpha_bounds[0] + alpha_bounds[1])
        init = moment_match_init(x, alpha=alpha0, symmetric_lambda=False)
    init = _clip_to_bounds(init, bounds)

    theta0 = _to_unconstrained(init, bounds)
    res = minimize(
        _ecf_objective,
        theta0,
        args=(phi_hat, u, w, bounds),
        method="Nelder-Mead",
        options={"maxiter": maxiter, "maxfev": 4 * maxiter,
                 "xatol": 1e-8, "fatol": 1e-14},
    )

    params = _to_constrained(res.x, bounds)
    hits, critical = _at_bound(params, bounds)
    # Note: `converged` reflects whether the *estimate is usable*, not whether
    # Nelder-Mead exhausted its iteration budget. Simplex methods routinely hit
    # the cap while sitting on a perfectly good optimum, so the optimizer's own
    # flag is recorded for information and the gate is degeneracy plus (in
    # fit_cts) the moment sanity checks.
    return CTSFitResult(
        params=params,
        method="ecf",
        objective=float(res.fun),
        n_obs=int(x.size),
        converged=not critical,
        message=(f"degenerate parameters at bound: {critical}" if critical
                 else str(res.message)),
        history={"init": tuple(init), "u_max": float(u_max),
                 "at_bound": hits, "critical_at_bound": critical,
                 "optimizer_success": bool(res.success),
                 "n_iter": int(res.nit)},
    )


def _clip_to_bounds(params, bounds):
    """Pull a parameter tuple strictly inside the admissible box."""
    alpha, C_p, C_m, lam_p, lam_m, mu = params

    def _inside(v, lo, hi, log=False):
        if log:
            v = float(np.clip(v, lo, hi))
            span = np.log(hi) - np.log(lo)
            v = float(np.exp(np.clip(np.log(max(v, lo)),
                                     np.log(lo) + 1e-6 * span,
                                     np.log(hi) - 1e-6 * span)))
            return v
        span = hi - lo
        return float(np.clip(v, lo + 1e-6 * span, hi - 1e-6 * span))

    return (
        _inside(alpha, *bounds["alpha"]),
        _inside(C_p, *bounds["C"], log=True),
        _inside(C_m, *bounds["C"], log=True),
        _inside(lam_p, *bounds["lam"], log=True),
        _inside(lam_m, *bounds["lam"], log=True),
        _inside(mu, *bounds["mu"]),
    )


# ---------------------------------------------------------------------------
# COS-based density, MLE and goodness of fit
# ---------------------------------------------------------------------------

def cts_density(params, Ncos: int = 4096, L: float = 10.0):
    """Reconstruct (x_grid, pdf, cdf, inv_cdf) for a CTS law via the COS method."""
    cf = lambda u: cts_cf(u, params)
    return cos_invert_cf_from_cffunc(cf, Ncos=Ncos, L=L)


def neg_loglik_params(params, x, Ncos: int = 2048, L: float = 10.0) -> float:
    """Negative log-likelihood of ``x`` under a CTS law with ``params``.

    Works on constrained (natural) parameters, so two candidate fits can be
    compared on the same scale regardless of how they were produced.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                x_grid, pdf, _, _ = cts_density(params, Ncos=Ncos, L=L)
    except (FloatingPointError, ValueError, OverflowError, np.linalg.LinAlgError):
        return 1e12
    if not np.all(np.isfinite(pdf)) or pdf.max() <= 0:
        return 1e12
    dx = x_grid[1] - x_grid[0]
    mass = float(np.sum(pdf) * dx)
    if not np.isfinite(mass) or mass <= 0:
        return 1e12
    pdf = pdf / mass
    dens = np.interp(x, x_grid, pdf, left=0.0, right=0.0)
    dens = np.maximum(dens, 1e-300)
    ll = float(np.sum(np.log(dens)))
    if not np.isfinite(ll):
        return 1e12
    return -ll


def _neg_loglik(theta, x, bounds, Ncos, L):
    params = _to_constrained(theta, bounds)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                x_grid, pdf, _, _ = cts_density(params, Ncos=Ncos, L=L)
    except (FloatingPointError, ValueError, OverflowError, np.linalg.LinAlgError):
        return 1e12
    if not np.all(np.isfinite(pdf)) or pdf.max() <= 0:
        return 1e12
    # Renormalise: COS truncation loses a little mass in the tails.
    dx = x_grid[1] - x_grid[0]
    mass = float(np.sum(pdf) * dx)
    if not np.isfinite(mass) or mass <= 0:
        return 1e12
    pdf = pdf / mass
    dens = np.interp(x, x_grid, pdf, left=0.0, right=0.0)
    dens = np.maximum(dens, 1e-300)
    ll = float(np.sum(np.log(dens)))
    if not np.isfinite(ll):
        return 1e12
    return -ll


def fit_cts_mle(
    x,
    init,
    alpha_bounds=(1.05, 1.95),
    Ncos: int = 2048,
    L: float = 10.0,
    maxiter: int = 300,
) -> CTSFitResult:
    """Refine an estimate by maximum likelihood, with the density from COS.

    This is materially slower than :func:`fit_cts_ecf` because every objective
    evaluation rebuilds a density, so it is meant as a refinement step starting
    from the ECF estimate rather than as a standalone fit.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    bounds = make_bounds(x, alpha_bounds)
    init = _clip_to_bounds(init, bounds)
    theta0 = _to_unconstrained(init, bounds)

    res = minimize(
        _neg_loglik,
        theta0,
        args=(x, bounds, Ncos, L),
        method="Nelder-Mead",
        options={"maxiter": maxiter, "maxfev": 4 * maxiter,
                 "xatol": 1e-7, "fatol": 1e-9},
    )
    params = _to_constrained(res.x, bounds)
    hits, critical = _at_bound(params, bounds)
    return CTSFitResult(
        params=params,
        method="mle",
        objective=float(res.fun),
        n_obs=int(x.size),
        converged=not critical,
        message=(f"degenerate parameters at bound: {critical}" if critical
                 else str(res.message)),
        history={"init": tuple(init), "neg_loglik": float(res.fun),
                 "at_bound": hits, "critical_at_bound": critical,
                 "optimizer_success": bool(res.success),
                 "n_iter": int(res.nit)},
    )


def ks_goodness_of_fit(x, params, Ncos: int = 4096, L: float = 10.0) -> tuple:
    """Kolmogorov-Smirnov statistic and asymptotic p-value against the fitted CTS.

    The p-value is the standard asymptotic one and is *optimistic* here, because
    the parameters were estimated from the same sample. Treat it as a relative
    diagnostic when comparing candidate fits, not as a formal test.
    """
    x = np.sort(np.asarray(x, dtype=float))
    n = x.size
    x_grid, pdf, cdf, _ = cts_density(params, Ncos=Ncos, L=L)
    cdf = cdf / cdf[-1] if cdf[-1] > 0 else cdf
    F = np.interp(x, x_grid, cdf, left=0.0, right=1.0)
    i = np.arange(1, n + 1)
    d_plus = np.max(i / n - F)
    d_minus = np.max(F - (i - 1) / n)
    d = float(max(d_plus, d_minus))

    # Asymptotic Kolmogorov distribution
    lam = (np.sqrt(n) + 0.12 + 0.11 / np.sqrt(n)) * d
    k = np.arange(1, 101)
    p = 2.0 * np.sum((-1.0) ** (k - 1) * np.exp(-2.0 * (k ** 2) * lam ** 2))
    p = float(np.clip(p, 0.0, 1.0))
    return d, p


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def fit_cts(
    x,
    alpha_bounds=(1.05, 1.95),
    refine: bool = True,
    verbose: bool = True,
    ecf_kwargs: dict | None = None,
    mle_kwargs: dict | None = None,
) -> CTSFitResult:
    """Estimate CTS parameters from a sample of returns.

    Runs method of moments -> empirical CF least squares -> (optionally) COS
    maximum likelihood, and attaches a KS statistic to the result.

    Parameters
    ----------
    x : array_like
        Sample of returns, e.g. ``(Close - Open) / Open``.
    alpha_bounds : (float, float)
        Branch of the stability parameter to search. Must not straddle 1 or 2.
    refine : bool
        Run the MLE refinement after the ECF fit.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    ecf_kwargs = dict(ecf_kwargs or {})
    mle_kwargs = dict(mle_kwargs or {})

    c1, c2, c3, c4 = sample_cumulants(x)
    sd = np.sqrt(c2)
    skew = c3 / sd ** 3
    exkurt = c4 / c2 ** 2
    if verbose:
        print(f"[cts_fit] n={x.size}  mean={c1:.6g}  sd={sd:.6g}  "
              f"skew={skew:.3f}  excess kurt={exkurt:.3f}")

    if exkurt <= 0:
        warnings.warn(
            f"sample excess kurtosis is {exkurt:.3f} (not heavy-tailed). CTS is a "
            "heavy-tail family, so the fit will drift towards its Gaussian limit "
            "and the individual parameters will be weakly identified. The box "
            "constraints keep the estimate finite, but treat the parameters as "
            "indicative and rely on the implied quantiles instead.",
            RuntimeWarning,
            stacklevel=2,
        )

    ecf = fit_cts_ecf(x, alpha_bounds=alpha_bounds, **ecf_kwargs)
    if verbose:
        print(f"[cts_fit] ECF objective={ecf.objective:.3e}  "
              f"alpha={ecf.params[0]:.4f}")

    best = ecf
    if refine:
        try:
            Ncos = mle_kwargs.get("Ncos", 2048)
            L = mle_kwargs.get("L", 10.0)
            # Score the ECF estimate on the likelihood scale so the two
            # candidates are compared like for like. Without this the MLE step
            # can be accepted even when it lands somewhere strictly worse.
            nll_ecf = neg_loglik_params(ecf.params, x, Ncos=Ncos, L=L)
            mle = fit_cts_mle(x, ecf.params, alpha_bounds=alpha_bounds, **mle_kwargs)
            if verbose:
                print(f"[cts_fit] neg-loglik  ECF={nll_ecf:.6g}  MLE={mle.objective:.6g}")
            if np.isfinite(mle.objective) and mle.objective < nll_ecf:
                best = mle
                if verbose:
                    print(f"[cts_fit] MLE improved the likelihood; "
                          f"alpha={mle.params[0]:.4f}")
            else:
                ecf.history["neg_loglik"] = float(nll_ecf)
                if verbose:
                    print("[cts_fit] MLE did not improve on the ECF fit; keeping ECF.")
        except Exception as exc:  # pragma: no cover - defensive
            if verbose:
                print(f"[cts_fit] MLE refinement failed ({exc}); keeping ECF fit.")

    try:
        d, p = ks_goodness_of_fit(x, best.params)
        best.ks_stat, best.ks_pvalue = d, p
    except Exception:  # pragma: no cover - defensive
        pass

    # Sanity checks: the fitted law should reproduce the sample location and
    # scale. A CTS fit can match the variance while putting the mean in a
    # completely wrong place (mu trades off against the Gamma(1-alpha) drift
    # term), which produces bands that miss the data entirely.
    problems = []
    try:
        fc = cts_cumulants(best.params)
        sd_ratio = np.sqrt(fc[1]) / sd
        mean_gap = abs(fc[0] - c1) / sd
        best.history["sd_ratio"] = float(sd_ratio)
        best.history["mean_gap_in_sd"] = float(mean_gap)
        if not (0.5 < sd_ratio < 2.0):
            problems.append(f"implied sd is {sd_ratio:.2f}x the sample sd")
        if mean_gap > 3.0:
            problems.append(f"implied mean is {mean_gap:.1f} sample sd away from the data mean")
    except Exception:  # pragma: no cover - defensive
        pass

    if problems:
        best.converged = False
        best.message += " | " + "; ".join(problems)
        warnings.warn(
            "CTS fit failed a sanity check (" + "; ".join(problems) + "). "
            "Downstream confidence bands built on it will be unreliable.",
            RuntimeWarning,
            stacklevel=2,
        )

    if verbose:
        print(best)
        if not best.converged:
            print(f"[cts_fit] WARNING: {best.message}")
    return best


def cts_rvs(params, size: int, random_state=None, Ncos: int = 8192, L: float = 12.0):
    """Draw samples from a CTS law by inverting the COS-reconstructed CDF.

    Used by :func:`self_test` to generate data with known parameters. Accuracy is
    limited by the COS grid resolution, which is fine for testing the estimator.
    """
    rng = np.random.default_rng(random_state)
    _, _, _, inv_cdf = cts_density(params, Ncos=Ncos, L=L)
    u = rng.uniform(1e-9, 1 - 1e-9, size=size)
    return np.asarray(inv_cdf(u), dtype=float)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def self_test(verbose: bool = True) -> bool:
    """Validate the cumulant formulas and the estimator on synthetic data."""
    ok = True

    # --- 1. cumulants against numerical derivatives of the CF -----------------
    params = (1.4, 2.0e-3, 1.5e-3, 9.5, 11.0, 1e-4)
    c1, c2, c3, c4 = cts_cumulants(params)

    h = 1e-3
    u = np.array([-2 * h, -h, 0.0, h, 2 * h])
    psi = cts_char_exponent(u, params)
    # central differences of the cumulant generating function psi(-i s) -> use
    # derivatives of psi wrt u and convert: c_n = (-i)^n * d^n psi / du^n
    d1 = (psi[3] - psi[1]) / (2 * h)
    d2 = (psi[3] - 2 * psi[2] + psi[1]) / h ** 2
    d3 = (psi[4] - 2 * psi[3] + 2 * psi[1] - psi[0]) / (2 * h ** 3)
    c1_num = float(np.real(-1j * d1))
    c2_num = float(np.real(-d2))
    c3_num = float(np.real(1j * d3))

    for name, a, b, tol in (
        ("c1", c1, c1_num, 1e-6),
        ("c2", c2, c2_num, 1e-8),
        ("c3", c3, c3_num, 1e-8),
    ):
        rel = abs(a - b) / max(abs(b), 1e-12)
        good = rel < 1e-3 or abs(a - b) < tol
        ok &= good
        if verbose:
            print(f"  cumulant {name}: closed form={a: .6e}  numerical={b: .6e}  "
                  f"{'OK' if good else 'MISMATCH'}")

    # --- 2. parameter recovery on synthetic data -----------------------------
    true_params = (1.45, 3.0e-3, 2.2e-3, 10.0, 12.0, 2e-4)
    x = cts_rvs(true_params, size=4000, random_state=20260922)
    fit = fit_cts(x, refine=False, verbose=False)

    tc = cts_cumulants(true_params)
    fc = cts_cumulants(fit.params)
    if verbose:
        print("\n  parameter recovery (n=4000, ECF only):")
        print(f"    {'':<10s} {'true':>12s} {'fitted':>12s}")
        for nm, tv, fv in zip(PARAM_NAMES, true_params, fit.params):
            print(f"    {nm:<10s} {tv:>12.5g} {fv:>12.5g}")
        print(f"    {'sd':<10s} {np.sqrt(tc[1]):>12.5g} {np.sqrt(fc[1]):>12.5g}")

    # The parameters are only weakly identified individually (C and lambda trade
    # off against each other), so judge recovery on the implied moments, which
    # are what the downstream bands actually depend on.
    sd_rel = abs(np.sqrt(fc[1]) - np.sqrt(tc[1])) / np.sqrt(tc[1])
    mean_abs = abs(fc[0] - tc[0])
    good = sd_rel < 0.15 and mean_abs < 0.01
    ok &= good
    if verbose:
        print(f"    implied sd rel. error = {sd_rel:.4f}  "
              f"mean abs. error = {mean_abs:.2e}  {'OK' if good else 'FAIL'}")

    if verbose:
        print(f"\n  self_test: {'PASSED' if ok else 'FAILED'}")
    return bool(ok)


if __name__ == "__main__":
    self_test()
