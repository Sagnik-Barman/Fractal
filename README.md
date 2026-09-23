# Recurrent Fractal Interpolation for Financial Data with Generalized Tempered Stable Noise

Python implementation of the recurrent fractal interpolation (RFIF) method of
Kumar, Upadhye & Chand, applied to weekly NIFTY 50 data, together with a
Fractal Complexity Coefficient (FCC) analysis and an FCC-guided regime-switching
pipeline.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Attribution

**The method is not mine.** This repository implements:

> Mohit Kumar, Neelesh S. Upadhye, A. K. B. Chand.
> *Recurrent fractal interpolation for data with generalized tempered stable noise:
> an application to NIFTY data.*
> **The European Physical Journal Special Topics**, 234(29): 9611–9629, 2025.
> [doi:10.1140/epjs/s11734-025-01639-3](https://doi.org/10.1140/epjs/s11734-025-01639-3)

**The original implementation is not mine either.** The RFIF core, the CTS
characteristic functions, the COS inversion helper and the propagation pipeline
were written by **Devharish N**, with the FCC modules contributed by
**V J Aswini**. Upstream repository:
[github.com/devharish1371/RFIF](https://github.com/devharish1371/RFIF).

This copy adds parameter estimation, correctness fixes and offline tests —
see [What this copy adds](#what-this-copy-adds). Everything else is upstream work.

---

## What this copy adds

Maintained by **Sagnik Barman** (M.Tech, Industrial Mathematics and Scientific
Computing, IIT Madras). Scope of the changes:

### 1. `cts_fit.py` — CTS parameters estimated from data (new)

The upstream pipeline looked for a variable named `params_hat` in the module
globals and, not finding it, fell back to a fixed vector
`(alpha, C+, C-, lam+, lam-, mu) = (1.4, 1e-3, 1e-3, 9.5, 9.5, mean)`. Because
`params_hat` was never defined anywhere in the repository, **every run used
those constants**. The confidence bands were therefore driven by hardcoded
numbers rather than by anything estimated from the NIFTY series.

`cts_fit.py` replaces that with three estimators:

| Function | Method | Notes |
|---|---|---|
| `moment_match_init` | Closed-form method of moments on the first four cumulants | Starting values; no optimisation |
| `fit_cts_ecf` | Weighted least squares between empirical and model characteristic functions | Fast, no density evaluation |
| `fit_cts_mle` | Maximum likelihood, density reconstructed via the existing COS machinery | Refinement step |

`fit_cts` chains them and attaches a Kolmogorov–Smirnov statistic.

Design decisions worth knowing about:

- **Box constraints scaled to the data.** Without an upper bound on the
  tempering rates, a sample that is not heavy-tailed drives the optimiser toward
  the Gaussian limit of the CTS family (`lambda -> inf`, `C -> inf`, with `mu`
  compensating). That limit fits fine but the individual parameters diverge and
  the characteristic function suffers catastrophic cancellation. `make_bounds`
  confines `lambda` to a window around the natural scale `1/sd`.
- **Like-for-like model selection.** The MLE refinement is accepted only when it
  beats the ECF estimate *on the likelihood scale*, computed via
  `neg_loglik_params`. Comparing an ECF objective against a negative
  log-likelihood is meaningless and silently keeps worse fits.
- **Sanity gates.** A fit whose implied standard deviation is off by more than
  2x, or whose implied mean sits more than 3 sample standard deviations from the
  data mean, is marked `converged=False`. `main.py` refuses to build bands on
  such a fit rather than reporting a misleading coverage number.
- **Weak identification is expected.** `C` and `lambda` trade off against one
  another, so individual parameters are not sharply pinned down even when the
  distribution is. Judge a fit by its implied moments and quantiles, which is
  what the bands actually depend on — not by the parameter values.

### 2. `RecFIF.py` — the interpolant now interpolates

A fractal interpolation function is *defined* by passing through its data. The
original construction did not.

| d_k | max knot error, % of data range |
|---|---|
| 0.00 | 0.09% (grid resolution) |
| 0.30 | 0.88% |
| 0.60 | 15.2% |
| 0.90 | 174.8% |
| 0.99 | 963.2% |

The error scaled with `d_k` and did **not** shrink with more iterations or a finer
grid — 100 iterations on an 800-point grid gave 11.27%; 3000 on 8000 points still
gave 9.87%. It was a property of the fixed point of the implemented operator, not a
convergence failure.

**Mechanism.** Writing `T` for the Read-Bajraktarevic operator,

```
(Tf)(t_{k-1}) = c_k*t_l + d_k*f(t_l) + e_k = y_{k-1} + d_k*( f(t_l) - y_l )
```

using the endpoint condition `c_k*t_l + d_k*y_l + e_k = y_{k-1}`. So `T` maps
interpolating functions to interpolating functions, it is a contraction with factor
`max|d_k| < 1`, and the interpolating set is closed — its unique fixed point
interpolates exactly. That is a theorem about the *exact* operator, and a numerical
implementation only inherits it if `f(t_l)` is represented exactly at the window
endpoints. The uniform evaluation grid did not contain the knots, so `f(t_l)`
carried an interpolation error, the residual `d_k*(f(t_l) - y_l)` was non-zero at
every iteration, and it was amplified rather than damped.

**Fix.** Put every knot on the evaluation grid, snap pre-images landing on a knot to
that knot exactly, and evaluate `T` in pull-back form so each grid point is written
by exactly one map — which also removes the deposit-and-average step and its
zero-masking bug (`deposited != 0.0` treated a legitimate zero as "no contribution").

**Result:** knot error at machine precision for every `d_k`, `d_k = 0` collapses to
exact linear interpolation, and the basis property `omega_i(t_j) = delta_ij` — which
the whole weighted-CF noise propagation rests on — goes from a maximum error of
**0.672** to **8.9e-16**. Roughness still grows with `d_k` (total variation 14.8 →
52.4 → 548.8 → 4151.7 for d = 0, 0.3, 0.6, 0.9), so the construction is still
fractal; it is now also an interpolation.

`build_recurrent_fif_legacy` keeps the original behaviour so earlier results remain
reproducible. `test_recfif.py` pins all of the above (7 tests).

> **`d_vec_opt.npy` is now stale.** It was tuned by minimising MAPE against the
> *broken* builder, so the tuner was partly suppressing the defect by driving `d_k`
> small — median 0.297, with 86.8% of values below 0.5, which puts the graph close
> to its piecewise-linear limit. Re-tune with `simple_tuner` before reading anything
> into the fitted roughness. All coverage results below predate the fix.

### 3. Correctness fixes

- **`main.py` could not run on any machine but the original author's.** It
  loaded the cached parameters from a hardcoded absolute path,
  `/Users/dev1371/Downloads/RFIF/d_vec_opt.npy`. Now resolved relative to the
  module, with a fallback to the coordinate tuner and a clear error when a
  cached `d_vec` does not match the current knot count.
- **`utils.fetch_weekly_nifty` failed opaquely on an empty download.** A failed
  or rate-limited fetch produced `KeyError: "['Date'] not in index"` several
  lines later. It now raises a `RuntimeError` naming the ticker and date range,
  and the resampled index is named explicitly so `reset_index()` is not at the
  mercy of what the data provider called the column.
- **`globals()` lookups replaced with real parameters.** `d_vec` and
  `params_noise` are now function arguments with a sensible resolution order.
- **O(n²) membership test** in the sampled-weeks fill replaced with a set lookup.
- **Dependency list corrected.** `seaborn` is imported by `fcc.py` but was never
  listed; `numba` and `tqdm` were listed but are imported nowhere. `numba` pins
  older numpy releases, so it could fail to install or silently downgrade numpy
  against the `numpy>=2.2` requirement alongside it. (Earlier documentation
  claimed Numba JIT compilation — there is no numba code in the repository.)
- **A command-line interface** so runs are reproducible without editing source,
  and `show_plot` / `save_plot` so the pipeline can run headless.

### 4. `test_cts_fit.py` and `test_recfif.py` — offline test suites (new)

Six tests that need no network access:

1. Closed-form cumulants checked against numerical derivatives of the characteristic function
2. Moment initialiser stays finite on a platykurtic sample
3. Fit reproduces sample scale and location on heavy-tailed data
4. MLE refinement is never accepted when it is worse than the ECF fit
5. An empty download raises a clear error
6. End-to-end propagation with a synthetic series

```bash
python test_cts_fit.py     # Passed: 6/6
python test_recfif.py      # Passed: 7/7
```

### 5. Honest out-of-sample evaluation

`noise_fit_fraction` fits the CTS parameters on a leading slice and reports
coverage separately on the held-out tail:

```bash
python main.py --noise-fit-fraction 0.7
```

This matters because the headline coverage number is otherwise partly
circular — the interpolant is tuned to minimise error against the same Open
series the bands are then scored on.

---

## Installation

```bash
python -m venv myenv
source myenv/bin/activate        # Windows: myenv\Scripts\activate
pip install -r requirements.txt
```

Requires Python 3.11+ and network access to Yahoo Finance for the price download.

## Usage

```bash
python test_recfif.py                         # interpolant correctness
python test_cts_fit.py                        # estimator + end-to-end
python main.py --sample-every 4 --ncos 512    # fast smoke run
python main.py                                # full run
python main.py --noise-fit-fraction 0.7       # with held-out coverage
python main.py --legacy-noise                 # original hardcoded CTS params
```

Fitting the noise directly:

```python
from utils import fetch_weekly_nifty
from cts_fit import fit_cts

df = fetch_weekly_nifty()
returns = (df['Close'].values - df['Open'].values) / df['Open'].values
fit = fit_cts(returns)
print(fit)                 # parameters, KS statistic, diagnostics
print(fit.as_dict())
```

## Method

Pipeline, in order:

1. **Data** — weekly Friday Open/Close bars for `^NSEI`, time rescaled to `[0, 1]`.
2. **Knots** — local extrema of the Open series plus endpoints.
3. **Recurrent FIF** — each interval gets an affine map from a *neighbourhood
   window* of width `m` (rather than the whole domain, as in a classical
   Barnsley FIF), with vertical scaling factor `d_k`. The attractor is found by
   iterating the maps on a grid to a fixed point; `|d_k| < 1` gives contraction.
4. **Tuning `d_vec`** — heuristic initialisation from local linear residuals,
   then greedy coordinate descent against MAPE.
5. **Basis functions** — one RFIF per knot, equal to 1 at that knot and 0
   elsewhere, so the interpolant is linear in knot values and noise can be
   pushed through as a weighted sum.
6. **Noise** — Classical Tempered Stable (CGMY) with characteristic exponent

   ```
   psi(u) = i*mu*u + Gamma(-alpha) * [ C+ ((lam+ - iu)^alpha - lam+^alpha)
                                     + C- ((lam- + iu)^alpha - lam-^alpha) ]
   ```

7. **Weighted CF** — infinite divisibility gives `exp(sum_i psi(omega_i u))`.
8. **COS inversion** — Fang & Oosterlee, to a density, CDF and inverse CDF.
9. **Bands** — 2.5% and 97.5% quantiles applied multiplicatively to the
   interpolated Open.

## Results

Weekly NIFTY 50, 2007-09-21 to 2022-01-21: 749 weeks, 380 extrema knots. CTS
parameters fitted on the first 524 weeks (70%), coverage then scored on the
remaining 225.

```
python evaluate_coverage.py --noise-fit-fraction 0.7 --no-show
```

### Fitted CTS parameters

| | |
|---|---|
| alpha | 1.48142 |
| C+ / C- | 0.00113541 / 0.000859029 |
| lambda+ / lambda- | 16.9112 / 8.56236 |
| mu | 0.00825319 |
| KS statistic | 0.0290 (p = 0.766) |

Two sanity checks pass. `alpha` lands at 1.481 against the 1.4 reported in the
source paper — an independent estimator agreeing with the published value.
And `lambda- < lambda+` by roughly a factor of two, i.e. the left tail is
tempered at half the rate of the right, so the fit has recovered the negative
skew of equity index returns from the data rather than being told about it.

Fitted mean and standard deviation reproduce the sample to within 0.002 sample
sd and 0.3% respectively. Skewness and excess kurtosis overshoot (-0.55 vs
-0.19, and 7.41 vs 4.05): the characteristic-function objective weights the
centre of the distribution, so the tails are extrapolated rather than fitted.

### Coverage

| | weeks | coverage |
|---|---|---|
| overall | 749 | **95.06%** |
| in-sample | 524 | 94.85% |
| held out | 225 | **95.56%** |

Against a nominal 95% that looks close to perfect, and the held-out number sits
0.08 corrected standard errors from target. **Do not read it that way.** The
aggregate is an average of two regimes that are each badly calibrated:

| | weeks | coverage |
|---|---|---|
| 2007-2009 and 2020 | 171 | **80.7%** |
| every other year | 578 | **99.3%** |

In the calm decade the bands are far too wide; in crises they are far too
narrow. The two errors cancel to 95.06%.

The formal tests agree. A Wald-Wolfowitz runs test gives 46 runs against 71.3
expected under independence (z = -9.93, p < 1e-5) — misses arrive in bunches,
not scattered. The miss indicator has lag-1 autocorrelation +0.352, which puts
the effective sample size at 359 of 749 and the true standard error at 1.15 pp
rather than the naive binomial 0.80 pp. The longest run of consecutive misses
is 5 weeks, starting 2009-03-20.

Direction is also one-sided: of 37 misses, **29 are above the band and 8 below**
(z = +3.45 against a symmetric split). That is a bias in where the band is
centred, not symmetric tail thinness.

### What this means

The model carries a single time-invariant noise law, and markets cluster their
volatility. No static distribution can serve both 2009 and 2014. A headline
coverage figure near nominal is therefore weak evidence on its own — the
regime split and the runs test are what tell you whether the bands mean
anything week to week.

This is direct motivation for the regime-switching direction already present
upstream in `fcc_gts_regime_pipeline.py`. Conditioning the CTS parameters on a
volatility or FCC regime, rather than fitting one set across fifteen years, is
the obvious next step.

`evaluate_coverage.py` also attributes each miss to either the interpolant being
mis-centred or a genuine tail event, by re-centring the same quantiles on the
actual Open. Run it to see the split for this configuration.

## Known limitations

- **Coverage is partly in-sample.** The interpolant is tuned on the same Open
  series used to score the bands. Use `--noise-fit-fraction` and quote the
  held-out number.
- **Knot noise is assumed independent.** The weighted-CF step depends on it, and
  it is unlikely to hold exactly for adjacent market extrema.
- **Parameters are weakly identified individually.** See above.
- **`alpha` is fitted on one branch at a time.** `Gamma(-alpha)` has poles at
  0, 1 and 2, so the search is confined to `(1.05, 1.95)` by default. Pass
  `alpha_bounds=(0.05, 0.95)` for the lower branch.
- **`d_vec_opt.npy` is tied to a specific date range.** Change the window and
  it must be re-tuned; `main.py` now says so instead of failing obscurely.
- **FCC modules are upstream work and have not been re-verified here.**

## Citation

Cite the paper, not this repository:

```bibtex
@article{kumar2025recurrent,
  title   = {Recurrent fractal interpolation for data with generalized tempered
             stable noise: an application to NIFTY data},
  author  = {Kumar, Mohit and Upadhye, Neelesh S. and Chand, A. K. B.},
  journal = {The European Physical Journal Special Topics},
  volume  = {234},
  number  = {29},
  pages   = {9611--9629},
  year    = {2025},
  doi     = {10.1140/epjs/s11734-025-01639-3}
}
```

## License

MIT — see [LICENSE](LICENSE). The licence and copyright notice are carried over
from upstream unchanged.
