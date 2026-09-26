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

**The implementation began as a course group project** by **Devharish N**,
**V J Aswini** and **Sagnik Barman**, in which I contributed to the original
codebase. Commits in the upstream history are under Devharish's account, which
does not reflect the division of work. The FCC modules are V J Aswini's.
Upstream repository:
[github.com/devharish1371/RFIF](https://github.com/devharish1371/RFIF).

This copy carries that work forward on my own: parameter estimation, correctness
fixes, regime-conditioned noise and a validation suite — see
[What this copy adds](#what-this-copy-adds) for exactly which files are which.

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

> **Was the tuned roughness an artefact of the defect?** It was reasonable to
> suspect so: `simple_tuner` minimises MAPE, and since large `d_k` broke the old
> builder, the tuner had an incentive to keep `d_k` small.
>
> **Measured on weekly NIFTY (380 knots): no.** Two re-tunes against the fixed
> builder, from different starting points:
>
> | | median d_k | sum abs d_k | box dim | MAPE (full res) |
> |---|---|---|---|---|
> | shipped vector | 0.2968 | 132.7 | 1.823 | 0.7981% |
> | cold start from heuristic | 0.3162 | 185.2 | 1.879 | 0.8532% |
> | warm start from shipped | **0.2091** | 119.8 | 1.806 | **0.5762%** |
>
> Roughness does not jump when the defect is removed — it drops slightly. The
> shipped values were a genuine fit to the data, not the tuner dodging a bug.
>
> **But the shipped vector was under-tuned.** A warm-started two-pass sweep cuts
> MAPE by 28%, with 320 then 289 of 379 coordinates still moving. `simple_tuner`
> is greedy coordinate descent, so a cold start from the heuristic lands in a worse
> local optimum than the incumbent already occupies — which is why `retune.py`
> warm-starts by default and refuses to write a vector that scores worse.
>
> Note also that "median `d_k` ~ 0.3" is **not** close to piecewise linear. Box
> dimension goes as `D = 1 + log(sum|d_k|)/log(N)`, and `sum|d_k| = 132.7` over
> `N = 379` maps gives `D ~ 1.82` — a genuinely rough graph. The dimension depends
> on the sum, not the median.

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

### 4. `regime_fit.py` — regime-conditioned noise (new)

The validation below shows a single time-invariant CTS law cannot calibrate across
market regimes: 100.0% coverage when volatility is low, 84.4% when it is high.
`regime_fit.py` fits one law per trailing-volatility tercile and lets each week
draw its band from the regime it is actually in.

```
python evaluate_coverage.py --noise-fit-fraction 0.7 --noise-regimes 3
```

Two things are enforced so the comparison means something:

- **Labels are causal.** Volatility is a trailing rolling standard deviation, so
  week `t` is labelled from returns up to and including `t`.
- **Thresholds come from the training slice only**, then are applied unchanged to
  the held-out period. Computing tercile cut points over the full sample would
  leak the future distribution of volatility into every label.

A regime with fewer than `min_obs` weeks, or whose fit fails the sanity gates in
`cts_fit`, falls back to the pooled fit rather than producing a degenerate one —
splitting 524 training weeks three ways leaves ~175 each, which is workable but
not generous.

`test_regime_fit.py` pins the causality and no-leak properties (5 tests).

### 5. `test_cts_fit.py`, `test_recfif.py`, `test_regime_fit.py` — offline test suites (new)

Six tests that need no network access:

1. Closed-form cumulants checked against numerical derivatives of the characteristic function
2. Moment initialiser stays finite on a platykurtic sample
3. Fit reproduces sample scale and location on heavy-tailed data
4. MLE refinement is never accepted when it is worse than the ECF fit
5. An empty download raises a clear error
6. End-to-end propagation with a synthetic series

```bash
python test_recfif.py      # Passed: 7/7
python test_cts_fit.py     # Passed: 6/6
python test_regime_fit.py  # Passed: 5/5
```

### 6. Honest out-of-sample evaluation

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
python test_regime_fit.py                     # regime causality / no-leak
python retune.py                              # re-tune d_vec (required once)
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

Weekly NIFTY 50, 2007-09-21 to 2022-01-21: 749 weeks, 380 extrema knots. `d_vec`
re-tuned against the corrected builder (MAPE 0.576%). CTS parameters fitted on the
first 524 weeks, coverage scored on all 749.

```
python retune.py
python evaluate_coverage.py --noise-fit-fraction 0.7 --save-plot results/bands.png
```

### Fitted CTS parameters

| | |
|---|---|
| alpha | 1.48142 |
| C+ / C- | 0.00113541 / 0.000859029 |
| lambda+ / lambda- | 16.9112 / 8.56236 |
| mu | 0.00825319 |
| KS statistic | 0.0290 (p = 0.766) |

Two independent checks pass. `alpha` lands at 1.481 against the 1.4 reported in the
source paper, from an estimator that never saw that value. And
`lambda- < lambda+` by roughly a factor of two — the left tail is tempered at half
the rate of the right, so the fit recovered the negative skew of equity index
returns from the data rather than being told about it.

Fitted mean and standard deviation reproduce the sample to within 0.002 sample sd
and 0.3%. Skewness and excess kurtosis overshoot (-0.55 against -0.19, and 7.41
against 4.05): the characteristic-function objective weights the centre of the
distribution, so the tails are extrapolated rather than fitted.

### Coverage: the aggregate is meaningless

| | weeks | coverage | z vs nominal |
|---|---|---|---|
| overall | 749 | 94.53% | -0.41 |
| in-sample | 524 | 93.70% | -0.93 |
| held out | 225 | 96.44% | +0.68 |

All within one standard error of the nominal 95%. **That tells you almost nothing.**
Split the same weeks by trailing realised volatility:

| regime | weeks | coverage | z vs nominal |
|---|---|---|---|
| low volatility | 250 | **100.0%** | **+2.49** |
| mid volatility | 249 | 99.2% | +2.08 |
| high volatility | 250 | **84.4%** | **-5.27** |

An honest 95% interval is 95% in *every* regime. This one is 100% when markets are
calm — the bands are far too wide, and not one week in 250 falls outside — and
84.4% when they are stressed. A 15.6 pp spread. The errors cancel to 94.53%.

Per calendar year the same thing: 2010 through 2019 run at 96-100%, while 2007
falls to 73.3%, 2008 and 2009 to 75.0%, and 2020 to 86.5%.

z-scores use dependence-corrected standard errors. The miss indicator has lag-1
autocorrelation +0.3605, giving an effective sample of 352 of 749 weeks, so the
true standard error is 1.16 pp rather than the naive binomial 0.80 pp.

### Misses cluster and lean one way

A Wald-Wolfowitz runs test gives **50 runs against 78.5 expected** under
independence, z = -10.13, p < 1e-5. Misses arrive in bunches. The longest run is
5 consecutive weeks from 2008-12-12.

Direction: of 41 misses, **29 fall above the band and 12 below** (z = +2.65 against
a symmetric split). Symmetric tail thinness would miss on both sides; a one-sided
pattern means the band centre is biased.

### What causes a miss

`evaluate_coverage.py` re-centres each band's quantiles on the *actual* Open. A miss
that disappears was an interpolation failure; one that survives is a genuine tail
event.

| cause | misses | share |
|---|---|---|
| interpolant mis-centred | 14 | 34% |
| genuine tail event | 27 | 66% |

Interpolation error is **4.4x higher in miss weeks than in hit weeks** (2.121%
against 0.487%, with 0.576% across all weeks), so it is concentrated exactly where
the bands fail. Re-centring every band on the true Open would lift coverage from
94.53% to 95.46%.

So roughly a third of the problem is the interpolant and two thirds is the noise
model being too narrow under stress.

### Effect of re-tuning

Re-tuning `d_vec` cut interpolant MAPE from 0.798% to 0.576% (-28%):

| | before | after |
|---|---|---|
| overall | 95.06% | 94.53% |
| held out | 95.56% | 96.44% |
| misses above / below | 29 / 8 | 29 / 12 |
| share above | 78% | 71% |
| runs-test z | -9.93 | -10.13 |

A better interpolant reduced the one-sidedness (z +3.45 to +2.65) without removing
it, which is consistent with the attribution above: the interpolant was part of the
story, not all of it. The clustering did not improve at all, because it never had
anything to do with the interpolant.

### What this means

The model carries a single time-invariant noise law, and markets cluster their
volatility. No static distribution serves both 2009 and 2014. The headline coverage
figure is an average over regimes in which the bands are separately far too wide and
far too narrow, and on its own it is close to uninformative — the regime split and
the runs test are what say whether the bands mean anything week to week.

This is direct motivation for conditioning the CTS parameters on a volatility or FCC
regime, which is the direction `fcc_gts_regime_pipeline.py` already points in.

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
