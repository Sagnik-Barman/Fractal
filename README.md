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

### 2. Correctness fixes

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
- **A command-line interface** so runs are reproducible without editing source,
  and `show_plot` / `save_plot` so the pipeline can run headless.

### 3. `test_cts_fit.py` — offline test suite (new)

Six tests that need no network access:

1. Closed-form cumulants checked against numerical derivatives of the characteristic function
2. Moment initialiser stays finite on a platykurtic sample
3. Fit reproduces sample scale and location on heavy-tailed data
4. MLE refinement is never accepted when it is worse than the ECF fit
5. An empty download raises a clear error
6. End-to-end propagation with a synthetic series

```bash
python test_cts_fit.py     # Passed: 6/6
```

### 4. Honest out-of-sample evaluation

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
python test_cts_fit.py                        # offline tests
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

Run it and fill this section in with your own numbers. The figures in earlier
versions of this README (coverage, knot counts, runtimes) were carried over
from upstream without re-verification, so they are deliberately not repeated
here.

What to report: overall coverage, in-sample vs held-out coverage at
`--noise-fit-fraction 0.7`, the fitted CTS parameters, and the KS statistic.

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
