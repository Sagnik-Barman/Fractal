import time
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")
from utils import fetch_weekly_nifty, extract_extrema
from RecFIF import build_recurrent_fif_with_dvec
from coordinate_tuner import simple_tuner
from cts_char import cts_char_exponent
from wCF import weighted_cf_from_omegas
from COS_inversion_helper import cos_invert_cf_from_cffunc
from cts_fit import fit_cts
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

# Directory containing this file, so cached artefacts resolve on any machine.
HERE = Path(__file__).resolve().parent
DVEC_PATH = HERE / "d_vec_opt.npy"


def load_d_vec(path=None):
    """Load the pre-tuned vertical scaling factors, or return None if absent."""
    p = Path(path) if path is not None else DVEC_PATH
    if p.exists():
        return np.load(p)
    return None


def main_propagation(full_run=True, sample_every=1, Ncos_per_week=1024, basis_iters=250,
                     basis_grid=1400, m=2, d_vec=None, params_noise=None,
                     fit_noise=True, noise_fit_fraction=1.0, show_plot=True,
                     save_plot=None, noise_regimes=0, regime_window=26):
    """
    full_run: if False, will run a sampled quick run (sample_every > 1)
    sample_every: process every k-th week (useful to check quickly)
    Ncos_per_week: COS terms per week (increase to 2048/4096 for final)
    basis_iters / basis_grid: resolution to precompute basis functions

    d_vec: pre-tuned vertical scaling factors. If None, d_vec_opt.npy is loaded
        from the repository directory; if that is missing too, the coordinate
        tuner is run as a fallback.
    params_noise: CTS parameter tuple (alpha, C_p, C_m, lam_p, lam_m, mu). If
        None and fit_noise is True, the parameters are estimated from the
        observed weekly returns by cts_fit.fit_cts.
    noise_fit_fraction: fraction of the sample (from the start) used to fit the
        CTS parameters. Values below 1.0 hold out the tail of the series so that
        band coverage can be reported out of sample.
    noise_regimes: 0 or 1 fits a single CTS law for the whole sample. 3 splits
        the training slice into trailing-volatility terciles and fits one law
        per regime, so each week draws its band from the regime it is in.
        Thresholds are estimated on the training slice only.
    regime_window: rolling window, in weeks, for the trailing volatility that
        defines the regimes.
    """
    print("Loading weekly NIFTY data...")
    df = fetch_weekly_nifty()
    open_prices = df['Open'].values
    idx = extract_extrema(open_prices, order=1)
    knots_t = df['t'].values[idx]; knots_y = open_prices[idx]
    K = len(knots_t)
    print(f"Knots selected: {K}")

    # Resolve the tuned vertical scaling factors: explicit argument, then the
    # cached file next to this module, then a fresh tuner run.
    if d_vec is None:
        d_vec = load_d_vec()
        if d_vec is not None:
            print(f"Loaded pre-tuned d_vec from {DVEC_PATH.name} ({len(d_vec)} entries).")
            print("  Re-tune with retune.py if the date range, extrema order or m")
            print("  has changed since this vector was fitted.")
    else:
        print(f"Using caller-supplied d_vec ({len(d_vec)} entries).")

    if d_vec is None:
        print("No d_vec_opt.npy found. Running fallback tuner (this will take a while)...")
        d_vec = simple_tuner(knots_t, knots_y, df, m=m, passes=1, candidate_factors=[0.8,0.9,1.0,1.1], fast_iters=120, fast_grid=800)
        print("Tuning completed (fallback).")

    d_vec = np.asarray(d_vec, dtype=float)
    if len(d_vec) != K - 1:
        raise ValueError(
            f"d_vec has {len(d_vec)} entries but {K} knots require {K - 1}. "
            "The cached d_vec_opt.npy was tuned for a different date range or "
            "extrema order; delete it to re-tune, or pass d_vec explicitly."
        )

    # Precompute basis recurrent-FIFs (one per knot)
    print(f"Precomputing {K} basis recurrent-FIFs (iters={basis_iters}, grid={basis_grid}) ...")
    t0_time = time.time()
    basis_fns = []
    for i in range(K):
        y_basis = np.zeros(K); y_basis[i] = 1.0
        _, _, f_basis_fn, _ = build_recurrent_fif_with_dvec(knots_t, y_basis, d_vec, m=m, iters=basis_iters, grid_len=basis_grid, verbose=False)
        basis_fns.append(f_basis_fn)
        if (i+1) % 50 == 0 or (i+1) == K:
            print(f"  basis {i+1}/{K} done")
    elapsed = time.time() - t0_time
    print(f"Basis precompute finished in {elapsed:.1f}s")

    # compute high-res interpolant for open prices using tuned d_vec (for band transformation)
    _, _, f_fn_full, _ = build_recurrent_fif_with_dvec(knots_t, knots_y, d_vec, m=m, iters=400, grid_len=2500, verbose=False)
    interp_open_vals = f_fn_full(df['t'].values)

    # CTS noise parameters. Previously this looked for a `params_hat` global
    # that was never defined, so every run silently used hardcoded constants.
    # They are now estimated from the observed weekly returns.
    returns = (df['Close'].values - df['Open'].values) / df['Open'].values
    n_fit = int(round(len(returns) * float(np.clip(noise_fit_fraction, 0.05, 1.0))))
    n_fit = max(n_fit, 30)
    fit_returns = returns[:n_fit]

    cts_result = None
    if params_noise is None:
        if fit_noise:
            print(f"Estimating CTS parameters from {n_fit} weekly returns "
                  f"({'in-sample' if n_fit == len(returns) else 'training slice'}) ...")
            cts_result = fit_cts(fit_returns, verbose=True)
            params_noise = cts_result.params
            if not cts_result.converged:
                raise RuntimeError(
                    "CTS estimation failed its sanity checks: "
                    f"{cts_result.message}\n"
                    "Bands built on this fit would be meaningless, so the run is "
                    "stopped here rather than reporting a misleading coverage "
                    "number. Options: pass params_noise=... explicitly, widen "
                    "alpha_bounds, check the return series for data problems, or "
                    "pass fit_noise=False (--legacy-noise) to reproduce the "
                    "original hardcoded-parameter behaviour."
                )
        else:
            # Explicit opt-out: the original hardcoded values, kept so old
            # results can still be reproduced for comparison.
            params_noise = (1.4, 1e-3, 1e-3, 9.5, 9.5, float(np.mean(returns)))
            print("fit_noise=False -- using the original hardcoded CTS parameters.")
    else:
        print("Using caller-supplied CTS parameters.")
    print("CTS params (alpha, C+, C-, lam+, lam-, mu):",
          tuple(float(v) for v in params_noise))

    # Optional: condition the noise on a trailing-volatility regime instead of
    # using one law for the whole sample. Thresholds come from the training
    # slice only and volatility is backward-looking, so labels never see ahead.
    week_params = None
    regime_info = None
    if noise_regimes and noise_regimes > 1 and fit_noise:
        from regime_fit import assign_regimes, fit_cts_by_regime, params_per_week
        print(f"\nConditioning the noise on {noise_regimes} trailing-volatility "
              f"regimes (window={regime_window} weeks) ...")
        train_labels, thresholds, _ = assign_regimes(
            fit_returns, window=regime_window, n_regimes=noise_regimes)
        regime_params = fit_cts_by_regime(
            fit_returns, train_labels, n_regimes=noise_regimes,
            pooled=params_noise, verbose=True)
        # Label every week with the TRAINING thresholds.
        all_labels, _, all_vol = assign_regimes(
            returns, window=regime_window, thresholds=thresholds,
            n_regimes=noise_regimes)
        week_params = params_per_week(all_labels, regime_params)
        regime_info = {"labels": all_labels, "thresholds": thresholds,
                       "vol": all_vol, "params": regime_params}
        print(f"\n[regime] week counts over the full sample: " +
              "  ".join(f"{nm}={int((all_labels == g).sum())}"
                        for g, nm in enumerate(("low", "mid", "high")[:noise_regimes])))

    # per-week propagation (optionally sampled)
    n = len(df)
    weeks_idx = np.arange(0, n, sample_every)
    q_low = np.zeros(n); q_high = np.zeros(n)
    t_start = time.time()
    for counter, i in enumerate(weeks_idx):
        tval = df['t'].values[i]
        # compute omegas from basis functions
        omegas = np.array([float(fn(tval)) for fn in basis_fns])
        # noise law for this week: regime-specific if conditioning is on
        p_week = week_params[i] if week_params is not None else params_noise
        # define CF function for this week
        cf_func = lambda u, _p=p_week: weighted_cf_from_omegas(u, omegas, _p)
        # COS inversion (moderate resolution)
        _, pdf_w, cdf_w, inv_cdf_w = cos_invert_cf_from_cffunc(cf_func, Ncos=Ncos_per_week, L=8)
        q_low[i] = float(inv_cdf_w(0.025)); q_high[i] = float(inv_cdf_w(0.975))
        if (counter+1) % 50 == 0:
            print(f"Propagated {counter+1}/{len(weeks_idx)} weeks (index {i})")
    total_time = time.time() - t_start
    print(f"Propagation finished in {total_time:.1f}s (sample_every={sample_every}, Ncos={Ncos_per_week})")

    # Fill in any weeks not computed (if sampled) via linear interpolation on quantiles
    computed_idx = weeks_idx
    # For simplicity, fill zeros for uncomputed weeks then interpolate
    if sample_every > 1:
        # simple linear interpolation over indices
        valid = computed_idx
        computed_set = set(int(v) for v in valid)   # O(1) membership, was O(n) per week
        qlow_vals = q_low[valid]; qhigh_vals = q_high[valid]
        qlow_interp = interpolate.interp1d(valid, qlow_vals, bounds_error=False, fill_value="extrapolate")
        qhigh_interp = interpolate.interp1d(valid, qhigh_vals, bounds_error=False, fill_value="extrapolate")
        for j in range(n):
            if j not in computed_set:
                q_low[j] = float(qlow_interp(j)); q_high[j] = float(qhigh_interp(j))

    # convert to close bands
    close_low = interp_open_vals * (1 + q_low)
    close_high = interp_open_vals * (1 + q_high)
    closes = df['Close'].values
    inside = (closes >= close_low) & (closes <= close_high)
    coverage = np.mean(inside) * 100
    print(f"Coverage of actual closes inside 95% bands: {coverage:.2f}%")

    # When the CTS parameters were fitted on a leading slice, report the held-out
    # coverage separately. The nominal target is 95%; a full-sample number is
    # partly in-sample because the interpolant was tuned on the same Opens.
    coverage_train = coverage_test = float('nan')
    if n_fit < len(returns):
        coverage_train = float(np.mean(inside[:n_fit]) * 100)
        coverage_test = float(np.mean(inside[n_fit:]) * 100)
        print(f"  in-sample  ({n_fit} weeks): {coverage_train:.2f}%")
        print(f"  out-of-sample ({len(returns) - n_fit} weeks): {coverage_test:.2f}%")

    # Plots
    plt.figure(figsize=(14,6))
    plt.plot(df['Date'], df['Open'].values, label='Open (actual)', alpha=0.6)
    plt.plot(df['Date'], interp_open_vals, label='Tuned recurrent FIF interpolant', lw=1.0)
    plt.fill_between(df['Date'], close_low, close_high, color='gray', alpha=0.25, label='95% per-week bands')
    plt.plot(df['Date'], df['Close'].values, label='Close (actual)', alpha=0.9)
    if n_fit < len(returns):
        plt.axvline(df['Date'].values[n_fit], color='crimson', ls='--', lw=1.0,
                    label='CTS fit / hold-out split')
    plt.legend(); plt.title(f'Per-week CTS propagation — Coverage: {coverage:.2f}%')
    if save_plot:
        # Create the parent directory if it does not exist, so --save-plot
        # results/bands.png works on a fresh clone without a manual mkdir.
        Path(save_plot).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_plot, dpi=150, bbox_inches='tight')
        print(f"Figure written to {save_plot}")
    if show_plot:
        plt.show()
    else:
        plt.close()

    return {
        'df': df,
        'knots_idx': idx,
        'd_vec': d_vec,
        'interp_open': interp_open_vals,
        'close_low': close_low,
        'close_high': close_high,
        'coverage': coverage,
        'coverage_train': coverage_train,
        'coverage_test': coverage_test,
        'params_noise': tuple(float(v) for v in params_noise),
        'cts_fit': cts_result,
        'n_fit': n_fit,
        'regime_info': regime_info,
        'week_params': week_params,
    }

# ----------------------------
# 8) Run main propagation
# ----------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RFIF + CTS weekly band propagation")
    parser.add_argument("--sample-every", type=int, default=1,
                        help="process every k-th week (4 is a fast smoke test)")
    parser.add_argument("--ncos", type=int, default=1024,
                        help="COS terms per week; raise to 2048/4096 for better tails")
    parser.add_argument("--basis-iters", type=int, default=250)
    parser.add_argument("--basis-grid", type=int, default=1400)
    parser.add_argument("--m", type=int, default=2, help="neighbourhood half-width")
    parser.add_argument("--noise-fit-fraction", type=float, default=1.0,
                        help="fraction of the series used to fit CTS; e.g. 0.7 "
                             "holds out the last 30%% for honest coverage")
    parser.add_argument("--legacy-noise", action="store_true",
                        help="use the original hardcoded CTS parameters instead of fitting")
    parser.add_argument("--noise-regimes", type=int, default=0,
                        help="fit one CTS law per trailing-volatility regime (try 3)")
    parser.add_argument("--regime-window", type=int, default=26,
                        help="rolling window in weeks for the volatility regimes")
    parser.add_argument("--save-plot", default=None, help="path to write the figure")
    parser.add_argument("--no-show", action="store_true", help="do not open a plot window")
    args = parser.parse_args()

    out = main_propagation(
        full_run=True,
        sample_every=args.sample_every,
        Ncos_per_week=args.ncos,
        basis_iters=args.basis_iters,
        basis_grid=args.basis_grid,
        m=args.m,
        fit_noise=not args.legacy_noise,
        noise_fit_fraction=args.noise_fit_fraction,
        show_plot=not args.no_show,
        save_plot=args.save_plot,
        noise_regimes=args.noise_regimes,
        regime_window=args.regime_window,
    )
    print("Done. Coverage:", out['coverage'])
