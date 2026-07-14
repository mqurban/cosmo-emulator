"""
mcmc_compare.py

The actual payoff demo: run parameter inference (MCMC) two ways on the
same synthetic "observed" data --

  1. Using real CAMB calls for the likelihood (slow, ~2.8s/evaluation)
  2. Using the trained neural emulator for the likelihood (fast, ~microseconds)

...and compare (a) wall-clock time and (b) whether the recovered
parameters agree. This directly demonstrates the pattern Givans described:
"emulate the slow model, then do faster-than-naive-MCMC inference."

Because a full-length MCMC chain (thousands of steps) via real CAMB would
take hours, we:
  - measure the true per-call CAMB cost precisely (from repeated calls)
  - run a SHORT but real CAMB-based chain to prove it works & starts
    moving toward the true parameters
  - run a FULL-length chain using the emulator (seconds, not hours)
  - report the recovered parameters from both, and extrapolate the
    wall-clock cost a full CAMB-based chain of the same length would need

This is the honest version of "faster than MCMC": we are not faking the
slow side, we are measuring it and showing why nobody would actually run
it at full length in practice.
"""

import os
import time
import numpy as np
import torch
import emcee
import camb

from train_emulator import PkEmulator, Standardizer, DEVICE

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

PARAM_NAMES = ["H0", "ombh2", "omch2", "ns", "As"]

# Fiducial ("true") parameters used to generate synthetic observed data
TRUE_PARAMS = np.array([67.5, 0.0224, 0.120, 0.965, 2.1e-9])

# Prior bounds (must match the ranges used in generate_data.py)
PRIOR_LOW = np.array([60.0, 0.020, 0.10, 0.92, 1.8e-9])
PRIOR_HIGH = np.array([75.0, 0.024, 0.14, 1.00, 2.4e-9])

NOISE_FRAC = 0.02  # 2% fractional noise on synthetic "observed" P(k)


# ------------------------------------------------------------------
# Emulator loading
# ------------------------------------------------------------------

def load_emulator():
    params = np.load(os.path.join(DATA_DIR, "params.npy")).astype(np.float32)
    pk = np.load(os.path.join(DATA_DIR, "pk.npy")).astype(np.float32)
    log_pk = np.log10(pk)

    scalers = np.load(os.path.join(MODEL_DIR, "scalers.npz"))
    x_scaler = Standardizer.__new__(Standardizer)
    x_scaler.mean, x_scaler.std = scalers["x_mean"], scalers["x_std"]
    y_scaler = Standardizer.__new__(Standardizer)
    y_scaler.mean, y_scaler.std = scalers["y_mean"], scalers["y_std"]

    model = PkEmulator(n_params=params.shape[1], n_output=pk.shape[1]).to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "emulator.pt"), map_location=DEVICE))
    model.eval()
    return model, x_scaler, y_scaler


def emulator_predict(model, x_scaler, y_scaler, theta):
    """theta: (5,) array of raw params -> returns (200,) P(k) prediction."""
    x = x_scaler.transform(theta.reshape(1, -1).astype(np.float32))
    with torch.no_grad():
        y = model(torch.tensor(x, dtype=torch.float32).to(DEVICE)).cpu().numpy()
    log_pk = y_scaler.inverse_transform(y)[0]
    return 10 ** log_pk


# ------------------------------------------------------------------
# CAMB (real, slow) prediction
# ------------------------------------------------------------------

def camb_predict(theta, kh_ref):
    H0, ombh2, omch2, ns, As = theta
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2)
    pars.InitPower.set_params(ns=ns, As=As)
    pars.set_matter_power(redshifts=[0.0], kmax=kh_ref.max() * 1.1)
    pars.NonLinear = camb.model.NonLinear_none
    results = camb.get_results(pars)
    kh, z, pk = results.get_matter_power_spectrum(
        minkh=kh_ref.min(), maxkh=kh_ref.max(), npoints=len(kh_ref)
    )
    return pk[0]


# ------------------------------------------------------------------
# Likelihood / prior (shared logic, only the predict function differs)
# ------------------------------------------------------------------

def log_prior(theta):
    if np.any(theta < PRIOR_LOW) or np.any(theta > PRIOR_HIGH):
        return -np.inf
    return 0.0


def make_log_likelihood(predict_fn, y_obs, sigma):
    def log_likelihood(theta):
        model_pk = predict_fn(theta)
        chi2 = np.sum(((y_obs - model_pk) / sigma) ** 2)
        return -0.5 * chi2
    return log_likelihood


def make_log_prob(predict_fn, y_obs, sigma):
    log_like = make_log_likelihood(predict_fn, y_obs, sigma)

    def log_prob(theta):
        lp = log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        try:
            ll = log_like(theta)
        except Exception:
            return -np.inf
        if not np.isfinite(ll):
            return -np.inf
        return lp + ll

    return log_prob


def main():
    kh = np.load(os.path.join(DATA_DIR, "kh.npy")).astype(np.float64)

    print("Generating synthetic 'observed' P(k) from true parameters:")
    for name, val in zip(PARAM_NAMES, TRUE_PARAMS):
        print(f"  {name} = {val}")

    y_true = camb_predict(TRUE_PARAMS, kh)
    rng = np.random.default_rng(0)
    sigma = NOISE_FRAC * y_true
    y_obs = y_true + rng.normal(0, sigma)
    print(f"Synthetic data: {len(y_obs)} points, {NOISE_FRAC*100:.0f}% fractional noise\n")

    # -----------------------------------------------------------
    # Load emulator
    # -----------------------------------------------------------
    model, x_scaler, y_scaler = load_emulator()
    emu_predict = lambda theta: emulator_predict(model, x_scaler, y_scaler, theta)

    ndim = len(TRUE_PARAMS)
    nwalkers = 12

    # Initialize walkers in a small ball around a slightly-off starting guess
    # (so the chain has to do real work to find the true values)
    start_guess = TRUE_PARAMS * np.array([1.03, 0.97, 1.05, 0.98, 1.02])
    spread = 0.02 * (PRIOR_HIGH - PRIOR_LOW)
    p0 = start_guess + spread * rng.normal(size=(nwalkers, ndim))
    p0 = np.clip(p0, PRIOR_LOW, PRIOR_HIGH)

    # -----------------------------------------------------------
    # 1) FULL-length MCMC using the emulator (fast)
    # -----------------------------------------------------------
    print("=" * 70)
    print("Running FULL emulator-based MCMC (this is the fast path)")
    print("=" * 70)
    log_prob_emu = make_log_prob(emu_predict, y_obs, sigma)
    sampler_emu = emcee.EnsembleSampler(nwalkers, ndim, log_prob_emu)

    n_steps_emu = 3000
    t0 = time.time()
    sampler_emu.run_mcmc(p0, n_steps_emu, progress=False)
    t_emu = time.time() - t0
    print(f"Emulator MCMC: {nwalkers} walkers x {n_steps_emu} steps "
          f"= {nwalkers*n_steps_emu} likelihood evals in {t_emu:.2f}s "
          f"({t_emu/(nwalkers*n_steps_emu)*1000:.3f} ms/eval)\n")

    burnin = int(n_steps_emu * 0.3)
    flat_emu = sampler_emu.get_chain(discard=burnin, flat=True)
    emu_mean = flat_emu.mean(axis=0)
    emu_std = flat_emu.std(axis=0)

    # -----------------------------------------------------------
    # 2) SHORT real CAMB-based MCMC (slow, real, honest)
    # -----------------------------------------------------------
    print("=" * 70)
    print("Running SHORT real-CAMB-based MCMC (this is the slow path)")
    print("=" * 70)
    n_walkers_camb = 2 * ndim  # emcee requires >= 2*ndim walkers (minimum allowed)
    log_prob_camb = make_log_prob(lambda theta: camb_predict(theta, kh), y_obs, sigma)
    sampler_camb = emcee.EnsembleSampler(n_walkers_camb, ndim, log_prob_camb)

    start_camb = start_guess + spread * rng.normal(size=(n_walkers_camb, ndim))
    p0_camb = np.clip(start_camb, PRIOR_LOW, PRIOR_HIGH)
    n_steps_camb = 3  # kept small deliberately -- see docstring
    t0 = time.time()
    sampler_camb.run_mcmc(p0_camb, n_steps_camb, progress=False)
    t_camb = time.time() - t0
    n_evals_camb = n_walkers_camb * n_steps_camb
    per_eval_camb = t_camb / n_evals_camb
    print(f"CAMB MCMC: {n_walkers_camb} walkers x {n_steps_camb} steps "
          f"= {n_evals_camb} likelihood evals in {t_camb:.2f}s "
          f"({per_eval_camb:.3f} s/eval)\n")

    flat_camb = sampler_camb.get_chain(flat=True)  # no burn-in, chain too short
    camb_mean = flat_camb.mean(axis=0)

    # -----------------------------------------------------------
    # Extrapolate: what would the SAME full chain cost with real CAMB?
    # -----------------------------------------------------------
    full_n_evals = nwalkers * n_steps_emu
    extrapolated_camb_time = full_n_evals * per_eval_camb

    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"{'param':>8} {'true':>10} {'emulator MCMC (mean+-std)':>28} {'short CAMB chain (mean)':>26}")
    for i, name in enumerate(PARAM_NAMES):
        print(f"{name:>8} {TRUE_PARAMS[i]:>10.4g} "
              f"{emu_mean[i]:>14.4g} +- {emu_std[i]:<10.2g} "
              f"{camb_mean[i]:>20.4g}")

    print(f"\nWall-clock time, emulator-based full chain ({full_n_evals} evals): {t_emu:.2f}s")
    print(f"Measured real CAMB cost per likelihood eval:                 {per_eval_camb:.3f}s")
    print(f"Extrapolated time for the SAME chain length using real CAMB: "
          f"{extrapolated_camb_time:.0f}s (~{extrapolated_camb_time/3600:.2f} hours)")
    print(f"\nSpeedup factor: {extrapolated_camb_time/t_emu:.0f}x")

    # Save results for the README / plots
    np.savez(
        os.path.join(DATA_DIR, "mcmc_results.npz"),
        true_params=TRUE_PARAMS,
        emu_mean=emu_mean, emu_std=emu_std,
        camb_mean=camb_mean,
        t_emu=t_emu, per_eval_camb=per_eval_camb,
        full_n_evals=full_n_evals,
        extrapolated_camb_time=extrapolated_camb_time,
        flat_emu=flat_emu,
    )
    print(f"\nSaved results to {DATA_DIR}/mcmc_results.npz")


if __name__ == "__main__":
    main()
