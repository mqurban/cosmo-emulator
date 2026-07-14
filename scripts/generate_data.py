"""
generate_data.py

Generates a training dataset for the cosmological emulator.

The idea: CAMB (a Boltzmann code) computes the matter power spectrum P(k)
given a set of cosmological parameters. This calculation is slow (a few
seconds per call), which is exactly the kind of bottleneck an emulator is
meant to remove. This script samples parameter combinations using a Latin
Hypercube (which spreads samples more evenly across the parameter space
than uniform random sampling) and calls CAMB for each one, storing the
input parameters and the resulting P(k) curve.

Parameters varied (standard CAMB / Planck-style base parameters):
    H0      - Hubble constant today [km/s/Mpc]
    ombh2   - physical baryon density (Omega_b * h^2)
    omch2   - physical cold dark matter density (Omega_c * h^2)
    ns      - scalar spectral index
    As      - amplitude of primordial curvature perturbations

Output:
    data/params.npy   -> shape (N, 5)   the sampled parameter combinations
    data/pk.npy       -> shape (N, npoints)  the resulting P(k) curves
    data/kh.npy       -> shape (npoints,)    the k values (same grid for all samples)
"""

import os
import time
import numpy as np
from scipy.stats import qmc
import camb

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

N_SAMPLES = 300          # number of CAMB evaluations to run
NPOINTS = 200            # number of k points in each P(k) curve
MINKH, MAXKH = 1e-4, 1.0 # k range [h/Mpc]
REDSHIFT = 0.0

# Parameter ranges (roughly centered on Planck 2018 best-fit values,
# with a generous range so the emulator learns real sensitivity)
PARAM_NAMES = ["H0", "ombh2", "omch2", "ns", "As"]
PARAM_BOUNDS = {
    "H0":    (60.0, 75.0),
    "ombh2": (0.020, 0.024),
    "omch2": (0.10, 0.14),
    "ns":    (0.92, 1.00),
    "As":    (1.8e-9, 2.4e-9),
}

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def sample_parameters(n_samples: int, seed: int = 42) -> np.ndarray:
    """Latin Hypercube sample the 5D parameter space, scaled to PARAM_BOUNDS."""
    sampler = qmc.LatinHypercube(d=len(PARAM_NAMES), seed=seed)
    unit_samples = sampler.random(n=n_samples)  # values in [0, 1)

    lower = np.array([PARAM_BOUNDS[p][0] for p in PARAM_NAMES])
    upper = np.array([PARAM_BOUNDS[p][1] for p in PARAM_NAMES])
    scaled = qmc.scale(unit_samples, lower, upper)
    return scaled


def run_camb(H0, ombh2, omch2, ns, As):
    """Run a single CAMB calculation and return the P(k) curve."""
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2)
    pars.InitPower.set_params(ns=ns, As=As)
    pars.set_matter_power(redshifts=[REDSHIFT], kmax=MAXKH * 1.1)
    pars.NonLinear = camb.model.NonLinear_none

    results = camb.get_results(pars)
    kh, z, pk = results.get_matter_power_spectrum(
        minkh=MINKH, maxkh=MAXKH, npoints=NPOINTS
    )
    sigma8 = results.get_sigma8()[0]
    return kh, pk[0], sigma8


def main(batch_size=None):
    """
    Generates data with checkpointing so the process can be safely resumed
    across multiple invocations (each CAMB call is ~3s, so large N can
    exceed a single command's time budget in some environments).

    If batch_size is given, only that many *new* samples are computed in
    this call, then progress is saved and the function returns. Re-running
    the script will pick up where it left off. If batch_size is None, all
    remaining samples (up to N_SAMPLES) are computed in one go.
    """
    os.makedirs(OUT_DIR, exist_ok=True)

    params_path = os.path.join(OUT_DIR, "params.npy")
    pk_path = os.path.join(OUT_DIR, "pk.npy")
    sigma8_path = os.path.join(OUT_DIR, "sigma8.npy")
    kh_path = os.path.join(OUT_DIR, "kh.npy")

    full_params = sample_parameters(N_SAMPLES)  # deterministic (fixed seed)

    # Resume from checkpoint if present
    if os.path.exists(pk_path):
        pk_curves = list(np.load(pk_path))
        sigma8_values = list(np.load(sigma8_path))
        kh_grid = np.load(kh_path)
        n_done = len(pk_curves)
        print(f"Resuming: {n_done}/{N_SAMPLES} samples already done.")
    else:
        pk_curves = []
        sigma8_values = []
        kh_grid = None
        n_done = 0

    if n_done >= N_SAMPLES:
        print("All samples already generated.")
        return True

    n_target = N_SAMPLES if batch_size is None else min(N_SAMPLES, n_done + batch_size)

    t_start = time.time()
    i = n_done
    while i < n_target:
        row = full_params[i]
        H0, ombh2, omch2, ns, As = row
        try:
            kh, pk, sigma8 = run_camb(H0, ombh2, omch2, ns, As)
            if kh_grid is None:
                kh_grid = kh
            pk_curves.append(pk)
            sigma8_values.append(sigma8)
        except Exception as e:
            print(f"[{i}] CAMB failed for params {row}: {e}")
            full_params = np.delete(full_params, i, axis=0)
            continue  # don't increment i, next row shifted into position i

        i += 1
        if i % 10 == 0 or i == n_target:
            elapsed = time.time() - t_start
            done_this_call = i - n_done
            rate = elapsed / max(done_this_call, 1)
            print(f"[{i}/{N_SAMPLES}] this-call elapsed={elapsed:.1f}s avg={rate:.2f}s/call")

    # Save checkpoint
    np.save(params_path, full_params[:i])
    np.save(pk_path, np.array(pk_curves))
    np.save(sigma8_path, np.array(sigma8_values))
    np.save(kh_path, kh_grid)
    with open(os.path.join(OUT_DIR, "param_names.txt"), "w") as f:
        f.write(",".join(PARAM_NAMES))

    print(f"\nCheckpoint saved: {i}/{N_SAMPLES} samples done.")
    if i < N_SAMPLES:
        print("Run the script again to continue generating remaining samples.")
        return False
    else:
        print("Dataset complete!")
        return True


if __name__ == "__main__":
    import sys
    batch = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    main(batch_size=batch)
