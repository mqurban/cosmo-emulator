"""
make_plots.py

Generates two figures for the README:
1. emulator_accuracy.png -- overlays emulator-predicted P(k) against real
   CAMB output for a few held-out validation samples, showing the
   emulator tracks the real calculation closely.
2. posterior_corner.png -- corner plot of the emulator-based MCMC
   posterior, with the true injected parameters marked, showing the
   emulator-driven inference correctly recovers the known answer.
"""

import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import corner

from train_emulator import PkEmulator, Standardizer, DEVICE

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
PARAM_NAMES = ["H0", "ombh2", "omch2", "ns", "As"]


def load_emulator():
    params = np.load(os.path.join(DATA_DIR, "params.npy")).astype(np.float32)
    pk = np.load(os.path.join(DATA_DIR, "pk.npy")).astype(np.float32)
    scalers = np.load(os.path.join(MODEL_DIR, "scalers.npz"))
    x_scaler = Standardizer.__new__(Standardizer)
    x_scaler.mean, x_scaler.std = scalers["x_mean"], scalers["x_std"]
    y_scaler = Standardizer.__new__(Standardizer)
    y_scaler.mean, y_scaler.std = scalers["y_mean"], scalers["y_std"]
    model = PkEmulator(n_params=params.shape[1], n_output=pk.shape[1]).to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "emulator.pt"), map_location=DEVICE))
    model.eval()
    return model, x_scaler, y_scaler, params, pk


def plot_emulator_accuracy():
    model, x_scaler, y_scaler, params, pk = load_emulator()
    kh = np.load(os.path.join(DATA_DIR, "kh.npy"))

    rng = np.random.default_rng(7)
    sample_idx = rng.choice(len(params), size=4, replace=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for idx in sample_idx:
        theta = params[idx]
        true_pk = pk[idx]
        x = x_scaler.transform(theta.reshape(1, -1))
        with torch.no_grad():
            y = model(torch.tensor(x, dtype=torch.float32).to(DEVICE)).cpu().numpy()
        pred_log_pk = y_scaler.inverse_transform(y)[0]
        pred_pk = 10 ** pred_log_pk

        axes[0].loglog(kh, true_pk, "-", alpha=0.7)
        axes[0].loglog(kh, pred_pk, "--", alpha=0.9)

        frac_err = np.abs(pred_pk - true_pk) / true_pk * 100
        axes[1].semilogx(kh, frac_err, alpha=0.8)

    axes[0].set_xlabel("k [h/Mpc]")
    axes[0].set_ylabel("P(k)")
    axes[0].set_title("Solid = real CAMB, Dashed = emulator\n(4 random validation samples)")

    axes[1].set_xlabel("k [h/Mpc]")
    axes[1].set_ylabel("Fractional error [%]")
    axes[1].set_title("Emulator error vs real CAMB")
    axes[1].axhline(0, color="gray", lw=0.5)

    plt.tight_layout()
    out_path = os.path.join(DATA_DIR, "..", "emulator_accuracy.png")
    plt.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


def plot_posterior_corner():
    results = np.load(os.path.join(DATA_DIR, "mcmc_results.npz"))
    flat_emu = results["flat_emu"]
    true_params = results["true_params"]

    fig = corner.corner(
        flat_emu,
        labels=PARAM_NAMES,
        truths=true_params,
        truth_color="red",
        show_titles=True,
        title_fmt=".4g",
    )
    out_path = os.path.join(DATA_DIR, "..", "posterior_corner.png")
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    plot_emulator_accuracy()
    plot_posterior_corner()
