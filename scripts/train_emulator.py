"""
train_emulator.py

Trains a PyTorch neural network emulator that predicts the matter power
spectrum P(k) directly from cosmological parameters, without calling CAMB.

This is the core idea Givans described: replace a slow, exact calculation
(CAMB, ~3s/call) with a fast neural network approximation (~microseconds),
so it can be called thousands of times inside an MCMC loop without the
computation itself becoming the bottleneck.

Design choices, explained:
- Inputs (5 params: H0, ombh2, omch2, ns, As) are standardized (zero mean,
  unit variance) since they live on very different scales (As ~ 1e-9,
  H0 ~ 70) -- without this the network would struggle to learn.
- The target P(k) spans several orders of magnitude, so we train on
  log10(P(k)) rather than P(k) directly. This is standard practice for
  emulating power spectra and makes the loss landscape far better behaved.
- A simple feedforward network is enough here: this is a smooth,
  well-behaved regression problem (5 inputs -> 200 smoothly-varying
  outputs), not something that needs convolutional or attention structure.
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Standardizer:
    """Standardize data to zero mean / unit variance, and invert later."""

    def __init__(self, data: np.ndarray):
        self.mean = data.mean(axis=0)
        self.std = data.std(axis=0)
        self.std[self.std == 0] = 1.0

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return data * self.std + self.mean


class PkEmulator(nn.Module):
    """Feedforward network: 5 cosmological params -> 200-point log10 P(k)."""

    def __init__(self, n_params=5, n_output=200, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_params, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, n_output),
        )

    def forward(self, x):
        return self.net(x)


def load_dataset():
    params = np.load(os.path.join(DATA_DIR, "params.npy")).astype(np.float32)
    pk = np.load(os.path.join(DATA_DIR, "pk.npy")).astype(np.float32)
    kh = np.load(os.path.join(DATA_DIR, "kh.npy")).astype(np.float32)
    return params, pk, kh


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    params, pk, kh = load_dataset()
    log_pk = np.log10(pk)

    n_total = params.shape[0]
    n_train = int(0.8 * n_total)
    rng = np.random.default_rng(42)
    idx = rng.permutation(n_total)
    train_idx, val_idx = idx[:n_train], idx[n_train:]

    x_scaler = Standardizer(params[train_idx])
    y_scaler = Standardizer(log_pk[train_idx])

    x_train = x_scaler.transform(params[train_idx])
    y_train = y_scaler.transform(log_pk[train_idx])
    x_val = x_scaler.transform(params[val_idx])
    y_val = y_scaler.transform(log_pk[val_idx])

    train_ds = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)

    x_val_t = torch.tensor(x_val, dtype=torch.float32).to(DEVICE)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).to(DEVICE)

    model = PkEmulator(n_params=params.shape[1], n_output=pk.shape[1]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=100
    )
    loss_fn = nn.MSELoss()

    n_epochs = 2000
    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    patience_limit = 300

    print(f"Training on {DEVICE}, {len(train_idx)} train / {len(val_idx)} val samples")

    for epoch in range(n_epochs):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_pred = model(x_val_t)
            val_loss = loss_fn(val_pred, y_val_t).item()

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 100 == 0 or epoch == n_epochs - 1:
            print(
                f"epoch {epoch:4d}  train_loss={np.mean(train_losses):.5f}  "
                f"val_loss={val_loss:.5f}  best_val={best_val_loss:.5f}"
            )

        if patience_counter >= patience_limit:
            print(f"Early stopping at epoch {epoch} (no improvement for {patience_limit} epochs)")
            break

    model.load_state_dict(best_state)

    # Save model + scalers for later use (inference / MCMC comparison)
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, "emulator.pt"))
    np.savez(
        os.path.join(MODEL_DIR, "scalers.npz"),
        x_mean=x_scaler.mean, x_std=x_scaler.std,
        y_mean=y_scaler.mean, y_std=y_scaler.std,
    )
    print(f"\nSaved model to {MODEL_DIR}/emulator.pt")
    print(f"Best validation loss (standardized log10 P(k) MSE): {best_val_loss:.5f}")

    # Quick accuracy report in real units
    model.eval()
    with torch.no_grad():
        val_pred_log_pk = y_scaler.inverse_transform(
            model(x_val_t).cpu().numpy()
        )
    val_true_log_pk = log_pk[val_idx]
    frac_err = np.abs(10**val_pred_log_pk - 10**val_true_log_pk) / (10**val_true_log_pk)
    print(f"Mean fractional error on validation P(k): {frac_err.mean()*100:.3f}%")
    print(f"Max fractional error on validation P(k):  {frac_err.max()*100:.3f}%")


if __name__ == "__main__":
    main()
