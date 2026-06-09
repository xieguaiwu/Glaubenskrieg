#!/usr/bin/env python3
"""Synthetic data experiments for Glaubenskrieg architecture validation.

Generates data with KNOWN signal structure to test whether the CTM architecture
can extract signals when they exist. This is the paper's critical "positive control"
experiment — if CTM fails on synthetic data too, the architecture is fundamentally
flawed; if CTM succeeds on synthetic but fails on real data, the data is the limit.

Experiments:
    S1 — AR(p) processes with known coefficients
    S2 — Sine wave + Gaussian noise at controlled SNR
    S3 — Factor model with known factor loadings
    S4 — Known IC level (generate returns with predetermined IC)

Usage:
    # Quick test all experiments
    PYTHONPATH=. python scripts/synthetic_data.py --quick-test

    # Full run with all SNRs and comparisons
    PYTHONPATH=. python scripts/synthetic_data.py --output results/synthetic.json

    # Single experiment
    PYTHONPATH=. python scripts/synthetic_data.py --experiment ar --snr 2.0
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model.multiasset_ctm import MultiAssetCTM

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("synthetic")

# ── Constants ─────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_N_ASSETS = 20     # assets per experiment
DEFAULT_N_TIMESTEPS = 2000  # time steps per asset
DEFAULT_SEQ_LEN = 63       # CTM sequence length
DEFAULT_N_EPOCHS = 50       # training epochs


# ═══════════════════════════════════════════════════════════════════
# Data Generators
# ═══════════════════════════════════════════════════════════════════

@dataclass
class SyntheticConfig:
    """Configuration for a single synthetic experiment."""
    name: str
    n_assets: int = DEFAULT_N_ASSETS
    n_timesteps: int = DEFAULT_N_TIMESTEPS
    n_features: int = 8       # matches real data input_dim - 1
    seq_len: int = DEFAULT_SEQ_LEN
    snr: float = 1.0           # signal-to-noise ratio (higher = cleaner)
    seed: int = 42


def _compute_snr_scale(signal: np.ndarray, noise_std: float, target_snr: float) -> float:
    """Scale signal to achieve target SNR.

    SNR = var(signal) / var(noise). Returns multiplier for signal.
    """
    sig_power = np.var(signal)
    if sig_power < 1e-12:
        return 1.0
    target_sig_power = target_snr * (noise_std ** 2)
    return float(np.sqrt(target_sig_power / sig_power))


# ── S1: AR(p) Process ────────────────────────────────────────────

def generate_ar_process(
    n_timesteps: int,
    n_assets: int,
    ar_coeffs: List[float],
    noise_std: float = 0.1,
    snr: float = 1.0,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate AR(p) process with known coefficients.

    Each asset has the same AR structure but independent noise.

    Args:
        n_timesteps: number of time steps
        n_assets: number of independent assets
        ar_coeffs: AR coefficients [φ₁, φ₂, ..., φₚ]
        noise_std: base noise std
        snr: target signal-to-noise ratio
        seed: random seed

    Returns:
        features: (n_timesteps, n_assets, n_features) — n_features=8 dummy features + target
        targets: (n_timesteps, n_assets) — the AR process values

    Note: For simplicity, features are the AR process lagged values.
    """
    rng = np.random.RandomState(seed)
    p = len(ar_coeffs)

    # Generate base AR process
    burn_in = 200
    total_len = n_timesteps + burn_in + p
    innovations = rng.randn(total_len, n_assets) * noise_std
    ar_series = np.zeros((total_len, n_assets))

    for t in range(p, total_len):
        for j, coeff in enumerate(ar_coeffs):
            ar_series[t] += coeff * ar_series[t - j - 1]
        ar_series[t] += innovations[t]

    ar_series = ar_series[burn_in + p:]  # remove burn-in

    # Scale to target SNR
    scale = _compute_snr_scale(ar_series, noise_std, snr)
    ar_series = ar_series * scale

    # Build feature set: p lags + (8-p) dummy noise features
    # This mimics real OHLCV data where only some features are predictive
    lag_features = np.zeros((n_timesteps, n_assets, min(p, 8)))
    for i in range(min(p, 8)):
        if i == 0:
            lag_features[:, :, i] = np.roll(ar_series, 1, axis=0)
        else:
            lag_features[:, :, i] = np.roll(ar_series, i + 1, axis=0)
    lag_features[:p] = 0  # zero-pad start

    # Add dummy noise features to fill to 8
    dummy = rng.randn(n_timesteps, n_assets, 8 - min(p, 8)) * noise_std
    features = np.concatenate([lag_features, dummy], axis=-1)

    # Targets: predict next AR value
    targets = np.roll(ar_series, -1, axis=0)
    targets[-1] = 0  # last step undefined

    return features, targets, ar_series


# ── S2: Sine + Noise ─────────────────────────────────────────────

def generate_sine_waves(
    n_timesteps: int,
    n_assets: int,
    periods: List[float],
    noise_std: float = 0.1,
    snr: float = 1.0,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate sine waves with controlled noise for periodic structure recovery.

    Args:
        periods: oscillation periods for each sine component
                 e.g., [10, 30] means 10- and 30-day cycles

    Returns:
        features: (n_timesteps, n_assets, 8)
        targets: (n_timesteps, n_assets)
    """
    rng = np.random.RandomState(seed)
    t = np.arange(n_timesteps).reshape(-1, 1)

    # Build signal from sine components
    signal = np.zeros((n_timesteps, n_assets))
    for period in periods:
        phase = rng.uniform(0, 2 * np.pi, size=n_assets)
        signal += np.sin(2 * np.pi * t / period + phase)

    signal /= len(periods)  # normalize

    # Add noise
    noise = rng.randn(n_timesteps, n_assets) * noise_std
    scale = _compute_snr_scale(signal, noise_std, snr)
    signal = signal * scale

    y = signal + noise

    # Features: 3 lagged values of y + 5 dummy noise
    features = np.zeros((n_timesteps, n_assets, 8))
    for i in range(3):
        features[:, :, i] = np.roll(y, i + 1, axis=0)
    features[:3] = 0
    for i in range(3, 8):
        features[:, :, i] = rng.randn(n_timesteps, n_assets) * noise_std

    # Target: predict next signal (clean, not noisy)
    target_signal = np.roll(signal, -1, axis=0)
    target_signal[-1] = 0

    return features, target_signal


# ── S3: Factor Model ──────────────────────────────────────────────

def generate_factor_model(
    n_timesteps: int,
    n_assets: int,
    n_factors: int = 3,
    noise_std: float = 0.1,
    snr: float = 1.0,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate data from a linear factor model.

    r_t = B @ f_t + ε_t

    where f_t ~ AR(1) factors, B is known loadings.

    Returns:
        features: (n_timesteps, n_assets, 8)
        targets: (n_timesteps, n_assets) — next-period returns r_{t+1}
        true_betas: (n_assets, n_factors) — for recovery comparison
    """
    rng = np.random.RandomState(seed)

    # Generate factors as AR(1)
    f = np.zeros((n_timesteps + 1, n_factors))
    for i in range(n_factors):
        phi = rng.uniform(0.3, 0.7)
        eps = rng.randn(n_timesteps + 1) * 0.1
        for t in range(1, n_timesteps + 1):
            f[t, i] = phi * f[t - 1, i] + eps[t]
    f = f[1:]

    # True loadings
    B = rng.randn(n_assets, n_factors) * 0.5

    # Returns: r_t = B @ f_t + ε_t
    returns = B @ f.T  # (n_assets, n_timesteps)
    returns = returns.T  # (n_timesteps, n_assets)

    # Scale to SNR
    noise = rng.randn(n_timesteps, n_assets) * noise_std
    scale = _compute_snr_scale(returns, noise_std, snr)
    returns = returns * scale + noise

    # Features: lagged returns + factor proxies + dummies
    features = np.zeros((n_timesteps, n_assets, 8))
    for i in range(3):
        features[:, :, i] = np.roll(returns, i + 1, axis=0)
    features[:3] = 0

    # Factor proxy features
    for i in range(n_factors):
        if 3 + i < 8:
            features[:, :, 3 + i] = np.broadcast_to(f[:, i], (n_assets, n_timesteps)).T
    # Fill remaining with noise
    for i in range(3 + n_factors, 8):
        features[:, :, i] = rng.randn(n_timesteps, n_assets) * noise_std

    # Target: predict next return
    target = np.roll(returns, -1, axis=0)
    target[-1] = 0

    return features, target, B


# ── S4: Known IC Level ────────────────────────────────────────────

def generate_known_ic(
    n_timesteps: int,
    n_assets: int,
    ic_level: float = 0.05,
    noise_std: float = 0.1,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate data where the optimal linear IC is known.

    Creates a single predictive feature z_t such that
    corr(z_t, r_{t+1}) = ic_level approximately.

    Returns:
        features: (n_timesteps, n_assets, 8)
        targets: (n_timesteps, n_assets) — returns with known IC
    """
    rng = np.random.RandomState(seed)

    # Generate predictive feature
    z = rng.randn(n_timesteps, n_assets)

    # Generate returns with known correlation
    # r = IC * z + sqrt(1 - IC²) * noise
    # This ensures corr(r, z) = IC approximately
    noise = rng.randn(n_timesteps, n_assets)
    returns = ic_level * z + np.sqrt(1 - ic_level ** 2) * noise
    returns = returns * noise_std  # scale to desired vol

    # Features: z (predictive) + 7 dummies
    features = np.zeros((n_timesteps, n_assets, 8))
    features[:, :, 0] = z
    for i in range(1, 8):
        features[:, :, i] = rng.randn(n_timesteps, n_assets) * noise_std

    # Target: next-period return
    target = np.roll(returns, -1, axis=0)
    target[-1] = 0

    return features, target


# ═══════════════════════════════════════════════════════════════════
# Sequence Builder
# ═══════════════════════════════════════════════════════════════════

def to_sequences(
    features: np.ndarray,
    targets: np.ndarray,
    seq_len: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert (T, N, D) + (T, N) → (B, N, T, D) + (B, T, N).

    Uses sliding window. B = T - seq_len + 1.
    """
    T, N, D = features.shape

    # sliding_window_view on (T, N, D) with axis=0
    # gives (B, N, D, seq_len) — needs axes 2,3 swapped to (B, N, seq_len, D)
    X = np.lib.stride_tricks.sliding_window_view(features, seq_len, axis=0)
    X = X.transpose(0, 1, 3, 2)  # (B, N, seq_len, D)

    # Targets: (T, N) → (B, seq_len, N)
    # sliding_window_view on 2D gives (B, N, seq_len) — needs axes 1,2 swapped
    Y = np.lib.stride_tricks.sliding_window_view(targets, seq_len, axis=0)
    Y = Y.transpose(0, 2, 1)  # (B, seq_len, N)

    return X.astype(np.float32), Y.astype(np.float32)


def train_val_test_split(
    X: np.ndarray, Y: np.ndarray,
    train_frac: float = 0.7, val_frac: float = 0.15,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sequential train/val/test split."""
    B = len(X)
    n_tr = int(B * train_frac)
    n_vl = int(B * val_frac)
    return (X[:n_tr], Y[:n_tr],
            X[n_tr:n_tr + n_vl], Y[n_tr:n_tr + n_vl],
            X[n_tr + n_vl:], Y[n_tr + n_vl:])


# ═══════════════════════════════════════════════════════════════════
# Model & Training
# ═══════════════════════════════════════════════════════════════════

def build_ctm(
    n_assets: int, input_dim: int,
    model_dim: int = 64, state_dim: int = 8, n_layers: int = 2,
) -> MultiAssetCTM:
    return MultiAssetCTM(
        n_assets=n_assets, input_dim=input_dim,
        model_dim=model_dim, state_dim=state_dim, n_layers=n_layers,
        output_dim=1, use_cross_attention=True,
        use_fused_attention=False, dropout=0.2,
        conv_kernel=3, use_decomp=False, bidirectional=False,
        parallel_scan=torch.cuda.is_available(),
        return_hidden=False, use_time_gate=False,
    )


def build_linear(n_assets: int, input_dim: int, seq_len: int) -> nn.Module:
    """Simple linear baseline: flattens sequence and regresses."""
    flat_dim = input_dim * seq_len
    class _LinearBaseline(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(flat_dim, 1)
        def forward(self, x):
            B, N, T, D = x.shape
            x_flat = x.reshape(B * N, T * D)
            pred = self.linear(x_flat)  # (B*N, 1)
            # Reshape to (B, N, 1) then expand to (B, N, T), then permute to (B, T, N)
            return pred.reshape(B, N, 1).expand(-1, -1, T).permute(0, 2, 1)
    return _LinearBaseline()


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_epochs: int = DEFAULT_N_EPOCHS,
    lr: float = 3e-4,
    weight_decay: float = 0.1,
    patience: int = 10,
    device: torch.device = DEVICE,
) -> Dict[str, Any]:
    """Train model and return metrics."""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    train_losses = []
    val_losses = []

    for epoch in range(1, n_epochs + 1):
        # Training
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred_raw = model(x)

            # Extract regression head (first N assets) for CTM
            if isinstance(model, MultiAssetCTM):
                pred = pred_raw[:, :, :model.n_assets]
            else:
                pred = pred_raw

            loss = nn.functional.mse_loss(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / max(n_batches, 1)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        n_vb = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred_raw = model(x)
                if isinstance(model, MultiAssetCTM):
                    pred = pred_raw[:, :, :model.n_assets]
                else:
                    pred = pred_raw
                val_loss += nn.functional.mse_loss(pred, y).item()
                n_vb += 1
        avg_val_loss = val_loss / max(n_vb, 1)
        val_losses.append(avg_val_loss)

        if avg_val_loss < best_val_loss - 1e-8:
            best_val_loss = avg_val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 10 == 0 or epoch == 1:
            logger.info("  epoch %3d/%d: train=%.6e val=%.6e (pat=%d)",
                        epoch, n_epochs, avg_train_loss, avg_val_loss, patience_counter)

        if patience_counter >= patience:
            logger.info("  early stopping at epoch %d", epoch)
            break

    # Restore best
    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "best_val_loss": float(best_val_loss),
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_loss_final": float(train_losses[-1]),
    }


def evaluate_ic(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device = DEVICE,
) -> Dict[str, float]:
    """Evaluate model: MSE and Spearman IC on last timestep."""
    model.to(device)
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            pred_raw = model(x)
            if isinstance(model, MultiAssetCTM):
                pred = pred_raw[:, :, :model.n_assets]
            else:
                pred = pred_raw
            all_preds.append(pred.cpu().numpy())
            all_targets.append(y.numpy())

    preds = np.concatenate(all_preds, axis=0)   # (B, T, N)
    targets = np.concatenate(all_targets, axis=0)

    # Last timestep
    p_last = preds[:, -1, :].reshape(-1)
    t_last = targets[:, -1, :].reshape(-1)

    mse = float(np.mean((p_last - t_last) ** 2))
    mae = float(np.mean(np.abs(p_last - t_last)))

    # Spearman IC
    from scipy.stats import spearmanr
    ic, ic_p = spearmanr(p_last, t_last)
    ic = float(ic) if not np.isnan(ic) else 0.0

    return {"mse": mse, "mae": mae, "ic": ic, "ic_pvalue": float(ic_p)}


# ═══════════════════════════════════════════════════════════════════
# Experiment Runner
# ═══════════════════════════════════════════════════════════════════

def run_experiment(
    name: str,
    generate_fn: callable,
    gen_kwargs: dict,
    model_dim: int = 64,
    state_dim: int = 8,
    n_layers: int = 2,
    n_assets: int = DEFAULT_N_ASSETS,
    seq_len: int = DEFAULT_SEQ_LEN,
    n_epochs: int = DEFAULT_N_EPOCHS,
    quick_test: bool = False,
) -> Dict[str, Any]:
    """Run a single synthetic experiment."""
    logger.info("=" * 50)
    logger.info("Experiment: %s", name)
    logger.info("Config: %s", gen_kwargs)

    # Generate data
    result = generate_fn(**gen_kwargs)
    if len(result) == 2:
        features, targets = result
        extra_info = {}
    elif len(result) == 3:
        features, targets, extra = result
        extra_info = {"true_betas_shape": list(extra.shape)} if isinstance(extra, np.ndarray) else {}
    else:
        raise ValueError(f"Unexpected return from {generate_fn.__name__}")

    N = features.shape[1]  # may differ from n_assets

    if quick_test:
        features = features[:500]
        targets = targets[:500]
        n_epochs = min(n_epochs, 10)

    # Build sequences
    X, Y = to_sequences(features, targets, seq_len)
    train_X, train_Y, val_X, val_Y, test_X, test_Y = train_val_test_split(X, Y)

    input_dim = X.shape[-1]
    logger.info("Data: X=%s Y=%s (B=%d, N=%d, D=%d)",
                X.shape, Y.shape, len(X), N, input_dim)

    # Build data loaders once, shared by both models
    batch_size = min(32, len(train_X))
    train_loader = DataLoader(TensorDataset(
        torch.from_numpy(train_X), torch.from_numpy(train_Y)),
        batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(TensorDataset(
        torch.from_numpy(val_X), torch.from_numpy(val_Y)),
        batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(
        torch.from_numpy(test_X), torch.from_numpy(test_Y)),
        batch_size=batch_size, shuffle=False)

    # ── Linear Baseline ──
    logger.info("Training linear baseline...")
    linear_model = build_linear(N, input_dim, seq_len)
    t0 = time.time()
    lin_train = train_model(linear_model, train_loader, val_loader, n_epochs=min(n_epochs, 20))
    lin_metrics = evaluate_ic(linear_model, test_loader)

    # ── CTM Baseline ──
    logger.info("Training CTM...")
    ctm_model = build_ctm(N, input_dim, model_dim, state_dim, n_layers)
    if quick_test:
        ctm_model = build_ctm(N, input_dim, model_dim=16, state_dim=2, n_layers=1)

    ctm_train = train_model(ctm_model, train_loader, val_loader, n_epochs=n_epochs)
    ctm_metrics = evaluate_ic(ctm_model, test_loader)
    ctm_params = sum(p.numel() for p in ctm_model.parameters())

    total_time = time.time() - t0

    result = {
        "experiment": name,
        "config": {**gen_kwargs, "model_dim": model_dim, "state_dim": state_dim,
                    "n_layers": n_layers, "n_epochs": n_epochs, "seq_len": seq_len},
        "data_shape": {"n_assets": N, "n_timesteps": features.shape[0],
                        "n_features": input_dim, "n_sequences": len(X)},
        "linear": {**lin_metrics, **lin_train},
        "ctm": {**ctm_metrics, **ctm_train, "params": ctm_params,
                "pt_ratio": ctm_params / max(len(train_X), 1)},
        "runtime_s": round(total_time, 1),
        **extra_info,
    }

    # Summary
    logger.info("%s results:", name)
    logger.info("  Linear: MSE=%.6e IC=%.4f", lin_metrics["mse"], lin_metrics["ic"])
    logger.info("  CTM:    MSE=%.6e IC=%.4f (params=%d, P/T=%.1f)",
                ctm_metrics["mse"], ctm_metrics["ic"], ctm_params,
                ctm_params / max(len(train_X), 1))

    return result


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Synthetic data experiments")
    parser.add_argument("--experiment", type=str, default=None,
                        choices=["ar", "sine", "factor", "ic", None],
                        help="Specific experiment to run (default: all)")
    parser.add_argument("--snr", type=float, default=1.0, help="Signal-to-noise ratio")
    parser.add_argument("--n-assets", type=int, default=DEFAULT_N_ASSETS)
    parser.add_argument("--n-timesteps", type=int, default=DEFAULT_N_TIMESTEPS)
    parser.add_argument("--model-dim", type=int, default=64)
    parser.add_argument("--state-dim", type=int, default=8)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--n-epochs", type=int, default=DEFAULT_N_EPOCHS)
    parser.add_argument("--quick-test", action="store_true",
                        help="Tiny data + few epochs for debugging")
    parser.add_argument("--output", type=str, default="results/synthetic_results.json")
    args = parser.parse_args()

    logger.info("Device: %s", DEVICE)
    if DEVICE.type == "cuda":
        logger.info("GPU: %s (%.1f GB)", torch.cuda.get_device_name(0),
                     torch.cuda.get_device_properties(0).total_memory / 1e9)

    experiments = []

    # S1: AR(p)
    if args.experiment in (None, "ar"):
        for p, coeffs in [(1, [0.9]), (3, [0.5, 0.2, 0.1]), (5, [0.4, 0.2, 0.1, 0.05, 0.025])]:
            experiments.append({
                "name": f"AR({p}) φ={coeffs}",
                "generate_fn": generate_ar_process,
                "gen_kwargs": {
                    "n_timesteps": args.n_timesteps, "n_assets": args.n_assets,
                    "ar_coeffs": coeffs, "noise_std": 0.1, "snr": args.snr, "seed": 42,
                },
            })

    # S2: Sine + noise
    if args.experiment in (None, "sine"):
        for periods in [[10], [30], [10, 30], [5, 15, 60]]:
            experiments.append({
                "name": f"Sine periods={periods}",
                "generate_fn": generate_sine_waves,
                "gen_kwargs": {
                    "n_timesteps": args.n_timesteps, "n_assets": args.n_assets,
                    "periods": periods, "noise_std": 0.1, "snr": args.snr, "seed": 42,
                },
            })

    # S3: Factor model
    if args.experiment in (None, "factor"):
        for n_factors in [1, 3, 5]:
            experiments.append({
                "name": f"Factor model K={n_factors}",
                "generate_fn": generate_factor_model,
                "gen_kwargs": {
                    "n_timesteps": args.n_timesteps, "n_assets": args.n_assets,
                    "n_factors": n_factors, "noise_std": 0.1, "snr": args.snr, "seed": 42,
                },
            })

    # S4: Known IC
    if args.experiment in (None, "ic"):
        for ic_level in [0.01, 0.03, 0.05, 0.10]:
            experiments.append({
                "name": f"Known IC={ic_level}",
                "generate_fn": generate_known_ic,
                "gen_kwargs": {
                    "n_timesteps": args.n_timesteps, "n_assets": args.n_assets,
                    "ic_level": ic_level, "noise_std": 0.1, "seed": 42,
                },
            })

    logger.info("Running %d experiments", len(experiments))

    all_results = []
    for exp in experiments:
        try:
            result = run_experiment(
                name=exp["name"],
                generate_fn=exp["generate_fn"],
                gen_kwargs=exp["gen_kwargs"],
                model_dim=args.model_dim,
                state_dim=args.state_dim,
                n_layers=args.n_layers,
                n_assets=args.n_assets,
                n_epochs=min(args.n_epochs, 10 if args.quick_test else args.n_epochs),
                quick_test=args.quick_test,
            )
            all_results.append(result)
        except Exception as e:
            logger.error("Experiment %s failed: %s", exp["name"], e)
            import traceback
            traceback.print_exc()

    output = {
        "device": str(DEVICE),
        "quick_test": args.quick_test,
        "experiments": all_results,
    }

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2, default=str)
        logger.info("Saved to %s", args.output)

    # Print comparison table
    print()
    print("=" * 80)
    print(f"{'Experiment':35s} {'Linear MSE':>12s} {'CTM MSE':>12s} {'Linear IC':>10s} {'CTM IC':>10s}")
    print("-" * 80)
    for r in all_results:
        lin_mse = r["linear"]["mse"]
        ctm_mse = r["ctm"]["mse"]
        lin_ic = r["linear"]["ic"]
        ctm_ic = r["ctm"]["ic"]
        print(f"{r['experiment']:35s} {lin_mse:12.6e} {ctm_mse:12.6e} {lin_ic:10.4f} {ctm_ic:10.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
