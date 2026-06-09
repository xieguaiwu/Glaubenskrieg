#!/usr/bin/env python3
"""Baseline comparison: CTM vs LSTM vs GRU vs Linear on synthetic data.

Tests whether Mamba SSM is justified at T=63 by comparing against simpler
recurrent architectures at matched parameter counts (weakness #1).

Usage:
    # Quick test
    PYTHONPATH=. python scripts/baseline_compare.py --quick-test --output results/baseline_quick.json

    # Full run
    PYTHONPATH=. python scripts/baseline_compare.py --output results/baseline_full.json

    # T-sweep (find where Mamba's advantage kicks in)
    PYTHONPATH=. python scripts/baseline_compare.py --t-sweep --output results/baseline_tsweep.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.synthetic_data import (
    generate_ar_process,
    generate_sine_waves,
    to_sequences,
    train_val_test_split,
    train_model,
    evaluate_ic,
    build_ctm,
    build_linear,
    DEVICE,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("baseline")

# ── LSTM/GRU Models ───────────────────────────────────────────────

class LSTMModel(nn.Module):
    """LSTM baseline: matches CTM's parameter count ~50K."""

    def __init__(self, n_assets: int, input_dim: int, seq_len: int,
                 hidden: int = 64, n_layers: int = 2):
        super().__init__()
        self.n_assets = n_assets
        self.seq_len = seq_len
        self.lstm = nn.LSTM(input_dim, hidden, n_layers, batch_first=True, dropout=0.2)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, T, D = x.shape
        x_flat = x.reshape(B * N, T, D)
        out, _ = self.lstm(x_flat)
        pred = self.head(out[:, -1:, :])  # (B*N, 1, 1)
        return pred.reshape(B, N, 1).expand(-1, -1, T).permute(0, 2, 1)


class GRUModel(nn.Module):
    """GRU baseline: matches CTM's parameter count ~50K."""

    def __init__(self, n_assets: int, input_dim: int, seq_len: int,
                 hidden: int = 64, n_layers: int = 2):
        super().__init__()
        self.n_assets = n_assets
        self.seq_len = seq_len
        self.gru = nn.GRU(input_dim, hidden, n_layers, batch_first=True, dropout=0.2)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, T, D = x.shape
        x_flat = x.reshape(B * N, T, D)
        out, _ = self.gru(x_flat)
        pred = self.head(out[:, -1:, :])
        return pred.reshape(B, N, 1).expand(-1, -1, T).permute(0, 2, 1)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ── Data Loader ───────────────────────────────────────────────────

def prepare_data(
    generate_fn: callable,
    gen_kwargs: dict,
    seq_len: int = 63,
    quick_test: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader, int, int]:
    """Generate data and return train/val/test loaders."""
    result = generate_fn(**gen_kwargs)
    features, targets = result[0], result[1]

    if quick_test:
        features = features[:500]
        targets = targets[:500]

    X, Y = to_sequences(features, targets, seq_len)
    tX, tY, vX, vY, teX, teY = train_val_test_split(X, Y)
    input_dim = X.shape[-1]
    n_assets = X.shape[1]

    bs = min(32, len(tX))
    train_loader = DataLoader(TensorDataset(
        torch.from_numpy(tX), torch.from_numpy(tY)), batch_size=bs, shuffle=True, drop_last=True)
    val_loader = DataLoader(TensorDataset(
        torch.from_numpy(vX), torch.from_numpy(vY)), batch_size=bs, shuffle=False)
    test_loader = DataLoader(TensorDataset(
        torch.from_numpy(teX), torch.from_numpy(teY)), batch_size=bs, shuffle=False)

    return train_loader, val_loader, test_loader, n_assets, input_dim


# ── Experiment ────────────────────────────────────────────────────

def run_single_comparison(
    name: str,
    generate_fn: callable,
    gen_kwargs: dict,
    seq_len: int = 63,
    n_epochs: int = 50,
    quick_test: bool = False,
    device: torch.device = DEVICE,
) -> Dict[str, Any]:
    """Run CTM vs LSTM vs GRU vs Linear on one dataset."""
    logger.info("=" * 60)
    logger.info("Dataset: %s", name)
    train_loader, val_loader, test_loader, n_assets, input_dim = prepare_data(
        generate_fn, gen_kwargs, seq_len, quick_test)

    models = {
        "Linear": build_linear(n_assets, input_dim, seq_len),
        "GRU": GRUModel(n_assets, input_dim, seq_len),
        "LSTM": LSTMModel(n_assets, input_dim, seq_len),
        "CTM": build_ctm(n_assets, input_dim),
    }

    results = {}
    for model_name, model in models.items():
        logger.info("Training %s...", model_name)
        params = count_params(model)

        n_ep = min(n_epochs, 10 if quick_test else n_epochs)
        n_ep = min(n_ep, 20 if model_name == "Linear" else n_ep)

        try:
            t0 = time.time()
            train_res = train_model(model, train_loader, val_loader,
                                     n_epochs=n_ep, device=device)
            eval_res = evaluate_ic(model, test_loader, device)
            rt = time.time() - t0

            results[model_name] = {
                "params": params,
                "pt_ratio": params / max(len(train_loader.dataset), 1),
                "runtime_s": round(rt, 1),
                **eval_res,
                **train_res,
            }
            logger.info("  %s: params=%d  P/T=%.1f  test_IC=%.4f  test_MSE=%.6e  (%.1fs)",
                        model_name, params, results[model_name]["pt_ratio"],
                        eval_res["ic"], eval_res["mse"], rt)
        except Exception as e:
            logger.error("  %s FAILED: %s", model_name, e)
            results[model_name] = {"error": str(e)}

    return {
        "dataset": name,
        "config": gen_kwargs,
        "models": results,
    }


# ── T-Sweep ───────────────────────────────────────────────────────

def run_t_sweep(
    T_values: List[int] = [10, 20, 40, 63, 120],
    n_epochs: int = 50,
    quick_test: bool = False,
    device: torch.device = DEVICE,
) -> List[Dict[str, Any]]:
    """Sweep T to find where Mamba's advantage over GRU/LSTM kicks in."""
    results = []
    for T in T_values:
        logger.info("T-Sweep: T=%d", T)
        gen_kwargs = {
            "n_timesteps": 2000, "n_assets": 10,
            "ar_coeffs": [0.5, 0.2, 0.1],
            "noise_std": 0.1, "snr": 1.0, "seed": 42,
        }
        if quick_test:
            gen_kwargs["n_timesteps"] = 500

        result = run_single_comparison(
            name=f"AR(3) T={T}",
            generate_fn=generate_ar_process,
            gen_kwargs=gen_kwargs,
            seq_len=T,
            n_epochs=n_epochs,
            quick_test=quick_test,
            device=device,
        )
        results.append({"T": T, **result})
    return results


# ── Main ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Baseline comparison: CTM vs LSTM vs GRU vs Linear")
    parser.add_argument("--quick-test", action="store_true")
    parser.add_argument("--t-sweep", action="store_true")
    parser.add_argument("--n-epochs", type=int, default=50)
    parser.add_argument("--output", type=str, default="results/baseline_results.json")
    args = parser.parse_args()

    device = DEVICE
    logger.info("Device: %s", device)
    if device.type == "cuda":
        logger.info("GPU: %s", torch.cuda.get_device_name(0))

    datasets = [
        ("AR(1) φ=0.9", generate_ar_process,
         {"n_timesteps": 2000, "n_assets": 10, "ar_coeffs": [0.9],
          "noise_std": 0.1, "snr": 1.0, "seed": 42}),
        ("AR(3) φ=[0.5,0.2,0.1]", generate_ar_process,
         {"n_timesteps": 2000, "n_assets": 10, "ar_coeffs": [0.5, 0.2, 0.1],
          "noise_std": 0.1, "snr": 1.0, "seed": 42}),
        ("Sine(10)", generate_sine_waves,
         {"n_timesteps": 2000, "n_assets": 10, "periods": [10],
          "noise_std": 0.1, "snr": 1.0, "seed": 42}),
        ("Sine(10,30)", generate_sine_waves,
         {"n_timesteps": 2000, "n_assets": 10, "periods": [10, 30],
          "noise_std": 0.1, "snr": 1.0, "seed": 42}),
    ]

    all_results: List[Dict] = []

    if args.t_sweep:
        logger.info("=" * 60)
        logger.info("T-SWEEP MODE")
        T_values = [10, 20, 40, 63] if args.quick_test else [10, 20, 40, 63, 120]
        tsweep_results = run_t_sweep(T_values, args.n_epochs, args.quick_test, device)
        all_results.append({"type": "t_sweep", "results": tsweep_results})

    for name, gen_fn, kwargs in datasets:
        if args.quick_test:
            kwargs["n_timesteps"] = 500
        result = run_single_comparison(
            name=name, generate_fn=gen_fn, gen_kwargs=kwargs,
            n_epochs=args.n_epochs, quick_test=args.quick_test, device=device)
        all_results.append(result)

    output = {
        "device": str(device),
        "quick_test": args.quick_test,
        "results": all_results,
    }

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2, default=str)
        logger.info("Saved to %s", args.output)

    # Print summary
    print("\n" + "=" * 100)
    fmt = "{:30s} {:>10s} {:>10s} {:>10s} {:>10s} {:>10s} {:>10s}"
    print(fmt.format("Dataset", "Model", "Params", "P/T", "Test_IC", "Test_MSE", "Time"))
    print("-" * 100)
    for r in all_results:
        if r.get("type") == "t_sweep":
            continue
        dset = r["dataset"]
        for mname, mres in r["models"].items():
            if "error" in mres:
                print(fmt.format(dset, mname, "ERR", "", "", "", ""))
            else:
                print(fmt.format(dset if mname == "Linear" else "",
                                  mname,
                                  str(mres["params"]),
                                  f"{mres['pt_ratio']:.1f}",
                                  f"{mres['ic']:.4f}",
                                  f"{mres['mse']:.2e}",
                                  f"{mres['runtime_s']:.1f}s"))
    print("=" * 100)

    # Winner matrix
    print("\nWinner matrix (IC):")
    for r in all_results:
        if r.get("type") == "t_sweep":
            continue
        ics = {m: r["models"][m]["ic"] for m in ["Linear", "GRU", "LSTM", "CTM"]
               if m in r["models"] and "error" not in r["models"][m]}
        if ics:
            winner = max(ics, key=ics.get)
            print(f"  {r['dataset']:25s} → {winner} (IC={ics[winner]:.4f})")


if __name__ == "__main__":
    main()
