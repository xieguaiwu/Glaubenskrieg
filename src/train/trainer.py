"""Training pipeline with walk-forward validation and early stopping."""

from __future__ import annotations

from typing import Any, Dict, List, Type

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
from scipy.stats import spearmanr
from ._walk_forward_utils import walk_forward_windows
from ..utils.metrics import sharpe_ratio_torch


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: Any,
    device: torch.device,
    grad_clip: float = 1.0,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_grad_norm = 0.0
    n_batches = 0
    n_samples = 0

    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        pred = model(batch_x)
        loss = loss_fn(pred, batch_y)

        optimizer.zero_grad()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        n = batch_x.size(0)
        total_loss += loss.item() * n
        total_grad_norm += float(grad_norm)
        n_batches += 1
        n_samples += n

    avg_loss = total_loss / max(n_samples, 1)
    avg_grad_norm = total_grad_norm / max(n_batches, 1)
    return {"avg_loss": avg_loss, "grad_norm": avg_grad_norm}


@torch.no_grad()
def _batch_ic(preds: torch.Tensor, targs: torch.Tensor) -> float:
    """Spearman rank IC between predictions and targets."""
    p = preds.cpu().numpy().ravel()
    t = targs.cpu().numpy().ravel()
    valid = np.isfinite(p) & np.isfinite(t)
    if valid.sum() < 2:
        return 0.0
    ic, _ = spearmanr(p[valid], t[valid])
    return float(ic) if np.isfinite(ic) else 0.0


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: Any,
    device: torch.device,
    num_regression: int | None = None,
    is_multi_asset: bool | None = None,
) -> Dict[str, float]:
    """Walk-forward validation.

    Parameters
    ----------
    model, dataloader, loss_fn, device : standard.
    num_regression : int or None
        Number of regression channels in the model output. When provided,
        the last dimension is split as ``[regression_channels, 3_class_logits]``
        and only regression channels are used for Sharpe/accuracy metrics.
        When None (default), falls back to heuristic detection.
    is_multi_asset : bool or None
        If True, treat output as multi-asset (use multi-asset mean pooling).
        If False, use single-asset output layout (last timestep, channel 0).
        If None (default), auto-detect by checking ``hasattr(model, 'n_assets')``.
    """
    model.eval()
    total_loss = 0.0
    n_samples = 0
    all_preds: List[torch.Tensor] = []
    all_targets: List[torch.Tensor] = []

    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        pred = model(batch_x)
        loss = loss_fn(pred, batch_y)
        n = batch_x.size(0)
        total_loss += loss.item() * n
        n_samples += n

        n_channels = pred.shape[-1]

        # Determine multi-asset status
        if num_regression is not None:
            n_assets_m = num_regression
            _is_ma = True
        elif is_multi_asset is None:
            _is_ma = hasattr(model, 'n_assets') and getattr(model, 'n_assets', 1) > 1
            n_assets_m = (n_channels // 4) if _is_ma else 0
        else:
            _is_ma = is_multi_asset
            n_assets_m = (n_channels // 4) if _is_ma else 0

        # Extract regression predictions (last timestep)
        if _is_ma:
            # Multi-asset: (B, T, N*C) → take per-asset regression channels: (B, N)
            reg_pred_per_asset = pred[:, -1, :n_assets_m]  # (B, N)
            # Per-asset targets (last timestep): (B, N)
            if batch_y.dim() == 3:
                reg_targ_per_asset = batch_y[:, -1, :]  # (B, N)
            elif batch_y.dim() == 4:
                # (B, T, N) shape common in multi-asset
                reg_targ_per_asset = batch_y[:, -1, :]  # (B, N)
            else:
                reg_targ_per_asset = batch_y[:, -1, 0].unsqueeze(-1).expand(-1, n_assets_m)
            # Flatten to (B*N,) for per-asset Sharpe / directional accuracy
            all_preds.append(reg_pred_per_asset.reshape(-1).cpu())
            all_targets.append(reg_targ_per_asset.reshape(-1).cpu())
        else:
            # Single-asset: (B, T, C) → take channel 0
            reg_pred = pred[:, -1, 0].cpu()  # (B,)
            if batch_y.dim() == 1:
                reg_targ = batch_y.cpu()
            elif batch_y.dim() >= 2:
                reg_targ = batch_y[:, -1, 0].cpu() if batch_y.dim() == 3 else batch_y[:, -1].cpu()
            else:
                reg_targ = batch_y.cpu()
            all_preds.append(reg_pred)
            all_targets.append(reg_targ)

    avg_loss = total_loss / max(n_samples, 1)

    preds = torch.cat(all_preds)
    targs = torch.cat(all_targets)

    # Compute Sharpe on strategy returns: sign(pred) × actual_return
    strategy_returns = preds.sign() * targs
    sharpe = sharpe_ratio_torch(strategy_returns.unsqueeze(0), annual_factor=252.0, ddof=1)

    eps = 0.005
    correct = ((preds < -eps) == (targs < -eps)) | ((preds > eps) == (targs > eps))
    dir_acc = correct.float().mean()

    ic = _batch_ic(preds, targs)

    return {
        "avg_loss": avg_loss,
        "sharpe_ratio": float(sharpe),
        "directional_accuracy": float(dir_acc),
        "spearman_ic": ic,
    }


class WalkForwardTrainer:
    """Walk-forward cross-validation trainer.

    Parameters
    ----------
    model_class : nn.Module class to instantiate for each window.
    model_params : dict of constructor kwargs.
    device : torch device.
    """
    def __init__(
        self,
        model_class: Type[nn.Module],
        model_params: Dict[str, Any],
        device: torch.device | str = "cpu",
    ) -> None:
        self.model_class = model_class
        self.model_params = model_params
        self.device = torch.device(device)

    def train_walk_forward(
        self,
        data: torch.Tensor,
        targets: torch.Tensor,
        train_window: int,
        val_window: int,
        purge_period: int,
        step_size: int,
        n_epochs: int,
        batch_size: int = 32,
        lr: float = 3e-4,
        weight_decay: float = 0.1,
        grad_clip: float = 1.0,
        patience: int = 10,
    ) -> List[Dict[str, Any]]:
        """Run walk-forward training.

        Returns list of per-window metrics dicts.
        """
        N = len(data)
        results: List[Dict[str, Any]] = []

        for pos, train_end, purge_end, val_end in walk_forward_windows(
            N, train_window, val_window, purge_period, step_size
        ):
            train_data = data[pos:train_end]
            train_targ = targets[pos:train_end]
            val_data = data[purge_end:val_end]
            val_targ = targets[purge_end:val_end]

            train_ds = TensorDataset(train_data, train_targ)
            val_ds = TensorDataset(val_data, val_targ)

            # No shuffle — walk-forward windows already provide causality.
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

            model = self.model_class(**self.model_params).to(self.device)
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=lr, weight_decay=weight_decay
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=n_epochs
            )
            # Extract regression channel (channel 0) before MSE to avoid
            # broadcasting mismatch with (B, T, 4) model output vs (B, T, 1) target
            def _mse_ch0(pred: torch.Tensor, targ: torch.Tensor) -> torch.Tensor:
                return nn.functional.mse_loss(pred[..., :1], targ)

            loss_fn = _mse_ch0

            best_sharpe = -float("inf")
            no_improve = 0
            window_metrics: List[Dict[str, float]] = []

            for epoch in range(n_epochs):
                train_m = train_epoch(model, train_loader, optimizer, loss_fn, self.device, grad_clip)
                val_m = validate(model, val_loader, loss_fn, self.device)
                scheduler.step()

                window_metrics.append({
                    "epoch": epoch,
                    "train_loss": train_m["avg_loss"],
                    "val_loss": val_m["avg_loss"],
                    "val_sharpe": val_m["sharpe_ratio"],
                    "val_dir_acc": val_m["directional_accuracy"],
                    "grad_norm": train_m["grad_norm"],
                })

                if val_m["sharpe_ratio"] > best_sharpe:
                    best_sharpe = val_m["sharpe_ratio"]
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        break

            results.append({
                "window_start": pos,
                "window_end": val_end,
                "best_sharpe": best_sharpe,
                "epochs_run": len(window_metrics),
                "metrics": window_metrics,
            })

        return results
