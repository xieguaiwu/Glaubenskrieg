"""Quantitative loss functions for CTM stock prediction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.metrics import sharpe_ratio_torch


def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean squared error: mean((pred - target)^2)."""
    return torch.mean((pred - target) ** 2)


def sharpe_loss(
    pred_returns: torch.Tensor,
    actual_returns: Optional[torch.Tensor] = None,
    annual_factor: float = 252.0,
) -> torch.Tensor:
    """Negative Sharpe ratio, computed on trading strategy returns.

    When ``actual_returns`` is provided, the Sharpe is computed on the
    **strategy PnL**: ``position = tanh(pred_returns)`` multiplied by
    ``actual_returns``.  This prevents the model from gaming the Sharpe
    by collapsing prediction variance or inflating prediction magnitude,
    since the Sharpe now depends on *correct directional bets*.

    When ``actual_returns`` is None (legacy), falls back to computing
    Sharpe on raw predictions — **not recommended** due to the
    variance-collapse pathology (see defect_patterns.md).

    Parameters
    ----------
    pred_returns : (B, T, C) predicted returns.
    actual_returns : (B, T, C) or None — ground-truth returns.
    annual_factor : scaling factor (252 for daily).

    Returns
    -------
    Scalar ``-Sharpe`` (minimised during training).
    """
    if pred_returns.dim() < 3:
        raise ValueError(
            f"sharpe_loss expects (B, T, C) input, got shape with {pred_returns.dim()} dims"
        )
    if actual_returns is not None:
        # Strategy returns: bounded positions × actual returns
        # tanh keeps positions in [-1, 1], preserving gradients everywhere
        positions = torch.tanh(pred_returns)
        strategy_returns = positions * actual_returns
        return -sharpe_ratio_torch(strategy_returns, annual_factor=annual_factor, ddof=1)
    import warnings
    warnings.warn(
        "sharpe_loss called without actual_returns — prone to variance collapse. "
        "Pass actual_returns to compute Sharpe on strategy PnL.",
        UserWarning, stacklevel=2,
    )
    return -sharpe_ratio_torch(pred_returns, annual_factor=annual_factor, ddof=1)


def directional_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Cross-entropy for 3-class direction prediction."""
    if logits.dim() > 2:
        # Move class dim to position -2 for F.cross_entropy (expects (N, C, ...)).
        # Works for (B,T,C) → (B,C,T) and (B,N,T,C) → (B,N,C,T).
        logits = logits.transpose(-2, -1)
    return F.cross_entropy(logits, targets, weight=weight)


def pinball_loss(pred: torch.Tensor, target: torch.Tensor, tau: float) -> torch.Tensor:
    """Quantile (pinball) loss for VaR estimation."""
    error = target - pred
    return torch.mean(torch.where(error >= 0, tau * error, (tau - 1.0) * error))


@dataclass
class LossConfig:
    lambda_mse: float = 1.0
    lambda_sharpe: float = 0.1
    lambda_directional: float = 1.0
    lambda_pinball: float = 0.1
    lambda_reg: float = 0.01
    pinball_tau: float = 0.05
    skip_l2_reg: bool = False
    class_weight: Optional[torch.Tensor] = None


class LearnableWeights(nn.Module):
    """Uncertainty-weighted multi-task loss (Kendall et al., 2018).

    Learns log σ² for each task. Loss = ½·exp(-log_var)·task_loss + ½·log_var.
    """
    def __init__(self) -> None:
        super().__init__()
        self.log_var_mse = nn.Parameter(torch.zeros(()))
        self.log_var_sharpe = nn.Parameter(torch.zeros(()))
        self.log_var_directional = nn.Parameter(torch.zeros(()))
        self.log_var_pinball = nn.Parameter(torch.zeros(()))

    def clamp(self, lo: float = -10.0, hi: float = 10.0) -> None:
        with torch.no_grad():
            for p in self.parameters():
                p.clamp_(lo, hi)


def composite_loss(
    predictions: torch.Tensor,
    regression_target: torch.Tensor,
    class_targets: torch.Tensor,
    model_parameters: List[nn.Parameter],
    config: LossConfig,
    learnable_weights: Optional[LearnableWeights] = None,
    num_regression: int = 1,
) -> torch.Tensor:
    """Composite multi-objective loss (MSE + Sharpe + Directional + Pinball + L2).

    Parameters
    ----------
    predictions : (B, T, C). C dim layout: [0:output_dim]=regression, [output_dim:]=class logits.
    regression_target : (B, T, 1).
    class_targets : (B, T) integer labels 0/1/2.
    model_parameters : weight params for L2 reg (exclude biases).
    config : LossConfig.
    learnable_weights : optional LearnableWeights.
    num_regression : int. Number of regression channels in predictions.
        Required for models like MultiAssetCTM where per-asset outputs are
        interleaved (e.g. N assets each with ``output_dim + 3`` channels).

    Returns
    -------
    Scalar loss.
    """
    loss = predictions.new_zeros(())
    output_dim = num_regression

    have_reg = regression_target.numel() > 0
    have_cls = class_targets.numel() > 0

    if have_reg:
        pred_reg = predictions[..., :output_dim]
        loss_mse = mse_loss(pred_reg, regression_target)
        loss_sr = sharpe_loss(pred_reg, actual_returns=regression_target)

        if learnable_weights is not None:
            learnable_weights.clamp()
            loss = loss + 0.5 * torch.exp(-learnable_weights.log_var_mse) * loss_mse \
                        + 0.5 * learnable_weights.log_var_mse
            loss = loss + 0.5 * torch.exp(-learnable_weights.log_var_sharpe) * loss_sr \
                        + 0.5 * learnable_weights.log_var_sharpe
        else:
            loss = loss + config.lambda_mse * loss_mse
            loss = loss + config.lambda_sharpe * loss_sr

        loss_pb = pinball_loss(pred_reg, regression_target, config.pinball_tau)
        if learnable_weights is not None:
            loss = loss + 0.5 * torch.exp(-learnable_weights.log_var_pinball) * loss_pb \
                        + 0.5 * learnable_weights.log_var_pinball
        else:
            loss = loss + config.lambda_pinball * loss_pb

    if have_cls:
        logits = predictions[..., output_dim:]
        cls_targets_2d = class_targets
        num_cls_channels = logits.shape[-1]
        n_assets_cls = num_cls_channels // 3 if num_cls_channels > 3 else 1
        if n_assets_cls > 1 and cls_targets_2d.dim() == 3:
            B, T, N = cls_targets_2d.shape
            assert N == n_assets_cls, \
                f"cls_targets last dim {N} != n_assets_cls {n_assets_cls}"
            logits = logits.reshape(B, T, N, 3).permute(0, 2, 1, 3).reshape(B * N, T, 3)
            cls_targets_2d = cls_targets_2d.reshape(B, T, N).permute(0, 2, 1).reshape(B * N, T)
        cw = config.class_weight.to(predictions.device) if config.class_weight is not None else None
        loss_dir = directional_loss(logits, cls_targets_2d, cw)
        if learnable_weights is not None:
            loss = loss + 0.5 * torch.exp(-learnable_weights.log_var_directional) * loss_dir \
                        + 0.5 * learnable_weights.log_var_directional
        else:
            loss = loss + config.lambda_directional * loss_dir

    if config.lambda_reg > 0.0 and not config.skip_l2_reg:
        reg = predictions.new_zeros(())
        for p in model_parameters:
            reg = reg + torch.sum(p * p)
        loss = loss + 0.5 * config.lambda_reg * reg

    return loss
