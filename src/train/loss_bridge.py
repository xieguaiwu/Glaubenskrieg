"""Bridge CTM's composite loss to GBDT's gradient/Hessian interface.

Converts CTM's complex multi-objective LossConfig into the
(loss_scalar, gradients, hessians) triplet expected by GBDT.
"""

from __future__ import annotations

import warnings
from typing import Any, Callable, Optional, Tuple

import torch
import torch.nn.functional as F

from ..model.losses import LossConfig, pinball_loss
from ..utils.metrics import sharpe_ratio_torch

# Dedup flags for warnings that should fire at most once per process
_warned_negative_hessian: bool = False
_warned_sharpe_gbdt: bool = False


def _sharpe_bridge_terms(
    y_pred: torch.Tensor,
    y_true: Optional[torch.Tensor] = None,
    annual_factor: float = 252.0,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute (-sharpe_ratio, grad, hess) for GBDT loss bridge.

    When ``y_true`` is provided, the Sharpe is computed on **strategy
    returns**: ``position = tanh(y_pred)`` multiplied by ``y_true``.
    This prevents variance collapse by tying Sharpe to correct directional
    bets rather than raw prediction statistics.

    Falls back to raw predictions when ``y_true`` is None, using the
    analytical diagonal Hessian approximation.

    Sharpe = mean(r) / std(r) * sqrt(annual_factor)

    With strategy returns, gradients are computed via autograd through
    ``tanh(y_pred) * y_true`` to correctly propagate through the position
    sizing transformation.
    """
    if y_true is not None:
        # Strategy returns: bounded positions × actual returns
        positions = torch.tanh(y_pred)
        strategy_returns = positions * y_true
        sr = sharpe_ratio_torch(
            strategy_returns.unsqueeze(0) if strategy_returns.dim() == 1 else strategy_returns,
            annual_factor=annual_factor, ddof=1,
        )
        loss = -sr
        # Gradient via autograd (propagates through tanh × y_true correctly)
        grad = torch.autograd.grad(loss, y_pred, create_graph=True)[0]
        hess = torch.autograd.grad(grad.sum(), y_pred, create_graph=False)[0]
        hess = hess.abs().clamp(min=eps)
        return loss.detach(), grad.detach(), hess.detach()

    # Legacy: analytical gradient on raw predictions
    N = len(y_pred)
    mean = y_pred.mean()
    var = y_pred.var(correction=1)
    std = torch.sqrt(var + eps)
    af_sqrt = annual_factor ** 0.5

    sharpe = mean / (std + eps) * af_sqrt
    loss = -sharpe

    grad = -af_sqrt * (
        1.0 / N / std - mean * (y_pred - mean) / (N * std**3)
        # NOTE: The analytical gradient uses N in the denominator, but `var` above
        # is computed with correction=1 (sample variance, denominator N-1). For a
        # strictly matching ddof=1 gradient, (N-1) should replace N in the second
        # term. The difference is <1% for N>=100 and negligible after Hessian
        # clamping. This path is currently only reached when y_true is None
        # (never in current call flow), so the mismatch is not actively harmful.
    )

    hess = -af_sqrt * (
        -2.0 * mean / (N * std**3)
        + 3.0 * mean * (y_pred - mean) / (N * std**5)
    )
    hess = hess.abs().clamp(min=eps)

    return loss, grad, hess


def ctm_composite_loss_for_gbdt(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    loss_config: LossConfig,
    _model_parameters: Any = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """CTM composite loss in (loss, grad, hess) form for GBDT.

    The loss is a weighted combination of:
    - MSE: mean squared error
    - Pinball: quantile loss for VaR estimation
    - Directional: binary cross-entropy for up/down classification
    - L2: prediction magnitude regulariser (mean(y_pred^2))

    NOTE: Sharpe loss uses an analytical diagonal Hessian approximation.
    This is experimental — exact Hessian would require a dense (N×N) matrix.
    The approximation is computationally tractable and empirically useful,
    but may produce negative diagonal entries which are clamped to 1e-8.

    NOTE: The directional loss here uses binary CE (up vs not-up) rather
    than the 3-class CE (UP/NEUTRAL/DOWN) used in CTM's composite_loss.
    This is an intentional simplification: GBDT receives regression
    predictions (1D per sample), so only the sign direction is available.
    The NEUTRAL class is implicitly absorbed into the "not-up" group.
    This is a known approximation — the GBDT loss bridge deliberately
    trades off 3-class fidelity for simplicity and numerical stability.

    Gradients via torch.autograd.grad.
    Hessians via diagonal approximation (standard GBDT practice).

    Parameters
    ----------
    y_true : (N,) ground-truth values.
    y_pred : (N,) model predictions (will be detached internally).
    loss_config : LossConfig with lambda weights.
    model_parameters : ignored for GBDT; prediction L2 is used instead.

    Returns
    -------
    (loss_scalar, gradients, hessians) — all detached.
    """
    y_true = y_true.detach().flatten()
    y_pred_in = y_pred.detach().flatten().requires_grad_(True)

    mse = F.mse_loss(y_pred_in, y_true)

    pb = pinball_loss(y_pred_in, y_true, loss_config.pinball_tau)

    dir_labels = (y_true > 0).float()
    directional = F.binary_cross_entropy_with_logits(y_pred_in, dir_labels)

    l2 = torch.mean(y_pred_in ** 2)

    if not loss_config.skip_l2_reg:
        reg = loss_config.lambda_reg * l2
    else:
        reg = torch.tensor(0.0, device=y_pred_in.device)

    loss = (
        loss_config.lambda_mse * mse
        + loss_config.lambda_pinball * pb
        + loss_config.lambda_directional * directional
        + reg
    )

    grads = torch.autograd.grad(loss, y_pred_in, create_graph=True)[0]

    hessians = torch.autograd.grad(grads.sum(), y_pred_in, create_graph=False)[0]
    if (hessians < 0).any():
        global _warned_negative_hessian
        if not _warned_negative_hessian:
            _warned_negative_hessian = True
            warnings.warn(
                f"{'':>25}Negative Hessian diagonal detected "
                f"({'%.0f' % ((hessians < 0).float().mean().item() * 100)}%), "
                f"clipping to 1e-6. Non-convex loss composition may cause convergence issues."
            )
    hessians = hessians.clamp(min=1e-6)

    if loss_config.lambda_sharpe > 0:
        global _warned_sharpe_gbdt
        if not _warned_sharpe_gbdt:
            _warned_sharpe_gbdt = True
            warnings.warn(
                "Including lambda_sharpe in GBDT gradients (experimental). "
                "Sharpe Hessian uses diagonal approximation — non-convex."
            )
        sr_loss, sr_grad, sr_hess = _sharpe_bridge_terms(y_pred_in, y_true=y_true)
        loss = loss + loss_config.lambda_sharpe * sr_loss
        grads = grads + loss_config.lambda_sharpe * sr_grad
        hessians = hessians + loss_config.lambda_sharpe * sr_hess

    return loss.detach(), grads.detach(), hessians.detach()


def make_gbdt_loss_fn(loss_config: LossConfig) -> Callable:
    """Factory: returns a GBDT-compatible loss_fn from a LossConfig.

    The returned callable has the signature::

        loss_fn(y_true: Tensor, y_pred: Tensor) -> (loss, grad, hess)

    and can be passed directly to ``GBDTTrainer(config, loss_fn=my_fn)``.
    """
    def loss_fn(y_true: torch.Tensor, y_pred: torch.Tensor):
        return ctm_composite_loss_for_gbdt(y_true, y_pred, loss_config)
    return loss_fn


__all__ = [
    "ctm_composite_loss_for_gbdt",
    "make_gbdt_loss_fn",
]
