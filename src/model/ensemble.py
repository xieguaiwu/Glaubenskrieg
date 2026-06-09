"""IC-weighted ensemble signal fusion for P0 integration.

Provides utilities for combining CTM and GBDT predictions
using rolling Information Coefficient (Spearman rank correlation)
as adaptive weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
from scipy.stats import spearmanr


def compute_ic(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Spearman rank correlation between predictions and ground truth.

    Parameters
    ----------
    y_true : (N,) array of ground-truth values.
    y_pred : (N,) array of predicted values.

    Returns
    -------
    Spearman ρ as a float. Returns 0.0 when inputs are constant
    (zero variance) or when fewer than 2 samples are provided.
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()

    if y_true.shape[0] < 2 or y_pred.shape[0] < 2:
        return 0.0

    y_true_std = np.std(y_true, ddof=1)
    y_pred_std = np.std(y_pred, ddof=1)
    if y_true_std < 1e-12 or y_pred_std < 1e-12:
        return 0.0

    result = spearmanr(y_true, y_pred)
    rho = float(result[0])  # type: ignore[arg-type]
    if np.isnan(rho) or np.isinf(rho):
        return 0.0
    return float(rho)


def ic_weighted_fusion(
    ctm_pred: np.ndarray,
    gbdt_pred: np.ndarray,
    y_true: np.ndarray,
    lookback: int = 252,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Fuse CTM and GBDT predictions via rolling IC-weighted average.

    For each time step t (starting after ``lookback`` samples), the
    IC of each model is computed over the trailing window of length
    ``lookback``.  The weight applied to the CTM at step t is:

        w = IC_ctm / (IC_ctm + IC_gbdt + 1e-8)

    and the fused signal is   w * ctm_pred + (1 - w) * gbdt_pred.

    Parameters
    ----------
    ctm_pred : (T,) array of CTM predictions.
    gbdt_pred : (T,) array of GBDT predictions.
    y_true : (T,) array of ground-truth targets.
    lookback : rolling window length (default 252).

    Returns
    -------
    fused_signal : (T,) array of IC-weighted fused predictions.
    weights_dict : dict with keys ``ctm_weight`` and ``gbdt_weight``,
        each an (T,) array of per-step weights.
    """
    ctm_pred = np.asarray(ctm_pred, dtype=np.float64).ravel()
    gbdt_pred = np.asarray(gbdt_pred, dtype=np.float64).ravel()
    y_true = np.asarray(y_true, dtype=np.float64).ravel()

    T = len(ctm_pred)
    ctm_weight = np.full(T, 0.5)
    gbdt_weight = np.full(T, 0.5)

    for t in range(lookback, T):
        start = t - lookback
        ic_c = compute_ic(y_true[start:t], ctm_pred[start:t])
        ic_g = compute_ic(y_true[start:t], gbdt_pred[start:t])

        # Use absolute IC values so weights stay in [0, 1] even when ICs
        # have opposite signs (prevents out-of-range weights that could
        # amplify noise in the fused prediction).
        ic_c_abs = abs(ic_c)
        ic_g_abs = abs(ic_g)
        denom = ic_c_abs + ic_g_abs + 1e-8
        ctm_weight[t] = ic_c_abs / denom
        gbdt_weight[t] = ic_g_abs / denom

    fused = ctm_weight * ctm_pred + gbdt_weight * gbdt_pred
    weights_dict = {"ctm_weight": ctm_weight, "gbdt_weight": gbdt_weight}
    return fused, weights_dict


@dataclass
class EnsembleConfig:
    """Configuration for the CTM + GBDT ensemble.

    When ``use_ic_weighting=False``, :attr:`ctm_weight` and :attr:`gbdt_weight`
    are normalised to sum to 1.0 before fusion.
    """

    ctm_weight: float = 0.5
    gbdt_weight: float = 0.5
    use_ic_weighting: bool = True
    ic_lookback: int = 252
    min_samples_for_ic: int = 20


@dataclass
class EnsembleSignal:
    """Output of the ensemble evaluation pipeline."""

    ctm_pred: np.ndarray
    gbdt_pred: np.ndarray
    fused: np.ndarray
    ic_ctm: float
    ic_gbdt: float
    fused_ic: float
    weights: Dict[str, np.ndarray]


def evaluate_ensemble(
    ctm_pred: np.ndarray,
    gbdt_pred: np.ndarray,
    y_true: np.ndarray,
    config: EnsembleConfig,
) -> EnsembleSignal:
    """Full ensemble evaluation pipeline.

    1. If ``config.use_ic_weighting`` and the sample count is ≥
       ``config.min_samples_for_ic``, apply IC-weighted fusion.
    2. Otherwise fall back to the static weights from ``config``.
    3. Compute overall Spearman IC for each model and the fused signal.

    Parameters
    ----------
    ctm_pred : (T,) array.
    gbdt_pred : (T,) array.
    y_true : (T,) array.
    config : EnsembleConfig.

    Returns
    -------
    EnsembleSignal with fused predictions, per-model / fused ICs,
    and per-step weight arrays.
    """
    ctm_pred = np.asarray(ctm_pred, dtype=np.float64).ravel()
    gbdt_pred = np.asarray(gbdt_pred, dtype=np.float64).ravel()
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    T = len(ctm_pred)

    if config.use_ic_weighting and T >= config.min_samples_for_ic:
        fused, weights = ic_weighted_fusion(
            ctm_pred, gbdt_pred, y_true, lookback=config.ic_lookback,
        )
    else:
        total = config.ctm_weight + config.gbdt_weight
        cw_val = config.ctm_weight / total if total > 0 else 0.5
        gw_val = config.gbdt_weight / total if total > 0 else 0.5
        cw = np.full(T, cw_val)
        gw = np.full(T, gw_val)
        fused = cw * ctm_pred + gw * gbdt_pred
        weights = {"ctm_weight": cw, "gbdt_weight": gw}

    ic_ctm = compute_ic(y_true, ctm_pred)
    ic_gbdt = compute_ic(y_true, gbdt_pred)
    ic_fused = compute_ic(y_true, fused)

    return EnsembleSignal(
        ctm_pred=ctm_pred,
        gbdt_pred=gbdt_pred,
        fused=fused,
        ic_ctm=ic_ctm,
        ic_gbdt=ic_gbdt,
        fused_ic=ic_fused,
        weights=weights,
    )


def _differentiable_ranking_helper(
    x: torch.Tensor, temperature: float = 0.1
) -> torch.Tensor:
    """Differentiable rank approximation via sigmoid-based pairwise comparison.

    Approximates ranks by computing P(x_i > x_j) with a sigmoid temperature.
    Returns normalised ranks in [0, 1].

    Parameters
    ----------
    x : (N,) tensor of values to rank.
    temperature : sigmoid temperature controlling rank approximation sharpness.

    Returns
    -------
    (N,) tensor of approximate ranks scaled to [0, 1].
    """
    N = x.shape[0]
    if N <= 1:
        return torch.zeros_like(x)

    x_i = x.unsqueeze(-1)  # (N, 1)
    x_j = x.unsqueeze(-2)  # (1, N)
    pairwise = torch.sigmoid((x_i - x_j) / temperature)  # (N, N)
    ranks = pairwise.sum(dim=-1) / N  # (N,)
    return ranks


def rankic_loss_gbdt_style(
    y_true: torch.Tensor, y_pred: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """RankIC loss in the form expected by the GBDT Python bridge.

    Returns (loss, grad, hess) so the GBDT framework can directly
    optimise the Spearman rank correlation.  Gradients are computed
    via a differentiable rank approximation.

    Parameters
    ----------
    y_true : (N,) ground-truth values.
    y_pred : (N,) predicted values.

    Returns
    -------
    loss : scalar tensor, loss = 1 - RankIC.
    grad : (N,) tensor, first derivative w.r.t. y_pred.
    hess : (N,) tensor, second derivative w.r.t. y_pred (constant
        Hessian approximation for numerical stability).
    """
    y_true = y_true.detach().flatten()
    # Detach is intentional — this function computes gradients w.r.t. prediction
    # values for GBDT tree splitting, not for neural network backprop.
    y_pred = y_pred.detach().flatten().requires_grad_(True)
    N = y_pred.shape[0]

    if N <= 1:
        return (
            torch.tensor(0.0, device=y_pred.device),
            torch.zeros_like(y_pred),
            torch.ones_like(y_pred),
        )

    true_rank = _differentiable_ranking_helper(y_true)
    pred_rank = _differentiable_ranking_helper(y_pred)

    # RankIC ≈ Pearson correlation on approximate ranks
    tr = true_rank - true_rank.mean()
    pr = pred_rank - pred_rank.mean()
    tr_std = tr.std(correction=1) + 1e-8
    pr_std = pr.std(correction=1) + 1e-8
    rank_ic = (tr * pr).mean() / (tr_std * pr_std)

    loss = 1.0 - rank_ic

    # Gradient w.r.t. y_pred  (d loss / d y_pred)
    grad = torch.autograd.grad(loss, y_pred, create_graph=False)[0]

    # Hessian diagonal ≈ constant approximation for stability
    hess = torch.ones_like(y_pred)

    return loss.detach(), grad.detach(), hess
