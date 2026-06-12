"""Quantitative finance loss functions for GBDT.

All per-sample loss functions follow the signature:
    loss_fn(y_true: Tensor, y_pred: Tensor) -> (loss: Tensor, grad: Tensor, hess: Tensor)

Where:
    loss  - scalar loss value (for monitoring)
    grad  - [N] tensor of first derivatives dL/dy_pred_i
    hess  - [N] tensor of second derivatives d²L/dy_pred_i² (diagonal Hessian)

Portfolio-level losses (sharpe_loss) have custom signatures documented inline.
"""

import torch
import torch.nn.functional as F
import warnings
from typing import Callable, Tuple, Optional

# ── Numerical stability constants ──────────────────────────────
_EPS_HEss: float = 1e-8     # small constant for near-zero Hessians (MAE/Quantile)
_EPS_DENOM: float = 1e-8    # denominator stability (correlation, Sharpe)
_EPS_GRAD: float = 1e-12    # floor for Hessian clamping in autograd bridge
_EPS_LEAF: float = 1e-20    # floor for leaf value denominator in C++ core

# ── RankIC memory limits ──────────────────────────────────────
_RANKIC_MAX_EXACT: int = 2048   # max batch size for exact O(N²) ranking
_RANKIC_CDF_BINS: int = 512     # number of CDF bins for large-batch approximation


# ──────────────────────────────────────────────
#  Helper: autograd-based grad/hess computation
# ──────────────────────────────────────────────


def compute_gradients(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    create_graph: bool = False,
    fast_diag_hessian: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute loss, gradients, and diagonal Hessians via torch.autograd.grad.

    This is the bridge between arbitrary PyTorch loss functions and the
    GBDT C++ core which expects per-sample (gradient, hessian) pairs.

    For *sample-separable* losses (MSE, MAE, Huber) the diagonal Hessian
    is exact under both modes.  For *cross-sample* losses (e.g. RankIC):

        * ``fast_diag_hessian=True`` (default) — returns the row-sum of
          the full Hessian matrix.  This is a standard diagonal
          approximation consistent with XGBoost / LightGBM practice.
          Fast: O(1) autograd calls.

        * ``fast_diag_hessian=False`` — computes the true per-sample
          diagonal Hessian ∂²L / ∂y_pred_i² separately for each sample.
          Exact but slow: O(N) autograd calls.  Use only for small
          datasets or when the row-sum approximation is insufficient.

    Args:
        y_true: Ground truth, shape [N] or [N, 1].
        y_pred: Model predictions, shape [N] or [N, 1] (will be detached
                and have requires_grad set internally).
        loss_fn: Callable (y_true, y_pred) -> scalar loss Tensor.
        create_graph: If True, keeps the graph for higher-order gradients.
        fast_diag_hessian: If True (default), use row-sum Hessian
            approximation (fast, standard for GBDT).  If False, compute
            exact per-sample diagonal Hessian (slow, O(N) autograd calls).

    Returns:
        (loss, gradients, hessians) where:
            loss      - scalar detached Tensor
            gradients - [N] Tensor detached
            hessians  - [N] Tensor detached (abs to ensure non-negative)
    """
    y_pred_in = y_pred.detach().requires_grad_(True)
    loss = loss_fn(y_true, y_pred_in)

    grads = torch.autograd.grad(loss, y_pred_in, create_graph=fast_diag_hessian)[0]

    if fast_diag_hessian:
        # Row-sum of full Hessian matrix — exact for sample-separable
        # losses, diagonal approximation for cross-sample losses.
        # Standard GBDT practice (XGBoost / LightGBM).
        grad_sum = grads.sum()
        hessians = torch.autograd.grad(
            grad_sum, y_pred_in, create_graph=create_graph
        )[0]
    else:
        # Exact per-sample diagonal Hessian: ∂²L / ∂y_pred_i² for each i.
        # Computed via d(grad_i) / d(y_pred_i).  retain_graph=True on all
        # but the last iteration to avoid O(N) memory accumulation.
        # Correct but O(N) autograd calls — use for small batches only.
        n_samples = grads.shape[0]
        if n_samples > 10000:
            warnings.warn(
                f"Exact diagonal Hessian on {n_samples} samples: "
                "O(N) autograd calls with O(N) memory. "
                "Consider using fast_diag_hessian=True for large batches.",
                stacklevel=2,
            )
        hessians = torch.zeros_like(grads)
        for i in range(n_samples):
            retain = i < n_samples - 1  # free graph on last iteration
            (hess_i,) = torch.autograd.grad(
                grads[i], y_pred_in, retain_graph=retain
            )
            hessians[i] = hess_i[i]

    # Use abs() to ensure non-negative Hessians (negative Hessians break
    # the Newton descent direction in GBDT), then clamp to avoid division
    # by zero. This is standard practice in XGBoost / LightGBM.
    return loss.detach(), grads.detach(), hessians.abs().clamp(min=_EPS_GRAD).detach()


# ──────────────────────────────────────────────
#  Per-sample loss functions
# ──────────────────────────────────────────────


def _differentiable_rank(x: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """Differentiable rank approximation.

    For N ≤ _RANKIC_MAX_EXACT (2048): exact O(N²) pairwise sigmoid comparisons.
    For larger N: O(N·K) CDF-based approximation with K = _RANKIC_CDF_BINS (512)
    query points.  This keeps memory O(N) instead of O(N²).

        rank(x)_i ≈ Σ_j sigmoid((x_i − x_j) / temperature)
    """
    N = x.shape[0]

    # ── small batches: exact O(N²) pairwise ──
    if N <= _RANKIC_MAX_EXACT:
        diff = x.view(-1, 1) - x.view(1, -1)
        return torch.sigmoid(diff / temperature).sum(dim=1)

    # ── large batches: O(N·K) CDF-based approximation ──
    K = min(N, _RANKIC_CDF_BINS)
    x_min, x_max = x.min(), x.max()
    if x_max - x_min < 1e-8:
        return torch.full_like(x, N / 2.0)

    queries = torch.linspace(x_min, x_max, K, device=x.device)

    # soft_le[j, k] = sigmoid((q_k - x_j) / temp)
    # counts[k]   = Σ_j soft_le[j, k] — the CDF at query q_k
    soft_le = torch.sigmoid(
        (queries.view(1, -1) - x.view(-1, 1)) / temperature
    )
    counts = soft_le.sum(dim=0)

    bin_idx = torch.searchsorted(queries, x)
    bin_idx = torch.clamp(bin_idx, 1, K - 1)

    left_count = counts[bin_idx - 1]
    right_count = counts[bin_idx]
    left_query = queries[bin_idx - 1]
    right_query = queries[bin_idx]

    frac = (x - left_query) / (right_query - left_query + _EPS_DENOM)
    frac = torch.clamp(frac, 0.0, 1.0)

    return left_count + frac * (right_count - left_count)


def mse_loss(
    y_true: torch.Tensor, y_pred: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mean Squared Error loss.

    Closed-form:
        L   = mean((y_pred - y_true)²)
        g_i = 2·(y_pred_i - y_true_i)
        h_i = 2

    The C++ GBDT core expects *raw* per-sample (gradient, Hessian) pairs.
    Normalisation (e.g. dividing by N) is handled internally by the
    tree builder, so gradients and Hessians are returned unscaled.

    Returns:
        (loss_scalar, gradients, hessians)
    """
    loss = F.mse_loss(y_pred, y_true)
    grad = 2.0 * (y_pred - y_true)
    hess = 2.0 * torch.ones_like(y_pred)
    return loss, grad, hess


def mae_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mean Absolute Error loss.

    Closed-form:
        L   = mean(|y_pred - y_true|)
        g_i = sign(y_pred_i - y_true_i)
        h_i = eps  (non-zero numerical stabiliser for GBDT)

    Gradients and Hessians are raw per-sample values (not divided by N);
    the C++ tree builder handles normalisation internally.

    Args:
        eps: Small constant for Hessian to avoid zeros in tree building.

    Returns:
        (loss_scalar, gradients, hessians)
    """
    loss = torch.abs(y_pred - y_true).mean()
    grad = torch.sign(y_pred - y_true)
    hess = eps * torch.ones_like(y_pred)
    return loss, grad, hess


def huber_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    delta: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Huber loss — blends MSE and MAE with a smooth transition at ``delta``.

    Loss per sample:
        |r| <= delta : 0.5 · r²
        |r| >  delta : delta · (|r| - 0.5 · delta)

    where r = y_pred - y_true.

    Closed-form gradients / Hessians:
        g_i = r_i                          if |r_i| <= delta
              delta · sign(r_i)            otherwise

        h_i = 1                            if |r_i| <= delta
              eps                          otherwise  (numerical stability)

    Gradients and Hessians are raw per-sample values (not divided by N);
    the C++ tree builder handles normalisation internally.

    Returns:
        (loss_scalar, gradients, hessians)
    """
    residual = y_pred - y_true
    abs_residual = torch.abs(residual)

    # Loss
    quadratic = torch.clamp(abs_residual, max=delta)
    linear = abs_residual - quadratic
    loss = (0.5 * quadratic ** 2 + delta * linear).mean()

    # Gradient (raw, not divided by N)
    grad = torch.where(
        abs_residual <= delta,
        residual,
        delta * torch.sign(residual),
    )

    # Hessian (raw, not divided by N)
    hess = torch.where(
        abs_residual <= delta,
        torch.ones_like(y_pred),
        torch.full_like(y_pred, 1e-8),
    )

    return loss, grad, hess


def quantile_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    alpha: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantile (pinball) loss.

    Given a target quantile α ∈ (0, 1), the pinball loss penalises
    over-prediction and under-prediction asymmetrically:

        ρ_α(r) = max(α·r, (α−1)·r)  where r = y_pred − y_true.

    Closed-form gradients / Hessians:

        g_i = α                         if r_i > 0
              (α − 1)                   otherwise

        h_i = 1e−8                      (constant; pinball has zero
                                         second derivative, but GBDT
                                         requires positive Hessians)

    Args:
        alpha: Target quantile.  α=0.5 → median regression (loss value
               equivalent to MAE, but gradients differ: ±0.5 vs ±1).

    Returns:
        (loss_scalar, gradients, hessians)
    """
    residual = y_pred - y_true
    loss = torch.maximum(alpha * residual, (alpha - 1) * residual).mean()
    grad = torch.where(residual > 0, alpha, alpha - 1.0)
    hess = torch.full_like(y_pred, 1e-8)
    return loss, grad, hess


def log_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    eps: float = 1e-12,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Binary cross-entropy loss for classification.

    Expects ``y_pred`` as raw logits (log-odds).  Sigmoid is applied
    internally.  Targets should be in [0, 1].

    Loss:
        L = −mean(y · log(σ) + (1−y) · log(1−σ))

    where σ = sigmoid(y_pred).

    Gradients and Hessians (raw per-sample, not divided by N):
        g_i = σ_i − y_i
        h_i = σ_i · (1 − σ_i)

    Returns:
        (loss_scalar, gradients, hessians)
    """
    loss = F.binary_cross_entropy_with_logits(y_pred, y_true.float())
    prob = torch.sigmoid(y_pred)
    grad = prob - y_true
    hess = prob * (1 - prob)
    return loss, grad, hess


def rankic_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    temperature: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Differentiable Rank Information Coefficient loss.

    Approximates 1 − Spearman rank correlation so that minimising the
    loss maximises the rank correlation between predictions and targets.

    The differentiable ranking uses pairwise sigmoid comparisons:

        rank(x)_i ≈ Σ_j sigmoid((x_i − x_j) / temperature)

    Spearman is then Pearson correlation on the approximate ranks.

    Because this is a cross-sample loss, gradients and Hessians are
    computed via :func:`compute_gradients` with autograd.  The returned
    Hessian is a diagonal approximation (row-sum of the full Hessian).

    Args:
        temperature: Scaling for the sigmoid pairwise comparison.
                     Smaller values → sharper rank approximation.
                     Typical range: [0.01, 1.0].

    Returns:
        (loss_scalar, gradients, hessians)
    """

    def _loss_fn(yt: torch.Tensor, yp: torch.Tensor) -> torch.Tensor:
        rank_true = _differentiable_rank(yt, temperature)
        rank_pred = _differentiable_rank(yp, temperature)

        rt_centred = rank_true - rank_true.mean()
        rp_centred = rank_pred - rank_pred.mean()

        cov = (rt_centred * rp_centred).sum()
        # Use sqrt(sum(x²) + eps) instead of norm() because norm() has an
        # undefined second derivative at zero (d(norm)/dx = x/norm → 0/0 → NaN
        # when all predictions are constant), which breaks the GBDT Hessian.
        std_pred = torch.sqrt((rp_centred ** 2).sum() + 1e-8)
        std_true = torch.sqrt((rt_centred ** 2).sum() + 1e-8)
        rho = cov / (std_true * std_pred + 1e-8)

        return 1.0 - rho

    # Use autograd bridge for cross-sample gradients / Hessians
    return compute_gradients(y_true, y_pred, _loss_fn)


# ──────────────────────────────────────────────
#  Portfolio-level loss
# ──────────────────────────────────────────────


def sharpe_loss(
    returns: torch.Tensor,
    weights: torch.Tensor,
    risk_free: float = 0.02,
    freq: int = 252,
    eps: float = 1e-8,
    use_rowsum_hessian: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Negative Sharpe ratio as a differentiable portfolio loss.

    .. note::
        This is a **portfolio-level** function, NOT a per-sample loss.
        It operates on full batches of shape [T, N].

        The returned ``gradients`` and ``hessians`` are w.r.t. the
        **weights** tensor (the decision variable), *not* any target.
        This is useful for gradient-based portfolio optimisation.

    Args:
        returns: Asset returns, shape [T, N]  (T time-steps, N assets).
        weights: Portfolio weights, shape [T, N] (differentiable).
        risk_free: Annual risk-free rate (e.g. 0.02 for 2 %).
        freq: Number of periods per year (252 for daily, 12 for monthly).
        eps: Small constant for numerical stability in denominator.
        use_rowsum_hessian: If True (default), use row-sum Hessian
            approximation.  If False, compute exact per-element diagonal
            Hessian (slow, O(T·N) autograd calls).

    Returns:
        (loss, gradients, hessians)
        loss      — negative Sharpe ratio (scalar).
        gradients — d(loss) / d(weights), shape [T, N].
        hessians  — diagonal of d²(loss) / d(weights)², shape [T, N]
                     (abs to ensure non-negative for GBDT-style usage).
    """
    # Clone to avoid mutating the input tensor, then enable gradients
    w = weights.clone().detach().requires_grad_(True)

    portfolio_returns = (returns * w).sum(dim=1)  # [T]
    excess = portfolio_returns.mean() - risk_free / freq
    volatility = portfolio_returns.std(unbiased=False)
    sharpe = excess / (volatility + eps)
    loss = -sharpe

    grads = torch.autograd.grad(loss, w, create_graph=True)[0]

    if use_rowsum_hessian:
        # Row-sum of full Hessian matrix — diagonal approximation
        grad_sum = grads.sum()
        hessians = torch.autograd.grad(grad_sum, w, create_graph=False)[0]
    else:
        # Exact per-element diagonal Hessian: ∂²L / ∂w_ti²
        hessians = torch.zeros_like(grads)
        for t in range(w.shape[0]):
            for i in range(w.shape[1]):
                # Only retain graph when there are more iterations ahead
                retain = (t < w.shape[0] - 1) or (i < w.shape[1] - 1)
                (hess_ti,) = torch.autograd.grad(
                    grads[t, i], w, retain_graph=retain
                )
                hessians[t, i] = hess_ti[t, i]

    return loss.detach(), grads.detach(), torch.abs(hessians).detach()


# ──────────────────────────────────────────────
#  Composite loss
# ──────────────────────────────────────────────


def composite_quant_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    alpha: float = 0.5,
    beta: float = 0.3,
    gamma: float = 0.2,
    temperature: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Composite quantitative loss (stateless, turnover penalises |ŷ|).

        L = α · MSE + β · (1 − RankIC) + γ · MSE(ŷ, 0)

    The turnover term penalises prediction magnitude (L2 on predictions)
    rather than actual inter-round change.  For proper turnover tracking
    across boosting rounds, use :class:`CompositeQuantLoss` instead.

    Returns:
        (loss_scalar, gradients, hessians)
    """

    def _composite_loss_fn(yt: torch.Tensor, yp: torch.Tensor) -> torch.Tensor:
        mse = F.mse_loss(yp, yt)
        rank_true = _differentiable_rank(yt, temperature)
        rank_pred = _differentiable_rank(yp, temperature)
        rt_c = rank_true - rank_true.mean()
        rp_c = rank_pred - rank_pred.mean()
        cov = (rt_c * rp_c).sum()
        std_true = torch.sqrt((rt_c ** 2).sum() + 1e-8)
        std_pred = torch.sqrt((rp_c ** 2).sum() + 1e-8)
        rho = cov / (std_true * std_pred + 1e-8)
        ic_loss = 1.0 - rho
        l2_penalty = F.mse_loss(yp, torch.zeros_like(yp))
        return alpha * mse + beta * ic_loss + gamma * l2_penalty

    return compute_gradients(y_true, y_pred, _composite_loss_fn)


class CompositeQuantLoss:
    """Composite loss with proper inter-round turnover tracking.

    Callable class suitable for use with ``GBDTTrainer(loss_fn=...)``.
    Tracks the previous iteration's predictions so the turnover penalty
    measures actual prediction changes between boosting rounds:

        L = α · MSE + β · (1 − RankIC) + γ · MSE(ŷ_t, ŷ_{t-1})

    On the first call (no previous predictions), turnover = 0.

    Args:
        alpha: MSE weight.
        beta: RankIC weight.
        gamma: Turnover-penalty weight.
        temperature: Sigmoid temperature for differentiable ranking.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.3,
        gamma: float = 0.2,
        temperature: float = 0.1,
    ):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.temperature = temperature
        self._prev_pred: Optional[torch.Tensor] = None

    def __call__(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        def _loss_fn(yt: torch.Tensor, yp: torch.Tensor) -> torch.Tensor:
            mse = F.mse_loss(yp, yt)
            rank_true = _differentiable_rank(yt, self.temperature)
            rank_pred = _differentiable_rank(yp, self.temperature)
            rt_c = rank_true - rank_true.mean()
            rp_c = rank_pred - rank_pred.mean()
            cov = (rt_c * rp_c).sum()
            std_true = torch.sqrt((rt_c ** 2).sum() + 1e-8)
            std_pred = torch.sqrt((rp_c ** 2).sum() + 1e-8)
            rho = cov / (std_true * std_pred + 1e-8)
            ic_loss = 1.0 - rho

            if self._prev_pred is not None:
                turnover = F.mse_loss(yp, self._prev_pred)
            else:
                turnover = torch.tensor(0.0, device=yp.device)

            return self.alpha * mse + self.beta * ic_loss + self.gamma * turnover

        result = compute_gradients(y_true, y_pred, _loss_fn)
        # Store detached predictions for next round's turnover computation
        self._prev_pred = y_pred.detach()
        return result


__all__ = [
    "compute_gradients",
    "mse_loss",
    "mae_loss",
    "huber_loss",
    "quantile_loss",
    "log_loss",
    "rankic_loss",
    "sharpe_loss",
    "composite_quant_loss",
    "CompositeQuantLoss",
    "_differentiable_rank",
]
