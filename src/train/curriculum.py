"""Progressive curriculum dropout for Variant C.

During training, CTM predictions at later time steps are randomly dropped
with probability p = (epoch/n_epochs) * (t/T), forcing the model to become
robust to long-horizon CTM degradation.

At inference: no dropout applied.
"""

from __future__ import annotations

import torch


def apply_curriculum_dropout(
    output: torch.Tensor,
    epoch: int,
    n_epochs: int,
    n_assets: int,
    output_dim: int,
    max_dropout: float = 0.3,
    gbdt_preds: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply progressive curriculum dropout to CTM regression predictions.

    With probability p = (epoch / n_epochs) * (t / T_max), zero out CTM
    predictions at time step t.  Optionally replace dropped predictions
    with GBDT predictions when available.

    Only applied during training (epoch >= 0).  Classification logits
    are never dropped.

    Parameters
    ----------
    output : (B, T_total, C) model output tensor
    epoch : current epoch (0-indexed). Pass -1 to disable dropout.
    n_epochs : total epochs
    n_assets : number of assets (N)
    output_dim : regression output dimension per asset
    max_dropout : cap on dropout probability (default 0.3)
    gbdt_preds : optional (B, N, T) GBDT predictions for replacement

    Returns
    -------
    (B, T_total, C) — same shape as input, with regression channels modified
    """
    if epoch < 0:
        return output  # disabled (validation/inference)

    B, T_total, C = output.shape
    num_reg = n_assets * output_dim

    epoch_frac = min(epoch / max(n_epochs - 1, 1), 0.5)  # [0, 0.5] — capped for safety

    tau = torch.arange(T_total, device=output.device).float() / max(T_total - 1, 1)
    p_drop = torch.clamp(epoch_frac * tau, max=max_dropout)  # (T,)  max=0.15 at any horizon

    # Extract regression slice
    reg_output = output[..., :num_reg].reshape(B, T_total, n_assets, output_dim)

    # Dropout mask: 1=keep CTM, 0=drop
    keep_prob = 1.0 - p_drop  # (T,)
    mask = torch.bernoulli(
        keep_prob.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        .expand(B, T_total, n_assets, output_dim)
    )

    if gbdt_preds is not None:
        # Blend: keep CTM where mask=1, use GBDT where mask=0
        gbdt_expanded = gbdt_preds.unsqueeze(-1)  # (B, N, T) → (B, N, T, 1)
        if gbdt_expanded.shape[2] == n_assets and gbdt_expanded.shape[1] != n_assets:
            gbdt_expanded = gbdt_expanded.permute(0, 3, 2, 1)  # normalize to (B, N, T, 1)
        # Broadcast mask over output_dim
        reg_output = mask * reg_output + (1 - mask) * gbdt_expanded
    else:
        # Pure dropout: zero where mask=0
        reg_output = reg_output * mask

    reg_flat = reg_output.reshape(B, T_total, num_reg)
    return torch.cat([reg_flat, output[..., num_reg:]], dim=-1)
