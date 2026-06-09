"""Shared quantitative metrics used across training and loss modules."""

from __future__ import annotations

import math

import torch


def sharpe_ratio_torch(
    values: torch.Tensor,
    annual_factor: float = 252.0,
    ddof: int = 1,
) -> torch.Tensor:
    """Annualised Sharpe ratio from a PyTorch tensor of returns.

    Computes ``sqrt(annual_factor) * mean(values) / std(values, ddof=ddof)``
    with a small epsilon guard against zero variance.

    The time dimension is auto-detected:
    - 1-D ``(T,)``           → time is the only dimension (dim=0)
    - 2-D ``(N, T)``         → time is the last dimension (dim=-1)
    - 3-D+ ``(B, T, C)``    → time is second-to-last (dim=-2)

    Parameters
    ----------
    values : (T,), (N, T), or (B, T, C) tensor of returns.
    annual_factor : scaling factor (252 for daily).
    ddof : delta degrees of freedom for std (1 = sample, 0 = population).

    Returns
    -------
    Scalar Sharpe ratio (negative for loss minimisation).
    """
    ndim = values.dim()
    if ndim == 1:
        time_dim = 0
    elif ndim == 2:
        time_dim = -1  # (N, T) — last dim is time
    else:
        time_dim = -2  # (B, T, C) — second-to-last is time

    N_time = values.shape[time_dim]
    if N_time < 2:
        return values.new_zeros(())
    mu = torch.mean(values, dim=time_dim)
    var = torch.var(values, dim=time_dim, correction=ddof)
    # Near-zero variance → return 0 (constant returns have undefined Sharpe)
    if torch.all(var < 1e-12):
        return values.new_zeros(())
    sigma = torch.sqrt(var + 1e-6)
    sr = math.sqrt(annual_factor) * mu / sigma
    return torch.mean(sr)
