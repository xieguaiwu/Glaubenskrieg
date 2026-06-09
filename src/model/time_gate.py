"""Horizon-dependent exponential decay gate for CTM ↔ GBDT blending.

At forecast horizon t (0 = 1-step-ahead, increasing = further future),
CTM weight decays as:  w(t) = σ(α · exp(-β · t/T) + γ)
This biases the ensemble toward CTM for near-term and GBDT for long-term.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TimeDecayGate(nn.Module):
    """Horizon-dependent exponential decay gate for CTM ↔ GBDT blending.
    At forecast horizon t (0 = 1-step-ahead, increasing = further future),
    CTM weight decays as:  w(t) = σ(α · exp(-β · t/T) + γ)
    This biases the ensemble toward CTM for near-term and GBDT for long-term.
    """

    def __init__(self, alpha_init=2.0, beta_init=4.0, gamma_init=0.0):
        # alpha: initial gate scale, sigma(2.0) ≈ 0.88 CTM at t=0
        # beta: decay rate, sigma(2.0-4.0) ≈ 0.12 CTM at t=T-1
        # gamma: floor weight, allows the gate to not fully go to 0
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha_init))
        self.beta = nn.Parameter(torch.tensor(beta_init))
        self.gamma = nn.Parameter(torch.tensor(gamma_init))

    def forward(self, ctm_pred, gbdt_pred):
        """
        Args:
            ctm_pred: (B, N, T, C) CTM regression predictions
            gbdt_pred: (B, N, T) GBDT scalar predictions per asset/step
        Returns:
            blended: (B, N, T, C) weighted average
            weights: (T,) CTM weight at each horizon for monitoring
        """
        T = ctm_pred.shape[2]
        device = ctm_pred.device
        t = torch.arange(T, device=device, dtype=torch.float32)
        # Normalize: t/T gives [0, 0.016, 0.032, ..., 0.984]
        tau = t / max(T - 1, 1)
        # w(t) = σ(α · exp(-β · τ) + γ)
        weights = torch.sigmoid(self.alpha * torch.exp(-self.beta * tau) + self.gamma)  # (T,)
        # Broadcast: (T,) → (1, 1, T, 1)
        w_4d = weights.view(1, 1, T, 1)
        gbdt_4d = gbdt_pred.unsqueeze(-1)  # (B, N, T) → (B, N, T, 1)
        blended = w_4d * ctm_pred + (1.0 - w_4d) * gbdt_4d
        return blended, weights

    def extra_repr(self):
        return f'alpha={self.alpha.item():.3f}, beta={self.beta.item():.3f}, gamma={self.gamma.item():.3f}'
