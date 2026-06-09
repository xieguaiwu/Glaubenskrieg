"""MambaBlock with parallel associative scan (Mamba-2 style).

The sequential scan in mamba_block.py is O(T) and preserves autograd via
torch.stack. This module implements the associative scan algorithm (Blelloch
1990, Martin & Cundy 2017), reducing time complexity to O(log T) on GPU.

On CPU the sequential scan is actually faster -- this is provided for:
  1. Reference implementation of the associative scan
  2. GPU training readiness
  3. Architectural completeness in the CTM guide
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F
import warnings

from .mamba_block import BaseMambaBlock


def associative_scan(
    A_bar: torch.Tensor, B_bar_x: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Parallel associative scan using log-space cumulative products.

    Solves the linear recurrence h_t = A_bar_t * h_{t-1} + B_bar_x_t
    with h_{-1} = 0. Closed-form solution via cumulative products:

        h_t = cp[t] * sum_{k=0}^{t} B_bar_x_k / cp[k]

    where cp[t] = prod_{i=0}^{t} A_bar_i.

    Uses log-space cumsum for cp to avoid underflow when A_bar ∈ (0,1].
    Reciprocal is computed as exp(-log_cp) for numerical stability.

    Parameters
    ----------
    A_bar : (B, T, D)
        Diagonal transition factors A_bar_t = exp(Δ_t * A). Must be > 0.
    B_bar_x : (B, T, D)
        Input contributions Δ_t * B_t * x_conv_t.

    Returns
    -------
    h : (B, T, D)
        Hidden states h_t for all t.
    h_final : (B, D)
        Final hidden state h_T.
    """
    B, T, D = A_bar.shape
    if T <= 1:
        return B_bar_x, B_bar_x[:, -1]

    # Log-space cumulative product: cp[t] = exp(log_cp[t])
    # Clamp A_bar away from zero to avoid log(0) = -inf
    log_A_bar = torch.log(torch.clamp(A_bar, min=1e-12))
    log_cp = torch.cumsum(log_A_bar, dim=1)   # (B, T, D)
    cp = torch.exp(log_cp)                     # (B, T, D)

    # Reciprocal via log-space: 1/cp[k] = exp(-log_cp[k]), always finite
    inv_cp = torch.exp(-log_cp)                # (B, T, D)

    # sum_term[t] = sum_{k=0}^{t} B_bar_x_k / cp[k]
    sum_term = torch.cumsum(B_bar_x * inv_cp, dim=1)  # (B, T, D)

    # h_t = cp[t] * sum_term[t]
    h = cp * sum_term  # (B, T, D)
    h_final = h[:, -1, :]  # (B, D)

    return h, h_final


class MambaBlockParallel(BaseMambaBlock):
    """MambaBlock using associative scan instead of sequential loop."""

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, _ = x.shape

        z, z1, z2, x_conv, delta, A_bar, B_proj, C_proj = self._forward_steps_1_3(x)

        assert not torch.isnan(z).any(), \
            f"NaN detected in MambaBlockParallel after input projection (step 1) — shape={z.shape}"

        # Step 4: Per-channel associative scan
        # For each state dimension s in 0..d_state, run associative scan on
        # the time series with transition A_bar and input B_bar_s * x_conv
        # h_t[s] = A_bar_t * h_{t-1}[s] + Δ_t * B_t[s] * x_conv_t
        #        = A_bar_t * h_{t-1}[s] + B_bar_x_t[s]

        # B_bar_x: (B, T, d_inner, d_state)
        B_bar_x = (delta.unsqueeze(-1) * B_proj.unsqueeze(-2)) * x_conv.unsqueeze(-1)
        # Reshape to merge inner and state dims for the scan
        # (B, T, d_inner * d_state)
        B_bar_x_2d = B_bar_x.reshape(B, T, self.d_inner * self.d_state)
        A_bar_rep = A_bar.unsqueeze(-1).expand(-1, -1, -1, self.d_state)
        A_bar_2d = A_bar_rep.reshape(B, T, self.d_inner * self.d_state)

        h_2d, _ = associative_scan(A_bar_2d, B_bar_x_2d)
        # Reshape back: (B, T, d_inner, d_state)
        h = h_2d.reshape(B, T, self.d_inner, self.d_state)
        h_final = h[:, -1]  # (B, d_inner, d_state)

        # Step 5: Output via C_t
        # y_t = C_t^T @ h_t + D * x_conv_t
        C_proj_exp = C_proj.unsqueeze(-1)  # (B, T, d_state, 1)
        y_ssm = torch.matmul(h, C_proj_exp).squeeze(-1)  # (B, T, d_inner)
        y_ssm = y_ssm + self.D * x_conv
        assert not torch.isnan(y_ssm).any(), \
            f"NaN detected in MambaBlockParallel after SSM scan (step 5) — shape={y_ssm.shape}"

        # Step 6: SiLU gating
        y = F.silu(z2) * y_ssm

        assert not torch.isnan(y).any(), \
            f"NaN detected in MambaBlockParallel after SiLU gating (step 6) — shape={y.shape}"

        return y, h_final

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, d_state={self.d_state}, "
            f"expand={self.expand}, scan=parallel_associative"
        )
