"""
MambaBlock: Selective State Space Model (S6) implementation in pure PyTorch.

Mathematical formulation follows Gu & Dao (2023) "Mamba: Linear-Time Sequence
Modeling with Selective State Spaces", adapted with Algorithm 1 from the
CTM Architecture Guide.

The selective scan processes each channel independently:
  h_t[i] = A_bar_t[i] * h_{t-1}[i] + B_bar_t[i,:] * x_conv_t[i]
  y_ssm_t[i] = C_t.T @ h_t[i] + D[i] * x_conv_t[i]

Key features:
  - Pure PyTorch (no CUDA kernels, no mamba-ssm package)
  - Preserves full autograd graph via torch.stack (no in-place ops)
  - Causal conv1d with left-padding (no look-ahead)
  - SiLU activation for conv output and gating
  - Softplus-based discretization step size
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import warnings

import torch.nn.functional as F


class BaseMambaBlock(nn.Module):
    """Shared base for sequential (MambaBlock) and parallel (MambaBlockParallel) SSM blocks.

    Provides common __init__, _init_weights, and forward steps 1-3
    (input projection, causal conv, selective parameters).
    """
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: str | int = "auto",
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = expand * d_model
        self._nan_counter = 0

        if dt_rank == "auto":
            self.dt_rank = max(1, int(math.ceil(d_model / 16)))
        else:
            self.dt_rank = int(dt_rank)

        self.W_in = nn.Parameter(torch.empty(2 * self.d_inner, d_model))
        self.W_conv = nn.Parameter(torch.empty(self.d_inner, d_conv))
        self.b_conv = nn.Parameter(torch.zeros(self.d_inner))
        self.A_log = nn.Parameter(torch.empty(self.d_inner))
        self.D = nn.Parameter(torch.zeros(self.d_inner))
        self.W_dt_proj_down = nn.Parameter(torch.empty(self.d_inner, self.dt_rank))
        self.W_dt_proj_up = nn.Parameter(torch.empty(self.dt_rank, self.d_inner))
        self.b_dt = nn.Parameter(torch.empty(self.d_inner))
        self.W_B = nn.Parameter(torch.empty(d_state, self.d_inner))
        self.W_C = nn.Parameter(torch.empty(d_state, self.d_inner))

        self._init_weights()

    def _init_weights(self) -> None:
        for name, tensor in [
            ("W_in", self.W_in),
            ("W_conv", self.W_conv),
            ("W_dt_proj_down", self.W_dt_proj_down),
            ("W_dt_proj_up", self.W_dt_proj_up),
            ("W_B", self.W_B),
            ("W_C", self.W_C),
        ]:
            nn.init.xavier_uniform_(tensor)
        with torch.no_grad():
            self.A_log.copy_(-torch.exp(torch.randn_like(self.A_log) - 2.0))
            # Clamp A_log to [-5, 5] so exp(A_log) ∈ [e⁻⁵≈0.0067, e⁵≈148]
            # prevents overflow/underflow when computing A_bar = exp(delta * (-exp(A_log)))
            self.A_log.data.clamp_(min=-5, max=5)
            self.b_dt.fill_(0.001)

    def _forward_steps_1_3(self, x: torch.Tensor):
        """Forward steps 1-3: input projection, causal conv, selective parameters.

        Returns
        -------
        Tuple[z, z1, z2, x_conv, delta, A_bar, B_proj, C_proj]
        """
        B, T, _ = x.shape
        d_inner = self.d_inner

        z = F.linear(x, self.W_in)
        z1, z2 = z[..., :d_inner], z[..., d_inner:]

        z1_3d = z1.transpose(1, 2)
        z1_padded = F.pad(z1_3d, (self.d_conv - 1, 0))
        w_3d = self.W_conv.unsqueeze(1)
        x_conv_3d = F.conv1d(z1_padded, w_3d, self.b_conv, groups=d_inner)
        x_conv = F.silu(x_conv_3d.transpose(1, 2))
        assert not torch.isnan(x_conv).any(), \
            f"NaN detected in MambaBlock after conv (step 2) — shape={x_conv.shape}"

        # Low-rank Δ projection: down-project to dt_rank, then up-project to d_inner
        delta = x_conv @ self.W_dt_proj_down                      # (B, T, dt_rank)
        delta = F.softplus(delta @ self.W_dt_proj_up + self.b_dt)  # (B, T, d_inner)
        delta = torch.clamp(delta, min=1e-5, max=20.0)

        # Safe A_bar computation: clamp A_log so exp(A_log) doesn't overflow,
        # and delta has bounds [1e-5, 20] so delta * (-exp(A_log)) stays well-behaved.
        A_bar = torch.exp(delta * (-torch.exp(self.A_log.clamp(min=-5, max=5))))

        B_proj = F.linear(x_conv, self.W_B)
        C_proj = F.linear(x_conv, self.W_C)

        return z, z1, z2, x_conv, delta, A_bar, B_proj, C_proj


class MambaBlock(BaseMambaBlock):
    """Selective SSM (S6) block — the core building block of the Mamba architecture.

    This implements a single Mamba layer consisting of:
      1. Input projection (2× expansion)
      2. Causal depthwise 1D convolution (local feature extraction)
      3. Selective scan (content-dependent state space recurrence)
      4. SiLU gating
    """

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, _ = x.shape
        d_inner = self.d_inner
        d_state = self.d_state

        z, z1, z2, x_conv, delta, A_bar, B_proj, C_proj = self._forward_steps_1_3(x)

        assert not torch.isnan(z).any(), \
            f"NaN detected in MambaBlock after input projection (step 1) — shape={z.shape}"

        # ── Step 4: Discretization ──
        B_bar = delta.unsqueeze(-1) * B_proj.unsqueeze(-2)

        # ── Step 5: Selective scan (sequential on CPU) ──
        h = torch.zeros(B, d_inner, d_state, device=x.device, dtype=x.dtype)
        y_ssm_steps: list[torch.Tensor] = []
        for t in range(T):
            a_t = A_bar[:, t].unsqueeze(-1)
            b_t = B_bar[:, t]
            x_t = x_conv[:, t].unsqueeze(-1)
            h = h * a_t + b_t * x_t
            # Clamp hidden state to prevent explosion/underflow in deep recurrence
            h = h.clamp(min=-1e4, max=1e4)
            c_t = C_proj[:, t].unsqueeze(-1)
            y_t = torch.matmul(h, c_t).squeeze(-1)
            y_t = y_t + self.D * x_conv[:, t]
            y_ssm_steps.append(y_t)

        y_ssm = torch.stack(y_ssm_steps, dim=1)
        h_final = h
        assert not torch.isnan(y_ssm).any(), \
            f"NaN detected in MambaBlock after SSM scan (step 5) — shape={y_ssm.shape}"

        # ── Step 6: SiLU gating ──
        y = F.silu(z2) * y_ssm
        assert not torch.isnan(y).any(), \
            f"NaN detected in MambaBlock after SiLU gating (step 6) — shape={y.shape}"

        return y, h_final

    def extra_repr(self) -> str:
        """String representation for print(model)."""
        return (
            f"d_model={self.d_model}, d_state={self.d_state}, "
            f"d_conv={self.d_conv}, expand={self.expand}, "
            f"dt_rank={self.dt_rank}, d_inner={self.d_inner}"
        )
