"""Fused multi-head cross-asset attention with GBDT modulation.

Extends CrossAssetAttention with multi-head QKV splitting, GBDT-derived
pairwise bias modulation, and attention dropout. Drop-in replacement for
CrossAssetAttention in MultiAssetCTM.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class GBDTModulator(nn.Module):
    """Converts per-asset GBDT predictions to pairwise attention biases.

    Architecture: MLP(1 -> hidden_dim -> hidden_dim -> 1) with LayerNorm,
    GELU activation, and Dropout.  Applied elementwise to each pairwise
    prediction difference.

    Parameters
    ----------
    hidden_dim : int
        Hidden dimension of the internal MLP (default 64).
    dropout : float
        Dropout rate applied after each GELU (default 0.1).
    """

    def __init__(self, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout

        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, gbdt_preds: torch.Tensor) -> torch.Tensor:
        """Compute pairwise prediction-difference biases.

        Parameters
        ----------
        gbdt_preds : (N,) or (B, N)
            Per-asset GBDT predictions.

        Returns
        -------
        (N, N) or (B, N, N)
            Pairwise bias matrix.  Entry [i, j] = MLP(preds[i] - preds[j]).
        """
        if gbdt_preds.dim() == 1:
            diff = gbdt_preds.unsqueeze(-1) - gbdt_preds.unsqueeze(-2)
            bias = self.net(diff.unsqueeze(-1)).squeeze(-1)
        else:
            diff = gbdt_preds.unsqueeze(-1) - gbdt_preds.unsqueeze(-2)
            flat = diff.reshape(-1, 1)
            bias = self.net(flat).reshape(diff.shape)
        return bias

    def param_count(self) -> int:
        """Return the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def extra_repr(self) -> str:
        return f"hidden_dim={self.hidden_dim}, dropout={self.dropout_rate}"


class FusedMultiHeadCrossAttention(nn.Module):
    """Multi-head cross-asset attention with fused bias sources.

    Supports learnable adjacency bias (following CrossAssetAttention),
    GBDT-derived modulation bias via GBDTModulator, and attention dropout
    for regularisation.

    Parameters
    ----------
    n_assets : int
        Number of stocks in the cross-section.
    d_model : int
        Model dimension (must be divisible by n_heads).
    n_heads : int
        Number of attention heads (default 4).
    dropout : float
        Dropout rate applied to attention weights after softmax (default 0.1).
    use_gbdt_bias : bool
        Whether to instantiate a GBDTModulator (default True).
    modulator_hidden : int
        Hidden dimension for the GBDTModulator MLP (default 64).
    """

    def __init__(
        self,
        n_assets: int,
        d_model: int,
        n_heads: int = 4,
        dropout: float = 0.1,
        use_gbdt_bias: bool = True,
        modulator_hidden: int = 64,
        use_time_amp: bool = False,
    ):
        super().__init__()

        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads}). "
                f"Got remainder {d_model % n_heads}."
            )

        self.n_assets = n_assets
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.dropout_rate = dropout
        self.use_gbdt_bias = use_gbdt_bias

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.adj_bias = nn.Parameter(torch.zeros(n_assets, n_assets))

        if use_gbdt_bias:
            self.gbdt_modulator = GBDTModulator(
                hidden_dim=modulator_hidden, dropout=dropout
            )
        else:
            self.gbdt_modulator = None

        self.attn_dropout = nn.Dropout(dropout)

        self.use_time_amp = use_time_amp
        if use_time_amp:
            self.time_gamma = nn.Parameter(torch.tensor(1.0))
        else:
            self.time_gamma = None
        # Cache B, T for time amplification
        self._B: Optional[int] = None
        self._T: Optional[int] = None

        self._gbdt_preds: Optional[torch.Tensor] = None

    def set_gbdt_predictions(self, preds: torch.Tensor) -> None:
        """Cache GBDT predictions for the next forward pass.

        Parameters
        ----------
        preds : (N,) or (batch, N)
            Per-asset GBDT predictions.  If batched, the batch dimension
            should match the B dimension of forward input x.
        """
        self._gbdt_preds = preds

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Cross-asset multi-head mixing.

        Parameters
        ----------
        x : (B, N, T, D) or (B*T, N, D)
            Hidden states, where B = batch, N = n_assets,
            T = time steps, D = d_model.

        Returns
        -------
        (B, N, T, D) or (B*T, N, D)
            Mixed representation - same shape as input.
        """
        needs_reshape = x.dim() == 4
        B = T = 0  # bound for type checker; only used when needs_reshape
        if needs_reshape:
            B, N, T, D = x.shape
            self._B = B
            self._T = T
            x_2d = x.reshape(B * T, N, D)
            BT = B * T
        else:
            BT, N, D = x.shape
            x_2d = x

        q = self.q_proj(x_2d)
        k = self.k_proj(x_2d)
        v = self.v_proj(x_2d)

        q = q.reshape(BT, N, self.n_heads, self.d_k).transpose(1, 2)
        k = k.reshape(BT, N, self.n_heads, self.d_k).transpose(1, 2)
        v = v.reshape(BT, N, self.n_heads, self.d_k).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn = attn + self.adj_bias.unsqueeze(0).unsqueeze(0)

        if self.use_gbdt_bias and self._gbdt_preds is not None:
            assert self.gbdt_modulator is not None
            mod_bias = self.gbdt_modulator(self._gbdt_preds)
            if mod_bias.dim() == 2:
                mod_bias = mod_bias.unsqueeze(0).unsqueeze(0)
            else:
                B_gbdt = mod_bias.size(0)
                mod_bias = mod_bias.unsqueeze(1)
                if B_gbdt > 1 and B_gbdt != BT:
                    if BT % B_gbdt != 0:
                        raise ValueError(
                            f"Batch alignment failed: BT={BT} not divisible by B_gbdt={B_gbdt}"
                        )
                    # Note: repeat_interleave is used intentionally (not repeat) to
                    # replicate per-time-step, preserving temporal order when the
                    # time axis carries per-asset biases.
                    T_repeat = BT // B_gbdt
                    mod_bias = mod_bias.repeat_interleave(T_repeat, dim=0)
            # ── Variant B: Time-amplified GBDT bias ──
            if self.use_time_amp and self.time_gamma is not None and needs_reshape:
                T_amp = self._T if self._T is not None else 1
                tau = torch.arange(T_amp, device=x.device, dtype=x.dtype) / max(T_amp - 1, 1)  # (T,) in [0,1]
                gamma_clamped = torch.clamp(self.time_gamma, 0.0, 5.0)
                amp = 1.0 + gamma_clamped * tau  # (T,) — grows from 1.0 to 1+gamma
                B_amp = self._B if self._B is not None else (mod_bias.shape[0] // T_amp if T_amp > 0 else 1)
                # Expand amp to match mod_bias: mod_bias is (BT, N, N) or (B*T, N, N)
                # amp is (T,) — need to repeat for each batch element then broadcast
                amp_exp = amp.repeat_interleave(B_amp, dim=0).view(-1, 1, 1)  # (B*T, 1, 1)
                mod_bias = mod_bias * amp_exp

            attn = attn + mod_bias

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(BT, N, self.d_model)
        out = self.out_proj(out)

        if needs_reshape:
            out = out.reshape(B, N, T, D)

        return out

    def param_count(self) -> int:
        """Return the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def extra_repr(self) -> str:
        return (
            f"n_assets={self.n_assets}, d_model={self.d_model}, "
            f"n_heads={self.n_heads}, d_k={self.d_k}, "
            f"use_gbdt_bias={self.use_gbdt_bias}"
        )
