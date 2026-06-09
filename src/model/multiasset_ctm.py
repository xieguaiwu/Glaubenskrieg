"""Multi-asset CTM model for cross-stock prediction.

Extends CTMStockModel to handle multiple stocks simultaneously with:
  1. Per-stock embedding (learnable stock ID → vector)
  2. Shared Mamba backbone across stocks
  3. Cross-asset mixing via graph attention
  4. Per-asset output heads

Architecture:
  Input(B, N, T, D) → Embedding → Shared Mamba×L → CrossAssetMix → Output(B, N, T, C)
"""

from __future__ import annotations

import math
from typing import Optional
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ctm_model import RMSNorm, CTMStockModel

from .loop_ctm import RecurrentCTM
from .fused_attention import FusedMultiHeadCrossAttention

try:
    from .time_gate import TimeDecayGate
except ImportError:
    TimeDecayGate = None  # type: ignore[assignment]


class CrossAssetAttention(nn.Module):
    """Lightweight cross-asset attention mixer.

    Learns which stocks influence which other stocks via a learnable
    adjacency matrix. Applied after Mamba encoding.
    """
    def __init__(self, n_assets: int, d_model: int, attn_dropout: float = 0.1):
        super().__init__()
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        rank = min(4, n_assets)
        self.adj_U = nn.Parameter(torch.randn(n_assets, rank) * 0.01)
        self.adj_V = nn.Parameter(torch.randn(n_assets, rank) * 0.01)
        self.attn_dropout = attn_dropout
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Cross-asset mixing.

        Parameters
        ----------
        x : (B, N, T, D) — batch, n_assets, time, d_model

        Returns
        -------
        (B, N, T, D) mixed representation.
        """
        B, N, T, D = x.shape

        # Merge batch and time for attention over assets
        x_2d = x.reshape(B * T, N, D)
        q = self.query(x_2d)
        k = self.key(x_2d)
        v = self.value(x_2d)

        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(D)
        adj_bias = self.adj_U @ self.adj_V.T  # low-rank (N, r) × (r, N) → (N, N)
        attn = attn + adj_bias.unsqueeze(0)
        attn = F.softmax(attn, dim=-1)
        attn = F.dropout(attn, p=self.attn_dropout, training=self.training)

        out = torch.matmul(attn, v)
        out = self.out_proj(out)
        return out.reshape(B, N, T, D)


class MultiAssetCTM(nn.Module):
    """Multi-asset CTM model for portfolio-level prediction.

    Parameters
    ----------
    n_assets : int
        Number of stocks/assets.
    input_dim : int
        Features per stock per time step.
    model_dim : int, default=64
        Hidden dimension.
    state_dim : int, default=16
        SSM state dimension.
    n_layers : int, default=3
        Number of Mamba layers.
    output_dim : int, default=1
        Regression outputs per asset.
    embedding_dim : int or None, default=None
        Learnable asset embedding dim. None = model_dim.
    use_cross_attention : bool, default=True
        Enable cross-asset attention.
    use_fused_attention : bool, default=False
        When True, replace ``CrossAssetAttention`` with
        ``FusedMultiHeadCrossAttention`` (multi-head with GBDT modulator).
        Requires ``src.model.fused_attention`` to be importable.
    n_heads : int, default=4
        Number of attention heads for fused attention.
    modulator_hidden : int, default=64
        Hidden dimension for the GBDT modulator in fused attention.
    dropout : float, default=0.1
    conv_kernel : int, default=3
        Causal conv kernel size for shared backbone.
    use_decomp : bool, default=False
        Enable seasonal-trend decomposition in backbone.
    bidirectional : bool, default=False
        Enable bidirectional Mamba in backbone.
    parallel_scan : bool, default=False
        Use parallel scan implementation in backbone.
    return_hidden : bool, default=False
        Return hidden states from backbone (unused by MultiAssetCTM).
    n_loop_iters : int, default=3
        Number of encoder loop iterations when ``use_fused_attention=True``
        (only used with ``RecurrentCTM`` backbone).
    loop_dropout : float, default=0.1
        Dropout rate for loop residual in ``RecurrentCTM``.
    """
    def __init__(
        self,
        n_assets: int,
        input_dim: int,
        model_dim: int = 64,
        state_dim: int = 16,
        n_layers: int = 3,
        output_dim: int = 1,
        embedding_dim: int | None = None,
        use_cross_attention: bool = True,
        use_fused_attention: bool = False,
        use_time_gate: bool = False,
        n_heads: int = 4,
        modulator_hidden: int = 64,
        use_time_amp: bool = False,
        dropout: float = 0.1,
        conv_kernel: int = 3,
        use_decomp: bool = False,
        bidirectional: bool = False,
        parallel_scan: bool = False,
        return_hidden: bool = False,
        n_loop_iters: int = 3,
        loop_dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.n_assets = n_assets
        self.model_dim = model_dim
        self.use_cross_attention = use_cross_attention
        self.use_fused_attention = use_fused_attention
        self.output_dim = output_dim
        embed_dim = embedding_dim if embedding_dim is not None else model_dim

        self._nan_counter = 0

        # Learnable per-asset embedding (e.g., sector, vol regime)
        self.asset_embed = nn.Embedding(n_assets, embed_dim)
        self.cond_proj = nn.Linear(embed_dim, model_dim)

        # Shared single-asset backbone per stock.
        # When use_fused_attention=True, use RecurrentCTM (supports in-loop
        # cross-asset attention via its encode(cross_attn=...) parameter).
        # Otherwise use the standard CTMStockModel.
        # output_dim=0 skips unused output heads when only encode() is called.
        backbone_output_dim = 0
        if use_fused_attention:
            # RecurrentCTM imported at module top — guaranteed available
            backbone_cls: type = RecurrentCTM
            backbone_kwargs: dict[str, object] = dict(
                n_loop_iters=n_loop_iters,
                loop_dropout=loop_dropout,
                in_loop_fusion=True,
                n_assets=n_assets,
            )
        else:
            backbone_cls = CTMStockModel
            backbone_kwargs: dict[str, object] = {}
        self.single_asset_model = backbone_cls(
            input_dim=input_dim,
            model_dim=model_dim,
            state_dim=state_dim,
            conv_kernel=conv_kernel,
            n_layers=n_layers,
            output_dim=backbone_output_dim,
            dropout=dropout,
            use_decomp=use_decomp,
            bidirectional=bidirectional,
            parallel_scan=parallel_scan,
            return_hidden=False,
            **backbone_kwargs,
        )

        # Cross-asset mixing
        if use_fused_attention:
            # FusedMultiHeadCrossAttention imported at module top
            self.cross_attn = FusedMultiHeadCrossAttention(
                n_assets=n_assets,
                d_model=model_dim,
                n_heads=n_heads,
                modulator_hidden=modulator_hidden,
                use_time_amp=use_time_amp,
            )
        elif use_cross_attention:
            self.cross_attn = CrossAssetAttention(n_assets, model_dim, attn_dropout=dropout)
        else:
            self.cross_attn = None
        self.cross_norm = RMSNorm(model_dim) if use_cross_attention else None
        self.cross_ff = nn.Sequential(
            nn.Linear(model_dim, model_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim * 2, model_dim),
        ) if use_cross_attention else None

        self.dropout = nn.Dropout(dropout)

        # Output heads (per-asset)
        self.head_regression = nn.Linear(model_dim, output_dim)
        self.head_classification = nn.Linear(model_dim, 3)

        # ── Time-decay gate (Variant A: progressive CTM→GBDT blending) ──
        if use_time_gate and TimeDecayGate is not None:
            self.time_gate = TimeDecayGate()
        else:
            self.time_gate = None

        # Total output channels for composite_loss slicing
        self.num_output_channels = output_dim + 3

    def forward(self, x: torch.Tensor, asset_ids: Optional[torch.Tensor] = None, gbdt_preds: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : (B, N, T, input_dim) features.
        asset_ids : (N,) or (B, N) — per-asset IDs for embedding.
            Default: range(n_assets).
        gbdt_preds : (B, N, T) or (B*T, N) or None
            GBDT predictions for fusion. Only used when
            ``use_fused_attention=True``. Passed to
            ``FusedMultiHeadCrossAttention.set_gbdt_predictions()``
            before the cross-attention step.

        Returns
        -------
        (B, T, N * num_output_channels) — flattened output.
            Reshape to (B, T, N, -1) for per-asset view.
        """
        B, N, T, D_in = x.shape
        device = x.device

        if asset_ids is None:
            asset_ids = torch.arange(N, device=device).expand(B, -1)

        # ── Per-stock embedding ──
        emb = self.asset_embed(asset_ids)  # (B, N, embed_dim)

        if emb.dim() == 2:
            emb = emb.unsqueeze(0).expand(B, -1, -1)

        # ── Process each stock independently through shared backbone ──
        # Reshape: (B*N, T, D_in) for single-asset model
        x_flat = x.reshape(B * N, T, D_in)
        emb_flat = emb.reshape(B * N, -1)

        # Shared Mamba backbone via encode() (includes input_proj, conv, Mamba blocks)
        cond_embed = emb_flat.unsqueeze(1).expand(-1, T, -1)  # (B*N, T, embed_dim)
        cond_proj = self.cond_proj(cond_embed)                  # (B*N, T, model_dim)

        if self.use_fused_attention:
            assert self.cross_attn is not None  # guaranteed by __init__
            if gbdt_preds is not None:
                # FusedMultiHeadCrossAttention.set_gbdt_predictions expects
                # (batch, N) — reshape (B, N, T) → (B*T, N) when needed
                if gbdt_preds.dim() == 3:  # (B, N, T)
                    gbdt_preds_2d = gbdt_preds.permute(0, 2, 1).reshape(-1, N)
                else:
                    gbdt_preds_2d = gbdt_preds
                self.cross_attn.set_gbdt_predictions(gbdt_preds_2d)  # type: ignore[union-attr]
            z = self.single_asset_model.encode(x_flat, cond=cond_proj, cross_attn=self.cross_attn)  # type: ignore[call-arg]
            # Reshape: flatten (B*N, T, D) back to (B, N, T, D)
            z = z.reshape(B, N, T, self.model_dim)
            # Skip post-loop cross-attention (already handled in-loop)
            if torch.isnan(z).any():
                self._nan_counter += 1
                warnings.warn("NaN detected in MultiAssetCTM after fused cross-attention")
                z = torch.nan_to_num(z, nan=0.0)
        else:
            z = self.single_asset_model.encode(x_flat, cond=cond_proj)
            # Reshape back to (B, N, T, model_dim)
            z = z.reshape(B, N, T, self.model_dim)

            # ── Cross-asset mixing (post-loop) ──
            if self.cross_attn is not None and self.cross_norm is not None:
                z_normed = self.cross_norm(z)  # (B, N, T, D)
                z_mixed = self.cross_attn(z_normed)
                z = z + self.dropout(z_mixed)
                if self.cross_ff is not None:
                    z = z + self.dropout(self.cross_ff(z))
                    if torch.isnan(z).any():
                        self._nan_counter += 1
                        warnings.warn("NaN detected in MultiAssetCTM after cross-ff")
                        z = torch.nan_to_num(z, nan=0.0)

        # ── Output heads (per-asset regression + classification) ──
        if torch.isnan(z).any():
            self._nan_counter += 1
            warnings.warn("NaN detected in MultiAssetCTM before output heads")
            z = torch.nan_to_num(z, nan=0.0)
        out_reg = self.head_regression(z)       # (B, N, T, output_dim)
        out_cls = self.head_classification(z)    # (B, N, T, 3)

        # ── Time-decay gate blending (Variant A) ──
        if self.time_gate is not None and gbdt_preds is not None:
            # Normalize gbdt_preds to (B, N, T): handle both (B,N,T) and (B*T,N)
            if gbdt_preds.dim() == 2:
                gbdt_3d = gbdt_preds.reshape(B, T, N).transpose(1, 2)  # (B*T,N) → (B,N,T)
            elif gbdt_preds.shape[1] == N and gbdt_preds.shape[2] == T:
                gbdt_3d = gbdt_preds  # already (B, N, T)
            elif gbdt_preds.shape[1] == T and gbdt_preds.shape[2] == N:
                gbdt_3d = gbdt_preds.permute(0, 2, 1)  # (B, T, N) → (B, N, T)
            else:
                gbdt_3d = gbdt_preds.reshape(B, N, T)  # fallback reshape
            # Detach GBDT preds so gradients flow only through gate params (α,β,γ)
            # and CTM predictions, NOT through GBDT weights.
            out_reg, self._last_gate_weights = self.time_gate(out_reg, gbdt_3d.detach())

        # Stack regression and class logits separately, then concatenate along channel dim.
        # This gives a flat layout: [reg_all_assets..., cls_all_assets...]
        # matching the expectation of composite_loss(num_regression=N*output_dim).
        # out_reg: (B, N, T, output_dim) → (B, T, N * output_dim)
        out_reg_b = out_reg.permute(0, 2, 1, 3).reshape(B, T, N * self.output_dim)
        # out_cls: (B, N, T, 3) → (B, T, N * 3)
        out_cls_b = out_cls.permute(0, 2, 1, 3).reshape(B, T, N * 3)
        return torch.cat([out_reg_b, out_cls_b], dim=-1)  # (B, T, N*(output_dim+3))

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def extract_features(self, x: torch.Tensor, chunk_size: int = 4096) -> torch.Tensor:
        """Extract hidden features without output heads.

        Processes in chunks along the batch dimension to avoid OOM
        when flattening many assets (e.g. 400 windows × 200 stocks = 80 000).

        Returns encoder output: ``(B*N, T, model_dim)`` for 4-D input
        ``(B, N, T, D)``, or ``(B, T, model_dim)`` for 3-D input
        ``(B, T, D)`` (delegated to ``single_asset_model`` directly).
        """
        if x.dim() == 3:
            return self.single_asset_model.extract_features(x)
        B, N, T, D_in = x.shape
        device = x.device
        asset_ids = torch.arange(N, device=device).expand(B, -1)
        emb = self.asset_embed(asset_ids)
        if emb.dim() == 2:
            emb = emb.unsqueeze(0).expand(B, -1, -1)
        x_flat = x.reshape(B * N, T, D_in)
        emb_flat = emb.reshape(B * N, -1)
        cond_embed = emb_flat.unsqueeze(1).expand(-1, T, -1)
        cond_proj = self.cond_proj(cond_embed)

        total = B * N
        if total <= chunk_size:
            z = self.single_asset_model.encode(x_flat, cond=cond_proj)
        else:
            chunks = []
            for start in range(0, total, chunk_size):
                end = min(start + chunk_size, total)
                chunk_z = self.single_asset_model.encode(
                    x_flat[start:end], cond=cond_proj[start:end]
                )
                chunks.append(chunk_z)
            z = torch.cat(chunks, dim=0)

        return z.reshape(B * N, T, self.model_dim)
