"""CTM: Conv-Temporal-Mamba model for stock prediction.

Full architecture: CausalConv → [SeasonalTrendDecomp] → MambaBlock×N →
[Bi-Mamba backward] → Multi-Task Output Heads.

References:
  - Gu & Dao (2023): Mamba: Linear-Time Sequence Modeling
  - Shi (2024): MambaStock: Selective SSM for Stock Prediction
  - Chen & Sun (2026): DMamba: Decomposition-enhanced Mamba for Time Series
"""

from __future__ import annotations

import torch
import torch.nn as nn
import warnings

import torch.nn.functional as F

from .mamba_block import MambaBlock
from .mamba_parallel import MambaBlockParallel


# ═══════════════════════════════════════════════════════════════════════════════
# Sub-modules
# ═══════════════════════════════════════════════════════════════════════════════


class CausalConv1d(nn.Module):
    """Depthwise causal 1D convolution for local temporal feature extraction.

    Parameters
    ----------
    channels : int
        Number of input/output channels (= d_model).
    kernel_size : int, default=3
        Convolution kernel size. Uses left-padding only for causality.
    """
    def __init__(self, channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.weight = nn.Parameter(torch.empty(channels, kernel_size))
        self.bias = nn.Parameter(torch.zeros(channels))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply causal depthwise conv1d.

        Parameters
        ----------
        x : (B, T, C) input.

        Returns
        -------
        (B, T, C) output.
        """
        x_t = x.transpose(1, 2)  # (B, C, T)
        x_t = F.pad(x_t, (self.kernel_size - 1, 0))
        w = self.weight.unsqueeze(1)  # (C, 1, K)
        y_t = F.conv1d(x_t, w, self.bias, groups=self.channels)
        return y_t.transpose(1, 2)


class SeasonalTrendDecomp(nn.Module):
    """Seasonal-trend decomposition via moving average, per DMamba (2026).

    x_t = s_t + t_t, where t_t is the MA trend, s_t is the residual (seasonal).
    """
    def __init__(self, period: int = 5) -> None:
        super().__init__()
        self.period = period
        kernel: torch.Tensor = torch.ones(1, 1, period) / period
        self.register_buffer("kernel", kernel)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Decompose input into seasonal and trend components.

        Parameters
        ----------
        x : (B, T, C) input.

        Returns
        -------
        seasonal : (B, T, C) high-frequency residual.
        trend : (B, T, C) low-frequency moving average.
        """
        B, T, C = x.shape
        # Run depthwise conv1d per channel for trend
        x_t = x.reshape(B * C, 1, T)
        padded = F.pad(x_t, (self.period - 1, 0))
        trend = F.conv1d(padded, self.get_buffer("kernel"), groups=1).reshape(B, C, T).transpose(1, 2)
        seasonal = x - trend
        return seasonal, trend


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Zhang & Sennrich, 2019)."""
    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight


# ═══════════════════════════════════════════════════════════════════════════════
# CTM Model
# ═══════════════════════════════════════════════════════════════════════════════


class CTMStockModel(nn.Module):
    """Conv-Temporal-Mamba stock prediction model.

    Architecture (simplified):
        Input(B,T,d_x) → Linear(d_x→d_model) → CausalConv → [Decomp] →
        MambaBlock×N → [Bi-Mamba backward] → Output Heads

    Output heads:
      - Regression: returns prediction (1 or output_dim)
      - Classification (optional): 3-class direction logits (UP/NEUTRAL/DOWN)

    Parameters
    ----------
    input_dim : int
        Number of input features.
    model_dim : int, default=64
        Hidden dimension d_model.
    state_dim : int, default=16
        SSM state dimension.
    conv_kernel : int, default=3
        Causal conv kernel size.
    n_layers : int, default=3
        Number of MambaBlock layers.
    output_dim : int, default=1
        Number of regression outputs (1 for single-stock, >1 for multi-stock).
    dropout : float, default=0.1
        Dropout rate between layers.
    use_decomp : bool, default=False
        Enable seasonal-trend decomposition.
    bidirectional : bool, default=False
        Enable bidirectional Mamba processing.
    """
    def __init__(
        self,
        input_dim: int,
        model_dim: int = 64,
        state_dim: int = 16,
        conv_kernel: int = 3,
        n_layers: int = 3,
        output_dim: int = 1,
        dropout: float = 0.1,
        use_decomp: bool = False,
        bidirectional: bool = False,
        return_hidden: bool = False,
        parallel_scan: bool = False,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.parallel_scan = parallel_scan
        self.model_dim = model_dim
        self.state_dim = state_dim
        self.n_layers = n_layers
        self.output_dim = output_dim
        self.use_decomp = use_decomp
        self.bidirectional = bidirectional
        self.return_hidden = return_hidden
        self._nan_counter = 0
        expand = 2  # MambaBlock default expansion factor
        d_inner = expand * model_dim

        # Input projection
        self.input_proj = nn.Linear(input_dim, model_dim)

        # Causal conv frontend
        self.conv = CausalConv1d(model_dim, conv_kernel)

        # Optional decomposition
        self.decomp = SeasonalTrendDecomp(period=5) if use_decomp else None

        # Mamba blocks with residual connections and output projections
        self.norms = nn.ModuleList([RMSNorm(model_dim) for _ in range(n_layers)])
        block_cls = MambaBlockParallel if parallel_scan else MambaBlock
        self.mamba_blocks = nn.ModuleList([
            block_cls(d_model=model_dim, d_state=state_dim, d_conv=conv_kernel, expand=expand)
            for _ in range(n_layers)
        ])
        self.mamba_out_projs = nn.ModuleList([
            nn.Linear(d_inner, model_dim) for _ in range(n_layers)
        ])

        # Optional backward Mamba stack for bidirectional processing
        if bidirectional:
            self.backward_mamba = block_cls(
                d_model=model_dim, d_state=state_dim, d_conv=conv_kernel, expand=expand
            )
            self.backward_norm = RMSNorm(model_dim)
            self.backward_out_proj = nn.Linear(d_inner, model_dim)
            self.backward_fuse = nn.Linear(model_dim * 2, model_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Output heads (skipped when output_dim <= 0, e.g. MultiAssetCTM encode-only)
        if output_dim > 0:
            self.head_regression = nn.Linear(model_dim, output_dim)
            self.head_classification = nn.Linear(model_dim, 3)
        else:
            self.head_regression = None
            self.head_classification = None

        # Track output channels for composite_loss slicing
        self.num_output_channels = output_dim + 3

    def _encode_blocks(self, x: torch.Tensor) -> torch.Tensor:
        """Shared encoder blocks: conv → [decomp] → Mamba stack → [bidirectional].

        Assumes x is already in model_dim space (after input_proj).
        Reused by both ``_encode_core`` and ``RecurrentCTM._encode_loop``.

        Parameters
        ----------
        x : (B, T, model_dim) hidden features.

        Returns
        -------
        (B, T, model_dim) encoded features.
        """
        x = self.conv(x)
        assert not torch.isnan(x).any(), \
            f"NaN detected in CTMStockModel after conv — shape={x.shape}"
        if self.decomp is not None:
            seasonal, _ = self.decomp(x)
            x = seasonal
        for block, out_proj, norm in zip(self.mamba_blocks, self.mamba_out_projs, self.norms):
            residual = x
            x_norm = norm(x)
            x_mamba, _ = block(x_norm)
            x = residual + self.dropout(out_proj(x_mamba))
            assert not torch.isnan(x).any(), \
                f"NaN detected in CTMStockModel after Mamba block {i} — shape={x.shape}"
        if self.bidirectional:
            x_rev = torch.flip(x, dims=[1])
            x_rev_norm = self.backward_norm(x_rev)
            x_rev_mamba, _ = self.backward_mamba(x_rev_norm)
            x_rev = torch.flip(self.backward_out_proj(x_rev_mamba), dims=[1])
            x = self.backward_fuse(torch.cat([x, x_rev], dim=-1))
        return x

    def _encode_core(self, x: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        """Shared encoding: input_proj → (cond add) → _encode_blocks.

        Parameters
        ----------
        x : (B, T, input_dim) or (B*N, T, input_dim) input.
        cond : (B*N, T, model_dim) or None — conditioning signal to add after input_proj.

        Returns
        -------
        (B, T, model_dim) encoder output (before output heads).
        """
        x = self.input_proj(x)
        if cond is not None:
            x = x + cond
        x = self._encode_blocks(x)
        return x

    def forward(self, x: torch.Tensor, return_hidden: bool | None = None) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        x : (B, T, input_dim) input features.
        return_hidden : bool | None
            If True, return ``(output, hidden_state)``.
            If False, return only ``output``.
            If None (default), use ``self.return_hidden``.

        Returns
        -------
        torch.Tensor | tuple[torch.Tensor, torch.Tensor]
            - If ``return_hidden`` resolves to False: ``(B, T, num_output_channels)`` predictions.
              Last dim layout: [0:output_dim]=regression, [output_dim:output_dim+3]=class logits.
            - If True: ``(output, hidden_state)`` where hidden_state is (B, T, model_dim).
        """
        hidden_state = self._encode_core(x)
        out_reg = self.head_regression(hidden_state) if self.head_regression is not None else hidden_state.new_zeros(hidden_state.shape[0], hidden_state.shape[1], 0)
        out_cls = self.head_classification(hidden_state) if self.head_classification is not None else hidden_state.new_zeros(hidden_state.shape[0], hidden_state.shape[1], 0)
        output = torch.cat([out_reg, out_cls], dim=-1)
        # Last-resort safety net: only place nan_to_num is acceptable.
        # All upstream NaN sources must be caught by assertions in conv / Mamba blocks.
        if torch.isnan(output).any():
            self._nan_counter += 1
            warnings.warn("NaN detected in CTMStockModel output heads (last resort)")
            output = torch.nan_to_num(output, nan=0.0)
        _return_hidden = return_hidden if return_hidden is not None else self.return_hidden
        if _return_hidden:
            return output, hidden_state
        return output

    def param_count(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def encode(self, x: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        """Encode input to hidden features, optionally with conditioning.

        Delegates to ``_encode_core``.  Provided as a public interface
        for :class:`MultiAssetCTM` and similar composite models.

        Parameters
        ----------
        x : (B, T, input_dim) or (B*N, T, input_dim) input
        cond : (B*N, T, model_dim) or None — conditioning signal to add after input_proj

        Returns
        -------
        (B, T, model_dim) encoder output (before output heads)
        """
        return self._encode_core(x, cond=cond)

    def extra_repr(self) -> str:
        return (
            f"input_dim={self.input_dim}, model_dim={self.model_dim}, "
            f"state_dim={self.state_dim}, n_layers={self.n_layers}, "
            f"output_dim={self.output_dim}, bidirectional={self.bidirectional}, "
            f"return_hidden={self.return_hidden}, parallel_scan={self.parallel_scan}"
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract hidden features without output heads.
        Returns (B, T, model_dim) encoder output.
        """
        return self._encode_core(x)
