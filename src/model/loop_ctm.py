"""RecurrentCTM: CTMStockModel with iterative refinement via residual loops.

Wraps ``CTMStockModel._encode_core`` in a fixed-iteration loop with
LayerNorm residual connections, letting the encoder iteratively
refine its hidden representation before the output heads are applied.

Architecture::

    x_raw → _encode_core (full: input_proj→conv→Mamba×N→[bidirectional]) → x
                                                                           ↓
    x ← Dropout(LayerNorm(h + x))        ←── _encode_loop(x)               │
                      ↓                         (conv→Mamba×N→[bidir],     │
    x ← h              (final iter)             no input_proj)              │
                      ↓                                                    │
    head_regression(x) ‖ head_classification(x)

References:
    - Gu & Dao (2023): Mamba: Linear-Time Sequence Modeling
    - Lan et al. (2020): ALBERT — sharing parameters across layers
"""

from __future__ import annotations

import torch
import torch.nn as nn
import warnings

from .ctm_model import CTMStockModel


class RecurrentCTM(CTMStockModel):
    """Recurrent CTM model with iterative encoder refinement.

    Loops ``_encode_core`` for ``n_loop_iters`` iterations, applying
    a LayerNorm residual connection after each intermediate iteration.
    The first pass uses the full encoder (including ``input_proj``);
    subsequent passes reuse the conv→Mamba→bidirectional stack on
    the model_dim representation directly.

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
        Number of MambaBlock layers inside each encoder pass.
    output_dim : int, default=1
        Number of regression outputs (1 for single-stock, >1 for multi-stock).
    dropout : float, default=0.1
        Dropout rate between Mamba layers.
    use_decomp : bool, default=False
        Enable seasonal-trend decomposition.
    bidirectional : bool, default=False
        Enable bidirectional Mamba processing.
    return_hidden : bool, default=False
        Default hidden-state return behaviour (see ``forward``).
    parallel_scan : bool, default=False
        Use parallel scan Mamba implementation.
    n_loop_iters : int, default=3
        Number of encoder loop iterations (≥1).
    loop_dropout : float, default=0.1
        Dropout rate applied after each loop residual connection.
    in_loop_fusion : bool, default=False
        When True, ``encode()`` and ``forward()`` accept a ``cross_attn``
        module and apply cross-asset attention after each ``_encode_blocks``
        call, before the residual connection.
    n_assets : int, default=1
        Number of assets. Required when ``in_loop_fusion=True`` to perform
        the (B*N, T, D) ↔ (B, N, T, D) reshape for cross-asset attention.
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
        n_loop_iters: int = 3,
        loop_dropout: float = 0.1,
        in_loop_fusion: bool = False,
        n_assets: int = 1,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            model_dim=model_dim,
            state_dim=state_dim,
            conv_kernel=conv_kernel,
            n_layers=n_layers,
            output_dim=output_dim,
            dropout=dropout,
            use_decomp=use_decomp,
            bidirectional=bidirectional,
            return_hidden=return_hidden,
            parallel_scan=parallel_scan,
        )

        if n_loop_iters < 1:
            raise ValueError(f"n_loop_iters must be >= 1, got {n_loop_iters}")
        self.n_loop_iters = n_loop_iters
        self.loop_norm = nn.LayerNorm(model_dim)
        self.loop_dropout = nn.Dropout(loop_dropout)
        self.in_loop_fusion = in_loop_fusion
        self.n_assets = n_assets
        self._nan_counter = 0
        self._cross_attn: nn.Module | None = None

    def _encode_loop(self, x: torch.Tensor) -> torch.Tensor:
        """Single encoder pass without input projection (x is already in model_dim space).

        Delegates to the shared ``CTMStockModel._encode_blocks``, reusing the
        same base-class submodules so weights are shared across loop iterations.
        """
        return self._encode_blocks(x)

    def set_cross_attn(self, attn_module: nn.Module) -> None:
        """Inject a cross-asset attention module for in-loop fusion.

        Parameters
        ----------
        attn_module : nn.Module
            A callable module with signature ``(B*T, N, D) → (B*T, N, D)``
            (e.g., ``FusedMultiHeadCrossAttention``). Called after each
            ``_encode_blocks`` pass when ``in_loop_fusion=True``.
        """
        self._cross_attn = attn_module

    def encode(self, x: torch.Tensor, cond: torch.Tensor | None = None, cross_attn: nn.Module | None = None) -> torch.Tensor:
        """Encode input with iterative refinement.

        Overrides base class single-pass encode to run the full loop.
        Returns the final hidden state (before output heads).

        Parameters
        ----------
        x : (B, T, input_dim) or (B*N, T, input_dim) input features.
        cond : (B, T, model_dim) or (B*N, T, model_dim) or None — conditioning signal.
        cross_attn : nn.Module or None
            Injected cross-asset attention module (e.g.,
            ``FusedMultiHeadCrossAttention``). When provided and
            ``in_loop_fusion=True``, applied after each ``_encode_blocks``
            call. Expects input/output shape ``(B*T, N, D)``.

        Returns
        -------
        (B, T, model_dim) or (B*N, T, model_dim) hidden state after loop refinement.
        """
        x = self.input_proj(x)
        if cond is not None:
            x = x + cond
        for i in range(self.n_loop_iters):
            h = self._encode_blocks(x)
            if torch.isnan(h).any():
                self._nan_counter += 1
                warnings.warn(f"NaN detected in RecurrentCTM after encode_blocks (iter {i})")
                h = torch.nan_to_num(h, nan=0.0)
            if cross_attn is not None and self.in_loop_fusion:
                B_N, T, D = h.shape
                N = self.n_assets
                assert B_N % N == 0, f"Batch-asset dimension mismatch: B_N={B_N} not divisible by N={N}"
                B = B_N // N
                h_4d = h.reshape(B, N, T, D).transpose(1, 2)
                h_attn = cross_attn(h_4d.reshape(B * T, N, D))
                h = h_attn.reshape(B, T, N, D).transpose(1, 2).reshape(B_N, T, D)
                if torch.isnan(h).any():
                    self._nan_counter += 1
                    warnings.warn(f"NaN detected in RecurrentCTM after cross-attn (iter {i})")
                    h = torch.nan_to_num(h, nan=0.0)
            if i < self.n_loop_iters - 1:
                x = self.loop_dropout(self.loop_norm(h + x))
                if torch.isnan(x).any():
                    self._nan_counter += 1
                    warnings.warn(f"NaN detected in RecurrentCTM after loop residual (iter {i})")
                    x = torch.nan_to_num(x, nan=0.0)
            else:
                x = h
        return x

    def forward(
        self,
        x: torch.Tensor,
        return_hidden: bool | None = None,
        cross_attn: nn.Module | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with iterative encoder refinement.

        Parameters
        ----------
        x : (B, T, input_dim) or (B*N, T, input_dim) input features.
        return_hidden : bool | None
            If True, return ``(output, hidden_state)``.
            If False, return only ``output``.
            If None (default), use ``self.return_hidden``.
        cross_attn : nn.Module or None
            Cross-asset attention module for in-loop fusion.
            When provided and ``in_loop_fusion=True``, applied after each
            ``_encode_blocks`` call with the reshape round-trip:
            ``(B*N, T, D) → (B, N, T, D) → attn → (B, N, T, D) → (B*N, T, D)``.

        Returns
        -------
        torch.Tensor | tuple[torch.Tensor, torch.Tensor]
            - If ``return_hidden`` resolves to False: ``(B, T, num_output_channels)`` predictions.
              Last dim layout: [0:output_dim]=regression, [output_dim:output_dim+3]=class logits.
            - If True: ``(output, hidden_state)`` where hidden_state is (B, T, model_dim).
        """
        # Iteration 0: full encoder pass (input_proj → conv → Mamba → bidirectional)
        x = self._encode_core(x)
        if torch.isnan(x).any():
            self._nan_counter += 1
            warnings.warn("NaN detected in RecurrentCTM after initial encode (iter 0)")
            x = torch.nan_to_num(x, nan=0.0)

        # Iterations 1 .. n_loop_iters-1: loop encode (no input_proj)
        n_total_iters = self.n_loop_iters
        for i in range(1, n_total_iters):
            h = self._encode_blocks(x)
            if torch.isnan(h).any():
                self._nan_counter += 1
                warnings.warn(f"NaN detected in RecurrentCTM after _encode_loop (iter {i})")
                h = torch.nan_to_num(h, nan=0.0)
            if cross_attn is not None and self.in_loop_fusion:
                B_N, T, D = h.shape
                N = self.n_assets
                assert B_N % N == 0, f"Batch-asset dimension mismatch: B_N={B_N} not divisible by N={N}"
                B = B_N // N
                h_4d = h.reshape(B, N, T, D).transpose(1, 2)
                h_attn = cross_attn(h_4d.reshape(B * T, N, D))
                h = h_attn.reshape(B, T, N, D).transpose(1, 2).reshape(B_N, T, D)
            if i < n_total_iters - 1:
                # Progressive dropout: anneal from loop_dropout → loop_dropout/3 across iterations
                # Higher dropout early (exploration), lower later (refinement)
                decay = 1.0 - 0.7 * (i - 1) / max(1, n_total_iters - 2)
                current_p = self.loop_dropout.p * decay
                x = torch.nn.functional.dropout(self.loop_norm(h + x), p=current_p, training=self.training)
                if torch.isnan(x).any():
                    self._nan_counter += 1
                    warnings.warn(f"NaN detected in RecurrentCTM after loop residual (iter {i})")
                    x = torch.nan_to_num(x, nan=0.0)
            else:
                x = h

        out_reg = self.head_regression(x) if self.head_regression is not None else x.new_zeros(x.shape[0], x.shape[1], 0)
        out_cls = self.head_classification(x) if self.head_classification is not None else x.new_zeros(x.shape[0], x.shape[1], 0)
        output = torch.cat([out_reg, out_cls], dim=-1) if out_reg.shape[-1] > 0 or out_cls.shape[-1] > 0 else out_reg
        if torch.isnan(output).any():
            self._nan_counter += 1
            warnings.warn("NaN detected in RecurrentCTM output heads")
            output = torch.nan_to_num(output, nan=0.0)

        _return_hidden = return_hidden if return_hidden is not None else self.return_hidden
        if _return_hidden:
            return output, x
        return output

    def param_count(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def extra_repr(self) -> str:
        return (
            f"input_dim={self.input_dim}, model_dim={self.model_dim}, "
            f"state_dim={self.state_dim}, n_layers={self.n_layers}, "
            f"output_dim={self.output_dim}, bidirectional={self.bidirectional}, "
            f"return_hidden={self.return_hidden}, parallel_scan={self.parallel_scan}, "
            f"n_loop_iters={self.n_loop_iters}, loop_dropout={self.loop_dropout.p}, "
            f"in_loop_fusion={self.in_loop_fusion}, n_assets={self.n_assets}"
        )
