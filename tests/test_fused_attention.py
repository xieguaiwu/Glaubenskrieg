"""Tests for FusedMultiHeadCrossAttention and GBDTModulator (fused_attention module)."""

import torch
import pytest

from src.model.fused_attention import FusedMultiHeadCrossAttention, GBDTModulator


# ═══════════════════════════════════════════════════════════════════
# GBDTModulator tests
# ═══════════════════════════════════════════════════════════════════


class TestGBDTModulator:
    """Tests for GBDTModulator — pairwise prediction-difference bias MLP."""

    def test_forward_shape_2d(self):
        """(batch, N) input → (batch, N, N) output."""
        modulator = GBDTModulator(hidden_dim=8)
        preds = torch.randn(3, 5)  # batch=3, assets=5
        bias = modulator(preds)
        assert bias.shape == (3, 5, 5), f"Expected (3,5,5), got {bias.shape}"

    def test_forward_shape_1d(self):
        """(N,) input → (N, N) output."""
        modulator = GBDTModulator(hidden_dim=8)
        preds = torch.randn(5)
        bias = modulator(preds)
        assert bias.shape == (5, 5), f"Expected (5,5), got {bias.shape}"

    def test_forward_zero_input(self):
        """All zeros input → all pairs have same diff (0), so output is uniform."""
        modulator = GBDTModulator(hidden_dim=8)
        modulator.eval()  # disable dropout for deterministic output
        preds = torch.zeros(5)
        bias = modulator(preds)
        # All pairwise diffs are zero → same MLP input → same output
        assert torch.allclose(bias, bias[0, 0].expand_as(bias), atol=1e-5), (
            "Zero-diff input should produce uniform bias matrix"
        )
        assert torch.isfinite(bias).all()

    def test_forward_identity(self):
        """Ones input → output has correct shape and finite values."""
        modulator = GBDTModulator(hidden_dim=8)
        preds = torch.ones(5)
        bias = modulator(preds)
        assert bias.shape == (5, 5)
        assert torch.isfinite(bias).all(), "Output should be finite"

    def test_extra_repr(self):
        """String representation includes hidden_dim and dropout."""
        modulator = GBDTModulator(hidden_dim=32, dropout=0.2)
        rep = modulator.extra_repr()
        assert "hidden_dim=32" in rep
        assert "dropout=0.2" in rep


# ═══════════════════════════════════════════════════════════════════
# FusedMultiHeadCrossAttention tests
# ═══════════════════════════════════════════════════════════════════


class TestFusedMultiHeadCrossAttention:
    """Tests for FusedMultiHeadCrossAttention — multi-head cross-asset attention."""

    def test_forward_shape(self):
        """(B, N, T, d_model) → (B, N, T, d_model) shape preserved."""
        attn = FusedMultiHeadCrossAttention(n_assets=5, d_model=16, n_heads=4)
        x = torch.randn(2, 5, 10, 16)  # B=2, N=5, T=10, D=16
        out = attn(x)
        assert out.shape == (2, 5, 10, 16), f"Expected (2,5,10,16), got {out.shape}"
        assert torch.isfinite(out).all()

    def test_multi_head_split(self):
        """d_model=16, n_heads=4 → each head gets d_k=4."""
        attn = FusedMultiHeadCrossAttention(n_assets=3, d_model=16, n_heads=4)
        assert attn.d_k == 4, f"Expected d_k=4, got {attn.d_k}"
        assert attn.n_heads == 4

    def test_gbdt_bias_injection(self):
        """set_gbdt_predictions + forward produces different output than without."""
        attn = FusedMultiHeadCrossAttention(n_assets=3, d_model=16, n_heads=4)
        x = torch.randn(1, 3, 4, 16)  # B=1, N=3, T=4, D=16

        # Forward without GBDT bias
        out_no_bias = attn(x).clone()

        # Forward with GBDT bias injected
        gbdt_preds = torch.tensor([0.1, -0.2, 0.05])
        attn.set_gbdt_predictions(gbdt_preds)
        out_with_bias = attn(x)

        assert not torch.allclose(out_no_bias, out_with_bias, atol=1e-5), (
            "Output should differ when GBDT bias is injected"
        )

    def test_different_n_heads(self):
        """Test with n_heads=1 and n_heads=4 — both produce valid output."""
        for n_heads in [1, 4]:
            attn = FusedMultiHeadCrossAttention(
                n_assets=3, d_model=16, n_heads=n_heads,
            )
            x = torch.randn(1, 3, 4, 16)
            out = attn(x)
            assert out.shape == (1, 3, 4, 16)
            assert torch.isfinite(out).all(), f"Output non-finite for n_heads={n_heads}"

    def test_no_gbdt_bias(self):
        """Forward without set_gbdt_predictions still works (no bias injected)."""
        attn = FusedMultiHeadCrossAttention(
            n_assets=3, d_model=16, n_heads=4, use_gbdt_bias=True,
        )
        x = torch.randn(1, 3, 4, 16)
        out = attn(x)  # No set_gbdt_predictions called — _gbdt_preds is None
        assert out.shape == (1, 3, 4, 16)
        assert torch.isfinite(out).all()

    def test_no_gbdt_bias_disabled(self):
        """use_gbdt_bias=False — modulator is None, forward works."""
        attn = FusedMultiHeadCrossAttention(
            n_assets=3, d_model=16, n_heads=4, use_gbdt_bias=False,
        )
        assert attn.gbdt_modulator is None
        x = torch.randn(1, 3, 4, 16)
        out = attn(x)
        assert out.shape == (1, 3, 4, 16)
        assert torch.isfinite(out).all()

    def test_invalid_d_model_raises(self):
        """d_model not divisible by n_heads → ValueError."""
        with pytest.raises(ValueError, match="divisible"):
            FusedMultiHeadCrossAttention(n_assets=3, d_model=15, n_heads=4)

    def test_backward(self):
        """Gradient flows through attention parameters."""
        attn = FusedMultiHeadCrossAttention(n_assets=3, d_model=16, n_heads=4)
        x = torch.randn(1, 3, 4, 16)

        # Inject GBDT bias to exercise the modulator path
        gbdt_preds = torch.tensor([0.1, -0.2, 0.05])
        attn.set_gbdt_predictions(gbdt_preds)

        out = attn(x)
        loss = out.sum()
        loss.backward()

        # At least one parameter should receive non-zero gradient
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in attn.parameters()
        )
        assert has_grad, "No parameter received gradients"

    def test_param_count(self):
        """Returns positive parameter count (QKV projections + output + adj_bias + modulator)."""
        attn = FusedMultiHeadCrossAttention(
            n_assets=3, d_model=16, n_heads=4, use_gbdt_bias=True,
        )
        n = attn.param_count()
        assert n > 0, f"Expected positive param count, got {n}"
        # At minimum: Q, K, V, out projections (4 × 16 × 16 = 1024)
        assert n > 500, f"Expected >500 params (projections + adj_bias + modulator), got {n}"

    def test_extra_repr(self):
        """String representation includes key attributes."""
        attn = FusedMultiHeadCrossAttention(
            n_assets=5, d_model=32, n_heads=8, use_gbdt_bias=False,
        )
        rep = attn.extra_repr()
        assert "n_assets=5" in rep
        assert "d_model=32" in rep
        assert "n_heads=8" in rep
        assert "use_gbdt_bias=False" in rep
