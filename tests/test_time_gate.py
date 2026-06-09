"""Tests for TimeDecayGate — horizon-dependent CTM↔GBDT blending module."""

import torch
import pytest

from src.model.time_gate import TimeDecayGate


class TestTimeDecayGate:
    """Tests for TimeDecayGate — exponential horizon-based NN→GBDT blending."""

    # ── Helpers ──
    @staticmethod
    def _make_dummy_inputs(B=2, N=5, T=20, C=1):
        """Create synthetic (B, N, T, C) CTM preds and (B, N, T) GBDT preds."""
        ctm_pred = torch.randn(B, N, T, C)
        gbdt_pred = torch.randn(B, N, T)
        return ctm_pred, gbdt_pred

    # ── Tests ──

    def test_gate_shape(self):
        """Blended output has the same shape as ctm_pred."""
        gate = TimeDecayGate()
        ctm, gbd = self._make_dummy_inputs(B=3, N=4, T=10, C=2)
        blended, weights = gate(ctm, gbd)
        assert blended.shape == ctm.shape, (
            f"Expected blended shape {ctm.shape}, got {blended.shape}"
        )
        assert weights.shape == (10,), f"Expected weights shape (10,), got {weights.shape}"

    def test_gate_in_range(self):
        """Weights are strictly in [0, 1] (sigmoid output)."""
        gate = TimeDecayGate()
        ctm, gbd = self._make_dummy_inputs(T=15)
        _, weights = gate(ctm, gbd)
        assert (weights >= 0.0).all(), f"Weight below 0: min={weights.min().item():.4f}"
        assert (weights <= 1.0).all(), f"Weight above 1: max={weights.max().item():.4f}"

    def test_gate_monotonic(self):
        """Weights decrease monotonically with t for alpha>0, beta>0."""
        gate = TimeDecayGate(alpha_init=2.0, beta_init=4.0)
        ctm, gbd = self._make_dummy_inputs(T=20)
        _, weights = gate(ctm, gbd)
        # Weights should be non-increasing: w[0] >= w[1] >= ... >= w[T-1]
        diffs = weights[:-1] - weights[1:]
        assert (diffs >= 0).all(), (
            f"Weights not monotonic decreasing: diffs={diffs.detach().numpy()}"
        )

    def test_gate_monotonic_negative_alpha(self):
        """With alpha<0 and beta>0, weights should increase with t."""
        gate = TimeDecayGate(alpha_init=-2.0, beta_init=4.0)
        ctm, gbd = self._make_dummy_inputs(T=20)
        _, weights = gate(ctm, gbd)
        diffs = weights[:-1] - weights[1:]
        # alpha negative → exp term is neg → sigmoid of decreasing→more neg → decreasing
        # Wait: α<0, exp(-β·τ)>0, so α·exp(...) < 0 and decreasing toward 0
        # So sigmoid of a negative decreasing→0 → increasing
        assert (diffs <= 0).all(), (
            f"Weights not monotonic increasing with negative alpha: "
            f"diffs={diffs.detach().numpy()}"
        )

    def test_gate_gradient(self):
        """Alpha, beta, gamma all receive non-zero gradients after loss.backward()."""
        gate = TimeDecayGate()
        ctm, gbd = self._make_dummy_inputs(B=2, N=3, T=10, C=1)
        blended, weights = gate(ctm, gbd)
        loss = blended.sum() + weights.sum()
        loss.backward()

        for name in ("alpha", "beta", "gamma"):
            param = getattr(gate, name)
            assert param.grad is not None, f"{name}.grad is None"
            assert param.grad.abs().sum() > 0, f"{name}.grad is zero"

    def test_gate_fallback(self):
        """Forward works with default parameters (alpha=2, beta=4, gamma=0)."""
        gate = TimeDecayGate()
        ctm, gbd = self._make_dummy_inputs(B=1, N=2, T=5, C=1)
        blended, weights = gate(ctm, gbd)
        assert torch.isfinite(blended).all(), "Blended output contains non-finite values"
        assert torch.isfinite(weights).all(), "Weights contain non-finite values"

    def test_gate_fallback_different_shapes(self):
        """Forward works with varied batch, asset, time, and channel sizes."""
        gate = TimeDecayGate()
        for B, N, T, C in [(1, 1, 2, 1), (3, 5, 4, 1), (2, 3, 8, 2)]:
            ctm = torch.randn(B, N, T, C)
            gbd = torch.randn(B, N, T)
            blended, weights = gate(ctm, gbd)
            assert blended.shape == (B, N, T, C)
            assert weights.shape == (T,)

    def test_gate_serialization(self):
        """State dict roundtrip: save → load → identical output."""
        gate1 = TimeDecayGate(alpha_init=1.5, beta_init=3.0, gamma_init=-0.2)
        ctm, gbd = self._make_dummy_inputs(B=1, N=2, T=6, C=1)

        # Record originals for comparison
        alpha1 = gate1.alpha.item()
        beta1 = gate1.beta.item()
        gamma1 = gate1.gamma.item()

        with torch.no_grad():
            blended1, weights1 = gate1(ctm, gbd)

        # Roundtrip: save → new instance → load
        state = gate1.state_dict()
        gate2 = TimeDecayGate()  # different default init
        gate2.load_state_dict(state)

        # Params are restored
        assert gate2.alpha.item() == alpha1, "alpha not restored"
        assert gate2.beta.item() == beta1, "beta not restored"
        assert gate2.gamma.item() == gamma1, "gamma not restored"

        # Output is identical
        with torch.no_grad():
            blended2, weights2 = gate2(ctm, gbd)
        assert torch.allclose(blended1, blended2, atol=1e-6), "Blended output differs"
        assert torch.allclose(weights1, weights2, atol=1e-6), "Weights differ"

    def test_gate_extra_repr(self):
        """String representation includes alpha, beta, gamma values."""
        gate = TimeDecayGate(alpha_init=2.5, beta_init=3.0, gamma_init=0.1)
        rep = gate.extra_repr()
        assert "alpha=" in rep
        assert "beta=" in rep
        assert "gamma=" in rep
        assert "2.5" in rep

    def test_gate_edge_case_t1(self):
        """Single time step (T=1) works without division by zero."""
        gate = TimeDecayGate()
        ctm = torch.randn(1, 1, 1, 1)  # B=1, N=1, T=1, C=1
        gbd = torch.randn(1, 1, 1)
        blended, weights = gate(ctm, gbd)
        assert blended.shape == (1, 1, 1, 1)
        assert weights.shape == (1,)
        assert torch.isfinite(blended).all()
