"""Tests for RecurrentCTM — CTMStockModel with looped encoding and Pre-LN residual."""

import torch
import pytest

from src.model.ctm_model import CTMStockModel
from src.model.loop_ctm import RecurrentCTM


def test_recurrent_ctm_forward_shape() -> None:
    """Forward pass returns (B, T, output_dim+3) = (4, 20, 4)."""
    model = RecurrentCTM(
        input_dim=10, model_dim=32, state_dim=8, n_layers=2,
        n_loop_iters=3, parallel_scan=True,
    )
    x = torch.randn(4, 20, 10)
    out = model(x)
    assert out.shape == (4, 20, 4), f"Expected (4,20,4), got {out.shape}"


def test_recurrent_ctm_output_finite() -> None:
    """Forward pass output contains only finite values."""
    model = RecurrentCTM(
        input_dim=10, model_dim=32, state_dim=8, n_layers=2,
        n_loop_iters=3, parallel_scan=True,
    )
    x = torch.randn(4, 20, 10)
    out = model(x)
    assert torch.isfinite(out).all(), "Output contains non-finite values"


def test_recurrent_ctm_param_count() -> None:
    """param_count() is positive and exceeds the base CTMStockModel count."""
    recurrent = RecurrentCTM(
        input_dim=10, model_dim=32, state_dim=8, n_layers=2,
        n_loop_iters=3, parallel_scan=True,
    )
    base = CTMStockModel(
        input_dim=10, model_dim=32, state_dim=8, n_layers=2,
        parallel_scan=True,
    )
    rec_count = recurrent.param_count()
    base_count = base.param_count()
    assert rec_count > 0, "RecurrentCTM param count should be positive"
    assert rec_count > base_count, (
        f"RecurrentCTM ({rec_count}) should have more params "
        f"than base CTMStockModel ({base_count})"
    )


def test_recurrent_ctm_return_hidden() -> None:
    """return_hidden=True returns (output, hidden) tuple with correct shapes."""
    model = RecurrentCTM(
        input_dim=10, model_dim=32, state_dim=8, n_layers=2,
        n_loop_iters=3, parallel_scan=True,
    )
    x = torch.randn(4, 20, 10)
    out = model(x, return_hidden=True)
    assert isinstance(out, tuple), "Expected tuple output when return_hidden=True"
    assert len(out) == 2, "Expected (output, hidden) tuple"
    pred, hidden = out
    assert pred.shape == (4, 20, 4), f"Expected pred (4,20,4), got {pred.shape}"
    assert hidden.shape == (4, 20, 32), f"Expected hidden (4,20,32), got {hidden.shape}"


def test_recurrent_ctm_n_loop_1_matches_base_output_dim() -> None:
    """n_loop_iters=1 produces same output channels as base CTMStockModel."""
    loop1 = RecurrentCTM(
        input_dim=10, model_dim=32, state_dim=8, n_layers=2,
        n_loop_iters=1, parallel_scan=True,
    )
    base = CTMStockModel(
        input_dim=10, model_dim=32, state_dim=8, n_layers=2,
        parallel_scan=True,
    )
    x = torch.randn(4, 20, 10)
    out_loop1 = loop1(x)
    out_base = base(x)
    assert out_loop1.shape == out_base.shape == (4, 20, 4), (
        f"Shape mismatch: loop1 {out_loop1.shape}, base {out_base.shape}"
    )


def test_recurrent_ctm_n_loop_3_produces_different_output() -> None:
    """Different n_loop_iters values produce different forward outputs."""
    torch.manual_seed(42)
    loop1 = RecurrentCTM(
        input_dim=10, model_dim=32, state_dim=8, n_layers=2,
        n_loop_iters=1, parallel_scan=True,
    )
    torch.manual_seed(42)
    loop3 = RecurrentCTM(
        input_dim=10, model_dim=32, state_dim=8, n_layers=2,
        n_loop_iters=3, parallel_scan=True,
    )
    x = torch.randn(4, 20, 10)
    out1 = loop1(x)
    out3 = loop3(x)
    assert not torch.allclose(out1, out3, atol=1e-4), (
        "n_loop_iters=1 and n_loop_iters=3 should produce different outputs"
    )


def test_recurrent_ctm_nan_detection() -> None:
    """NaN input triggers a UserWarning and is recovered via nan_to_num."""
    model = RecurrentCTM(
        input_dim=10, model_dim=32, state_dim=8, n_layers=2,
        n_loop_iters=3, parallel_scan=True,
    )
    x = torch.full((4, 20, 10), float("nan"))
    with pytest.warns(UserWarning, match="NaN detected"):
        out = model(x)
    assert torch.isfinite(out).all(), "NaN input should be recovered to finite output"


def test_recurrent_ctm_gradient_flow() -> None:
    """loss.backward() produces finite gradients for all parameters."""
    model = RecurrentCTM(
        input_dim=10, model_dim=32, state_dim=8, n_layers=2,
        n_loop_iters=3, parallel_scan=True,
    )
    model.train()
    x = torch.randn(4, 20, 10)
    out = model(x)
    loss = out.pow(2).mean()
    loss.backward()
    for name, p in model.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), (
                f"Non-finite gradient in {name}"
            )


def test_recurrent_ctm_residual_loop_preserves_shape() -> None:
    """_encode_loop preserves (B, T, model_dim) through conv + Mamba blocks."""
    model = RecurrentCTM(
        input_dim=10, model_dim=32, state_dim=8, n_layers=2,
        n_loop_iters=3, parallel_scan=True,
    )
    model.eval()
    x = torch.randn(4, 20, 10)
    # Project to model_dim first (_encode_loop skips input_proj)
    h = model.input_proj(x)
    h = model._encode_loop(h)
    assert h.shape == (4, 20, 32), f"Expected (4,20,32), got {h.shape}"
    assert torch.isfinite(h).all(), "Hidden state contains non-finite values"


def test_recurrent_ctm_bidirectional() -> None:
    """Forward pass with bidirectional=True produces correct shape and finite output."""
    model = RecurrentCTM(
        input_dim=10, model_dim=32, state_dim=8, n_layers=2,
        n_loop_iters=3, bidirectional=True, parallel_scan=True,
    )
    x = torch.randn(4, 20, 10)
    out = model(x)
    assert out.shape == (4, 20, 4), f"Expected (4,20,4), got {out.shape}"
    assert torch.isfinite(out).all(), "Output contains non-finite values"


def test_recurrent_ctm_decomp() -> None:
    """Forward pass with use_decomp=True produces correct shape and finite output."""
    model = RecurrentCTM(
        input_dim=10, model_dim=32, state_dim=8, n_layers=2,
        n_loop_iters=3, use_decomp=True, parallel_scan=True,
    )
    x = torch.randn(4, 20, 10)
    out = model(x)
    assert out.shape == (4, 20, 4), f"Expected (4,20,4), got {out.shape}"
    assert torch.isfinite(out).all(), "Output contains non-finite values"


def test_recurrent_ctm_return_hidden_constructor() -> None:
    """return_hidden=True in constructor is respected when calling forward() without kwarg."""
    model = RecurrentCTM(
        input_dim=10, model_dim=32, state_dim=8, n_layers=2,
        n_loop_iters=3, return_hidden=True, parallel_scan=True,
    )
    x = torch.randn(4, 20, 10)
    out = model(x)
    assert isinstance(out, tuple), "Expected tuple output when return_hidden=True in constructor"
    assert len(out) == 2, "Expected (output, hidden) tuple"
    pred, hidden = out
    assert pred.shape == (4, 20, 4), f"Expected pred (4,20,4), got {pred.shape}"
    assert hidden.shape == (4, 20, 32), f"Expected hidden (4,20,32), got {hidden.shape}"


def test_recurrent_ctm_extra_repr() -> None:
    """extra_repr() contains n_loop_iters and loop_dropout."""
    model = RecurrentCTM(
        input_dim=10, model_dim=32, state_dim=8, n_layers=2,
        n_loop_iters=3, loop_dropout=0.1, parallel_scan=True,
    )
    repr_str = repr(model)
    assert "n_loop_iters" in repr_str, (
        "extra_repr should contain n_loop_iters"
    )
    assert "loop_dropout" in repr_str, (
        "extra_repr should contain loop_dropout"
    )
