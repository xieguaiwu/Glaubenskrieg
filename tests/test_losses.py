"""Tests for loss functions (losses, ensemble)."""

import math

import torch
import torch.nn.functional as F
import numpy as np
import pytest

from src.model.losses import (
    mse_loss,
    sharpe_loss,
    directional_loss,
    pinball_loss,
    composite_loss,
    LossConfig,
)
from src.model.ensemble import compute_ic, rankic_loss_gbdt_style, evaluate_ensemble, EnsembleConfig
from src.utils.metrics import sharpe_ratio_torch


def test_mse_loss_value():
    pred = torch.tensor([1.0, 2.0, 3.0])
    target = torch.tensor([1.5, 2.5, 2.5])
    ours = mse_loss(pred, target)
    ref = F.mse_loss(pred, target)
    assert torch.allclose(ours, ref), f"{ours} != {ref}"


def test_sharpe_loss_shape():
    pred = torch.randn(4, 10, 1)
    loss = sharpe_loss(pred)
    assert loss.dim() == 0, f"Expected scalar, got shape {loss.shape}"


def test_sharpe_loss_constant_input():
    pred = torch.zeros(4, 10, 1)
    loss = sharpe_loss(pred)
    assert torch.allclose(loss, torch.zeros_like(loss), atol=1e-6), f"Expected 0, got {loss}"


def test_sharpe_ratio_torch_1d_and_2d_equivalent():
    """1D input and (1,N) 2D input should produce the same Sharpe ratio."""
    data = torch.tensor([0.01, 0.02, 0.015, 0.025, 0.01])
    sr_1d = sharpe_ratio_torch(data, annual_factor=1.0, ddof=1)
    sr_2d = sharpe_ratio_torch(data.unsqueeze(0), annual_factor=1.0, ddof=1)
    assert torch.allclose(sr_1d, sr_2d, atol=1e-6), (
        f"1D Sharpe {sr_1d} != 2D (1,N) Sharpe {sr_2d}"
    )


def test_sharpe_ratio_torch_2d_value():
    """Verify (1,N) sharpe_ratio_torch returns positive value matching analytical form."""
    data = torch.tensor([[0.01, 0.02, 0.03, 0.04, 0.05]])
    result = sharpe_ratio_torch(data, annual_factor=1.0, ddof=1)
    # Known positive Sharpe for increasing sequence
    assert result.item() > 0, f"Expected positive Sharpe, got {result.item()}"
    # Verify at a level reasonably close to analytical (accounting for 1e-6 eps)
    mu = data.mean().item()
    std = data.std(correction=1).item()
    expected = mu / (std + 1e-6)
    assert abs(result.item() - expected) < 0.01, (
        f"Sharpe {result.item():.6f} diverges from analytical {expected:.6f}"
    )


def test_directional_loss_shape():
    logits = torch.randn(4, 20, 3)
    targets = torch.randint(0, 3, (4, 20))
    loss = directional_loss(logits, targets)
    assert loss.dim() == 0, f"Expected scalar, got shape {loss.shape}"


def test_pinball_loss_symmetric():
    pred = torch.tensor([1.0, 2.0, 3.0])
    target = pred.clone()
    loss = pinball_loss(pred, target, tau=0.5)
    assert torch.allclose(loss, torch.zeros_like(loss), atol=1e-6), f"Expected 0, got {loss}"


def test_composite_loss_backward():
    model = torch.nn.Linear(10, 4)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    x = torch.randn(4, 20, 10)
    pred = model(x)
    reg_target = torch.randn(4, 20, 1)
    cls_targets = torch.randint(0, 3, (4, 20))
    cfg = LossConfig()
    loss = composite_loss(pred, reg_target, cls_targets, list(model.parameters()), cfg, num_regression=1)
    loss.backward()
    assert model.weight.grad is not None, "Gradients should be computed"
    assert torch.isfinite(model.weight.grad).all(), "Gradients should be finite"


def test_compute_ic_perfect():
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    ic = compute_ic(y_true, y_pred)
    assert abs(ic - 1.0) < 1e-6, f"Expected 1.0, got {ic}"


def test_compute_ic_negative():
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    ic = compute_ic(y_true, y_pred)
    assert abs(ic - (-1.0)) < 1e-6, f"Expected -1.0, got {ic}"


def test_compute_ic_constant():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([5.0, 5.0, 5.0])
    ic = compute_ic(y_true, y_pred)
    assert abs(ic) < 1e-6, f"Expected 0.0, got {ic}"


def test_rankic_loss_gbdt_style_shape():
    y_true = torch.randn(10)
    y_pred = torch.randn(10)
    loss, grad, hess = rankic_loss_gbdt_style(y_true, y_pred)
    assert loss.dim() == 0, f"Expected scalar loss, got shape {loss.shape}"
    assert grad.shape == (10,), f"Expected (10,) grad, got {grad.shape}"
    assert hess.shape == (10,), f"Expected (10,) hess, got {hess.shape}"


def test_rankic_loss_gbdt_style_gradients_finite():
    y_true = torch.randn(10)
    y_pred = torch.randn(10)
    loss, grad, hess = rankic_loss_gbdt_style(y_true, y_pred)
    assert torch.isfinite(grad).all(), "Gradients contain non-finite values"
    assert torch.isfinite(hess).all(), "Hessians contain non-finite values"


def test_evaluate_ensemble_identical():
    N = 100
    y_true = np.random.randn(N)
    config = EnsembleConfig(
        use_ic_weighting=True, ic_lookback=20, min_samples_for_ic=20
    )
    result = evaluate_ensemble(y_true, y_true, y_true, config)
    assert abs(result.ic_ctm - 1.0) < 1e-6
    assert abs(result.ic_gbdt - 1.0) < 1e-6
    assert abs(result.fused_ic - 1.0) < 1e-6
