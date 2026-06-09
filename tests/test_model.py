"""Tests for model architecture (ctm_model, mamba_block, multiasset_ctm)."""

import torch
import pytest

from src.model.ctm_model import CTMStockModel
from src.model.mamba_block import MambaBlock
from src.model.mamba_parallel import MambaBlockParallel
from src.model.multiasset_ctm import MultiAssetCTM


def test_ctm_model_forward_shape():
    model = CTMStockModel(input_dim=10, model_dim=32, state_dim=8, n_layers=2)
    x = torch.randn(4, 20, 10)
    out = model(x)
    assert out.shape == (4, 20, 4), f"Expected (4,20,4), got {out.shape}"


def test_ctm_model_output_range():
    model = CTMStockModel(input_dim=10, model_dim=32, state_dim=8, n_layers=2)
    x = torch.randn(4, 20, 10)
    out = model(x)
    assert torch.isfinite(out).all(), "Output contains non-finite values"


def test_ctm_model_param_count():
    model = CTMStockModel(input_dim=10, model_dim=32, state_dim=8, n_layers=2)
    n = model.param_count()
    assert n > 0, f"Expected positive param count, got {n}"


def test_ctm_extract_features_shape():
    model = CTMStockModel(input_dim=10, model_dim=32, state_dim=8, n_layers=2)
    x = torch.randn(4, 20, 10)
    h = model.extract_features(x)
    assert h.shape == (4, 20, 32), f"Expected (4,20,32), got {h.shape}"


def test_ctm_return_hidden():
    model = CTMStockModel(input_dim=10, model_dim=32, state_dim=8, n_layers=2, return_hidden=True)
    x = torch.randn(4, 20, 10)
    out = model(x)
    assert isinstance(out, tuple), "Expected tuple output when return_hidden=True"
    assert len(out) == 2, "Expected (output, hidden) tuple"
    pred, hidden = out
    assert pred.shape == (4, 20, 4)
    assert hidden.shape == (4, 20, 32)


def test_mamba_block_forward():
    block = MambaBlock(d_model=32, d_state=8)
    x = torch.randn(2, 10, 32)
    y, h_final = block(x)
    assert y.shape == (2, 10, 64), f"Expected (2,10,64), got {y.shape}"
    assert h_final.shape == (2, 64, 8), f"Expected (2,64,8), got {h_final.shape}"


def test_mamba_block_state_evolution():
    block = MambaBlock(d_model=32, d_state=8)
    x1 = torch.randn(2, 10, 32)
    x2 = torch.randn(2, 10, 32) + 10.0
    _, h1 = block(x1)
    _, h2 = block(x2)
    assert not torch.allclose(h1, h2, atol=1e-4), "h_final should differ for different inputs"


def test_multiasset_ctm_forward():
    model = MultiAssetCTM(n_assets=3, input_dim=10, model_dim=32)
    x = torch.randn(2, 3, 20, 10)
    out = model(x)
    assert out.shape == (2, 20, 12), f"Expected (2,20,12), got {out.shape}"


def test_multiasset_ctm_output_finite():
    model = MultiAssetCTM(n_assets=3, input_dim=10, model_dim=32)
    x = torch.randn(2, 3, 20, 10)
    out = model(x)
    assert torch.isfinite(out).all(), "MultiAssetCTM output contains non-finite values"


def test_ctm_model_nan_detection():
    model = CTMStockModel(input_dim=10, model_dim=32, state_dim=8, n_layers=2)
    x = torch.full((4, 20, 10), float("nan"))
    with pytest.warns(UserWarning, match="NaN detected"):
        model(x)


def test_mamba_parallel_forward():
    block = MambaBlockParallel(d_model=32, d_state=8, d_conv=3, expand=2)
    x = torch.randn(2, 10, 32)
    y, h = block(x)
    assert y.shape == (2, 10, 64)
    assert h.shape == (2, 64, 8)
    assert torch.isfinite(y).all()


def test_ctm_parallel_scan_flag():
    model = CTMStockModel(input_dim=10, model_dim=32, state_dim=8, n_layers=2, parallel_scan=True)
    for block in model.mamba_blocks:
        assert block.__class__.__name__ == "MambaBlockParallel"
    x = torch.randn(2, 20, 10)
    out = model(x)
    assert out.shape == (2, 20, 4)
    assert torch.isfinite(out).all()
