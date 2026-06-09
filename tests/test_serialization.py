"""Tests for model serialization module."""

import pytest

import torch
from src.utils.serialization import (
    save_ctm_model,
    load_ctm_model,
    save_gbdt_model,
    load_gbdt_model,
    save_ensemble,
)


def test_serialization_module_imports():
    """Verify all serialization functions are importable."""
    assert callable(save_ctm_model)
    assert callable(load_ctm_model)
    assert callable(save_gbdt_model)
    assert callable(load_gbdt_model)
    assert callable(save_ensemble)


def test_save_ctm_model_smoke(tmp_path):
    """Smoke test: save and load a simple Linear model."""
    model = torch.nn.Linear(4, 2)
    state_path, _ = save_ctm_model(model, str(tmp_path), "test_model")
    assert state_path.endswith(".pt")

    loaded = load_ctm_model(
        torch.nn.Linear,
        {"in_features": 4, "out_features": 2},
        state_path,
    )
    assert loaded is not None
    assert isinstance(loaded, torch.nn.Linear)
